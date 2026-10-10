"""Detect, track and heat-map players in a (handheld) phone video.

    python -m basket_tracking.run --video data/clip.MOV --detector roboflow
"""

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from .detectors import make_detector
from .court import HALF_LENGTH, WIDTH
from .heatmaps import TEAM_COLORS, build_panorama, panorama_canvas, save_court_heatmaps, save_heatmaps
from .stabilize import Stabilizer, transform_boxes, transform_points


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def detect_all(frames, args, cache_path):
    """Run the detector on every frame, caching results since this is the slow step."""
    if cache_path.exists() and not args.no_cache:
        data = np.load(cache_path, allow_pickle=True)
        print(f"Loaded cached detections from {cache_path}")
        return list(data["boxes"]), list(data["confs"])
    detector = make_detector(args.detector)
    boxes, confs = [], []
    for i, frame in enumerate(frames):
        det = detector(frame)
        boxes.append(det.xyxy.astype(np.float32))
        confs.append(det.confidence.astype(np.float32) if det.confidence is not None else np.ones(len(det), np.float32))
        if i % 30 == 0:
            print(f"  detect {i}/{len(frames)}: {len(det)} people")
    np.savez(cache_path, boxes=np.array(boxes, dtype=object), confs=np.array(confs, dtype=object))
    return boxes, confs


def torso_color(frame, box):
    """Mean Lab color of the upper-body region, used to split players into teams."""
    x1, y1, x2, y2 = box.astype(int)
    h, w = y2 - y1, x2 - x1
    crop = frame[max(y1 + int(0.2 * h), 0):max(y1 + int(0.5 * h), 1), max(x1 + int(0.25 * w), 0):max(x2 - int(0.25 * w), 1)]
    if crop.size == 0:
        return np.full(3, np.nan, np.float32)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(axis=0)


def on_court(points, court, margin=0.0):
    """True for points at least `margin` pixels inside the court polygon. Without a
    polygon everything counts as on court."""
    if court is None:
        return np.ones(len(points), bool)
    poly = np.asarray(court, np.float32)
    return np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), True) >= margin for x, y in points], bool)


def on_court_m(court_xy, margin=0.0):
    """True for points (meters) within the full court, `margin` meters inside the lines."""
    x, y = court_xy[:, 0], court_xy[:, 1]
    return (x >= margin) & (x <= WIDTH - margin) & (y >= margin) & (y <= 2 * HALF_LENGTH - margin)


def court_homography(calibration):
    """Reference-frame pixels -> court meters, from matched landmark points."""
    if not calibration:
        return None
    H, _ = cv2.findHomography(np.float32(calibration["image_points"]), np.float32(calibration["court_points"]), 0)
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--detector", choices=["roboflow", "yolo"], default="roboflow")
    ap.add_argument("--config", type=Path, help="JSON with optional 'reference_frame', 'court_calibration', 'court' and 'court_margin' (see README)")
    ap.add_argument("--min-track-seconds", type=float, default=2.0)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    config = json.loads(args.config.read_text()) if args.config else {}
    out = args.out / args.video.stem / args.detector
    out.mkdir(parents=True, exist_ok=True)

    frames, fps = read_frames(args.video)
    print(f"{len(frames)} frames at {fps:.1f} fps, {frames[0].shape[1]}x{frames[0].shape[0]}")
    boxes, confs = detect_all(frames, args, out / "detections.npz")

    ref_idx = config.get("reference_frame", 0)
    court = config.get("court")
    H_court = court_homography(config.get("court_calibration"))
    margin = config.get("court_margin", 0.0)
    stabilizer = Stabilizer(frames[ref_idx])
    tracker = sv.ByteTrack(frame_rate=int(round(fps)), lost_track_buffer=int(fps * 2))
    homographies, rows = [], []

    for i, frame in enumerate(frames):
        H = stabilizer.update(frame, boxes[i])
        homographies.append(H)
        img_boxes = boxes[i]
        feet_img = np.c_[(img_boxes[:, 0] + img_boxes[:, 2]) / 2, img_boxes[:, 3]] if len(img_boxes) else np.zeros((0, 2))
        feet_ref = transform_points(H, feet_img)
        feet_court = transform_points(H_court, feet_ref) if H_court is not None else np.full((len(feet_ref), 2), np.nan)
        # Hard court check: people whose feet are off court are never handed to the tracker.
        keep = on_court_m(feet_court, margin) if H_court is not None else on_court(feet_ref, court, margin)

        # Track in reference coordinates so camera pans don't look like player motion.
        det = sv.Detections(
            xyxy=transform_boxes(H, img_boxes[keep]).astype(np.float32),
            confidence=confs[i][keep],
            class_id=np.zeros(int(keep.sum()), int),
            data={"idx": np.flatnonzero(keep)},
        )
        det = tracker.update_with_detections(det)
        for tid, idx in zip(det.tracker_id, det.data["idx"]):
            rows.append(dict(
                frame=i, time=i / fps, track_id=int(tid),
                box=img_boxes[idx], foot_ref=feet_ref[idx], foot_court=feet_court[idx], conf=float(confs[i][idx]),
                color=torso_color(frame, img_boxes[idx]),
            ))
        if i % 60 == 0:
            print(f"  track {i}/{len(frames)}: {len(det)} tracked")

    # Keep tracks long enough to say something about where a player spends time.
    lengths = {}
    for r in rows:
        lengths[r["track_id"]] = lengths.get(r["track_id"], 0) + 1
    long_ids = sorted((t for t, n in lengths.items() if n / fps >= args.min_track_seconds), key=lambda t: -lengths[t])
    rows = [r for r in rows if r["track_id"] in long_ids]

    # Two teams from jersey color: k-means on each track's median torso Lab color. Lightness
    # is down-weighted so shadows matter less, but kept so navy vs white still separates.
    team = {t: 2 for t in long_ids}
    team_names = ["team A", "team B", "unknown team"]
    if len(long_ids) >= 2:
        weights = np.array([0.5, 1.0, 1.0], np.float32)
        colors = np.array([np.nanmedian([r["color"] * weights for r in rows if r["track_id"] == t], axis=0) for t in long_ids], np.float32)
        _, labels, centers = cv2.kmeans(colors, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1), 10, cv2.KMEANS_PP_CENTERS)
        team = dict(zip(long_ids, labels.ravel().tolist()))
        # Name teams by jersey shade so the labels stay meaningful across runs.
        lighter = int(np.argmax(centers[:, 0]))
        team_names[lighter], team_names[1 - lighter] = "light jerseys", "dark jerseys"

    with open(out / "tracks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "track_id", "team", "x1", "y1", "x2", "y2", "foot_ref_x", "foot_ref_y", "court_x_m", "court_y_m", "conf"])
        for r in rows:
            w.writerow([r["frame"], f"{r['time']:.3f}", r["track_id"], team[r["track_id"]], *np.round(r["box"], 1), *np.round(r["foot_ref"], 1), *np.round(r["foot_court"], 2), f"{r['conf']:.3f}"])

    # Heat maps over a player-free panorama of the court, in reference-frame coordinates.
    T, size = panorama_canvas(frames[0].shape, homographies)
    sample = range(0, len(frames), max(1, len(frames) // 25))
    pano = build_panorama([frames[i] for i in sample], [homographies[i] for i in sample], T, size)
    cv2.imwrite(str(out / "panorama.jpg"), pano)
    # T maps reference pixels onto panorama.jpg; keep it to calibrate the court from the panorama.
    np.savez(out / "camera.npz", homographies=np.array(homographies), panorama_T=T, reference_frame=ref_idx)
    tracks = {}
    for t in long_ids:
        pts = np.array([r["foot_ref"] for r in rows if r["track_id"] == t])
        tracks[t] = dict(points=transform_points(T, pts), team=team[t], team_name=team_names[team[t]],
                         label=f"Player {t} ({team_names[team[t]]})")
    save_heatmaps(out / "heatmaps", pano, tracks, size, fps)
    if H_court is not None:
        for t in long_ids:
            tracks[t]["court"] = np.array([r["foot_court"] for r in rows if r["track_id"] == t])
        save_court_heatmaps(out / "court_heatmaps", tracks, fps)

    # Annotated video in the original camera view.
    writer = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, frames[0].shape[1::-1])
    by_frame = {}
    for r in rows:
        by_frame.setdefault(r["frame"], []).append(r)
    ui = max(frames[0].shape[:2]) / 640  # scale lines and text with the video size
    thick = max(1, round(ui))
    for i, frame in enumerate(frames):
        frame = frame.copy()
        if court is not None:
            outline = transform_points(np.linalg.inv(homographies[i]), np.asarray(court, np.float32))
            cv2.polylines(frame, [outline.astype(np.int32)], True, (0, 220, 255), thick, cv2.LINE_AA)
        for r in by_frame.get(i, []):
            hexc = TEAM_COLORS[team[r["track_id"]]].lstrip("#")
            bgr = tuple(int(hexc[k:k + 2], 16) for k in (4, 2, 0))
            x1, y1, x2, y2 = r["box"].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, thick)
            cv2.putText(frame, str(r["track_id"]), (x1, y1 - 3 * thick), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * ui, bgr, thick, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    if shutil.which("ffmpeg"):
        # OpenCV writes MPEG-4 Part 2, which phones and browsers often won't play.
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(out / "annotated.mp4"), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(out / "annotated_h264.mp4")], check=False)
        (out / "annotated_h264.mp4").replace(out / "annotated.mp4")

    print(f"{len(long_ids)} tracks >= {args.min_track_seconds}s. Outputs in {out}")


if __name__ == "__main__":
    main()
