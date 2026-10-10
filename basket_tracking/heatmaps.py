"""Background panorama and per-player heat maps."""

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

# Single-hue sequential ramp (light -> dark blue). Low values fade to transparent so
# the court stays visible underneath.
_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
HEAT_CMAP = LinearSegmentedColormap.from_list("heat_blue", _RAMP)
TEAM_COLORS = ["#d03b3b", "#2a78d6", "#7a7a7a"]  # used only for box outlines/labels


def panorama_canvas(frame_shape, homographies, max_side=1600):
    """Size of a canvas holding every frame warped into reference coordinates, and the
    transform (shift, then downscale to at most `max_side` px) from reference pixels onto it."""
    h, w = frame_shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    pts = np.concatenate([cv2.perspectiveTransform(corners, H) for H in homographies]).reshape(-1, 2)
    # Guard against a bad homography blowing the canvas up.
    lo = np.maximum(pts.min(axis=0), [-2 * w, -2 * h])
    hi = np.minimum(pts.max(axis=0), [3 * w, 3 * h])
    scale = min(1.0, max_side / max(hi - lo))
    T = np.array([[scale, 0, -lo[0] * scale], [0, scale, -lo[1] * scale], [0, 0, 1]], dtype=np.float64)
    size = (int(np.ceil((hi[0] - lo[0]) * scale)), int(np.ceil((hi[1] - lo[1]) * scale)))
    return T, size


def build_panorama(frames, homographies, T, size):
    """Median of warped frames: a clean background of the court without players."""
    stack = []
    for frame, H in zip(frames, homographies):
        warped = cv2.warpPerspective(frame, T @ H, size)
        valid = cv2.warpPerspective(np.ones(frame.shape[:2], np.uint8), T @ H, size) > 0
        stack.append(np.where(valid[..., None], warped, np.nan).astype(np.float32))
    stack = np.stack(stack)
    pano = np.nanmedian(stack, axis=0)
    return np.nan_to_num(pano, nan=0).astype(np.uint8)


def density(points, weights, size, sigma):
    w, h = size
    hist, _, _ = np.histogram2d(points[:, 1], points[:, 0], bins=(h, w), range=((0, h), (0, w)), weights=weights)
    return gaussian_filter(hist, sigma)


def plot_heatmap(ax, background_rgb, dens, title):
    ax.imshow(background_rgb)
    if dens.max() > 0:
        norm = dens / dens.max()
        rgba = HEAT_CMAP(norm)
        rgba[..., 3] = np.clip(norm * 1.4, 0, 0.85)  # fade out near-zero density
        ax.imshow(rgba)
    ax.set_title(title, fontsize=10, color="#222")
    ax.set_axis_off()


def save_heatmaps(out_dir, background_bgr, tracks, size, fps, sigma=None, crop=None):
    """tracks: {track_id: dict(points=Nx2 canvas pixels, team=int, label=str)}"""
    sigma = sigma or max(size) / 100  # blur radius relative to the canvas
    out_dir.mkdir(parents=True, exist_ok=True)
    bg = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2RGB)
    if crop is None and tracks:
        # Zoom to where play happens, with some court around it for context.
        pts = np.concatenate([tr["points"] for tr in tracks.values()])
        pad = 0.25 * max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 100)
        lo = np.maximum(pts.min(axis=0) - pad, 0).astype(int)
        hi = np.minimum(pts.max(axis=0) + pad, size).astype(int)
        crop = (*lo, *hi)
    x0, y0, x1, y1 = crop if crop is not None else (0, 0, size[0], size[1])
    bg = bg[y0:y1, x0:x1]

    dens_by_track = {}
    for tid, tr in tracks.items():
        pts = tr["points"]
        dens_by_track[tid] = density(pts, np.full(len(pts), 1.0 / fps), size, sigma)[y0:y1, x0:x1]

    aspect = bg.shape[0] / bg.shape[1]
    for tid, tr in tracks.items():
        fig, ax = plt.subplots(figsize=(8, 8 * aspect + 0.4), dpi=120)
        plot_heatmap(ax, bg, dens_by_track[tid], f"{tr['label']}  ·  {len(tr['points']) / fps:.1f}s tracked")
        fig.savefig(out_dir / f"player_{tid:03d}.png", bbox_inches="tight")
        plt.close(fig)

    # Small multiples: every player on the same background, same scale per panel.
    ids = list(tracks)
    if ids:
        cols = min(4, len(ids))
        rows = int(np.ceil(len(ids) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * aspect * rows + 0.6), dpi=120, squeeze=False)
        for ax in axes.flat:
            ax.set_axis_off()
        for ax, tid in zip(axes.flat, ids):
            tr = tracks[tid]
            plot_heatmap(ax, bg, dens_by_track[tid], f"{tr['label']} · {len(tr['points']) / fps:.1f}s")
        fig.suptitle("Where each tracked player spent their time", fontsize=13, color="#222")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out_dir.parent / "heatmaps_grid.png", bbox_inches="tight")
        plt.close(fig)

        all_dens = sum(dens_by_track.values())
        fig, ax = plt.subplots(figsize=(8, 8 * aspect + 0.4), dpi=120)
        plot_heatmap(ax, bg, all_dens, "All tracked players")
        fig.savefig(out_dir.parent / "heatmap_all.png", bbox_inches="tight")
        plt.close(fig)


COURT_BG = "#f4efe6"   # light wood tone so the blue heat reads clearly
COURT_RES = 0.05       # meters per density cell


def court_density(points, fps, sigma_m=0.6):
    from .court import HALF_LENGTH, WIDTH

    nx, ny = int((WIDTH + 1) / COURT_RES), int((HALF_LENGTH + 1) / COURT_RES)
    hist, _, _ = np.histogram2d(
        points[:, 1], points[:, 0], bins=(ny, nx),
        range=((-0.5, HALF_LENGTH + 0.5), (-0.5, WIDTH + 0.5)),
        weights=np.full(len(points), 1.0 / fps),
    )
    return gaussian_filter(hist, sigma_m / COURT_RES)  # seconds per cell, smoothed


def plot_court_heatmap(ax, dens, title, vmax=None):
    from .court import HALF_LENGTH, WIDTH, draw_court

    ax.set_facecolor(COURT_BG)
    ax.add_patch(plt.Rectangle((-0.5, -0.5), WIDTH + 1, HALF_LENGTH + 1, color=COURT_BG, zorder=0))
    vmax = vmax or dens.max()
    if vmax > 0:
        norm = np.clip(dens / vmax, 0, 1)
        rgba = HEAT_CMAP(norm)
        rgba[..., 3] = np.clip(norm * 1.4, 0, 0.9)
        ax.imshow(rgba, extent=(-0.5, WIDTH + 0.5, HALF_LENGTH + 0.5, -0.5), interpolation="bilinear", zorder=2)
    draw_court(ax)
    ax.set_title(title, fontsize=10, color="#222")


def save_court_heatmaps(out_dir, tracks, fps):
    """Top-down heat maps on a FIBA half court. tracks[tid]['court'] holds Nx2 meters."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dens = {t: court_density(tr["court"], fps) for t, tr in tracks.items()}

    def title(t):
        return f"{tracks[t]['label']} · {len(tracks[t]['court']) / fps:.1f}s"

    for t in tracks:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=130)
        plot_court_heatmap(ax, dens[t], title(t))
        fig.savefig(out_dir / f"player_{t:03d}.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)

    ids = list(tracks)
    if not ids:
        return
    cols = min(4, len(ids))
    rows = int(np.ceil(len(ids) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.6 * rows + 0.5), dpi=120, squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, t in zip(axes.flat, ids):
        plot_court_heatmap(ax, dens[t], title(t))
    fig.suptitle("Where each tracked player spent their time (top-down)", fontsize=13, color="#222")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir.parent / "court_heatmaps_grid.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # All players, then one panel per team.
    teams = sorted({tr["team_name"] for tr in tracks.values()})
    panels = [("All tracked players", ids)] + [(name.capitalize(), [t for t in ids if tracks[t]["team_name"] == name]) for name in teams]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 5.8), dpi=130, squeeze=False)
    for ax, (name, members) in zip(axes.flat, panels):
        plot_court_heatmap(ax, sum(dens[t] for t in members), f"{name} · {len(members)} tracks")
    fig.tight_layout()
    fig.savefig(out_dir.parent / "court_heatmap_teams.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
