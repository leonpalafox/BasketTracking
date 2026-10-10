# BasketTracking

Track basketball players in phone footage and make a heat map per player.

```
phone video ─► detect players ─► camera-motion compensation ─► ByteTrack ─► team split (jersey color)
            ─► per-player heat maps on a player-free panorama of the court
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
# Roboflow's basketball-tuned model (RF-DETR, hosted). Free API key: https://app.roboflow.com/settings/api
export ROBOFLOW_API_KEY=...
.venv/bin/python -m basket_tracking.run --video data/clip01.MOV --detector roboflow

# Generic COCO YOLO "person" detector, runs locally (also picks up refs, coaches and spectators)
.venv/bin/python -m basket_tracking.run --video data/clip01.MOV --detector yolo
```

Optional `--config clip.json`:

```json
{
  "reference_frame": 0,
  "court_calibration": {
    "image_points": [[x, y], ...],
    "court_points": [[x_m, y_m], ...]
  },
  "court": [[x, y], [x, y], ...],
  "court_margin": 0
}
```

`court_calibration` maps the video onto a top-down FIBA court (15 m wide; `x` from the
left sideline, `y` from the baseline). Give 4+ court landmarks (corners, lane corners,
where the half-court line meets a sideline, ...) as reference-frame pixels and the matching
court positions in meters. With it you get positions in meters in `tracks.csv` and
top-down heat maps, and the court check uses the real court lines (`court_margin` is
then in meters; negative lets feet slightly over a line count).

To find landmarks on a panning clip, run once without calibration, pick the points on
`panorama.jpg` (it shows the whole court), and convert them to reference-frame pixels
with the inverse of `panorama_T` from `camera.npz`. `configs/clip02.json` is a worked
example.

Without calibration, `court` is the playing area as a polygon in reference-frame pixels
(frame `reference_frame` of the clip). It is a hard check: anyone whose feet (bottom-center of the box) fall
outside it, such as the bench, coaches, the scorer's table or spectators, is never
tracked. `court_margin` > 0 requires feet to be that many pixels inside the line. The
outline is drawn on `annotated.mp4` so you can check it follows the court as the camera
pans.

Outputs land in `outputs/<clip>/<detector>/`:

| file | what |
|---|---|
| `annotated.mp4` | original video with track IDs, colored by team |
| `tracks.csv` | per-frame boxes, team and foot position in reference-frame pixels |
| `panorama.jpg` | court background stitched from the clip with players removed |
| `heatmaps/player_XXX.png` | one heat map per track |
| `heatmaps_grid.png`, `heatmap_all.png` | all players side by side / combined (camera view) |
| `court_heatmaps/`, `court_heatmaps_grid.png`, `court_heatmap_teams.png` | top-down court heat maps per player, and combined per team (needs `court_calibration`) |
| `camera.npz` | per-frame homographies and the panorama transform |

Detections are cached in `detections.npz`; pass `--no-cache` to re-detect.

## How it works

- **Detection** – `basket_tracking/detectors.py`. The Roboflow model
  (`basketball-player-detection-3-ycjdo`) separates players from referees, so it is the
  default. Classes `player`, `player-in-possession`, `player-jump-shot`,
  `player-layup-dunk`, `player-shot-block` are kept.
- **Camera motion** – `basket_tracking/stabilize.py`. Handheld footage pans, so every
  frame is registered to a reference frame with SIFT + RANSAC homography (players
  masked out). All positions are expressed in reference-frame pixels.
- **Tracking** – ByteTrack runs on the stabilized boxes, so a camera pan doesn't look
  like every player moving at once.
- **Teams** – k-means (k=2) on each track's median torso color.
- **Heat maps** – each player's foot point (bottom-center of the box), weighted by
  time, blurred, drawn over the panorama.

## Limitations / next steps

- Court calibration assumes FIBA dimensions; adjust `basket_tracking/court.py` for
  other courts.
- Track IDs fragment when players cross or leave the frame. Stitching fragments into
  10 real identities (jersey numbers, or a quick manual label step) is the next step.
- For best results film from a tripod, high up, with the whole court in view.
