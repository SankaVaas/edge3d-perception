# edge3d-perception

CPU-first 3D scene perception pipeline: **sparse-anchor depth alignment**, object detection, INT8 ONNX optimization, and an accuracy-latency benchmark harness.

> **Status: MVP in progress.** Data loading, geometry alignment and metrics are implemented and tested. Models, optimization and benchmarks are being added (see the roadmap). No results are reported until they are measured.

## Why this project

Monocular depth networks give detailed but *relative* depth (unknown scale and shift). Robots need *metric* depth. This project explores how a few sparse metric measurements (LiDAR points, a ToF sensor, SfM landmarks) can fix the scale of a learned depth map, and what that costs in latency on commodity CPUs.

The focus is system design under constraints: an explicit accuracy / latency / memory trade-off, reproducible runs, and tests, rather than training a new model.

## Architecture

```
 RGB frame ──► Monocular depth (relative) ──┐
                                            ├─► Scale/shift alignment ─► Metric depth ─┐
 Sparse metric anchors ─────────────────────┘      (RANSAC / Huber)                     │
                                                                                        ├─► 3D scene output
 RGB frame ──► Object detector ─────────────────────────────────────────────────────────┘

 Cross-cutting: ONNX export + INT8 quantization · per-stage latency profiler · config-driven runs · CI
```

| Layer | Module | Status |
|---|---|---|
| Data | `edge3d.data.tum_rgbd` | done |
| Geometry (alignment) | `edge3d.geometry.scale_align` | done |
| Metrics | `edge3d.bench.metrics` | done |
| Depth + detection models | `edge3d.models` | planned |
| Optimization (ONNX, INT8) | `edge3d.optimize` | planned |
| Latency benchmark | `edge3d.bench.latency` | planned |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

pytest                              # run the unit tests
bash scripts/download_data.sh       # TUM RGB-D freiburg1_xyz (~0.4 GB)
```

Minimal usage of the alignment module:

```python
from edge3d.data.tum_rgbd import TUMRGBDDataset
from edge3d.geometry.scale_align import sample_anchors, align_depth

ds = TUMRGBDDataset("data/rgbd_dataset_freiburg1_xyz", stride=5, max_frames=50)
sample = ds[0]

ys, xs, anchor_d = sample_anchors(sample.depth, n=200)   # simulated sparse sensor
# relative_disp = <output of a monocular depth model, shape (H, W)>
# metric_depth, fit = align_depth(relative_disp, ys, xs, anchor_d, space="disparity", method="ransac")
```

## Evaluation protocol

- **Dataset:** TUM RGB-D `freiburg1_xyz` (RGB + ground-truth metric depth from a Kinect).
- **Sparse anchors are simulated** by sampling pixels from the ground-truth depth map. This stands in for a real sparse sensor and is an assumption of this benchmark, not a hardware result.
- **Metrics:** AbsRel, RMSE, and delta < 1.25 over valid pixels (0.1 m to 8 m).
- **Latency:** fixed thread count, warmup runs, median and p95 per stage (planned).

## Results

_Not yet measured. This table will be filled from `scripts/benchmark.py`._

| Variant | AbsRel ↓ | RMSE (m) ↓ | δ<1.25 ↑ | Latency (ms, CPU) ↓ |
|---|---|---|---|---|
| Monocular, global median scaling | – | – | – | – |
| Monocular + sparse-anchor alignment (RANSAC) | – | – | – | – |
| + INT8 ONNX | – | – | – | – |

## Roadmap

- [x] TUM RGB-D loader, scale/shift alignment (lstsq / RANSAC / Huber), depth metrics, tests
- [ ] Pretrained depth and detection model wrappers
- [ ] ONNX export and INT8 quantization
- [ ] Latency benchmark and accuracy-latency Pareto plot
- [ ] Architecture Decision Records, Dockerfile, CI
- [ ] Multi-object tracking and bird's-eye-view occupancy grid
- [ ] Frame skipping with optical-flow propagation
- [ ] ROS 2 node
- [ ] Corruption-robustness suite (blur, noise, low light)

## License

MIT
