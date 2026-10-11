import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from edge3d.data.tum_rgbd import TUMRGBDDataset, associate, read_file_list


def _make_sequence(root, rgb_ts, depth_ts):
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir(parents=True)
    rgb_lines, depth_lines = ["# color images", "# timestamp filename"], ["# depth images"]
    for i, t in enumerate(rgb_ts):
        red_bgr = np.zeros((4, 6, 3), np.uint8)
        red_bgr[..., 2] = 255  # pure red in BGR
        cv2.imwrite(str(root / "rgb" / f"{i}.png"), red_bgr)
        rgb_lines.append(f"{t} rgb/{i}.png")
    for i, t in enumerate(depth_ts):
        raw = np.full((4, 6), 5000, np.uint16)  # == 1.0 m
        raw[0, 0] = 0                           # one invalid pixel
        cv2.imwrite(str(root / "depth" / f"{i}.png"), raw)
        depth_lines.append(f"{t} depth/{i}.png")
    (root / "rgb.txt").write_text("\n".join(rgb_lines) + "\n")
    (root / "depth.txt").write_text("\n".join(depth_lines) + "\n")


def test_read_file_list_skips_comments(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("# c\n\n1.5 a.png\n2.5 b.png\n")
    assert read_file_list(f) == [(1.5, "a.png"), (2.5, "b.png")]


def test_associate_respects_max_diff():
    rgb = [(1.00, "r0"), (2.00, "r1"), (3.00, "r2")]
    depth = [(1.01, "d0"), (2.50, "d1"), (3.015, "d2")]
    pairs = associate(rgb, depth, max_diff=0.02)
    assert pairs == [(1.00, "r0", "d0"), (3.00, "r2", "d2")]  # r1 has no close depth


def test_dataset_loads_and_converts(tmp_path):
    _make_sequence(tmp_path, [0.0, 0.1, 0.2, 0.3], [0.005, 0.105, 0.205, 0.305])
    ds = TUMRGBDDataset(tmp_path, max_time_diff=0.02)
    assert len(ds) == 4
    s = ds[0]
    assert s.rgb.shape == (4, 6, 3) and s.rgb.dtype == np.uint8
    assert tuple(s.rgb[0, 0]) == (255, 0, 0)       # BGR -> RGB conversion
    assert s.depth.dtype == np.float32
    assert s.depth[1, 1] == pytest.approx(1.0)      # 5000 raw -> 1 m
    assert s.depth[0, 0] == 0.0                     # invalid stays 0


def test_dataset_stride_and_max_frames(tmp_path):
    _make_sequence(tmp_path, [0.0, 0.1, 0.2, 0.3], [0.0, 0.1, 0.2, 0.3])
    assert len(TUMRGBDDataset(tmp_path, stride=2)) == 2
    assert len(TUMRGBDDataset(tmp_path, max_frames=3)) == 3


def test_missing_sequence_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data.sh"):
        TUMRGBDDataset(tmp_path / "nope")
