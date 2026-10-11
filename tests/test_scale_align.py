import numpy as np
import pytest

from edge3d.geometry.scale_align import (
    align_depth,
    fit_scale_shift,
    sample_anchors,
)


def _synthetic(n=300, seed=0, scale=2.5, shift=0.3):
    """x = relative prediction, y = metric target with y = scale * x + shift."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.2, 1.5, n)
    y = scale * x + shift
    return x, y, scale, shift


def test_lstsq_recovers_exact_affine():
    x, y, s, t = _synthetic()
    s_hat, t_hat, mask = fit_scale_shift(x, y, method="lstsq")
    assert s_hat == pytest.approx(s, abs=1e-9)
    assert t_hat == pytest.approx(t, abs=1e-9)
    assert mask.all()


@pytest.mark.parametrize("method", ["ransac", "huber"])
def test_robust_methods_survive_outliers(method):
    x, y, s, t = _synthetic(n=400)
    rng = np.random.default_rng(1)
    bad = rng.choice(x.size, size=int(0.2 * x.size), replace=False)
    y_corrupt = y.copy()
    y_corrupt[bad] += rng.choice([-1.0, 1.0], size=bad.size) * rng.uniform(2.0, 6.0, bad.size)

    # Plain least squares is visibly corrupted by the outliers...
    s_ls, _, _ = fit_scale_shift(x, y_corrupt, method="lstsq")
    robust_s, robust_t, mask = fit_scale_shift(x, y_corrupt, method=method)

    assert abs(robust_s - s) < abs(s_ls - s)
    assert robust_s == pytest.approx(s, rel=0.05)
    assert robust_t == pytest.approx(t, abs=0.1)
    # Corrupted anchors should (mostly) be flagged as outliers.
    assert mask[bad].mean() < 0.2


def test_ransac_is_deterministic_for_fixed_seed():
    x, y, _, _ = _synthetic()
    y = y + np.random.default_rng(2).normal(0, 0.05, y.size)
    a = fit_scale_shift(x, y, method="ransac", seed=7)
    b = fit_scale_shift(x, y, method="ransac", seed=7)
    assert a[0] == b[0] and a[1] == b[1]


def test_fit_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        fit_scale_shift(np.array([1.0]), np.array([2.0]))
    with pytest.raises(ValueError):
        fit_scale_shift(np.ones(10), np.arange(10.0))  # no variation in x
    with pytest.raises(ValueError):
        fit_scale_shift(np.arange(5.0), np.arange(4.0))  # shape mismatch


def _scene(h=48, w=64):
    """A smooth synthetic metric depth map in [1, 5] metres."""
    yy, xx = np.mgrid[0:h, 0:w]
    return (1.0 + 4.0 * (xx / w) * 0.6 + 4.0 * (yy / h) * 0.4).astype(np.float64)


def test_align_depth_recovers_metric_depth_in_disparity_space():
    depth_gt = _scene()
    disp_gt = 1.0 / depth_gt
    scale, shift = 0.7, 0.05
    pred = (disp_gt - shift) / scale  # what a relative-disparity network might output

    ys, xs, d = sample_anchors(depth_gt, 150, min_depth=0.1, max_depth=10.0, seed=0)
    out, res = align_depth(pred, ys, xs, d, space="disparity", method="ransac")

    assert out.shape == depth_gt.shape and out.dtype == np.float32
    assert res.scale == pytest.approx(scale, rel=1e-3)
    assert res.shift == pytest.approx(shift, abs=1e-3)
    assert np.allclose(out, depth_gt, rtol=1e-2)


def test_align_depth_in_depth_space():
    depth_gt = _scene()
    pred = (depth_gt - 0.4) / 1.8
    ys, xs, d = sample_anchors(depth_gt, 100, seed=3)
    out, res = align_depth(pred, ys, xs, d, space="depth", method="lstsq")
    assert res.scale == pytest.approx(1.8, rel=1e-6)
    assert np.allclose(out, depth_gt, atol=1e-3)


def test_align_depth_ignores_invalid_anchors_and_clips_range():
    depth_gt = _scene()
    pred = 1.0 / depth_gt
    ys, xs, d = sample_anchors(depth_gt, 80, seed=0)
    d = d.copy()
    d[:5] = 0.0          # invalid sensor returns
    d[5:8] = np.nan
    out, _ = align_depth(pred, ys, xs, d, space="disparity", method="lstsq",
                         min_depth=0.5, max_depth=6.0)
    assert out.min() >= 0.5 and out.max() <= 6.0


def test_align_depth_needs_two_valid_anchors():
    pred = np.random.rand(10, 10)
    with pytest.raises(ValueError):
        align_depth(pred, np.array([1, 2]), np.array([1, 2]), np.array([0.0, np.nan]))


def test_sample_anchors_respects_valid_range_and_seed():
    depth = _scene()
    depth[:10, :] = 0.0  # invalid region
    ys, xs, d = sample_anchors(depth, 50, min_depth=0.1, max_depth=10.0, seed=5)
    assert len(ys) == 50 and (ys >= 10).all() and (d > 0).all()
    ys2, xs2, _ = sample_anchors(depth, 50, min_depth=0.1, max_depth=10.0, seed=5)
    assert (ys == ys2).all() and (xs == xs2).all()
