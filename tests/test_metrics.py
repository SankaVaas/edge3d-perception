import numpy as np
import pytest

from edge3d.bench.metrics import abs_rel, delta_accuracy, depth_metrics, rmse


def test_perfect_prediction():
    gt = np.array([[1.0, 2.0], [3.0, 4.0]])
    m = depth_metrics(gt.copy(), gt)
    assert m["abs_rel"] == 0.0 and m["rmse"] == 0.0 and m["delta1"] == 1.0
    assert m["n_valid"] == 4


def test_known_values_for_ten_percent_overestimate():
    gt = np.array([1.0, 2.0, 4.0])
    pred = 1.1 * gt
    assert abs_rel(pred, gt) == pytest.approx(0.1)
    assert rmse(pred, gt) == pytest.approx(np.sqrt(0.07))
    assert delta_accuracy(pred, gt) == 1.0


def test_delta_is_symmetric_over_and_under_estimation():
    gt = np.array([2.0, 2.0])
    pred = np.array([2.0 * 1.3, 2.0 / 1.3])  # both outside 1.25
    assert delta_accuracy(pred, gt) == 0.0


def test_invalid_pixels_are_masked():
    gt = np.array([0.0, 1.0, 2.0, 50.0, np.nan])
    pred = np.array([1.0, 1.0, 2.0, 50.0, 1.0])
    m = depth_metrics(pred, gt, min_depth=0.1, max_depth=10.0)
    assert m["n_valid"] == 2  # only gt=1 and gt=2 are in range
    assert m["abs_rel"] == 0.0


def test_no_valid_pixels_returns_nan():
    m = depth_metrics(np.ones((2, 2)), np.zeros((2, 2)))
    assert m["n_valid"] == 0 and np.isnan(m["abs_rel"])


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        depth_metrics(np.ones((2, 2)), np.ones((3, 3)))
