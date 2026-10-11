"""Standard monocular-depth evaluation metrics (AbsRel, RMSE, delta<1.25)."""
from __future__ import annotations

from typing import Dict

import numpy as np


def valid_mask(
    pred: np.ndarray, gt: np.ndarray, min_depth: float = 0.1, max_depth: float = 10.0
) -> np.ndarray:
    """Pixels where both ground truth and prediction are finite and in range."""
    return (
        np.isfinite(gt)
        & np.isfinite(pred)
        & (gt > min_depth)
        & (gt < max_depth)
        & (pred > 0)
    )


def abs_rel(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean absolute relative error: mean(|pred - gt| / gt)."""
    return float(np.mean(np.abs(pred - gt) / gt))


def rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    """Root mean squared error in metres."""
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def delta_accuracy(pred: np.ndarray, gt: np.ndarray, threshold: float = 1.25) -> float:
    """Fraction of pixels with max(pred/gt, gt/pred) < threshold."""
    ratio = np.maximum(pred / gt, gt / pred)
    return float(np.mean(ratio < threshold))


def depth_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
) -> Dict[str, float]:
    """Compute AbsRel, RMSE, delta1 over valid pixels.

    Returns NaN metrics (and ``n_valid == 0``) if no pixel is valid.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    m = valid_mask(pred, gt, min_depth, max_depth)
    n = int(m.sum())
    if n == 0:
        return {"abs_rel": float("nan"), "rmse": float("nan"), "delta1": float("nan"), "n_valid": 0}
    p, g = pred[m], gt[m]
    return {
        "abs_rel": abs_rel(p, g),
        "rmse": rmse(p, g),
        "delta1": delta_accuracy(p, g, 1.25),
        "n_valid": n,
    }
