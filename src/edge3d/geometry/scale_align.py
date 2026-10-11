"""Sparse-anchor scale/shift alignment for monocular depth.

Monocular depth networks (e.g. Depth Anything) predict *relative* depth that is
only defined up to an unknown affine transform, usually in inverse-depth
(disparity) space. A sparse metric depth source (LiDAR points, a ToF sensor,
triangulated SfM landmarks) can recover the missing scale and shift:

    target = scale * prediction + shift        (fitted at the anchor pixels)

This module implements that fit with three solvers (plain least squares, RANSAC
and Huber IRLS) so that outlier anchors (depth-edge pixels, sensor noise) do
not corrupt the result, and exposes ``align_depth`` to apply it to a full map.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

Method = Literal["lstsq", "ransac", "huber"]
Space = Literal["depth", "disparity"]

_MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class AlignResult:
    """Outcome of an affine alignment fit."""

    scale: float
    shift: float
    inlier_mask: np.ndarray  # bool, one entry per anchor
    method: str
    space: str

    @property
    def n_inliers(self) -> int:
        return int(self.inlier_mask.sum())


# --------------------------------------------------------------------------
# Anchor sampling (simulates a sparse depth sensor from dense ground truth)
# --------------------------------------------------------------------------
def sample_anchors(
    depth_gt: np.ndarray,
    n: int,
    *,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly sample ``n`` valid pixels from a dense metric depth map.

    Returns ``(ys, xs, depths)``. If fewer than ``n`` valid pixels exist, all
    valid pixels are returned. Raises ``ValueError`` if fewer than 2 are valid.
    """
    valid = np.isfinite(depth_gt) & (depth_gt > min_depth) & (depth_gt < max_depth)
    ys_all, xs_all = np.nonzero(valid)
    if ys_all.size < 2:
        raise ValueError("Fewer than 2 valid depth pixels available for anchors.")
    rng = np.random.default_rng(seed)
    k = min(n, ys_all.size)
    idx = rng.choice(ys_all.size, size=k, replace=False)
    ys, xs = ys_all[idx], xs_all[idx]
    return ys, xs, depth_gt[ys, xs].astype(np.float64)


# --------------------------------------------------------------------------
# Affine solvers
# --------------------------------------------------------------------------
def _solve_affine(
    x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """(Weighted) least squares for y ~ s*x + t."""
    A = np.stack([x, np.ones_like(x)], axis=1)
    if w is not None:
        sw = np.sqrt(w)
        A = A * sw[:, None]
        y = y * sw
    (s, t), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(s), float(t)


def _mad(v: np.ndarray) -> float:
    return float(np.median(np.abs(v - np.median(v))))


def fit_scale_shift(
    x: np.ndarray,
    y: np.ndarray,
    method: Method = "ransac",
    *,
    n_iters: int = 200,
    inlier_thresh: Optional[float] = None,
    huber_iters: int = 20,
    seed: int = 0,
) -> Tuple[float, float, np.ndarray]:
    """Fit ``y ~ scale * x + shift`` robustly.

    Args:
        x: predicted (relative) values at the anchors.
        y: target (metric) values at the anchors, in the same space as the fit.
        method: ``lstsq`` (no outlier handling), ``ransac`` or ``huber``.
        n_iters: RANSAC hypotheses.
        inlier_thresh: absolute residual threshold for RANSAC inliers. If None,
            ``0.2 * 1.4826 * MAD(y)`` is used. Tune per sensor noise level.
        huber_iters: IRLS iterations for the Huber solver.
        seed: RNG seed (RANSAC), for reproducibility.

    Returns:
        ``(scale, shift, inlier_mask)``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have the same number of anchors.")
    if x.size < 2:
        raise ValueError("Need at least 2 anchors to fit scale and shift.")
    if np.ptp(x) < 1e-12:
        raise ValueError("Anchor predictions have no variation; fit is degenerate.")

    all_true = np.ones(x.size, dtype=bool)

    if method == "lstsq":
        s, t = _solve_affine(x, y)
        return s, t, all_true

    if method == "ransac":
        thresh = inlier_thresh
        if thresh is None:
            thresh = max(0.2 * _MAD_TO_SIGMA * _mad(y), 1e-9)
        rng = np.random.default_rng(seed)
        best_mask, best_count = None, -1
        for _ in range(n_iters):
            i, j = rng.choice(x.size, size=2, replace=False)
            dx = x[i] - x[j]
            if abs(dx) < 1e-12:
                continue
            s = (y[i] - y[j]) / dx
            t = y[i] - s * x[i]
            mask = np.abs(s * x + t - y) <= thresh
            count = int(mask.sum())
            if count > best_count:
                best_mask, best_count = mask, count
        if best_mask is None or best_count < 2 or np.ptp(x[best_mask]) < 1e-12:
            s, t = _solve_affine(x, y)  # fall back to plain least squares
            return s, t, all_true
        s, t = _solve_affine(x[best_mask], y[best_mask])
        final_mask = np.abs(s * x + t - y) <= thresh
        return s, t, final_mask

    if method == "huber":
        s, t = _solve_affine(x, y)
        delta = 1e-9
        for _ in range(huber_iters):
            r = y - (s * x + t)
            delta = max(1.345 * _MAD_TO_SIGMA * _mad(r), 1e-9)
            w = delta / np.maximum(np.abs(r), delta)
            s, t = _solve_affine(x, y, w)
        r = y - (s * x + t)
        return s, t, np.abs(r) <= 2.0 * delta

    raise ValueError(f"Unknown method: {method!r}")


# --------------------------------------------------------------------------
# Dense alignment
# --------------------------------------------------------------------------
def align_depth(
    pred: np.ndarray,
    anchor_ys: np.ndarray,
    anchor_xs: np.ndarray,
    anchor_depth: np.ndarray,
    *,
    space: Space = "disparity",
    method: Method = "ransac",
    min_depth: float = 1e-3,
    max_depth: float = 1e3,
    **fit_kwargs,
) -> Tuple[np.ndarray, AlignResult]:
    """Align a relative depth/disparity map to metric depth using anchors.

    Args:
        pred: (H, W) relative prediction from a monocular network.
        anchor_ys, anchor_xs: integer pixel coordinates of the anchors.
        anchor_depth: metric depth (metres) at the anchors, must be > 0.
        space: ``"disparity"`` if ``pred`` is relative inverse depth (Depth
            Anything, MiDaS), ``"depth"`` if it is relative depth.
        method: robust solver, see :func:`fit_scale_shift`.
        min_depth, max_depth: clip range for the returned metric depth.

    Returns:
        ``(metric_depth (H, W) float32, AlignResult)``.
    """
    pred = np.asarray(pred, dtype=np.float64)
    ys = np.asarray(anchor_ys, dtype=np.intp)
    xs = np.asarray(anchor_xs, dtype=np.intp)
    d = np.asarray(anchor_depth, dtype=np.float64)

    ok = np.isfinite(d) & (d > 0) & np.isfinite(pred[ys, xs])
    if ok.sum() < 2:
        raise ValueError("Fewer than 2 valid anchors after filtering.")
    ys, xs, d = ys[ok], xs[ok], d[ok]

    x = pred[ys, xs]
    target = 1.0 / d if space == "disparity" else d

    s, t, mask = fit_scale_shift(x, target, method=method, **fit_kwargs)

    fitted = s * pred + t
    if space == "disparity":
        eps = 1.0 / max_depth
        depth = 1.0 / np.maximum(fitted, eps)
    elif space == "depth":
        depth = fitted
    else:
        raise ValueError(f"Unknown space: {space!r}")

    depth = np.clip(depth, min_depth, max_depth).astype(np.float32)
    return depth, AlignResult(s, t, mask, method, space)
