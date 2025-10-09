"""Repetition segmentation utilities based on hip vertical trajectory."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

PoseTimeSeries = Dict[str, Any]
RepWindow = Dict[str, Any]


def hip_vertical_signal(series: PoseTimeSeries) -> np.ndarray:
    """Return the normalized vertical hip midpoint trajectory.

    The input series is expected to be the cleaned output from ``ml.preprocess``.
    Missing hip coordinates are forward-filled. If no valid observation has been
    seen yet, ``0.0`` is used as a neutral starting value.
    """
    frames = series.get("frames", []) or []
    frame_count = len(frames)
    if frame_count == 0:
        return np.zeros(0, dtype=float)

    values = np.zeros(frame_count, dtype=float)
    last_valid = 0.0
    has_valid = False

    for idx, frame in enumerate(frames):
        landmarks = frame.get("landmarks", {}) or {}
        left = landmarks.get("LEFT_HIP") or {}
        right = landmarks.get("RIGHT_HIP") or {}

        left_y = _extract_float(left.get("y"))
        right_y = _extract_float(right.get("y"))

        candidates = [val for val in (left_y, right_y) if not np.isnan(val)]
        if candidates:
            mean_val = float(np.mean(candidates))
            last_valid = mean_val
            has_valid = True
        else:
            mean_val = last_valid if has_valid else 0.0

        values[idx] = mean_val

    return values


def _extract_float(value: Any) -> float:
    """Convert a value to ``float`` preserving NaN semantics."""
    if value is None:
        return float("nan")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if np.isnan(result):
        return float("nan")
    return result


def find_minima(y: np.ndarray, fps: float, cfg: Dict[str, Any]) -> List[int]:
    """Locate local minima corresponding to squat bottoms.

    Parameters
    ----------
    y:
        Hip midpoint vertical trajectory.
    fps:
        Effective frames per second of the series.
    cfg:
        Configuration dictionary with a ``"segmentation"`` section containing
        ``hip_min_drop_norm``, ``peak_prominence``, and ``min_rep_duration_s``.
    """
    if y.size < 3:
        return []

    seg_cfg = (cfg or {}).get("segmentation", {}) if cfg is not None else {}
    hip_min_drop = float(seg_cfg.get("hip_min_drop_norm", 0.0))
    peak_prominence = float(seg_cfg.get("peak_prominence", 0.0))
    min_rep_duration = float(seg_cfg.get("min_rep_duration_s", 0.0))

    window_radius = max(1, int(round(0.15 * float(fps))))
    window_size = max(3, window_radius * 2 + 1)
    kernel = np.ones(window_size, dtype=float) / window_size
    y_smooth = np.convolve(y, kernel, mode="same")

    min_samples = max(1, int(round(min_rep_duration * float(fps))))

    minima: List[int] = []
    last_peak_idx: int | None = None
    last_peak_val: float | None = None

    for idx in range(1, y_smooth.size - 1):
        prev_val = y_smooth[idx - 1]
        curr_val = y_smooth[idx]
        next_val = y_smooth[idx + 1]

        if prev_val < curr_val > next_val:
            neighbor_high = max(prev_val, next_val)
            if curr_val - neighbor_high >= peak_prominence:
                last_peak_idx = idx
                last_peak_val = curr_val

        if prev_val > curr_val < next_val:
            if minima and idx - minima[-1] < min_samples:
                continue
            if last_peak_idx is None or last_peak_idx >= idx:
                continue
            if last_peak_val is None:
                continue
            drop = last_peak_val - curr_val
            if drop >= hip_min_drop:
                minima.append(idx)
                last_peak_idx = None
                last_peak_val = None

    return minima


def windows_from_minima(min_idx: List[int], fps: float) -> List[RepWindow]:
    """Convert consecutive minima indices into rep windows."""
    if len(min_idx) < 2:
        return []

    windows: List[RepWindow] = []
    time_scale = 1.0 / fps if fps > 0 else 0.0

    for rep_id, (start_idx, end_idx) in enumerate(zip(min_idx[:-1], min_idx[1:]), start=1):
        window: RepWindow = {
            "rep_id": rep_id,
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "start_t": float(start_idx) * time_scale,
            "end_t": float(end_idx) * time_scale,
        }
        windows.append(window)

    return windows


def segment(series: PoseTimeSeries, cfg: Dict[str, Any]) -> List[RepWindow]:
    """Segment cleaned pose series into squat repetition windows."""
    fps = float(series.get("fps", 0.0))
    signal = hip_vertical_signal(series)
    minima = find_minima(signal, fps, cfg)
    return windows_from_minima(minima, fps)
