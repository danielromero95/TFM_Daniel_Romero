"""Rep segmentation utilities for squat detection."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np


def _safe_float(value: Any) -> float:
    """Convert a value to float, returning NaN when conversion fails."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def hip_vertical_signal(series: Dict[str, Any]) -> np.ndarray:
    """Return the vertical hip midpoint trajectory for a pose time-series."""
    frames: Iterable[Dict[str, Any]] = series.get("frames", []) or []
    frame_list = list(frames)
    frame_count = len(frame_list)
    signal = np.zeros(frame_count, dtype=float)

    prev_valid = None
    for idx, frame in enumerate(frame_list):
        landmarks = frame.get("landmarks", {}) or {}
        left = landmarks.get("LEFT_HIP") or {}
        right = landmarks.get("RIGHT_HIP") or {}

        left_y = _safe_float(left.get("y"))
        right_y = _safe_float(right.get("y"))
        values = np.array([left_y, right_y], dtype=float)
        mask = ~np.isnan(values)

        if mask.any():
            current = float(values[mask].mean())
            prev_valid = current
        elif prev_valid is None:
            current = 0.0
        else:
            current = prev_valid

        signal[idx] = current

    return signal


def find_minima(y: np.ndarray, fps: float, cfg: Dict[str, Any]) -> List[int]:
    """Detect local minima in a signal subject to temporal and drop thresholds."""
    if y.size < 3:
        return []

    fps = float(fps) if fps and fps > 0 else 30.0
    seg_cfg = cfg or {}
    min_drop = float(seg_cfg.get("hip_min_drop_norm", 0.08))
    min_duration_s = float(seg_cfg.get("min_rep_duration_s", 0.8))
    min_samples = max(1, int(round(min_duration_s * fps)))

    window = max(3, 2 * int(round(0.15 * fps)) + 1)
    if window > y.size:
        window = y.size if y.size % 2 == 1 else max(3, y.size - 1)
    if window < 3:
        window = 3

    kernel = np.ones(window, dtype=float) / window
    y_smooth = np.convolve(y, kernel, mode="same")

    minima: List[int] = []
    last_peak_idx: int | None = None
    last_peak_val: float | None = None
    last_min_idx: int | None = None

    # Seed the peak with the first value to handle monotonic drops at the start.
    last_peak_idx = 0
    last_peak_val = float(y_smooth[0])

    for idx in range(1, y_smooth.size - 1):
        prev_val = y_smooth[idx - 1]
        curr_val = y_smooth[idx]
        next_val = y_smooth[idx + 1]

        if prev_val < curr_val and curr_val > next_val:
            last_peak_idx = idx
            last_peak_val = float(curr_val)
            continue

        if not (prev_val > curr_val and curr_val < next_val):
            continue

        if last_min_idx is not None and idx - last_min_idx < min_samples:
            continue

        if last_peak_idx is None or last_peak_val is None or last_peak_idx >= idx:
            continue

        if last_peak_val - curr_val < min_drop:
            continue

        minima.append(idx)
        last_min_idx = idx
        last_peak_idx = None
        last_peak_val = None

    return minima


def windows_from_minima(min_idx: List[int], fps: float) -> List[Dict[str, Any]]:
    """Convert consecutive minima indices to rep windows with timing metadata."""
    if len(min_idx) < 2:
        return []

    fps = float(fps) if fps and fps > 0 else 30.0
    windows: List[Dict[str, Any]] = []

    for rep_id, (start_idx, end_idx) in enumerate(zip(min_idx[:-1], min_idx[1:]), start=1):
        if end_idx <= start_idx:
            continue
        start_idx = int(start_idx)
        end_idx = int(end_idx)
        start_t = start_idx / fps
        end_t = end_idx / fps
        windows.append(
            {
                "rep_id": rep_id,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_t": start_t,
                "end_t": end_t,
            }
        )

    return windows


def segment(series_pre_norm: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Full segmentation pipeline operating on the pre-normalized series."""
    if series_pre_norm is None:
        return []

    fps = float(series_pre_norm.get("fps", 0.0) or 0.0)
    if fps <= 0:
        fps = float(cfg.get("fps_cap", 30.0))

    seg_cfg = cfg.get("segmentation", {}) if cfg is not None else {}
    signal = hip_vertical_signal(series_pre_norm)
    minima = find_minima(signal, fps, seg_cfg)
    return windows_from_minima(minima, fps)
