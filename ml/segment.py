"""Rep segmentation utilities for squat detection."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

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
    signal = np.full(frame_count, np.nan, dtype=float)

    last_valid: float | None = None
    for idx, frame in enumerate(frame_list):
        landmarks = frame.get("landmarks", {}) or {}
        left = landmarks.get("LEFT_HIP") or {}
        right = landmarks.get("RIGHT_HIP") or {}

        left_y = _safe_float(left.get("y"))
        right_y = _safe_float(right.get("y"))
        values = np.array([left_y, right_y], dtype=float)
        mask = np.isfinite(values)

        if mask.any():
            current = float(values[mask].mean())
            last_valid = current
        elif last_valid is not None:
            current = last_valid
        else:
            current = np.nan

        signal[idx] = current

    if frame_count:
        valid_mask = np.isfinite(signal)
        if valid_mask.any():
            first_valid_idx = int(np.argmax(valid_mask))
            first_valid_value = float(signal[first_valid_idx])
            signal[:first_valid_idx] = first_valid_value

    return signal


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


def _resolve_fps(series: Dict[str, Any] | None, cfg: Dict[str, Any] | None) -> float:
    fps = 0.0
    if series is not None:
        fps = float(series.get("fps", 0.0) or 0.0)
    if fps <= 0.0:
        fps = float((cfg or {}).get("fps_cap", 0.0) or 0.0)
    if fps <= 0.0:
        fps = 30.0
    return fps


def _moving_average_window(fps: float, length: int) -> int:
    if length <= 0:
        return 1
    fps = float(fps) if fps and fps > 0 else 30.0
    k = max(1, int(round(0.125 * fps)))
    window = 2 * k + 1
    window = max(3, window)
    if window > length:
        window = length if length % 2 == 1 else length - 1
        if window <= 0:
            window = length
        if window < 3 and length >= 3:
            window = 3 if 3 <= length else window
    if window <= 0:
        window = 1
    if window > length:
        window = length
    if window % 2 == 0:
        window = max(1, window - 1)
    return max(1, window)


def _moving_average(y: np.ndarray, fps: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = int(y.size)
    if n == 0:
        return y.copy()

    window = _moving_average_window(fps, n)
    kernel = np.ones(window, dtype=float) / float(window)
    smooth = np.convolve(y, kernel, mode="same")
    return smooth


def _scale_range(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    p5 = float(np.nanpercentile(y, 5))
    p95 = float(np.nanpercentile(y, 95))
    range_abs = p95 - p5
    if not np.isfinite(range_abs) or range_abs < 1e-8:
        range_abs = float(np.nanstd(y))
    if not np.isfinite(range_abs) or range_abs < 1e-8:
        return 0.0
    return range_abs


def _local_extrema(smooth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if smooth.size < 3:
        empty = np.array([], dtype=int)
        return empty, empty

    valid_center = np.isfinite(smooth[1:-1])
    left_valid = np.isfinite(smooth[:-2])
    right_valid = np.isfinite(smooth[2:])

    local_min_mask = (
        (smooth[1:-1] < smooth[:-2])
        & (smooth[1:-1] < smooth[2:])
        & valid_center
        & left_valid
        & right_valid
    )
    local_max_mask = (
        (smooth[1:-1] > smooth[:-2])
        & (smooth[1:-1] > smooth[2:])
        & valid_center
        & left_valid
        & right_valid
    )

    minima_idx = np.where(local_min_mask)[0] + 1
    peaks_idx = np.where(local_max_mask)[0] + 1
    return peaks_idx, minima_idx


def find_minima(y: np.ndarray, fps: float, cfg: Dict[str, Any]) -> List[int]:
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return []

    smooth = _moving_average(y, fps)

    range_abs = _scale_range(smooth)
    if range_abs < 1e-6:
        return []

    seg_cfg = (cfg or {}).get("segmentation", {}) or {}
    hip_min_drop_norm = float(seg_cfg.get("hip_min_drop_norm", 0.10))
    peak_prominence = float(seg_cfg.get("peak_prominence", 0.05))
    min_rep_duration_s = float(seg_cfg.get("min_rep_duration_s", 1.0))

    min_distance = max(1, int(round(min_rep_duration_s * float(fps if fps and fps > 0 else 30.0))))
    drop_threshold = hip_min_drop_norm * range_abs
    prominence_threshold = peak_prominence * range_abs

    peaks_idx, minima_idx = _local_extrema(smooth)

    if minima_idx.size == 0 or peaks_idx.size == 0:
        return []

    accepted: List[int] = []
    last_min_idx: int | None = None

    for min_idx in minima_idx:
        if last_min_idx is not None and min_idx - last_min_idx < min_distance:
            continue

        preceding_peaks = peaks_idx[peaks_idx < min_idx]
        if preceding_peaks.size == 0:
            continue

        peak_idx = int(preceding_peaks[-1])
        peak_val = float(smooth[peak_idx])
        min_val = float(smooth[min_idx])
        if not (np.isfinite(peak_val) and np.isfinite(min_val)):
            continue

        drop_abs = peak_val - min_val
        if drop_abs < drop_threshold:
            continue

        prev_min_candidates = minima_idx[minima_idx < peak_idx]
        next_min_candidates = minima_idx[minima_idx > peak_idx]
        neighbor_candidates: List[Tuple[int, float]] = []

        if prev_min_candidates.size:
            prev_idx = int(prev_min_candidates[-1])
            prev_val = float(smooth[prev_idx])
            if np.isfinite(prev_val):
                neighbor_candidates.append((abs(peak_idx - prev_idx), prev_val))

        if next_min_candidates.size:
            next_idx = int(next_min_candidates[0])
            next_val = float(smooth[next_idx])
            if np.isfinite(next_val):
                neighbor_candidates.append((abs(next_idx - peak_idx), next_val))

        prominence_abs = drop_abs
        if neighbor_candidates:
            _, neighbor_val = min(neighbor_candidates, key=lambda item: item[0])
            prominence_abs = peak_val - neighbor_val

        if prominence_abs < prominence_threshold:
            continue

        accepted.append(int(min_idx))
        last_min_idx = int(min_idx)

    return accepted


def diagnose(series_pre_norm: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    fps = _resolve_fps(series_pre_norm, cfg)
    hip_signal = hip_vertical_signal(series_pre_norm)
    smooth = _moving_average(hip_signal, fps) if hip_signal.size else hip_signal.copy()
    peaks_idx, minima_idx = _local_extrema(smooth)
    accepted_minima = find_minima(hip_signal, fps, cfg)

    seg_cfg = (cfg or {}).get("segmentation", {}) or {}
    params = {
        "hip_min_drop_norm": float(seg_cfg.get("hip_min_drop_norm", 0.10)),
        "peak_prominence": float(seg_cfg.get("peak_prominence", 0.05)),
        "min_rep_duration_s": float(seg_cfg.get("min_rep_duration_s", 1.0)),
        "window_size": _moving_average_window(fps, hip_signal.size),
    }

    if fps <= 0:
        fps = 30.0

    t = (
        np.arange(hip_signal.size, dtype=float) / fps
        if hip_signal.size
        else np.array([], dtype=float)
    )

    return {
        "t": t,
        "hip_signal": hip_signal,
        "smooth": smooth,
        "peaks_idx": peaks_idx,
        "mins_idx": np.array(accepted_minima, dtype=int),
        "params": params,
    }


def segment(series_pre_norm: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Full segmentation pipeline operating on the pre-normalized series."""
    fps = _resolve_fps(series_pre_norm, cfg)
    hip_signal = hip_vertical_signal(series_pre_norm)
    if hip_signal.size == 0:
        return []

    minima = find_minima(hip_signal, fps, cfg)
    if not minima:
        return []

    return windows_from_minima(minima, fps)
