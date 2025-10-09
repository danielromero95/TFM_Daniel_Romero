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
        fps = float((cfg or {}).get("fps_cap", 30.0))
    if fps <= 0.0:
        fps = 30.0
    return fps


def _segmentation_params(cfg: Dict[str, Any] | None) -> Dict[str, float]:
    seg_cfg = (cfg or {}).get("segmentation", {}) or {}
    params = {
        "hip_min_drop_norm": float(seg_cfg.get("hip_min_drop_norm", 0.08)),
        "peak_prominence": float(seg_cfg.get("peak_prominence", 0.03)),
        "min_rep_duration_s": float(seg_cfg.get("min_rep_duration_s", 0.8)),
    }
    return params


def _moving_average_window(fps: float, length: int) -> int:
    if length <= 1:
        return max(1, length)
    base = max(1, int(round(0.15 * max(fps, 1.0))))
    window = 2 * base + 1
    window = min(window, length)
    if window % 2 == 0:
        window = max(1, window - 1)
    return max(1, window)


def _smooth_signal(signal: np.ndarray, window_size: int) -> np.ndarray:
    if signal.size == 0:
        return signal.copy()
    window_size = max(1, int(window_size))
    kernel = np.ones(window_size, dtype=float) / float(window_size)
    return np.convolve(signal, kernel, mode="same")


def _local_extrema_indices(smooth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if smooth.size < 3:
        return np.array([], dtype=int), np.array([], dtype=int)

    rising = smooth[1:-1] > smooth[:-2]
    falling = smooth[1:-1] > smooth[2:]
    peaks = np.nonzero(rising & falling)[0] + 1

    rising_min = smooth[1:-1] < smooth[:-2]
    falling_min = smooth[1:-1] < smooth[2:]
    mins = np.nonzero(rising_min & falling_min)[0] + 1
    return peaks.astype(int), mins.astype(int)


def _select_minima(
    smooth: np.ndarray,
    peaks_idx: np.ndarray,
    mins_idx: np.ndarray,
    fps: float,
    params: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    hip_min_drop_norm = float(params.get("hip_min_drop_norm", 0.08))
    peak_prominence = float(params.get("peak_prominence", 0.03))
    min_rep_duration_s = float(params.get("min_rep_duration_s", 0.8))
    min_samples = max(1, int(round(min_rep_duration_s * fps)))

    if smooth.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    accepted_mins: List[int] = []
    accepted_peaks: List[int] = []
    last_min_idx: int | None = None
    last_peak_idx: int | None = None

    if mins_idx.size == 0 or peaks_idx.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    peak_prominences: Dict[int, float] = {}
    for peak_idx in peaks_idx:
        left_candidates = mins_idx[mins_idx < peak_idx]
        right_candidates = mins_idx[mins_idx > peak_idx]
        left_val = smooth[left_candidates[-1]] if left_candidates.size else smooth[peak_idx]
        right_val = smooth[right_candidates[0]] if right_candidates.size else smooth[peak_idx]
        reference = max(left_val, right_val)
        prominence = float(smooth[peak_idx] - reference)
        peak_prominences[int(peak_idx)] = max(0.0, prominence)

    for min_idx in mins_idx:
        if last_min_idx is not None and min_idx - last_min_idx < min_samples:
            continue

        preceding_mask = peaks_idx < min_idx
        if last_peak_idx is not None:
            preceding_mask &= peaks_idx > last_peak_idx
        preceding_peaks = peaks_idx[preceding_mask]
        if preceding_peaks.size == 0:
            continue

        peak_idx = int(preceding_peaks[-1])
        drop = float(smooth[peak_idx] - smooth[min_idx])
        if drop < hip_min_drop_norm:
            continue

        prominence = peak_prominences.get(peak_idx, drop)
        if prominence < peak_prominence:
            continue

        accepted_mins.append(int(min_idx))
        accepted_peaks.append(peak_idx)
        last_min_idx = int(min_idx)
        last_peak_idx = peak_idx

    return np.array(accepted_peaks, dtype=int), np.array(accepted_mins, dtype=int)


def _detect_reps(
    signal: np.ndarray,
    fps: float,
    params: Dict[str, float],
) -> Dict[str, Any]:
    length = int(signal.size)
    window_size = _moving_average_window(fps, length)
    smooth = _smooth_signal(signal, window_size)
    peaks_idx, mins_idx_candidates = _local_extrema_indices(smooth)
    selected_peaks, selected_mins = _select_minima(
        smooth, peaks_idx, mins_idx_candidates, fps, params
    )
    return {
        "smooth": smooth,
        "peaks_idx": selected_peaks,
        "mins_idx": selected_mins,
        "window_size": int(window_size),
        "params": {
            "hip_min_drop_norm": float(params.get("hip_min_drop_norm", 0.08)),
            "peak_prominence": float(params.get("peak_prominence", 0.03)),
            "min_rep_duration_s": float(params.get("min_rep_duration_s", 0.8)),
            "window_size": int(window_size),
        },
    }


def _segment_full(
    series_pre_norm: Dict[str, Any] | None,
    cfg: Dict[str, Any] | None,
    params: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    if series_pre_norm is None:
        fps = _resolve_fps(None, cfg)
        empty = np.array([], dtype=float)
        detection = {
            "hip_signal": empty,
            "smooth": empty,
            "peaks_idx": np.array([], dtype=int),
            "mins_idx": np.array([], dtype=int),
            "window_size": 1,
            "params": {
                "hip_min_drop_norm": float((params or {}).get("hip_min_drop_norm", 0.08)),
                "peak_prominence": float((params or {}).get("peak_prominence", 0.03)),
                "min_rep_duration_s": float((params or {}).get("min_rep_duration_s", 0.8)),
                "window_size": 1,
            },
        }
        detection["rep_windows"] = []
        detection["fps"] = fps
        return detection

    fps = _resolve_fps(series_pre_norm, cfg)
    signal = hip_vertical_signal(series_pre_norm)
    params_resolved = params or _segmentation_params(cfg)
    detection = _detect_reps(signal, fps, params_resolved)
    detection["hip_signal"] = signal
    detection["fps"] = fps
    detection["rep_windows"] = windows_from_minima(detection["mins_idx"].tolist(), fps)
    return detection


def segment(series_pre_norm: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Full segmentation pipeline operating on the pre-normalized series."""
    result = _segment_full(series_pre_norm, cfg)
    return result.get("rep_windows", [])


def diagnose(series_pre_norm: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return diagnostics for segmentation including signals and extrema."""
    result = _segment_full(series_pre_norm, cfg)
    signal = result.get("hip_signal", np.array([], dtype=float))
    fps = float(result.get("fps", 0.0) or 0.0)
    if fps <= 0:
        fps = 30.0
    t = np.arange(signal.size, dtype=float) / fps if signal.size else np.array([], dtype=float)
    diagnostics = {
        "t": t,
        "hip_signal": signal,
        "smooth": result.get("smooth", np.array([], dtype=float)),
        "peaks_idx": result.get("peaks_idx", np.array([], dtype=int)),
        "mins_idx": result.get("mins_idx", np.array([], dtype=int)),
        "params": result.get("params", {}),
    }
    return diagnostics


def auto_tune(
    series_pre_norm: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Attempt segmentation with relaxed thresholds if no reps are found."""
    base_params = _segmentation_params(cfg)
    base_result = _segment_full(series_pre_norm, cfg, base_params)
    rep_windows = base_result.get("rep_windows", [])
    if 1 <= len(rep_windows) <= 30:
        return rep_windows, {}

    tuned_params: Dict[str, float] = {}
    floors = {"hip_min_drop_norm": 0.03, "peak_prominence": 0.01}
    factors = [(0.75, 0.75), (0.5, 0.5)]

    for drop_factor, prom_factor in factors:
        new_params = dict(base_params)
        new_params["hip_min_drop_norm"] = max(base_params["hip_min_drop_norm"] * drop_factor, floors["hip_min_drop_norm"])
        new_params["peak_prominence"] = max(base_params["peak_prominence"] * prom_factor, floors["peak_prominence"])

        tuned_result = _segment_full(series_pre_norm, cfg, new_params)
        rep_windows = tuned_result.get("rep_windows", [])
        if 1 <= len(rep_windows) <= 30:
            tuned_params = {
                "hip_min_drop_norm": float(new_params["hip_min_drop_norm"]),
                "peak_prominence": float(new_params["peak_prominence"]),
            }
            return rep_windows, tuned_params

    return [], {}
