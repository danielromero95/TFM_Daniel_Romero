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
        "max_rep_duration_s": float(seg_cfg.get("max_rep_duration_s", 0.0)),
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


def _estimate_scale_range(signal: np.ndarray) -> float:
    if signal.size == 0:
        return 0.0
    finite_mask = np.isfinite(signal)
    if not np.any(finite_mask):
        return 0.0
    finite_vals = signal[finite_mask]
    p95 = float(np.percentile(finite_vals, 95))
    p5 = float(np.percentile(finite_vals, 5))
    return float(p95 - p5)


def _strict_extrema_indices(smooth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if smooth.size < 3:
        return np.array([], dtype=int), np.array([], dtype=int)

    values_left = smooth[:-2]
    values_center = smooth[1:-1]
    values_right = smooth[2:]

    peaks_mask = (values_center >= values_left) & (values_center > values_right)
    mins_mask = (values_center <= values_left) & (values_center < values_right)

    peaks = np.nonzero(peaks_mask)[0] + 1
    mins = np.nonzero(mins_mask)[0] + 1
    return peaks.astype(int), mins.astype(int)


def _derivative_extrema_indices(smooth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    length = int(smooth.size)
    if length < 3:
        return np.array([], dtype=int), np.array([], dtype=int)

    gradient = np.diff(smooth)
    if gradient.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    sign = np.sign(gradient)
    # Propagate zero slopes outward to capture flat regions.
    for idx in range(1, sign.size):
        if sign[idx] == 0.0:
            sign[idx] = sign[idx - 1]
    for idx in range(sign.size - 2, -1, -1):
        if sign[idx] == 0.0:
            sign[idx] = sign[idx + 1]

    peaks_idx = np.nonzero((sign[:-1] > 0) & (sign[1:] < 0))[0] + 1
    mins_idx = np.nonzero((sign[:-1] < 0) & (sign[1:] > 0))[0] + 1
    return peaks_idx.astype(int), mins_idx.astype(int)


def _local_extrema_indices(smooth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return merged extrema candidates using derivative and value tests."""

    strict_peaks, strict_mins = _strict_extrema_indices(smooth)
    deriv_peaks, deriv_mins = _derivative_extrema_indices(smooth)
    peaks = np.unique(np.concatenate([strict_peaks, deriv_peaks])).astype(int)
    mins = np.unique(np.concatenate([strict_mins, deriv_mins])).astype(int)
    return peaks, mins


def _refine_extrema(
    smooth: np.ndarray, indices: np.ndarray, radius: int, mode: str
) -> np.ndarray:
    if indices.size == 0:
        return np.array([], dtype=int)

    radius = max(1, int(radius))
    refined: List[int] = []
    for idx in indices:
        center = int(idx)
        start = max(0, center - radius)
        end = min(smooth.size, center + radius + 1)
        if end <= start:
            continue
        window = smooth[start:end]
        if mode == "max":
            local_offset = int(np.argmax(window))
        else:
            local_offset = int(np.argmin(window))
        refined_idx = start + local_offset
        refined.append(refined_idx)

    if not refined:
        return np.array([], dtype=int)

    refined_unique = np.unique(np.asarray(refined, dtype=int))
    return refined_unique


def _select_minima(
    smooth: np.ndarray,
    peaks_idx: np.ndarray,
    mins_idx: np.ndarray,
    fps: float,
    params: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    hip_min_drop_norm = float(params.get("hip_min_drop_norm", 0.08))
    peak_prominence = float(params.get("peak_prominence", 0.03))
    abs_drop = float(params.get("abs_drop", hip_min_drop_norm))
    abs_prom = float(params.get("abs_prom", peak_prominence))
    min_rep_duration_s = float(params.get("min_rep_duration_s", 0.8))
    min_samples = max(1, int(round(min_rep_duration_s * fps)))

    if smooth.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int), {}

    accepted_mins: List[int] = []
    accepted_peaks: List[int] = []
    last_min_idx: int | None = None
    last_peak_idx: int | None = None

    if mins_idx.size == 0 or peaks_idx.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int), {}

    rejection_counts: Dict[str, int] = {
        "too_close": 0,
        "no_prev_peak": 0,
        "no_next_peak": 0,
        "drop_too_small": 0,
        "prominence_low": 0,
        "rise_too_small": 0,
    }

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
            rejection_counts["too_close"] += 1
            continue

        preceding_mask = peaks_idx < min_idx
        if last_peak_idx is not None:
            preceding_mask &= peaks_idx > last_peak_idx
        preceding_peaks = peaks_idx[preceding_mask]
        if preceding_peaks.size == 0:
            rejection_counts["no_prev_peak"] += 1
            continue

        following_peaks = peaks_idx[peaks_idx > min_idx]
        if following_peaks.size == 0:
            rejection_counts["no_next_peak"] += 1
            continue

        peak_before = int(preceding_peaks[-1])
        peak_after = int(following_peaks[0])

        drop = float(smooth[peak_before] - smooth[min_idx])
        if drop < abs_drop:
            rejection_counts["drop_too_small"] += 1
            continue

        prominence_before = peak_prominences.get(peak_before, drop)
        prominence_after = peak_prominences.get(peak_after, float(smooth[peak_after] - smooth[min_idx]))
        if prominence_before < abs_prom and prominence_after < abs_prom:
            rejection_counts["prominence_low"] += 1
            continue

        rise = float(smooth[peak_after] - smooth[min_idx])
        if rise < max(abs_prom, abs_drop * 0.5):
            rejection_counts["rise_too_small"] += 1
            continue

        accepted_mins.append(int(min_idx))
        accepted_peaks.append(peak_before)
        last_min_idx = int(min_idx)
        last_peak_idx = peak_before

    return (
        np.array(accepted_peaks, dtype=int),
        np.array(accepted_mins, dtype=int),
        rejection_counts,
    )


def _detect_reps(
    signal: np.ndarray,
    fps: float,
    params: Dict[str, float],
) -> Dict[str, Any]:
    length = int(signal.size)
    window_size = _moving_average_window(fps, length)
    smooth = _smooth_signal(signal, window_size)
    candidate_peaks_idx, candidate_mins_idx = _local_extrema_indices(smooth)

    refine_radius = max(1, window_size // 4)
    candidate_peaks_idx = _refine_extrema(smooth, candidate_peaks_idx, refine_radius, "max")
    candidate_mins_idx = _refine_extrema(smooth, candidate_mins_idx, refine_radius, "min")

    hip_min_drop_norm = float(params.get("hip_min_drop_norm", 0.08))
    peak_prominence = float(params.get("peak_prominence", 0.03))
    min_rep_duration_s = float(params.get("min_rep_duration_s", 0.8))

    scale_range = _estimate_scale_range(smooth)
    if scale_range <= 0.0:
        fallback_scale = float(np.nanstd(smooth))
        if fallback_scale > 0.0:
            scale_range = fallback_scale

    def _to_absolute(value: float, reference: float) -> float:
        if 0.0 < value <= 1.0:
            if reference > 0.0:
                return float(value * reference)
            return 0.0
        return float(value)

    abs_drop = _to_absolute(hip_min_drop_norm, scale_range)
    abs_prom = _to_absolute(peak_prominence, scale_range)

    selected_peaks: np.ndarray
    selected_mins: np.ndarray
    rejection_counts: Dict[str, int] = {}
    if scale_range > 0.0:
        selection_params = dict(params)
        selection_params["abs_drop"] = abs_drop
        selection_params["abs_prom"] = abs_prom
        selection_params["min_rep_duration_s"] = min_rep_duration_s
        (
            selected_peaks,
            selected_mins,
            rejection_counts,
        ) = _select_minima(
            smooth, candidate_peaks_idx, candidate_mins_idx, fps, selection_params
        )
    else:
        selected_peaks = np.array([], dtype=int)
        selected_mins = np.array([], dtype=int)
        rejection_counts = {}

    return {
        "smooth": smooth,
        "peaks_idx": selected_peaks,
        "mins_idx": selected_mins,
        "candidate_peaks_idx": candidate_peaks_idx,
        "candidate_mins_idx": candidate_mins_idx,
        "window_size": int(window_size),
        "params": {
            "hip_min_drop_norm": hip_min_drop_norm,
            "peak_prominence": peak_prominence,
            "min_rep_duration_s": min_rep_duration_s,
            "window_size": int(window_size),
            "max_rep_duration_s": float(params.get("max_rep_duration_s", 0.0) or 0.0),
            "abs_drop": float(abs_drop),
            "abs_prom": float(abs_prom),
        },
        "scale_range": float(scale_range),
        "abs_drop_used": float(abs_drop),
        "abs_prom_used": float(abs_prom),
        "rejection_counts": rejection_counts,
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
            "candidate_peaks_idx": np.array([], dtype=int),
            "candidate_mins_idx": np.array([], dtype=int),
            "window_size": 1,
            "params": {
                "hip_min_drop_norm": float((params or {}).get("hip_min_drop_norm", 0.08)),
                "peak_prominence": float((params or {}).get("peak_prominence", 0.03)),
                "min_rep_duration_s": float((params or {}).get("min_rep_duration_s", 0.8)),
                "max_rep_duration_s": float((params or {}).get("max_rep_duration_s", 0.0)),
                "window_size": 1,
                "abs_drop": 0.0,
                "abs_prom": 0.0,
            },
            "scale_range": 0.0,
            "abs_drop_used": 0.0,
            "abs_prom_used": 0.0,
            "rejection_counts": {},
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
        "candidate_peaks_idx": result.get("candidate_peaks_idx", np.array([], dtype=int)),
        "candidate_mins_idx": result.get("candidate_mins_idx", np.array([], dtype=int)),
        "peaks_idx": result.get("peaks_idx", np.array([], dtype=int)),
        "mins_idx": result.get("mins_idx", np.array([], dtype=int)),
        "params": result.get("params", {}),
        "scale_range": float(result.get("scale_range", 0.0) or 0.0),
        "abs_drop_used": float(result.get("abs_drop_used", 0.0) or 0.0),
        "abs_prom_used": float(result.get("abs_prom_used", 0.0) or 0.0),
        "rejection_counts": result.get("rejection_counts", {}),
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
    rep_count = len(rep_windows)
    if 1 <= rep_count <= 30:
        return rep_windows, {}

    tuned_params: Dict[str, float] = {}
    scale_range = float(base_result.get("scale_range", 0.0) or 0.0)
    signal = base_result.get("hip_signal", np.array([], dtype=float))
    fps = float(base_result.get("fps", 0.0) or 0.0)
    min_rep_duration_s = float(base_params.get("min_rep_duration_s", 0.8))
    total_duration = float(signal.size / fps) if fps > 0 else 0.0
    expected_reps = total_duration / max(min_rep_duration_s, 1e-6) if total_duration > 0 else 0.0

    need_tune = rep_count == 0 or (rep_count <= 1 and expected_reps >= 3.0)
    if scale_range <= 0.0 or not need_tune:
        return rep_windows, {}

    drop_fraction = max(float(base_params.get("hip_min_drop_norm", 0.08)), 0.0)
    prom_fraction = max(float(base_params.get("peak_prominence", 0.03)), 0.0)

    if drop_fraction > 1.0 and scale_range > 0.0:
        drop_fraction = drop_fraction / scale_range
    if prom_fraction > 1.0 and scale_range > 0.0:
        prom_fraction = prom_fraction / scale_range

    floors = {"hip_min_drop_norm": 0.02, "peak_prominence": 0.005}
    duration_floor = max(0.6, min_rep_duration_s * 0.75)
    duration_ceiling = max(min_rep_duration_s, min_rep_duration_s * 1.2)

    attempts = [
        (0.85, 0.85, 1.0),
        (0.7, 0.7, 0.9),
        (0.6, 0.65, 0.85),
    ]

    def _from_fraction(original: float, fraction: float) -> float:
        fraction = max(fraction, 0.0)
        if original <= 0.0:
            return float(fraction)
        if 0.0 < original <= 1.0:
            return float(fraction)
        if scale_range > 0.0:
            return float(fraction * scale_range)
        return float(original)

    for drop_factor, prom_factor, duration_factor in attempts:
        new_params = dict(base_params)
        candidate_drop_fraction = max(drop_fraction * drop_factor, floors["hip_min_drop_norm"])
        candidate_prom_fraction = max(prom_fraction * prom_factor, floors["peak_prominence"])
        new_params["hip_min_drop_norm"] = _from_fraction(
            base_params.get("hip_min_drop_norm", 0.08), candidate_drop_fraction
        )
        new_params["peak_prominence"] = _from_fraction(
            base_params.get("peak_prominence", 0.03), candidate_prom_fraction
        )
        tuned_duration = float(min_rep_duration_s * duration_factor)
        tuned_duration = min(max(tuned_duration, duration_floor), duration_ceiling)
        new_params["min_rep_duration_s"] = tuned_duration

        tuned_result = _segment_full(series_pre_norm, cfg, new_params)
        rep_windows = tuned_result.get("rep_windows", [])
        if 1 <= len(rep_windows) <= 30:
            tuned_params = {
                "hip_min_drop_norm": float(new_params["hip_min_drop_norm"]),
                "peak_prominence": float(new_params["peak_prominence"]),
                "min_rep_duration_s": float(new_params["min_rep_duration_s"]),
            }
            return rep_windows, tuned_params

    return rep_windows, {}
