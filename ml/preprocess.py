"""Pose landmark preprocessing utilities: interpolate → smooth → normalize."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, MutableMapping

import numpy as np

from .pose3d import LANDMARK_NAMES

PoseFrame = Dict[str, Any]
PoseTimeSeries = Dict[str, Any]
LandmarkArrays = Dict[str, np.ndarray]


def _init_landmark_arrays(frame_count: int) -> LandmarkArrays:
    """Create landmark → (N×3) array dictionaries filled with NaNs."""
    return {
        name: np.full((frame_count, 3), np.nan, dtype=float) for name in LANDMARK_NAMES
    }


def _nanmean_pair(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return element-wise NaN-aware mean of two (N×3) arrays."""
    stacked = np.stack([first, second])
    counts = np.sum(~np.isnan(stacked), axis=0)
    sums = np.nansum(stacked, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.divide(sums, counts, where=counts > 0)
    mean[counts == 0] = np.nan
    return mean


def _collect_landmark_arrays(frames: Iterable[PoseFrame]) -> LandmarkArrays:
    """Convert PoseFrame landmarks into stacked NumPy arrays keyed by landmark name."""
    frames_list = list(frames)
    frame_count = len(frames_list)
    arrays = _init_landmark_arrays(frame_count)

    for idx, frame in enumerate(frames_list):
        landmarks = frame.get("landmarks", {}) or {}
        for name, point in landmarks.items():
            if name not in arrays:
                continue
            arr = arrays[name]
            x_val = point.get("x")
            y_val = point.get("y")
            z_val = point.get("z")
            arr[idx, 0] = float(x_val) if x_val is not None else np.nan
            arr[idx, 1] = float(y_val) if y_val is not None else np.nan
            arr[idx, 2] = float(z_val) if z_val is not None else np.nan

    return arrays


def _rebuild_series(series: PoseTimeSeries, arrays: LandmarkArrays) -> PoseTimeSeries:
    """Reconstruct a PoseTimeSeries from landmark arrays, preserving metadata."""
    original_frames = series.get("frames", []) or []
    new_frames: list[PoseFrame] = []

    for idx, frame in enumerate(original_frames):
        original_landmarks = frame.get("landmarks", {}) or {}
        new_landmarks: Dict[str, MutableMapping[str, Any]] = {}
        for name in LANDMARK_NAMES:
            arr = arrays.get(name)
            if arr is None or idx >= arr.shape[0]:
                continue
            point = dict(original_landmarks.get(name, {}))
            point["x"] = float(arr[idx, 0])
            point["y"] = float(arr[idx, 1])
            point["z"] = float(arr[idx, 2])
            for extra_key in ("visibility", "presence"):
                if extra_key not in point:
                    point[extra_key] = float("nan")
            new_landmarks[name] = point

        new_frame = dict(frame)
        new_frame["landmarks"] = new_landmarks
        new_frames.append(new_frame)

    new_series = dict(series)
    new_series["frames"] = new_frames
    return new_series


def interpolate(series: PoseTimeSeries, cfg: Mapping[str, Any]) -> PoseTimeSeries:
    """Fill missing pose landmarks over time using linear interpolation.

    Parameters
    ----------
    series:
        Input PoseTimeSeries produced by pose extraction.
    cfg:
        Configuration dictionary (unused but accepted for signature stability).
    """
    del cfg  # Configuration currently unused; kept for future parity.

    frames = series.get("frames", []) or []
    frame_count = len(frames)
    if frame_count == 0:
        return dict(series)

    arrays = _collect_landmark_arrays(frames)
    indices = np.arange(frame_count)

    for arr in arrays.values():
        for axis in range(3):
            values = arr[:, axis]
            mask = np.isnan(values)
            if mask.all():
                continue
            valid_idx = indices[~mask]
            valid_values = values[~mask]
            arr[:, axis] = np.interp(indices, valid_idx, valid_values)

    return _rebuild_series(series, arrays)


def smooth(series: PoseTimeSeries, cfg: Mapping[str, Any]) -> PoseTimeSeries:
    """Apply exponential smoothing frame-by-frame for each landmark coordinate."""
    alpha = float(cfg.get("smoothing_alpha", 0.3)) if cfg is not None else 0.3
    alpha = float(np.clip(alpha, 1e-6, 1.0))

    frames = series.get("frames", []) or []
    frame_count = len(frames)
    if frame_count == 0:
        return dict(series)

    arrays = _collect_landmark_arrays(frames)

    for arr in arrays.values():
        for axis in range(3):
            values = arr[:, axis]
            mask = np.isnan(values)
            if mask.all():
                continue
            first_valid = int(np.flatnonzero(~mask)[0])
            prev = values[first_valid]
            values[first_valid] = prev
            if first_valid > 0:
                values[:first_valid] = prev
            for idx in range(first_valid + 1, frame_count):
                current = values[idx]
                if np.isnan(current):
                    current = prev
                prev = alpha * current + (1.0 - alpha) * prev
                values[idx] = prev

    return _rebuild_series(series, arrays)


def normalize(series: PoseTimeSeries, cfg: Mapping[str, Any]) -> PoseTimeSeries:
    """Center poses at hip midpoint and scale by torso length per frame."""
    del cfg  # Reserved for future runtime options.

    frames = series.get("frames", []) or []
    frame_count = len(frames)
    if frame_count == 0:
        return dict(series)

    arrays = _collect_landmark_arrays(frames)

    left_hip = arrays.get("LEFT_HIP")
    right_hip = arrays.get("RIGHT_HIP")
    left_shoulder = arrays.get("LEFT_SHOULDER")
    right_shoulder = arrays.get("RIGHT_SHOULDER")

    if left_hip is None:
        left_hip = np.full((frame_count, 3), np.nan)
    if right_hip is None:
        right_hip = np.full((frame_count, 3), np.nan)
    if left_shoulder is None:
        left_shoulder = np.full((frame_count, 3), np.nan)
    if right_shoulder is None:
        right_shoulder = np.full((frame_count, 3), np.nan)

    hip_mid = _nanmean_pair(left_hip, right_hip)
    shoulder_mid = _nanmean_pair(left_shoulder, right_shoulder)

    diff = shoulder_mid - hip_mid
    diff_sq = np.square(diff)
    diff_sq = np.where(np.isnan(diff_sq), 0.0, diff_sq)
    scale = np.sqrt(np.sum(diff_sq, axis=1))
    scale = np.maximum(scale, 1e-6)
    scale = scale.reshape(-1, 1)

    for name, arr in arrays.items():
        arrays[name] = (arr - hip_mid) / scale

    return _rebuild_series(series, arrays)


def clean(series: PoseTimeSeries, cfg: Mapping[str, Any]) -> PoseTimeSeries:
    """Composite preprocessing pipeline: interpolate → smooth → normalize."""
    interpolated = interpolate(series, cfg)
    smoothed = smooth(interpolated, cfg)
    normalized = normalize(smoothed, cfg)
    return normalized
