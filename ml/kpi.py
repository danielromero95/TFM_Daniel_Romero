"""KPI computation utilities for squat analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Tuple

import numpy as np

from . import segment

PoseTimeSeries = Dict[str, Any]
RepWindow = Mapping[str, Any]
RepMetrics = Dict[str, float]


@dataclass(frozen=True)
class FrameMetrics:
    """Container for per-frame KPI series."""

    values: Dict[str, np.ndarray]

    def get(self, key: str) -> np.ndarray:
        return self.values.get(key, np.empty((0,), dtype=float))

    @property
    def frame_count(self) -> int:
        if not self.values:
            return 0
        any_series = next(iter(self.values.values()))
        return int(any_series.shape[0])


def _init_array(frame_count: int) -> np.ndarray:
    return np.full((frame_count, 3), np.nan, dtype=float)


def _extract_landmarks(series: PoseTimeSeries, names: Iterable[str]) -> Dict[str, np.ndarray]:
    frames: Iterable[MutableMapping[str, Any]] = series.get("frames", []) or []
    frame_list = list(frames)
    frame_count = len(frame_list)
    arrays = {name: _init_array(frame_count) for name in names}

    axis_keys = ("x", "y", "z")
    for idx, frame in enumerate(frame_list):
        landmarks = frame.get("landmarks", {}) or {}
        for name in names:
            if name not in landmarks:
                continue
            point = landmarks.get(name) or {}
            arr = arrays[name]
            for axis, axis_key in enumerate(axis_keys):
                value = point.get(axis_key)
                arr[idx, axis] = float(value) if value is not None else np.nan

    return arrays


def _angle_deg(vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
    if vec_a.size == 0 or vec_b.size == 0:
        return np.array([], dtype=float)

    dot = np.sum(vec_a * vec_b, axis=1)
    norm_a = np.linalg.norm(vec_a, axis=1)
    norm_b = np.linalg.norm(vec_b, axis=1)
    denom = norm_a * norm_b

    valid = (denom > 0.0) & ~np.isnan(dot)
    cos = np.full(dot.shape, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos[valid] = dot[valid] / denom[valid]
    valid_cos = ~np.isnan(cos)
    cos[valid_cos] = np.clip(cos[valid_cos], -1.0, 1.0)

    angles = np.full(dot.shape, np.nan, dtype=float)
    angles[valid_cos] = np.degrees(np.arccos(cos[valid_cos]))
    return angles


def _nanmean_pair(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if first.size == 0:
        return second.copy()
    if second.size == 0:
        return first.copy()
    stacked = np.stack([first, second])
    with np.errstate(invalid="ignore"):
        return np.nanmean(stacked, axis=0)


def _frame_metrics_lateral(series: PoseTimeSeries) -> FrameMetrics:
    landmark_names = (
        "LEFT_HIP",
        "RIGHT_HIP",
        "LEFT_KNEE",
        "RIGHT_KNEE",
        "LEFT_ANKLE",
        "RIGHT_ANKLE",
        "LEFT_SHOULDER",
        "RIGHT_SHOULDER",
    )
    arrays = _extract_landmarks(series, landmark_names)

    left_hip = arrays.get("LEFT_HIP", np.empty((0, 3)))
    right_hip = arrays.get("RIGHT_HIP", np.empty((0, 3)))
    left_knee = arrays.get("LEFT_KNEE", np.empty((0, 3)))
    right_knee = arrays.get("RIGHT_KNEE", np.empty((0, 3)))
    left_ankle = arrays.get("LEFT_ANKLE", np.empty((0, 3)))
    right_ankle = arrays.get("RIGHT_ANKLE", np.empty((0, 3)))
    left_shoulder = arrays.get("LEFT_SHOULDER", np.empty((0, 3)))
    right_shoulder = arrays.get("RIGHT_SHOULDER", np.empty((0, 3)))

    left_thigh = left_hip - left_knee
    left_shank = left_ankle - left_knee
    right_thigh = right_hip - right_knee
    right_shank = right_ankle - right_knee

    left_knee_angle = _angle_deg(left_thigh, left_shank)
    right_knee_angle = _angle_deg(right_thigh, right_shank)

    left_torso = left_shoulder - left_hip
    right_torso = right_shoulder - right_hip
    left_hip_vec = left_knee - left_hip
    right_hip_vec = right_knee - right_hip

    left_hip_angle = _angle_deg(left_torso, left_hip_vec)
    right_hip_angle = _angle_deg(right_torso, right_hip_vec)

    hip_mid = _nanmean_pair(left_hip, right_hip)
    shoulder_mid = _nanmean_pair(left_shoulder, right_shoulder)
    trunk_vec = shoulder_mid - hip_mid
    frame_count = trunk_vec.shape[0] if trunk_vec.ndim == 2 else 0
    vertical_vec = np.tile(np.array([[0.0, -1.0, 0.0]], dtype=float), (frame_count, 1))
    trunk_angle = _angle_deg(trunk_vec, vertical_vec)

    with np.errstate(invalid="ignore"):
        knee_angle = np.nanmean(np.stack([left_knee_angle, right_knee_angle]), axis=0)
        hip_angle = np.nanmean(np.stack([left_hip_angle, right_hip_angle]), axis=0)

    return FrameMetrics(
        {
            "knee_angle_deg_left": left_knee_angle,
            "knee_angle_deg_right": right_knee_angle,
            "knee_angle_deg": knee_angle,
            "hip_angle_deg_left": left_hip_angle,
            "hip_angle_deg_right": right_hip_angle,
            "hip_angle_deg": hip_angle,
            "trunk_angle_deg": trunk_angle,
        }
    )


def _frame_metrics_frontal(series: PoseTimeSeries) -> FrameMetrics:
    landmark_names = (
        "LEFT_KNEE",
        "RIGHT_KNEE",
        "LEFT_ANKLE",
        "RIGHT_ANKLE",
        "LEFT_HIP",
        "RIGHT_HIP",
    )
    arrays = _extract_landmarks(series, landmark_names)

    left_knee = arrays.get("LEFT_KNEE", np.empty((0, 3)))
    right_knee = arrays.get("RIGHT_KNEE", np.empty((0, 3)))
    left_ankle = arrays.get("LEFT_ANKLE", np.empty((0, 3)))
    right_ankle = arrays.get("RIGHT_ANKLE", np.empty((0, 3)))
    left_hip = arrays.get("LEFT_HIP", np.empty((0, 3)))
    right_hip = arrays.get("RIGHT_HIP", np.empty((0, 3)))

    lateral_axis = 0  # X-axis captures medial-lateral deviation.
    left_offset = left_knee[:, lateral_axis] - left_ankle[:, lateral_axis]
    right_offset = right_knee[:, lateral_axis] - right_ankle[:, lateral_axis]
    hip_width = np.abs(left_hip[:, lateral_axis] - right_hip[:, lateral_axis])

    with np.errstate(invalid="ignore"):
        normalized_left = np.divide(left_offset, hip_width, where=hip_width > 0)
        normalized_right = np.divide(right_offset, hip_width, where=hip_width > 0)
        symmetry = np.abs(normalized_left - normalized_right)

    return FrameMetrics(
        {
            "valgus_offset_left": normalized_left,
            "valgus_offset_right": normalized_right,
            "valgus_symmetry": symmetry,
        }
    )


def frame_metrics(series: PoseTimeSeries, view: str) -> FrameMetrics:
    view_normalized = (view or "").strip().lower()
    if view_normalized == "frontal":
        return _frame_metrics_frontal(series)
    return _frame_metrics_lateral(series)


def _slice_indices(window: RepWindow, frame_count: int) -> Tuple[int, int]:
    start_idx = int(window.get("start_idx", 0) or 0)
    end_idx = int(window.get("end_idx", start_idx) or start_idx)
    start_idx = int(np.clip(start_idx, 0, max(frame_count - 1, 0)))
    end_idx = int(np.clip(end_idx, start_idx, max(frame_count - 1, start_idx)))
    return start_idx, end_idx


def angles_lateral(series: PoseTimeSeries, windows: List[RepWindow], cfg: Mapping[str, Any]) -> List[RepMetrics]:
    del cfg
    metrics = frame_metrics(series, "lateral")
    frame_count = metrics.frame_count
    results: List[RepMetrics] = []

    knee_angles = metrics.get("knee_angle_deg")
    hip_angles = metrics.get("hip_angle_deg")
    trunk_angles = metrics.get("trunk_angle_deg")

    for window in windows:
        start_idx, end_idx = _slice_indices(window, frame_count)
        idx_slice = slice(start_idx, end_idx + 1)

        min_knee = float(np.nanmin(knee_angles[idx_slice])) if knee_angles.size else float("nan")
        min_hip = float(np.nanmin(hip_angles[idx_slice])) if hip_angles.size else float("nan")
        max_trunk = float(np.nanmax(trunk_angles[idx_slice])) if trunk_angles.size else float("nan")

        results.append(
            {
                "rep_id": int(window.get("rep_id", len(results) + 1)),
                "min_knee_angle_deg": min_knee,
                "min_hip_angle_deg": min_hip,
                "trunk_max_angle_deg": max_trunk,
            }
        )

    return results


def alignment_frontal(series: PoseTimeSeries, windows: List[RepWindow], cfg: Mapping[str, Any]) -> List[RepMetrics]:
    del cfg
    metrics = frame_metrics(series, "frontal")
    frame_count = metrics.frame_count
    symmetry = metrics.get("valgus_symmetry")
    left_offset = metrics.get("valgus_offset_left")
    right_offset = metrics.get("valgus_offset_right")

    results: List[RepMetrics] = []
    for window in windows:
        start_idx, end_idx = _slice_indices(window, frame_count)
        idx_slice = slice(start_idx, end_idx + 1)

        rep_symmetry = float(np.nanmean(symmetry[idx_slice])) if symmetry.size else float("nan")
        left_max = float(np.nanmax(np.abs(left_offset[idx_slice]))) if left_offset.size else float("nan")
        right_max = float(np.nanmax(np.abs(right_offset[idx_slice]))) if right_offset.size else float("nan")

        results.append(
            {
                "rep_id": int(window.get("rep_id", len(results) + 1)),
                "valgus_symmetry": rep_symmetry,
                "valgus_offset_left_max": left_max,
                "valgus_offset_right_max": right_max,
            }
        )

    return results


def tempo_rom(
    series: PoseTimeSeries,
    windows: List[RepWindow],
    cfg: Mapping[str, Any],
    metrics: FrameMetrics | None = None,
) -> List[RepMetrics]:
    fps = float(series.get("fps", 0.0) or 0.0)
    if fps <= 0.0:
        fps = float(cfg.get("fps_cap", 30.0) or 30.0)

    if metrics is None:
        metrics = frame_metrics(series, "lateral")
    frame_count = metrics.frame_count

    hip_angles = metrics.get("hip_angle_deg")
    knee_angles = metrics.get("knee_angle_deg")
    hip_signal = segment.hip_vertical_signal(series)

    results: List[RepMetrics] = []
    for window in windows:
        start_idx, end_idx = _slice_indices(window, frame_count)
        idx_slice = slice(start_idx, end_idx + 1)

        duration = max(0.0, (end_idx - start_idx) / fps) if end_idx > start_idx else 0.0

        hip_slice = hip_signal[idx_slice] if hip_signal.size else np.array([], dtype=float)
        bottom_idx = start_idx
        if hip_slice.size:
            valid_mask = ~np.isnan(hip_slice)
            if valid_mask.any():
                relative_min_idx = int(np.nanargmin(hip_slice))
                bottom_idx = start_idx + relative_min_idx
            else:
                bottom_idx = start_idx + (end_idx - start_idx) // 2

        tempo_down = max(0.0, (bottom_idx - start_idx) / fps) if bottom_idx >= start_idx else 0.0
        tempo_up = max(0.0, (end_idx - bottom_idx) / fps) if end_idx >= bottom_idx else 0.0

        hip_rom = 0.0
        knee_rom = 0.0
        if hip_angles.size:
            hip_vals = hip_angles[idx_slice]
            hip_rom = float(np.nanmax(hip_vals) - np.nanmin(hip_vals))
        if knee_angles.size:
            knee_vals = knee_angles[idx_slice]
            knee_rom = float(np.nanmax(knee_vals) - np.nanmin(knee_vals))

        results.append(
            {
                "rep_id": int(window.get("rep_id", len(results) + 1)),
                "tempo_down_s": float(tempo_down),
                "tempo_up_s": float(tempo_up),
                "rom_hip_deg": float(hip_rom),
                "rom_knee_deg": float(knee_rom),
                "duration_s": float(duration),
            }
        )

    return results


def _merge_metrics(lists: List[List[RepMetrics]]) -> List[RepMetrics]:
    merged: Dict[int, Dict[str, float]] = {}
    for metrics_list in lists:
        for entry in metrics_list:
            rep_id = int(entry.get("rep_id", 0))
            if rep_id not in merged:
                merged[rep_id] = {"rep_id": rep_id}
            merged[rep_id].update(entry)
    return [merged[key] for key in sorted(merged)]


def compute(series: PoseTimeSeries, windows: List[RepWindow], cfg: Mapping[str, Any], view: str) -> List[RepMetrics]:
    if not windows:
        return []

    metrics = frame_metrics(series, view)

    tempo_metrics = tempo_rom(series, windows, cfg, metrics)

    view_normalized = (view or "").strip().lower()
    if view_normalized == "frontal":
        angle_metrics = alignment_frontal(series, windows, cfg)
    else:
        angle_metrics = angles_lateral(series, windows, cfg)

    combined = _merge_metrics([tempo_metrics, angle_metrics])

    # Preserve start/end timestamps when available.
    for window in windows:
        rep_id = int(window.get("rep_id", 0))
        for entry in combined:
            if entry.get("rep_id") == rep_id:
                entry.setdefault("start_t", float(window.get("start_t", 0.0) or 0.0))
                entry.setdefault("end_t", float(window.get("end_t", 0.0) or 0.0))
                entry.setdefault("duration_s", float(window.get("end_t", 0.0) or 0.0) - float(window.get("start_t", 0.0) or 0.0))
                break

    return combined
