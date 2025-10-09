"""Utilities to render pose landmark overlays on video frames."""
from __future__ import annotations

import contextlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import cv2
import numpy as np

from . import pose3d

# Mapping of landmark names to their canonical indices.
LANDMARK_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(pose3d.LANDMARK_NAMES)}

# Minimal skeleton connectivity using BlazePose landmark names.
EDGES: Tuple[Tuple[str, str], ...] = (
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "LEFT_KNEE"),
    ("LEFT_KNEE", "LEFT_ANKLE"),
    ("LEFT_ANKLE", "LEFT_HEEL"),
    ("LEFT_HEEL", "LEFT_FOOT_INDEX"),
    ("RIGHT_HIP", "RIGHT_KNEE"),
    ("RIGHT_KNEE", "RIGHT_ANKLE"),
    ("RIGHT_ANKLE", "RIGHT_HEEL"),
    ("RIGHT_HEEL", "RIGHT_FOOT_INDEX"),
    ("NOSE", "LEFT_EYE"),
    ("LEFT_EYE", "LEFT_EAR"),
    ("NOSE", "RIGHT_EYE"),
    ("RIGHT_EYE", "RIGHT_EAR"),
)


@dataclass(frozen=True)
class _Point:
    x: int
    y: int
    confidence: float


def _frame_landmarks(
    frame_data: Dict[str, Any],
    width: int,
    height: int,
    conf_thresh: float,
) -> Dict[str, _Point]:
    """Convert normalized coordinates to pixel positions for a frame."""

    landmarks = frame_data.get("landmarks", {})
    converted: Dict[str, _Point] = {}
    for name, payload in landmarks.items():
        if name not in LANDMARK_INDEX:
            continue
        u = float(payload.get("u", 0.0))
        v = float(payload.get("v", 0.0))
        visibility = float(payload.get("visibility", 0.0))
        presence = float(payload.get("presence", 0.0))
        confidence = max(visibility, presence)
        if confidence < conf_thresh:
            continue
        px = int(round(np.clip(u, 0.0, 1.0) * max(width - 1, 1)))
        py = int(round(np.clip(v, 0.0, 1.0) * max(height - 1, 1)))
        if px < 0 or px >= width or py < 0 or py >= height:
            continue
        converted[name] = _Point(px, py, confidence)
    return converted


def _draw_overlay(
    frame: np.ndarray,
    points: Dict[str, _Point],
) -> None:
    """Draw skeleton edges and landmark points on the frame in-place."""

    # Lines
    for start_name, end_name in EDGES:
        start_point = points.get(start_name)
        end_point = points.get(end_name)
        if not start_point or not end_point:
            continue
        edge_conf = float(min(start_point.confidence, end_point.confidence))
        edge_conf = float(np.clip(edge_conf, 0.0, 1.0))
        intensity = int(round(180 + 60 * edge_conf))
        intensity = int(np.clip(intensity, 0, 255))
        tail = int(round(200 - 120 * edge_conf))
        tail = int(np.clip(tail, 0, 255))
        color = (0, intensity, tail)
        cv2.line(frame, (start_point.x, start_point.y), (end_point.x, end_point.y), color, 2, cv2.LINE_AA)

    # Points
    for point in points.values():
        radius = 3
        point_conf = float(np.clip(point.confidence, 0.0, 1.0))
        intensity = int(round(100 + 155 * point_conf))
        intensity = int(np.clip(intensity, 0, 255))
        color = (intensity, intensity, 0)
        cv2.circle(frame, (point.x, point.y), radius, color, -1, lineType=cv2.LINE_AA)


def _rotate_points(
    points: Dict[str, _Point],
    width: int,
    height: int,
    rotation: int,
) -> Dict[str, _Point]:
    """Rotate landmark pixel coordinates to match the rotated frame."""

    if rotation not in (90, 180, 270):
        return points

    rotated: Dict[str, _Point] = {}
    for name, point in points.items():
        if rotation == 90:
            new_x = max(0, min(height - 1, height - 1 - point.y))
            new_y = max(0, min(width - 1, point.x))
            rotated[name] = _Point(new_x, new_y, point.confidence)
        elif rotation == 180:
            new_x = max(0, min(width - 1, width - 1 - point.x))
            new_y = max(0, min(height - 1, height - 1 - point.y))
            rotated[name] = _Point(new_x, new_y, point.confidence)
        else:  # 270 degrees
            new_x = max(0, min(height - 1, point.y))
            new_y = max(0, min(width - 1, width - 1 - point.x))
            rotated[name] = _Point(new_x, new_y, point.confidence)
    return rotated


def render_overlay(
    video_input: pose3d.VideoInput,
    series: Dict[str, Any],
    cfg: Dict[str, Any],
    *,
    out_fps: float = 10.0,
    out_width: int = 640,
    conf_thresh: float = 0.3,
) -> Tuple[bytes, str, str]:
    """Render a downsampled overlay preview for the provided series.

    Returns
    -------
    Tuple[bytes, str, str]
        A tuple containing the encoded video bytes, the mime type, and the
        resulting file extension (without the dot).
    """

    if out_fps <= 0:
        raise ValueError("out_fps must be positive")
    if out_width <= 0:
        raise ValueError("out_width must be positive")

    frames_series: Iterable[Dict[str, Any]] = series.get("frames", [])  # type: ignore[assignment]
    frame_lookup = {
        int(entry.get("frame_idx")): entry
        for entry in frames_series
        if isinstance(entry.get("frame_idx"), int)
    }

    keep_original = isinstance(video_input, (str, Path))
    video_path, _ = pose3d._ensure_path(video_input, keep_temp=True)

    rotation = 0
    capture = cv2.VideoCapture(str(video_path))
    try:
        if capture.isOpened():
            orientation_prop = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
            if orientation_prop is not None:
                try:
                    rotation = int(capture.get(orientation_prop) or 0)
                except Exception:  # noqa: BLE001
                    rotation = 0
    finally:
        capture.release()

    rotation = int(rotation) % 360 if rotation else 0
    if rotation not in (90, 180, 270):
        rotation = 0

    rotation_code = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }.get(rotation)

    tmp_path: str | None = None
    mime_type = ""
    file_ext = ""
    writer: cv2.VideoWriter | None = None

    try:
        frame_count = 0
        frame_iter = pose3d.iter_frames(str(video_path), cfg)
        for frame_bgr, _timestamp_ms, frame_idx, _src_fps in frame_iter:
            orig_height, orig_width = frame_bgr.shape[:2]
            frame_to_draw = frame_bgr.copy()

            frame_entry = frame_lookup.get(frame_idx)
            points: Dict[str, _Point] = {}
            if frame_entry:
                points = _frame_landmarks(frame_entry, orig_width, orig_height, conf_thresh)

            if rotation_code is not None:
                frame_to_draw = cv2.rotate(frame_to_draw, rotation_code)
                if points:
                    points = _rotate_points(points, orig_width, orig_height, rotation)
            rotated_height, rotated_width = frame_to_draw.shape[:2]

            if points:
                _draw_overlay(frame_to_draw, points)

            scale = min(1.0, out_width / float(rotated_width)) if rotated_width > 0 else 1.0
            target_width = int(round(rotated_width * scale)) if rotated_width > 0 else out_width
            target_height = (
                int(round(rotated_height * scale)) if rotated_height > 0 else int(out_width * 9 / 16)
            )
            target_width = max(target_width, 1)
            target_height = max(target_height, 1)

            if writer is None:
                for fourcc_code, ext, mime in (("VP90", "webm", "video/webm"), ("VP80", "webm", "video/webm"), ("mp4v", "mp4", "video/mp4")):
                    candidate_file = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False)
                    candidate_path = candidate_file.name
                    candidate_file.close()

                    fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
                    candidate_writer = cv2.VideoWriter(
                        candidate_path,
                        fourcc,
                        float(out_fps),
                        (target_width, target_height),
                    )
                    if candidate_writer.isOpened():
                        writer = candidate_writer
                        tmp_path = candidate_path
                        mime_type = mime
                        file_ext = ext
                        break

                    candidate_writer.release()
                    with contextlib.suppress(FileNotFoundError):
                        os.unlink(candidate_path)

                if writer is None or tmp_path is None:
                    raise RuntimeError("Failed to open video writer for overlay preview.")

            if target_width != rotated_width or target_height != rotated_height:
                interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                frame_to_write = cv2.resize(
                    frame_to_draw,
                    (target_width, target_height),
                    interpolation=interpolation,
                )
            else:
                frame_to_write = frame_to_draw

            writer.write(frame_to_write)
            frame_count += 1

        if frame_count == 0:
            raise RuntimeError("No frames available for overlay rendering.")
    finally:
        if writer is not None:
            writer.release()
        if not keep_original:
            with contextlib.suppress(FileNotFoundError):
                Path(video_path).unlink()

    if tmp_path is None:
        raise RuntimeError("Overlay writer did not produce an output file.")

    try:
        with open(tmp_path, "rb") as f:
            data = f.read()
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_path)

    return data, mime_type, file_ext

