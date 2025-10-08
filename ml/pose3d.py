"""Pose extraction utilities using MediaPipe Pose Landmarker 3D."""

from __future__ import annotations

import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.solutions.pose import PoseLandmark
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

__all__ = ["iter_frames", "extract"]

logger = logging.getLogger(__name__)


VideoInput = Union[str, os.PathLike[str], io.BufferedIOBase, io.BytesIO]
FrameYield = Tuple[np.ndarray, int, int, float]


def _resolve_video_input(video_input: VideoInput) -> Tuple[str, Optional[Callable[[], None]]]:
    """Return a filesystem path for ``video_input`` and an optional cleanup callback."""

    if isinstance(video_input, (str, os.PathLike)):
        path = Path(video_input).expanduser().resolve()
        return str(path), None

    if hasattr(video_input, "read"):
        data = video_input.read()
        if hasattr(video_input, "seek"):
            try:
                video_input.seek(0)
            except (OSError, AttributeError):
                pass
        suffix = Path(getattr(video_input, "name", "upload.mp4")).suffix or ".mp4"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            tmp.write(data)
            tmp.flush()
        finally:
            tmp.close()

        def _cleanup() -> None:
            try:
                os.remove(tmp.name)
            except OSError:
                pass

        return tmp.name, _cleanup

    raise TypeError("Unsupported video input type; expected path or file-like object")


def _compute_fps(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps) or fps <= 0:
        logger.warning("Falling back to default FPS=30.0 (metadata missing)")
        return 30.0
    return float(fps)


def _fps_step(src_fps: float, fps_cap: float) -> int:
    if fps_cap <= 0:
        return 1
    return max(1, int(round(src_fps / fps_cap)))


def iter_frames(video_input: VideoInput, cfg: Dict[str, Any]) -> Iterable[FrameYield]:
    """Yield frames from ``video_input`` with timestamps and frame indices.

    Parameters
    ----------
    video_input:
        Filesystem path or file-like object containing a video.
    cfg:
        Runtime configuration dictionary. Must contain ``fps_cap``.

    Yields
    ------
    tuple[np.ndarray, int, int, float]
        Frame in BGR, timestamp (ms), frame index, and source FPS.
    """

    video_path, cleanup = _resolve_video_input(video_input)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if cleanup:
            cleanup()
        raise RuntimeError(f"Could not open video file: {video_path}")

    src_fps = _compute_fps(cap)
    fps_cap = float(cfg.get("fps_cap", src_fps))
    step = _fps_step(src_fps, fps_cap)
    logger.info(
        "Iterating frames with src_fps=%.2f, fps_cap=%.2f, step=%d", src_fps, fps_cap, step
    )

    def _generator() -> Iterator[FrameYield]:
        try:
            frame_idx = 0
            while True:
                success, frame = cap.read()
                if not success or frame is None:
                    break

                if frame_idx % step == 0:
                    t_ms = int(round((frame_idx / src_fps) * 1000.0))
                    yield frame, t_ms, frame_idx, src_fps

                frame_idx += 1
        finally:
            cap.release()
            if cleanup:
                cleanup()

    return _generator()


def extract(video_input: VideoInput, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a :class:`PoseTimeSeries` using MediaPipe Pose Landmarker 3D."""

    model_path = cfg.get("models", {}).get("pose_landmarker_path")
    if not model_path:
        raise FileNotFoundError("Pose Landmarker model path missing in configuration")

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Pose Landmarker model not found: {model_path}")

    fps_cap = float(cfg.get("fps_cap", 30.0))

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False,
    )

    frames: list[Dict[str, Any]] = []
    src_fps: Optional[float] = None

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for frame_bgr, t_ms, frame_idx, src_fps_value in iter_frames(video_input, cfg):
            if src_fps is None:
                src_fps = src_fps_value
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect_for_video(mp_image, t_ms)

            pose_frame: Dict[str, Any] = {
                "frame_idx": int(frame_idx),
                "t": float(t_ms) / 1000.0,
                "landmarks": {},
                "conf": 0.0,
            }

            if result.pose_world_landmarks:
                world_landmarks = result.pose_world_landmarks[0]
                pose_frame["landmarks"] = {
                    PoseLandmark(i).name: {
                        "x": float(lmk.x),
                        "y": float(lmk.y),
                        "z": float(lmk.z),
                        "visibility": float(getattr(lmk, "visibility", 1.0)),
                        "presence": float(getattr(lmk, "presence", 1.0)),
                    }
                    for i, lmk in enumerate(world_landmarks.landmark)
                }
                pose_frame["conf"] = 1.0
            else:
                logger.warning("No pose detected at frame_idx=%d (t=%.3fs)", frame_idx, pose_frame["t"])

            frames.append(pose_frame)

    if src_fps is None:
        raise RuntimeError("No frames processed; video may be empty or unreadable")

    step = _fps_step(src_fps, fps_cap)
    effective_fps = src_fps / step

    logger.info("Pose extraction complete: %d frames (effective_fps=%.2f)", len(frames), effective_fps)

    return {
        "fps": float(effective_fps),
        "frames": frames,
    }
