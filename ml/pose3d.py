"""Pose3D extraction utilities using MediaPipe Tasks Pose Landmarker 3D."""
from __future__ import annotations

import contextlib
import logging
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

LOGGER = logging.getLogger(__name__)

# BlazePose landmark names (33 points) in canonical order.
LANDMARK_NAMES: Tuple[str, ...] = (
    "NOSE",
    "LEFT_EYE_INNER",
    "LEFT_EYE",
    "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER",
    "RIGHT_EYE",
    "RIGHT_EYE_OUTER",
    "LEFT_EAR",
    "RIGHT_EAR",
    "MOUTH_LEFT",
    "MOUTH_RIGHT",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_PINKY",
    "RIGHT_PINKY",
    "LEFT_INDEX",
    "RIGHT_INDEX",
    "LEFT_THUMB",
    "RIGHT_THUMB",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
)

FrameTuple = Tuple[np.ndarray, int, int, float]
VideoInput = Union[str, Path, bytes, "UploadedFile", "BytesIO"]


def _as_landmark_list(entry: Any) -> List[Any]:
    """Return a list of landmark-like objects from MediaPipe results."""
    if entry is None:
        return []
    if hasattr(entry, "landmark"):
        return list(getattr(entry, "landmark")) or []
    try:
        return list(entry)
    except TypeError:
        return []


def _ensure_path(video_input: VideoInput, keep_temp: bool) -> Tuple[Path, Callable[[], None]]:
    """Persist the provided video input to disk and return its path and cleanup."""
    cleanup: Callable[[], None] = lambda: None

    if isinstance(video_input, (str, Path)):
        path = Path(video_input)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")
        return path, cleanup

    if isinstance(video_input, bytes):
        data = video_input
    elif hasattr(video_input, "read"):
        # Streamlit's UploadedFile behaves like BytesIO. Ensure pointer reset afterwards.
        data = video_input.read()  # type: ignore[assignment]
        with contextlib.suppress(Exception):
            video_input.seek(0)  # type: ignore[attr-defined]
    else:
        raise TypeError("Unsupported video_input type; expected path or file-like object.")

    if not data:
        raise ValueError("Received empty video input stream.")

    suffix = ".mp4"
    name_attr = getattr(video_input, "name", "")
    if isinstance(name_attr, str):
        candidate = Path(name_attr).suffix
        if candidate:
            suffix = candidate

    tmp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp_file.write(data)
        tmp_file.flush()
        tmp_path = Path(tmp_file.name)
    finally:
        tmp_file.close()

    def _cleanup() -> None:
        if keep_temp:
            return
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()

    return tmp_path, _cleanup


def iter_frames(video_input: VideoInput, cfg: Dict[str, Any]) -> Iterable[FrameTuple]:
    """Yield frames, timestamps, and source FPS from a video input.

    Parameters
    ----------
    video_input:
        Path-like string or file-like object representing a video.
    cfg:
        Runtime configuration dictionary containing `fps_cap` and `keep_intermediates` keys.

    Yields
    ------
    Tuple[np.ndarray, int, int, float]
        A tuple containing the BGR frame, timestamp in milliseconds, frame index,
        and source frames-per-second value.
    """
    keep_temp = bool(cfg.get("keep_intermediates", False))
    video_path, cleanup = _ensure_path(video_input, keep_temp)

    LOGGER.info("Opening video: %s", video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        cleanup()
        raise RuntimeError(f"Unable to open video: {video_path}")

    try:
        src_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if src_fps <= 0:
            fallback_fps = float(cfg.get("fps_cap", 30.0)) or 30.0
            LOGGER.warning(
                "Video reports invalid FPS; falling back to %.2f fps.",
                fallback_fps,
            )
            src_fps = fallback_fps

        fps_cap = float(cfg.get("fps_cap", src_fps))
        if fps_cap <= 0:
            LOGGER.warning("fps_cap <= 0 detected; processing all frames.")
            fps_cap = src_fps

        step = max(1, round(src_fps / fps_cap)) if fps_cap > 0 else 1
        LOGGER.info(
            "Iterating video at %.2f fps with downsample step %d (cap=%.2f).",
            src_fps,
            step,
            fps_cap,
        )

        frame_idx = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_idx % step == 0:
                if src_fps > 0:
                    timestamp_ms = int(round(frame_idx / src_fps * 1000.0))
                else:
                    timestamp_ms = int(frame_idx * 1000.0)
                yield frame, timestamp_ms, frame_idx, src_fps
            frame_idx += 1
    finally:
        capture.release()
        cleanup()


def extract(video_input: VideoInput, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract 3D pose landmarks from the provided video.

    Returns a PoseTimeSeries dictionary matching the repository contract.
    """
    model_cfg = cfg.get("models", {})
    model_path_value = model_cfg.get("pose_landmarker_path")
    if not model_path_value:
        raise KeyError("Missing cfg['models']['pose_landmarker_path'] entry.")

    model_path = Path(model_path_value)
    if not model_path.is_file():
        raise FileNotFoundError(f"Pose landmarker model not found: {model_path}")

    LOGGER.info("Loading MediaPipe Pose Landmarker model from %s", model_path)

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

    frames: List[Dict[str, Any]] = []
    src_fps_value: float | None = None

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for frame_bgr, timestamp_ms, frame_idx, src_fps in iter_frames(video_input, cfg):
            if src_fps_value is None:
                src_fps_value = src_fps
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            pose_world = result.pose_world_landmarks or []
            pose_norm = result.pose_landmarks or []

            world_list = _as_landmark_list(pose_world[0]) if pose_world else []
            norm_list = _as_landmark_list(pose_norm[0]) if pose_norm else []

            if not world_list:
                LOGGER.warning(
                    "No pose detected at frame %d (t=%.3fs).", frame_idx, timestamp_ms / 1000.0
                )
                frames.append(
                    {
                        "frame_idx": frame_idx,
                        "t": timestamp_ms / 1000.0,
                        "landmarks": {},
                        "conf": 0.0,
                    }
                )
                continue

            landmarks_payload: Dict[str, Dict[str, float]] = {}
            visibility_values: List[float] = []

            for idx, name in enumerate(LANDMARK_NAMES):
                if idx >= len(world_list):
                    break
                wlm = world_list[idx]
                nvis = 0.0
                npres = 0.0
                if idx < len(norm_list):
                    nlm = norm_list[idx]
                    nvis = float(getattr(nlm, "visibility", 0.0))
                    npres = float(getattr(nlm, "presence", 0.0))
                    visibility_values.append(nvis)

                landmarks_payload[name] = {
                    "x": float(getattr(wlm, "x", 0.0)),
                    "y": float(getattr(wlm, "y", 0.0)),
                    "z": float(getattr(wlm, "z", 0.0)),
                    "visibility": nvis,
                    "presence": npres,
                }

            conf = float(np.mean(visibility_values)) if visibility_values else 0.0
            if conf < 0.3:
                LOGGER.warning(
                    "Low confidence (%.2f) at frame %d. Consider better lighting or positioning.",
                    conf,
                    frame_idx,
                )

            frames.append(
                {
                    "frame_idx": frame_idx,
                    "t": timestamp_ms / 1000.0,
                    "landmarks": landmarks_payload,
                    "conf": conf,
                }
            )

    if not frames:
        raise RuntimeError("Pose extraction yielded no frames. Ensure the subject is visible.")

    if not src_fps_value or src_fps_value <= 0:
        src_fps_value = float(cfg.get("fps_cap", 30.0)) or 30.0

    fps_cap = float(cfg.get("fps_cap", src_fps_value))
    if fps_cap <= 0:
        fps_cap = src_fps_value
    step = max(1, round(src_fps_value / fps_cap)) if fps_cap > 0 else 1
    effective_fps = src_fps_value / step if step > 0 else src_fps_value

    LOGGER.info(
        "Pose extraction complete: %d frames processed at %.2f fps (effective %.2f fps).",
        len(frames),
        src_fps_value,
        effective_fps,
    )

    return {
        "fps": float(effective_fps),
        "frames": frames,
    }

