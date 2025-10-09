"""Pose3D extraction utilities using MediaPipe Tasks Pose Landmarker 3D."""
from __future__ import annotations

import contextlib
import logging
import math
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple, Union

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
PoseFrame = Dict[str, Any]
PoseTimeSeries = Dict[str, Any]


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


def _performance_settings(cfg: Dict[str, Any]) -> Tuple[int, int, int]:
    """Return sanitized performance settings (sample rate, resize, workers)."""
    performance_cfg = cfg.get("performance") or {}

    sample_rate_raw = performance_cfg.get("sample_rate", 1)
    try:
        sample_rate = int(sample_rate_raw)
    except (TypeError, ValueError):
        sample_rate = 1
    sample_rate = max(1, sample_rate)

    preprocess_cfg = performance_cfg.get("preprocess_size") or {}
    longest_edge_raw = preprocess_cfg.get("longest_edge_px", 0)
    try:
        longest_edge_px = int(longest_edge_raw)
    except (TypeError, ValueError):
        longest_edge_px = 0
    longest_edge_px = max(0, longest_edge_px)

    workers_raw = performance_cfg.get("max_workers", 0)
    try:
        max_workers = int(workers_raw)
    except (TypeError, ValueError):
        max_workers = 0

    return sample_rate, longest_edge_px, max_workers


def _resolve_capture_fps(capture: cv2.VideoCapture, cfg: Dict[str, Any]) -> float:
    """Determine the source FPS, falling back to cfg.fps_cap when invalid."""
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(fps) or fps <= 0.0:
        fallback_fps = float(cfg.get("fps_cap", 30.0)) or 30.0
        LOGGER.warning("Video reports invalid FPS; falling back to %.2f fps.", fallback_fps)
        fps = fallback_fps
    return fps


def _timestamp_ms(frame_idx: int, fps: float) -> int:
    """Compute timestamp in milliseconds for the provided frame index and FPS."""
    if fps <= 0.0:
        return int(round(frame_idx * 1000.0))
    return int(round(frame_idx / fps * 1000.0))


def _resize_frame_if_needed(frame_bgr: np.ndarray, longest_edge_px: int) -> np.ndarray:
    """Resize the frame so that its longest edge does not exceed the target size."""
    if longest_edge_px <= 0:
        return frame_bgr

    height, width = frame_bgr.shape[:2]
    longest_edge = max(height, width)
    if longest_edge <= 0 or longest_edge <= longest_edge_px:
        return frame_bgr

    scale = float(longest_edge_px) / float(longest_edge)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(frame_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized


def _landmarker_options(model_path: Path) -> vision.PoseLandmarkerOptions:
    """Create Pose Landmarker options for the provided model path."""
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    return vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )


def _process_pose_frame(
    landmarker: vision.PoseLandmarker,
    frame_bgr: np.ndarray,
    frame_idx: int,
    timestamp_ms: int,
    longest_edge_px: int,
) -> PoseFrame:
    """Run pose inference on a single frame and format the result."""
    processed = _resize_frame_if_needed(frame_bgr, longest_edge_px)
    frame_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    pose_world = result.pose_world_landmarks or []
    pose_norm = result.pose_landmarks or []

    world_list = _as_landmark_list(pose_world[0]) if pose_world else []
    norm_list = _as_landmark_list(pose_norm[0]) if pose_norm else []

    timestamp_s = timestamp_ms / 1000.0
    if not world_list:
        LOGGER.warning(
            "No pose detected at frame %d (t=%.3fs).",
            frame_idx,
            timestamp_s,
        )
        return {
            "frame_idx": frame_idx,
            "t": timestamp_s,
            "landmarks": {},
            "conf": 0.0,
        }

    landmarks_payload: Dict[str, Dict[str, float]] = {}
    visibility_values: List[float] = []

    for idx, name in enumerate(LANDMARK_NAMES):
        if idx >= len(world_list):
            break
        wlm = world_list[idx]
        nvis = 0.0
        npres = 0.0
        nu = 0.0
        nv = 0.0
        if idx < len(norm_list):
            nlm = norm_list[idx]
            nvis = float(getattr(nlm, "visibility", 0.0))
            npres = float(getattr(nlm, "presence", 0.0))
            visibility_values.append(nvis)
            nu = float(getattr(nlm, "x", 0.0))
            nv = float(getattr(nlm, "y", 0.0))
            nu = float(np.clip(nu, 0.0, 1.0))
            nv = float(np.clip(nv, 0.0, 1.0))

        landmarks_payload[name] = {
            "x": float(getattr(wlm, "x", 0.0)),
            "y": float(getattr(wlm, "y", 0.0)),
            "z": float(getattr(wlm, "z", 0.0)),
            "u": nu,
            "v": nv,
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

    return {
        "frame_idx": frame_idx,
        "t": timestamp_s,
        "landmarks": landmarks_payload,
        "conf": conf,
    }


def _resolve_worker_count(max_workers_cfg: int) -> int:
    """Determine the number of parallel workers to launch."""
    if max_workers_cfg and max_workers_cfg > 0:
        return max(1, max_workers_cfg)
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def _parallel_worker(args: Tuple[str, str, Sequence[Tuple[int, int]], float, int, int]) -> List[PoseFrame]:
    """Worker entry point for ProcessPoolExecutor."""
    video_path, model_path, chunk, orig_fps, sample_rate, longest_edge_px = args

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video in worker: {video_path}")

    effective_fps = orig_fps / sample_rate if sample_rate > 0 else orig_fps
    if effective_fps <= 0.0:
        effective_fps = orig_fps if orig_fps > 0 else 30.0

    options = _landmarker_options(Path(model_path))
    frames: List[PoseFrame] = []

    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            for kept_idx, frame_number in chunk:
                capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_number))
                ok, frame_bgr = capture.read()
                if not ok or frame_bgr is None:
                    continue
                timestamp_ms = _timestamp_ms(kept_idx, effective_fps)
                frame = _process_pose_frame(
                    landmarker,
                    frame_bgr,
                    kept_idx,
                    timestamp_ms,
                    longest_edge_px,
                )
                frames.append(frame)
    finally:
        capture.release()

    return frames


def iter_frames(video_input: VideoInput, cfg: Dict[str, Any]) -> Iterable[FrameTuple]:
    """Yield sampled frames from the video according to cfg.performance.sample_rate."""
    keep_temp = bool(cfg.get("keep_intermediates", False))
    video_path, cleanup = _ensure_path(video_input, keep_temp)

    LOGGER.info("Opening video: %s", video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        cleanup()
        raise RuntimeError(f"Unable to open video: {video_path}")

    sample_rate, _, _ = _performance_settings(cfg)

    try:
        orig_fps = _resolve_capture_fps(capture, cfg)
        effective_fps = orig_fps / sample_rate if sample_rate > 0 else orig_fps
        LOGGER.info(
            "Iterating video at %.2f fps with sample_rate=%d (effective %.2f fps).",
            orig_fps,
            sample_rate,
            effective_fps,
        )

        frame_counter = -1
        kept_idx = 0
        while True:
            grabbed = capture.grab()
            if not grabbed:
                break
            frame_counter += 1
            if frame_counter % sample_rate != 0:
                continue
            ok, frame = capture.retrieve()
            if not ok or frame is None:
                continue
            timestamp_ms = _timestamp_ms(kept_idx, effective_fps)
            yield frame, timestamp_ms, kept_idx, orig_fps
            kept_idx += 1
    finally:
        capture.release()
        cleanup()


def extract(video_input: VideoInput, cfg: Dict[str, Any]) -> PoseTimeSeries:
    """Extract 3D pose landmarks from the provided video sequentially."""
    model_cfg = cfg.get("models", {})
    model_path_value = model_cfg.get("pose_landmarker_path")
    if not model_path_value:
        raise KeyError("Missing cfg['models']['pose_landmarker_path'] entry.")

    model_path = Path(model_path_value)
    if not model_path.is_file():
        raise FileNotFoundError(f"Pose landmarker model not found: {model_path}")

    sample_rate, longest_edge_px, _ = _performance_settings(cfg)
    options = _landmarker_options(model_path)

    frames: List[PoseFrame] = []
    orig_fps_value: float | None = None
    effective_fps = 0.0

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for frame_bgr, timestamp_ms, frame_idx, orig_fps in iter_frames(video_input, cfg):
            if orig_fps_value is None:
                orig_fps_value = float(orig_fps)
                effective_fps = orig_fps_value / sample_rate if sample_rate > 0 else orig_fps_value
            frame = _process_pose_frame(
                landmarker,
                frame_bgr,
                frame_idx,
                timestamp_ms,
                longest_edge_px,
            )
            frames.append(frame)

    if not frames:
        raise RuntimeError("Pose extraction yielded no frames. Ensure the subject is visible.")

    if orig_fps_value is None or orig_fps_value <= 0.0:
        orig_fps_value = float(cfg.get("fps_cap", 30.0)) or 30.0
    if effective_fps <= 0.0:
        effective_fps = orig_fps_value / sample_rate if sample_rate > 0 else orig_fps_value

    LOGGER.info(
        "Pose extraction complete: %d frames processed at %.2f fps (effective %.2f fps).",
        len(frames),
        orig_fps_value,
        effective_fps,
    )

    return {
        "orig_fps": float(orig_fps_value),
        "fps": float(effective_fps),
        "sample_rate": int(sample_rate),
        "preprocess_size": {"longest_edge_px": int(longest_edge_px)},
        "frames": frames,
    }


def extract_parallel(video_input: VideoInput, cfg: Dict[str, Any]) -> PoseTimeSeries:
    """Extract 3D pose landmarks using multiple worker processes."""
    model_cfg = cfg.get("models", {})
    model_path_value = model_cfg.get("pose_landmarker_path")
    if not model_path_value:
        raise KeyError("Missing cfg['models']['pose_landmarker_path'] entry.")

    model_path = Path(model_path_value)
    if not model_path.is_file():
        raise FileNotFoundError(f"Pose landmarker model not found: {model_path}")

    sample_rate, longest_edge_px, max_workers_cfg = _performance_settings(cfg)
    keep_temp = bool(cfg.get("keep_intermediates", False))
    video_path, cleanup = _ensure_path(video_input, keep_temp)

    try:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")
        try:
            orig_fps_value = _resolve_capture_fps(capture, cfg)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            capture.release()

        if frame_count <= 0:
            LOGGER.warning("Frame count unavailable; falling back to sequential extraction.")
            return extract(video_path, cfg)

        indices = list(range(0, frame_count, sample_rate))
        if len(indices) <= 1:
            LOGGER.info("Clip too short for parallel processing; using sequential extractor.")
            return extract(video_path, cfg)

        workers = _resolve_worker_count(max_workers_cfg)
        if workers <= 1:
            LOGGER.info("Parallel disabled (workers=%d); using sequential extractor.", workers)
            return extract(video_path, cfg)

        chunk_size = math.ceil(len(indices) / workers)
        if chunk_size <= 0:
            chunk_size = len(indices)

        chunks: List[List[Tuple[int, int]]] = []
        for start in range(0, len(indices), chunk_size):
            chunk_indices = indices[start : start + chunk_size]
            chunk_pairs = [
                (kept_idx, frame_number)
                for kept_idx, frame_number in enumerate(chunk_indices, start)
            ]
            if chunk_pairs:
                chunks.append(chunk_pairs)

        if len(chunks) <= 1:
            LOGGER.info("Single chunk after splitting; using sequential extractor.")
            return extract(video_path, cfg)

        workers = min(workers, len(chunks))
        effective_fps = orig_fps_value / sample_rate if sample_rate > 0 else orig_fps_value
        if effective_fps <= 0.0:
            effective_fps = orig_fps_value if orig_fps_value > 0 else 30.0

        tasks = [
            (
                str(video_path),
                str(model_path),
                chunk,
                float(orig_fps_value),
                sample_rate,
                longest_edge_px,
            )
            for chunk in chunks
        ]

        pose_frames: List[PoseFrame] = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for chunk_frames in executor.map(_parallel_worker, tasks):
                pose_frames.extend(chunk_frames)

        pose_frames.sort(key=lambda item: item.get("frame_idx", 0))

        LOGGER.info(
            "Parallel pose extraction complete: %d frames processed at %.2f fps (effective %.2f fps) using %d workers.",
            len(pose_frames),
            orig_fps_value,
            effective_fps,
            workers,
        )

        if not pose_frames:
            raise RuntimeError("Parallel pose extraction yielded no frames.")

        return {
            "orig_fps": float(orig_fps_value),
            "fps": float(effective_fps),
            "sample_rate": int(sample_rate),
            "preprocess_size": {"longest_edge_px": int(longest_edge_px)},
            "frames": pose_frames,
        }
    finally:
        cleanup()
