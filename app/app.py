from __future__ import annotations

import base64
import contextlib
import hashlib
import logging
import sys
from pathlib import Path
from collections.abc import Iterable
from typing import Any, Dict, Tuple

import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd

from ml import kpi, overlay, preprocess, segment  # noqa: E402
from ml.pose3d import extract, iter_frames  # noqa: E402

CFG_PATH = ROOT / "config" / "default.yml"


@st.cache_resource
def load_cfg() -> Dict[str, Any]:
    with CFG_PATH.open("r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f) or {}
    models_cfg = cfg.setdefault("models", {})
    models_cfg.setdefault(
        "pose_landmarker_path", str(ROOT / "assets" / "models" / "pose_landmarker_lite.task")
    )
    return cfg


cfg = load_cfg()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Squat MVP", layout="wide")
st.title("AI-Powered Exercise Technique Analyzer — MVP")
st.caption("Streamlit + MediaPipe 3D | CPU-only | Frontal & Lateral views")

DEFAULT_OVERLAY_WIDTH = 720
DEFAULT_OVERLAY_FPS = 8.0
DEFAULT_OVERLAY_CONF = 0.3


def _first_frame_shape(video_bytes: bytes, cfg: Dict[str, Any]) -> Tuple[int, int]:
    iterator = iter_frames(video_bytes, cfg)
    try:
        frame_bgr, *_ = next(iterator)
    except StopIteration:
        return 0, 0
    finally:
        with contextlib.suppress(Exception):
            iterator.close()  # type: ignore[attr-defined]

    height, width = frame_bgr.shape[:2]
    return int(height), int(width)


def _cache_video_shape(cfg: Dict[str, Any]) -> None:
    video_bytes = st.session_state.get("video_bytes")
    if not video_bytes:
        st.session_state.pop("video_shape", None)
        return

    if "video_shape" in st.session_state and st.session_state["video_shape"]:
        return

    try:
        height, width = _first_frame_shape(video_bytes, cfg)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unable to determine video dimensions: %s", exc)
        st.session_state["video_shape"] = (0, 0)
    else:
        st.session_state["video_shape"] = (height, width)


def _ensure_overlay_preview(cfg: Dict[str, Any]) -> None:
    video_bytes = st.session_state.get("video_bytes")
    series_raw = st.session_state.get("series_raw")
    if not video_bytes or series_raw is None:
        return

    video_hash = hashlib.md5(video_bytes).hexdigest()
    cached_hash = st.session_state.get("overlay_preview_video_hash")
    if (
        st.session_state.get("overlay_preview_bytes")
        and st.session_state.get("overlay_preview_mime")
        and cached_hash == video_hash
    ):
        return

    try:
        with st.spinner("Rendering overlay preview…"):
            data, mime, ext = overlay.render_overlay(
                video_bytes,
                series_raw,
                cfg,
                out_fps=DEFAULT_OVERLAY_FPS,
                out_width=DEFAULT_OVERLAY_WIDTH,
                conf_thresh=DEFAULT_OVERLAY_CONF,
            )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Overlay rendering failed: {exc}")
        logger.exception("Overlay rendering failed")
        st.session_state.pop("overlay_preview_bytes", None)
        st.session_state.pop("overlay_preview_mime", None)
        st.session_state.pop("overlay_preview_name", None)
        st.session_state.pop("overlay_preview_video_hash", None)
        return

    st.session_state["overlay_preview_bytes"] = data
    st.session_state["overlay_preview_mime"] = mime
    st.session_state["overlay_preview_name"] = (
        f"overlay_preview.{ext}" if ext else "overlay_preview"
    )
    st.session_state["overlay_preview_video_hash"] = video_hash


def _render_overlay_player(data: bytes, mime: str, *, max_width: int = 720) -> None:
    if not data or not mime:
        return

    try:
        encoded = base64.b64encode(data).decode("utf-8")
    except Exception:  # noqa: BLE001
        st.video(data, format=mime)
        return

    video_html = f"""
        <video controls playsinline style="width: min(100%, {max_width}px); border-radius: 4px;">
            <source src="data:{mime};base64,{encoded}" type="{mime}">
            Your browser does not support the video tag.
        </video>
    """
    st.markdown(video_html, unsafe_allow_html=True)


def _plot_lateral_chart(container: Any, diagnostics: Dict[str, Any]) -> None:
    times = np.asarray(diagnostics.get("t") or [], dtype=float)
    hip_series = diagnostics.get("smooth")
    if hip_series is None:
        hip_series = diagnostics.get("hip_signal")

    y = np.asarray(hip_series or [], dtype=float)
    n = min(times.size, y.size)
    if n == 0:
        container.info("Hip depth chart unavailable for this clip.")
        return

    times = times[:n]
    y = y[:n]
    max_y = float(np.nanmax(y))
    min_y = float(np.nanmin(y))
    if not np.isfinite(max_y) or not np.isfinite(min_y) or abs(max_y - min_y) < 1e-9:
        container.info("Hip depth chart unavailable for this clip.")
        return

    depth_pct = (max_y - y) / (max_y - min_y + 1e-9) * 100.0

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(times, depth_pct)

    mins_idx = diagnostics.get("mins_idx")
    if isinstance(mins_idx, Iterable):  # type: ignore[arg-type]
        valid_idx = [
            int(idx)
            for idx in mins_idx
            if isinstance(idx, (int, np.integer)) and 0 <= int(idx) < n
        ]
        if valid_idx:
            idx_array = np.asarray(valid_idx, dtype=int)
            ax.plot(times[idx_array], depth_pct[idx_array], marker="v", linestyle="None", markersize=6)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hip depth (% of clip range)")
    ax.set_ylim(0, 100)
    ax.set_yticks(np.linspace(0, 100, 6))
    container.pyplot(fig, clear_figure=True)
    plt.close(fig)
    container.caption("0% = highest hips; 100% = lowest (visualization only).")


def _plot_frontal_chart(
    container: Any,
    diagnostics: Dict[str, Any],
    frame_metrics_obj: Any,
) -> None:
    times = np.asarray(diagnostics.get("t") or [], dtype=float)

    knee_angles = np.array([], dtype=float)
    if frame_metrics_obj is not None:
        knee_angles = np.asarray(frame_metrics_obj.get("knee_angle_deg"), dtype=float)
        if knee_angles.size == 0:
            left = np.asarray(frame_metrics_obj.get("knee_angle_deg_left"), dtype=float)
            right = np.asarray(frame_metrics_obj.get("knee_angle_deg_right"), dtype=float)
            candidates = [arr for arr in (left, right) if arr.size]
            if len(candidates) == 2:
                with np.errstate(invalid="ignore"):
                    knee_angles = np.nanmean(np.stack(candidates), axis=0)
            elif candidates:
                knee_angles = candidates[0]

    n = min(times.size, knee_angles.size)
    if n == 0:
        container.info("Knee angle chart unavailable for this clip.")
        return

    times = times[:n]
    knee_angles = knee_angles[:n]
    if not np.isfinite(knee_angles).any():
        container.info("Knee angle chart unavailable for this clip.")
        return

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(times, knee_angles)

    mins_idx = diagnostics.get("mins_idx")
    if isinstance(mins_idx, Iterable):  # type: ignore[arg-type]
        valid_idx = [
            int(idx)
            for idx in mins_idx
            if isinstance(idx, (int, np.integer)) and 0 <= int(idx) < n
        ]
        if valid_idx:
            idx_array = np.asarray(valid_idx, dtype=int)
            ax.plot(times[idx_array], knee_angles[idx_array], marker="v", linestyle="None", markersize=6)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Knee flexion (°)")
    ax.set_ylim(0, 180)
    container.pyplot(fig, clear_figure=True)
    plt.close(fig)

# Sidebar controls
view = st.sidebar.selectbox("Camera view", ["Lateral", "Frontal"], index=0)
st.sidebar.markdown("**Config snapshot**")
st.sidebar.json({"fps_cap": cfg.get("fps_cap"), "smoothing_alpha": cfg.get("smoothing_alpha")})
auto_tune_enabled = st.sidebar.toggle("Auto-tune segmentation if 0 reps", value=True)

uploaded = st.file_uploader("Upload a short squat video (.mp4/.mov)", type=["mp4", "mov"])  # noqa: E501
analyze = st.button("Analyze", type="primary", disabled=uploaded is None)

tabs = st.tabs(["Overview", "Per-Rep", "Visuals", "Export"])  # placeholders

if analyze and uploaded is not None:
    st.session_state["video_bytes"] = uploaded.getvalue()
    uploaded.seek(0)
    with st.spinner("Extracting 3D pose landmarks…"):
        try:
            series_raw = extract(uploaded, cfg)
            series_interp = preprocess.interpolate(series_raw, cfg)
            series_smooth = preprocess.smooth(series_interp, cfg)
            rep_windows = segment.segment(series_smooth, cfg)
            tuned_params: Dict[str, float] = {}
            if auto_tune_enabled and len(rep_windows) == 0:
                rep_windows, tuned_params = segment.auto_tune(series_smooth, cfg)
            if tuned_params:
                seg_cfg_diag = dict(cfg.get("segmentation", {}))
                seg_cfg_diag.update(tuned_params)
                cfg_for_diag: Dict[str, Any] = dict(cfg)
                cfg_for_diag["segmentation"] = seg_cfg_diag
            else:
                cfg_for_diag = cfg
            diagnostics = segment.diagnose(series_smooth, cfg_for_diag)
            series_clean = preprocess.normalize(series_smooth, cfg)
            frame_metrics = kpi.frame_metrics(series_smooth, view)
            rep_metrics = kpi.compute(series_smooth, rep_windows, cfg, view)
        except FileNotFoundError as exc:
            st.error(f"Model missing: {exc}")
            logger.exception("Pose extraction failed: model missing")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Pose extraction failed: {exc}")
            logger.exception("Pose extraction failed")
        else:
            st.session_state["series_raw"] = series_raw
            st.session_state["series_smooth"] = series_smooth
            st.session_state["series_clean"] = series_clean
            st.session_state["rep_windows"] = rep_windows
            st.session_state["frame_metrics"] = frame_metrics
            st.session_state["rep_metrics"] = rep_metrics
            st.session_state["view"] = view
            st.session_state["seg_tuned_params"] = tuned_params
            st.session_state["seg_diagnostics"] = diagnostics
            st.session_state.pop("overlay_preview_bytes", None)
            st.session_state.pop("overlay_preview_mime", None)
            st.session_state.pop("overlay_preview_name", None)
            st.session_state.pop("overlay_preview_video_hash", None)
            st.session_state.pop("video_shape", None)
            _cache_video_shape(cfg)
            _ensure_overlay_preview(cfg)

with tabs[0]:
    if "series_raw" in st.session_state:
        series_raw = st.session_state["series_raw"]
        frames_processed = len(series_raw.get("frames", []))
        st.success(f"Frames processed: {frames_processed}")
        st.caption(f"Effective FPS: {series_raw.get('fps', 0.0):.1f}")
        if "series_clean" in st.session_state:
            series_clean = st.session_state["series_clean"]
            st.caption(f"Cleaned frames: {len(series_clean.get('frames', []))}")
        rep_windows_state = st.session_state.get("rep_windows")
        rep_metrics_state = st.session_state.get("rep_metrics") or []
        tuned_params_state = st.session_state.get("seg_tuned_params") or {}
        if rep_windows_state is not None:
            rep_count = len(rep_windows_state)
            st.caption(f"Rep count: {rep_count}")
            if tuned_params_state:
                hip_drop = tuned_params_state.get("hip_min_drop_norm")
                prominence = tuned_params_state.get("peak_prominence")
                if hip_drop is not None and prominence is not None:
                    st.caption(f"Auto-tuned: hip_drop={hip_drop:.3f}, prominence={prominence:.3f}")
            if rep_metrics_state:
                rep_df = pd.DataFrame(rep_metrics_state)
                summary_cols = {
                    "Avg min knee angle": rep_df.get("min_knee_angle_deg"),
                    "Avg min hip angle": rep_df.get("min_hip_angle_deg"),
                    "Avg trunk max angle": rep_df.get("trunk_max_angle_deg"),
                    "Avg tempo down": rep_df.get("tempo_down_s"),
                    "Avg tempo up": rep_df.get("tempo_up_s"),
                }
                summary_items = []
                for label, series_values in summary_cols.items():
                    if series_values is not None and not series_values.empty:
                        summary_items.append(f"{label}: {series_values.mean():.2f}")
                if summary_items:
                    st.caption(" | ".join(summary_items))
    else:
        st.info("Upload a video and click Analyze to see results.")

with tabs[1]:
    rep_metrics = st.session_state.get("rep_metrics") or []
    rep_windows = st.session_state.get("rep_windows")
    if rep_metrics:
        rep_df = pd.DataFrame(rep_metrics)
        desired_cols = [
            "rep_id",
            "start_t",
            "end_t",
            "duration_s",
            "tempo_down_s",
            "tempo_up_s",
            "min_knee_angle_deg",
            "min_hip_angle_deg",
            "trunk_max_angle_deg",
            "rom_knee_deg",
            "rom_hip_deg",
        ]
        available_cols = [col for col in desired_cols if col in rep_df.columns]
        st.dataframe(rep_df[available_cols], use_container_width=True)
    elif rep_windows is not None:
        st.warning("No reps detected.")
    else:
        st.info("Run an analysis to populate per-rep metrics.")

with tabs[2]:
    st.markdown("### Visuals")
    overlay_bytes = st.session_state.get("overlay_preview_bytes")
    overlay_mime = st.session_state.get("overlay_preview_mime")
    overlay_name = st.session_state.get("overlay_preview_name") or "overlay_preview"
    diagnostics = st.session_state.get("seg_diagnostics") or {}
    series_for_overlay = st.session_state.get("series_raw")
    video_shape = st.session_state.get("video_shape")
    frame_metrics_state = st.session_state.get("frame_metrics")
    analyzed_view = st.session_state.get("view") or view

    if series_for_overlay and not video_shape:
        _cache_video_shape(cfg)
        video_shape = st.session_state.get("video_shape")

    if series_for_overlay and (not overlay_bytes or not overlay_mime):
        _ensure_overlay_preview(cfg)
        overlay_bytes = st.session_state.get("overlay_preview_bytes")
        overlay_mime = st.session_state.get("overlay_preview_mime")
        overlay_name = st.session_state.get("overlay_preview_name") or overlay_name

    if not overlay_bytes or not overlay_mime:
        st.info("Run an analysis to see the overlay and diagnostics.")
    else:
        left_col, right_col = st.columns([3, 2], gap="medium")
        with left_col:
            _render_overlay_player(overlay_bytes, overlay_mime, max_width=DEFAULT_OVERLAY_WIDTH)
            st.download_button(
                "Download overlay",
                data=overlay_bytes,
                file_name=overlay_name,
                mime=overlay_mime,
            )

        with right_col:
            normalized_view = (analyzed_view or "").strip().lower()
            if normalized_view == "frontal":
                _plot_frontal_chart(right_col, diagnostics, frame_metrics_state)
            else:
                _plot_lateral_chart(right_col, diagnostics)

            params = diagnostics.get("params") or {}
            if params:
                hip_drop = params.get("hip_min_drop_norm")
                prominence = params.get("peak_prominence")
                min_duration = params.get("min_rep_duration_s")
                window_size = params.get("window_size")
                hip_drop_str = f"{hip_drop:.3f}" if hip_drop is not None else "nan"
                prominence_str = f"{prominence:.3f}" if prominence is not None else "nan"
                min_duration_str = f"{min_duration:.2f}" if min_duration is not None else "nan"
                try:
                    window_str = str(int(window_size))
                except (TypeError, ValueError):
                    window_str = "?"
                right_col.caption(
                    "Segmentation params: "
                    f"hip_drop={hip_drop_str}, "
                    f"prominence={prominence_str}, "
                    f"min_duration={min_duration_str}s, "
                    f"window={window_str}"
                )

with tabs[3]:
    st.write("Export options will appear once the full pipeline is available.")
