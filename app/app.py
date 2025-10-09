from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd

from ml import kpi, overlay, preprocess, segment  # noqa: E402
from ml.pose3d import extract  # noqa: E402

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
            st.session_state.pop("overlay_preview_meta", None)

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
    st.markdown("### Overlay preview")
    video_bytes_state = st.session_state.get("video_bytes")
    series_for_overlay = st.session_state.get("series_raw")
    if not video_bytes_state or series_for_overlay is None:
        st.info("Run an analysis to enable the overlay preview.")
    else:
        with st.form("overlay_preview_form"):
            preview_width = st.slider("Preview width (px)", 480, 960, 640, step=40)
            preview_fps = st.slider("Preview FPS", 6, 12, 8, step=1)
            preview_conf = st.slider("Confidence threshold", 0.2, 0.6, 0.3, step=0.05)
            render_clicked = st.form_submit_button("Render overlay preview")

        if render_clicked:
            with st.spinner("Rendering overlay preview…"):
                try:
                    preview_bytes = overlay.render_overlay(
                        video_bytes_state,
                        series_for_overlay,
                        cfg,
                        out_fps=float(preview_fps),
                        out_width=int(preview_width),
                        conf_thresh=float(preview_conf),
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Overlay rendering failed: {exc}")
                else:
                    st.session_state["overlay_preview_bytes"] = preview_bytes
                    st.session_state["overlay_preview_meta"] = {
                        "frames": len(series_for_overlay.get("frames", [])),
                        "fps": float(preview_fps),
                        "width": int(preview_width),
                        "conf": float(preview_conf),
                    }

        preview_bytes_state = st.session_state.get("overlay_preview_bytes")
        preview_meta_state = st.session_state.get("overlay_preview_meta") or {}
        if preview_bytes_state:
            st.video(preview_bytes_state)
            st.download_button(
                "Download overlay MP4",
                data=preview_bytes_state,
                file_name="overlay_preview.mp4",
                mime="video/mp4",
            )
            if preview_meta_state:
                frames_overlaid = preview_meta_state.get("frames")
                fps_preview = preview_meta_state.get("fps")
                width_preview = preview_meta_state.get("width")
                st.caption(
                    f"Rendered {frames_overlaid} frames at {fps_preview} fps, width {width_preview}px"
                )

    st.markdown("### Segmentation diagnostics")
    diagnostics = st.session_state.get("seg_diagnostics")
    if diagnostics:
        hip_signal = diagnostics.get("hip_signal")
        smooth = diagnostics.get("smooth")
        time_axis = diagnostics.get("t")
        hip_signal_arr = (
            np.asarray(hip_signal, dtype=float)
            if hip_signal is not None
            else np.array([], dtype=float)
        )
        smooth_arr = (
            np.asarray(smooth, dtype=float)
            if smooth is not None
            else np.array([], dtype=float)
        )
        time_axis_arr = (
            np.asarray(time_axis, dtype=float)
            if time_axis is not None
            else np.array([], dtype=float)
        )
        mins_idx = diagnostics.get("mins_idx")
        mins_idx_arr = (
            np.asarray(mins_idx, dtype=int)
            if mins_idx is not None
            else np.array([], dtype=int)
        )
        if (
            hip_signal_arr.size > 0
            and smooth_arr.size > 0
            and time_axis_arr.size > 0
        ):
            fig, ax = plt.subplots()
            ax.plot(time_axis_arr, hip_signal_arr, label="Hip signal")
            ax.plot(time_axis_arr, smooth_arr, label="Smoothed")
            if mins_idx_arr.size > 0:
                ax.scatter(
                    time_axis_arr[mins_idx_arr],
                    smooth_arr[mins_idx_arr],
                    marker="o",
                    label="Minima",
                )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Normalized height")
            ax.legend(loc="best")
            st.pyplot(fig)
            plt.close(fig)
            params = diagnostics.get("params", {})
            hip_drop = params.get("hip_min_drop_norm")
            prominence = params.get("peak_prominence")
            min_duration = params.get("min_rep_duration_s")
            window_size = params.get("window_size")
            hip_drop_str = f"{hip_drop:.3f}" if hip_drop is not None else "nan"
            prominence_str = f"{prominence:.3f}" if prominence is not None else "nan"
            min_duration_str = f"{min_duration:.2f}" if min_duration is not None else "nan"
            window_str = str(window_size) if window_size is not None else "?"
            st.caption(
                "Params used: "
                f"hip_drop={hip_drop_str}, "
                f"prominence={prominence_str}, "
                f"min_duration={min_duration_str}s, "
                f"window={window_str}"
            )
        else:
            st.info("Diagnostics unavailable for plotting.")
    else:
        st.info("Run an analysis to see segmentation diagnostics.")

with tabs[3]:
    st.write("Export options will appear once the full pipeline is available.")
