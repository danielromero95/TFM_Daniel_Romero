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

import pandas as pd

from ml import kpi, preprocess, segment  # noqa: E402
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

uploaded = st.file_uploader("Upload a short squat video (.mp4/.mov)", type=["mp4", "mov"])  # noqa: E501
analyze = st.button("Analyze", type="primary", disabled=uploaded is None)

tabs = st.tabs(["Overview", "Per-Rep", "Visuals", "Export"])  # placeholders

if analyze and uploaded is not None:
    with st.spinner("Extracting 3D pose landmarks…"):
        try:
            series_raw = extract(uploaded, cfg)
            series_interp = preprocess.interpolate(series_raw, cfg)
            series_smooth = preprocess.smooth(series_interp, cfg)
            rep_windows = segment.segment(series_smooth, cfg)
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
        if rep_windows_state is not None:
            rep_count = len(rep_windows_state)
            st.caption(f"Rep count: {rep_count}")
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
    st.write("Visuals — pose overlay and charts to be added in later epics.")

with tabs[3]:
    st.write("Export options will appear once the full pipeline is available.")
