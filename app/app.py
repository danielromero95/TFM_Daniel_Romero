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
            series = extract(uploaded, cfg)
        except FileNotFoundError as exc:
            st.error(f"Model missing: {exc}")
            logger.exception("Pose extraction failed: model missing")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Pose extraction failed: {exc}")
            logger.exception("Pose extraction failed")
        else:
            st.session_state["series"] = series
            st.session_state["view"] = view

with tabs[0]:
    if "series" in st.session_state:
        series = st.session_state["series"]
        frames_processed = len(series.get("frames", []))
        st.success(f"Frames processed: {frames_processed}")
        st.caption(f"Effective FPS: {series.get('fps', 0.0):.1f}")
    else:
        st.info("Upload a video and click Analyze to see results.")

with tabs[1]:
    st.write("Per-rep table — pending downstream pipeline implementation.")

with tabs[2]:
    st.write("Visuals — pose overlay and charts to be added in later epics.")

with tabs[3]:
    st.write("Export options will appear once the full pipeline is available.")
