import yaml
import streamlit as st

from ml.pose3d import extract

st.set_page_config(page_title="Squat MVP", layout="wide")


@st.cache_resource
def load_cfg():
    with open("config/default.yml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _clear_error() -> None:
    if "analysis_error" in st.session_state:
        del st.session_state["analysis_error"]


def _set_error(message: str) -> None:
    st.session_state["analysis_error"] = message
    if "series" in st.session_state:
        del st.session_state["series"]


cfg = load_cfg()

st.title("AI-Powered Exercise Technique Analyzer — MVP")
st.caption("Streamlit + MediaPipe 3D | CPU-only | Frontal & Lateral views")

view = st.sidebar.selectbox("Camera view", ["Lateral", "Frontal"], index=0)
st.session_state["view"] = view
st.sidebar.markdown("**Config snapshot**")
st.sidebar.json({"fps_cap": cfg.get("fps_cap"), "smoothing_alpha": cfg.get("smoothing_alpha")})

uploaded = st.file_uploader("Upload a short squat video (.mp4/.mov)", type=["mp4", "mov"])
analyze = st.button("Analyze", type="primary", disabled=uploaded is None)

tabs = st.tabs(["Overview", "Per-Rep", "Visuals", "Export"])

if analyze and uploaded is not None:
    with st.spinner("Running pose extraction…"):
        try:
            series = extract(uploaded, cfg)
        except Exception as exc:  # pragma: no cover - UI side effect
            _set_error(str(exc))
        else:
            st.session_state["series"] = series
            _clear_error()

series = st.session_state.get("series")
error = st.session_state.get("analysis_error")

with tabs[0]:
    if series is not None:
        st.success(f"Frames processed: {len(series['frames'])}")
    elif error:
        st.error(error)
    else:
        st.info("Upload a video and click Analyze to see results.")

with tabs[1]:
    st.write("Per-rep table — pending downstream pipeline")

with tabs[2]:
    st.write("Visuals — pending downstream pipeline")

with tabs[3]:
    st.write("Export options will appear once the pipeline is complete.")
