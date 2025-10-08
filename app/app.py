import io
import time
import yaml
import streamlit as st

st.set_page_config(page_title="Squat MVP", layout="wide")

# Load config once
@st.cache_resource
def load_cfg():
    with open("config/default.yml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = load_cfg()

st.title("AI-Powered Exercise Technique Analyzer — MVP")
st.caption("Streamlit + MediaPipe 3D | CPU-only | Frontal & Lateral views")


# Sidebar controls
view = st.sidebar.selectbox("Camera view", ["Lateral", "Frontal"], index=0)
st.sidebar.markdown("**Config snapshot**")  # light peek for debugging
st.sidebar.json({"fps_cap": cfg.get("fps_cap"), "smoothing_alpha": cfg.get("smoothing_alpha")})

uploaded = st.file_uploader("Upload a short squat video (.mp4/.mov)", type=["mp4", "mov"])  # noqa: E501

analyze = st.button("Analyze", type="primary", disabled=uploaded is None)

tabs = st.tabs(["Overview", "Per-Rep", "Visuals", "Export"])  # placeholders

if analyze and uploaded is not None:
    with st.spinner("Running analysis (pose → preprocess → segment → KPIs → faults)…"):
        time.sleep(0.5)  # placeholder for pipeline latency
        # TODO: call the pipeline functions once implemented (per 02_TASK_LIST.md)
        # from ml.pose3d import extract
        # from ml.preprocess import clean
        # from ml.segment import segment
        # from ml.kpi import compute
        # from ml.faults import detect
        # from ml.feedback import to_text
        # series = extract(uploaded, cfg)
        # series = clean(series, cfg)
        # windows = segment(series, cfg)
        # rep_metrics = compute(series, windows, cfg, view)
        # flags = [detect(m, cfg, view) for m in rep_metrics]
        # feedback = [to_text(f, view) for f in flags]
        st.session_state["_demo_results"] = {
            "rep_count": 3,
            "avg_knee_angle_deg": 98.7,
            "faults": {"insufficient_depth": 1, "knee_valgus": 0, "excessive_trunk_lean": 1},
            "view": view
        }

if "_demo_results" in st.session_state:
    res = st.session_state["_demo_results"]
    with tabs[0]:
        col1, col2, col3 = st.columns(3)
        col1.metric("Rep count", res["rep_count"])
        col2.metric("Avg knee angle (°)", f"{res['avg_knee_angle_deg']:.1f}")
        col3.json(res["faults"])

    with tabs[1]:
        st.write("Per-rep table — placeholder (to be filled by pipeline output)")

    with tabs[2]:
        st.write("Visuals — placeholder for overlay player and angle charts")  # charts must use matplotlib without custom colors

    with tabs[3]:
        st.download_button("Download CSV (placeholder)", data="rep,depth_ok\n1,True\n2,False\n3,True\n", file_name="results_demo.csv", mime="text/csv")  # noqa: E501
else:
    with tabs[0]:
        st.info("Upload a video and click Analyze to see results.")
