from __future__ import annotations

import base64
import contextlib
import copy
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
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
from ml.pose3d import extract, extract_parallel, iter_frames  # noqa: E402

CFG_PATH = ROOT / "config" / "default.yml"


@st.cache_resource
def load_cfg() -> Dict[str, Any]:
    with CFG_PATH.open("r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f) or {}
    models_cfg = cfg.setdefault("models", {})
    models_cfg.setdefault(
        "pose_landmarker_path", str(ROOT / "assets" / "models" / "pose_landmarker_lite.task")
    )
    performance_cfg = cfg.setdefault("performance", {})
    performance_cfg.setdefault("sample_rate", 1)
    preprocess_cfg = performance_cfg.setdefault("preprocess_size", {})
    preprocess_cfg.setdefault("longest_edge_px", 0)
    performance_cfg.setdefault("max_workers", 0)
    performance_cfg.setdefault("parallel", False)
    return cfg


cfg = load_cfg()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Squat MVP", layout="wide")
st.title("AI-Powered Exercise Technique Analyzer — MVP")
st.caption("Streamlit + MediaPipe 3D | CPU-only | Frontal & Lateral views")

DEFAULT_OVERLAY_WIDTH = 640
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


def _render_overlay_player(
    data: bytes, mime: str, *, max_width: int = DEFAULT_OVERLAY_WIDTH
) -> None:
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


def _safe_get(mapping: Any, key: str) -> Any:
    if mapping is None:
        return None
    getter = getattr(mapping, "get", None)
    if callable(getter):
        return getter(key)
    try:
        return mapping[key]
    except Exception:  # noqa: BLE001
        return None


def _ensure_array(value: Any, *, dtype: Any) -> np.ndarray:
    if value is None:
        return np.array([], dtype=dtype)
    try:
        arr = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError):
        return np.array([], dtype=dtype)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    else:
        arr = arr.reshape(-1)
    return arr.astype(dtype, copy=False)


def _configure_axes(ax: plt.Axes) -> None:
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.tick_params(labelsize=9)
    ax.margins(x=0)


def _render_empty_chart(container: Any, *, ylabel: str, caption: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10, color="#666666")
    container.pyplot(fig, clear_figure=True)
    plt.close(fig)
    container.caption(caption)


def _valid_indices(index_array: np.ndarray, upper: int) -> np.ndarray:
    if index_array.size == 0 or upper <= 0:
        return np.array([], dtype=int)
    idx = index_array.astype(int, copy=False)
    mask = (idx >= 0) & (idx < upper)
    if not np.any(mask):
        return np.array([], dtype=int)
    return np.unique(idx[mask])


def _plot_lateral_chart(container: Any, diagnostics: Dict[str, Any]) -> None:
    times = _ensure_array(_safe_get(diagnostics, "t"), dtype=float)
    hip_series = _safe_get(diagnostics, "smooth")
    if hip_series is None:
        hip_series = _safe_get(diagnostics, "hip_signal")

    y = _ensure_array(hip_series, dtype=float)
    n = min(times.size, y.size)
    if n == 0:
        _render_empty_chart(
            container,
            ylabel="Hip depth (% of clip range)",
            caption="Hip depth chart unavailable for this clip.",
        )
        return

    times = times[:n]
    y = y[:n]
    finite_y = y[np.isfinite(y)]
    if finite_y.size == 0:
        _render_empty_chart(
            container,
            ylabel="Hip depth (% of clip range)",
            caption="Hip depth chart unavailable for this clip.",
        )
        return

    highest = float(np.nanmax(finite_y))
    lowest = float(np.nanmin(finite_y))
    denom = highest - lowest
    if not np.isfinite(highest) or not np.isfinite(lowest) or abs(denom) <= 1e-9:
        _render_empty_chart(
            container,
            ylabel="Hip depth (% of clip range)",
            caption="Hip depth chart unavailable for this clip.",
        )
        return

    with np.errstate(invalid="ignore"):
        depth_pct = (highest - y) / denom * 100.0
    depth_pct = np.clip(depth_pct, -5.0, 105.0)

    valid_mask = np.isfinite(times) & np.isfinite(depth_pct)
    if not np.any(valid_mask):
        _render_empty_chart(
            container,
            ylabel="Hip depth (% of clip range)",
            caption="Hip depth chart unavailable for this clip.",
        )
        return

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(times, depth_pct, color="#1f77b4", linewidth=1.8, label="Hip depth")

    mins_idx = _ensure_array(_safe_get(diagnostics, "mins_idx"), dtype=int)
    valid_idx = _valid_indices(mins_idx, n)
    if valid_idx.size:
        marker_y = depth_pct[valid_idx]
        marker_t = times[valid_idx]
        marker_mask = np.isfinite(marker_t) & np.isfinite(marker_y)
        if np.any(marker_mask):
            ax.scatter(
                marker_t[marker_mask],
                marker_y[marker_mask],
                marker="v",
                s=36,
                color="#d62728",
                label="Rep bottom",
                zorder=5,
            )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hip depth (% of clip range)")
    ax.set_ylim(-2, 102)
    ax.set_yticks(np.linspace(0, 100, 6))
    _configure_axes(ax)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=9)
    container.pyplot(fig, clear_figure=True)
    plt.close(fig)
    container.caption("0% = highest hips; 100% = lowest (visualization only).")


def _plot_frontal_chart(
    container: Any,
    diagnostics: Dict[str, Any],
    frame_metrics_obj: Any,
) -> None:
    times = _ensure_array(_safe_get(diagnostics, "t"), dtype=float)

    series_map: Dict[str, np.ndarray] = {}
    if frame_metrics_obj is not None:
        left = _ensure_array(_safe_get(frame_metrics_obj, "knee_angle_deg_left"), dtype=float)
        right = _ensure_array(_safe_get(frame_metrics_obj, "knee_angle_deg_right"), dtype=float)
        if left.size:
            series_map["Left knee"] = left
        if right.size:
            series_map["Right knee"] = right

    if not series_map:
        _render_empty_chart(
            container,
            ylabel="Knee angle (°)",
            caption="Knee angle chart unavailable for this clip.",
        )
        return

    lengths = [times.size]
    lengths.extend(arr.size for arr in series_map.values())
    n = min(lengths)
    if n <= 0:
        _render_empty_chart(
            container,
            ylabel="Knee angle (°)",
            caption="Knee angle chart unavailable for this clip.",
        )
        return

    times = times[:n]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))

    for label, arr in series_map.items():
        values = arr[:n]
        if values.size != n:
            continue
        ax.plot(times, values, linewidth=1.6, label=label)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Knee angle (°)")
    ax.set_ylim(0, 180)
    _configure_axes(ax)
    ax.legend(loc="upper right", fontsize=9)
    container.pyplot(fig, clear_figure=True)
    plt.close(fig)

# Sidebar controls
view = st.sidebar.selectbox("Camera view", ["Lateral", "Frontal"], index=0)
auto_tune_enabled = st.sidebar.toggle("Auto-tune segmentation if 0 reps", value=True)

performance_defaults = cfg.get("performance", {}) or {}
preprocess_defaults = performance_defaults.get("preprocess_size", {}) or {}
default_sample_rate = int(performance_defaults.get("sample_rate", 1) or 1)
default_longest_edge = int(preprocess_defaults.get("longest_edge_px", 0) or 0)
default_parallel = bool(performance_defaults.get("parallel", False))
default_max_workers = int(performance_defaults.get("max_workers", 0) or 0)

st.sidebar.markdown("### Performance tuning")
sample_rate_sidebar = st.sidebar.number_input(
    "Frame sample rate (process every Nth frame)",
    min_value=1,
    step=1,
    value=default_sample_rate,
    help="Process every Nth frame; 1 keeps all frames.",
)
longest_edge_sidebar = st.sidebar.number_input(
    "Preprocess longest edge (px)",
    min_value=0,
    step=32,
    value=default_longest_edge,
    help="Resize frames before pose inference when the longest edge exceeds this value.",
)
parallel_enabled_sidebar = st.sidebar.toggle(
    "Enable parallel pose extraction",
    value=default_parallel,
)
if parallel_enabled_sidebar:
    max_workers_sidebar = st.sidebar.number_input(
        "Max workers (0 = auto)",
        min_value=0,
        step=1,
        value=default_max_workers,
    )
else:
    max_workers_sidebar = default_max_workers

st.sidebar.markdown("**Config snapshot**")
st.sidebar.json(
    {
        "fps_cap": cfg.get("fps_cap"),
        "smoothing_alpha": cfg.get("smoothing_alpha"),
        "performance": {
            "sample_rate": int(sample_rate_sidebar),
            "preprocess_longest_edge_px": int(longest_edge_sidebar),
            "parallel": bool(parallel_enabled_sidebar),
            "max_workers": int(max_workers_sidebar),
        },
    }
)

uploaded = st.file_uploader("Upload a short squat video (.mp4/.mov)", type=["mp4", "mov"])  # noqa: E501
analyze = st.button("Analyze", type="primary", disabled=uploaded is None)

tabs = st.tabs(["Overview", "Per-Rep", "Visuals", "Export"])  # placeholders

if analyze and uploaded is not None:
    st.session_state["video_bytes"] = uploaded.getvalue()
    uploaded.seek(0)
    session_cfg = copy.deepcopy(cfg)
    performance_cfg = session_cfg.setdefault("performance", {})
    preprocess_cfg = performance_cfg.setdefault("preprocess_size", {})
    performance_cfg["sample_rate"] = int(sample_rate_sidebar)
    preprocess_cfg["longest_edge_px"] = int(longest_edge_sidebar)
    performance_cfg["max_workers"] = int(max_workers_sidebar)
    performance_cfg["parallel"] = bool(parallel_enabled_sidebar)
    with st.spinner("Extracting 3D pose landmarks…"):
        try:
            if parallel_enabled_sidebar:
                series_raw = extract_parallel(uploaded, session_cfg)
            else:
                series_raw = extract(uploaded, session_cfg)
            series_interp = preprocess.interpolate(series_raw, session_cfg)
            series_smooth = preprocess.smooth(series_interp, session_cfg)
            rep_windows = segment.segment(series_smooth, session_cfg)
            tuned_params: Dict[str, float] = {}
            if auto_tune_enabled and len(rep_windows) == 0:
                rep_windows, tuned_params = segment.auto_tune(series_smooth, session_cfg)
            if tuned_params:
                seg_cfg_diag = dict(session_cfg.get("segmentation", {}))
                seg_cfg_diag.update(tuned_params)
                cfg_for_diag: Dict[str, Any] = dict(session_cfg)
                cfg_for_diag["segmentation"] = seg_cfg_diag
            else:
                cfg_for_diag = session_cfg
            diagnostics = segment.diagnose(series_smooth, cfg_for_diag)
            series_clean = preprocess.normalize(series_smooth, session_cfg)
            frame_metrics = kpi.frame_metrics(series_smooth, view)
            rep_metrics = kpi.compute(series_smooth, rep_windows, session_cfg, view)
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
            st.session_state["run_cfg"] = session_cfg
            st.session_state["seg_tuned_params"] = tuned_params
            st.session_state["seg_diagnostics"] = diagnostics
            st.session_state.pop("overlay_preview_bytes", None)
            st.session_state.pop("overlay_preview_mime", None)
            st.session_state.pop("overlay_preview_name", None)
            st.session_state.pop("overlay_preview_video_hash", None)
            st.session_state.pop("video_shape", None)
            _cache_video_shape(session_cfg)
            _ensure_overlay_preview(session_cfg)

with tabs[0]:
    if "series_raw" in st.session_state:
        series_raw = st.session_state["series_raw"]
        frames_processed = len(series_raw.get("frames", []))
        st.success(f"Frames processed: {frames_processed}")
        effective_fps_display = float(series_raw.get("fps", 0.0) or 0.0)
        orig_fps_display = float(series_raw.get("orig_fps", effective_fps_display) or 0.0)
        sample_rate_display = int(series_raw.get("sample_rate", 1) or 1)
        st.caption(
            f"Original FPS: {orig_fps_display:.1f} | Effective FPS: {effective_fps_display:.1f}"
        )
        st.caption(f"Frame sample rate: {sample_rate_display}")
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
    run_cfg = st.session_state.get("run_cfg") or cfg

    if series_for_overlay and not video_shape:
        _cache_video_shape(run_cfg)
        video_shape = st.session_state.get("video_shape")

    if series_for_overlay and (not overlay_bytes or not overlay_mime):
        _ensure_overlay_preview(run_cfg)
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
    series_raw_state = st.session_state.get("series_raw")
    rep_metrics_state = st.session_state.get("rep_metrics") or []
    view_state = st.session_state.get("view") or view
    run_cfg_state = st.session_state.get("run_cfg") or cfg

    if not series_raw_state:
        st.info("Run an analysis to enable exports.")
    else:
        performance_state = run_cfg_state.get("performance", {}) or {}
        preprocess_state = performance_state.get("preprocess_size", {}) or {}
        preprocess_series = series_raw_state.get("preprocess_size", {}) or {}
        longest_edge_value = preprocess_series.get(
            "longest_edge_px",
            preprocess_state.get("longest_edge_px", 0),
        )
        sample_rate_value = series_raw_state.get(
            "sample_rate", performance_state.get("sample_rate", 1)
        )
        timestamp_utc = datetime.now(timezone.utc)
        timestamp_str = timestamp_utc.strftime("%Y%m%dT%H%M%SZ")
        metadata = {
            "analysis_timestamp_utc": timestamp_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "view": str(view_state),
            "frames_processed": int(len(series_raw_state.get("frames", []))),
            "orig_fps": float(series_raw_state.get("orig_fps", 0.0) or 0.0),
            "fps": float(series_raw_state.get("fps", 0.0) or 0.0),
            "sample_rate": int(sample_rate_value or 1),
            "preprocess_longest_edge_px": int(longest_edge_value or 0),
            "parallel_enabled": bool(performance_state.get("parallel", False)),
            "parallel_max_workers": int(performance_state.get("max_workers", 0) or 0),
        }

        rep_df = pd.DataFrame(rep_metrics_state)
        if rep_df.empty:
            rep_df = pd.DataFrame([{**metadata}])
        else:
            rep_df = rep_df.copy()
            for key, value in metadata.items():
                rep_df[key] = value

        csv_bytes = rep_df.to_csv(index=False).encode("utf-8")
        json_payload = json.dumps(
            {
                "metadata": metadata,
                "rep_metrics": rep_metrics_state,
            },
            indent=2,
        ).encode("utf-8")

        st.markdown("#### Export metadata")
        st.json(metadata)
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name=f"results_{timestamp_str}.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download JSON",
            data=json_payload,
            file_name=f"results_{timestamp_str}.json",
            mime="application/json",
        )
