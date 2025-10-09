# 02_TASK_LIST — The “How‑To” (MVP, Small Tasks for Agents)

**Purpose:** a concise backlog of **small, self‑contained tasks** for agents to implement the MVP in `00_PROJECT_OVERVIEW.md`, aligned with `01_PROJECT_STRUCTURE.md`.

**Legend:** `[Agent]` tasks can be done by agents. `[Human]` tasks require manual steps (env, credentials, consented media).

**Key clarifications (applied below):**
- **Multi‑view support**: user selects **Lateral** (sagittal) or **Frontal** view in the UI; faults evaluated accordingly.
- **Explicit config passing**: all `ml/` functions that need settings **must accept `cfg`**; no global file reads.

---

## Epic 0 — Setup & Foundations
**Goal:** minimal working skeleton to run Streamlit and commit code consistently.

- [Human] **T00.1 — Create environment**
  - Add `environment.yml` (mamba/conda, CPU‑only).
  - Commands: `mamba env create -f environment.yml` → `mamba activate squat-mvp`.
  - _DoD:_ `python -c "import streamlit, cv2, mediapipe, numpy, pandas"` runs.

- [Agent] **T00.2 — Repo files**
  - Create `.gitignore` (Python, Streamlit, logs, reports).
  - Add empty folders: `reports/`, `logs/`, `assets/`.
  - _DoD:_ Git status clean; folders present.

- [Human] **T00.3 — Demo clips (optional)**
  - Place 1–2 short, consented clips in `assets/demo_clips/` (one **lateral**, one **frontal**).
  - _DoD:_ Files accessible locally; not committed if consent unclear.

---

## Epic 1 — Streamlit Shell
**Goal:** single‑page app with upload → analyze, config load, **view selector**, and tab placeholders.

- [Agent] **T01.1 — App entry**
  - File: `app/app.py` → set page config (title, layout).

- [Agent] **T01.2 — Config load**
  - Read `config/default.yml` once; create `cfg` object; pass `cfg` to pipeline calls.

- [Agent] **T01.3 — Uploader & actions**
  - `st.file_uploader` (mp4/mov), “Analyze” button, `st.spinner` during run.

- [Agent] **T01.4 — Tabs**
  - Tabs: **Overview**, **Per‑Rep**, **Visuals**, **Export** (placeholders).

- [Agent] **T01.5 — View selector**
  - `st.selectbox("Camera view", ["Lateral", "Frontal"])` → variable `view: str` propagated through pipeline.

_**DoD (Epic 1):**_ App launches; upload works; `cfg` loaded; `view` captured; no pipeline yet, no crashes.

---

## Epic 2 — Pose Extraction (MediaPipe 3D)
**Goal:** return 3D world landmarks per frame, capped FPS.

- [Agent] **T02.1 — Video iterator**
  - File: `ml/pose3d.py` → `iter_frames(video_input, cfg) -> Iterable[Frame]` using OpenCV; timestamp each frame; obey `cfg.fps_cap`.

- [Agent] **T02.2 — Pose Landmarker 3D**
  - Load MediaPipe Tasks — Pose Landmarker 3D; run per frame; collect `pose_world_landmarks` and confidences.

- [Agent] **T02.3 — Extract API**
  - `extract(video_input, cfg) -> PoseTimeSeries` (no file reads inside).

_**DoD (Epic 2):**_ On a short clip, returns non‑empty `PoseTimeSeries` respecting `cfg.fps_cap`.

---

## Epic 3 — Preprocessing
**Goal:** clean and normalize landmark series.

- [Agent] **T03.1 — Interpolation**
  - File: `ml/preprocess.py` → `interpolate(series, cfg) -> series` for low‑confidence gaps.

- [Agent] **T03.2 — Smoothing**
  - `smooth(series, cfg) -> series` with low‑pass using `cfg.smoothing_alpha`.

- [Agent] **T03.3 — Normalization**
  - `normalize(series, cfg) -> series` (scale by shoulder distance; center body).

- [Agent] **T03.4 — Composite clean**
  - `clean(series, cfg) -> series` chaining the three steps above.

_**DoD (Epic 3):**_ Output preserves length; smoke tests pass on synthetic noisy data.

---

## Epic 4 — Rep Segmentation
**Goal:** detect start/end of reps from hip vertical trajectory (view‑agnostic).

- [Agent] **T04.1 — Trajectory extraction**
  - File: `ml/segment.py` → `hip_vertical_signal(series, cfg) -> np.ndarray` (time‑aligned).

- [Agent] **T04.2 — Peaks/Hysteresis**
  - `segment(series, cfg) -> list[RepWindow]` using prominence + hysteresis; enforce `cfg.segmentation.*`.

- [Agent] **T04.3 — Windows**
  - Return `RepWindow[]` with indices/timestamps; expose debug arrays for charts.

_**DoD (Epic 4):**_ Counts within ±1 vs. manual for demo clip; windows non‑overlapping.

---

## Epic 5 — KPI Computation (view‑aware)
**Goal:** compute per‑rep KPIs; some metrics depend on `view`.

- [Agent] **T05.1 — Angles (lateral focus)**
  - File: `ml/kpi.py` → `angles_lateral(series, windows, cfg) -> list[...values...]` (knee, hip, trunk).

- [Agent] **T05.2 — Alignment (frontal focus)**
  - `alignment_frontal(series, windows, cfg) -> list[...values...]` (knee‑to‑ankle alignment; left/right symmetry proxy).

- [Agent] **T05.3 — Tempo & ROM**
  - `tempo_rom(series, windows, cfg) -> list[...values...]` (eccentric/concentric durations; hip/knee ROM).

- [Agent] **T05.4 — Compose metrics**
  - `compute(series, windows, cfg, view: str) -> list[RepMetrics]` (fill only relevant fields per `view`).

_**DoD (Epic 5):**_ Returns expected fields; values change sensibly between views on demo clips.

---

## Epic 6 — Fault Rules (view‑aware)
**Goal:** threshold‑based flags; rules chosen by `view`.

- [Agent] **T06.1 — Rule config**
  - File: `ml/faults.py` → read `cfg.thresholds.*` (depth, valgus, trunk).

- [Agent] **T06.2 — Implement flags**
  - `detect(metrics: RepMetrics, cfg, view: str) -> FaultFlags`

  - Lateral: insufficient depth, excessive trunk lean.

  - Frontal: knee valgus (and optionally asymmetry if time permits).

_**DoD (Epic 6):**_ Changing `view` toggles which faults are evaluated; flags respond to threshold edits in config.

---

## Epic 7 — Feedback Mapping
**Goal:** map flags to short advice strings.

- [Agent] **T07.1 — Mapper**
  - File: `ml/feedback.py` → `to_text(flags: FaultFlags, view: str) -> list[str]` (stable wording).

_**DoD (Epic 7):**_ Returns concise messages (≤ 8 words each).

---

## Epic 8 — Results UI
**Goal:** populate tabs from the pipeline output.

- [Agent] **T08.1 — Overview tab**
  - Show rep count, averages, fault histogram (respect `view`).

- [Agent] **T08.2 — Per‑Rep tab**
  - Table with rep #, angles/alignment, depth/tempo, flags ✔/✖.

_**DoD (Epic 8):**_ End‑to‑end run produces visible tables without exceptions.

---

## Epic 9 — Visuals & Charts
**Goal:** overlay + angle/alignment‑over‑time charts.

- [Agent] **T09.1 — Charts**
  - Matplotlib line charts for relevant signals (no custom colors).

- [Agent] **T09.2 — Overlay (basic)**
  - Player with skeleton landmarks; if heavy, generate short GIF preview.

_**DoD (Epic 9):**_ Visuals render on both lateral and frontal demo clips within reasonable time.

---

## Epic 10 — Export
**Goal:** CSV/JSON download of per‑rep data.

- [Agent] **T10.1 — Serialization**
  - Convert per‑rep results to pandas DataFrame/dict, include `view` in metadata/header.

- [Agent] **T10.2 — Download buttons**
  - `st.download_button` for `results_{timestamp}.csv` and `.json`.

_**DoD (Epic 10):**_ Files download and match on‑screen values; `view` recorded.

---

## Epic 11 — Evaluation (Mini)
**Goal:** quick metrics on a tiny labeled set (both views).

- [Agent] **T11.1 — Script**
  - File: `tests/eval_min.py` → prints rep count exact‑match %, PR/F1 for applicable faults per view.

_**DoD (Epic 11):**_ Produces `reports/eval_summary.json` with per‑view metrics.

---

## Epic 12 — Logging & Privacy
**Goal:** friendly messages + no persistent video by default.

- [Agent] **T12.1 — Logging**
  - INFO timings, WARN on low confidence; logs to `logs/run_{timestamp}.log`.

- [Agent] **T12.2 — Temp handling**
  - Auto‑delete intermediates unless `keep_intermediates: true` in config.

_**DoD (Epic 12):**_ Log file exists; temp deletion works.

---

## Epic 13 — README (Minimal)
**Goal:** quickstart for new users.

- [Agent] **T13.1 — Write README.md**
  - Include env create/activate; `streamlit run`; **supports Frontal & Lateral views (user‑selected)**; CPU‑only; troubleshooting.

_**DoD (Epic 13):**_ A teammate can run the app in 5 minutes.

---

## Epic 14 — Deploy (Optional)
**Goal:** Streamlit Community Cloud.

- [Human] **T14.1 — Connect repo to Streamlit Cloud**
  - Provide account/repo permissions; set secrets if needed.

_**DoD (Epic 14):**_ Public link loads and runs within CPU time constraints.

---

## Dependencies between tasks (edges)

```
T00.1 → T01.* → T02.* → T03.* → T04.* → T05.* → T06.* → T07.* → T08.* → T09.* → T10.*
                                                          ↘───────────── T11.*
T01.* → T13.1
T02.*/T06.*/T08.* → T10.*
All → T12.*
T01.* → T14.1 (optional)
```

---

## Global DoD (MVP)

A new user can run the app, select **Frontal or Lateral** view, upload a squat video, and obtain **rep count + core KPIs + ≥2–3 fault flags** relevant to the selected view, with overlay and charts, and download **CSV/JSON**—within **reasonable CPU latency**—plus a minimal per‑view evaluation summary and README.

---

## Agent‑Editable Appendix — Task Completion Ledger
**Purpose:** Safe, append‑only area for agents to log task completions.

**Format (one item per line):**
`[YYYY-MM-DD] Txx.y — [DONE] — short note`

Examples:
`[2025-10-08] T02.3 — [DONE] — Pose3D extract() returns PoseTimeSeries`
`[2025-10-08] T06.2 — [DONE] — faults.detect(view-aware) wired to cfg thresholds`

<!-- === BEGIN: AGENT-EDITABLE TASK COMPLETION LEDGER === -->
<!-- (append completed task entries below this line) -->
[2025-10-08] T02.1–T02.3 — [DONE] — Pose3D extract() returns PoseTimeSeries; wired to app
[2025-10-08] T02.FIX — [DONE] — PoseLandmarker result unpacking handles list and .landmark
[2025-10-08] T03.1–T03.5 — [DONE] — Preprocess interpolate/smooth/normalize clean wired to UI
[2025-10-09] T04.1–T04.4 — [DONE] — Rep segmentation on pre-normalized series wired to UI
<!-- === END: AGENT-EDITABLE TASK COMPLETION LEDGER === -->
