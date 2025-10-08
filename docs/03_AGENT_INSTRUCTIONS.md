# 03_AGENT_INSTRUCTIONS — The “Rules” (MVP)

**Scope:** Implement the MVP defined in `00_PROJECT_OVERVIEW.md`, using the paths/contracts in `01_PROJECT_STRUCTURE.md`, and the tasks in `02_TASK_LIST.md`.

**Priority:** The AI pipeline (pose → preprocess → segment → KPIs → faults → feedback) is the focus. Streamlit is a thin UI.

**Hard constraints:** CPU‑only, offline (no external network I/O), Streamlit + MediaPipe Pose **3D**, minimal dependencies, deterministic runs.

**Multi‑view:** The project **supports Frontal & Lateral (sagittal) views**; the user selects the view in the UI; downstream logic must respect it.

**Config passing:** All `ml/` functions that use settings **must accept `cfg`** explicitly; no global file reads inside libraries.

---

## 1) Ground‑truth documents (do not diverge)
- `00_PROJECT_OVERVIEW.md` — what & why (MVP definition)
- `01_PROJECT_STRUCTURE.md` — repository map, I/O contracts, config keys
- `02_TASK_LIST.md` — small, ordered tasks with DoD and multi‑view clarifications

> If code needs a change that affects these contracts, **update the docs first**, then code.

---

## 2) Operational workflow (revised for autonomous documentation)
For **every assigned task**, follow these steps in your response:

1. **Acknowledge the task** — restate the task ID and its goal.
2. **State your plan** — list the files you will create/modify.
3. **Generate complete code** — provide the full code for all files in your plan.
4. **Verify the DoD** — explicitly explain how the task’s DoD is met (include quick evidence/logs when relevant).
5. **Update Task List** — propose the exact patch to `02_TASK_LIST.md` to mark the task as complete by **appending** ` — [DONE 2025-10-08]` at the end of the relevant task line (do **not** rename tasks or change ordering).
6. **Update Project Structure** — if you created new files **that were already planned** in `01_PROJECT_STRUCTURE.md` or `02_TASK_LIST.md`, propose the exact text to append to `01_PROJECT_STRUCTURE.md` with a short description under the correct folder. If a file was **not planned**, **stop** and report why it’s needed; await approval.
7. **Confirm completion** — end with a clear statement that the code and documentation updates are complete.

> Always provide doc updates as **precise text edits** (patches), not free‑form prose.

---

## 3) Stable interfaces (frozen for MVP)
Use **exact names and signatures** below. Add docstrings & type hints.

```python
# ml/pose3d.py
def iter_frames(video_input, cfg) -> "Iterable[Frame]": ...
def extract(video_input, cfg) -> "PoseTimeSeries": ...  # uses MediaPipe Tasks Pose Landmarker 3D

# ml/preprocess.py
def interpolate(series: "PoseTimeSeries", cfg) -> "PoseTimeSeries": ...
def smooth(series: "PoseTimeSeries", cfg) -> "PoseTimeSeries": ...
def normalize(series: "PoseTimeSeries", cfg) -> "PoseTimeSeries": ...
def clean(series: "PoseTimeSeries", cfg) -> "PoseTimeSeries": ...

# ml/segment.py
def hip_vertical_signal(series: "PoseTimeSeries", cfg) -> "np.ndarray": ...
def segment(series: "PoseTimeSeries", cfg) -> "list[RepWindow]": ...

# ml/kpi.py
def angles_lateral(series: "PoseTimeSeries", windows: "list[RepWindow]", cfg) -> "list[dict]": ...
def alignment_frontal(series: "PoseTimeSeries", windows: "list[RepWindow]", cfg) -> "list[dict]": ...
def tempo_rom(series: "PoseTimeSeries", windows: "list[RepWindow]", cfg) -> "list[dict]": ...
def compute(series: "PoseTimeSeries", windows: "list[RepWindow]", cfg, view: str) -> "list[RepMetrics]": ...

# ml/faults.py
def detect(metrics: "RepMetrics", cfg, view: str) -> "FaultFlags": ...

# ml/feedback.py
def to_text(flags: "FaultFlags", view: str) -> "list[str]": ...
```

**Data contracts** (keys) are defined in `01_PROJECT_STRUCTURE.md` §4 and must remain stable.

---

## 4) Documentation Updates (Autonomous Workflow)
As part of your operational workflow, you are required to log your progress in the designated **Agent‑Editable Appendix** sections of the relevant documents.

**Task Completion:** To mark a task as complete, **append** a new line to the **Task Completion Ledger** at the end of `02_TASK_LIST.md`. Use the exact format:  
`[YYYY-MM-DD] Txx.y — [DONE] — short note on completion`

**File Creation:** When you create a **planned** file, **append** a new line to the **Project Structure Updates** appendix at the end of `01_PROJECT_STRUCTURE.md`. Use the exact format:  
`[YYYY-MM-DD] path/to/file — brief purpose`

**Guardrail:** You are **strictly prohibited** from editing any part of the documentation **outside** of the designated append‑only blocks.



## 5) Coding standards
- **Python 3.11**; PEP 8; add **type hints** everywhere.
- Keep functions **small & pure**; no mutable globals; avoid hidden state.
- **No `print`** in libraries. Use `logging` (INFO timings, WARN low confidence). UI can show `st.error`/`st.warning`.
- **Determinism:** set seeds where relevant; respect `cfg.fps_cap` and processing order.
- **Performance:** for a 10–15 s clip, end‑to‑end ≤ **~15 s** on CPU. Vectorize with NumPy; avoid Python loops inside hot paths.
- **Charts:** use **matplotlib** only; **do not set custom colors**; no seaborn.
- **Dependencies:** use only what’s in `environment.yml`. Do not edit that file.

---

## 6) Files, paths, and I/O
- Read/write **only** under the folders defined in `01_PROJECT_STRUCTURE.md`.
- Default behavior: **do not persist user videos**; intermediates stay in memory; temp files auto‑deleted unless `cfg.keep_intermediates` is true.
- Name outputs exactly as specified (e.g., `results_{timestamp}.csv/json` with UTC ISO8601 compact).

---

## 7) Error handling & UX
- Raise **clear exceptions** in `ml/` modules; catch in `app/app.py` and render friendly messages in Streamlit.
- Graceful degradation: if no person detected or pose is low confidence, abort the pipeline with a clear UI message.
- Log remediation hints: “Ensure lateral view”, “Increase lighting”, “Keep subject fully visible”.

---

## 8) Multi‑view rules
- UI passes `view ∈ {"Lateral", "Frontal"}` into the pipeline.
- `ml.kpi.compute(..., view)` computes only the relevant metrics per view.
- `ml.faults.detect(..., view)` evaluates only the applicable rules, e.g.:
  - **Lateral:** insufficient depth, excessive trunk lean.
  - **Frontal:** knee valgus (optionally asymmetry if added).
- Include `view` in exported metadata (CSV/JSON header or fields).

---

## 9) Git workflow & commits
- Branch naming: `feature/Txx-short-name` (e.g., `feature/T05-kpi-angles`).
- **Conventional Commits** required:
  - `feat: ` new user‑visible behavior
  - `fix: ` bug fix
  - `refactor: ` no behavior change
  - `docs: ` documentation only
  - `test: ` adding/updating tests
- Example: `feat: add lateral angles and tempo computation (T05.1,T05.3)`.
- Each task completion PR should **include** corresponding doc patches from Sections 2.5 and 2.6.

---

## 10) Testing & evaluation
- **Unit tests:** add minimal tests for segmentation and fault rules (`tests/test_segment.py`, `tests/test_faults.py`).
- **Evaluation mini‑script:** produce `reports/eval_summary.json` with rep count exact‑match % and PR/F1 **per view** (see `02_TASK_LIST.md` T11.1).
- Prefer **deterministic** tests with synthetic sequences for segmentation validation.

---

## 11) Prohibited actions
- Creating unplanned files/folders or changing top‑level layout without prior doc updates and approval.
- Reading config files from inside `ml/` modules (config must be **passed in**).
- Network calls, cloud uploads, or fetching external models at runtime.
- Large refactors that change function signatures or data keys without doc updates.

---

## 12) Acceptance criteria (per task)
A task is **done** when:
- DoD in `02_TASK_LIST.md` for that task is met.
- Unit or smoke tests pass.
- End‑to‑end sanity works on at least one **lateral** and one **frontal** demo clip.
- **Doc patches** for Task List and (if applicable) Project Structure are included and correct.

---

## 13) When in doubt
- Prefer **simpler**, **more interpretable** logic (rules > complex ML) for the MVP.
- If a requirement is unclear, **update the docs** (PR to `.md`) before coding.
- Keep the UI minimal; focus on reliable, reproducible analysis.
