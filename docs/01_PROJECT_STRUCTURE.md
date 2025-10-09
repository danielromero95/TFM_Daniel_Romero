# 01_PROJECT_STRUCTURE — The “Where” (MVP)

**Purpose:** give AI agents a precise map of where things live in the repo and where to read/write.

---

## Repository layout (canonical paths)

```
/                      # repo root
├─ app/               # Streamlit MVP (single page)
│  └─ app.py          # entrypoint: streamlit run app/app.py
├─ ml/                # analysis pipeline (pure Python)
│  ├─ pose3d.py       # MediaPipe Pose Landmarker 3D wrapper
│  ├─ preprocess.py   # smoothing, interpolation, normalization
│  ├─ segment.py      # rep segmentation & counting
│  ├─ kpi.py          # KPI calculations (angles, ROM, tempo)
│  ├─ faults.py       # rule-based fault detection
│  └─ feedback.py     # map fault flags → short tips
├─ config/
│  └─ default.yml     # runtime parameters (see below)
├─ assets/
│  ├─ models/          # MediaPipe models (.task)
│  │  ├─ pose_landmarker_lite.task   # CPU-friendly (recommended)
│  │  └─ pose_landmarker_full.task   # slower but more accurate
│  ├─ demo_clips/     # short consented example videos (optional)
│  └─ icons/          # UI icons if needed
├─ reports/           # exported CSV/JSON results
├─ logs/              # runtime logs (rotated)
├─ tests/             # minimal unit tests (segmentation, rules)
├─ environment.yml    # mamba/conda environment (CPU)
├─ 00_PROJECT_OVERVIEW.md
├─ 01_PROJECT_STRUCTURE.md
├─ 02_TASK_LIST.md
└─ 03_AGENT_INSTRUCTIONS.md
```

> Treat these paths as contracts. Don’t add new top-level folders without updating this file.

---

## I/O contracts

- **Input video:** uploaded via Streamlit; processed in memory or temp file. No persistent storage by default.  
- **Intermediate cache (optional):**
  - Per-frame pose landmarks kept in memory; if persisted, use `./logs/pose_cache_{run_id}.npz`.
- **Outputs:**
  - **Per-rep table** → `./reports/results_{timestamp}.csv` and `./reports/results_{timestamp}.json`.
  - **Summary JSON** (optional) → `./reports/summary_{timestamp}.json`.
- **Visual overlays** (optional) → `./reports/overlay_{timestamp}.mp4` or `.gif` if enabled.

**Naming rule:** `{timestamp}` = UTC ISO8601 compact (`YYYYMMDDTHHMMSSZ`).

---

## Runtime config (read from `config/default.yml`)

Minimal keys used across modules (agents must not rename):

```yaml
fps_cap: 25
smoothing_alpha: 0.30
thresholds:
  min_squat_depth_angle: 95.0   # deg
  valgus_dev_max: 0.12          # normalized deviation
  trunk_angle_max: 45.0         # deg
segmentation:
  hip_min_drop_norm: 0.08
  peak_prominence: 0.03
  min_rep_duration_s: 0.8
```

Load once in `app/app.py` and pass down to modules.

Additional performance tuning knobs (all optional, defaults shown):

```yaml
performance:
  sample_rate: 1                  # process every frame (1 = no skipping)
  preprocess_size:
    longest_edge_px: 0            # 0 disables resize before pose inference
  max_workers: 0                  # 0 ⇒ cpu_count() - 1
  parallel: false                 # toggle parallel extraction in the UI
```

---

## Module handshakes (data shapes)

Data structures are simple dicts (or dataclasses) with fixed keys.

```yaml
PosePoint: {x: float, y: float, z: float, visibility: float, presence: float}
PoseFrame: {frame_idx: int, t: float, landmarks: {name: PosePoint}, conf: float}
PoseTimeSeries:
  orig_fps: float
  fps: float
  sample_rate: int
  preprocess_size: {longest_edge_px: int}
  frames: PoseFrame[]

RepWindow: {rep_id: int, start_t: float, end_t: float, start_idx: int, end_idx: int}

RepMetrics:
  rep_id: int
  min_knee_angle_deg: float
  min_hip_angle_deg: float
  depth_ok: bool
  trunk_max_angle_deg: float
  tempo_down_s: float
  tempo_up_s: float
  rom_hip_deg: float
  rom_knee_deg: float

FaultFlags:
  insufficient_depth: bool
  knee_valgus: bool
  excessive_trunk_lean: bool

AnalysisSummary:
  rep_count: int
  avg_min_knee_angle_deg: float
  avg_min_hip_angle_deg: float
  fault_histogram: {insufficient_depth: int, knee_valgus: int, excessive_trunk_lean: int}
```

---

## Execution commands

- Create env: `mamba env create -f environment.yml`
- Activate: `mamba activate squat-mvp`
- Run app: `streamlit run app/app.py`

---

## Logging & files

- Default log path: `./logs/run_{timestamp}.log` (INFO level; DEBUG optional via env var `MVP_DEBUG=1`).
- Do not store user videos persistently unless explicitly enabled in config.

---

## Assumptions

- Single exercise (**back squat**); supports Frontal & Lateral views (user-selected); CPU-only.
- Short clips (~10–15 s). If FPS > `fps_cap`, downsample.
- Time base in seconds; angles in degrees; timestamps UTC.

---

## Agent‑Editable Appendix — Project Structure Updates
**Purpose:** Safe, append‑only area for agents to document **planned** files/folders created during tasks.

> Do **not** list unplanned artifacts here. If a new file/folder was not defined in `01_PROJECT_STRUCTURE.md` or `02_TASK_LIST.md`, stop and request approval.

**Format (one item per line):**
`[YYYY-MM-DD] path/to/file_or_folder — brief purpose (≤ 12 words)`

<!-- === BEGIN: AGENT-EDITABLE PROJECT STRUCTURE APPENDIX === -->
<!-- (append new planned file/folder entries below this line) -->
[2025-10-08] assets/models/ — model assets directory (.task files)
[2025-10-08] assets/models/pose_landmarker_lite.task — MediaPipe Pose 3D (Lite)
[2025-10-08] assets/models/pose_landmarker_full.task — MediaPipe Pose 3D (Full)
<!-- === END: AGENT-EDITABLE PROJECT STRUCTURE APPENDIX === -->
