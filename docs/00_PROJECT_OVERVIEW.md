# PROJECT_OVERVIEW: MVP of an AI-Powered Exercise Technique Analyzer

## 1. Project Vision (TFM Focus)
To develop and validate a computer vision system that analyzes the technique of strength-training exercises and provides corrective feedback. The final deliverable is a Minimum Viable Product (MVP) implemented as a Streamlit web application.

The primary focus is the research, implementation, and validation of the AI pipeline. The UI is a thin layer to demonstrate functionality, not a product in itself.

---

## 2. Main Goal & Core Analysis Pipeline
To design a pipeline that processes a video of a user performing a **back squat** from either a **lateral (side) or frontal (front)** perspective to extract quantitative metrics and identify technical faults.

**Key Deliverables:**
* **Repetition Count**: Automatically count successful reps.
* **Key Performance Indicators (KPIs)**: Per rep, calculate relevant metrics based on the camera view (e.g., squat depth and torso angle from lateral view; knee alignment from frontal view).
* **Fault Flags**: Identify common errors using predefined biomechanical thresholds, with specific faults being view-dependent.
    * From **Lateral View**: Insufficient Depth, Excessive Forward Lean.
    * From **Frontal View**: Knee Valgus (knees caving in).

**Core Pipeline Stages:**
1.  **Pose Estimation**: Use **MediaPipe Tasks — Pose Landmarker 3D** to extract 3D world landmarks (`pose_world_landmarks`) for key joints in each frame.
2.  **Data Processing**: Clean and smooth landmark time-series data and normalize it to reduce camera-distance effects.
3.  **Rep Segmentation**: Detect repetition boundaries from the vertical trajectory of the hips.
4.  **Feature Calculation**: Compute view-specific KPIs (joint angles, distances, velocities) per segmented rep.
5.  **Rule-Based Fault Detection**: Compare KPIs to configurable thresholds to raise fault flags. The set of rules applied will depend on the user-selected camera view.
6.  **Feedback Generation**: Map triggered faults to clear, actionable text for the user.

**Config Placeholders for Agents**: `FPS_CAP`, `SMOOTHING_ALPHA`, `MIN_SQUAT_DEPTH_ANGLE`, `VALGUS_DEV_MAX`, `TRUNK_ANGLE_MAX`.

---

## 3. MVP Structure & Functionality
A single-page Streamlit dashboard.

**User Flow & Features:**
1.  **Video Upload**: A user uploads a short `.mp4` or `.mov` video clip.
2.  **View Selection**: The user **must specify the camera angle** of the video using a radio button: "Lateral View" or "Frontal View".
3.  **Analysis**: Upon clicking "Analyze," a spinner (`st.spinner`) indicates processing.
4.  **Results Dashboard (tabbed)**:
    * **Overview**: Total rep count, a summary of detected faults.
    * **Detailed Analysis**: A per-rep table showing relevant KPIs and fault flags (✔/✖).
    * **Visuals**: Processed video with skeleton overlay and interactive charts for relevant angles.
5.  **Export Results**: A download button (`st.download_button`) for a CSV/JSON report with per-rep metrics.

---

## 4. Technology Stack
* **Environment Manager**: Mamba / Conda (`environment.yml`)
* **Language**: Python
* **UI**: Streamlit
* **Computer Vision**: MediaPipe (Pose Landmarker 3D), OpenCV
* **Data Manipulation**: NumPy, Pandas

---

## 5. Definition of Done & Assumptions
**The MVP is complete when a user can:**
1.  Launch the Streamlit app.
2.  Upload a squat video and select either "Lateral" or "Frontal" view.
3.  Receive an accurate rep count and view-appropriate fault flags.
4.  View the results, including the skeleton overlay and charts.
5.  Download a structured CSV/JSON report.

**Assumptions for Agents:**
* The project MVP focuses on a single exercise: the **back squat**.
* The system must handle two distinct camera views, which the user will specify. **Fault-detection logic will be view-dependent.**
* Execution will be on the CPU.
* All fault detection thresholds are managed through a simple configuration mechanism.