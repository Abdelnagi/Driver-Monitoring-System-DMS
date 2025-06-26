# 🚗 Driver Fatigue & Distraction Monitoring System

This module is responsible for detecting fatigue (microsleep, yawning, blinking) and distractions (abnormal gaze/head movements) using computer vision (yolo and mediapipe). It streams live feedback through a Flask web app,and provides alerts for critical driver states.

---

## 📦 Project Structure

| File / Folder                 | Purpose |
|-------------------------------|---------|
| `Run_fatigueAndDistraction.py`| Main runner script for fatigue and distraction detection |
| `gazeHeadDetection.py`        | Gaze direction and head movement tracking using Mediapipe |
| `yawnBlinkDetection.py`       | Fatigue detection using YOLO for yawn and blink detection |
| `two_cameras_updated.py`      | Initializes dual USB cameras or video files, and serves them via Flask 
| `thresholds.py`               | Configuration for blink, yawn, and microsleep thresholds |
| `driver_assistant.json`       | Stores parameters needed to integrate with NOVA |
| `status_driver_fatigue.json`  | Stores the real-time outputs from the fatigue module |

---

## 🔧 Installation And Running

### 1. Clone the Repository

git clone https://github.com/farahahmed09/DMS_EECE25_CU.git
cd https://github.com/farahahmed09/DMS_EECE25_CU.git


### 2. Install Dependencies
pip install -r requirements.txt
If a package fails during installation, install it manually using pip install package-name.

### 3. ▶️ Running the System
First -> Run two_cameras.py
Second -> Run Run_fatigueAndDistraction.py

