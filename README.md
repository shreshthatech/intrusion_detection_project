🔍 Multimodal Intrusion Detection & Incident Response System

🛡️ Project Summary: Edge-Efficient Multimodal Security

This repository contains an academic prototype for an Edge-Efficient Multimodal Intrusion Detection & Incident Response (EMIDIR) system.

The core goal is to leverage advanced AI techniques to fuse highly diverse, asynchronous sensor signals—RGB video, thermal video, audio, and RF spectrum—to dramatically reduce false alarms and determine the optimal security patrol dispatch under strict edge computing constraints (low latency, minimal power, and memory). This project focuses on learning, experimentation, and building a complete pipeline structure.

👤 Author Information

Detail

Value

Author

Shreshtha Gupta

Roll No

25SCS1003004880

Program

B.Tech CSE (AI/ML)

University

IILM University – Greater Noida

📌 Project Overview & Status

✔ Honest Current Status

The project is currently in the initial development (scaffolding and prototyping) stage.

Folder Structure: Fully organized and professional modular layout is complete.

Code Templates: Necessary templates for all modules are complete.

Prototype Scripts: Several foundational scripts for data preparation and vision/tracking demos are partially working.

Models & Fusion: Models are not fully trained or optimized; multimodal fusion is planned.

Datasets: Planned but not yet downloaded/integrated.

🎯 Project Aim

This work serves as a student-level research project focused on:

Mastering Multimodal AI (Vision + Audio + RF) concepts.

Implementing Edge AI Optimization (Quantization, Pruning).

Designing a Reinforcement Learning environment for real-world decision support.

Practicing professional software architecture and documentation standards.

🧱 Features (Implemented / Partial)

✔ Organized Modular Architecture

All modules required for a full Intrusion Detection System (IDS) are scaffolded: Vision, Tracking, Data Pipeline, Fusion, Acoustic/RF Analytics, Decision Support, and Edge Optimization.

✔ Prototype Functional Components

Some scripts currently work, including:

Motion mask demos

Hard negatives generator

Sliding window builder

Synthetic intrusion samples

RGB/Thermal detection preprocessing

Tracking demos (SORT/Kalman)

Data profiling / alignment utilities

✔ Processed Output Folders

The data/processed/ directory includes folders for storing intermediate results:

motion_masks/

hard_negatives/

rgb_detections/

thermal_detections/

synthetic_intrusion/

aligned/

fusion_detections/

windows/

edge_metrics/

📂 Project Structure (Final)

intrusion_detection_project/
│
├── src/
│   ├── vision/                # RGB/Thermal processing & detectors (B)
│   ├── tracking/              # SORT, Kalman, trajectory tools (D)
│   ├── fusion/                # Fusion templates (C)
│   ├── acoustic/              # Audio feature extraction (E)
│   ├── rf/                    # RF spectrum analytics (E)
│   ├── decision_support/      # RL dispatching, risk scoring (F)
│   ├── edge/                  # Quantization, pruning, distillation (G)
│   ├── data_pipeline/         # Sync, sliding windows, labeling (A)
│   └── utils/                 # Scalers, helpers
│
├── models/
│   ├── vision/                # Tiny CNN, motion models, Mobilenet SSD
│   ├── fusion/
│   └── scalers/
│
├── data/
│   ├── raw/                   # Raw datasets (local only)
│   ├── processed/             # Outputs (motion masks, windows, detections)
│   └── config/                # YAML configs
│
├── logs/                      # Runtime logs
├── dashboard/                 # (Future)
└── README.md


▶️ Running the Current Code

Since this is a developing project, only certain scripts are runnable right now.

1️⃣ Vision Preprocessing & Masking

# Creates binary masks from movement
python src/vision/motion_mask_cnn.py

# Or run background subtraction:
python src/vision/background_subtraction.py

# Filters wildlife, trees, empty scenes (Hard Negative Mining)
python src/vision/hard_negative_mining.py

# Normalizes + enhances thermal imagery
python src/vision/thermal_preprocess.py


2️⃣ Data & Detection Pipeline

# Generates demo intrusion videos in processed/synthetic_intrusion/
python src/data_pipeline/synthetic_generator.py

# Stores fixed-size windows into processed/windows/
python src/data_pipeline/sliding_window_builder.py

# Saves final object detections into processed/rgb_detections/ and processed/thermal_detections/
python src/vision/postprocess_detections.py


3️⃣ Tracking & Fusion Demos

# Runs tracking on sample detections (SORT + Kalman)
python src/tracking/sort_tracker.py

# Detects loitering behavior from track history
python src/tracking/loitering_detector.py

# Computes Mahalanobis OOD (Out-of-Distribution) scores
python src/fusion/ood_mahalanobis.py

# Calibrates per-sensor detection thresholds
python src/fusion/threshold_calibration.py


🍱 Datasets (Planned for Future Training)

These publicly available datasets are planned for use to train the individual modality models, providing a strong foundation for the multimodal fusion component.

Modality

Dataset

Purpose

Vision

CrowdHuman, MOT20

General human detection and multi-object tracking.

Vision/Anomaly

UCF-Crime, ExDark

Anomaly behavior and low-light/night scene robustness.

Audio

ESC-50, UrbanSound8K, VOICe

Classification of environmental sounds, sirens, engines, and potential panic/violence.

RF Spectrum

RadioML 2018.01A

Training model for RF modulation and anomaly detection (e.g., jammers).

⚠️ These datasets are not stored in this repository. They must be downloaded and placed inside data/raw/ on the local machine for training.

🗺️ 70-Task Roadmap (Development Blueprint)

This project is rigorously guided by a 70-task development roadmap, ensuring comprehensive coverage of all project goals, from data pre-processing to final deployment optimization.

Section

Focus

Examples of Tasks (Planned)

A) Data Pipeline (1–10)

Synchronization, Augmentation, Labeling

Timestamp alignment with drift correction, Synthetic data generation, Sliding-window dataset builder.

B) Vision (11–20)

RGB/Thermal Perception

Tiny CNN for motion masks, Lightweight object detectors (MobileNet-SSD), Privacy filter (blurring).

C) Fusion (21–30)

Multimodal Integration & Scoring

Time-sync buffer, Learnable feature fusion (Attention), Bayesian updating of threat score, Uncertainty quantification.

D) Tracking & Re-ID (31–40)

Sequence Modeling & Identity

SORT/ByteTrack-style tracker, Kalman filter, LSTM/GRU for intent prediction, Micro-movement "loitering" detector.

E) Audio & RF Analytics (41–50)

Non-Visual Sensing

MFCC feature extractor, Tiny 1D-CNN classifier, RF waterfall preprocessor, Change-point detection in spectrum.

F) Decision Support (51–60)

RL & Response Optimization

Risk scoring function, Incident priority queue, Q-learning for patrol dispatch, Counterfactual evaluation.

G) Edge Optimization (61–70)

Compression & MLOps

Static quantization (INT8), Knowledge distillation (Teacher→Student), Drift monitor (KS test/PSI), Edge health metrics.

🔮 Future Scope (Planned Enhancements)

The following areas are targeted for development beyond the initial academic scope:

Adaptive Compression: Implement dynamic model pruning/quantization based on real-time hardware thermals and CPU load (G.67) to ensure graceful degradation.

Sensor Reliability: Incorporate a dynamic, learned weighting scheme for sensor reliability based on real-time environmental data (e.g., heavy rain or high ambient noise) to improve fusion accuracy.

Full MLOps Pipeline: Integrate a robust Drift Monitor (KS test/PSI) with auto-recalibration hooks (G.69) to maintain model accuracy as attack patterns or environment change over time.

⭐ Support the Project

If you’re interested in AI, Computer Vision, or Multimodal Systems, please ⭐ star the repository or follow for updates as this academic research project progresses!