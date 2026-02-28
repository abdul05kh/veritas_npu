# 🛡️ Veritas-NPU: The Reality Firewall

**An Explainable, Multimodal Deepfake Detection Engine (Edge AI + Cybersecurity)**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)  
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)  
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Vision-orange.svg)](https://developers.google.com/mediapipe)  
[![AMD](https://img.shields.io/badge/Optimized_for-AMD_Ryzen™_AI-ed1c24.svg)](https://www.amd.com/en/products/processors/consumer/ryzen-ai.html)

**Built by Team Void Breakers for the AMD Slingshot Hackathon**  
*Theme 6: AI + Cybersecurity & Privacy*

Developed by: Team Void Breakers (Mohammad Abdul Kalam Hussain & Team)


---

## Table of Contents

- [The Problem](#-the-problem)
- [The Solution](#-the-solution)
- [Core Detection System](#-core-detection-system)
- [Explainable AI Layer](#-explainable-ai-layer-killer-feature)
- [Enterprise UI System](#-enterprise-ui-system)
- [Reporting Engine](#-reporting-engine)
- [Hardware Acceleration](#-hardware-acceleration)
- [Quick Start](#-quick-start)
- [Modes](#-modes)
- [Key Technical Features](#key-technical-features)
- [Installation](#-installation)
- [Execution & UI](#-execution--ui)
- [Project Vision](#project-vision)
- [Built For](#-built-for)
- [Team](#-team)
- [Final Statement](#-final-statement)


## ⚠️ The Problem

Modern deepfakes are no longer visual tricks — they are **biologically convincing synthetic identities**.

They can:

* Bypass biometric authentication
* Fake executive presence in real-time
* Perform voice cloning + lip-sync attacks
* Evade traditional pixel-based detection

👉 Most systems fail because they:

* Rely only on visual artifacts
* Ignore biological signals
* Are cloud-dependent (privacy risk)

---

## 💡 The Solution

**Veritas-NPU = A Local “Reality Firewall”**

Instead of asking *“does this look real?”*
Veritas asks:

> **“Does this behave like a real human?”**

It performs **multimodal forensic interrogation** across:

* Visual signals
* Biological signals
* Audio signals
* Temporal consistency

All in **real-time, on-device**

---

## ⚙️ Core Detection System

### Engine Highlights

- Holographic 40% Opacity Alpha‑Mesh (biometric mesh overlay)
- Inferno ELA Thermal Mapping for explainable variance visualization
- Dual‑Export Reporting (Executive TXT + STIX‑style JSON)
- AV‑Desync / Lip‑Sync Forensics
- Direct FFmpeg subprocess routing for high‑fidelity audio extraction
- SciPy Butterworth bandpass filter used for robust rPPG pulse extraction
- Explicit AMD Vitis AI execution provider hooks (ONNX runtime)


### 🔬 1. Spectral Texture Analysis (FFT)

Detects over-smoothing caused by diffusion models
→ Synthetic faces lack high-frequency skin detail

---

### 🎨 2. Chrominance Variance (C-VAR)

Analyzes blood-flow realism via color distribution
→ Fake skin shows abnormal saturation patterns

---

### 🧬 3. Biometric Mesh Asymmetry

468-point facial geometry analysis
→ AI faces often unnaturally symmetrical

---

### ⏱️ 4. Temporal Jitter Detection

Frame-to-frame instability detection
→ Deepfakes produce micro inconsistencies

---

### ❤️ 5. rPPG Pulse Detection (Biological Liveness)

Extracts heart rate from skin color fluctuations

* Real human → measurable BPM
* Deepfake → flatline / noise

---

### 🔊 6. Audio Forensics + Entropy Analysis

Analyzes speech randomness + structure

* Detects voice cloning
* Flags unnatural spectral patterns

---

### 🎭 7. AV-Desync Detection (Lip Sync)

Cross-validates:

* Mouth motion (MAR)
* Audio entropy

→ Detects **fake talking faces**

---

## 🧠 Explainable AI Layer (Killer Feature)

Veritas doesn’t just detect — it explains.

### 🧾 “WHY FLAGGED” Panel

Shows top causes:

* Low spectral energy
* No pulse signal
* Lip-sync mismatch
* Color anomalies

---

### 📊 Confidence Breakdown

Per-signal contribution visualization:

* FFT
* Asymmetry
* rPPG
* ELA

---

### 🗣️ Live AI Narration

Real-time reasoning:

> “Critical: No pulse signature. Synthetic content suspected.”

---

## 🖥️ Enterprise UI System

* 🎯 Tactical Threat Dashboard
* 📈 Live Telemetry Graph
* 🔥 Inferno ELA Thermal Mapping
* 🧊 Holographic Biometric Mesh Overlay
* 🎯 Multi-face tracking (ID-based)
* 🚨 Threat-based visual highlighting

---

## 📦 Reporting Engine

Automatic dual export:

* 📄 Executive TXT Report
* 🧠 STIX-style JSON Threat Intel

Includes:

* MITRE ATT&CK mapping
* Subject-level analysis
* Aggregated forensic metrics

---

## ⚡ Hardware Acceleration

Supports:

* AMD Ryzen™ AI (Vitis AI EP)
* DirectML (fallback)
* CPU (XNNPACK)

Auto-detects best available provider at runtime.

---

## 🚀 Quick Start Guide

### **Prerequisites**
- Python **3.9+**  
- A functional webcam  
- Windows 11 (optimized for AMD Ryzen™ processors with **Ryzen AI**)  

---

## Project Vision

Veritas‑NPU is a high‑performance media forensics suite designed to restore trust in digital communications. By offloading complex computer vision and signal processing tasks to the AMD Ryzen™ AI NPU, the engine provides real‑time, local verification of media authenticity without the latency or privacy risks of cloud‑based detection.

## Key Technical Features

1. Multimodal Forensic Core

- Visual Domain (Spatial/Geometric): ELA (Error Level Analysis), FFT (Spectral Frequency analysis), and Geometric Asymmetry checks using a 468‑point face landmarker.
- Biological Domain (rPPG): Remote photoplethysmography for pulse detection and liveness verification using a Scipy Butterworth bandpass filter to extract BPM.
- Acoustic Domain (Spectral Entropy): Spectral entropy analysis for voice cloning detection and AV‑Desync (Lip‑Sync) correlation using Mouth Aspect Ratio (MAR).

2. Absolute Zoning UI Architecture

- Volumetric 3D Integrity Field: Holographic gradient overlay adapting to head tilt and Z‑depth.
- Thermal ELA Projection: Inferno colormap visualization of ELA variance for explainability.
- Tactical Telemetry: Real‑time smoothed threat probability graphs.

3. Enterprise Intelligence Engine

- FFmpeg Direct Subprocess Routing: High‑performance audio extraction pipeline.
- Dual‑Format Reporting: Executive TXT summaries and STIX‑style JSON threat intel mapped to MITRE ATT&CK (T1586.002).

Hardware Optimization

Veritas‑NPU targets AMD Ryzen™ AI and leverages the Vitis AI Execution Provider (via ONNX Runtime when available) for ultra‑low latency, efficient local inference and reduced CPU/GPU load.


## 🔧 Installation

### **1. Clone the repository**
```bash
git clone https://github.com/abdul05kh/veritas_npu.git
cd veritas_npu
```

### **2. Create an isolated virtual environment**
```bash
python -m venv venv
```

**Activate (Windows):**
```bash
venv\Scripts\activate
```

**Activate (macOS/Linux):**
```bash
source venv/bin/activate
```

### **3. Install forensic dependencies**
```bash
pip install opencv-python numpy mediapipe scipy sounddevice imageio-ffmpeg onnxruntime
```

---

## ▶️ Execution & UI

### **Launch the Enterprise Command Center GUI**
```bash
python main.py
```

**Note:**  
On first launch, veritas_npu automatically downloads the required `face_landmarker.task` model.

---

## 🖥️ Enterprise Dashboard Features

### **Dynamic Delaunay Triangulation**
A real‑time OpenCV geometric wireframe maps the topography of tracked subjects.

### **Multi‑Target Deep Tracking**
Aggressive **0.15 confidence thresholds** allow tracking of up to **10 faces simultaneously**, even in the background.

### **Glassmorphism HUD**
Live Threat Telemetry is plotted on a sleek, semi‑transparent tactical area graph.

### **Crash‑Proof Automated Reporting**
When the feed is terminated (`q` key or window close):

- Veritas generates a detailed `.txt` forensic breakdown  
- Each tracked ID receives its own report  
- Reports are stored in the **Forensic_Reports/** directory  

---

Built with 💻 and ☕ by **Team Void Breakers**.