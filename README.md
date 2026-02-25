# 🛡️ Veritas‑NPU: The Reality Firewall
**A Real‑Time, Hardware‑Accelerated Deepfake & Synthetic Media Detection Engine**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)  
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)  
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Vision-orange.svg)](https://developers.google.com/mediapipe)  
[![AMD](https://img.shields.io/badge/Optimized_for-AMD_Ryzen™_AI-ed1c24.svg)](https://www.amd.com/en/products/processors/consumer/ryzen-ai.html)

**Built by Team Void Breakers for the AMD Slingshot Hackathon**  
*Theme 6: AI + Cybersecurity & Privacy*

Developed by: Team Void Breakers (Mohammad Abdul Kalam Hussain & Team)

Hardware Target: AMD Ryzen™ AI NPU (via Vitis AI Execution Provider)

## Table of Contents

- [Project Vision](#project-vision)
- [Key Technical Features](#key-technical-features)
- [Quick Start Guide](#quick-start-guide)
- [Installation](#installation)
- [Execution & UI](#execution--ui)
- [Enterprise Dashboard Features](#enterprise-dashboard-features)
- [Crash‑Proof Automated Reporting](#crash-proof-automated-reporting)

---

## 🚨 The Vulnerability
In the era of Latent Diffusion and high‑fidelity generative AI, standard digital forensics have become obsolete.  
Threat actors now deploy real‑time deepfakes and face‑swaps that are mathematically trained to bypass basic texture analysis.  
They are used to:

- Bypass biometric authentication  
- Impersonate executives  
- Execute sophisticated social‑engineering attacks  

Existing detection systems fail because they are:  
❌ Cloud‑dependent (privacy & latency risks)  
❌ Reliant on outdated visual‑spectrum algorithms  
❌ Too computationally heavy for real‑time edge deployment  

---

## 💡 The Solution: Veritas‑NPU
Veritas‑NPU acts as a **local, OS‑level Reality Firewall**.

Instead of analyzing pixels, Veritas interrogates the **biological, spectral, and cryptographic signatures** of a video feed.

All computation happens **locally at the edge**, ensuring:

- Zero PII leaves the device  
- Zero‑Trust compliance  
- Real‑time threat detection  

---

## ⚙️ The Empirical "Zero‑Trust" Architecture
Veritas‑NPU abandons standard texture mapping and implements **four DOD‑level physical sensor forensic tests**:

### **1. Spectral Texture Deficit (FFT)**
AI diffusion models over‑smooth high‑frequency microscopic pores.  
A 2D FFT isolates this energy:

- Real skin → chaotic high‑frequency spectrum  
- Synthetic skin → smooth, low‑energy void  

### **2. Hyper‑Saturation Chrominance Analysis (C‑VAR)**
AI‑generated skin lacks natural sub‑surface blood flow.  
We convert frames to **YCbCr** and isolate the **Cr (Red‑Difference)** channel to expose biological inconsistencies.

### **3. Biometric Mesh Asymmetry (BMA)**
Generative models often produce unnaturally perfect symmetry.  
Using a **468‑point face landmarker**, Veritas measures:

- Jawline asymmetry  
- Focal plane distortion  
- Geometric irregularities  

### **4. Micro‑Temporal Jitter (MSE)**
Deepfake generators struggle with sub‑pixel temporal consistency.  
We compute **Mean Squared Error** across consecutive frames to detect:

- Rendering tears  
- Pixel‑shift jitter  
- Artificial edge‑bleeding  

---

## ⚡ Why AMD Ryzen™ AI?
Running FFTs, chrominance mapping, and multi‑target biometric isolation at 30 FPS is computationally expensive.

Ryzen™ AI provides:

- **Dedicated NPU acceleration** for matrix operations  
- **Zero‑latency inference** for live video interrogation  
- **Power‑efficient always‑on monitoring**  

Veritas‑NPU offloads the entire forensic pipeline to the NPU, freeing CPU/GPU resources for the user’s workflow.

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
git clone https://github.com/VoidBreakers/Veritas-NPU.git
cd Veritas-NPU
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
pip install opencv-python numpy mediapipe
```

---

## ▶️ Execution & UI

### **Launch the Enterprise Command Center GUI**
```bash
python main.py
```

**Note:**  
On first launch, Veritas‑NPU automatically downloads the required `face_landmarker.task` model.

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