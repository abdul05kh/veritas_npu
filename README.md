# 🛡️ Veritas‑NPU: The Reality Firewall  
**A Real‑Time, Hardware‑Accelerated Deepfake & Synthetic Media Detection Engine**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)  
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)  
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Vision-orange.svg)](https://developers.google.com/mediapipe)  
[![AMD](https://img.shields.io/badge/Optimized_for-AMD_Ryzen™_AI-ed1c24.svg)](https://www.amd.com/en/products/processors/consumer/ryzen-ai.html)

**Built by Team Void Breakers for the AMD Slingshot Hackathon**  
*Theme 6: AI + Cybersecurity & Privacy*

---

## 📖 Table of Contents
1. [The Problem](#-the-problem)  
2. [The Solution](#-the-solution)  
3. [The Tripartite Architecture](#-the-tripartite-architecture)  
4. [Why AMD Ryzen™ AI?](#-why-amd-ryzen-ai)  
5. [Quick Start Guide](#-quick-start-guide)  
6. [Execution](#-execution)  
7. [UI Controls](#-ui-controls)

---

## 🚨 The Problem
In the era of generative AI, *“Seeing is Believing”* has become a critical security vulnerability.  
Threat actors now deploy real‑time deepfakes, face‑swaps, and synthetic video to:

- Bypass biometric authentication  
- Impersonate executives during high‑stakes video calls  
- Execute social‑engineering attacks on Zoom, Teams, Meet, etc.

Most existing detection systems are:

❌ Cloud‑based (privacy‑invasive)  
❌ High‑latency  
❌ Not suitable for real‑time defense  

---

## 💡 The Solution: Veritas‑NPU
Veritas‑NPU acts as a **local, OS‑level Reality Firewall**.

It performs real‑time digital forensics on inbound video feeds using:

- Compression anomaly detection  
- Frequency‑domain biometric texture analysis  
- Temporal jitter tracking  

All computation happens **locally**, ensuring:

✔ Zero PII leaves the device  
✔ Zero‑trust compliance  
✔ Real‑time threat detection  

---

## ⚙️ The Tripartite Architecture

### **1. Spatial Forensics — Error Level Analysis (ELA)**
Generative models leave unnaturally smooth compression signatures.  
Veritas intentionally re‑compresses each frame and computes the variance of the absolute difference to expose synthetic noise patterns.

### **2. Frequency Forensics — Spectral Analysis (FFT)**
Deepfakes erase high‑frequency micro‑textures like pores and stubble.  
A 2D FFT isolates high‑frequency energy; real skin shows chaotic energy, while deepfakes appear as smooth voids.

### **3. Temporal Forensics — Micro‑Jitter Detection (MSE)**
Deepfake generators struggle with frame‑to‑frame consistency.  
We compute MSE between consecutive frames to detect microscopic jitter and pixel‑shift artifacts.

---

## ⚡ Why AMD Ryzen™ AI?
Running ELA, FFTs, and biometric isolation at 30 FPS on a CPU is computationally expensive.

Ryzen™ AI provides:

- **Dedicated NPU acceleration** for matrix operations  
- **Zero‑latency inference** for live video  
- **Edge‑native privacy** (no cloud dependency)  
- **Power efficiency** for long‑duration calls  

Veritas‑NPU is engineered to offload heavy forensics to the NPU, freeing CPU/GPU resources for user workloads.

---

## 🚀 Quick Start Guide

### **Prerequisites**
- Python **3.9+**  
- A functional webcam  
- (Optional but recommended) AMD Ryzen™ processor with **Ryzen AI** enabled  

---

### **Installation**

#### **1. Clone the repository**
```bash
git clone https://github.com/VoidBreakers/Veritas-NPU.git
cd Veritas-NPU
```

#### **2. Create an isolated virtual environment**
```bash
python -m venv venv
```

**On Windows:**
```bash
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

#### **3. Install forensic dependencies**
```bash
pip install opencv-python numpy mediapipe
```

---

## ▶️ Execution

Launch the Enterprise Command Center:

```bash
python main.py
```

**Note:**  
On first launch, Veritas‑NPU will automatically download the required `.tflite` biometric tracking model.

---

## 🖥️ UI Controls

- The dashboard automatically locks onto the primary biometric target  
- The **Tactical Area Graph** displays smoothed threat telemetry  
- Press **q** to securely terminate the engine and close the port  

---

Built with 💻 and ☕ by **Team Void Breakers**.