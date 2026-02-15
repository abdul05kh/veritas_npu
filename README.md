# 🛡️ Veritas-NPU: The Reality Firewall
**A Real-Time, Hardware-Accelerated Deepfake & Synthetic Media Detection Engine.**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Vision-orange.svg)](https://developers.google.com/mediapipe)
[![AMD](https://img.shields.io/badge/Optimized_for-AMD_Ryzen™_AI-ed1c24.svg)](https://www.amd.com/en/products/processors/consumer/ryzen-ai.html)

**Built by Team Void Breakers for the AMD Slingshot Hackathon.**
*Targeting Theme 6: AI + Cybersecurity & Privacy.*

---

## 📖 Table of Contents
1. [The Problem](#-the-problem)
2. [The Solution](#-the-solution)
3. [The Tripartite Architecture](#-the-tripartite-architecture)
4. [Why AMD Ryzen™ AI?](#-why-amd-ryzen-ai)
5. [Quick Start Guide](#-quick-start-guide)

---

## 🚨 The Problem
In the era of generative AI, "Seeing is Believing" is a critical security vulnerability. Threat actors are utilizing real-time deepfakes, face-swaps, and synthetic video to bypass biometric locks and execute high-stakes social engineering scams over video conferencing platforms (Zoom, Teams). 

Current detection methods are cloud-based (violating user privacy by streaming video to external servers) and suffer from severe latency.

## 💡 The Solution: Veritas-NPU
Veritas-NPU operates as a localized, OS-level "Reality Firewall." It leverages advanced digital forensics and biometric tracking to interrogate inbound video feeds in real-time. By calculating mathematical anomalies in video compression, pixel frequency, and temporal rendering, Veritas catches synthetic media before the user can be compromised. 

Crucially, it is designed to run entirely on the edge, fulfilling the mandate for **privacy-preserving analytics with no raw PII exposure.**

---

## ⚙️ The Tripartite Architecture

Veritas-NPU abandons outdated "rule-based" behavioral tracking (like counting blinks) in favor of a mathematically rigorous, multi-modal ensemble engine.

<details>
<summary><b>1. Spatial Forensics: Error Level Analysis (ELA)</b></summary>
<br>
We isolate the biometric target and execute a real-time Error Level Analysis pipeline. Generative AI models leave microscopic, unnaturally smooth compression signatures. By intentionally degrading the frame and calculating the variance of the absolute mathematical difference, we expose the hidden "noise" of synthetic generation.
</details>

<details>
<summary><b>2. Frequency Forensics: Spectral Analysis (FFT)</b></summary>
<br>
Deepfakes act as digital airbrushes, destroying high-frequency micro-textures (pores, stubble). We run a 2D Discrete Fast Fourier Transform on the biometric crop, converting the spatial pixels into the frequency domain. We then mask the low frequencies and measure the high-frequency energy. Real human skin maintains a chaotic, high-energy frequency; deepfakes present as a smooth, low-energy void.
</details>

<details>
<summary><b>3. Temporal Forensics: Micro-Jitter Detection (MSE)</b></summary>
<br>
While a single synthetic frame might pass spatial inspection, generative models struggle with temporal consistency. We calculate the Mean Squared Error (MSE) between consecutive frames to detect the microscopic rendering stutters and pixel-shifting (jitter) inherent in live deepfake generation.
</details>

---

## ⚡ Why AMD Ryzen™ AI?
Executing ELA, 2D FFTs, and Biometric Isolation at 30 frames per second on a standard CPU causes severe thermal throttling and destroys battery life. Sending this data to the cloud violates zero-trust privacy architectures.

Veritas-NPU is designed specifically to offload these heavy matrix multiplications to the **AMD Ryzen™ AI NPU**. This provides:
* **Zero-Latency Inference:** Crucial for live video conferencing.
* **Edge-Native Privacy:** Video feeds never leave the user's local machine.
* **Power Efficiency:** Freeing up the CPU/GPU for the user's actual workloads.

---

## 🚀 Quick Start Guide

### Prerequisites
* Python 3.9 or higher.
* A functional webcam.
* (Optional but recommended) AMD Ryzen™ processor with Ryzen AI enabled.

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/VoidBreakers/Veritas-NPU.git](https://github.com/VoidBreakers/Veritas-NPU.git)
   cd Veritas-NPU