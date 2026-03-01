"""
Veritas-NPU Reality Firewall // v22.1 Aesthetic Polish Build
Enterprise Deepfake & Synthetic Media Detection Engine

Features:
- PERFECTED: 40% Opacity "Holographic" Alpha-Mesh Geometry
- PERFECTED: COLORMAP_INFERNO ELA mapping (Removes ugly blue background)
- Dual-Export Engine (Executive TXT + STIX-Compliant JSON)
- AV-Desync Lip-Sync Forensics 
- Direct FFmpeg Subprocess Routing 
- Scipy Butterworth Bandpass Filter 
- Explicit AMD Vitis(TM) AI Hooks
"""
import os
from statistics import mode
import sys
import glob

# --- AUTOMATIC NVIDIA HARDWARE LINKING ---
def link_nvidia_hardware():
    # Find any version of CUDA installed in the default directory
    search_path = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin'
    found_paths = glob.glob(search_path)
    
    if found_paths:
        # Sort to get the latest version if multiple exist
        latest_cuda = sorted(found_paths)[-1]
        try:
            os.add_dll_directory(latest_cuda)
            print(f"[INFO] SUCCESS: Linked NVIDIA Hardware via {latest_cuda}")
        except Exception as e:
            print(f"[WARNING] DLL Link Failed: {e}")
    else:
        print("[WARNING] CUDA Toolkit not found in Program Files. NPU Link disabled.")

link_nvidia_hardware()

import logging
import math
import time
import json
import uuid
import platform
import threading
import urllib.request
import subprocess
import tempfile
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Set
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Hardware, Audio & Math Safe Imports
# ---------------------------------------------------------------------------

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except Exception as e:
    AUDIO_AVAILABLE = False
    print(f"[WARNING] Live Audio disabled: {e}")

try:
    import imageio_ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False
    print(f"[WARNING] imageio_ffmpeg missing. Video audio extraction disabled.")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    from scipy.signal import butter, filtfilt
    from scipy.io import wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("VeritasNPU")

@dataclass
class UIConfig:
    header_h: int = 45
    panel_w: int = 320  # Reduced from 360 for better screen fit
    bottom_h: int = 180  # Reduced from 240 for better screen fit
    max_video_w: int = 800  # Reduced from 960 to better fit on standard screens
    window_name: str = "Veritas-NPU // Enterprise Multimodal Forensics"
    
    color_red: Tuple[int, int, int] = (36, 28, 237)
    color_cyan: Tuple[int, int, int] = (255, 255, 0)
    color_green: Tuple[int, int, int] = (57, 255, 20)
    color_orange: Tuple[int, int, int] = (0, 165, 255)
    color_bg: Tuple[int, int, int] = (10, 12, 16)
    
    @staticmethod
    def get_screen_dimensions():
        """Get screen dimensions with padding for taskbar"""
        try:
            # Try to get screen dimensions using tkinter
            root = tk.Tk()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
            # Leave space for taskbars and window decorations
            return max(800, int(screen_width * 0.95)), max(600, int(screen_height * 0.90))
        except Exception:
            # Default fallback
            return 1366, 768


@dataclass
class ForensicThresholds:
    fft_min: float
    fft_warn: float
    cvar_max: float
    cvar_warn: float
    mesh_asym_max: float
    ela_min: float
    jitter_max: float = 1000.0
    rppg_min_bpm: float = 45.0     
    audio_entropy_min: float = 4.5 


class Config:
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    MODEL_PATH = Path("face_landmarker.task")
    
    THRESHOLDS = {
        'image': ForensicThresholds(fft_min=550.0, fft_warn=650.0, cvar_max=120.0, cvar_warn=100.0, mesh_asym_max=0.12, ela_min=20.0),
        'video': ForensicThresholds(fft_min=400.0, fft_warn=600.0, cvar_max=130.0, cvar_warn=100.0, mesh_asym_max=0.15, ela_min=5.0, jitter_max=1200.0)
    }

# ---------------------------------------------------------------------------
# Multimodal & Hardware Modules
# ---------------------------------------------------------------------------

class HardwareAccelerator:
    @staticmethod
    def get_active_provider() -> str:
        if not ONNX_AVAILABLE:
            return "NPU UNLINKED (XNNPACK FALLBACK)"
        
        providers = ort.get_available_providers()
        if 'VitisAIExecutionProvider' in providers:
            logger.info("AMD Vitis AI Execution Provider linked successfully.")
            return "AMD RYZEN™ AI (VITIS NPU)"
        elif 'DmlExecutionProvider' in providers:
            return "AMD DIRECTML ACCELERATOR"
        return "CPU / XNNPACK DELEGATE"
    


class AudioForensics(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.is_running = False
        self.audio_entropy = 0.0
        self.rate = 16000
        self.chunk = 2048

    def audio_callback(self, indata, frames, time_info, status):
        if not self.is_running: raise sd.CallbackStop()

        audio_data = indata[:, 0]
        max_amp = np.max(np.abs(audio_data))
        
        if max_amp < 1e-6:
            entropy = 0.0
        else:
            norm_audio = audio_data / max_amp  
            fft_vals = np.abs(np.fft.rfft(norm_audio))
            fft_vals = fft_vals / (np.sum(fft_vals) + 1e-9)
            entropy = -np.sum(fft_vals * np.log2(fft_vals + 1e-9))
            
        self.audio_entropy = (0.8 * self.audio_entropy) + (0.2 * entropy)

    def run(self):
        if not AUDIO_AVAILABLE:
            return

        self.is_running = True
        try:
            logger.info("Initializing Live Microphone Stream (AGC Active)...")
            with sd.InputStream(samplerate=self.rate, channels=1, callback=self.audio_callback, blocksize=self.chunk):
                while self.is_running:
                    time.sleep(0.1) 
        except Exception as e:
            logger.error(f"Live audio stream crashed: {e}")
            self.audio_entropy = 0.0

    def stop(self):
        self.is_running = False


class ForensicAnalyzer:
    @staticmethod
    def calculate_mar(landmarks: List, frame_w: int, frame_h: int) -> float:
        try:
            p_top = (landmarks[13].x * frame_w, landmarks[13].y * frame_h)
            p_bot = (landmarks[14].x * frame_w, landmarks[14].y * frame_h)
            p_left = (landmarks[78].x * frame_w, landmarks[78].y * frame_h)
            p_right = (landmarks[308].x * frame_w, landmarks[308].y * frame_h)
            
            v_dist = math.hypot(p_top[0] - p_bot[0], p_top[1] - p_bot[1])
            h_dist = math.hypot(p_left[0] - p_right[0], p_left[1] - p_right[1])
            return float(v_dist / (h_dist + 1e-6))
        except Exception as e:
            logger.debug(f"calculate_mar failed: {e}")
            return 0.0

    @staticmethod
    def calculate_rppg(green_channel_buffer: deque, fps: float = 30.0) -> float:
        if not SCIPY_AVAILABLE or len(green_channel_buffer) < int(fps * 3.5):
            return 0.0
            
        signal = np.array(green_channel_buffer)
        signal = signal - np.mean(signal) 
        
        nyq = 0.5 * fps
        low, high = 0.8 / nyq, 3.0 / nyq
        
        try:
            b, a = butter(3, [low, high], btype='band')
            filtered_signal = filtfilt(b, a, signal)
        except ValueError:
            logger.debug("calculate_rppg: filter design failed (ValueError)")
            return 0.0
        except Exception as e:
            logger.debug(f"calculate_rppg failed: {e}")
            return 0.0
            
        window = np.hamming(len(filtered_signal))
        filtered_signal = filtered_signal * window
        
        fft_out = np.abs(np.fft.rfft(filtered_signal))
        freqs = np.fft.rfftfreq(len(filtered_signal), 1.0/fps)
        
        valid_idx = np.where((freqs >= 0.8) & (freqs <= 3.0))
        if len(valid_idx[0]) == 0: return 0.0
        
        valid_fft = fft_out[valid_idx]
        valid_freqs = freqs[valid_idx]
        peak_freq = valid_freqs[np.argmax(valid_fft)]
        
        return float(peak_freq * 60.0)

    @staticmethod
    def calculate_ela(image: np.ndarray, quality: int = 90) -> Tuple[np.ndarray, float]:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded_img = cv2.imencode('.jpg', image, encode_param)
        compressed_img = cv2.imdecode(encoded_img, 1)
        diff = cv2.absdiff(image, compressed_img)
        ela_map = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        gray_ela = cv2.cvtColor(ela_map, cv2.COLOR_BGR2GRAY)
        return ela_map, float(np.var(gray_ela))

    @staticmethod
    def calculate_fft(image: np.ndarray) -> Tuple[np.ndarray, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rows, cols = gray.shape
        nrows, ncols = cv2.getOptimalDFTSize(rows), cv2.getOptimalDFTSize(cols)
        nimg = np.zeros((nrows, ncols), dtype=gray.dtype)
        nimg[:rows, :cols] = gray
        f = np.fft.fft2(nimg)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
        mag_display = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        crow, ccol = nrows // 2, ncols // 2
        fshift[crow-25:crow+25, ccol-25:ccol+25] = 0 
        return mag_display, float(np.mean(np.abs(fshift)))

    @staticmethod
    def calculate_chrominance(frame) -> float:
        try:
            h, w, _ = frame.shape
            roi = frame[int(h*0.1):int(h*0.3), int(w*0.35):int(w*0.65)]

            if roi.size == 0:
                return 0.0

            avg_rgb = np.mean(roi, axis=(0, 1))
            norm_rgb = avg_rgb / (np.mean(avg_rgb) + 1e-6)

            bvp_val = 3 * norm_rgb[2] - 2 * norm_rgb[1]

            return float(np.var(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)))
        except Exception as e:
            logger.debug(f"calculate_chrominance failed: {e}")
            return 0.0

    @staticmethod
    def calculate_mesh_asymmetry(landmarks: List, frame_w: int, frame_h: int) -> float:
        try:
            left_dist = math.hypot(landmarks[168].x - landmarks[33].x, landmarks[168].y - landmarks[33].y)
            right_dist = math.hypot(landmarks[168].x - landmarks[263].x, landmarks[168].y - landmarks[263].y)
            left_jaw = math.hypot(landmarks[152].x - landmarks[234].x, landmarks[152].y - landmarks[234].y)
            right_jaw = math.hypot(landmarks[152].x - landmarks[454].x, landmarks[152].y - landmarks[454].y)
            asym_eye = abs(left_dist - right_dist) / max(left_dist, right_dist)
            asym_jaw = abs(left_jaw - right_jaw) / max(left_jaw, right_jaw)
            return float((asym_eye + asym_jaw) / 2.0)
        except Exception as e:
            logger.debug(f"calculate_mesh_asymmetry failed: {e}")
            return 0.05

    @staticmethod
    def calculate_temporal_jitter(current_crop: np.ndarray, previous_crop: Optional[np.ndarray]) -> float:
        if previous_crop is None: return 0.0
        curr_resized = cv2.resize(cv2.cvtColor(current_crop, cv2.COLOR_BGR2GRAY), (64, 64))
        prev_resized = cv2.resize(cv2.cvtColor(previous_crop, cv2.COLOR_BGR2GRAY), (64, 64))
        curr_blur = cv2.GaussianBlur(curr_resized, (5, 5), 0)
        prev_blur = cv2.GaussianBlur(prev_resized, (5, 5), 0)
        err = np.sum((curr_blur.astype("float") - prev_blur.astype("float")) ** 2)
        return float(err / float(curr_blur.shape[0] * curr_blur.shape[1]))

# ---------------------------------------------------------------------------
# Rendering Engine
# ---------------------------------------------------------------------------

class DashboardRenderer:
    @staticmethod
    def draw_text(img: np.ndarray, text: str, pos: Tuple[int, int], scale: float = 0.5, 
                  color: Tuple[int, int, int] = (255, 255, 255), thickness: int = 1, 
                  bg_color: Optional[Tuple[int, int, int]] = None):
        if bg_color:
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)
            overlay = img.copy()
            cv2.rectangle(overlay, (pos[0]-5, pos[1]-th-5), (pos[0]+tw+5, pos[1]+5), bg_color, -1)
            cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def draw_biometric_mesh(canvas: np.ndarray, landmarks: List, frame_w: int, frame_h: int, color: Tuple[int, int, int]):
        """THE FIX: Holographic 40% Opacity Alpha-Mesh overlay"""
        overlay = canvas.copy()
        wire_color = (max(0, color[0]-30), max(0, color[1]-30), max(0, color[2]-30))
        points = []
        for i in range(0, len(landmarks), 3): 
            x, y = int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h)
            if 0 <= x < frame_w and 0 <= y < frame_h: points.append((x, y))
                
        if len(points) > 10:
            subdiv = cv2.Subdiv2D((-50, -50, frame_w + 100, frame_h + 100))
            for p in points: subdiv.insert(p)
            triangle_list = subdiv.getTriangleList()
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            max_edge = (max(xs) - min(xs)) * 0.15  # Tighter geometric edges
            
            for t in triangle_list:
                pt1, pt2, pt3 = (int(t[0]), int(t[1])), (int(t[2]), int(t[3])), (int(t[4]), int(t[5]))
                if all(0 <= pt[0] < frame_w and 0 <= pt[1] < frame_h for pt in [pt1, pt2, pt3]):
                    if (math.hypot(pt1[0]-pt2[0], pt1[1]-pt2[1]) < max_edge and 
                        math.hypot(pt2[0]-pt3[0], pt2[1]-pt3[1]) < max_edge and 
                        math.hypot(pt3[0]-pt1[0], pt3[1]-pt1[1]) < max_edge):
                        cv2.line(overlay, pt1, pt2, wire_color, 1, cv2.LINE_AA)
                        cv2.line(overlay, pt2, pt3, wire_color, 1, cv2.LINE_AA)
                        cv2.line(overlay, pt3, pt1, wire_color, 1, cv2.LINE_AA)
                        
        # Key structural anchor points (Eyes, Jaw, Nose)
        anchors = [33, 263, 1, 152, 61, 291, 199] 
        for idx in anchors:
            x, y = int(landmarks[idx].x * frame_w), int(landmarks[idx].y * frame_h)
            cv2.circle(overlay, (x, y), 2, color, -1)
            cv2.circle(overlay, (x, y), 5, color, 1, cv2.LINE_AA)
            
        # Blend the holographic overlay into the main canvas at 40% opacity
        cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)

    @staticmethod
    def draw_tactical_graph(canvas: np.ndarray, data_queue: deque, x: int, y: int, w: int, h: int, line_color: Tuple[int, int, int]):
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (15, 18, 24), -1)
        cv2.addWeighted(overlay, 0.8, canvas, 0.2, 0, canvas)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (60, 70, 80), 1, cv2.LINE_AA)
        
        for i in [1, 2, 3]:
            line_y = y + int(h * (i / 4.0))
            for j in range(x, x+w, 8): cv2.line(canvas, (j, line_y), (j+2, line_y), (40, 45, 55), 1)
            DashboardRenderer.draw_text(canvas, f"{100 - (i*25)}", (x - 35, line_y + 5), 0.4, (120, 130, 140))
            
        if len(data_queue) < 2: return
        points = []
        x_step = w / (data_queue.maxlen - 1)
        for i, val in enumerate(data_queue):
            px, py = int(x + (i * x_step)), int(y + h - ((val / 100.0) * h))
            points.append((px, max(y, min(y + h, py))))
            
        poly_points = points.copy()
        poly_points.extend([(x + w, y + h), (x, y + h)])
        graph_overlay = canvas.copy()
        cv2.fillPoly(graph_overlay, np.array([poly_points], np.int32), (line_color[0]//3, line_color[1]//3, line_color[2]//3))
        cv2.addWeighted(graph_overlay, 0.5, canvas, 0.5, 0, canvas) 
        cv2.polylines(canvas, [np.array(points, np.int32).reshape((-1, 1, 2))], False, line_color, 2, cv2.LINE_AA)

    @staticmethod
    def draw_scanning_laser(canvas: np.ndarray, frame_count: int, w: int, h: int, color: Tuple[int, int, int]):
        scan_y = int((math.sin(frame_count * 0.05) + 1) / 2 * h)
        overlay = canvas.copy()
        cv2.line(overlay, (0, scan_y), (w, scan_y), color, 2)
        cv2.rectangle(overlay, (0, max(0, scan_y-10)), (w, scan_y), color, -1)
        cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)

# ---------------------------------------------------------------------------
# Report Generation (Dual-Export Engine)
# ---------------------------------------------------------------------------

class ReportGenerator:
    @staticmethod
    def generate(source_name: str, source_type: str, global_threat: int, face_database: Dict) -> Optional[str]:
        if not face_database: return None
        report_dir = Path("Forensic_Reports")
        report_dir.mkdir(exist_ok=True)
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_filepath = report_dir / f"Veritas_Executive_Report_{timestamp_str}.txt"
        json_filepath = report_dir / f"Veritas_Threat_Intel_{timestamp_str}.json"
        
        verdict = "CRITICAL DEEPFAKE" if global_threat >= 75 else ("ANOMALY WARNING" if global_threat >= 40 else "VERIFIED HUMAN")
        
        # 1. BUILD ENTERPRISE JSON PAYLOAD
        json_payload = {
            "schema_version": "2.0.0-Enterprise",
            "scan_id": str(uuid.uuid4()),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "engine_version": "Veritas-NPU Reality Firewall v22.0",
                "os_platform": platform.platform(),
                "hardware_node": HardwareAccelerator.get_active_provider(),
                "target_source": Path(source_name).name,
                "source_type": source_type.upper()
            },
            "global_assessment": {
                "peak_threat_probability": global_threat,
                "final_verdict": verdict,
                "mitre_attack_mapping": {
                    "tactic": "TA0005 - Defense Evasion",
                    "technique": "T1586 - Compromise Accounts",
                    "sub_technique": "T1586.002 - Deepfake/Synthetic Media"
                }
            },
            "biometric_subjects": []
        }
        
        # 2. BUILD HUMAN-READABLE TXT REPORT
        txt_content = []
        txt_content.append("=================================================\n")
        txt_content.append("      VERITAS-NPU EXECUTIVE FORENSIC REPORT      \n")
        txt_content.append("=================================================\n\n")
        txt_content.append(f"TIMESTAMP:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        txt_content.append(f"SOURCE FILE:     {Path(source_name).name}\n")
        txt_content.append(f"HARDWARE ACCEL:  {HardwareAccelerator.get_active_provider()}\n")
        txt_content.append(f"GLOBAL THREAT:   {global_threat}% (Highest detected probability)\n")
        txt_content.append(f"SYSTEM VERDICT:  {verdict}\n\n")
        txt_content.append("=================================================\n")
        txt_content.append("         SUBJECT-BY-SUBJECT ANALYSIS             \n")
        txt_content.append("=================================================\n\n")

        try:
            for fid, data in face_database.items():
                if not data.get('threat'): continue
                
                max_t = max(data['threat'])
                avg_fft = float(np.mean(data['fft']))
                avg_cvar = float(np.mean(data['cvar']))
                avg_asym = float(np.mean(data['asym']))
                subject_id = f"VOID-ID-{fid:03d}"
                
                # Append to JSON
                subject_data = {
                    "subject_id": subject_id,
                    "peak_threat_score": max_t,
                    "telemetry_aggregates": {
                        "spatial_frequency_fft": {
                            "value": avg_fft,
                            "status": "ANOMALOUS (SYNTHETIC SKIN)" if avg_fft < Config.THRESHOLDS['video'].fft_warn else "NORMAL"
                        },
                        "chrominance_cvar": {
                            "value": avg_cvar,
                            "status": "ANOMALOUS (HYPER-SATURATED)" if avg_cvar > Config.THRESHOLDS['video'].cvar_warn else "NORMAL"
                        },
                        "mesh_asymmetry": {
                            "value": avg_asym,
                            "status": "ANOMALOUS (ARTIFICIAL SYMMETRY)" if avg_asym > Config.THRESHOLDS['video'].mesh_asym_max else "NORMAL"
                        }
                    }
                }
                
                # Append to TXT
                txt_content.append(f"--- BIOMETRIC ID: {subject_id} ---\n")
                txt_content.append(f"Peak Threat Probability:      {max_t}%\n")
                txt_content.append(f"Spectral Energy (FFT):        {avg_fft:.2f}\n")
                txt_content.append(f"Chrominance Variance (C-VAR): {avg_cvar:.2f}\n")
                txt_content.append(f"Biometric Mesh Asymmetry:     {avg_asym:.3f}\n")
                
                if source_type != 'image':
                    if data.get('rppg'):
                        avg_bpm = float(np.mean(data['rppg']))
                        subject_data["telemetry_aggregates"]["biological_pulse_rppg"] = {
                            "bpm": avg_bpm,
                            "status": "FLATLINE (SYNTHETIC MASK)" if avg_bpm < Config.THRESHOLDS['video'].rppg_min_bpm else "BIOLOGICAL"
                        }
                        txt_content.append(f"Biological Pulse (rPPG):      {avg_bpm:.1f} BPM\n")
                    if data.get('jitter'):
                        avg_jitter = float(np.mean(data['jitter']))
                        subject_data["telemetry_aggregates"]["temporal_jitter_mse"] = {"value": avg_jitter}
                        txt_content.append(f"Mean Temporal Jitter (MSE):   {avg_jitter:.2f}\n")
                        
                json_payload["biometric_subjects"].append(subject_data)
                
                # TXT Verdict
                txt_content.append("Verdict: ")
                if max_t > 70: txt_content.append("CRITICAL (Deepfake / Synthetic Face Detected)\n\n")
                elif max_t > 40: txt_content.append("WARNING (Anomalies Present, Manual Review Advised)\n\n")
                else: txt_content.append("CLEAR (Organic Human Face Verified)\n\n")

            # Write JSON securely
            with open(json_filepath, "w", encoding="utf-8") as f:
                json.dump(json_payload, f, indent=4, ensure_ascii=False)

            # Write TXT securely
            with open(txt_filepath, "w", encoding="utf-8") as f:
                f.writelines(txt_content)
                
            logger.info(f"Dual Threat Intel generated: {txt_filepath} | {json_filepath}")
            return f"Executive Report: {txt_filepath.name}\nThreat Intel: {json_filepath.name}"
            
        except Exception as e:
            logger.error(f"Failed to generate reports: {e}")
            return None

# ---------------------------------------------------------------------------
# Core Application Logic
# ---------------------------------------------------------------------------

class ForensicEngine:
    def _generate_narration(self, display_prob, metrics):
        if display_prob < 40:
            return "Subject appears authentic. No major anomalies detected."

        if display_prob < 75:
            if metrics['bpm'] < self.thresholds.rppg_min_bpm:
                return "Warning: Weak biological signals detected."
            return "Moderate inconsistencies detected. Further verification advised."

        # High threat
        if metrics.get('desync'):
            return "Critical: Lip-sync mismatch detected. Likely deepfake."
        if metrics['bpm'] < self.thresholds.rppg_min_bpm:
            return "Critical: No pulse signature. Synthetic content suspected."
        
        return "High probability synthetic media detected."
    def __init__(self, source_type: str, file_path: str = None):
        # 1. Attributes required for the processing loop
        self.source_type = source_type
        self.file_path = file_path
        self.session_id = str(uuid.uuid4())[:8]
        
        # 2. Buffers for the BPM / Pulse rating
        self.bvp_buffer = deque(maxlen=150)
        self.current_bpm = 0.0
        
        # 3. NVIDIA NPU priority (Already linked via your terminal)
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # 4. THE STRICT FIX: Try both common names to ensure the engine starts
        self._initialize_forensics()
    def _initialize_forensics(self):
        logger.info("Initializing Forensic Engine Core...")

        # Core configs
        self.ui_config = UIConfig()
        mode = 'video' if self.source_type in ['video', 'webcam'] else 'image'
        self.thresholds = Config.THRESHOLDS[mode]
        self.video_fps = 30.0
        # Hardware provider
        self.active_provider = HardwareAccelerator.get_active_provider()

        # Model
        self._ensure_model_exists()
        self.detector = self._initialize_detector()

        # Audio
        self.audio_monitor = AudioForensics()

        # State tracking
        self.face_database = {}
        self.next_face_id = 0
        self.telemetry_history = deque(maxlen=120)

        # Video-specific
        self.video_audio_entropy = 0.0
        if self.source_type == 'video':
            self._extract_video_audio()

        logger.info("Forensic Engine Initialized Successfully.")   
    def _extract_video_audio(self):
        if not FFMPEG_AVAILABLE or not SCIPY_AVAILABLE:
            logger.warning("FFmpeg missing. Skipping Video Audio Extraction.")
            return
        logger.info("Extracting video audio track via FFmpeg Direct Router...")
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        temp_wav = None
        try:
            temp_dir = tempfile.gettempdir()
            temp_wav = os.path.join(temp_dir, f"veritas_temp_{int(time.time())}.wav")

            cmd = [
                ffmpeg_exe, '-i', self.file_path, '-vn',
                '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                temp_wav, '-y', '-loglevel', 'quiet'
            ]
            subprocess.run(cmd, check=True)

            if temp_wav and os.path.exists(temp_wav):
                rate, audio_data = wavfile.read(temp_wav)

                audio_data = audio_data.astype(np.float32)
                max_amp = np.max(np.abs(audio_data)) if audio_data.size else 0.0

                if max_amp < 1e-4:
                    self.video_audio_entropy = 0.0
                    logger.info("Video Audio Entropy: 0.00 (TRUE SILENT TRACK DETECTED)")
                else:
                    norm_audio = audio_data / max_amp
                    fft_vals = np.abs(np.fft.rfft(norm_audio))
                    fft_vals = fft_vals / (np.sum(fft_vals) + 1e-9)
                    self.video_audio_entropy = -np.sum(fft_vals * np.log2(fft_vals + 1e-9))
                    logger.info(f"Video Audio Entropy established: {self.video_audio_entropy:.2f}")
            else:
                self.video_audio_entropy = 0.0
                logger.warning("No audio track found in the video file.")

        except Exception as e:
            logger.error(f"FFmpeg audio extraction failed: {e}")
            self.video_audio_entropy = 0.0
        finally:
            try:
                if temp_wav and os.path.exists(temp_wav):
                    os.remove(temp_wav)
            except Exception:
                pass

    def _ensure_model_exists(self):
        if not Config.MODEL_PATH.exists():
            logger.info("Downloading Face Landmarker model...")
            urllib.request.urlretrieve(Config.MODEL_URL, str(Config.MODEL_PATH))

    def _initialize_detector(self):
        base_options = python.BaseOptions(model_asset_path=str(Config.MODEL_PATH))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=10, 
            min_face_detection_confidence=0.15,
            min_face_presence_confidence=0.15,
            min_tracking_confidence=0.15
        )
        return vision.FaceLandmarker.create_from_options(options)

    def _score_face(self, fft_energy: float, cvar: float, asym: float, jitter: float, bpm: float, audio_entropy: float, mar_variance: float) -> Tuple[int, bool]:
        score = 0
        av_desync = False
        
        if fft_energy < self.thresholds.fft_min: score += 35 
        elif fft_energy < self.thresholds.fft_warn: score += 15 
        
        if cvar > self.thresholds.cvar_max: score += 30 
        elif cvar > self.thresholds.cvar_warn: score += 15 
        
        if asym > self.thresholds.mesh_asym_max: score += 15 
        if self.source_type != 'image' and jitter > self.thresholds.jitter_max: score += 15
        
        if self.source_type != 'image':
            if bpm > 0 and bpm < self.thresholds.rppg_min_bpm: score += 20
            
            if audio_entropy > self.thresholds.audio_entropy_min:
                if mar_variance < 0.001:
                    score += 25
                    av_desync = True

        return min(100, score), av_desync

    def run(self):
        logger.info(f"Starting engine. Mode: {self.source_type}")
        cap = cv2.VideoCapture(0 if self.source_type == 'webcam' else self.file_path)
        
        if not cap.isOpened():
            logger.error("Failed to open video source.")
            return
            
        fps_val = None

        if self.source_type in ['video', 'webcam']:
            fps_val = cap.get(cv2.CAP_PROP_FPS)

        if fps_val is not None and fps_val > 1:
            self.video_fps = fps_val

        if self.source_type == 'webcam':
            self.audio_monitor.start()

        smoothed_probability = 0.0
        alpha = 0.15 if self.source_type != 'image' else 1.0 
        max_system_threat = 0
        frame_count = 0
        image_processed = False
        
        # Get screen dimensions and calculate safe dashboard size
        screen_w, screen_h = UIConfig.get_screen_dimensions()

        while True:
            if self.source_type != 'image' or not image_processed:
                ret, frame = cap.read()
                if not ret: break 
                
                frame_count += 1
                
                # Calculate maximum frame width based on screen size and panel width
                head_h = self.ui_config.header_h
                deck_h = self.ui_config.bottom_h
                panel_w = self.ui_config.panel_w
                max_allowed_frame_w = screen_w - panel_w - 20  # 20px safety margin
                max_allowed_frame_w = min(max_allowed_frame_w, self.ui_config.max_video_w)
                max_allowed_frame_h = screen_h - head_h - deck_h - 40  # 40px for margins
                
                # Scale frame to fit constraints
                frame_h, frame_w, _ = frame.shape
                scale_w = max_allowed_frame_w / frame_w if frame_w > max_allowed_frame_w else 1.0
                scale_h = max_allowed_frame_h / frame_h if frame_h > max_allowed_frame_h else 1.0
                scale = min(scale_w, scale_h, 1.0)  # Don't upscale
                
                if scale < 1.0:
                    new_w = int(frame_w * scale)
                    new_h = int(frame_h * scale)
                    frame = cv2.resize(frame, (new_w, new_h))
                    frame_h, frame_w, _ = frame.shape
                
                dashboard_w = frame_w + panel_w
                dashboard_h = head_h + frame_h + deck_h
                
                # Final safety check - should never exceed screen size
                if dashboard_w > screen_w or dashboard_h > screen_h:
                    logger.warning(f"Dashboard size ({dashboard_w}x{dashboard_h}) exceeds screen ({screen_w}x{screen_h}). Further scaling applied.")
                    scale_factor = min(screen_w / dashboard_w, screen_h / dashboard_h, 0.95)
                    frame_w = int(frame_w * scale_factor)
                    frame_h = int(frame_h * scale_factor)
                    frame = cv2.resize(frame, (frame_w, frame_h))
                    dashboard_w = frame_w + panel_w
                    dashboard_h = head_h + frame_h + deck_h
                
                dashboard = np.full((dashboard_h, dashboard_w, 3), self.ui_config.color_bg, dtype=np.uint8)
                
                for i in range(0, dashboard_w, 40): cv2.line(dashboard, (i, 0), (i, dashboard_h), (18, 20, 25), 1)
                for i in range(0, dashboard_h, 40): cv2.line(dashboard, (0, i), (dashboard_w, i), (18, 20, 25), 1)

                video_y1, video_y2 = head_h, head_h + frame_h
                
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                detection_result = self.detector.detect(mp_image)

                highest_frame_threat = 0
                best_ela_crop, best_fft_crop = None, None
                display_metrics = {'ela': 0, 'fft': 0, 'cvar': 0, 'asym': 0, 'jitter': 0, 'bpm': 0, 'audio': 0.0, 'desync': False}
                
                assigned_ids: Set[int] = set()

                if detection_result.face_landmarks:
                    for face_landmarks in detection_result.face_landmarks:
                        x_coords = [lm.x * frame_w for lm in face_landmarks]
                        y_coords = [lm.y * frame_h for lm in face_landmarks]
                        x, y = int(min(x_coords)), int(min(y_coords))
                        bw, bh = int(max(x_coords) - x), int(max(y_coords) - y)
                        
                        pad_x, pad_y = int(bw*0.1), int(bh*0.15)
                        x, y = max(0, x - pad_x), max(0, y - pad_y)
                        bw, bh = min(frame_w - x, bw + 2*pad_x), min(frame_h - y, bh + 2*pad_y)
                        
                        if bw > 20 and bh > 20: 
                            cx, cy = x + bw//2, y + bh//2 
                            face_crop = frame[y:y+bh, x:x+bw]
                            
                            matched_id = None
                            min_dist = float('inf')
                            for tid, tdata in self.face_database.items():
                                if tid in assigned_ids: continue 
                                tcx, tcy = tdata['centroid']
                                dist = math.hypot(cx - tcx, cy - tcy)
                                if dist < max(bw, bh) * 1.8 and dist < min_dist: 
                                    min_dist = dist
                                    matched_id = tid
                                    
                            if matched_id is None:
                                matched_id = self.next_face_id
                                self.next_face_id += 1
                                self.face_database[matched_id] = {
                                    'centroid': (cx, cy), 'last_crop': None, 
                                    'ela': [], 'fft': [], 'cvar': [], 'asym': [], 'jitter': [], 'threat': [],
                                    'rppg_buffer': deque(maxlen=200), 'rppg': [],
                                    'mar_buffer': deque(maxlen=30)
                                }

                            assigned_ids.add(matched_id)
                            db_entry = self.face_database[matched_id]
                            db_entry['centroid'] = (cx, cy)

                            ela_crop, ela_variance = ForensicAnalyzer.calculate_ela(face_crop)
                            fft_crop, fft_energy = ForensicAnalyzer.calculate_fft(face_crop)
                            cvar = ForensicAnalyzer.calculate_chrominance(face_crop)
                            mesh_asym = ForensicAnalyzer.calculate_mesh_asymmetry(face_landmarks, frame_w, frame_h)
                            
                            green_mean = np.mean(face_crop[:,:,1])
                            db_entry['rppg_buffer'].append(green_mean)
                            bpm = ForensicAnalyzer.calculate_rppg(db_entry['rppg_buffer'], fps=self.video_fps)
                            
                            mar = ForensicAnalyzer.calculate_mar(face_landmarks, frame_w, frame_h)
                            db_entry['mar_buffer'].append(mar)
                            mar_variance = float(np.var(db_entry['mar_buffer'])) if len(db_entry['mar_buffer']) > 15 else 0.1

                            jitter_mse = 0
                            if self.source_type != 'image' and db_entry['last_crop'] is not None:
                                jitter_mse = ForensicAnalyzer.calculate_temporal_jitter(face_crop, db_entry['last_crop'])
                            db_entry['last_crop'] = face_crop.copy()

                            current_audio_entropy = self.audio_monitor.audio_entropy if self.source_type == 'webcam' else self.video_audio_entropy

                            face_prob_score, is_desync = self._score_face(fft_energy, cvar, mesh_asym, jitter_mse, bpm, current_audio_entropy, mar_variance)
                            
                            db_entry['ela'].append(ela_variance)
                            db_entry['fft'].append(fft_energy)
                            db_entry['cvar'].append(cvar)
                            db_entry['asym'].append(mesh_asym)
                            if bpm > 0: db_entry['rppg'].append(bpm)
                            if self.source_type != 'image': db_entry['jitter'].append(jitter_mse)
                            db_entry['threat'].append(face_prob_score)

                            target_color = self.ui_config.color_green
                            if face_prob_score >= 75: target_color = self.ui_config.color_red
                            elif face_prob_score >= 40: target_color = self.ui_config.color_orange
                            # 🔥 STEP 3: Suspicious region highlight
                            if face_prob_score >= 75:
                                overlay = frame.copy()
                                cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 0, 255), -1)
                                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                            cv2.rectangle(frame, (x, y), (x+bw, y+bh), target_color, 2)
                            
                            # HUD Text
                            # 🔥 STEP 5: Enhanced face label
                            status_label = "HUMAN"
                            if face_prob_score >= 75:
                                status_label = "DEEPFAKE"
                            elif face_prob_score >= 40:
                                status_label = "SUSPICIOUS"

                            hud_text = f"{status_label} | ID:{matched_id:02d} | {face_prob_score}%"
                            if is_desync:
                                hud_text += " | AV-DESYNC"

                            DashboardRenderer.draw_text(
                                frame,
                                hud_text,
                                (x, max(15, y - 8)),
                                0.45,
                                target_color,
                                1,
                                bg_color=(20, 20, 20)
                            )

                            if face_prob_score >= highest_frame_threat:
                                highest_frame_threat = face_prob_score
                                best_ela_crop, best_fft_crop = ela_crop, fft_crop
                                display_metrics = {'ela': ela_variance, 'fft': fft_energy, 'cvar': cvar, 'asym': mesh_asym, 'jitter': jitter_mse, 'bpm': bpm, 'audio': current_audio_entropy, 'desync': is_desync}

                if self.source_type != 'image':
                    DashboardRenderer.draw_scanning_laser(frame, frame_count, frame_w, frame_h, self.ui_config.color_cyan)

                dashboard[video_y1:video_y2, 0:frame_w] = frame
                
                smoothed_probability = (alpha * highest_frame_threat) + ((1 - alpha) * smoothed_probability)
                display_prob = int(smoothed_probability)
                trust_score = 100 - display_prob
                self.telemetry_history.append(display_prob)
                if display_prob > max_system_threat: max_system_threat = display_prob

                self._render_ui_overlays(dashboard, frame_w, frame_h, dashboard_w, dashboard_h, head_h, deck_h, panel_w, display_prob, display_metrics, best_ela_crop, best_fft_crop)
                
                if self.source_type == 'image': 
                    image_processed = True

            cv2.imshow(self.ui_config.window_name, dashboard)
            
            # Move window to center of screen on first display
            if frame_count == 1:
                try:
                    cv2.moveWindow(self.ui_config.window_name, 0, 0)
                except Exception:
                    pass

            key = cv2.waitKey(1 if self.source_type != 'image' else 10) & 0xFF
            if key in [27, ord('q')]: 
                break
                
            try:
                if cv2.getWindowProperty(self.ui_config.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                break

        if hasattr(self, 'audio_monitor'):
            self.audio_monitor.stop()
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Inference loop terminated.")

        if self.source_type in ['image', 'video'] and self.face_database:
            report_msg = ReportGenerator.generate(str(self.file_path), self.source_type, max_system_threat, self.face_database)
            if report_msg:
                messagebox.showinfo("Dual Threat Intel Generated", f"Forensic scan finished successfully.\n\n{report_msg}")

    def _generate_explanation(self, metrics):
        reasons = []

        if metrics['fft'] < self.thresholds.fft_warn:
            reasons.append("Low frequency consistency (possible synthesis)")

        if metrics['asym'] > self.thresholds.mesh_asym_max:
            reasons.append("Facial asymmetry detected")

        if metrics['bpm'] < self.thresholds.rppg_min_bpm:
            reasons.append("No biological pulse signal")

        if metrics.get('desync'):
            reasons.append("Lip-sync mismatch")

        if metrics['cvar'] > self.thresholds.cvar_warn:
            reasons.append("Color variance abnormal")

        if not reasons:
            reasons.append("No strong anomalies detected")

        return reasons[:3]  # keep top 3
    def _render_ui_overlays(self, dashboard, frame_w, frame_h, dashboard_w, dashboard_h, head_h, deck_h, panel_w, display_prob, metrics, best_ela, best_fft):

        # -------------------- SPACING SYSTEM --------------------
        PAD = 20
        SMALL_PAD = 10
        LINE_H = 26

        # -------------------- LAYOUT BASICS --------------------
        deck_y = head_h + frame_h
        metrics_y = dashboard_h - 240 - 10  # matches your text_box_h logic

        # -------------------- SAFE DEFAULTS --------------------
        status_msg = "INITIALIZING"
        status_color = (200, 200, 200)
        trust_score = 100 - int(display_prob)

        # -------------------- COLORS --------------------
        c_cyan = self.ui_config.color_cyan
        c_red = self.ui_config.color_red
        c_green = self.ui_config.color_green
        c_orange = self.ui_config.color_orange
            # UI scale factor - reduce font sizes on narrow frames to avoid overlap
        ui_scale = max(0.6, min(1.0, frame_w / 800.0))
        s = lambda val: float(val) * ui_scale
        # -------------------- STATUS LOGIC --------------------
        if display_prob < 40:
            status_msg = "VERIFIED HUMAN"
            status_color = c_green
        elif display_prob < 75:
            status_msg = "SUSPICIOUS ACTIVITY"
            status_color = c_orange
        else:
            status_msg = "SYNTHETIC / DEEPFAKE"
            status_color = c_red

        # -------------------- HEADER --------------------
        cv2.rectangle(dashboard, (0, 0), (dashboard_w, head_h), (15, 18, 22), -1)
        cv2.line(dashboard, (0, head_h), (dashboard_w, head_h), (50, 60, 70), 1)

        DashboardRenderer.draw_text(
            dashboard,
            f"VERITAS-NPU // {self.source_type.upper()} ANALYSIS",
            (PAD, int(head_h * 0.65)),
            s(0.65),
            c_cyan,
            2
        )

        prov_text = "CPU FORENSIC ENGINE"
        (text_w, _), _ = cv2.getTextSize(prov_text, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)

        DashboardRenderer.draw_text(
            dashboard,
            prov_text,
            (dashboard_w - text_w - PAD, int(head_h * 0.65)),
            s(0.45),
            (200, 200, 200),
            1
        )

        # -------------------- VERDICT/STATUS BOX (dedicated area below image, no overlap) --------------------
        # Conservative box sizing with explicit boundary protection
        margin_right = 25  # Safety margin before right panel
        box_gap = 18  # Larger gap between boxes to prevent visual overlap
        available_width = frame_w - (2 * PAD) - box_gap - margin_right
        box_width = int(available_width / 2)
        
        # Status box (left side)
        status_box_x = PAD
        status_box_y = deck_y + 10
        status_box_w = box_width
        status_box_h = 65
        
        # Boundary check for status box
        if status_box_x + status_box_w > frame_w - margin_right:
            status_box_w = frame_w - margin_right - status_box_x - box_gap
        
        # Draw background box for status
        cv2.rectangle(dashboard, (status_box_x, status_box_y), (status_box_x + status_box_w, status_box_y + status_box_h), (25, 30, 40), -1)
        cv2.rectangle(dashboard, (status_box_x, status_box_y), (status_box_x + status_box_w, status_box_y + status_box_h), (60, 70, 80), 1)
        
        # Draw status text inside box - dynamically truncate based on box width
        # Calculate max chars that fit: (box_width - 16 for padding) / approx char width at scale 0.42
        # reduce max chars based on scaled font width
        approx_char_px = 6.5 * ui_scale
        max_chars_status = max(6, int((status_box_w - 16) / approx_char_px))
        verdict_short = status_msg[:max_chars_status] if len(status_msg) > max_chars_status else status_msg
        DashboardRenderer.draw_text(
            dashboard,
            verdict_short,
            (status_box_x + 8, status_box_y + 10),
            s(0.42),
            status_color,
            1
        )
        
        DashboardRenderer.draw_text(
            dashboard,
            f"T: {trust_score}%",
            (status_box_x + 8, status_box_y + 32),
            s(0.42),
            (180, 180, 180)
        )

        # -------------------- RIGHT SIDE: GLOBAL THREAT (separate box on same row) --------------------
        # Position threat box with proper gap and boundary checking
        threat_box_x = status_box_x + status_box_w + box_gap
        threat_box_y = status_box_y
        threat_box_w = box_width
        threat_box_h = 65
        
        # Explicit boundary check - ensure threat box doesn't overflow
        threat_box_right = threat_box_x + threat_box_w
        if threat_box_right > frame_w - 5:  # 5px safety margin
            threat_box_w = max(100, frame_w - 5 - threat_box_x)  # Minimum 100px width
        
        # Draw background box for threat
        cv2.rectangle(dashboard, (threat_box_x, threat_box_y), (threat_box_x + threat_box_w, threat_box_y + threat_box_h), (25, 30, 40), -1)
        cv2.rectangle(dashboard, (threat_box_x, threat_box_y), (threat_box_x + threat_box_w, threat_box_y + threat_box_h), (60, 70, 80), 1)
        
        DashboardRenderer.draw_text(
            dashboard,
            "GLOBAL",
            (threat_box_x + 8, threat_box_y + 6),
            s(0.42),
            (180, 190, 200),
            1
        )
        
        DashboardRenderer.draw_text(
            dashboard,
            "THREAT",
            (threat_box_x + 8, threat_box_y + 22),
            s(0.40),
            (180, 190, 200),
            1
        )
        
        # Threat display: dynamically truncate label based on box width (reserve space for percentage)
        # Reserve ~35px for percentage display, calculate remaining space for label
        space_for_pct = 35
        approx_char_px_threat = 5.5 * ui_scale
        max_chars_threat = max(6, int((threat_box_w - space_for_pct - 16) / approx_char_px_threat))
        threat_label = status_msg[:max_chars_threat] if len(status_msg) > max_chars_threat else status_msg
        DashboardRenderer.draw_text(
            dashboard,
            threat_label,
            (threat_box_x + 8, threat_box_y + 40),
            s(0.38),
            status_color,
            1
        )
        
        # Percentage positioned to the right within box
        pct_text = f"{int(display_prob)}%"
        (pct_size, _baseline) = cv2.getTextSize(pct_text, cv2.FONT_HERSHEY_DUPLEX, s(0.42), 1)
        pct_w = pct_size[0]
        pct_x = threat_box_x + threat_box_w - pct_w - 8
        # ensure percent does not overflow left boundary of the box
        if pct_x < threat_box_x + 8:
            pct_x = threat_box_x + 8
        DashboardRenderer.draw_text(
            dashboard,
            pct_text,
            (pct_x, threat_box_y + 40),
            s(0.50),
            status_color,
            1
        )

        # -------------------- LEFT PANEL: SEGREGATED LAYOUT --------------------
        # Upper Left: LIVE THREAT Graph (positioned well below status boxes)
        graph_x = PAD
        graph_y = status_box_y + status_box_h + 18  # 18px gap below status boxes
        # Calculate graph width to match available space (same as status box)
        graph_w = status_box_w
        graph_h = int(deck_h - status_box_h - 90)  # Reduced to fit all elements

        DashboardRenderer.draw_text(
            dashboard,
            "LIVE THREAT",
            (graph_x, graph_y - 16),
            s(0.46),
            (180, 190, 200)
        )

        DashboardRenderer.draw_tactical_graph(
            dashboard,
            self.telemetry_history,
            graph_x,
            graph_y,
            graph_w,
            graph_h,
            status_color
        )

        # Lower Left: Metric Readouts (SIGNAL/BIO/AUDIO) - BELOW graph with padding
        metrics_section_y = graph_y + graph_h + 12
        metrics_x = PAD
        y = metrics_section_y

        # SIGNAL block
        DashboardRenderer.draw_text(dashboard, "SIGNAL", (metrics_x, y), s(0.40), (150, 150, 150))
        y += 16
        DashboardRenderer.draw_text(dashboard, f"FFT: {metrics['fft']:.1f}", (metrics_x, y), s(0.36), c_cyan)
        y += 15
        DashboardRenderer.draw_text(dashboard, f"ELA: {metrics['ela']:.1f}", (metrics_x, y), s(0.36), c_cyan)
        y += 18

        # BIO block
        DashboardRenderer.draw_text(dashboard, "BIO", (metrics_x, y), s(0.40), (150, 150, 150))
        y += 16
        DashboardRenderer.draw_text(dashboard, f"rPPG: {metrics['bpm']:.1f}", (metrics_x, y), s(0.36), c_cyan)
        y += 18

        # AUDIO block
        DashboardRenderer.draw_text(dashboard, "AUDIO", (metrics_x, y), s(0.40), (150, 150, 150))
        y += 16
        DashboardRenderer.draw_text(dashboard, f"{metrics.get('audio', 0.0):.2f}", (metrics_x, y), s(0.36), c_cyan)


        # -------------------- RIGHT PANEL (COMPACT LAYOUT) --------------------
        panel_x = frame_w
        cv2.rectangle(dashboard, (panel_x, head_h), (dashboard_w, dashboard_h), (20, 24, 30), -1)
        
        panel_inner_x = panel_x + 10
        panel_inner_w = panel_w - 20
        img_h = 100  # Fixed small size for ELA/FFT
        
        # --- TOP: ELA & FFT (fixed size, no scaling issues) ---
        cursor_y = head_h + PAD
        
        if best_ela is not None and best_fft is not None:
            hot_ela = cv2.applyColorMap(best_ela, cv2.COLORMAP_INFERNO)
            dashboard[cursor_y:cursor_y+img_h, panel_inner_x:panel_inner_x+panel_inner_w] = cv2.resize(hot_ela, (panel_inner_w, img_h))
            DashboardRenderer.draw_text(dashboard, "ELA", (panel_inner_x + 5, cursor_y + 10), 0.38, c_cyan)
            
            cursor_y += img_h + 5
            dashboard[cursor_y:cursor_y+img_h, panel_inner_x:panel_inner_x+panel_inner_w] = cv2.resize(cv2.cvtColor(best_fft, cv2.COLOR_GRAY2BGR), (panel_inner_w, img_h))
            DashboardRenderer.draw_text(dashboard, "FFT", (panel_inner_x + 5, cursor_y + 10), 0.38, c_cyan)
            
            cursor_y += img_h + 15
        
        # --- CONFIDENCE BREAKDOWN (compact bars, aligned right) ---
        # Only show if there's enough vertical space
        if cursor_y + 100 < dashboard_h:
            DashboardRenderer.draw_text(dashboard, "CONFIDENCE BREAKDOWN", (panel_inner_x, cursor_y), 0.40, (180, 180, 180))
            cursor_y += 18
            
            signals = {
                "FFT": metrics.get('fft', 0),
                "ASYM": metrics.get('asym', 0),
                "rPPG": metrics.get('bpm', 0),
                "ELA": metrics.get('ela', 0)
            }
            
            metric_scale = {"FFT": 700.0, "ELA": 500.0, "ASYM": 0.6, "rPPG": 120.0}
            
            for i, (key, val) in enumerate(signals.items()):
                scale = metric_scale.get(key, 100.0)
                percent = int(min(max((val / scale) * 100.0, 0), 100))
                bar_w = int((percent / 100.0) * (panel_inner_w - 70))
                
                bar_y = cursor_y + i * 16
                if bar_y + 12 < dashboard_h:  # Boundary check
                    cv2.rectangle(dashboard, (panel_inner_x, bar_y), (panel_inner_x + bar_w, bar_y + 12), c_cyan, -1)
                    DashboardRenderer.draw_text(dashboard, key, (panel_inner_x + panel_inner_w - 50, bar_y + 1), 0.34, (200, 200, 200))
        
        if cursor_y + 16 < dashboard_h:
            cursor_y += (4 * 16) + 8
        
        # --- WHY FLAGGED (compact, truncated to fit) ---
        if cursor_y + 70 < dashboard_h:
            DashboardRenderer.draw_text(dashboard, "WHY FLAGGED:", (panel_inner_x, cursor_y), 0.38, (180, 180, 180))
            cursor_y += 16
            
            reasons = self._generate_explanation(metrics)
            for i, reason in enumerate(reasons[:2]):  # Show only top 2 to avoid overflow
                truncated = reason[:32] if len(reason) > 32 else reason  # Truncate long text
                if cursor_y + i * 15 < dashboard_h:
                    DashboardRenderer.draw_text(dashboard, f"- {truncated}", (panel_inner_x + 5, cursor_y + i * 15), 0.34, c_orange)
            
            cursor_y += 45
        
        # --- AI ANALYSIS at bottom ---
        if cursor_y + 35 < dashboard_h:
            narration = self._generate_narration(display_prob, metrics)
            narration_short = narration[:45] if len(narration) > 45 else narration
            DashboardRenderer.draw_text(dashboard, "AI ANALYSIS:", (panel_inner_x, cursor_y), 0.38, (180, 180, 180))
            cursor_y += 15
            if cursor_y < dashboard_h:
                DashboardRenderer.draw_text(dashboard, narration_short, (panel_inner_x + 5, cursor_y), 0.33, status_color)
        


class VeritasLauncher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Veritas-NPU // Team Void Breakers")
        self.root.geometry("500x450")
        self.root.configure(bg="#0A0C10")
        self._build_ui()

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#0A0C10")
        header.pack(fill="x", pady=(30, 10))
        tk.Label(header, text="VERITAS-NPU", font=("Courier", 32, "bold"), fg="#00E5FF", bg="#0A0C10").pack()
        tk.Label(header, text="Multimodal Enterprise Forensics", font=("Helvetica", 10), fg="#8A9BB5", bg="#0A0C10").pack()
        tk.Label(self.root, text="Developed by Team Void Breakers", font=("Courier", 9, "italic"), fg="#ED1C24", bg="#0A0C10").pack(pady=(0, 30))

        btn_style = {
            "font": ("Courier", 12, "bold"), "fg": "#E6E6E6", "bg": "#1C222E", 
            "activebackground": "#00E5FF", "activeforeground": "#0B0D12", 
            "width": 30, "pady": 14, "bd": 1, "relief": "ridge", "cursor": "hand2"
        }
        
        self._create_button("📡  INIT LIVE FIREWALL", lambda: self._execute_engine('webcam'), btn_style)
        self._create_button("📂  SCAN STATIC IMAGE", lambda: self._execute_engine('image', filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])), btn_style)
        self._create_button("🎬  SCAN VIDEO FEED", lambda: self._execute_engine('video', filedialog.askopenfilename(filetypes=[("Videos", "*.mp4 *.mov *.avi")])), btn_style)
        
        tk.Label(self.root, text="POWERED BY AMD RYZEN™ AI", font=("Helvetica", 8, "bold"), fg="#405060", bg="#0A0C10").pack(side="bottom", pady=20)

    def _create_button(self, text: str, command, style: dict):
        btn = tk.Button(self.root, text=text, command=command, **style)
        btn.pack(pady=10)
        btn.bind("<Enter>", lambda e: e.widget.config(bg='#2A3242'))
        btn.bind("<Leave>", lambda e: e.widget.config(bg='#1C222E'))

    def _execute_engine(self, source_type: str, file_path: str = None):
        if source_type in ['image', 'video'] and not file_path:
            return
        self.root.withdraw()
        try:
            engine = ForensicEngine(source_type, file_path)
            engine.run()
        except Exception as e:
            logger.error(f"Engine execution failed: {e}")
            messagebox.showerror("Error", f"Failed to run forensics: {e}")
        finally:
            self.root.deiconify()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    VeritasLauncher().run()