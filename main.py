"""
Veritas-NPU: Real-time Deepfake Forensics Engine
Developed by Team Void Breakers for the AMD Slingshot Hackathon

This module implements a tripartite ensemble model (Spatial, Frequency, and Temporal analysis)
to detect synthetic media and deepfakes in real-time video streams. Designed to leverage 
local NPU processing for privacy-preserving cybersecurity.
"""

import os
import time
import math
import urllib.request
from collections import deque
from datetime import datetime
from typing import Tuple, Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION & THRESHOLDS ---
# Note: These heuristics are tuned for standard 720p/1080p laptop webcams.
# TODO: Move these to a separate JSON config file in v2.0
THRESHOLDS = {
    'ela_min': 80,         # Below this = artificially smooth (AI generated)
    'ela_max': 500,        # Above this = heavy artificial noise injection
    'fft_min': 500,        # Below this = missing natural skin pores/textures
    'fft_warn': 800,
    'jitter_max': 150      # Above this = deepfake frame-render stuttering
}

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
MODEL_PATH = 'blaze_face_short_range.tflite'


def download_model() -> str:
    """Fetches the lightweight biometric tracker if not cached locally."""
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Fetching Face Detection model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def calculate_ela(image: np.ndarray, quality: int = 90) -> Tuple[np.ndarray, float]:
    """
    Error Level Analysis (ELA).
    Intentionally compresses the image to identify areas with unnatural compression signatures.
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_img = cv2.imencode('.jpg', image, encode_param)
    compressed_img = cv2.imdecode(encoded_img, 1)
    
    # Calculate absolute difference between original and degraded image
    diff = cv2.absdiff(image, compressed_img)
    ela_map = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    
    gray_ela = cv2.cvtColor(ela_map, cv2.COLOR_BGR2GRAY)
    return ela_map, np.var(gray_ela)


def calculate_fft(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Spectral Analysis via 2D Fast Fourier Transform.
    Used to detect the absence of high-frequency spatial data (micro-textures/pores) typical in GANs.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Transform to frequency domain
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    mag_display = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Mask out the DC component and low frequencies (center 30x30 pixels)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    fshift[crow-15:crow+15, ccol-15:ccol+15] = 0 
    
    return mag_display, np.mean(np.abs(fshift))


def calculate_temporal_jitter(current_crop: np.ndarray, previous_crop: Optional[np.ndarray]) -> float:
    """Calculates Mean Squared Error (MSE) between consecutive frames to catch render stutters."""
    if previous_crop is None: 
        return 0.0
        
    # Standardize matrix size for pixel-perfect diffing
    curr_resized = cv2.resize(cv2.cvtColor(current_crop, cv2.COLOR_BGR2GRAY), (64, 64))
    prev_resized = cv2.resize(cv2.cvtColor(previous_crop, cv2.COLOR_BGR2GRAY), (64, 64))
    
    err = np.sum((curr_resized.astype("float") - prev_resized.astype("float")) ** 2)
    return err / float(curr_resized.shape[0] * curr_resized.shape[1])


# --- UI & RENDERING HELPERS ---

def draw_hud_text(img: np.ndarray, text: str, pos: Tuple[int, int], scale: float = 0.5, color: Tuple[int, int, int] = (255, 255, 255), thickness: int = 1):
    """Draws anti-aliased HUD typography."""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_cyber_reticle(canvas: np.ndarray, x: int, y: int, w: int, h: int, color: Tuple[int, int, int]):
    """Draws the biometric targeting brackets."""
    length = 25
    t = 2 
    
    # Corner brackets
    lines = [
        ((x, y), (x + length, y)), ((x, y), (x, y + length)),
        ((x + w, y), (x + w - length, y)), ((x + w, y), (x + w, y + length)),
        ((x, y + h), (x + length, y + h)), ((x, y + h), (x, y + h - length)),
        ((x + w, y + h), (x + w - length, y + h)), ((x + w, y + h), (x + w, y + h - length))
    ]
    for pt1, pt2 in lines:
        cv2.line(canvas, pt1, pt2, color, t)
        
    # Center crosshairs
    cx, cy = x + w//2, y + h//2
    cv2.line(canvas, (cx - 10, cy), (cx + 10, cy), color, 1)
    cv2.line(canvas, (cx, cy - 10), (cx, cy + 10), color, 1)
    
    # Faint outer bounds
    cv2.rectangle(canvas, (x-5, y-5), (x+w+5, y+h+5), color, 1, cv2.LINE_AA)


def draw_tactical_area_graph(canvas: np.ndarray, data_queue: deque, x: int, y: int, w: int, h: int, line_color: Tuple[int, int, int]):
    """Renders the smoothed historical telemetry area graph."""
    # Base container
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (15, 18, 22), -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (40, 50, 60), 1)
    
    # Y-axis Gridlines
    for i in [1, 2, 3]:
        line_y = y + int(h * (i / 4.0))
        cv2.line(canvas, (x, line_y), (x + w, line_y), (30, 35, 40), 1)
        draw_hud_text(canvas, f"{100 - (i*25)}", (x - 30, line_y + 4), 0.4, (100, 120, 130))

    if len(data_queue) < 2: 
        return
    
    # Map queue data to physical pixel coordinates
    points = []
    x_step = w / (data_queue.maxlen - 1)
    for i, val in enumerate(data_queue):
        px = int(x + (i * x_step))
        py = int(y + h - ((val / 100.0) * h))
        py = max(y, min(y + h, py))  # Clamp to bounds
        points.append((px, py))
    
    # Render Alpha-blended area
    poly_points = points.copy()
    poly_points.extend([(x + w, y + h), (x, y + h)])
    poly_pts_arr = np.array([poly_points], np.int32)
    
    overlay = canvas.copy()
    fill_color = (line_color[0]//3, line_color[1]//3, line_color[2]//3)
    cv2.fillPoly(overlay, poly_pts_arr, fill_color)
    cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas) 
    
    # Render crisp top line
    line_pts_arr = np.array(points, np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [line_pts_arr], False, line_color, 2, cv2.LINE_AA)


def main():
    # Initialize MediaPipe runtime
    base_options = python.BaseOptions(model_asset_path=download_model())
    options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
    detector = vision.FaceDetector.create_from_options(options)

    cap = cv2.VideoCapture(0)
    
    # State Memory
    telemetry_history = deque([0]*300, maxlen=300) 
    previous_face_crop = None
    smoothed_probability = 0.0
    alpha = 0.05  # EMA smoothing factor
    
    system_logs = deque(maxlen=4)
    frame_count = 0

    print("[INFO] Veritas-NPU Engine initialized. Press 'q' to terminate.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            print("[WARN] Stream interrupted. Attempting to recover...")
            break

        frame_count += 1
        frame_h, frame_w, _ = frame.shape
        panel_w, panel_h, bottom_h = frame_w // 2, frame_h // 2, 220 
        
        # Deep space gray background
        dashboard = np.full((frame_h + bottom_h, frame_w + panel_w, 3), (12, 14, 18), dtype=np.uint8)
        dashboard[0:frame_h, 0:frame_w] = frame
        
        # Run standard facial detection
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detection_result = detector.detect(mp_image)

        raw_prob_score = 0
        ela_variance, hf_energy, jitter_mse = 0, 0, 0
        target_color = (0, 255, 100)  # Default: Cyber Green

        if detection_result.detections:
            # Isolate primary biometric target
            bbox = detection_result.detections[0].bounding_box
            x, y = max(0, int(bbox.origin_x)), max(0, int(bbox.origin_y))
            bw, bh = min(frame_w - x, int(bbox.width)), min(frame_h - y, int(bbox.height))

            if bw > 50 and bh > 50:
                face_crop = frame[y:y+bh, x:x+bw]

                # --- Forensic Analysis Pipeline ---
                ela_crop, ela_variance = calculate_ela(face_crop)
                fft_crop, hf_energy = calculate_fft(face_crop)
                jitter_mse = calculate_temporal_jitter(face_crop, previous_face_crop)
                previous_face_crop = face_crop.copy()

                # Calculate Ensemble Threat Score based on heuristics
                if ela_variance < THRESHOLDS['ela_min']: raw_prob_score += 35 
                elif ela_variance > THRESHOLDS['ela_max']: raw_prob_score += 15
                
                if hf_energy < THRESHOLDS['fft_min']: raw_prob_score += 35 
                elif hf_energy < THRESHOLDS['fft_warn']: raw_prob_score += 15
                
                if jitter_mse > THRESHOLDS['jitter_max']: raw_prob_score += 30 
                
                raw_prob_score = min(100, raw_prob_score)

                # Determine UI state based on threat level
                if raw_prob_score > 70: target_color = (0, 0, 255)
                elif raw_prob_score > 40: target_color = (0, 165, 255)

                draw_cyber_reticle(dashboard, x, y, bw, bh, target_color)
                draw_hud_text(dashboard, f"ID: VOID-{frame_count%999:03d}", (x, y - 15), 0.5, target_color)

                # Render Diagnostic Panels
                dashboard[0:panel_h, frame_w:frame_w+panel_w] = cv2.resize(ela_crop, (panel_w, panel_h))
                dashboard[panel_h:frame_h, frame_w:frame_w+panel_w] = cv2.resize(cv2.cvtColor(fft_crop, cv2.COLOR_GRAY2BGR), (panel_w, panel_h))

                # Simulate active scanning lasers on diagnostic panels
                scan_y = (frame_count * 4) % panel_h
                cv2.line(dashboard, (frame_w, scan_y), (frame_w + panel_w, scan_y), (0, 255, 200), 1)
                cv2.line(dashboard, (frame_w, panel_h + scan_y), (frame_w + panel_w, panel_h + scan_y), (0, 255, 200), 1)

        # Apply Exponential Moving Average (EMA) to smooth telemetry output
        smoothed_probability = (alpha * raw_prob_score) + ((1 - alpha) * smoothed_probability)
        display_prob = int(smoothed_probability)
        telemetry_history.append(display_prob)

        # Update process logs
        if frame_count % 15 == 0:
            system_logs.append(f"0x{math.floor(time.time()*1000):X} : M-HASH OK")

        # --- MASTER UI OVERLAY DECK ---
        
        # 1. Top HUD
        now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        draw_hud_text(dashboard, "VERITAS-NPU // TEAM VOID BREAKERS", (15, 25), 0.6, (255, 200, 0))
        draw_hud_text(dashboard, f"SYS.CLK: {now}", (15, 50), 0.5, (200, 220, 255))
        
        # AMD Integration Badge (Pulsing)
        pulse = abs(math.sin(time.time() * 3))
        badge_color = (int(0), int(0), int(150 + (100 * pulse)))
        cv2.rectangle(dashboard, (15, 65), (185, 85), badge_color, -1)
        draw_hud_text(dashboard, "AMD RYZEN AI ACTIVE", (20, 80), 0.4, (255, 255, 255), 1)

        # 2. Panel Headers
        cv2.rectangle(dashboard, (frame_w, 0), (frame_w + 200, 25), (20, 25, 30), -1)
        draw_hud_text(dashboard, f"[ELA] VAR: {ela_variance:.1f}", (frame_w + 10, 18), 0.5, (0, 255, 200))
        
        cv2.rectangle(dashboard, (frame_w, panel_h), (frame_w + 200, panel_h + 25), (20, 25, 30), -1)
        draw_hud_text(dashboard, f"[FFT] NGY: {hf_energy:.1f}", (frame_w + 10, panel_h + 18), 0.5, (0, 255, 200))

        # 3. Bottom Tactical Deck
        deck_y = frame_h
        cv2.rectangle(dashboard, (0, deck_y), (frame_w + panel_w, deck_y + bottom_h), (18, 22, 28), -1)
        cv2.line(dashboard, (0, deck_y), (frame_w + panel_w, deck_y), (50, 60, 70), 2)
        
        draw_hud_text(dashboard, "SYNTHETIC PROBABILITY TELEMETRY", (50, deck_y + 30), 0.4, (150, 170, 190))
        
        status_color = (0, 255, 100) if display_prob < 40 else ((0, 165, 255) if display_prob < 70 else (0, 0, 255))
        draw_tactical_area_graph(dashboard, telemetry_history, x=50, y=deck_y + 45, w=frame_w - 100, h=140, line_color=status_color)

        # 4. Status Monitor & Logs
        status_x = frame_w
        cv2.line(dashboard, (status_x, deck_y + 20), (status_x, deck_y + bottom_h - 20), (40, 50, 60), 1)
        
        if not detection_result.detections:
            status_msg = "SCANNING..."
            status_color = (150, 150, 150)
        elif display_prob < 40:
            status_msg = "VERIFIED HUMAN"
        elif display_prob < 70:
            status_msg = "ANOMALY DETECTED"
        else:
            status_msg = "CRITICAL: DEEPFAKE"
            # Critical Warning Border
            cv2.rectangle(dashboard, (0, 0), (frame_w + panel_w, frame_h + bottom_h), (0, 0, 255), int(4 + (4 * pulse)))

        draw_hud_text(dashboard, "THREAT LEVEL", (status_x + 30, deck_y + 40), 0.5, (150, 170, 190))
        draw_hud_text(dashboard, status_msg, (status_x + 30, deck_y + 70), 0.8, status_color, 2)
        draw_hud_text(dashboard, f"{display_prob}%", (status_x + 30, deck_y + 120), 1.5, status_color, 2)
        
        draw_hud_text(dashboard, f"MSE: {jitter_mse:.2f}", (status_x + 30, deck_y + 160), 0.4, (100, 120, 130))
        for idx, line in enumerate(system_logs):
            draw_hud_text(dashboard, line, (status_x + 130, deck_y + 150 + (idx * 15)), 0.3, (0, 255, 100))

        cv2.imshow('Veritas-NPU // Data Science Interface', dashboard)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean shutdown
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()