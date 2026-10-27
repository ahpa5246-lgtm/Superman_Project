"""
Spy_Vision.py  —  Superman Eyes  |  Sensor Fusion: Camera + WiFi CSI
======================================================================
FIX NOTE — MediaPipe 0.10+ API change
--------------------------------------
MediaPipe ≥ 0.10 dropped the legacy  `mp.solutions.*`  interface entirely.
The top-level `mediapipe` module now exposes ONLY:
    mediapipe.Image, mediapipe.ImageFormat, mediapipe.tasks

`mp.solutions.pose` → AttributeError: module 'mediapipe' has no attribute 'solutions'

This file uses the correct NEW Tasks API:
    mediapipe.tasks.python.vision.PoseLandmarker  (VIDEO mode, no callbacks)

Required model file (downloaded automatically on first run):
    pose_landmarker_lite.task   (~5.5 MB, placed next to this script)

Dependencies
------------
    pip install mediapipe opencv-python pyserial numpy

Run
---
    python Spy_Vision.py                 # real ESP32 on COM3
    python Spy_Vision.py --demo          # simulated CSI, no hardware
    python Spy_Vision.py --port COM5     # different serial port
"""

import argparse
import math
import pathlib
import sys
import threading
import time
import urllib.request

import cv2
import numpy as np

# ── MediaPipe Tasks API (v0.10+) ─────────────────────────────────────────────
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# ─────────────────────────── constants ───────────────────────────────────────

MODEL_PATH    = pathlib.Path(__file__).parent / "pose_landmarker_lite.task"
MODEL_URL     = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)

# Skeleton connections (MediaPipe Pose 33-point set, landmark indices)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),          # face left
    (0, 4), (4, 5), (5, 6), (6, 8),          # face right
    (9, 10),                                   # mouth
    (11, 12),                                  # shoulders
    (11, 13), (13, 15),                        # left arm
    (12, 14), (14, 16),                        # right arm
    (15, 17), (15, 19), (15, 21),             # left hand
    (16, 18), (16, 20), (16, 22),             # right hand
    (11, 23), (12, 24), (23, 24),             # torso
    (23, 25), (25, 27), (27, 29), (29, 31),   # left leg
    (24, 26), (26, 28), (28, 30), (30, 32),   # right leg
]

NEON_GREEN   = (0,  255,  0)
NEON_LIME    = (50, 255, 80)
NEON_CYAN    = (0,  230, 230)
RED_ALERT    = (0,   0,  255)
ORANGE_WARN  = (0, 140,  255)
DIM_GREEN    = (0,   80,   0)

CSI_ALERT_THRESHOLD = 40.0     # variance level that triggers "behind wall" alert
RADAR_RINGS         = 4        # concentric radar rings to draw

# ─────────────────────── model auto-download ─────────────────────────────────

def ensure_model():
    if MODEL_PATH.exists():
        return
    print(f"[INFO] Downloading pose model (~5.5 MB) → {MODEL_PATH.name} …")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[INFO] Model saved: {MODEL_PATH.stat().st_size // 1024} KB")
    except Exception as e:
        print(f"[ERROR] Could not download model: {e}")
        print(f"        Place  pose_landmarker_lite.task  next to this script.")
        sys.exit(1)

# ─────────────────────── CSI radar thread ────────────────────────────────────

class CSIRadar(threading.Thread):
    """
    Background thread that reads CSI frames from the ESP32 serial port.

    Expected line format:
        CSI_DATA:imag0,real0,imag1,real1,...

    Computes per-frame amplitude variance as a proxy for movement energy
    behind the wall, with exponential smoothing.
    """

    def __init__(self, port: str = "COM3", baud: int = 115200,
                 demo: bool = False):
        super().__init__(daemon=True)
        self.port     = port
        self.baud     = baud
        self.demo     = demo
        self.variance = 0.0
        self.connected = False
        self._t       = 0.0

    def run(self):
        if not self.demo and SERIAL_AVAILABLE:
            try:
                ser = serial.Serial(self.port, self.baud, timeout=1)
                self.connected = True
                print(f"[INFO] CSI radar connected on {self.port}")
                while True:
                    try:
                        line = ser.readline().decode("utf-8", errors="ignore").strip()
                        if "CSI_DATA:" in line:
                            self._process(line)
                    except Exception:
                        pass
            except Exception as e:
                print(f"[WARN] Serial {self.port} unavailable ({e}) — using demo CSI.")
                self.demo = True

        # Demo / fallback
        print("[INFO] CSI running in demo/simulation mode.")
        while True:
            time.sleep(0.05)
            self._t += 0.05
            # simulate breathing motion + occasional spike (person walking past)
            base    = 20 + 10 * math.sin(self._t * 0.4)
            spike   = 80 * max(0, math.sin(self._t * 0.18)) ** 6
            self.variance = base + spike + np.random.uniform(-3, 3)

    def _process(self, line: str):
        try:
            payload = line.split(":")[1].strip()
            raw     = np.fromstring(payload, dtype=np.int32, sep=",")
            if raw.size < 4:
                return
            amps = np.sqrt(raw[0::2].astype(np.float32) ** 2 +
                           raw[1::2].astype(np.float32) ** 2)
            v = float(np.var(amps))
            self.variance = self.variance * 0.75 + v * 0.25
        except Exception:
            pass

# ─────────────────────── drawing helpers ─────────────────────────────────────

def draw_skeleton(frame: np.ndarray, landmarks, h: int, w: int):
    """Draw neon-green skeleton with glowing joints on `frame` (in-place)."""
    pts = []
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        pts.append((x, y))

    # connections — bright neon line + dim wider glow underneath
    for (a, b) in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], DIM_GREEN,   5, cv2.LINE_AA)
            cv2.line(frame, pts[a], pts[b], NEON_GREEN,  2, cv2.LINE_AA)

    # joints
    for (x, y) in pts:
        cv2.circle(frame, (x, y), 6, DIM_GREEN,  -1, cv2.LINE_AA)  # glow
        cv2.circle(frame, (x, y), 3, NEON_LIME,  -1, cv2.LINE_AA)  # core
        cv2.circle(frame, (x, y), 4, NEON_GREEN,  1, cv2.LINE_AA)  # ring


def draw_radar(frame: np.ndarray, cx: int, cy: int,
               variance: float, t: float):
    """
    Draw pulsing concentric radar rings + particle cloud scaled to CSI variance.
    Rings: red/orange glow with animated breathing radius.
    Particles: scattered dots whose density reflects movement magnitude.
    """
    base_r   = int(min(variance * 1.5, 220))
    pulse    = int(15 * math.sin(t * 6))            # breathing animation
    alpha_ov = np.zeros_like(frame, dtype=np.uint8)

    # rings
    for i in range(RADAR_RINGS, 0, -1):
        r       = base_r * i // RADAR_RINGS + pulse
        opacity = max(0, 220 - i * 45)
        color   = (0, max(0, 100 - i * 20), 255)    # dark orange → red
        thick   = max(1, 3 - i)
        cv2.circle(alpha_ov, (cx, cy), max(r, 2), color, thick, cv2.LINE_AA)

    # outer faint sweep line (rotating)
    angle    = (t * 90) % 360
    rad_ang  = math.radians(angle)
    ex       = int(cx + (base_r + 40) * math.cos(rad_ang))
    ey       = int(cy + (base_r + 40) * math.sin(rad_ang))
    cv2.line(alpha_ov, (cx, cy), (ex, ey), (0, 60, 200), 1, cv2.LINE_AA)

    # particle cloud — density ~ variance
    n_particles = int(min(variance / 2, 80))
    rng = np.random.default_rng(int(t * 10) % 9999)
    angles  = rng.uniform(0, 2 * math.pi, n_particles)
    radii   = rng.uniform(20, base_r + 20, n_particles)
    for ang, rad in zip(angles, radii):
        px = int(cx + rad * math.cos(ang))
        py = int(cy + rad * math.sin(ang))
        if 0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]:
            cv2.circle(alpha_ov, (px, py), 2, RED_ALERT, -1, cv2.LINE_AA)

    cv2.addWeighted(alpha_ov, 0.75, frame, 1.0, 0, dst=frame)


def draw_hud(frame: np.ndarray, h: int, w: int,
             variance: float, skeleton_detected: bool,
             connected: bool, t: float):
    """Cinematic spy HUD: corner brackets, status bars, alert text."""

    blink = (int(t * 2) % 2 == 0)     # 2 Hz blink

    # ── corner bracket decorations ────────────────────────────────────────
    br_len, br_thick, br_col = 30, 2, NEON_GREEN
    corners = [(0, 0, 1, 1), (0, h, 1, -1), (w, 0, -1, 1), (w, h, -1, -1)]
    for (cx, cy, sx, sy) in corners:
        cv2.line(frame, (cx, cy), (cx + sx * br_len, cy),            br_col, br_thick)
        cv2.line(frame, (cx, cy), (cx,               cy + sy * br_len), br_col, br_thick)

    # ── top-left status block ─────────────────────────────────────────────
    font  = cv2.FONT_HERSHEY_SIMPLEX
    small = 0.48
    med   = 0.60

    link_color = NEON_GREEN if connected else ORANGE_WARN
    link_text  = "LINK: ESP32 ACTIVE" if connected else "LINK: SIMULATION MODE"

    cv2.putText(frame, "SUPERMAN EYES  v2.0",       (14, 22), font, 0.55, NEON_LIME,  1, cv2.LINE_AA)
    cv2.putText(frame, link_text,                   (14, 42), font, small, link_color,1, cv2.LINE_AA)
    cv2.putText(frame, f"CSI VARIANCE : {variance:>7.1f}", (14, 60), font, small, NEON_GREEN, 1, cv2.LINE_AA)

    # variance bar
    bar_w = int(min(variance / 200.0, 1.0) * 180)
    cv2.rectangle(frame, (14, 66), (194, 74), DIM_GREEN, 1)
    bar_col = RED_ALERT if variance > CSI_ALERT_THRESHOLD else NEON_GREEN
    if bar_w > 0:
        cv2.rectangle(frame, (14, 66), (14 + bar_w, 74), bar_col, -1)

    # ── bottom-left timestamp ─────────────────────────────────────────────
    ts = time.strftime("%H:%M:%S")
    cv2.putText(frame, f"SYS TIME  {ts}", (14, h - 14), font, small, DIM_GREEN, 1, cv2.LINE_AA)

    # ── top-right detection state ─────────────────────────────────────────
    if skeleton_detected:
        state_col  = NEON_GREEN
        state_text = "TARGET: VISUAL LOCK"
    elif variance > CSI_ALERT_THRESHOLD:
        state_col  = RED_ALERT if blink else ORANGE_WARN
        state_text = "TARGET: BEHIND WALL" if blink else "THROUGH-WALL SIGNAL"
    else:
        state_col  = DIM_GREEN
        state_text = "TARGET: NONE"

    tw, _ = cv2.getTextSize(state_text, font, med, 1)[0], 0
    cv2.putText(frame, state_text, (w - tw[0] - 16, 26), font, med, state_col, 1, cv2.LINE_AA)

    # ── centre alert banner (flashes when behind-wall) ────────────────────
    if not skeleton_detected and variance > CSI_ALERT_THRESHOLD and blink:
        banner = ">> MOTION DETECTED BEHIND WALL <<"
        bw, _  = cv2.getTextSize(banner, font, 0.72, 2)[0], 0
        bx     = (w - bw[0]) // 2
        cv2.putText(frame, banner, (bx, 46), font, 0.72, RED_ALERT, 2, cv2.LINE_AA)

    # ── bottom-right CSI mode label ───────────────────────────────────────
    mode_text = "MODE: X-RAY / CSI RADAR"
    mw, _     = cv2.getTextSize(mode_text, font, small, 1)[0], 0
    cv2.putText(frame, mode_text, (w - mw[0] - 14, h - 14),
                font, small, DIM_GREEN, 1, cv2.LINE_AA)

# ─────────────────────────── main loop ───────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Spy_Vision — Superman Eyes")
    parser.add_argument("--port",  default="COM3",   help="Serial port (default: COM3)")
    parser.add_argument("--baud",  type=int, default=115200)
    parser.add_argument("--demo",  action="store_true", help="Simulated CSI (no hardware)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height",type=int, default=720)
    args = parser.parse_args()

    # ── model check ───────────────────────────────────────────────────────
    ensure_model()

    # ── CSI radar thread ──────────────────────────────────────────────────
    radar = CSIRadar(port=args.port, baud=args.baud, demo=args.demo)
    radar.start()

    # ── MediaPipe PoseLandmarker (VIDEO mode — synchronous, no thread) ────
    options = PoseLandmarkerOptions(
        base_options   = BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode   = RunningMode.VIDEO,
        num_poses      = 1,
        min_pose_detection_confidence = 0.5,
        min_pose_presence_confidence  = 0.5,
        min_tracking_confidence       = 0.5,
    )
    landmarker = PoseLandmarker.create_from_options(options)

    # ── camera ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        sys.exit(1)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    CX, CY = W // 2, H // 2

    print(f"[INFO] Camera: {W}×{H}   |   Press Q to quit.")
    print(f"[INFO] MediaPipe Pose Landmarker loaded from: {MODEL_PATH.name}")

    frame_idx = 0
    t_start   = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now_ms = int((time.time() - t_start) * 1000)
        t      = time.time() - t_start

        # ── Night Vision filter: grayscale → neon green ───────────────────
        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # slight CLAHE contrast enhancement for night-vision feel
        clahe     = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced  = clahe.apply(gray)
        blank     = np.zeros_like(enhanced)
        nv_frame  = cv2.merge([blank, enhanced, blank])    # (B=0, G, R=0)

        # ── MediaPipe pose detection ──────────────────────────────────────
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
        result = landmarker.detect_for_video(mp_image, now_ms)

        skeleton_detected = False
        if result.pose_landmarks:
            skeleton_detected = True
            draw_skeleton(nv_frame, result.pose_landmarks[0], H, W)

        # ── CSI radar overlay (only when no skeleton) ──────────────────────
        variance = radar.variance
        if not skeleton_detected and variance > 5:
            draw_radar(nv_frame, CX, CY, variance, t)

        # ── HUD ───────────────────────────────────────────────────────────
        draw_hud(nv_frame, H, W, variance, skeleton_detected, radar.connected, t)

        # ── FPS counter ───────────────────────────────────────────────────
        frame_idx += 1
        elapsed = time.time() - t_start
        fps     = frame_idx / max(elapsed, 0.001)
        cv2.putText(nv_frame, f"FPS {fps:4.1f}",
                    (W - 80, H - 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.44, DIM_GREEN, 1, cv2.LINE_AA)

        cv2.imshow("Superman Eyes — Spy Vision", nv_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    print("[INFO] Spy Vision terminated.")


if __name__ == "__main__":
    main()