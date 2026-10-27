"""
Matrix_radar.py — ESP32 CSI Live Radar Visualizer
====================================================
Project : ESP32-CSI-Tool  (StevenMHernandez)
Author  : Antigravity AI

Description
-----------
Reads continuous CSI frames from a serial port (default COM3 @ 115200 baud),
decodes the raw IQ pairs into per-subcarrier AMPLITUDE values, and renders
them in real-time as a scrolling WATERFALL / HEATMAP — styled to look like
a cinematic "Matrix" radar screen:

  • Pure black background
  • Neon-green colour map (custom linear gradient: black → bright green)
  • Scanline sweep animation overlay
  • Phase-unwrap / linear-trend removal to cancel static-path distortion
  • Live statistics HUD (packet rate, RSSI, active subcarriers)

Serial data format expected
---------------------------
Every line that contains the token  CSI_DATA  follows the ESP32-CSI-Tool
CSV layout (26 header fields + bracketed IQ array):

  CSI_DATA,<role>,<mac>,<rssi>,<rate>,<sig_mode>,<mcs>,<bw>,<smooth>,
  <not_snd>,<aggr>,<stbc>,<fec>,<sgi>,<noise_floor>,<ampdu_cnt>,
  <channel>,<sec_ch>,<local_ts>,<ant>,<sig_len>,<rx_state>,
  <real_time_set>,<real_ts>,<len>,[<IQ bytes …>]

The bracketed array holds alternating  imaginary / real  int8 values.
Amplitude = sqrt(imag² + real²)

Usage
-----
  python Matrix_radar.py                  # default COM3, 115200
  python Matrix_radar.py --port COM5
  python Matrix_radar.py --port COM3 --baud 921600
  python Matrix_radar.py --demo           # run with simulated data (no hardware)

Dependencies
------------
  pip install pyserial matplotlib numpy
"""

import argparse
import collections
import math
import re
import sys
import threading
import time

import matplotlib
matplotlib.use("TkAgg")          # works on Windows; change to Qt5Agg if needed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import numpy as np

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[WARN] pyserial not found – forcing --demo mode.")

# ─────────────────────────── constants ───────────────────────────────────────

WATERFALL_ROWS   = 200          # number of time-steps shown vertically
MIN_SUBCARRIERS  = 1            # guard against empty frames
MAX_SUBCARRIERS  = 128          # typical ESP32 802.11n LLTF + HT-LTF
AMPLITUDE_CLAMP  = 80.0         # clip value for colour normalisation
REFRESH_INTERVAL = 80           # ms between animation frames

# ─────────────────────── custom neon-green colormap ──────────────────────────

_NEON_COLORS = [
    (0.00, "#000000"),   # black  – no signal
    (0.15, "#003300"),   # very dark green
    (0.35, "#006600"),   # dark green
    (0.55, "#00aa00"),   # medium green
    (0.75, "#00dd00"),   # bright green
    (0.90, "#39ff14"),   # neon green (electric)
    (1.00, "#ccffcc"),   # near-white green – peak
]
_NEON_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "matrix_neon",
    [(v, c) for v, c in _NEON_COLORS]
)

# ─────────────────────────── CSI parsing ─────────────────────────────────────

_BRACKET_RE = re.compile(r"\[([^\]]+)\]")

def parse_csi_line(line: str):
    """
    Parse a line in the custom dashboard format:

        CSI_DATA:imag1,real1,imag2,real2,...

    The token before the colon must contain 'CSI_DATA'.
    Everything after the colon is a flat, comma-separated list of integers
    interleaved as  [imag0, real0, imag1, real1, …].

    Returns
    -------
    (amplitudes: np.ndarray, rssi: int | None)  or  (None, None) on failure.

    Distortion correction
    ---------------------
    After computing amplitudes we apply a simple linear-trend removal
    (detrending) across sub-carriers so that the static multipath baseline
    is subtracted, leaving only the *dynamic* variation visible —
    effectively reversing slow channel distortion.
    """
    if "CSI_DATA" not in line:
        return None, None

    # ── split at the colon to isolate the IQ payload ──────────────────────
    colon_idx = line.find(":")
    if colon_idx == -1:
        return None, None

    payload = line[colon_idx + 1:].strip()

    # ── parse comma-separated integers directly into a NumPy array ────────
    try:
        raw = np.fromstring(payload, dtype=np.int32, sep=",")
    except Exception:
        return None, None

    if raw.size < 2:
        return None, None

    # ── split interleaved IQ pairs with binary stride ──────────────────────
    imag = raw[0::2].astype(np.float32)   # imaginary components
    real = raw[1::2].astype(np.float32)   # real components
    n    = min(len(imag), len(real))
    if n == 0:
        return None, None

    # ── amplitude = √(I² + R²) — vectorised ───────────────────────────────
    amps = np.sqrt(imag[:n] ** 2 + real[:n] ** 2)

    # ── linear-trend removal (detrend) to cancel static distortion ────────
    if n >= 4:
        x      = np.arange(n, dtype=np.float32)
        coeffs = np.polyfit(x, amps, 1)
        trend  = np.polyval(coeffs, x)
        amps   = amps - trend + trend.mean()   # subtract slope, keep DC

    # clip negatives introduced by detrending
    np.clip(amps, 0, None, out=amps)
    return amps, None          # custom format carries no RSSI field

# ─────────────────────── simulated demo data ─────────────────────────────────

# ─────────────────────── simulated demo data ─────────────────────────────────

class DemoSource:
    """Generates synthetic CSI data matching the updated custom format."""

    def __init__(self):
        self._t = 0.0

    def readline(self) -> str:
        time.sleep(0.04)          # ~25 Hz
        self._t += 0.04
        n = 64                    # 64 subcarriers × 2 = 128 IQ components
        imag = []
        real = []
        for k in range(n):
            base   = 20 * math.sin(k * 0.3 + 0.5)
            breath = 8  * math.sin(2 * math.pi * 0.3 * self._t + k * 0.15)
            noise  = np.random.uniform(-3, 3)
            amp    = base + breath + noise + 25
            phase  = math.sin(self._t * 0.7 + k * 0.2)
            imag.append(int(amp * math.sin(phase)))
            real.append(int(amp * math.cos(phase)))

        pairs = []
        for i, r in zip(imag, real):
            pairs += [i, r]

        # ⬇️ التعديل الجديد (بسيط ومباشر) ⬇️
        payload_str = ",".join(str(v) for v in pairs)
        return f"CSI_DATA:{payload_str}"

# ─────────────────────────── serial reader thread ────────────────────────────

class SerialReader(threading.Thread):
    """
    Background thread that continuously reads from the serial port and
    pushes decoded CSI amplitude vectors into a thread-safe deque.
    """

    def __init__(self, port: str, baud: int,
                 waterfall_buf: np.ndarray,
                 lock: threading.Lock,
                 stats: dict,
                 demo: bool = False):
        super().__init__(daemon=True)
        self.port         = port
        self.baud         = baud
        self.waterfall    = waterfall_buf
        self.lock         = lock
        self.stats        = stats
        self.demo         = demo
        self._running     = True
        self._pkt_count   = 0
        self._last_rate_t = time.time()

    def run(self):
        if self.demo:
            src = DemoSource()
            while self._running:
                line = src.readline()
                self._process(line)
        else:
            if not SERIAL_AVAILABLE:
                print("[ERROR] pyserial missing. Run with --demo flag.")
                sys.exit(1)
            try:
                ser = serial.Serial(self.port, self.baud, timeout=1)
            except serial.SerialException as e:
                print(f"[ERROR] Cannot open {self.port}: {e}")
                sys.exit(1)

            print(f"[INFO]  Serial port {self.port} opened at {self.baud} baud.")
            # flush initial boot/debug messages until first CSI line
            print("[INFO]  Waiting for first CSI_DATA frame …")
            while self._running:
                try:
                    raw = ser.readline()
                    line = raw.decode("utf-8", errors="replace").strip()
                except Exception:
                    continue
                if "CSI_DATA" in line:
                    self._process(line)
                    break

            print("[INFO]  CSI stream acquired. Rendering …")
            while self._running:
                try:
                    raw  = ser.readline()
                    line = raw.decode("utf-8", errors="replace").strip()
                    if "CSI_DATA" in line:
                        self._process(line)
                except Exception:
                    pass

    def _process(self, line: str):
        amps, rssi = parse_csi_line(line)
        if amps is None:
            return

        # normalise length to MAX_SUBCARRIERS for consistent display
        n = len(amps)
        if n > MAX_SUBCARRIERS:
            amps = amps[:MAX_SUBCARRIERS]
            n    = MAX_SUBCARRIERS
        elif n < MAX_SUBCARRIERS:
            amps = np.pad(amps, (0, MAX_SUBCARRIERS - n))

        with self.lock:
            # roll waterfall up and insert new row at bottom
            self.waterfall[:-1, :] = self.waterfall[1:, :]
            self.waterfall[-1,  :] = amps

            self._pkt_count += 1
            now = time.time()
            dt  = now - self._last_rate_t
            if dt >= 1.0:
                self.stats["rate"]    = self._pkt_count / dt
                self.stats["rssi"]    = rssi if rssi is not None else self.stats.get("rssi", 0)
                self.stats["n_sub"]   = int(np.sum(amps > 0.5))
                self.stats["total"]  += self._pkt_count
                self._pkt_count       = 0
                self._last_rate_t     = now

    def stop(self):
        self._running = False

# ─────────────────────────── main visualiser ─────────────────────────────────

class MatrixRadar:
    """
    Sets up the Matplotlib figure and drives the animation.

    Layout
    ------
    ┌──────────────────────────────────────────────────────┐
    │  ██ MATRIX RADAR — ESP32 CSI Live Monitor ██   HUD  │
    │                                                      │
    │   [WATERFALL HEATMAP — time scrolls downward]       │
    │                                                      │
    │   [AMPLITUDE LINE PLOT — current frame]             │
    └──────────────────────────────────────────────────────┘
    """

    def __init__(self, port: str, baud: int, demo: bool):
        self.lock       = threading.Lock()
        self.waterfall  = np.zeros((WATERFALL_ROWS, MAX_SUBCARRIERS),
                                   dtype=np.float32)
        self.stats      = {"rate": 0.0, "rssi": 0, "n_sub": 0, "total": 0}
        self._scan_y    = 0      # scanline position

        # ── reader thread ─────────────────────────────────────────────────
        self.reader = SerialReader(port, baud, self.waterfall,
                                   self.lock, self.stats, demo=demo)
        self.reader.start()

        # ── figure ────────────────────────────────────────────────────────
        plt.style.use("dark_background")
        self.fig = plt.figure(
            figsize=(14, 8),
            facecolor="#000000",
            num="Matrix Radar — ESP32 CSI"
        )
        self.fig.patch.set_facecolor("#000000")

        # grid: waterfall (top, 65%) + amplitude line (bottom, 35%)
        gs = self.fig.add_gridspec(
            2, 1,
            height_ratios=[2.2, 1],
            hspace=0.08,
            left=0.06, right=0.98, top=0.91, bottom=0.07
        )

        self.ax_wf   = self.fig.add_subplot(gs[0])   # waterfall
        self.ax_line = self.fig.add_subplot(gs[1])   # amplitude line

        self._style_axes()
        self._build_waterfall_plot()
        self._build_line_plot()
        self._build_hud()
        self._build_scanline()

    # ── styling ───────────────────────────────────────────────────────────

    def _style_axes(self):
        BLACK  = "#000000"
        GREEN  = "#00cc00"
        DGREEN = "#004400"

        for ax in (self.ax_wf, self.ax_line):
            ax.set_facecolor(BLACK)
            for spine in ax.spines.values():
                spine.set_edgecolor(DGREEN)
                spine.set_linewidth(0.8)
            ax.tick_params(colors=GREEN, labelsize=8)
            ax.xaxis.label.set_color(GREEN)
            ax.yaxis.label.set_color(GREEN)

        self.ax_wf.set_title(
            "▸  CHANNEL IMPULSE RESPONSE  ◂   [time ↓  ·  subcarrier →]",
            color="#39ff14", fontsize=10, fontfamily="monospace", pad=6
        )
        self.ax_wf.set_ylabel("Time  (frames)", color=GREEN, fontsize=8)
        self.ax_wf.set_xlabel("Subcarrier index", color=GREEN, fontsize=8)
        self.ax_wf.set_yticks(
            np.linspace(0, WATERFALL_ROWS, 6, dtype=int),
            labels=[f"-{int(v)}" for v in np.linspace(WATERFALL_ROWS, 0, 6, dtype=int)],
            fontsize=7
        )

        self.ax_line.set_ylabel("Amplitude", color=GREEN, fontsize=8)
        self.ax_line.set_xlabel("Subcarrier index", color=GREEN, fontsize=8)
        self.ax_line.set_xlim(0, MAX_SUBCARRIERS - 1)
        self.ax_line.set_ylim(0, AMPLITUDE_CLAMP)
        self.ax_line.grid(color=DGREEN, linestyle="--", linewidth=0.4, alpha=0.6)

    # ── waterfall image ───────────────────────────────────────────────────

    def _build_waterfall_plot(self):
        self.im = self.ax_wf.imshow(
            self.waterfall,
            aspect="auto",
            cmap=_NEON_CMAP,
            vmin=0,
            vmax=AMPLITUDE_CLAMP,
            interpolation="nearest",
            origin="upper",
            extent=[0, MAX_SUBCARRIERS, WATERFALL_ROWS, 0],
        )
        cbar = self.fig.colorbar(self.im, ax=self.ax_wf, pad=0.01, fraction=0.015)
        cbar.ax.yaxis.set_tick_params(color="#39ff14", labelsize=7)
        cbar.outline.set_edgecolor("#004400")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#39ff14")

    # ── amplitude line plot ───────────────────────────────────────────────

    def _build_line_plot(self):
        x = np.arange(MAX_SUBCARRIERS)
        self.line_fill = self.ax_line.fill_between(
            x, np.zeros(MAX_SUBCARRIERS), np.zeros(MAX_SUBCARRIERS),
            color="#003300", alpha=0.5
        )
        self.line_plot, = self.ax_line.plot(
            x, np.zeros(MAX_SUBCARRIERS),
            color="#39ff14", linewidth=1.2, alpha=0.92
        )
        self.ax_line.set_title(
            "▸  CURRENT FRAME — AMPLITUDE SPECTRUM  ◂",
            color="#00cc00", fontsize=9, fontfamily="monospace", pad=4
        )

    # ── HUD text ──────────────────────────────────────────────────────────

    def _build_hud(self):
        hud_style = dict(
            transform=self.fig.transFigure,
            fontfamily="monospace",
            fontsize=9,
            color="#39ff14",
            alpha=0.9,
        )
        self.hud_title = self.fig.text(
            0.5, 0.965,
            "█  MATRIX RADAR  ·  ESP32-CSI-Tool  ·  LIVE CHANNEL STATE INFORMATION  █",
            ha="center", va="top",
            fontsize=11, fontfamily="monospace", color="#39ff14",
            fontweight="bold", transform=self.fig.transFigure
        )
        self.hud_stats = self.fig.text(
            0.985, 0.965,
            "",
            ha="right", va="top",
            **hud_style
        )

    # ── scanline overlay ──────────────────────────────────────────────────

    def _build_scanline(self):
        self.scan_line = self.ax_wf.axhline(
            y=0, color="#39ff14", linewidth=1.0, alpha=0.55, linestyle="-"
        )

    # ── animation update ──────────────────────────────────────────────────

    def update(self, _frame):
        with self.lock:
            wf_snap = self.waterfall.copy()

        # --- waterfall ---
        self.im.set_data(wf_snap)

        # --- current-frame line ---
        last_row = wf_snap[-1, :]
        self.line_plot.set_ydata(last_row)

        # rebuild fill (must remove old collection)
        self.line_fill.remove()
        x = np.arange(MAX_SUBCARRIERS)
        self.line_fill = self.ax_line.fill_between(
            x, 0, last_row, color="#003300", alpha=0.45
        )

        # --- scanline sweep ---
        self._scan_y = (self._scan_y + 2) % WATERFALL_ROWS
        self.scan_line.set_ydata([self._scan_y, self._scan_y])

        # --- HUD ---
        r     = self.stats.get("rate",  0.0)
        rssi  = self.stats.get("rssi",  0)
        n_sub = self.stats.get("n_sub", 0)
        total = self.stats.get("total", 0)
        self.hud_stats.set_text(
            f"PKT/s: {r:5.1f}  │  RSSI: {rssi:4d} dBm  │  "
            f"SUB: {n_sub:3d}  │  TOTAL: {total:6d}"
        )

        return (self.im, self.line_plot, self.scan_line,
                self.line_fill, self.hud_stats)

    # ── entry point ───────────────────────────────────────────────────────

    def run(self):
        self.ani = animation.FuncAnimation(
            self.fig,
            self.update,
            interval=REFRESH_INTERVAL,
            blit=False,       # blit=True breaks fill_between dynamic update
            cache_frame_data=False,
        )
        plt.show()
        self.reader.stop()


# ─────────────────────────── CLI ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Matrix Radar — ESP32 CSI live waterfall visualiser"
    )
    parser.add_argument(
        "--port", default="COM3",
        help="Serial port (default: COM3)"
    )
    parser.add_argument(
        "--baud", type=int, default=115200,
        help="Baud rate (default: 115200)"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run with simulated data — no hardware required"
    )
    args = parser.parse_args()

    if not args.demo and not SERIAL_AVAILABLE:
        print("[WARN] pyserial not installed.  Switching to --demo mode.")
        args.demo = True

    radar = MatrixRadar(port=args.port, baud=args.baud, demo=args.demo)
    radar.run()


if __name__ == "__main__":
    main()
