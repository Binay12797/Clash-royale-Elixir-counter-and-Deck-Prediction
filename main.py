"""
main.py

Entry point for the Clash Royale Elixir Counter & Deck Predictor.

Wires together:
    WindowCapture       → grabs frames from the game window
    Detector            → runs YOLOv8 on each frame
    CycleTracker        → tracks opponent card cycle
    ElixirEstimator     → estimates opponent elixir
    GameStartTrigger    → SPACE key fires start_game / reset
    OverlayPanel        → PyQt5 HUD displayed left of the game window

Run:
    python main.py
    python main.py --confidence 0.65
    python main.py --debug          ← shows annotated OpenCV window
    python main.py --no-overlay     ← headless, prints state to console

Controls (while running):
    SPACE       — start / restart match clock
    Q           — quit
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import cv2
from PyQt5.QtCore    import QRect, QTimer
from PyQt5.QtWidgets import QApplication

# Project imports
from capture.window_capture         import WindowCapture
from tracker.cycle_tracker          import CycleTracker
from tracker.elixir_estimator       import ElixirEstimator
from tracker.game_start_trigger     import GameStartTrigger
from detection.detector             import Detector
from overlay.panel                  import OverlayPanel


# ---------------------------------------------------------------------------
# Config paths
# ---------------------------------------------------------------------------
ROI_CONFIG_PATH  = "config/roi_config.json"
CARD_DB_PATH     = "data/card_database.json"
MODEL_PATH       = "models/clash_royale_yolov8s.pt"
GAME_WINDOW_NAME = "Clash Royale"   # title of the Google Play Games window

# Main loop target — YOLO is slow, 15 fps is plenty
TARGET_FPS       = 15
FRAME_INTERVAL   = 1.0 / TARGET_FPS

# How often the overlay refreshes (ms) — independent of capture rate
OVERLAY_REFRESH_MS = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_roi_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"[main] WARNING: roi_config not found at '{path}' — using defaults.")
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_game_window_rect(capture: WindowCapture) -> QRect:
    """Convert WindowCapture's window rect to a QRect for the overlay."""
    try:
        r = capture.get_window_rect()   # expects (x, y, w, h)
        return QRect(r[0], r[1], r[2], r[3])
    except Exception:
        return QRect(0, 0, 800, 600)    # safe fallback


# ---------------------------------------------------------------------------
# Main application class
# ---------------------------------------------------------------------------
class App:
    def __init__(self, args: argparse.Namespace):
        self.args      = args
        self.running   = False

        print("[main] ── Initialising ──────────────────────────────────────")

        # ── Config ────────────────────────────────────────────────────
        self.roi_config = load_roi_config(ROI_CONFIG_PATH)

        # ── Capture ───────────────────────────────────────────────────
        print(f"[main] Attaching to window: '{GAME_WINDOW_NAME}'")
        self.capture = WindowCapture(GAME_WINDOW_NAME)

        # ── Tracker / Estimator ───────────────────────────────────────
        self.tracker   = CycleTracker()
        self.estimator = ElixirEstimator(card_db_path=CARD_DB_PATH)

        # ── Detector ──────────────────────────────────────────────────
        print(f"[main] Loading YOLO model: {MODEL_PATH}")
        self.detector = Detector(
            model_path = MODEL_PATH,
            roi_config = self.roi_config,
            confidence = args.confidence,
        )

        # ── Game-start trigger (SPACE) ────────────────────────────────
        self.trigger = GameStartTrigger()
        self.trigger.register(self.estimator.start_game)
        self.trigger.register(self.tracker.reset)
        self.trigger.register(self._on_game_start)
        self.trigger.start()

        # ── PyQt5 overlay ─────────────────────────────────────────────
        self.qt_app = QApplication.instance() or QApplication(sys.argv)

        if not args.no_overlay:
            game_rect = get_game_window_rect(self.capture)
            self.overlay = OverlayPanel(game_window_rect=game_rect,
                                        refresh_ms=OVERLAY_REFRESH_MS)
            self.overlay.show_waiting()
            self.overlay.show()
        else:
            self.overlay = None

        # ── Main loop timer (drives capture + detection) ───────────────
        self._loop_timer = QTimer()
        self._loop_timer.timeout.connect(self._loop_tick)
        self._loop_timer.start(int(FRAME_INTERVAL * 1000))

        print("[main] ── Ready ─────────────────────────────────────────────")
        print("[main] Press SPACE when 'FIGHT!' appears to start tracking.")
        print("[main] Press Q (in debug window) or Ctrl-C to quit.\n")

    # ------------------------------------------------------------------
    # Game start callback
    # ------------------------------------------------------------------

    def _on_game_start(self) -> None:
        """Called by GameStartTrigger on SPACE press."""
        print("[main] *** Match started — tracking active ***")
        if self.overlay:
            # Re-anchor overlay in case user moved/resized the game window
            game_rect = get_game_window_rect(self.capture)
            self.overlay.set_game_window(game_rect)

    # ------------------------------------------------------------------
    # Main loop tick  (runs every ~67ms at 15 fps)
    # ------------------------------------------------------------------

    def _loop_tick(self) -> None:
        t0 = time.monotonic()

        # ── Grab frame ────────────────────────────────────────────────
        frame = self.capture.get_frame()
        if frame is None:
            return

        # ── YOLO detection ────────────────────────────────────────────
        detections = self.detector.detect(frame)

        # ── Feed detections to tracker / estimator ────────────────────
        if self.trigger.game_active:
            collector_count = 0

            for d in detections:
                if d.belong == "enemy":
                    self.tracker.card_played(d.card_name, d.timestamp)
                    self.estimator.card_played(d.card_name, d.timestamp)

                    # Count elixir collectors on field for bonus regen
                    if "elixir-collector" in d.card_name:
                        collector_count += 1

            self.estimator.set_collector_count(collector_count)

        # ── Pull state ────────────────────────────────────────────────
        cycle_state  = self.tracker.get_state()
        elixir_state = self.estimator.get_state()

        # ── Push to overlay ───────────────────────────────────────────
        if self.overlay:
            if self.trigger.game_active:
                self.overlay.update_state(cycle_state, elixir_state)
            else:
                self.overlay.show_waiting()

        # ── Debug window (--debug flag) ───────────────────────────────
        if self.args.debug:
            annotated = self.detector.draw(frame, detections)
            self._overlay_debug_text(annotated, elixir_state)
            cv2.imshow("Royal Scout — Debug", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.quit()
                return

        # ── Console output (--no-overlay) ────────────────────────────
        if self.args.no_overlay and self.trigger.game_active:
            self._print_state(cycle_state, elixir_state)

        # ── Frame-rate throttle ───────────────────────────────────────
        elapsed = time.monotonic() - t0
        if elapsed > FRAME_INTERVAL * 1.5:
            print(f"[main] WARNING: frame took {elapsed*1000:.0f}ms "
                  f"(target {FRAME_INTERVAL*1000:.0f}ms) — consider lowering --confidence")

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def _overlay_debug_text(self, frame, elixir_state) -> None:
        """Burn elixir value and game phase onto the debug frame."""
        if elixir_state is None:
            return
        phase = ("TRIPLE" if elixir_state.is_triple_elixir
                 else "DOUBLE" if elixir_state.is_double_elixir
                 else "NORMAL")
        mins = int(elixir_state.game_time) // 60
        secs = int(elixir_state.game_time) % 60
        text = (f"Elixir ~{elixir_state.estimated_elixir:.1f}  "
                f"{phase}  {mins}:{secs:02d}")
        cv2.putText(frame, text, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (177, 74, 237), 2, cv2.LINE_AA)

    def _print_state(self, cycle_state, elixir_state) -> None:
        if elixir_state:
            mins = int(elixir_state.game_time) // 60
            secs = int(elixir_state.game_time) % 60
            phase = ("TRIPLE" if elixir_state.is_triple_elixir
                     else "DOUBLE" if elixir_state.is_double_elixir
                     else "normal")
            print(f"  [{mins}:{secs:02d}] elixir ~{elixir_state.estimated_elixir:.1f}  "
                  f"({phase})", end="")
        if cycle_state:
            seen = len(cycle_state.seen_cards)
            hand = list(cycle_state.current_hand)
            print(f"  |  deck {seen}/8  hand {hand}", end="")
        print()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.running = True
        sys.exit(self.qt_app.exec_())

    def quit(self) -> None:
        print("\n[main] Shutting down…")
        self.trigger.stop()
        self._loop_timer.stop()
        cv2.destroyAllWindows()
        self.qt_app.quit()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clash Royale Elixir Counter & Deck Predictor"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.60,
        help="YOLO detection confidence threshold (default 0.60)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Show annotated OpenCV window with bounding boxes"
    )
    parser.add_argument(
        "--no-overlay", action="store_true",
        help="Disable PyQt5 overlay — print state to console instead"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    # Graceful Ctrl-C
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = App(args)
    app.run()