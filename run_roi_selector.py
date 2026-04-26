"""
run_roi_selector.py
===================
Captures a live frame from Clash Royale and opens
the ROI selector on it. Saves result to config/roi_config.json

Usage:
    python run_roi_selector.py
"""

import sys
import os
import cv2

# ── path setup so subfolders are importable ───────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from capture.window_capture import WindowCapture, bring_to_front
from roi.roi_selector import run as run_roi_selector

OUTPUT_CONFIG = "config/roi_config.json"


def main():
    print("=" * 50)
    print("  Clash Royale — ROI Selector")
    print("=" * 50)

    # ── Step 1: capture a frame from the game ─────────────────────────────
    print("\n[1/3] Looking for Clash Royale window...")
    cap = WindowCapture()
    if not cap.start():
        print("[ERROR] Could not find Clash Royale window.")
        print("        Make sure the game is open and visible.")
        sys.exit(1)

    print(f"[INFO] Window found: {cap.region}")

    # ── Step 2: grab one clean frame ──────────────────────────────────────
    print("\n[2/3] Capturing reference frame...")
    print("      Switch to the game now — capturing in 3 seconds...")

    import time
    for i in range(3, 0, -1):
        print(f"      {i}...")
        time.sleep(1)

    bring_to_front(cap.region.hwnd)
    time.sleep(0.3)   # let window paint after coming to front

    frame = cap.get_frame()
    cap.stop()

    if frame is None:
        print("[ERROR] Failed to capture frame.")
        sys.exit(1)

    # save reference frame so you can review it later
    os.makedirs("config", exist_ok=True)
    cv2.imwrite("config/reference_frame.png", frame)
    print(f"[INFO] Reference frame saved → config/reference_frame.png")
    print(f"[INFO] Frame size: {frame.shape[1]}x{frame.shape[0]}")

    # ── Step 3: open ROI selector on that frame ────────────────────────────
    print("\n[3/3] Opening ROI selector...")
    print("      Draw your ROI boxes on the game frame.")
    print("      Controls: [SPACE] confirm  [N] next  [S] save  [Q] quit")
    print()

    run_roi_selector(source=frame, output_path=OUTPUT_CONFIG)


if __name__ == "__main__":
    main()