"""
roi_selector.py
===============
Draw and save ROI coordinates for:
  1. game_timer         - single box
  2. opponent_half      - single box
  3. our_half           - single box
  4. card_deploy_zones  - multiple boxes

Controls:
  [SPACE] Confirm ROI   [R] Redraw    [U] Undo last
  [N] Next ROI          [S] Save & exit   [Q] Quit
"""

import cv2
import json
import sys
import os
import numpy as np
from pathlib import Path
from datetime import datetime

# ── ROI definitions ───────────────────────────────────────────────────────────
ROI_SEQUENCE = [
    {
        "key":   "game_timer",
        "label": "GAME TIMER",
        "color": (0, 255, 255),        # cyan
        "hint":  "Draw box around the game countdown timer",
        "multi": False,
    },
    {
        "key":   "opponent_half",
        "label": "OPPONENT'S HALF",
        "color": (0, 0, 255),          # red
        "hint":  "Draw box covering opponent's side of the arena",
        "multi": False,
    },
    {
        "key":   "our_half",
        "label": "OUR HALF",
        "color": (255, 0, 255),        # magenta
        "hint":  "Draw box covering our side of the arena",
        "multi": False,
    },
    {
        "key":   "card_deploy_zones",
        "label": "CARD DEPLOY ZONE",
        "color": (0, 255, 0),          # green
        "hint":  "Draw each card slot in hand, SPACE after each, N when done",
        "multi": True,
    },
]

# ── Globals shared with mouse callback ────────────────────────────────────────
drawing         = False
start_pt        = (-1, -1)
current_pt      = (-1, -1)
confirmed_rects = []
temp_rect       = None


def mouse_callback(event, x, y, flags, param):
    global drawing, start_pt, current_pt, temp_rect

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing   = True
        start_pt  = (x, y)
        temp_rect = None

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_pt = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing    = False
        current_pt = (x, y)
        x1 = min(start_pt[0], x)
        y1 = min(start_pt[1], y)
        x2 = max(start_pt[0], x)
        y2 = max(start_pt[1], y)
        if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
            temp_rect = (x1, y1, x2, y2)


def draw_overlay(frame, roi_def, confirmed_rects, temp_rect, roi_index, total_rois):
    canvas = frame.copy()
    color  = roi_def["color"]

    for i, (x1, y1, x2, y2) in enumerate(confirmed_rects):
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = roi_def["label"] + (f" #{i+1}" if roi_def["multi"] else "")
        _draw_label(canvas, label, x1, y1, color)

    if drawing and start_pt != (-1, -1):
        x1 = min(start_pt[0], current_pt[0])
        y1 = min(start_pt[1], current_pt[1])
        x2 = max(start_pt[0], current_pt[0])
        y2 = max(start_pt[1], current_pt[1])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)

    elif temp_rect:
        x1, y1, x2, y2 = temp_rect
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        _draw_label(canvas, "SPACE to confirm  |  R to redraw",
                    x1, y1, color, bg=(40, 40, 40))

    _draw_hud(canvas, roi_def, roi_index, total_rois, len(confirmed_rects))
    return canvas


def _draw_label(img, text, x, y, color, bg=None):
    font      = cv2.FONT_HERSHEY_SIMPLEX
    scale     = 0.55
    thickness = 1
    (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
    tx = max(x, 2)
    ty = max(y - 6, th + 4)
    if bg:
        cv2.rectangle(img, (tx - 2, ty - th - 2),
                      (tx + tw + 2, ty + bl), bg, -1)
    cv2.putText(img, text, (tx, ty), font, scale, color, thickness, cv2.LINE_AA)


def _draw_hud(img, roi_def, roi_index, total_rois, n_confirmed):
    h, w  = img.shape[:2]
    bar_h = 70
    cv2.rectangle(img, (0, h - bar_h), (w, h), (20, 20, 20), -1)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    color = roi_def["color"]

    cv2.putText(img,
                f"Step {roi_index+1}/{total_rois}  |  {roi_def['label']}",
                (10, h - bar_h + 22), font, 0.6, color, 1, cv2.LINE_AA)

    cv2.putText(img, roi_def["hint"],
                (10, h - bar_h + 44), font, 0.48,
                (180, 180, 180), 1, cv2.LINE_AA)

    controls = "[SPACE] Confirm  [R] Redraw  [U] Undo  [N] Next  [S] Save & Exit  [Q] Quit"
    if roi_def["multi"]:
        controls = f"({n_confirmed} confirmed)  " + controls
    cv2.putText(img, controls,
                (10, h - bar_h + 62), font, 0.40,
                (130, 130, 130), 1, cv2.LINE_AA)


def rect_to_dict(x1, y1, x2, y2, img_w, img_h):
    return {
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "w":  x2 - x1, "h": y2 - y1,
        "cx": (x1 + x2) // 2, "cy": (y1 + y2) // 2,
        "rel": {
            "x1": round(x1 / img_w, 4),
            "y1": round(y1 / img_h, 4),
            "x2": round(x2 / img_w, 4),
            "y2": round(y2 / img_h, 4),
        }
    }


# ── Main entry — accepts numpy frame, file path, or webcam index ──────────────
def run(source=0, output_path="roi_config.json"):
    global drawing, start_pt, current_pt, confirmed_rects, temp_rect

    if isinstance(source, np.ndarray):
        frame = source

    elif isinstance(source, str) and os.path.isfile(source):
        frame = cv2.imread(source)
        if frame is None:
            print(f"[ERROR] Cannot read image: {source}")
            sys.exit(1)

    elif isinstance(source, int):
        cap = cv2.VideoCapture(source)
        print("[INFO] Press any key to capture reference frame from webcam...")
        while True:
            ret, frame = cap.read()
            cv2.imshow("Capture", frame)
            if cv2.waitKey(1) != -1:
                break
        cap.release()
        cv2.destroyAllWindows()

    else:
        print(f"[ERROR] Invalid source: {source}")
        sys.exit(1)

    img_h, img_w = frame.shape[:2]
    print(f"[INFO] Frame size: {img_w}x{img_h}")

    WIN = "ROI Selector"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, min(img_w, 1280), min(img_h + 70, 800))
    cv2.setMouseCallback(WIN, mouse_callback)

    saved_rois = {}
    roi_idx    = 0

    while roi_idx < len(ROI_SEQUENCE):
        roi_def = ROI_SEQUENCE[roi_idx]

        drawing         = False
        start_pt        = (-1, -1)
        current_pt      = (-1, -1)
        temp_rect       = None
        confirmed_rects = []

        print(f"\n── {roi_def['label']} ──")
        print(f"   {roi_def['hint']}")

        while True:
            canvas = draw_overlay(frame, roi_def, confirmed_rects,
                                  temp_rect, roi_idx, len(ROI_SEQUENCE))
            cv2.imshow(WIN, canvas)
            key = cv2.waitKey(16) & 0xFF

            if key == ord(' '):
                if temp_rect:
                    confirmed_rects.append(temp_rect)
                    print(f"   ✓ Confirmed: {temp_rect}")
                    temp_rect = None
                    if not roi_def["multi"]:
                        break
                else:
                    print("   [WARN] Nothing to confirm — draw a box first.")

            elif key == ord('r'):
                temp_rect = None
                print("   [INFO] Cleared — draw again.")

            elif key == ord('u'):
                if confirmed_rects:
                    print(f"   [INFO] Undone: {confirmed_rects.pop()}")
                elif temp_rect:
                    temp_rect = None

            elif key == ord('n'):
                if roi_def["multi"]:
                    if confirmed_rects:
                        break
                    else:
                        print("   [WARN] Confirm at least one zone first.")
                else:
                    if confirmed_rects:
                        break
                    elif temp_rect:
                        confirmed_rects.append(temp_rect)
                        temp_rect = None
                        break
                    else:
                        print("   [WARN] Draw and confirm a box first.")

            elif key == ord('s'):
                if temp_rect:
                    confirmed_rects.append(temp_rect)
                if confirmed_rects:
                    _store_roi(saved_rois, roi_def, confirmed_rects, img_w, img_h)
                _save_and_exit(saved_rois, output_path, img_w, img_h)
                cv2.destroyAllWindows()
                return

            elif key == ord('q'):
                print("[INFO] Quit without saving.")
                cv2.destroyAllWindows()
                return

        _store_roi(saved_rois, roi_def, confirmed_rects, img_w, img_h)
        roi_idx += 1

    cv2.destroyAllWindows()
    _save_and_exit(saved_rois, output_path, img_w, img_h)


def _store_roi(saved, roi_def, rects, img_w, img_h):
    if roi_def["multi"]:
        saved[roi_def["key"]] = [rect_to_dict(*r, img_w, img_h) for r in rects]
    else:
        saved[roi_def["key"]] = rect_to_dict(*rects[0], img_w, img_h)
    print(f"   → Stored '{roi_def['key']}'")


def _save_and_exit(saved, path, img_w, img_h):
    data = {
        "meta": {
            "created_at":   datetime.now().isoformat(timespec="seconds"),
            "frame_width":  img_w,
            "frame_height": img_h,
        },
        "rois": saved,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2))
    print(f"\n✅  Saved {len(saved)} ROI(s) → {path}")


# ── Standalone usage ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", default="0")
    parser.add_argument("-o", "--output", default="roi_config.json")
    args = parser.parse_args()
    src  = int(args.source) if args.source.isdigit() else args.source
    run(source=src, output_path=args.output)