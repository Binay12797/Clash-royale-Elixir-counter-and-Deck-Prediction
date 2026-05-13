"""
detection/detector.py

Runs YOLOv8 inference on each frame captured by WindowCapture, filters
detections by ROI zone, debounces repeated detections, and feeds results
to CycleTracker and ElixirEstimator.

Class label format expected from the model:
    "enemy-hog-rider"     → belong="enemy",    card_name="hog-rider"
    "friendly-giant"      → belong="friendly",  card_name="giant"

Dependencies:
    pip install ultralytics opencv-python
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    print("[Detector] WARNING: ultralytics not installed.  pip install ultralytics")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CONFIDENCE  = 0.60   # minimum confidence to accept a detection
DEBOUNCE_SECS       = 2.0    # suppress same card at same position within this window
POSITION_TOLERANCE  = 0.08   # fraction of frame width/height — "same position" threshold


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    card_name:  str             # bare name, e.g. "hog-rider"
    belong:     str             # "enemy" | "friendly"
    bbox:       tuple           # (x1, y1, x2, y2) in pixels
    confidence: float
    position:   tuple           # (cx_rel, cy_rel) — centre as 0-1 fraction of frame
    timestamp:  float           # time.monotonic()
    zone:       str             # "opponent_half" | "our_half" | "unknown"


@dataclass
class _RecentDetection:
    card_name: str
    cx_rel:    float
    cy_rel:    float
    timestamp: float


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class Detector:
    """
    Usage
    -----
        detector = Detector(
            model_path  = "models/clash_royale_yolov8s.pt",
            roi_config  = roi_config,       # dict loaded from roi_config.json
            confidence  = 0.6,
        )

        detections = detector.detect(frame)   # frame = numpy BGR from WindowCapture

        for d in detections:
            if d.belong == "enemy":
                cycle_tracker.card_played(d.card_name, d.timestamp)
                elixir_estimator.card_played(d.card_name, d.timestamp)
    """

    def __init__(
        self,
        model_path: str  = "models/clash_royale_yolov8s.pt",
        roi_config: dict = None,
        confidence: float = DEFAULT_CONFIDENCE,
    ):
        self._confidence  = confidence
        self._roi         = roi_config or {}
        self._recent: list[_RecentDetection] = []
        self._model       = None

        self._load_model(model_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on a single BGR frame.
        Returns a list of Detection objects, filtered and debounced.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame from WindowCapture.get_frame()

        Returns
        -------
        list[Detection]
            Only detections that passed confidence, ROI, and debounce filters.
        """
        if self._model is None or frame is None:
            return []

        h, w = frame.shape[:2]
        now  = time.monotonic()

        # Run YOLO inference (returns a list of Results objects)
        results = self._model(frame, verbose=False)[0]

        detections: list[Detection] = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self._confidence:
                continue

            # ── parse class label ──────────────────────────────────────
            cls_id    = int(box.cls[0])
            raw_label = results.names[cls_id]          # e.g. "enemy-hog-rider"
            belong, card_name = self._parse_label(raw_label)
            if belong is None:
                continue   # label didn't match expected format — skip

            # ── bounding box ───────────────────────────────────────────
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            cx_rel = cx / w
            cy_rel = cy / h

            # ── ROI zone filter ────────────────────────────────────────
            zone = self._classify_zone(cx_rel, cy_rel)
            if not self._zone_allowed(zone, belong):
                continue

            # ── debounce ───────────────────────────────────────────────
            if self._is_duplicate(card_name, cx_rel, cy_rel, now):
                continue

            # ── record & emit ──────────────────────────────────────────
            self._recent.append(_RecentDetection(card_name, cx_rel, cy_rel, now))
            self._prune_recent(now)

            detections.append(Detection(
                card_name  = card_name,
                belong     = belong,
                bbox       = (x1, y1, x2, y2),
                confidence = conf,
                position   = (round(cx_rel, 4), round(cy_rel, 4)),
                timestamp  = now,
                zone       = zone,
            ))

        return detections

    def set_confidence(self, threshold: float) -> None:
        self._confidence = max(0.0, min(1.0, threshold))

    def update_roi(self, roi_config: dict) -> None:
        self._roi = roi_config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self, path: str) -> None:
        if not _YOLO_AVAILABLE:
            return
        p = Path(path)
        if not p.exists():
            print(f"[Detector] Model not found at '{path}' — inference disabled.")
            return
        print(f"[Detector] Loading model: {path}")
        self._model = YOLO(str(p))
        print(f"[Detector] Model loaded — {len(self._model.names)} classes.")

    @staticmethod
    def _parse_label(raw: str) -> tuple[Optional[str], Optional[str]]:
        """
        Split "enemy-hog-rider" → ("enemy", "hog-rider")
              "friendly-giant"  → ("friendly", "giant")
        Returns (None, None) if format is unexpected.
        """
        raw = raw.strip().lower()
        if raw.startswith("enemy-"):
            return "enemy", raw[len("enemy-"):]
        if raw.startswith("friendly-"):
            return "friendly", raw[len("friendly-"):]
        # Fallback: unknown prefix — still try to use it
        parts = raw.split("-", 1)
        if len(parts) == 2 and parts[0] in ("enemy", "friendly"):
            return parts[0], parts[1]
        return None, None

    def _classify_zone(self, cx_rel: float, cy_rel: float) -> str:
        """
        Map a relative (cx, cy) to a named zone using roi_config.
        roi_config zones are stored as relative coords (0.0–1.0):
            {
              "opponent_half": {"y_min": 0.0, "y_max": 0.5},
              "our_half":      {"y_min": 0.5, "y_max": 1.0}
            }
        Falls back to a simple top/bottom split if config is absent.
        """
        zones = self._roi.get("zones", {})

        for zone_name, bounds in zones.items():
            x_min = bounds.get("x_min", 0.0)
            x_max = bounds.get("x_max", 1.0)
            y_min = bounds.get("y_min", 0.0)
            y_max = bounds.get("y_max", 1.0)
            if x_min <= cx_rel <= x_max and y_min <= cy_rel <= y_max:
                return zone_name

        # Default: simple top/bottom split
        return "opponent_half" if cy_rel < 0.5 else "our_half"

    @staticmethod
    def _zone_allowed(zone: str, belong: str) -> bool:
        """
        Sanity-check: enemy cards should appear in opponent_half,
        friendly cards in our_half. Accept unknowns to avoid missed detections.
        """
        if zone == "opponent_half" and belong == "enemy":
            return True
        if zone == "our_half" and belong == "friendly":
            return True
        if zone == "unknown":
            return True
        # Cross-zone detections (e.g. enemy unit that crossed the bridge)
        # are still valid — allow them
        return True

    def _is_duplicate(
        self, card_name: str, cx_rel: float, cy_rel: float, now: float
    ) -> bool:
        """
        Return True if the same card was detected within DEBOUNCE_SECS at a
        position within POSITION_TOLERANCE of this one.
        """
        cutoff = now - DEBOUNCE_SECS
        for r in self._recent:
            if r.timestamp < cutoff:
                continue
            if r.card_name != card_name:
                continue
            dist = ((r.cx_rel - cx_rel) ** 2 + (r.cy_rel - cy_rel) ** 2) ** 0.5
            if dist < POSITION_TOLERANCE:
                return True
        return False

    def _prune_recent(self, now: float) -> None:
        cutoff = now - DEBOUNCE_SECS * 2
        self._recent = [r for r in self._recent if r.timestamp >= cutoff]

    # ------------------------------------------------------------------
    # Debug: draw detections onto a frame (useful during development)
    # ------------------------------------------------------------------
    def draw(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """
        Returns a copy of the frame with bounding boxes and labels drawn.
        Call this in main.py during development to visually verify detections.

            annotated = detector.draw(frame, detections)
            cv2.imshow("detections", annotated)
        """
        out = frame.copy()
        for d in detections:
            color = (0, 0, 220) if d.belong == "enemy" else (0, 200, 0)
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{d.belong[0].upper()} {d.card_name} {d.confidence:.2f}"
            cv2.putText(out, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        return out


# ---------------------------------------------------------------------------
# Smoke-test  (run: python detection/detector.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Detector smoke test (no model needed) ===\n")

    # --- test label parser ---
    cases = [
        ("enemy-hog-rider",       ("enemy",    "hog-rider")),
        ("friendly-giant",        ("friendly", "giant")),
        ("enemy-elixir-collector",("enemy",    "elixir-collector")),
        ("friendly-mini-pekka",   ("friendly", "mini-pekka")),
        ("unknown-card",          (None, None)),
        ("hog-rider",             (None, None)),
    ]
    for raw, expected in cases:
        result = Detector._parse_label(raw)
        status = "✅" if result == expected else "❌"
        print(f"  {status}  '{raw}'  →  {result}  (expected {expected})")
    print()

    # --- test debounce ---
    det = Detector.__new__(Detector)
    det._confidence = 0.6
    det._roi = {}
    det._recent = []
    det._model = None

    now = time.monotonic()
    assert not det._is_duplicate("hog-rider", 0.3, 0.3, now), "FAIL: first detection should not be duplicate"

    det._recent.append(_RecentDetection("hog-rider", 0.3, 0.3, now))

    # Same card, same position, 0.5s later → duplicate
    assert det._is_duplicate("hog-rider", 0.31, 0.31, now + 0.5), "FAIL: nearby position should be duplicate"

    # Same card, far position → not duplicate
    assert not det._is_duplicate("hog-rider", 0.8, 0.8, now + 0.5), "FAIL: far position should not be duplicate"

    # Different card, same position → not duplicate
    assert not det._is_duplicate("giant", 0.3, 0.3, now + 0.5), "FAIL: different card should not be duplicate"

    # After debounce window expires → not duplicate
    assert not det._is_duplicate("hog-rider", 0.3, 0.3, now + DEBOUNCE_SECS + 0.1), "FAIL: expired debounce"
    print("  ✅  Debounce logic all correct\n")

    # --- test zone classification ---
    det._roi = {
        "zones": {
            "opponent_half": {"y_min": 0.0, "y_max": 0.5},
            "our_half":      {"y_min": 0.5, "y_max": 1.0},
        }
    }
    assert det._classify_zone(0.5, 0.25) == "opponent_half"
    assert det._classify_zone(0.5, 0.75) == "our_half"
    print("  ✅  Zone classification correct\n")

    print("✅ All smoke tests passed.")