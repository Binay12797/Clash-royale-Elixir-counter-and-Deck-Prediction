"""
window_capture.py
=================
Finds the Clash Royale window (Google Play Games),
captures frames using mss, and provides a live preview.

Usage (standalone test):
    python window_capture.py
"""

import win32gui
import win32con
import mss
import numpy as np
import cv2
from dataclasses import dataclass

# ── Config ────────────────────────────────────────────────────────────────────
WINDOW_TITLE_KEYWORD = "Clash Royale - BINAYBISTA09"   # partial match — handles "Clash Royale - BINAYBISTA09"
TARGET_FPS           = 30


# ── Window region dataclass ───────────────────────────────────────────────────
@dataclass
class WindowRegion:
    hwnd:   int
    left:   int
    top:    int
    width:  int
    height: int
    title:  str

    def to_mss_monitor(self) -> dict:
        """Convert to mss monitor dict."""
        return {
            "left":   self.left,
            "top":    self.top,
            "width":  self.width,
            "height": self.height,
        }

    def __str__(self):
        return (f"[{self.title}]  "
                f"pos=({self.left},{self.top})  "
                f"size={self.width}x{self.height}")


# ── Window finder ─────────────────────────────────────────────────────────────
def find_window(keyword: str = WINDOW_TITLE_KEYWORD) -> WindowRegion | None:
    """
    Search all open windows for one whose title contains `keyword`.
    Returns a WindowRegion or None if not found.
    """
    found = []

    def _callback(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if keyword.lower() in title.lower():
            rect = win32gui.GetWindowRect(hwnd)   # (left, top, right, bottom)
            left, top, right, bottom = rect
            width  = right  - left
            height = bottom - top
            if width > 0 and height > 0:
                found.append(WindowRegion(
                    hwnd=hwnd, left=left, top=top,
                    width=width, height=height, title=title
                ))

    win32gui.EnumWindows(_callback, None)

    if not found:
        return None
    if len(found) > 1:
        print(f"[WARN] Multiple windows matched '{keyword}':")
        for i, w in enumerate(found):
            print(f"  [{i}] {w}")
        print(f"  → Using [{0}]")
    return found[0]


def bring_to_front(hwnd: int):
    """Bring the target window to the foreground."""
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
    except Exception as e:
        print(f"[WARN] Could not bring window to front: {e}")


# ── Frame capturer ────────────────────────────────────────────────────────────
class WindowCapture:
    """
    Continuously captures frames from the Clash Royale window.

    Usage:
        cap = WindowCapture()
        cap.start()
        frame = cap.get_frame()   # numpy BGR array
        cap.stop()
    """

    def __init__(self, keyword: str = WINDOW_TITLE_KEYWORD):
        self.keyword  = keyword
        self.region   = None
        self._sct     = None
        self._running = False

    # ── public ────────────────────────────────────────────────────────────
    def start(self) -> bool:
        """Find the window and initialise mss. Returns True on success."""
        self.region = find_window(self.keyword)
        if self.region is None:
            print(f"[ERROR] Window containing '{self.keyword}' not found.")
            print("        Make sure Clash Royale is open and visible.")
            return False

        print(f"[INFO] Found window: {self.region}")
        self._sct     = mss.mss()
        self._running = True
        return True

    def get_frame(self) -> np.ndarray | None:
        """
        Capture one frame from the game window.
        Refreshes window position each call (handles window moves).
        Returns BGR numpy array, or None on failure.
        """
        if not self._running or self._sct is None:
            return None

        # re-query window position in case it moved
        updated = find_window(self.keyword)
        if updated is None:
            print("[WARN] Window lost — is the game still open?")
            return None
        self.region = updated

        monitor = self.region.to_mss_monitor()
        raw     = self._sct.grab(monitor)

        # mss returns BGRA → convert to BGR for cv2
        frame = np.array(raw)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def stop(self):
        self._running = False
        if self._sct:
            self._sct.close()
            self._sct = None
        print("[INFO] Capture stopped.")

    @property
    def window_info(self) -> WindowRegion | None:
        return self.region


# ── Standalone preview (run directly to test) ─────────────────────────────────
def _preview():
    cap = WindowCapture()
    if not cap.start():
        return

    bring_to_front(cap.region.hwnd)

    win_name = "Clash Royale — Capture Preview  (Q to quit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # scale preview to fit screen nicely
    preview_w = min(cap.region.width, 720)
    preview_h = int(preview_w * cap.region.height / cap.region.width)
    cv2.resizeWindow(win_name, preview_w, preview_h)

    print("[INFO] Live preview started. Press Q to quit.")
    delay = max(1, int(1000 / TARGET_FPS))

    while True:
        frame = cap.get_frame()
        if frame is None:
            break

        # info overlay
        info = (f"{cap.region.width}x{cap.region.height}  |  "
                f"pos ({cap.region.left},{cap.region.top})")
        cv2.putText(frame, info, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(win_name, frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _preview()