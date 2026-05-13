"""
overlay/panel.py

Always-on-top, borderless, semi-transparent PyQt5 overlay panel.
Sits on the LEFT side of the game window.

Layout:
    ┌─────────────────────────┐
    │  ◈ ROYAL SCOUT          │  ← title
    ├─────────────────────────┤
    │  OPPONENT DECK  6/8     │
    │  [H][F][S][C][?][?][][]  │
    ├─────────────────────────┤
    │  HAND  [H][?][?][?]     │
    │  Next → [F]             │
    ├─────────────────────────┤
    │  ON FIELD               │
    │  Giant · Musketeer      │
    ├─────────────────────────┤
    │  ELIXIR  ████████░░ ~8  │
    │  ◈ DOUBLE ELIXIR        │
    └─────────────────────────┘

Design:
    - Deep charcoal background (#0d0f14), 88% opacity
    - Elixir purple accent (#b14aed)
    - Double-elixir gold (#f5c518)
    - Triple-elixir red (#ff4444)

Dependencies:
    pip install PyQt5
"""

from __future__ import annotations

import time
from typing import Optional

from PyQt5.QtCore    import Qt, QTimer, QRect
from PyQt5.QtGui     import (QPainter, QColor, QFont,
                              QPen, QBrush, QPainterPath, QLinearGradient)
from PyQt5.QtWidgets import QWidget, QApplication


# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------
PANEL_WIDTH      = 240
CARD_SLOT_SIZE   = 26
CARD_SLOT_GAP    = 4
SECTION_GAP      = 10
CORNER_RADIUS    = 8

C_BG             = QColor(13,  15,  20,  224)
C_BORDER         = QColor(255, 255, 255, 18)
C_TEXT_HEADING   = QColor(160, 170, 190)
C_TEXT_MAIN      = QColor(220, 225, 235)
C_TEXT_DIM       = QColor(90,  100, 120)

C_ELIXIR_NORMAL  = QColor(177,  74, 237)
C_ELIXIR_DOUBLE  = QColor(245, 197,  24)
C_ELIXIR_TRIPLE  = QColor(255,  68,  68)
C_ELIXIR_EMPTY   = QColor(40,   30,  55)

C_CARD_SEEN      = QColor(180,  60,  60)
C_CARD_UNKNOWN   = QColor(35,   40,  55)
C_CARD_HAND      = QColor(200,  80,  40)
C_CARD_FIELD     = QColor(220, 100,  30)

FONT_MONO        = "Courier New"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _elixir_color(state) -> QColor:
    if state is None:
        return C_ELIXIR_NORMAL
    if getattr(state, "is_triple_elixir", False):
        return C_ELIXIR_TRIPLE
    if getattr(state, "is_double_elixir", False):
        return C_ELIXIR_DOUBLE
    return C_ELIXIR_NORMAL


def _abbreviate(name: str) -> str:
    """'hog-rider' → 'HR',  'giant' → 'GI',  '?' → '?'"""
    if name == "?":
        return "?"
    parts = name.replace("-", " ").replace("_", " ").split()
    if len(parts) == 1:
        return parts[0][:2].upper()
    return "".join(pt[0] for pt in parts[:3]).upper()


# ---------------------------------------------------------------------------
# Main overlay widget
# ---------------------------------------------------------------------------
class OverlayPanel(QWidget):
    """
    Borderless always-on-top overlay.

    Typical usage from main.py
    --------------------------
        panel = OverlayPanel()
        panel.show()
        # inside main loop:
        panel.update_state(cycle_state, elixir_state)

    Or pass a state_provider for automatic polling:
        panel = OverlayPanel(state_provider=lambda: (tracker.get_state(),
                                                      estimator.get_state()))
    """

    def __init__(
        self,
        game_window_rect: Optional[QRect] = None,
        state_provider=None,
        refresh_ms: int = 500,
        parent=None,
    ):
        super().__init__(parent)

        self._cycle_state    = None
        self._elixir_state   = None
        self._state_provider = state_provider
        self._waiting        = True

        self._setup_window()
        self._setup_fonts()

        if game_window_rect:
            self.set_game_window(game_window_rect)
        else:
            self._position_default()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start(refresh_ms)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_game_window(self, rect: QRect) -> None:
        """Reposition panel to the left of the game window rect."""
        x = rect.x() - PANEL_WIDTH - 6
        y = rect.y()
        h = rect.height()
        self.setGeometry(x, y, PANEL_WIDTH, h)

    def update_state(self, cycle_state, elixir_state) -> None:
        """Push new state in from the main loop."""
        self._cycle_state  = cycle_state
        self._elixir_state = elixir_state
        self._waiting      = False
        self.update()

    def show_waiting(self) -> None:
        """Switch to 'Press SPACE to start' screen."""
        self._waiting      = True
        self._cycle_state  = None
        self._elixir_state = None
        self.update()

    # ------------------------------------------------------------------
    # Window setup
    # ------------------------------------------------------------------

    def _setup_window(self) -> None:
        self.setWindowFlags(
            Qt.FramelessWindowHint      |
            Qt.WindowStaysOnTopHint     |
            Qt.Tool                     |
            Qt.WindowTransparentForInput
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setWindowTitle("Royal Scout")

    def _setup_fonts(self) -> None:
        self._f_title   = QFont(FONT_MONO, 9,  QFont.Bold)
        self._f_heading = QFont(FONT_MONO, 7,  QFont.Bold)
        self._f_body    = QFont(FONT_MONO, 8)
        self._f_card    = QFont(FONT_MONO, 6)
        self._f_elixir  = QFont(FONT_MONO, 9,  QFont.Bold)

    def _position_default(self) -> None:
        screen = QApplication.primaryScreen().availableGeometry()
        self.setGeometry(screen.x(), screen.y(), PANEL_WIDTH, screen.height())

    # ------------------------------------------------------------------
    # Timer
    # ------------------------------------------------------------------

    def _on_timer(self) -> None:
        if self._state_provider:
            try:
                cs, es = self._state_provider()
                self.update_state(cs, es)
            except Exception:
                pass
        else:
            self.update()

    # ------------------------------------------------------------------
    # Paint dispatch
    # ------------------------------------------------------------------

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        self._draw_background(p, w, h)

        if self._waiting:
            self._draw_waiting(p, w, h)
        else:
            y = 0
            y = self._draw_title(p, w, y)
            y = self._draw_deck_section(p, w, y)
            y = self._draw_hand_section(p, w, y)
            y = self._draw_field_section(p, w, y)
            self._draw_elixir_section(p, w, y)

        p.end()

    # ------------------------------------------------------------------
    # Background
    # ------------------------------------------------------------------

    def _draw_background(self, p: QPainter, w: int, h: int) -> None:
        path = QPainterPath()
        path.addRoundedRect(0, 0, w, h, CORNER_RADIUS, CORNER_RADIUS)
        p.fillPath(path, C_BG)
        p.setPen(QPen(C_BORDER, 1))
        p.drawPath(path)

    # ------------------------------------------------------------------
    # Waiting screen
    # ------------------------------------------------------------------

    def _draw_waiting(self, p: QPainter, w: int, h: int) -> None:
        p.setPen(C_ELIXIR_NORMAL)
        p.setFont(self._f_title)
        p.drawText(QRect(0, h // 2 - 40, w, 24), Qt.AlignCenter, "◈ ROYAL SCOUT")

        p.setPen(C_TEXT_HEADING)
        p.setFont(self._f_body)
        p.drawText(QRect(0, h // 2 - 10, w, 22), Qt.AlignCenter, "Press SPACE")

        p.setPen(C_TEXT_DIM)
        p.setFont(self._f_card)
        p.drawText(QRect(0, h // 2 + 14, w, 18), Qt.AlignCenter, "when FIGHT! appears")

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------

    def _draw_title(self, p: QPainter, w: int, y: int) -> int:
        p.setPen(C_ELIXIR_NORMAL)
        p.setFont(self._f_title)
        p.drawText(QRect(0, y, w, 28), Qt.AlignCenter, "◈  ROYAL SCOUT")
        p.setPen(QPen(C_BORDER, 1))
        p.drawLine(8, y + 28, w - 8, y + 28)
        return y + 34

    # ------------------------------------------------------------------
    # Deck section
    # ------------------------------------------------------------------

    def _draw_deck_section(self, p: QPainter, w: int, y: int) -> int:
        cs = self._cycle_state
        seen_names  = list(cs.seen_cards.keys()) if cs else []
        hand_names  = list(cs.current_hand)      if cs else []
        field_names = list(cs.on_field)          if cs else []

        label = f"OPPONENT DECK  {len(seen_names)}/8"
        y = self._section_label(p, w, y, label)

        slots   = seen_names + ["?"] * (8 - len(seen_names))
        cols    = 4
        total_w = cols * CARD_SLOT_SIZE + (cols - 1) * CARD_SLOT_GAP
        x0      = (w - total_w) // 2

        for i, name in enumerate(slots[:8]):
            col = i % cols
            row = i // cols
            sx  = x0 + col * (CARD_SLOT_SIZE + CARD_SLOT_GAP)
            sy  = y  + row * (CARD_SLOT_SIZE + CARD_SLOT_GAP)
            self._card_chip(p, sx, sy, name,
                            on_hand=name in hand_names,
                            on_field=name in field_names)

        rows = (min(len(slots), 8) + cols - 1) // cols
        y   += rows * (CARD_SLOT_SIZE + CARD_SLOT_GAP) + SECTION_GAP
        self._divider(p, w, y)
        return y + 6

    # ------------------------------------------------------------------
    # Hand section
    # ------------------------------------------------------------------

    def _draw_hand_section(self, p: QPainter, w: int, y: int) -> int:
        cs        = self._cycle_state
        hand      = list(cs.current_hand) if cs else []
        next_card = getattr(cs, "next_card", None) if cs else None

        y = self._section_label(p, w, y, "HAND")

        cols    = 4
        total_w = cols * CARD_SLOT_SIZE + (cols - 1) * CARD_SLOT_GAP
        x0      = (w - total_w) // 2

        for i in range(4):
            sx   = x0 + i * (CARD_SLOT_SIZE + CARD_SLOT_GAP)
            name = hand[i] if i < len(hand) else "?"
            self._card_chip(p, sx, y, name, on_hand=True, on_field=False)
        y += CARD_SLOT_SIZE + 6

        if next_card:
            p.setPen(C_TEXT_DIM)
            p.setFont(self._f_card)
            p.drawText(QRect(x0, y, 40, CARD_SLOT_SIZE), Qt.AlignVCenter, "Next →")
            self._card_chip(p, x0 + 46, y, next_card, on_hand=False, on_field=False)
            y += CARD_SLOT_SIZE + 4

        y += SECTION_GAP
        self._divider(p, w, y)
        return y + 6

    # ------------------------------------------------------------------
    # On-field section
    # ------------------------------------------------------------------

    def _draw_field_section(self, p: QPainter, w: int, y: int) -> int:
        cs       = self._cycle_state
        on_field = list(cs.on_field) if cs else []

        y = self._section_label(p, w, y, "ON FIELD")

        if not on_field:
            p.setPen(C_TEXT_DIM)
            p.setFont(self._f_body)
            p.drawText(QRect(0, y, w, 18), Qt.AlignCenter, "—")
            y += 18
        else:
            x, row_y = 10, y
            for name in on_field:
                if x + CARD_SLOT_SIZE > w - 10:
                    x = 10
                    row_y += CARD_SLOT_SIZE + CARD_SLOT_GAP
                self._card_chip(p, x, row_y, name, on_hand=False, on_field=True)
                x += CARD_SLOT_SIZE + CARD_SLOT_GAP
            y = row_y + CARD_SLOT_SIZE + 4

        y += SECTION_GAP
        self._divider(p, w, y)
        return y + 6

    # ------------------------------------------------------------------
    # Elixir section
    # ------------------------------------------------------------------

    def _draw_elixir_section(self, p: QPainter, w: int, y: int) -> int:
        es         = self._elixir_state
        elixir     = es.estimated_elixir if es else 5.0
        bar_fill   = es.elixir_bar_fill  if es else 0.5
        is_double  = getattr(es, "is_double_elixir", False) if es else False
        is_triple  = getattr(es, "is_triple_elixir", False) if es else False
        game_time  = getattr(es, "game_time", 0.0)          if es else 0.0

        y = self._section_label(p, w, y, "ELIXIR")

        bar_color = _elixir_color(es)
        bar_x, bar_w, bar_h = 10, w - 20, 14
        fill_w = int(bar_w * _clamp(bar_fill, 0.0, 1.0))

        # Empty trough
        bg_path = QPainterPath()
        bg_path.addRoundedRect(bar_x, y, bar_w, bar_h, 4, 4)
        p.fillPath(bg_path, C_ELIXIR_EMPTY)

        # Filled gradient
        if fill_w > 0:
            grad = QLinearGradient(bar_x, y, bar_x + bar_w, y)
            grad.setColorAt(0.0, bar_color.lighter(130))
            grad.setColorAt(1.0, bar_color)
            fill_path = QPainterPath()
            fill_path.addRoundedRect(bar_x, y, fill_w, bar_h, 4, 4)
            p.fillPath(fill_path, QBrush(grad))

        # Count label centred on bar
        p.setPen(C_TEXT_MAIN)
        p.setFont(self._f_elixir)
        p.drawText(QRect(bar_x, y, bar_w, bar_h),
                   Qt.AlignCenter, f"~{int(elixir)}")
        y += bar_h + 6

        # Phase badge
        if is_triple:
            p.setPen(C_ELIXIR_TRIPLE)
            p.setFont(self._f_heading)
            p.drawText(QRect(0, y, w, 16), Qt.AlignCenter, "◈  TRIPLE ELIXIR")
            y += 18
        elif is_double:
            p.setPen(C_ELIXIR_DOUBLE)
            p.setFont(self._f_heading)
            p.drawText(QRect(0, y, w, 16), Qt.AlignCenter, "◈  DOUBLE ELIXIR")
            y += 18

        # Game clock
        mins = int(game_time) // 60
        secs = int(game_time) % 60
        p.setPen(C_TEXT_DIM)
        p.setFont(self._f_card)
        p.drawText(QRect(0, y, w, 14), Qt.AlignCenter, f"{mins}:{secs:02d}")
        return y + 16

    # ------------------------------------------------------------------
    # Shared draw utilities
    # ------------------------------------------------------------------

    def _section_label(self, p: QPainter, w: int, y: int, text: str) -> int:
        p.setPen(C_TEXT_HEADING)
        p.setFont(self._f_heading)
        p.drawText(QRect(10, y, w - 20, 16), Qt.AlignLeft | Qt.AlignVCenter, text)
        return y + 18

    def _divider(self, p: QPainter, w: int, y: int) -> None:
        p.setPen(QPen(C_BORDER, 1))
        p.drawLine(8, y, w - 8, y)

    def _card_chip(
        self,
        p: QPainter,
        x: int, y: int,
        name: str,
        on_hand: bool,
        on_field: bool,
    ) -> None:
        if name == "?":
            bg, fg = C_CARD_UNKNOWN, C_TEXT_DIM
        elif on_field:
            bg, fg = C_CARD_FIELD,   C_TEXT_MAIN
        elif on_hand:
            bg, fg = C_CARD_HAND,    C_TEXT_MAIN
        else:
            bg, fg = C_CARD_SEEN,    C_TEXT_MAIN

        path = QPainterPath()
        path.addRoundedRect(x, y, CARD_SLOT_SIZE, CARD_SLOT_SIZE, 3, 3)
        p.fillPath(path, bg)
        p.setPen(QPen(C_BORDER, 1))
        p.drawPath(path)

        p.setPen(fg)
        p.setFont(self._f_card)
        p.drawText(QRect(x, y, CARD_SLOT_SIZE, CARD_SLOT_SIZE),
                   Qt.AlignCenter, _abbreviate(name))


# ---------------------------------------------------------------------------
# Smoke-test — shows the overlay with dummy data for 8 seconds
# Run:  python overlay/panel.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from dataclasses import dataclass

    app = QApplication(sys.argv)

    @dataclass
    class FakeCycleState:
        seen_cards:   dict
        current_hand: list
        on_field:     list
        next_card:    str

    @dataclass
    class FakeElixirState:
        estimated_elixir: float
        elixir_bar_fill:  float
        is_double_elixir: bool
        is_triple_elixir: bool
        game_time:        float

    cycle_state = FakeCycleState(
        seen_cards   = {"hog-rider": 1, "fireball": 1, "musketeer": 1,
                        "ice-spirit": 1, "skeletons": 1},
        current_hand = ["hog-rider", "fireball", "musketeer", "ice-spirit"],
        on_field     = ["hog-rider", "musketeer"],
        next_card    = "skeletons",
    )
    elixir_state = FakeElixirState(
        estimated_elixir = 7.2,
        elixir_bar_fill  = 0.72,
        is_double_elixir = True,
        is_triple_elixir = False,
        game_time        = 135.0,
    )

    panel = OverlayPanel()
    panel.update_state(cycle_state, elixir_state)
    panel.show()

    QTimer.singleShot(8000, app.quit)
    print("Overlay open — closes in 8 seconds.")
    sys.exit(app.exec_())