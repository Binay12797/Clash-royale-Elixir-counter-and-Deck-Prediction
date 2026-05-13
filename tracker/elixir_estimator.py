"""
tracker/elixir_estimator.py

Estimates the opponent's current elixir based on:
  - Passive regeneration (time-based)
  - Elixir drain when a card is played (YOLO detection)
  - Double-elixir phase (final 60s of regulation: 2:00–3:00 elapsed)
  - Triple-elixir phase (final 60s of overtime: 4:00–5:00 elapsed)
  - Elixir Collector bonus regen if detected on field

Elixir rules (Clash Royale — confirmed from official wiki):
  - Range:  0 – 10
  - Start:  5
  - Normal regen:         +1 per 2.8 s   (0:00 – 2:00 elapsed)
  - Double-elixir regen:  +1 per 1.4 s   (2:00 – 3:00 elapsed, final min of regulation)
  - Overtime:             +1 per 1.4 s   (3:00 – 4:00 elapsed, first min of OT)
  - Triple-elixir regen:  +1 per ~0.9 s  (4:00+ elapsed, final min of OT)
  - Elixir Collector:     +1 per 13 s extra per collector on field
                          (Nov 2025 balance update)

Game start trigger:
  - User presses SPACE when "FIGHT!" appears on screen
  - main.py listens for the keypress via GameStartTrigger and calls estimator.start_game()
  - See tracker/game_start_trigger.py
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Game-clock thresholds (seconds elapsed from match start)
# Source: https://clashroyale.fandom.com/wiki/Elixir
# ---------------------------------------------------------------------------
DOUBLE_ELIXIR_START  = 120.0   # 2:00 elapsed → final 60s of regulation + all of OT
TRIPLE_ELIXIR_START  = 240.0   # 4:00 elapsed → final 60s of overtime
OVERTIME_START       = 180.0   # 3:00 elapsed → sudden-death overtime begins

REGEN_NORMAL   = 1.0 / 2.8    # elixir per second
REGEN_DOUBLE   = 1.0 / 1.4    # elixir per second
REGEN_TRIPLE   = 1.0 / 0.9    # elixir per second (triple elixir overtime)

# Post-May 2024 nerf: was 9s → 11s → now 12s (Nov 2025 update: 13s; use latest)
ELIXIR_COLLECTOR_RATE = 1.0 / 13.0  # bonus elixir per second, per collector

ELIXIR_MIN   = 0.0
ELIXIR_MAX   = 10.0
ELIXIR_START = 5.0

# How long (seconds) to suppress duplicate detections at same position
DEBOUNCE_WINDOW = 2.0


@dataclass
class _PlayEvent:
    card_name: str
    timestamp: float   # wall-clock time (time.monotonic())
    cost: int


@dataclass
class ElixirState:
    estimated_elixir: float          # 0.0 – 10.0
    elixir_bar_fill: float           # 0.0 – 1.0  (for UI progress bar)
    is_double_elixir: bool
    is_triple_elixir: bool
    game_time: float                  # seconds elapsed


class ElixirEstimator:
    """
    Usage
    -----
    estimator = ElixirEstimator(card_db_path="data/card_database.json")
    estimator.start_game()                         # call when match begins
    estimator.card_played("hog-rider", timestamp)  # call on YOLO detection
    estimator.set_collector_count(1)               # call when collector detected
    state = estimator.get_state()                  # call each overlay refresh
    """

    def __init__(self, card_db_path: str = "data/card_database.json"):
        self._card_db: dict[str, int] = self._load_card_db(card_db_path)

        # Game state
        self._game_start_wall: Optional[float] = None   # time.monotonic()
        self._elixir: float = ELIXIR_START
        self._last_update_wall: Optional[float] = None

        # Elixir Collector tracking
        self._collector_count: int = 0   # number of collectors on opponent field

        # Debounce: track recent play events to avoid double-counting
        self._recent_plays: list[_PlayEvent] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_game(self) -> None:
        """Call this when a new match begins (e.g. main.py detects game start)."""
        self._game_start_wall = time.monotonic()
        self._last_update_wall = self._game_start_wall
        self._elixir = ELIXIR_START
        self._collector_count = 0
        self._recent_plays.clear()

    def card_played(self, card_name: str, timestamp: Optional[float] = None) -> None:
        """
        Call when YOLO detects the opponent deploying a card.

        Parameters
        ----------
        card_name : str
            The *bare* card name (no enemy/friendly prefix), e.g. "hog-rider".
            Strip the prefix before calling:
                raw_class = "enemy-hog-rider"
                name = raw_class.split("-", 1)[1]   # → "hog-rider"
        timestamp : float, optional
            time.monotonic() of detection. Defaults to now.
        """
        if timestamp is None:
            timestamp = time.monotonic()

        # --- flush old regen first so elixir is current ---
        self._tick(timestamp)

        # --- debounce ---
        if self._is_duplicate(card_name, timestamp):
            return

        cost = self._get_cost(card_name)

        # Record event
        self._recent_plays.append(_PlayEvent(card_name, timestamp, cost))
        self._prune_recent_plays(timestamp)

        # Drain elixir
        self._elixir = max(ELIXIR_MIN, self._elixir - cost)

        # If it was an Elixir Collector, bump collector count
        if self._is_collector(card_name):
            self._collector_count += 1

    def collector_destroyed(self) -> None:
        """Call when YOLO detects an Elixir Collector being destroyed."""
        self._collector_count = max(0, self._collector_count - 1)

    def set_collector_count(self, count: int) -> None:
        """Directly set collector count (useful if you recount each frame)."""
        self._collector_count = max(0, count)

    def reset_game(self) -> None:
        """Alias for start_game — call between matches."""
        self.start_game()

    def get_state(self) -> ElixirState:
        """
        Tick regen up to *now* and return a snapshot of the current state.
        Safe to call every 500 ms from the overlay refresh loop.
        """
        now = time.monotonic()
        self._tick(now)

        game_time = self._get_game_time(now)
        is_double = self._is_double_elixir(game_time)
        is_triple = self._is_triple_elixir(game_time)

        return ElixirState(
            estimated_elixir=round(self._elixir, 2),
            elixir_bar_fill=self._elixir / ELIXIR_MAX,
            is_double_elixir=is_double,
            is_triple_elixir=is_triple,
            game_time=game_time,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tick(self, now: float) -> None:
        """Advance elixir regen from _last_update_wall to *now*."""
        if self._game_start_wall is None or self._last_update_wall is None:
            # Game hasn't started yet — nothing to tick
            return

        dt = now - self._last_update_wall
        if dt <= 0:
            return

        game_time_at_last = self._get_game_time(self._last_update_wall)
        game_time_now     = self._get_game_time(now)

        # Handle the transition from normal → double-elixir mid-tick
        elixir_gained = self._integrate_regen(
            game_time_at_last, game_time_now, dt
        )

        # Collector bonus
        elixir_gained += self._collector_count * ELIXIR_COLLECTOR_RATE * dt

        self._elixir = min(ELIXIR_MAX, self._elixir + elixir_gained)
        self._last_update_wall = now

    def _integrate_regen(
        self, t_start: float, t_end: float, dt: float
    ) -> float:
        """
        Correctly handle ticks that straddle phase boundaries.
        Phases (seconds elapsed):
          0–120  : normal (1/2.8s)
          120–240: double (1/1.4s)  — final min of regulation + first min OT
          240+   : triple (1/0.9s)  — final min of overtime
        Returns total elixir regenerated over [t_start, t_end].
        """
        B1 = DOUBLE_ELIXIR_START   # 120
        B2 = TRIPLE_ELIXIR_START   # 240
        elixir = 0.0

        # Normal phase slice
        s, e = t_start, min(t_end, B1)
        if e > s:
            elixir += REGEN_NORMAL * (e - s)

        # Double elixir phase slice
        s, e = max(t_start, B1), min(t_end, B2)
        if e > s:
            elixir += REGEN_DOUBLE * (e - s)

        # Triple elixir phase slice
        s, e = max(t_start, B2), t_end
        if e > s:
            elixir += REGEN_TRIPLE * (e - s)

        return elixir

    def _get_game_time(self, wall: float) -> float:
        """Seconds elapsed since game start (0 if not started)."""
        if self._game_start_wall is None:
            return 0.0
        return max(0.0, wall - self._game_start_wall)

    def _is_double_elixir(self, game_time: float) -> bool:
        return DOUBLE_ELIXIR_START <= game_time < TRIPLE_ELIXIR_START

    def _is_triple_elixir(self, game_time: float) -> bool:
        return game_time >= TRIPLE_ELIXIR_START

    def _get_cost(self, card_name: str) -> int:
        """Look up elixir cost; default 3 if unknown."""
        # Try exact match first, then fuzzy prefix strip
        name_lower = card_name.lower().strip()
        cost = self._card_db.get(name_lower)
        if cost is not None:
            return cost

        # Try replacing spaces/underscores with hyphens (normalise)
        normalised = name_lower.replace(" ", "-").replace("_", "-")
        cost = self._card_db.get(normalised)
        if cost is not None:
            return cost

        # Fallback: average elixir cost
        print(f"[ElixirEstimator] Unknown card '{card_name}', defaulting cost to 3")
        return 3

    def _is_collector(self, card_name: str) -> bool:
        name = card_name.lower().replace(" ", "-").replace("_", "-")
        return "elixir-collector" in name or name == "collector"

    def _is_duplicate(self, card_name: str, timestamp: float) -> bool:
        """Return True if this card was already registered within DEBOUNCE_WINDOW."""
        cutoff = timestamp - DEBOUNCE_WINDOW
        for event in self._recent_plays:
            if event.card_name == card_name and event.timestamp >= cutoff:
                return True
        return False

    def _prune_recent_plays(self, now: float) -> None:
        cutoff = now - DEBOUNCE_WINDOW * 2
        self._recent_plays = [e for e in self._recent_plays if e.timestamp >= cutoff]

    @staticmethod
    def _load_card_db(path: str) -> dict[str, int]:
        """
        Load card_database.json.
        Expected format (flexible):
          { "hog-rider": {"elixir": 4, ...}, ... }
          OR
          { "hog-rider": 4, ... }
        Returns a flat dict: { card_name_lower: elixir_cost }
        """
        p = Path(path)
        if not p.exists():
            print(f"[ElixirEstimator] Warning: card_database.json not found at '{path}'")
            return {}

        with p.open("r", encoding="utf-8") as f:
            raw: dict = json.load(f)

        db: dict[str, int] = {}
        for key, value in raw.items():
            name = key.lower().strip()
            if isinstance(value, dict):
                cost = value.get("elixir") or value.get("elixir_cost") or value.get("cost")
            elif isinstance(value, (int, float)):
                cost = int(value)
            else:
                cost = None

            if cost is not None:
                db[name] = int(cost)

        return db


# ---------------------------------------------------------------------------
# Quick smoke-test  (run: python tracker/elixir_estimator.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import math

    print("=== ElixirEstimator smoke test ===\n")

    estimator = ElixirEstimator.__new__(ElixirEstimator)
    # Inject a tiny card db for testing without the real file
    estimator._card_db = {
        "hog-rider": 4,
        "fireball": 4,
        "minions": 3,
        "elixir-collector": 6,
        "giant": 5,
    }
    estimator._game_start_wall = None
    estimator._last_update_wall = None
    estimator._elixir = ELIXIR_START
    estimator._collector_count = 0
    estimator._recent_plays = []

    estimator.start_game()
    start = estimator._game_start_wall

    def fake_time(seconds_elapsed: float) -> float:
        return start + seconds_elapsed

    # T=0 → elixir should be 5
    estimator._last_update_wall = fake_time(0)
    state = estimator.get_state.__func__(estimator)  # won't tick past "now"

    # Simulate 5 seconds passing
    estimator._last_update_wall = fake_time(0)
    estimator._tick(fake_time(5))
    expected = min(10.0, 5.0 + REGEN_NORMAL * 5)
    print(f"After 5s normal regen: {estimator._elixir:.3f}  (expected ~{expected:.3f})")
    assert math.isclose(estimator._elixir, expected, abs_tol=0.01), "FAIL: normal regen"

    # Opponent plays Hog Rider (cost 4) at T=5
    estimator.card_played("hog-rider", fake_time(5))
    expected -= 4
    print(f"After hog rider played: {estimator._elixir:.3f}  (expected ~{expected:.3f})")
    assert math.isclose(estimator._elixir, expected, abs_tol=0.01), "FAIL: card drain"

    # Debounce: same card 1s later should NOT drain again
    before = estimator._elixir
    estimator._last_update_wall = fake_time(5)
    estimator._tick(fake_time(6))
    estimator.card_played("hog-rider", fake_time(6))
    # Only regen should have changed, no drain
    regen_1s = REGEN_NORMAL * 1
    print(f"After debounce duplicate: {estimator._elixir:.3f}  (expected ~{before + regen_1s:.3f})")
    assert estimator._elixir > before, "FAIL: no regen during debounce window"

    # Simulate double-elixir phase transition (T=119→130)
    estimator._last_update_wall = fake_time(119)
    estimator._elixir = 3.0
    estimator._tick(fake_time(130))
    # 1s normal (119→120) + 10s double (120→130)
    expected_de = min(10.0, 3.0 + REGEN_NORMAL * 1 + REGEN_DOUBLE * 10)
    print(f"After double-elixir transition: {estimator._elixir:.3f}  (expected ~{expected_de:.3f})")
    assert math.isclose(estimator._elixir, expected_de, abs_tol=0.05), "FAIL: double elixir"

    # Simulate triple-elixir transition (T=238→245)
    estimator._last_update_wall = fake_time(238)
    estimator._elixir = 2.0
    estimator._tick(fake_time(245))
    # 2s double (238→240) + 5s triple (240→245)
    expected_te = min(10.0, 2.0 + REGEN_DOUBLE * 2 + REGEN_TRIPLE * 5)
    print(f"After triple-elixir transition: {estimator._elixir:.3f}  (expected ~{expected_te:.3f})")
    assert math.isclose(estimator._elixir, expected_te, abs_tol=0.05), "FAIL: triple elixir"

    # is_triple_elixir flag
    state = estimator.get_state()
    # game_time will be whatever time.monotonic() returns, not fake — just check flags are booleans
    assert isinstance(state.is_double_elixir, bool)
    assert isinstance(state.is_triple_elixir, bool)
    print(f"ElixirState flags ok  (is_double={state.is_double_elixir}, is_triple={state.is_triple_elixir})")

    # Elixir capped at 10
    estimator._elixir = 9.9
    estimator._last_update_wall = fake_time(130)
    estimator._tick(fake_time(145))
    assert estimator._elixir == ELIXIR_MAX, f"FAIL: cap exceeded ({estimator._elixir})"
    print(f"Elixir cap respected: {estimator._elixir}")

    print("\n✅ All smoke tests passed.")