"""
tracker/game_start_trigger.py

Listens for the SPACE key (global hotkey, works even when the game window has
focus) and fires registered callbacks — primarily estimator.start_game() and
cycle_tracker.reset().

Uses the `keyboard` library (pip install keyboard) which hooks at the OS level
on Windows, so it works while Google Play Games has focus.

Usage (in main.py)
------------------
    from tracker.game_start_trigger import GameStartTrigger
    from tracker.elixir_estimator  import ElixirEstimator
    from tracker.cycle_tracker     import CycleTracker

    estimator = ElixirEstimator()
    tracker   = CycleTracker()
    trigger   = GameStartTrigger()

    trigger.register(estimator.start_game)
    trigger.register(tracker.reset)        # add any reset you need
    trigger.start()

    # ... main loop ...

    trigger.stop()   # clean up on exit

Press SPACE when "FIGHT!" appears. A 3-second cooldown prevents accidental
double-fires if space is held down.
"""

import threading
import time
from typing import Callable

try:
    import keyboard
    _KEYBOARD_AVAILABLE = True
except ImportError:
    _KEYBOARD_AVAILABLE = False


TRIGGER_KEY     = "space"
COOLDOWN_SECS   = 3.0   # ignore repeated presses within this window


class GameStartTrigger:
    """
    Global hotkey listener. Fires all registered callbacks when SPACE is pressed.

    Thread-safe: callbacks are invoked on the keyboard listener thread,
    which is a daemon thread — safe to call time.monotonic() and set
    simple Python attributes from it.
    """

    def __init__(self, key: str = TRIGGER_KEY, cooldown: float = COOLDOWN_SECS):
        self._key        = key
        self._cooldown   = cooldown
        self._callbacks: list[Callable[[], None]] = []
        self._last_fire  = 0.0          # time.monotonic() of last trigger
        self._running    = False
        self._hook       = None
        self._game_active = False       # True while a match is in progress

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, callback: Callable[[], None]) -> None:
        """Add a zero-argument callable to fire on SPACE press."""
        self._callbacks.append(callback)

    def start(self) -> None:
        """
        Start listening. Non-blocking — the hook runs on a background thread
        managed by the `keyboard` library.
        """
        if not _KEYBOARD_AVAILABLE:
            print(
                "[GameStartTrigger] WARNING: 'keyboard' package not installed.\n"
                "  Run:  pip install keyboard\n"
                "  Falling back to manual trigger — call trigger.fire() directly."
            )
            return

        if self._running:
            return

        self._running = True
        self._hook = keyboard.on_press_key(self._key, self._on_keypress, suppress=False)
        print(f"[GameStartTrigger] Listening — press [{self._key.upper()}] when 'FIGHT!' appears.")

    def stop(self) -> None:
        """Unregister the hotkey hook."""
        if not _KEYBOARD_AVAILABLE or not self._running:
            return
        if self._hook:
            keyboard.unhook(self._hook)
            self._hook = None
        self._running = False
        print("[GameStartTrigger] Stopped.")

    def fire(self) -> None:
        """
        Manually trigger game start (useful for testing or if `keyboard`
        is unavailable). Respects the cooldown.
        """
        self._trigger()

    @property
    def game_active(self) -> bool:
        """True after SPACE has been pressed and the match clock is running."""
        return self._game_active

    def mark_game_over(self) -> None:
        """
        Call this when the match ends (e.g. tower destroyed detected by YOLO)
        so the overlay can show a 'waiting' state and the trigger is ready
        for the next game.
        """
        self._game_active = False
        print("[GameStartTrigger] Game marked as over — ready for next match.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_keypress(self, event) -> None:
        self._trigger()

    def _trigger(self) -> None:
        now = time.monotonic()
        if now - self._last_fire < self._cooldown:
            return   # cooldown active — ignore

        self._last_fire   = now
        self._game_active = True

        print(f"[GameStartTrigger] *** GAME STARTED *** — firing {len(self._callbacks)} callback(s)")
        for cb in self._callbacks:
            try:
                cb()
            except Exception as exc:
                print(f"[GameStartTrigger] Callback {cb.__name__} raised: {exc}")


# ---------------------------------------------------------------------------
# Smoke-test  (run: python tracker/game_start_trigger.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fired_at: list[float] = []

    def fake_start_game():
        fired_at.append(time.monotonic())
        print(f"  → fake_start_game() called  (total fires: {len(fired_at)})")

    trigger = GameStartTrigger(cooldown=1.0)
    trigger.register(fake_start_game)

    if _KEYBOARD_AVAILABLE:
        trigger.start()
        print("\nPress SPACE to test. Press Ctrl+C to exit.\n")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        trigger.stop()
    else:
        # No keyboard library — test manual fire and cooldown
        print("keyboard not installed — testing manual .fire()\n")

        trigger.fire()
        assert len(fired_at) == 1, "FAIL: first fire"

        # Rapid second fire should be suppressed by cooldown
        trigger.fire()
        assert len(fired_at) == 1, "FAIL: cooldown did not suppress second fire"
        print("Cooldown suppression: OK")

        # After cooldown expires, should fire again
        trigger._last_fire = 0.0
        trigger.fire()
        assert len(fired_at) == 2, "FAIL: fire after cooldown reset"
        print("Fire after cooldown: OK")

        assert trigger.game_active is True
        trigger.mark_game_over()
        assert trigger.game_active is False
        print("game_active flag: OK")

        print("\n✅ All smoke tests passed.")