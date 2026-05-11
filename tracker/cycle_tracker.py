"""
cycle_tracker.py
================
Tracks the opponent's card cycle in Clash Royale.

Core mechanic:
  - Deck = 8 cards in fixed rotation
  - Hand = 4 cards visible at a time
  - Playing a card removes it from hand, next card enters
  - Played card returns after 4 OTHER cards are played

Availability rule:
  Card played at play N → available again at play N+5
  (must wait for 4 other cards to be played first)

Knowledge phases:
  Phase 1 (< 8 unique seen):
    → Track which cards are cooling down vs available
    → Hand = available cards + ? for unknowns
    → Predictions are partial

  Phase 2 (all 8 seen):
    → Full cycle order known
    → Hand and next card are exact forever

Place at: tracker/cycle_tracker.py
"""

from dataclasses import dataclass
from typing      import Optional

# ── Constants ─────────────────────────────────────────────────────────────────
DECK_SIZE       = 8
HAND_SIZE       = 4
COOLDOWN_PLAYS  = 4    # must wait for 4 OTHER cards before returning
UNKNOWN         = "?"


# ── Data structures ───────────────────────────────────────────────────────────
@dataclass
class CardEvent:
    card_name:  str
    timestamp:  float    # seconds since game start
    position:   tuple    # (x, y) pixel position
    play_index: int      # sequential play number (1, 2, 3...)


@dataclass
class CycleState:
    # deck knowledge
    deck:             list[str]   # 8 slots, ? for undiscovered
    cards_discovered: int         # unique cards seen (0-8)
    cycle_complete:   bool        # True when all 8 seen

    # hand state
    hand:             list[str]   # 4 slots — confirmed or ?
    next_card:        str         # 5th slot — confirmed or ?
    hand_known:       int         # how many hand slots confirmed

    # availability
    available:        list[str]   # cards that CAN be in hand now
    cooling_down:     list[str]   # cards definitely NOT in hand

    # meta
    total_plays:      int
    events:           list


# ── Cycle Tracker ─────────────────────────────────────────────────────────────
class CycleTracker:
    """
    Tracks opponent card cycle using the availability model.

    Usage:
        tracker = CycleTracker()
        tracker.card_played("hog-rider", timestamp=5.0, position=(300,200))
        state = tracker.get_state()
        print(state.hand)
        print(state.next_card)
        print(state.available)
        print(state.cooling_down)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Call at the start of each new game."""
        # ordered list of unique cards in discovery order
        self._unique_deck:  list[str]       = []

        # full play history in order (includes repeats)
        # each entry = card_name
        self._play_history: list[str]       = []

        # maps card_name → play_index of its most recent play
        self._last_played_at: dict[str,int] = {}

        self._total_plays:  int             = 0
        self._events:       list            = []

        print("[CycleTracker] Reset — ready for new game.")

    # ── public API ────────────────────────────────────────────────────────
    def card_played(self,
                    card_name: str,
                    timestamp: float,
                    position:  tuple = (0, 0)):
        """
        Call every time opponent deploys a card.

        Args:
            card_name : e.g. "hog-rider", "giant", "fireball"
            timestamp : seconds since game start
            position  : (x, y) screen coordinates of detection
        """
        card_name = card_name.strip().lower()
        self._total_plays += 1

        # record event
        event = CardEvent(
            card_name  = card_name,
            timestamp  = timestamp,
            position   = position,
            play_index = self._total_plays,
        )
        self._events.append(event)
        self._play_history.append(card_name)
        self._last_played_at[card_name] = self._total_plays

        # register new unique card
        if card_name not in self._unique_deck:
            self._unique_deck.append(card_name)
            print(f"[CycleTracker] New card: '{card_name}' "
                  f"({len(self._unique_deck)}/{DECK_SIZE})")

        self._log_state()

    def get_state(self) -> CycleState:
        deck             = self._build_deck()
        available, cooling = self._get_availability()
        hand, next_card  = self._calculate_hand(available, cooling)
        hand_known       = sum(1 for c in hand if c != UNKNOWN)

        return CycleState(
            deck             = deck,
            cards_discovered = len(self._unique_deck),
            cycle_complete   = self._is_complete(),
            hand             = hand,
            next_card        = next_card,
            hand_known       = hand_known,
            available        = available,
            cooling_down     = cooling,
            total_plays      = self._total_plays,
            events           = list(self._events),
        )

    @property
    def cycle_complete(self) -> bool:
        return self._is_complete()

    # ── internal ──────────────────────────────────────────────────────────
    def _is_complete(self) -> bool:
        return len(self._unique_deck) == DECK_SIZE

    def _build_deck(self) -> list[str]:
        """8-slot deck in discovery order, ? for unknowns."""
        deck = list(self._unique_deck)
        while len(deck) < DECK_SIZE:
            deck.append(UNKNOWN)
        return deck

    def _get_availability(self) -> tuple[list[str], list[str]]:
        """
        Split known cards into available vs cooling_down.

        A card is COOLING DOWN if it was played within the last
        COOLDOWN_PLAYS (4) plays — meaning it hasn't had 4 other
        cards played after it yet.

        A card is AVAILABLE if:
          - It has been played before AND enough plays have passed
          - OR it has never been played yet (we just don't know
            its position in the cycle)
        """
        available    = []
        cooling_down = []

        for card in self._unique_deck:
            last_play = self._last_played_at.get(card, 0)
            plays_since = self._total_plays - last_play

            if plays_since < COOLDOWN_PLAYS:
                # played too recently — definitely NOT in hand
                cooling_down.append(card)
            else:
                # enough plays have passed — CAN be in hand
                available.append(card)

        return available, cooling_down

    def _calculate_hand(self,
                        available: list[str],
                        cooling:   list[str]) -> tuple[list[str], str]:
        """
        Calculate hand and next card.

        Phase 1 — cycle incomplete (< 8 unique cards seen):
          Fill hand slots with available cards.
          Remaining slots = ? (could be undiscovered cards
          or available cards we haven't placed yet)

        Phase 2 — cycle complete (all 8 seen):
          Use play history to determine exact rotation position.
          Hand and next card are always exact.
        """
        if self._is_complete():
            return self._exact_hand()
        else:
            return self._partial_hand(available)

    def _exact_hand(self) -> tuple[list[str], str]:
        """
        Phase 2 — all 8 cards known.
        Use play history to find exact cycle position.

        The cycle order = unique_deck in discovery order.
        The cycle pointer = position right after the last played card.
        """
        deck = self._unique_deck   # 8 cards in fixed cycle order
        n    = len(deck)

        # find position of last played card in deck
        last_card    = self._play_history[-1]
        last_idx     = deck.index(last_card)

        # hand starts one position after last played
        hand = []
        for i in range(HAND_SIZE):
            idx = (last_idx + 1 + i) % n
            hand.append(deck[idx])

        # next card = 5th position
        next_idx  = (last_idx + 1 + HAND_SIZE) % n
        next_card = deck[next_idx]

        return hand, next_card

    def _partial_hand(self,
                      available: list[str]) -> tuple[list[str], str]:
        """
        Phase 1 — cycle incomplete.
        Fill hand with available cards, rest are ?
        We can't know exact slot order so we list them without order.
        """
        hand = []

        # fill up to HAND_SIZE with available cards
        for card in available[:HAND_SIZE]:
            hand.append(card)

        # remaining slots unknown
        while len(hand) < HAND_SIZE:
            hand.append(UNKNOWN)

        # next card — the card that becomes available on the NEXT play
        # = play_history[total_plays - COOLDOWN_PLAYS]
        next_idx  = self._total_plays - COOLDOWN_PLAYS
        next_card = self._play_history[next_idx] if 0 <= next_idx < len(self._play_history) else UNKNOWN

        return hand, next_card

    def _log_state(self):
        state = self.get_state()
        print(f"[CycleTracker] Play #{state.total_plays}")
        print(f"[CycleTracker] Cooling down : {state.cooling_down}")
        print(f"[CycleTracker] Available    : {state.available}")
        print(f"[CycleTracker] Hand         : {state.hand} "
              f"({state.hand_known}/4 known)")
        print(f"[CycleTracker] Next         : {state.next_card}")
        if state.cycle_complete:
            print(f"[CycleTracker] ✅ Cycle complete — predictions exact!")


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Cycle Tracker — Availability Model Test")
    print("=" * 60)

    tracker = CycleTracker()

    # simulate a real game
    # actual deck order (unknown to tracker):
    # hog-rider(1) fireball(2) skeletons(3) cannon(4)
    # ice-spirit(5) musketeer(6) log(7) ice-golem(8)
    plays = [
        # discovery phase
        ("hog-rider",   5.0,  (300, 200)),  # play 1
        ("fireball",   12.0,  (320, 150)),  # play 2
        ("skeletons",  18.0,  (290, 210)),  # play 3
        ("cannon",     24.0,  (310, 230)),  # play 4
        ("ice-spirit", 30.0,  (305, 195)),  # play 5
        # hog-rider available again (4 cards played since play 1)
        ("musketeer",  38.0,  (280, 180)),  # play 6
        ("log",        45.0,  (315, 160)),  # play 7
        ("ice-golem",  52.0,  (295, 220)),  # play 8 ← all 8 discovered!
        # cycle complete — exact predictions from here
        ("hog-rider",  60.0,  (300, 200)),  # play 9
        ("fireball",   68.0,  (320, 150)),  # play 10
        ("skeletons",  74.0,  (290, 210)),  # play 11
    ]

    for card, ts, pos in plays:
        print(f"\n{'─'*60}")
        print(f"  ▶ Opponent plays: [{card}]  t={ts}s")
        print(f"{'─'*60}")
        tracker.card_played(card, ts, pos)
        state = tracker.get_state()

        print(f"\n  Deck      : {state.deck}")
        print(f"  Cooling   : {state.cooling_down}")
        print(f"  Available : {state.available}")
        print(f"  Hand      : {state.hand}")
        print(f"  Next      : {state.next_card}")
        print(f"  Complete  : {state.cycle_complete}")