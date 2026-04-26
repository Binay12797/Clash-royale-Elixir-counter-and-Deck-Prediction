"""
template_matcher.py
===================
Loads card templates from templates/cards/ and matches
them against a frame region using cv2 template matching.

Place this file at: detection/template_matcher.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 CARD TEMPLATES NEEDED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Download card icons (the small square icons
 shown in the game UI, NOT full artwork) and
 place them in:

   templates/cards/<card_name>.png

 Naming rules:
   - all lowercase
   - spaces → underscores
   - must match "template_file" in card_database.json

 Priority cards to get FIRST (most common in meta):
 ── Troops ──────────────────────────────────
   hog_rider.png          knight.png
   giant.png              musketeer.png
   valkyrie.png           mini_pekka.png
   witch.png              wizard.png
   pekka.png              mega_knight.png
   goblin_gang.png        barbarians.png
   minions.png            minion_horde.png
   archers.png            skeleton_army.png
   prince.png             dark_prince.png
   miner.png              bandit.png
   night_witch.png        electro_wizard.png
   baby_dragon.png        inferno_dragon.png
   balloon.png            lava_hound.png
   royal_giant.png        elite_barbarians.png
   bowler.png             executioner.png
   firecracker.png        bats.png
   skeletons.png          ice_spirit.png
   fire_spirit.png        electro_spirit.png
   heal_spirit.png        ice_golem.png
   goblin_giant.png       electro_giant.png
   golem.png              three_musketeers.png
   wall_breakers.png      skeleton_barrel.png
   dart_goblin.png        fisherman.png
   magic_archer.png       sparky.png
   royal_ghost.png        ice_wizard.png
   princess.png           guards.png
   cannon_cart.png        flying_machine.png
   zappies.png            rascals.png
   royal_hogs.png         royal_recruits.png
   bomber.png             spear_goblins.png
   goblins.png            mega_minion.png
   electro_dragon.png     battle_healer.png
   elixir_golem.png       skeleton_dragons.png

 ── Champions ───────────────────────────────
   golden_knight.png      skeleton_king.png
   archer_queen.png       monk.png
   little_prince.png      boss_bandit.png

 ── Spells ──────────────────────────────────
   fireball.png           arrows.png
   zap.png                rocket.png
   lightning.png          freeze.png
   poison.png             goblin_barrel.png
   earthquake.png         graveyard.png
   tornado.png            rage.png
   clone.png              mirror.png
   barbarian_barrel.png   royal_delivery.png
   the_log.png            snowball.png
   giant_snowball.png

 ── Buildings ───────────────────────────────
   cannon.png             mortar.png
   tesla.png              inferno_tower.png
   bomb_tower.png         x_bow.png
   tombstone.png          goblin_hut.png
   barbarian_hut.png      furnace.png
   elixir_collector.png

 You can add templates gradually — the matcher
 will only use cards it finds in the folder.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import cv2
import json
import os
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
TEMPLATES_DIR  = BASE_DIR / "templates" / "cards"
DATABASE_PATH  = BASE_DIR / "data" / "card_database.json"

# ── Matching config ───────────────────────────────────────────────────────────
MATCH_THRESHOLD   = 0.75    # 0-1, higher = stricter. Tune this after testing.
MATCH_SCALES      = [0.8, 0.9, 1.0, 1.1, 1.2]   # handle slight size differences
NMS_OVERLAP       = 0.3     # non-max suppression overlap threshold


@dataclass
class CardMatch:
    card_name:    str
    elixir_cost:  int
    card_type:    str           # troop / spell / building
    deploy_side:  str           # opponent / our / any
    ability:      dict
    confidence:   float
    bbox:         tuple         # (x1, y1, x2, y2) in frame coordinates


@dataclass
class TemplateData:
    card_name:   str
    image:       np.ndarray     # loaded template image
    db_entry:    dict           # full card_database entry


# ── Template Loader ───────────────────────────────────────────────────────────
class TemplateLoader:
    """
    Loads all card templates from templates/cards/ that
    also exist in card_database.json.
    """

    def __init__(self,
                 templates_dir: Path = TEMPLATES_DIR,
                 database_path: Path = DATABASE_PATH):
        self.templates_dir = templates_dir
        self.database_path = database_path
        self.templates: list[TemplateData] = []
        self._db: dict = {}

    def load(self) -> int:
        """
        Load database + templates.
        Returns number of templates successfully loaded.
        """
        self._load_database()
        self._load_templates()
        return len(self.templates)

    def _load_database(self):
        if not self.database_path.exists():
            raise FileNotFoundError(
                f"Card database not found: {self.database_path}\n"
                f"Make sure data/card_database.json exists."
            )
        with open(self.database_path, "r") as f:
            data = json.load(f)
        self._db = data["cards"]
        print(f"[TemplateLoader] Database loaded — {len(self._db)} cards defined.")

    def _load_templates(self):
        if not self.templates_dir.exists():
            print(f"[TemplateLoader] Templates folder not found: {self.templates_dir}")
            print(f"                 Create it and add card PNG images.")
            return

        loaded   = 0
        skipped  = 0
        missing  = 0

        for card_name, db_entry in self._db.items():
            template_file = db_entry.get("template_file", f"{card_name}.png")
            template_path = self.templates_dir / template_file

            if not template_path.exists():
                missing += 1
                continue

            img = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"[TemplateLoader] ⚠ Could not read: {template_path}")
                skipped += 1
                continue

            self.templates.append(TemplateData(
                card_name=card_name,
                image=img,
                db_entry=db_entry,
            ))
            loaded += 1

        print(f"[TemplateLoader] Templates loaded:  {loaded}")
        print(f"[TemplateLoader] Templates missing: {missing}  ← add PNGs to templates/cards/")
        if skipped:
            print(f"[TemplateLoader] Templates skipped (unreadable): {skipped}")

    @property
    def loaded_cards(self) -> list[str]:
        return [t.card_name for t in self.templates]


# ── Template Matcher ──────────────────────────────────────────────────────────
class TemplateMatcher:
    """
    Runs multi-scale template matching on a frame region
    and returns CardMatch results above the confidence threshold.
    """

    def __init__(self,
                 loader: TemplateLoader,
                 threshold: float = MATCH_THRESHOLD,
                 scales: list    = MATCH_SCALES):
        self.loader    = loader
        self.threshold = threshold
        self.scales    = scales

    def match_frame(self,
                    frame: np.ndarray,
                    roi: dict | None = None) -> list[CardMatch]:
        """
        Match all loaded templates against frame (or an ROI within it).

        Args:
            frame: Full BGR frame from window capture.
            roi:   Optional dict with x1,y1,x2,y2 to restrict search area.
                   If None, searches entire frame.

        Returns:
            List of CardMatch sorted by confidence descending.
        """
        if roi:
            region = frame[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]]
            offset = (roi["x1"], roi["y1"])
        else:
            region = frame
            offset = (0, 0)

        all_matches: list[CardMatch] = []

        for template_data in self.loader.templates:
            matches = self._match_template(region, template_data, offset)
            all_matches.extend(matches)

        # suppress duplicates
        all_matches = self._non_max_suppression(all_matches)

        # sort by confidence
        all_matches.sort(key=lambda m: m.confidence, reverse=True)
        return all_matches

    def match_single_slot(self,
                          frame: np.ndarray,
                          slot_roi: dict) -> CardMatch | None:
        """
        Match against a single card slot (e.g. one deploy zone box).
        Returns the best match or None if below threshold.
        """
        matches = self.match_frame(frame, roi=slot_roi)
        return matches[0] if matches else None

    # ── internal ──────────────────────────────────────────────────────────
    def _match_template(self,
                        region: np.ndarray,
                        template_data: TemplateData,
                        offset: tuple) -> list[CardMatch]:
        results = []
        tmpl    = template_data.image
        rh, rw  = region.shape[:2]

        for scale in self.scales:
            # resize template to this scale
            th = int(tmpl.shape[0] * scale)
            tw = int(tmpl.shape[1] * scale)

            if th > rh or tw > rw:
                continue    # template larger than region — skip

            scaled = cv2.resize(tmpl, (tw, th))
            result = cv2.matchTemplate(region, scaled, cv2.TM_CCOEFF_NORMED)
            locs   = np.where(result >= self.threshold)

            for pt in zip(*locs[::-1]):   # (x, y)
                x1 = pt[0] + offset[0]
                y1 = pt[1] + offset[1]
                x2 = x1 + tw
                y2 = y1 + th
                conf = float(result[pt[1], pt[0]])

                db = template_data.db_entry
                results.append(CardMatch(
                    card_name   = template_data.card_name,
                    elixir_cost = db["elixir_cost"],
                    card_type   = db["type"],
                    deploy_side = db["deploy_side"],
                    ability     = db["ability"],
                    confidence  = conf,
                    bbox        = (x1, y1, x2, y2),
                ))

        return results

    def _non_max_suppression(self, matches: list[CardMatch]) -> list[CardMatch]:
        """Remove overlapping matches keeping highest confidence."""
        if not matches:
            return []

        boxes = np.array([m.bbox for m in matches], dtype=np.float32)
        scores = np.array([m.confidence for m in matches], dtype=np.float32)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep  = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w   = np.maximum(0.0, xx2 - xx1)
            h   = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            order = order[np.where(iou <= NMS_OVERLAP)[0] + 1]

        return [matches[i] for i in keep]


# ── Debug visualiser ──────────────────────────────────────────────────────────
def draw_matches(frame: np.ndarray, matches: list[CardMatch]) -> np.ndarray:
    """Draw bounding boxes and labels on frame for debugging."""
    canvas = frame.copy()
    for m in matches:
        x1, y1, x2, y2 = m.bbox
        color = {
            "troop":    (0, 255, 0),
            "spell":    (0, 165, 255),
            "building": (255, 0, 255),
        }.get(m.card_type, (255, 255, 255))

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{m.card_name} {m.elixir_cost}E ({m.confidence:.2f})"
        cv2.putText(canvas, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(BASE_DIR))

    print("Loading templates...")
    loader = TemplateLoader()
    count  = loader.load()

    if count == 0:
        print("\n[TEST] No templates loaded yet.")
        print(f"       Add card PNG images to: {TEMPLATES_DIR}")
        print(f"       Then re-run this test.")
    else:
        print(f"\n[TEST] {count} templates ready.")
        print(f"       Loaded cards: {', '.join(loader.loaded_cards)}")

        # test on a screenshot if provided
        if len(sys.argv) > 1:
            img_path = sys.argv[1]
            frame    = cv2.imread(img_path)
            if frame is None:
                print(f"[ERROR] Cannot read image: {img_path}")
            else:
                matcher = TemplateMatcher(loader)
                matches = matcher.match_frame(frame)
                print(f"\n[TEST] Found {len(matches)} match(es):")
                for m in matches:
                    print(f"  {m.card_name:20s} | {m.elixir_cost}E | "
                          f"conf={m.confidence:.3f} | bbox={m.bbox}")

                result = draw_matches(frame, matches)
                cv2.imshow("Template Match Test", result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print("       Pass a screenshot path to test matching:")
            print("       python detection/template_matcher.py path/to/screenshot.png")