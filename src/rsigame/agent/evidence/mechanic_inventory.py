# Copyright 2026 The RSIGame authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Explorer v1 — mechanic inventory.

See docs/explorer_design.md §2 and §10. v1 is still LLM-free: the inventory is
derived from sources that already exist, in this priority order:

  1. runtime scene introspection  -- the strongest signal, because it describes
     what the built game ACTUALLY has (a `towersGroup`, a `cursorGridX`, a set
     of interactive Rectangles), not what a design doc claims it has.
  2. on-screen HUD / instruction text -- how the game tells a player to operate
     it, e.g. hajimi's "1-4 choose . CLICK place/upgrade . SPACE wave".
  3. gameConfig.json / GAME_DESIGN.md -- declared intent.

Why detection is runtime-first: the four tower-defense games in this corpus
implement "place a defender" three different ways. `field_commander` uses
clickable type buttons plus a scene-level pointer handler over a
`towerSlotGroup`; `garden_guard` and `signal_relay` expose ZERO interactive
objects and drive a KEYBOARD cursor (`cursorGridX`/`cursorGridY`/`cursorKeys`);
hajimi announces "1-4 choose . CLICK place". A hardcoded "tower defense = click
a cell" rule gets three of the four wrong -- which is exactly the assumption
that produced a day of false "this mechanic is broken" conclusions.

Each mechanic carries a SUCCESS PREDICATE evaluated against real scene state;
the substrate never decides success on its own (design §3.5).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

# --- status taxonomy (design doc §1) ----------------------------------------




# --- declarative, executable effects ----------------------------------------
#
# A mechanic's success used to be a sentence that only a hand-written branch in
# the evaluator could act on ("if mech.id == 'place_defender': ..."). That does
# not scale: `base_takes_damage` declared `core_effect = "KEEP decreases"` and
# still evaluated to "no predicate implemented", because nobody had written its
# branch. Every newly discovered mechanic would need one more branch, which caps
# exactly the generality v2 exists to provide.
#
# So an effect is DATA the evaluator can execute against any before/after pair:
#     Effect(field="scene.towers", op="increases")
#     Effect(field="hud.KEEP",     op="decreases")
#     Effect(field="hud_token.FOCUS", op="changes")
#
# Field namespaces:
#   scene.<key>      -- numeric key of the runtime scene-state probe
#   hud.<LABEL>      -- number parsed after LABEL in the HUD text
#   hud_token.<LABEL>-- the WORD after LABEL (e.g. FOCUS CLEAR -> FOCUS ACTIVE)









# --- scene-state probe every predicate reads --------------------------------








def hud_numbers(hud_text: str) -> dict:
    """Every `LABEL <number>` pair on screen. The resource a build spends is
    almost always one of these, and which one differs per game (GOLD / SUN /
    SEEDS / Kibble), so it is discovered rather than assumed."""
    out = {}
    toks = (hud_text or "").replace("/", " / ").split()
    for i, w in enumerate(toks[:-1]):
        lab = re.sub(r"[^A-Za-z]", "", w).upper()
        nxt = toks[i + 1]
        if not lab:
            continue
        # A "label" whose PREVIOUS token is a bare number is a unit suffix, not a
        # label: in "35 K 2 SIAMESE" the K belongs to the 35 and the 2 belongs to
        # the next entry. Reading it as K=2 invents a resource the game does not
        # display. One letter is never a HUD label worth trusting either.
        if len(lab) < 2:
            continue
        # The unit-suffix guard used to reject ANY label preceded by a bare
        # number, which is the normal HUD layout, not an exception:
        #
        #   KEEP 18 / 18 GOLD 170 WAVE 1 / 4 SCORE 0
        #
        # gave `{KEEP: 18}` and dropped GOLD, WAVE and SCORE, because each of
        # them follows a value. Measured on field_commander, whose HUD plainly
        # shows GOLD 170. A unit suffix is SHORT -- KG, MS, PX -- and one-letter
        # labels are already rejected above, so bound the guard by length
        # instead of firing it on every second label on the bar.
        if len(lab) <= 2 and i > 0 and re.match(r"^\d+$", toks[i - 1]):
            continue
        m = re.match(r"^(\d+)$", nxt)
        if m:
            out.setdefault(lab, int(m.group(1)))
        else:
            # forms like "SEEDS20/20" or "SUN150" glued together
            m2 = re.match(r"^([A-Za-z]+)(\d+)", w)
            if m2:
                out.setdefault(m2.group(1).upper(), int(m2.group(2)))
    return out






