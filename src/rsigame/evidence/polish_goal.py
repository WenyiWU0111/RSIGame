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
"""Turn the polish reader's finding into one thing to ask for this round.

The old framework did this inside the round planner: polish evidence arrived
beside the open requirements under the heading "a reader judged how finished
the game looks and feels", and the planner wrote a concrete, ranked goal from
it. That step is what made polish useful -- the reader says what is weak, and
something else has to decide what to ask for about it.

Pasting the reader's finding straight into the repair packet, which is what
this replaces, loses two things. It is not a request, so nobody has decided
what would count as doing it. And nobody checks it against what was already
asked: four rounds running on one game were handed the same sentence about
thirty shipped images and three loaded ones.

So this call gets the counted facts as well as the reader's prose. The facts
are the part that can change between rounds and be seen to have changed --
`images_used` going 3 to 5 while `images_present` goes 30 to 150 says the last
round generated art and never wired it up, and the right ask is then to wire it
up, not to generate more.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUN))

_PROMPT = """A game is being improved a round at a time. A reader looks at every
build and says how finished it looks and feels. Your job is to turn what it
said into ONE thing to ask for this round.

====================
WHAT THE READER SAYS ABOUT THIS BUILD
====================
{reader}

====================
WHAT IS ACTUALLY IN THE BUILD, counted off the source
====================
{facts}

====================
WHAT WAS ALREADY ASKED FOR, oldest first, verbatim
====================
{history}

====================
WHAT TO DECIDE
====================
Read the previous asks before you write. A request reworded is the same
request, and repeating it is how a round gets spent on a sentence the last
three rounds already produced.

The counts above are the part you can check. If art was generated and the code
still does not load it, the thing to ask for is the wiring, not more art -- a
file nobody loads changes nothing on screen and costs real money to make.

Ask for something a player would see. Not "improve the visuals": say what
should look different and where. One or two sentences.

Reply with strict JSON and nothing else:

{{"goal": "<what to do this round>",
 "why_now": "<what in the evidence above makes this the thing to ask>",
 "repeats_earlier": true|false,
 "dimension": "visual|feedback"}}
"""


def _facts_text(f: dict) -> str:
    if not f:
        return '  (not counted)'
    return '\n'.join([
        f"  image files present            {f.get('images_present')}",
        f"  referenced by code             {f.get('images_used')}",
        f"  never referenced               {f.get('images_unused')}"
        + (f"   e.g. {', '.join((f.get('unused_names') or [])[:6])}"
           if f.get('unused_names') else ''),
        f"  audio files present            {f.get('audio_present')}",
        f"  draws that place authored art  {f.get('texture_draws')}",
        f"  draws that are primitives      {f.get('primitive_draws')}",
        f"  share of drawing using art     {(f.get('authored_share') or 0) * 100:.0f}%",
    ])


def ask(reader: dict, facts: dict, history: list[str], *,
        model: str | None = None) -> dict:
    """One request for this round. `history` is previous goals, verbatim."""
    import rsigame.verify.post_verify as PV

    hr = reader.get('headroom') or {}
    mi = reader.get('main_issue') or {}
    rd = '\n'.join(f"  {d}: headroom {hr.get(d, '?')} -- {str(mi.get(d, ''))[:400]}"
                   for d in ('visual', 'feedback'))
    # The reader's own answer to "what would you do", which was being dropped:
    # it was read under the wrong key for 120 rounds. It is one reading of the
    # same evidence, not an instruction -- the history and the counts below can
    # still say it is the wrong ask this round.
    if reader.get('goal'):
        rd += (f"\n\n  what the reader itself would do next ({reader.get('focus') or '?'}): "
               f"{str(reader['goal'])[:400]}"
               + (f"\n  because: {str(reader.get('why'))[:200]}"
                  if reader.get('why') else ''))
    body = _PROMPT.format(
        reader=rd or '  (nothing reported)', facts=_facts_text(facts),
        history='\n'.join(f'  round {i}: {g}' for i, g in history) or '  (none yet)')
    from rsigame.controller.stage_brief import text as _stage_brief
    sb = _stage_brief()
    if sb:
        # The ask, not the reading: the reader's headroom above was produced
        # without the brief and stays that way.
        body = body.replace('====================\nWHAT TO DECIDE', sb + (
            '\nWhen the reader\'s findings and the counts support more than one ask, choose '
            'the one that serves the stage objective.\n\n'
            '====================\nWHAT TO DECIDE'), 1)
    parsed, _ = PV._ask([{'role': 'user', 'content': body}],
                        model=model or os.environ.get('RSIGAME_MODELS_LOOP_MODEL'), timeout=120.0)
    if not isinstance(parsed, dict):
        return {'error': 'the reading returned nothing after three tries'}
    d = parsed.get('dimension')
    return {'goal': str(parsed.get('goal') or '')[:400],
            'why_now': str(parsed.get('why_now') or '')[:300],
            'repeats_earlier': bool(parsed.get('repeats_earlier')),
            'dimension': d if d in ('visual', 'feedback') else None,
            'headroom': hr}
