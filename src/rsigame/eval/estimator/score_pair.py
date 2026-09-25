#!/usr/bin/env python
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
"""One demo, two builds, sixteen criteria -- scored in a single call.

BOTH SIDES IN ONE CALL, because the quantity that matters is the difference
and a difference judged across two conversations is two judgements subtracted.
The model sees the same frames it would see anyway, labelled `old:` and `new:`,
and answers each criterion twice.

WHAT THE MODEL IS NOT ASKED. It never says "improved" or "regressed" -- that
follows from the two numbers and the harness does the subtraction. It is never
told which round produced which build, what anyone was trying to fix, how many
rounds have been spent, or what the official score was. Measurement is kept
away from control, the same separation `monitor/comparator.py` was built with
and for the same reason.

UNOBSERVED IS AN ANSWER. A criterion the frames never exercise scores `null`
on that side, and `null` is never 0 -- a game is not bad at something it was
never asked to do in front of the camera.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
sys.path.insert(0, str(RUN))

_PROMPT = """Two builds of the same game were replayed with the SAME fixed
input script. You are scoring both against a fixed checklist.

Nobody has told you which build is newer, what anyone was trying to change, or
whether anything was supposed to improve -- and you should not guess. Score
what the frames show.

====================
WHAT YOU ARE LOOKING AT
====================
Each build's recording was sampled evenly from its first frame to its
last. EVERY IMAGE IS PRECEDED BY ITS LABEL, in square brackets, like
`[old: frame 4/16 (20% in)]` -- the label names the image that comes AFTER it.
They arrive in pairs: the two frames numbered 4 sit at the same point of the
same fixed script, so they are the two builds answering the same moment.

THE SAME MOMENT IS NOT THE SAME SITUATION. A build that starts drawing late,
crashes, or stalls will be somewhere else in the game at 20% than a build that
did not, even though both received the same inputs. If one side shows nothing
-- a black frame, an engine logo, a frozen screen -- say what you see and
score it; do not assume the feature was removed, and do not read the other
side's picture into it.

Two things in every frame are not part of the game: the mouse pointer, which
the recorder draws as a small cross, and the fact that a recording is not
frame-exact -- the same build replayed twice can cut a moment a frame or two
apart, which alone can make a counter read one different.

Scenario: {scenario}

====================
THE CHECKLIST
====================
Score EVERY item for BOTH builds.

{items}

====================
HOW TO SCORE
====================
- Use only 0.0, 0.5 or 1.0, matching the anchors above.
- Use null when THESE frames never exercise the item. Null is not zero: a game
  is not bad at something the script never asked it to do. If the old frames
  show it and the new ones do not, score old and leave new null.
- Judge the two builds independently. Do not decide one is better and then
  score to match; score each side from its own frames.
- Cite the frames you used, by their labels.
- Do not say whether anything improved. Give two scores; the comparison is
  computed elsewhere.

Reply with strict JSON and nothing else:

{{"items": [{{"id": "F1",
             "old_score": 0.0,
             "new_score": 1.0,
             "old_evidence_refs": ["old: frame 7/16 (40% in)"],
             "new_evidence_refs": ["new: frame 7/16 (40% in)"],
             "reason": "<one sentence about what the frames show>"}}, ...]}}
"""


def _items_text(rubric: dict) -> str:
    out = []
    for it in rubric['items']:
        out.append(f"{it['id']}  {it['criterion']}")
        for k in ('0.0', '0.5', '1.0'):
            out.append(f"      {k}  {it['anchors'][k]}")
    return '\n'.join(out)


def score(demo: dict, rubric: dict, *, model: str | None = None) -> dict:
    """One call. Returns the validated per-item rows plus what went wrong."""
    from rsigame.eval.estimator.rubric import ask
    if not demo.get('content'):
        return {'demo_id': demo.get('demo_id'), 'items': [],
                'error': 'no frames', 'problems': demo.get('problems') or []}
    body = _PROMPT.format(scenario=demo.get('scenario') or demo['demo_id'],
                          items=_items_text(rubric))
    content = [{'type': 'text', 'text': body}] + list(demo['content'])
    parsed, raw = ask([{'role': 'user', 'content': content}], max_tokens=6000)
    if not isinstance(parsed, dict):
        return {'demo_id': demo['demo_id'], 'items': [],
                'error': f'returned no parsable JSON: {str(raw)[:160]}'}
    known = {it['id'] for it in rubric['items']}
    rows, bad = [], []
    seen = set()
    for it in (parsed.get('items') or []):
        if not isinstance(it, dict):
            continue
        i = it.get('id')
        if i not in known:
            bad.append(f'unknown id {i!r}')
            continue
        if i in seen:
            bad.append(f'duplicate {i}')
            continue
        seen.add(i)

        def num(v, where):
            if v is None:
                return None
            try:
                f = float(v)
            except (TypeError, ValueError):
                bad.append(f'{i}.{where}={v!r} not a number')
                return None
            if f not in (0.0, 0.5, 1.0):
                # SNAPPED, AND RECORDED. The anchors are three points; a 0.7 is
                # the model declining the scale, not a finer reading of it.
                near = min((0.0, 0.5, 1.0), key=lambda x: abs(x - f))
                bad.append(f'{i}.{where}={f} not one of the three levels, mapped to {near}')
                return near
            return f

        rows.append({
            'id': i,
            'old_score': num(it.get('old_score'), 'old'),
            'new_score': num(it.get('new_score'), 'new'),
            'old_evidence_refs': [str(x)[:120] for x in
                                  (it.get('old_evidence_refs') or [])][:4],
            'new_evidence_refs': [str(x)[:120] for x in
                                  (it.get('new_evidence_refs') or [])][:4],
            'reason': str(it.get('reason') or '')[:300]})
    for i in sorted(known - seen):
        # Silence is not `unobserved`. An item the reader skipped was not
        # looked at, and saying so keeps it out of the coverage numerator.
        rows.append({'id': i, 'old_score': None, 'new_score': None,
                     'old_evidence_refs': [], 'new_evidence_refs': [],
                     'reason': '(the reading did not mention it)'})
        bad.append(f'{i} unanswered')
    order = [it['id'] for it in rubric['items']]
    rows.sort(key=lambda r: order.index(r['id']))
    return {'demo_id': demo['demo_id'], 'items': rows,
            'schema_problems': bad,
            'n_images': sum(1 for c in demo['content']
                            if c.get('type') == 'image_url')}
