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
"""M2 -- one demo, two builds, what changed.

This is the only place in the monitor that asks a model anything about pixels,
and it is deliberately narrow. It answers "what changed in this scenario",
never "what should we do next": direction and budget belong to the global
stage, and mixing them here would let development intent bias what the model
claims to see. The comparator is therefore never told which round produced the
new build, what the repair was trying to do, or how many rounds have been spent.

Two properties from M1 shape the call:

  * the evidence is already action-aligned, so a pair of frames can be labelled
    with the input that produced it rather than with an index;
  * independent replays of one behaviour collapse to a mean absolute difference
    below 2 at 128x72 while a real behavioural change measures above 7, so a
    deterministic screen can pick which events are worth showing. Sending every
    event costs images and invites the model to narrate noise.

The screen only ever *adds* the events it is confident about; a demo whose
events all look unchanged still sends its first and last event, so "nothing
changed" remains a conclusion the model reaches rather than one the harness
forces by showing it nothing.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

RUN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUN))

CHANGE = ('better', 'same', 'worse', 'unobserved')
MAGNITUDE = ('major', 'medium', 'minor')
DIRECTION = ('correctness', 'completeness', 'feedback', 'presentation')

# From M1: two replays of one behaviour measure 0.26-1.91 apart; a real change
# measures 7.5-10.8. The gate sits in the empty band between them, nearer the
# noise floor so a small real change is not screened out.
CHANGED_AT = 3.0
SCREEN_SIZE = (128, 72)
MAX_EVENTS = 6            # per demo, per side; configurable


def frame_delta(a: str | None, b: str | None) -> float | None:
    if not a or not b or not Path(a).is_file() or not Path(b).is_file():
        return None
    x = np.asarray(Image.open(a).convert('L').resize(SCREEN_SIZE), dtype=np.float32)
    y = np.asarray(Image.open(b).convert('L').resize(SCREEN_SIZE), dtype=np.float32)
    return float(np.abs(x - y).mean())


def screen(pair, max_events: int = MAX_EVENTS) -> list[dict]:
    """Which events to show, and why.

    Returns every event with its measured deltas so the caller can log the
    whole screen, with `show` marking the ones that go to the model.
    """
    rows = []
    for e in pair.events:
        db = frame_delta(e.old_before, e.new_before)
        da = frame_delta(e.old_after[0] if e.old_after else None,
                         e.new_after[0] if e.new_after else None)
        rows.append({'n': e.n, 'what': e.what, 'trace_frame': e.trace_frame,
                     'delta_before': db, 'delta_after': da,
                     'delta': max([v for v in (db, da) if v is not None] or [0.0]),
                     'show': False})
    changed = [r for r in rows if r['delta'] >= CHANGED_AT]
    changed.sort(key=lambda r: -r['delta'])
    for r in changed[:max_events]:
        r['show'] = True
    if not any(r['show'] for r in rows) and rows:
        # Nothing crossed the gate. Show the ends anyway: "unchanged" must be
        # something the model concludes, not something we arrange.
        rows[0]['show'] = True
        rows[-1]['show'] = True
    return rows


_PROMPT = """Two builds of the same game were replayed with the SAME fixed
input script. Your job is to say what changed in this scenario, and nothing
else.

SCENARIO
{scenario}

WHAT THE TASK REQUIRES, for the items this scenario can speak to
{items}

WHAT YOU ARE LOOKING AT
The script is fixed, so both builds received the same inputs at the same
moments. For each moment below you get the OLD build just before and just
after the input, then the NEW build at the same two moments. A frame labelled
FINAL is the last frame of the recording.

Two things in every frame are not part of the game. The mouse pointer, drawn
by the recorder as a small cross, is never a defect. And the recording is not
frame-exact: replaying one build twice cuts the same moment up to a frame or
two apart, which by itself makes a score read one point different, an
animation look half a beat behind, or a falling piece sit one row lower. That
is the recorder, not the build.

You get two frames after each input for exactly this reason. A difference that
is gone by the second frame, or that later moments undo, is timing. A
difference that is still there at the next moment and at the FINAL frame is
the build. Say `same` for the first kind.

{events}

RULES
- Compare only what these frames support.
- Do not assume the newer build is better, and do not assume a
  difference you cannot explain is a defect. Both tilts are errors.
- If a behaviour is not exercised here, its criterion is `unobserved`. Not
  seeing something is not seeing it missing.
- Explicit task requirements outrank optional polish.
- Report anything that got worse separately, however small the rest.
- Judge relative change. Do not score either build.
- Do not propose code edits, repairs, or next steps. That is not your job.

Reply with strict JSON and nothing else:

{{"changes": [{{"criterion": "<requirement id, or a short phrase for a general
                 quality>",
               "direction": "correctness|completeness|feedback|presentation",
               "change": "better|same|worse|unobserved",
               "magnitude": "major|medium|minor",
               "evidence": "<which moment above, e.g. 'step 4 after'>",
               "note": "<one sentence, what you saw>"}}],
 "regressions": [{{"what": "<one sentence>", "evidence": "<which moment>",
                  "severity": "major|medium|minor"}}],
 "remaining_issues": [{{"direction": "correctness|completeness|feedback|presentation",
                       "severity": "major|medium|minor",
                       "issue": "<one sentence>"}}]}}
"""


def _items_text(items: list[dict]) -> str:
    if not items:
        return '(no task requirements were matched to this scenario)'
    return '\n'.join(f"  {it.get('id')}  {it.get('requirement')}" for it in items)


def compare(pair, items: list[dict], *, model: str | None = None,
            max_events: int = MAX_EVENTS) -> dict:
    """One reading of one demo pair. Returns the parsed record plus its screen."""
    from rsigame.agent.evolve.verifier_agent import _data_uri
    import rsigame.verify.post_verify as PV

    rows = screen(pair, max_events)
    shown = [r for r in rows if r['show']]
    by_n = {e.n: e for e in pair.events}

    lines, content_imgs = [], []
    for r in shown:
        e = by_n[r['n']]
        lines.append(f"MOMENT {e.n} -- {e.what} (script frame {e.trace_frame})")
        for who, before, after in (('OLD', e.old_before, e.old_after),
                                   ('NEW', e.new_before, e.new_after)):
            if before:
                content_imgs.append((f'moment {e.n} {who} before', before))
            for j, f in enumerate(after[:2], start=1):
                content_imgs.append((f'moment {e.n} {who} after {j}', f))
        lines.append(f"  images: moment {e.n} OLD before, after 1, after 2; "
                     f"then moment {e.n} NEW before, after 1, after 2")
    if pair.old_final and pair.new_final:
        lines.append('FINAL frame of the recording, OLD then NEW')
        content_imgs.append(('FINAL OLD', pair.old_final))
        content_imgs.append(('FINAL NEW', pair.new_final))

    body = _PROMPT.format(scenario=pair.scenario or pair.demo_id,
                          items=_items_text(items), events='\n'.join(lines))
    content = [{'type': 'text', 'text': body}]
    for label, fp in content_imgs:
        p = Path(fp)
        if not p.is_file():
            continue
        content.append({'type': 'image_url', 'image_url': {'url': _data_uri(p)}})
        content.append({'type': 'text', 'text': f'({label})'})

    out = {'demo_id': pair.demo_id, 'scenario': pair.scenario,
           'old': pair.old_label, 'new': pair.new_label,
           'screen': rows, 'n_shown': len(shown), 'n_images': len(content_imgs)}
    try:
        parsed, raw = PV._ask([{'role': 'user', 'content': content}],
                              model=model or os.environ.get('RSIGAME_MODELS_LOOP_MODEL'))
    except Exception as exc:
        out['error'] = f'{type(exc).__name__}: {str(exc)[:200]}'
        return out
    if not isinstance(parsed, dict):
        out['error'] = 'the reading returned nothing after three tries'
        return out
    out.update(validate(parsed))
    return out


def validate(d: dict) -> dict:
    """Coerce to the schema and record what had to be dropped.

    A field outside its enum is dropped rather than guessed at: a `change` of
    "slightly better" silently mapped to `better` would be the harness
    inventing a reading.
    """
    bad = []

    def enum(v, allowed, where):
        if v in allowed:
            return v
        bad.append(f'{where}={v!r}')
        return None

    changes = []
    for i, c in enumerate(d.get('changes') or []):
        if not isinstance(c, dict):
            bad.append(f'changes[{i}] not an object')
            continue
        ch = enum(c.get('change'), CHANGE, f'changes[{i}].change')
        dr = enum(c.get('direction'), DIRECTION, f'changes[{i}].direction')
        mg = enum(c.get('magnitude'), MAGNITUDE, f'changes[{i}].magnitude')
        if ch is None:
            continue
        changes.append({'criterion': str(c.get('criterion') or '')[:120],
                        'direction': dr, 'change': ch, 'magnitude': mg,
                        'evidence': str(c.get('evidence') or '')[:120],
                        'note': str(c.get('note') or '')[:300]})
    regs = []
    for i, r in enumerate(d.get('regressions') or []):
        if not isinstance(r, dict):
            bad.append(f'regressions[{i}] not an object')
            continue
        regs.append({'what': str(r.get('what') or '')[:300],
                     'evidence': str(r.get('evidence') or '')[:120],
                     'severity': enum(r.get('severity'), MAGNITUDE,
                                      f'regressions[{i}].severity')})
    rem = []
    for i, r in enumerate(d.get('remaining_issues') or []):
        if not isinstance(r, dict):
            continue
        rem.append({'direction': enum(r.get('direction'), DIRECTION,
                                      f'remaining_issues[{i}].direction'),
                    'severity': enum(r.get('severity'), MAGNITUDE,
                                     f'remaining_issues[{i}].severity'),
                    'issue': str(r.get('issue') or '')[:300]})
    return {'changes': changes, 'regressions': regs, 'remaining_issues': rem,
            'schema_problems': bad}
