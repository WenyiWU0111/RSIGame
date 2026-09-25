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
"""Read one exploration trace and say what is wrong with the build.

This is the half `explore_arm` refuses to do. The agent that played was not
allowed to judge, because an agent asked to find bugs while it plays finds
them -- 140 clicks that placed no tower were once written up as a broken
mechanic and retracted the same day when the clicks turned out to be landing on
an invisible overlay. So judging happens here, once, with the whole session
visible at the same time instead of one turn at a time.

WHAT SEEING THE WHOLE TRACE BUYS. The strongest finding of the first sweep was
not in any single step: platformer-friction answered nine consecutive inputs
with 0.0% of pixels changed -- Enter, Space, a click, a three-second wait, a
held direction, a jump, an action key -- while its state object stayed null.
No step says "the level never constructed". The sequence does, and only a
reader holding all of it can see that.

TWO DIMENSIONS, because the repair paths differ. VISUAL is what the frames
show: something invisible, misplaced, unreadable, or drawn wrong. MECHANISM is
what the readings show: an input that changes nothing, a counter that moves on
its own, a rule the game states and does not keep. A defect can appear in both,
and is reported under the one whose evidence is stronger.

WHAT DECIDES A DEFECT IS EVIDENCE, NOT THE PLAYER'S OWN FILING. The first live
run got this backwards in the safe direction: told never to turn `not_reached`
into a defect, it deferred to the player's framing even where the steps showed
four clicks at four locations with the tower count stuck at zero, and reported
neither of hajimi's two known defects. Burying a finding behind "we did not get
there" costs exactly as much as inventing one -- the first leaves the bug in
the game, the second sends someone to repair working code. The `not_reached`
FIELD still comes back untouched; what the reader may say about the steps is
governed by how much they show.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .verifier_agent import _chat

# How many frames go into the request. The trace holds up to twenty; the ones
# that carry the argument are the ones beside an INPUT, so those are preferred
# and the rest fill in.
MAX_FRAMES_IN = 12





INPUT_ACTIONS = ('click', 'press_key', 'hold_key', 'wait')


def _pick_frames(steps: list, cap: int = MAX_FRAMES_IN) -> list:
    """Frames beside an input first, then the rest, back in step order.

    A frame after `read_text` shows the same screen the previous one did. A
    frame after a click is the one that carries the argument, so when the cap
    bites it should bite the passive ones.
    """
    marked = [(s.get('n'), s.get('marked_frame') or s.get('frame'))
              for s in steps if (s.get('marked_frame') or s.get('frame'))]
    if len(marked) <= cap:
        return marked
    acted = {s.get('n') for s in steps if s.get('action') in INPUT_ACTIONS}
    first = [m for m in marked if m[0] in acted]
    rest = [m for m in marked if m[0] not in acted]
    keep = (first + rest)[:cap]
    return sorted(keep, key=lambda m: m[0] if m[0] is not None else 0)



def click_marker_note(exp: dict, *, default: bool | None = None) -> str:
    """How to describe the click marker on these frames, to whoever reads them.

    From the trace, not from the environment: `CLICK_EVIDENCE` is an env var
    read at import, and a reader re-reading a stored trace -- offline, or under
    another arm's env -- would describe a marker that was never drawn. It did
    exactly that. Every arm's critique was told the crosshair is pink; EV1
    draws a red one with a 3x inset of what sits under the cursor, and the
    critique was never told the inset was there. That inset is the whole answer
    to the confusion that cost horror-tape-archive four repair rounds on a
    working button, so leaving it undescribed wasted the fix.

    Traces written before `click_evidence` was recorded do not say, and nothing
    else in them does -- the per-step note that would have given it away goes
    into the live conversation and is never stored. So a caller who knows which
    arm produced the trace passes `default`; EV1's stored traces are red, every
    other arm's are pink. With neither, this guesses pink, which is right for
    all four arms on disk except EV1.
    """
    flag = (exp or {}).get('click_evidence')
    if flag is None:
        flag = default
    if not flag:
        return ("A pink crosshair is drawn by the harness wherever a click "
                "landed; it is not part of the game.")
    return ("A red crosshair is drawn by the harness wherever a click landed, "
            "with a magnified inset showing what sits under the cursor at 3x. "
            "Neither is part of the game. Read the inset before concluding "
            "anything about hit areas: a click that landed on a caption or on "
            "empty space is a miss, not a broken button.")


def _pace(exp: dict) -> str:
    """How much real time separates the steps -- the fact that was missing.

    A reader with no clock reaches for the only duration in front of it, which
    is the action's own. `hold_key 900ms` then reads as "900 ms passed", and a
    countdown that fell sixteen units between two steps looks like a timer
    running sixty times too fast. It was four games before anyone asked.
    """
    steps = exp.get('steps') or []
    stamped = [s for s in steps if s.get('since_prev_s') is not None]
    if stamped:
        gaps = [s['since_prev_s'] for s in stamped]
        return (f"EACH STEP BELOW CARRIES `since_prev_s`: the REAL seconds "
                f"since the previous step. They run from {min(gaps):.0f}s to "
                f"{max(gaps):.0f}s here, because most of the gap is the time "
                f"taken to decide the next action, not the action itself. Use "
                f"these when judging anything about rates or elapsed time.\n")
    n, wall = exp.get('n_steps') or len(steps), exp.get('wall_s') or 0
    if not (n and wall):
        return ''
    return (f"HOW FAR APART THE STEPS ARE: this session ran {wall:.0f} seconds "
            f"over {n} steps, so roughly {wall / n:.0f} SECONDS OF REAL TIME "
            f"passed between one step and the next. Almost all of that is the "
            f"time taken to decide the next action; the duration attached to "
            f"an action -- `hold_key 900ms` -- is only how long the key was "
            f"held, NOT the interval between two readings.\n\n"
            f"So before calling any rate wrong, work out the real interval. A "
            f"countdown that falls sixteen units between two steps "
            f"{wall / n:.0f} seconds apart is running SLOWER than real time, "
            f"not faster. And a value that goes back UP between readings is a "
            f"level restarting -- no rate can be read across one.\n")


def _story(exp: dict) -> str:
    """The session as text, in order, with the readings attached."""
    out = []
    vp = exp.get('viewport') or {}
    if vp and not vp.get('error'):
        line = f"The window was {vp.get('view_w')}x{vp.get('view_h')}"
        i = vp.get('internal')
        if i:
            line += (f", while the game internally thinks it is "
                     f"{i.get('w')}x{i.get('h')} -- so a coordinate in a state "
                     f"reading is NOT in the same space as a click")
        out.append(line + '.')
    out.append('')
    for s in exp.get('steps') or []:
        bits = [f"step {s.get('n')}: {s.get('action')}"]
        if s.get('since_prev_s') is not None:
            bits.append(f"+{s['since_prev_s']:.0f}s since the last step")
        if s.get('args'):
            bits.append(json.dumps(s['args']))
        if s.get('refused'):
            bits.append('REFUSED by the harness, it never reached the game')
        if 'pixels_changed_pct' in s:
            bits.append(f"screen changed {s['pixels_changed_pct']}%")
        out.append('  ' + '  '.join(bits))
        if s.get('thought'):
            out.append(f"      they were trying to: {s['thought']}")
        o = s.get('observed')
        if isinstance(o, dict) and 'fields' in o:
            o = o['fields']
        if isinstance(o, dict) and 'hud_text' in o:
            t = ' '.join(str(o.get('hud_text') or '').split())
            out.append(f"      text on screen: {t[:400] or '(none at all)'}")
        elif o:
            out.append(f"      reading: {json.dumps(o, default=str)[:400]}")
        if s.get('error'):
            out.append(f"      the tool said: {str(s['error'])[:200]}")
    return '\n'.join(out)




