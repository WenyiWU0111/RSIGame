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
"""Turn a play-agent session into a replayable script.

The play agent explores adaptively and costs two to three minutes a session.
What it leaves behind, though, is a list of inputs it actually sent, each with
the second it was sent at -- and that is a demo script in everything but
format. Frozen, the same investigation replays in the thirty seconds a fixed
demo takes, deterministically, and can be paired frame by frame against a later
build the way a fixed demo can.

That is the point: it makes the design doc's rule -- the probe that found a
problem becomes the acceptance test for its repair -- true for the expensive
kind of probe too, not just for demos that already existed.

What is dropped, and why:

  refused steps      the agent named a target the grounder could not find, and
                     nothing was clicked. 135 of them across the corpus, every
                     one refused; a successful click always carries x,y (785
                     out of 785). Replaying a refusal would replay nothing.
  boot / screenshot  not inputs. `boot_scenario` is not dropped -- it becomes
                     the launch flag, because a session that started in a
                     scenario cannot be reproduced without it.

Timing comes from each step's `at_s`, so the agent's own pauses survive. A
fixed gap would compress the wait it took after a slow transition, and the
replay would then race past the thing it was waiting for.
"""
from __future__ import annotations

import json
from pathlib import Path

FPS = 30
TAIL_FRAMES = 90   # three seconds after the last input, so it can finish

# The gap between two of the agent's actions is mostly the model thinking, not
# the game needing time: a 13-step session spans 109 seconds of wall clock and
# sends 11 inputs. Replaying that faithfully would spend most of it on nothing,
# and `replay_trace` cuts a recording off at 90 seconds anyway, so a long
# session would simply lose its tail.
#
# What the game genuinely needs is bounded and known: the harness settles for
# 1500ms after an action, and the agent's own explicit waits run to 5000ms. So
# gaps are kept up to five seconds and compressed past that, and an explicit
# `wait` keeps its full length because someone chose it.
MAX_GAP_FRAMES = 5 * FPS
MIN_GAP_FRAMES = 15

# The agent writes keys the way a person says them; the replay harness wants
# the engine's own names.
_KEYS = {' ': 'SPACE', 'space': 'SPACE', 'enter': 'ENTER', 'return': 'ENTER',
         'escape': 'ESCAPE', 'esc': 'ESCAPE', 'tab': 'TAB', 'shift': 'SHIFT',
         'ctrl': 'CTRL', 'control': 'CTRL', 'alt': 'ALT', 'up': 'UP',
         'down': 'DOWN', 'left': 'LEFT', 'right': 'RIGHT',
         'backspace': 'BACKSPACE', 'delete': 'DELETE'}


def _key(raw) -> str | None:
    if raw is None:
        return None
    s = str(raw)
    if s in _KEYS:
        return _KEYS[s]
    low = s.strip().lower()
    if low in _KEYS:
        return _KEYS[low]
    if len(s.strip()) == 1:
        return s.strip().upper()
    return s.strip().upper() or None


def freeze(exploration: str | Path | dict, out_path: str | Path,
           *, question: str = '') -> dict:
    """Write the session's inputs as a demo script. Returns what was kept."""
    t = (exploration if isinstance(exploration, dict)
         else json.loads(Path(exploration).read_text()))
    steps = t.get('steps') or []

    events, scenario, dropped = [], None, {'refused': 0, 'not_input': 0,
                                           'no_timing': 0, 'unusable': 0}
    for s in steps:
        act = s.get('action')
        args = s.get('args') or {}
        if act == 'boot_scenario':
            scenario = args.get('id') or scenario
            continue
        if not s.get('ok'):
            dropped['refused'] += 1
            continue
        if act in ('boot', 'screenshot', 'shot', 'read_state', 'note'):
            dropped['not_input'] += 1
            continue
        at = s.get('at_s')
        if at is None:
            dropped['no_timing'] += 1
            continue
        gap = float(s.get('since_prev_s') or 0)
        want = min(max(gap * FPS, MIN_GAP_FRAMES), MAX_GAP_FRAMES)
        f = int(round((events[-1]['frame'] if events else 0) + want))

        if act == 'click':
            x, y = args.get('x'), args.get('y')
            if x is None or y is None:
                dropped['unusable'] += 1
                continue
            events.append({'frame': f, 'type': 'mouse_click',
                           'button': args.get('button') or 'left',
                           'x': int(x), 'y': int(y)})
        elif act == 'press_key':
            k = _key(args.get('key') or args.get('keycode'))
            if not k:
                dropped['unusable'] += 1
                continue
            events.append({'frame': f, 'type': 'key_press', 'keycode': k})
        elif act == 'hold_key':
            k = _key(args.get('key') or args.get('keycode'))
            if not k:
                dropped['unusable'] += 1
                continue
            held = max(1, int(round(float(args.get('ms') or 0) / 1000 * FPS)))
            events.append({'frame': f, 'type': 'key_down', 'keycode': k})
            events.append({'frame': f + held, 'type': 'key_up', 'keycode': k})
        elif act == 'wait':
            # A wait sends nothing, so it is not an event -- but it was a
            # deliberate choice about how long the game needs, so it survives
            # in full as a gap rather than being capped like thinking time.
            ms = float(args.get('ms') or 0)
            if ms > 0 and events:
                events[-1]['_pad'] = max(events[-1].get('_pad', 0),
                                         int(round(ms / 1000 * FPS)))
            continue
        else:
            dropped['unusable'] += 1

    # apply the deliberate waits, shifting everything after them
    shift = 0
    for e in events:
        e['frame'] += shift
        shift += e.pop('_pad', 0)
    events.sort(key=lambda e: e['frame'])
    last = events[-1]['frame'] if events else 0
    script = {'scenario': scenario,
              'duration_frames': last + TAIL_FRAMES,
              'events': events,
              # Not part of the replay format; carried so a later reader knows
              # this script is a frozen investigation and what it was after.
              'frozen_from': str(exploration) if not isinstance(exploration, dict) else 'inline',
              'question': question[:400]}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(script, indent=2))
    return {'path': str(out_path), 'n_events': len(events),
            'scenario': scenario, 'duration_frames': script['duration_frames'],
            'dropped': dropped, 'n_steps': len(steps)}
