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
"""Carry out the exploration plan the controller wrote.

This is the step that was missing: the controller has been choosing between
three modes since it was built, and `extend_existing_demo` -- the one it picks
most -- had no implementation at all. What it picked was never executed, so
"can it write a usable extension" was an untested assumption.

Three modes, three costs:

  reuse / none            nothing runs. The evidence is already on disk.
  replay_existing_demo    replay a frozen script. No agent is involved: the
                          value of a fixed demo is that it sends exactly the
                          same inputs every time.
  extend_existing_demo    the play agent, told how the demo reaches the state
                          and what to find out once it is there.
  custom_targeted_probe   the play agent, told only what to find out.

The last two differ in what the agent is given, not in how they run. An
extension is not a longer script: the controller decides WHAT needs finding
out, and the agent decides which keys to press, because it is the one with the
screen in front of it. A keystroke chosen without seeing the screen is a guess
the agent then has to work around, and `explore(goal=...)` has taken a
question rather than a script since it was written.

Nothing here writes to `demo_outputs/`. The demo set is frozen -- the refine
prompt forbids editing it and a run records `demo_outputs_rewritten` if
anything touches it.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUN))

def _route(script: Path, demo: str, extension: str, question: str) -> str:
    """Tell the agent how the demo gets to the state, then what to find out.

    The script is quoted as a route, not as an order: it is one way to reach
    the state the question is about, and the agent may find the same state
    another way or find the route no longer works, which is itself an answer.
    """
    import rsigame.verify.post_repair_replay as PRR
    t = json.loads(script.read_text())
    evs = [e for e in (t.get('events') or []) if e.get('type') in PRR.INPUT_TYPES]
    steps = '; '.join(PRR._describe(e) for e in evs) or '(no input)'
    out = [f'First get to where the demo `{demo}` ends up. That demo does: '
           f'{steps}. Follow it if it works, or reach the same state your own '
           f'way if it does not -- and say so if it does not.',
           f'Then: {question}']
    if extension:
        out.append(f'What is needed beyond the demo: {extension}')
    return '\n\n'.join(out)


def run(plan: dict, *, work: str | Path, out_dir: str | Path,
        question: str = '', items: list[str] | None = None,
        port: int = 7960) -> dict:
    """Execute one controller plan. Returns what ran and where it landed."""
    import rsigame.verify.post_repair_replay as PRR

    work, out_dir = Path(work), Path(out_dir)
    mode = plan.get('mode') or 'none'
    demo = plan.get('demo_id') or ''
    rec = {'mode': mode, 'demo_id': demo, 'ran': False}

    if mode == 'none':
        rec['why'] = 'the controller judged the existing evidence sufficient'
        return rec

    if mode == 'replay_existing_demo':
        script = work / 'demo_outputs' / f'{demo}.json'
        if not script.is_file():
            rec['error'] = f'no such demo: {demo}'
            return rec
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            res = PRR.replay_one(work, script, out_dir)
        except Exception as exc:
            rec['error'] = f'{type(exc).__name__}: {str(exc)[:200]}'
            return rec
        # The events carry the before/after frame paths, and whoever reads
        # this evidence needs them. Returning a count and the last frame left
        # the grounding stage with one picture of a finished board and no way
        # to see what any input did -- which it said, and was right about.
        rec.update(ran=True, n_events=res.get('n_events'),
                   events=res.get('events') or [],
                   final_frame=res.get('final_frame'),
                   replay_error=res.get('replay_error'))
        return rec

    if mode in ('extend_existing_demo', 'custom_targeted_probe'):
        from rsigame.agent.evolve.explore_arm import explore
        ask = question
        scenario = None
        if mode == 'extend_existing_demo':
            script = work / 'demo_outputs' / f'{demo}.json'
            if not script.is_file():
                rec['error'] = f'no such demo: {demo}'
                return rec
            ask = _route(script, demo, plan.get('extension') or '', question)
            scenario = (json.loads(script.read_text()) or {}).get('scenario')
        try:
            ex = explore(work, game_id=f'ctrl-{mode}',
                         steps=int(os.environ.get('RSIGAME_CONTROLLER_PROBE_STEPS') or 8),
                         port=port, out_dir=out_dir / 'trace',
                         model=os.environ.get('RSIGAME_MODELS_PLAY_MODEL'),
                         goal={'ask': ask, 'items': items or [],
                               'scenarios': [scenario] if scenario else []})
        except Exception as exc:
            rec['error'] = f'{type(exc).__name__}: {str(exc)[:200]}'
            return rec
        rec.update(ran=True, ask=ask, n_steps=ex.n_steps,
                   wall_s=round(ex.wall_s, 1), observation=ex.closing_note,
                   stopped_by=ex.stopped_by)
        return rec

    rec['error'] = f'unknown mode {mode!r}'
    return rec
