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
"""Read this round's evidence against the requirement list, every round.

The old loop did this and this one did not, which is the single largest thing
that went missing in the redesign. There, every round ended with one call that
answered "which of these requirements did what you just watched actually show,
and did it show them working or failing", and the answer was folded into a
persistent per-item state through `checklist_state.update`. Here the statuses
only moved when the global monitor ran -- four rounds in twelve -- and only for
the handful of items its frame comparison happened to cover. Everything else
sat at `unverified` for the whole run, which is why the ranker so often had
nothing eligible to repair and why the monitor read the same board four times.

Two things make this a wiring job rather than a new mechanism:

  the reader        `bootstrap_verifier.bootstrap` already does exactly this,
                    gates included -- a claimed gap has to cite a step that
                    exists and either quote the record or point at a frame that
                    was shown. Nothing here re-implements any of it.

  the evidence      `bootstrap` wants an exploration record. Half the rounds
                    here replay a fixed demo instead, which produces events
                    with before/after frames and no session text. That is a
                    format difference, not a missing observation, so a replay
                    is rewritten into the shape the reader expects: one step
                    per input, the event's own description as the action, the
                    first after-frame as the step's frame, and the round's
                    written account as the closing note so a quote-based claim
                    still has something to be checked against.

Not a new judgement anywhere. The reader is the old one, the gates are the old
ones, and the transition rules in `checklist_state` are what decide whether an
answer is allowed to move an item at all.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUN))


def as_exploration(ex: dict, probe_dir: Path, observation: str = '') -> dict | None:
    """The round's evidence in the shape the checklist reader reads.

    A real play session is returned as it was written. A replay is rebuilt:
    what is lost in the rebuild is the agent's thinking, which the reader was
    never entitled to treat as evidence anyway.
    """
    real = Path(probe_dir) / 'trace' / 'exploration.json'
    if real.is_file():
        try:
            return json.loads(real.read_text())
        except Exception:
            pass
    events = ex.get('events') or []
    if not events:
        return None
    steps = []
    for e in events:
        after = (e.get('after') or [None])[0]
        steps.append({'n': e.get('n'), 'action': 'replayed_input',
                      'args': {'input': e.get('what') or ''},
                      'frame': after or e.get('before'),
                      'ok': True})
    return {'steps': steps, 'reached': [], 'not_reached': [],
            'closing_note': (observation or '')[:4000],
            'stopped_by': 'replay finished',
            'replayed_demo': ex.get('demo_id') or ''}


def read(game: str, *, exp: dict, exp_path: Path, state_path: Path,
         static_view: str = '', round_n: int = 0, edited: bool = False,
         model: str | None = None) -> dict:
    """One reading, folded into the persistent state. Returns what moved.

    `edited` says the PREVIOUS round changed at least one file, which is what
    makes a `satisfied` item stale -- the code behind it may have moved since
    anyone confirmed it. This round's repair has not run yet.
    """
    import rsigame.checklist.bootstrap_verifier as BV
    import rsigame.checklist.checklist_state as CS

    exp_path = Path(exp_path)
    exp_path.write_text(json.dumps(exp, ensure_ascii=False, indent=1))
    state = json.loads(Path(state_path).read_text())
    items = [{'id': i['id'], 'requirement': i['requirement']}
             for i in state.get('items') or [] if i.get('source') != 'general']
    if not items:
        return {'error': 'no checklist items'}

    bs = BV.bootstrap(game, exp_path, checklist={'items': items},
                      static_view=static_view,
                      model=model or os.environ.get('RSIGAME_MODELS_LOOP_MODEL'))
    if bs.get('error'):
        return {'error': bs['error']}

    moved = CS.update(state, bs, round_n, edited=edited)
    CS.save(state, state_path)
    tal = CS.tally(state)
    return {'moves': moved['moves'], 'ignored': len(moved['ignored']),
            'tally': tal, 'gated': _gated(bs), 'reader_tally': bs.get('tally')}


def _gated(bs: dict) -> list:
    """Claims the gates refused, so a run can say whether they did anything."""
    out = []
    for s in (bs.get('statuses') or []) + (bs.get('general') or []):
        if s.get('gate') and s['gate'] != 'not answered':
            out.append({'id': s['id'], 'gate': s['gate'][:90]})
    return out[:12]


def from_comparison(summary: dict, round_n: int) -> dict:
    """The monitor's frame comparison, in the same shape the reader answers in.

    The comparator is the one reading that talks about requirements by id, so
    it can move the board too -- but through `checklist_state.update` like
    everything else, rather than by assigning statuses directly. Written
    straight into the state it was skipping the transition rules entirely, so a
    `satisfied` item could be sent anywhere and no move was ever recorded.
    """
    cov = summary.get('coverage') or {}
    observed = list(cov.get('observed') or [])
    faulted = set()
    for k in ('gating', 'discounted'):
        for x in (summary.get('regressions') or {}).get(k) or []:
            faulted |= {t for t in observed if t in str(x.get('what', ''))}
    return {'statuses': [
        {'id': i, 'status': 'confirmed_gap' if i in faulted else 'satisfied',
         'cites': [], 'basis': 'frame', 'quote': '', 'gate': '',
         'note': 'from the checkpoint comparison'} for i in observed],
        'general': []}
