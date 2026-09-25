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
"""The checklist as one object that survives the round. PART 3, first half.

Part 1 says what the task asked for. Part 2 says what ONE session shows. This
carries that forward: it is the thing a round updates rather than recomputes,
and it is what makes "is this better than last round?" a question anyone can
answer.

WHY IT CANNOT JUST TAKE THE LATEST READING. Measured, on the five pilot games:

  a session confirms 9.8 of a 15-24 item checklist   -- under half is touched
  repeated readings of ONE trace agree on 90% of items -- 10% flips per call
  visualnovel went 12 satisfied -> 5 between rounds 1 and 2

That last number is not the game getting worse. It is round 2's session going
somewhere else. A state that overwrites itself from each round's reading would
spend the run oscillating, and every stage decision built on it would be noise.

SO SILENCE DOES NOT WRITE. An item the round did not observe keeps the status
it had. Only evidence moves an item, in either direction -- including
`satisfied` -> `confirmed_gap`, because repairs do break things and a state
that cannot record a regression is worse than none.

WHAT `stale` IS FOR. A `satisfied` established four rounds ago, on code that
has been edited four times since, is weaker than one established just now, and
nothing in the three statuses can say so. `stale` says it, and it is NOT a
downgrade: it orders what to re-probe (Part 4) and never changes a status by
itself. The honest version needs the Project Map (Part 6) to know which files
back which item; until then the proxy is "this round edited something".

`deferred` IS A FLAG, NOT A FOURTH STATUS. Handoff 5 fixes the vocabulary at
three, and an item whose stage ran out of budget is still a confirmed gap --
what changed is that nobody is going to work on it now. Keeping it as a flag
means it stays countable and visible instead of disappearing into a status
nobody else knows how to read.
"""

from __future__ import annotations

import json
from pathlib import Path

STATUSES = ('satisfied', 'confirmed_gap', 'unverified')
# Polish is NOT a checklist stage -- see the Polish handoff. This vocabulary
# (satisfied / confirmed_gap / unverified) cannot express "the explosion is a
# plain orange circle", and forcing it to left every polish item sitting at
# `satisfied` while producing no work at all.
STAGES = ('correct', 'complete')

# Which transitions evidence is allowed to make. Absence is not in here at all:
# there is no path back to `unverified`, which is the whole point.
_ALLOWED = {
    ('unverified', 'satisfied'),
    ('unverified', 'confirmed_gap'),
    ('confirmed_gap', 'satisfied'),      # fixed
    ('satisfied', 'confirmed_gap'),      # regressed
}

MAX_EVIDENCE = 6          # per item; the run record keeps the rest


def init_state(checklist: dict, bootstrap: dict, *,
               stages: dict | None = None) -> dict:
    """The persistent object, from Part 1's list and Part 2's first reading.

    `stages` maps item id -> stage and is decided ONCE, here, then frozen: a
    per-round classification would add a second noisy signal on top of the
    status, and then a stage transition could not be attributed to either.
    """
    stages = stages or {}
    items = []
    seen = {s['id']: s for s in (bootstrap.get('statuses') or [])}
    for it in checklist.get('items') or []:
        r = seen.get(it['id']) or {}
        items.append(_item(it['id'], it['requirement'], 'task',
                           stages.get(it['id'], 'correct'), r, 0))
    for g in bootstrap.get('general') or []:
        items.append(_item(g['id'], g.get('check', ''), 'general',
                           stages.get(g['id'], 'correct'), g, 0))
    return {'items': items, 'round': 0, 'history': []}


def _item(iid, requirement, source, stage, reading, rnd) -> dict:
    st = reading.get('status') if reading.get('status') in STATUSES \
        else 'unverified'
    it = {'id': iid, 'requirement': requirement, 'source': source,
          'stage': stage if stage in STAGES else 'correct',
          'status': st, 'stale': False, 'deferred': False,
          'last_change_round': rnd if st != 'unverified' else None,
          'evidence_refs': []}
    if st != 'unverified':
        it['evidence_refs'].append(_ref(reading, rnd))
    return it


def _ref(reading: dict, rnd: int) -> dict:
    return {'round': rnd, 'status': reading.get('status'),
            'cites': reading.get('cites') or [], 'basis': reading.get('basis'),
            'note': (reading.get('note') or '')[:200]}


def update(state: dict, bootstrap: dict, rnd: int, *,
           edited: bool = False) -> dict:
    """Fold one round's reading into the state. Returns what moved.

    `edited` says the round changed at least one file, which is what marks
    `satisfied` items stale. It is a proxy for "the code behind this item may
    have moved" and will be replaced by the Project Map.
    """
    by_id = {s['id']: s for s in (bootstrap.get('statuses') or [])}
    by_id.update({g['id']: g for g in (bootstrap.get('general') or [])})
    moves, ignored = [], []

    for it in state['items']:
        r = by_id.get(it['id'])
        want = (r or {}).get('status')
        if not r or want not in STATUSES:
            continue
        if want == 'unverified':
            # The round looked and could not tell. That is not news about the
            # item; it is news about the round, and it stays out of the state.
            if it['status'] != 'unverified':
                ignored.append({'id': it['id'], 'kept': it['status'],
                                'reason': 'this round could not tell'})
            continue
        if want == it['status']:
            it['evidence_refs'] = (it['evidence_refs']
                                   + [_ref(r, rnd)])[-MAX_EVIDENCE:]
            if want == 'satisfied':
                it['stale'] = False        # re-confirmed just now
            continue
        if (it['status'], want) in _ALLOWED:
            moves.append({'id': it['id'], 'from': it['status'], 'to': want,
                          'round': rnd, 'cites': r.get('cites') or []})
            it['status'] = want
            it['last_change_round'] = rnd
            it['stale'] = False
            it['deferred'] = False if want == 'satisfied' else it['deferred']
            it['evidence_refs'] = (it['evidence_refs']
                                   + [_ref(r, rnd)])[-MAX_EVIDENCE:]

    if edited:
        for it in state['items']:
            if it['status'] == 'satisfied' and it['last_change_round'] != rnd:
                it['stale'] = True

    state['round'] = rnd
    state['history'].append({'round': rnd, 'moves': moves,
                             'ignored': len(ignored), 'edited': edited})
    return {'moves': moves, 'ignored': ignored}


def tally(state: dict, stage: str | None = None) -> dict:
    items = [i for i in state['items']
             if stage is None or i['stage'] == stage]
    out = {k: sum(1 for i in items if i['status'] == k) for k in STATUSES}
    out['deferred'] = sum(1 for i in items if i['deferred'])
    out['stale'] = sum(1 for i in items if i['stale'])
    out['n'] = len(items)
    return out


def save(state: dict, path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=1))
    return p


def render(state: dict) -> str:
    mark = {'satisfied': 'OK ', 'confirmed_gap': 'GAP', 'unverified': '?  '}
    out = []
    for st in STAGES:
        items = [i for i in state['items'] if i['stage'] == st]
        if not items:
            continue
        t = tally(state, st)
        out.append(f"[{st.upper()}]  {t['satisfied']} ok / "
                   f"{t['confirmed_gap']} gap / {t['unverified']} unverified"
                   + (f"  ({t['deferred']} deferred)" if t['deferred'] else ''))
        for i in items:
            flags = ''.join(c for c, on in (('*', i['stale']),
                                            ('D', i['deferred'])) if on)
            out.append(f"  {mark[i['status']]} {i['id']:<4}{flags:<2} "
                       f"{i['requirement'][:62]}")
    return '\n'.join(out)
