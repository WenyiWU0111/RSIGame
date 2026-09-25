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
"""M3 -- turn the per-demo readings into the one summary the global stage reads.

No model is called here, by design (redesign doc S21): the model interprets,
the system counts. Anything in this file that needed a model to decide it would
be a bug, and the test asserts the module makes no network call.

Two things this layer exists to get right.

FIRST, the noise floor. M2's control -- one build replayed twice -- still
produced one `minor` regression that survived every prompt-level defence: a
score one point lower, persisting to the final frame, because a frame of
recorder jitter at spawn cost the piece one cell of hard-drop credit and score
is cumulative. A transient difference had become a permanent one, so
"does it persist" cannot separate it. What does separate it is agreement
across demos:

    a real regression   4 major, 1 medium, 1 minor -- and every one of the
                        five demos reported at least one
    pure replay noise   1 minor, on one demo

So `critical` requires a major regression, or a lesser one that two demos see
independently. A lone minor finding is recorded but does not gate anything.

SECOND, `unobserved` is not `unsatisfied`. The comparator says `unobserved`
when a demo never exercises a requirement, and it says so often and correctly.
Rolling those into "unresolved" would leave the monitor believing in a backlog
of defects that were never observed to exist, and it would never stop
spending rounds on them.

On mapping criteria to checklist ids: the model is given the ids and quotes
them back, so the ids are matched as whole tokens against the frozen
vocabulary -- not read out of prose. Criteria that match no id are kept
separately as general qualities and counted, so a comparator that stops
quoting ids shows up as a number rather than as silence.
"""
from __future__ import annotations

import re
from collections import defaultdict

DIRECTIONS = ('correctness', 'completeness', 'feedback', 'presentation')
RANK = {'major': 3, 'medium': 2, 'minor': 1}

# A lesser regression must be seen by at least this many demos to gate.
AGREEMENT = 2


def _ids_in(criterion: str, vocab: set[str]) -> list[str]:
    """Whole-token matches against the frozen id vocabulary, longest first."""
    found = []
    for tok in re.findall(r'[A-Za-z]+[0-9]+', criterion or ''):
        if tok in vocab:
            found.append(tok)
    return found


def consolidate(records: list[dict], items: list[dict], *,
                old_label: str = '', new_label: str = '') -> dict:
    """`records` are M2 outputs, one per demo. `items` is the frozen checklist."""
    vocab = {i['id'] for i in items}

    by_dir = {d: defaultdict(int) for d in DIRECTIONS}
    observed, unobserved = set(), set()
    general = defaultdict(int)
    unmapped = []
    demos_ok, demos_failed = [], []

    # regressions, keyed by the demo that saw them
    reg_by_demo: dict[str, list[dict]] = {}

    for r in records:
        demo = r.get('demo_id') or '?'
        if r.get('error'):
            demos_failed.append({'demo_id': demo, 'error': r['error']})
            continue
        demos_ok.append(demo)

        for c in r.get('changes') or []:
            ids = _ids_in(c.get('criterion', ''), vocab)
            if ids:
                (unobserved if c['change'] == 'unobserved' else observed).update(ids)
            else:
                unmapped.append({'demo_id': demo, 'criterion': c.get('criterion', '')})
                general[c['change']] += 1
            d, m = c.get('direction'), c.get('magnitude')
            if d not in DIRECTIONS:
                continue
            if c['change'] == 'better' and m:
                by_dir[d][f'improved_{m}'] += 1
            elif c['change'] == 'worse' and m:
                by_dir[d][f'worsened_{m}'] += 1
            elif c['change'] == 'same':
                by_dir[d]['unchanged'] += 1
            elif c['change'] == 'unobserved':
                by_dir[d]['unobserved'] += 1

        for g in r.get('remaining_issues') or []:
            d, sev = g.get('direction'), g.get('severity')
            if d in DIRECTIONS and sev:
                by_dir[d][f'unresolved_{sev}'] += 1

        rs = [g for g in (r.get('regressions') or []) if g.get('severity')]
        if rs:
            reg_by_demo[demo] = rs

    # -- the gate ---------------------------------------------------------
    flat = [dict(g, demo_id=d) for d, gs in reg_by_demo.items() for g in gs]
    major = [g for g in flat if g['severity'] == 'major']
    lesser = [g for g in flat if g['severity'] != 'major']
    n_demos_with_lesser = len({g['demo_id'] for g in lesser})

    if major:
        critical, why = True, (
            f'{len(major)} major regression(s) across '
            f'{len({g["demo_id"] for g in major})} demo(s)')
        gating, discounted = major + lesser, []
    elif n_demos_with_lesser >= AGREEMENT:
        critical, why = True, (
            f'no major regression, but {len(lesser)} lesser one(s) seen '
            f'independently by {n_demos_with_lesser} demos')
        gating, discounted = lesser, []
    else:
        critical = False
        gating, discounted = [], lesser
        why = ('no regression' if not lesser else
               f'{len(lesser)} lesser regression(s) on {n_demos_with_lesser} demo '
               f'only -- below the {AGREEMENT}-demo agreement floor, recorded '
               f'but not gating')

    # Regressions carry no direction in the comparator's schema, so they are
    # not rolled into `by_direction` -- attributing them to one would be the
    # aggregation layer inventing a reading.
    substantive = sum(by_dir[d][k] for d in DIRECTIONS
                      for k in ('improved_major', 'improved_medium'))

    n_items = len(items)
    seen = observed | unobserved
    return {
        'old': old_label, 'new': new_label,
        'demos': {'read': len(demos_ok), 'failed': demos_failed},
        'coverage': {
            'observed': sorted(observed),
            'unobserved_only': sorted(unobserved - observed),
            'never_mentioned': sorted({i['id'] for i in items} - seen),
            'n_items': n_items,
            'rate': round(len(observed) / n_items, 3) if n_items else 0.0,
        },
        'by_direction': {d: dict(by_dir[d]) for d in DIRECTIONS},
        'regressions': {
            'critical': critical, 'why': why,
            'gating': gating, 'discounted': discounted,
        },
        'substantive_improvements': substantive,
        'general_criteria': dict(general),
        'unmapped_criteria': unmapped,
    }


def direction_history(rows: list[dict]) -> list[dict]:
    """Redesign doc S23. The harness counts rounds; the model never does.

    `rows` are per-round records: {'direction': ..., 'accepted': bool}.
    """
    out = {d: {'direction': d, 'rounds': 0, 'accepted': 0, 'failed': 0,
               'last_progress': 'untested', 'consecutive_stall': 0}
           for d in DIRECTIONS}
    for r in rows:
        d = r.get('direction')
        if d not in out:
            continue
        e = out[d]
        e['rounds'] += 1
        if r.get('accepted'):
            e['accepted'] += 1
            e['last_progress'] = 'improved'
            e['consecutive_stall'] = 0
        else:
            e['failed'] += 1
            e['last_progress'] = 'similar'
            e['consecutive_stall'] += 1
    return [out[d] for d in DIRECTIONS]
