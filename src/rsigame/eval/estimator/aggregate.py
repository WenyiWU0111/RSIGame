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
"""From per-demo item scores to one number per category. No model, no guessing.

EVERY ARITHMETIC DECISION IS HERE AND NOWHERE ELSE, which is the same split
`controller/verify.py` uses: the model says what it sees, the harness says what
that adds up to. A model asked to also compute its own aggregate will quietly
weight what it thinks matters.

THREE RULES THAT LOOK SMALL AND ARE NOT:

  1. `null` never becomes 0. An unobserved criterion leaves the numerator AND
     the denominator -- it lowers coverage, not the score. A game is not bad
     at something the script never asked it to do on camera.

  2. `max` and `mean` are per criterion, fixed in the rubric, applied ACROSS
     DEMOS. "Can the game reach its end screen" is answered by the one demo
     that reached it; "is the UI legible" is not redeemed by one tidy screen
     among five crowded ones.

  3. Headroom is never reported without its coverage. `1 - F` on a category
     observed in one demo out of three is a number about the sample, and the
     monitor has to be able to see that.
"""
from __future__ import annotations

import statistics
from collections import defaultdict

CATS = ('F', 'C', 'I', 'P')
CAT_NAME = {'F': 'functional', 'C': 'content', 'I': 'feedback',
            'P': 'presentation'}


def _combine(vals: list[float], how: str) -> float:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return max(vals) if how == 'max' else statistics.mean(vals)


def one_side(per_demo: list[dict], rubric: dict, side: str, only: set | None = None) -> dict:
    """side is 'old' or 'new'. -> per-item, per-category, overall.

    `only` restricts the category and overall means to a given set of criteria.
    `pair` passes the criteria BOTH sides were seen on, so a difference is a
    difference in the game rather than in what the two recordings happened to
    show -- see the note there.
    """
    how = {it['id']: it['aggregation'] for it in rubric['items']}
    cat = {it['id']: it['category'] for it in rubric['items']}
    order = [it['id'] for it in rubric['items']]
    # A stage checklist weighs its criteria (estimator/stage_rubric.py): the
    # brief's own criteria count more than the task document's, and a saturated
    # criterion kept only to catch a regression counts least. A rubric without
    # weights behaves exactly as before -- every weight is 1.
    wt = {it['id']: float(it.get('weight') or 1.0) for it in rubric['items']}
    org = {it['id']: it.get('origin') for it in rubric['items']}

    by_item = defaultdict(list)
    for d in per_demo:
        for row in (d.get('items') or []):
            v = row.get(f'{side}_score')
            if v is not None:
                by_item[row['id']].append(float(v))

    items = {}
    for i in order:
        items[i] = {'value': _combine(by_item.get(i, []), how[i]),
                    'n_demos': len(by_item.get(i, [])),
                    'aggregation': how[i], 'category': cat[i],
                    'weight': wt[i], 'origin': org[i]}

    def wmean(ids):
        if only is not None:
            ids = [i for i in ids if i in only]
        seen = [(items[i]['value'], wt[i]) for i in ids if items[i]['value'] is not None]
        w = sum(x[1] for x in seen)
        return (sum(v * x for v, x in seen) / w) if w else None

    cats = {}
    for c in CATS:
        ids = [i for i in order if cat[i] == c]
        seen = [items[i]['value'] for i in ids if items[i]['value'] is not None]
        # headroom the guard used to sit on `seen` , but `seen` is computed without `only` filtering,
        # while wmean only looks at `only` the filtered items —— so seen was non-empty while wmean was None,
        # `1.0 - None` and it crashed (9 games during the Phaser monitor replay).
        # the guard belongs on wmean's result.
        _v = wmean(ids)
        cats[c] = {
            'value': _v,
            'coverage': len(seen) / len(ids) if ids else 0.0,
            'observed': len(seen), 'total': len(ids),
            # Reported together, always. A headroom computed from one observed
            # criterion out of four is a fact about the replay, not the game.
            'headroom': (1.0 - _v) if _v is not None else None}

    have = [cats[c]['value'] for c in CATS if cats[c]['value'] is not None]
    overall = sum(have) / len(have) if have else None
    # The brief's own half, reported separately: on a stage checklist that is
    # what the round was asked to move, and a mean over four categories can
    # hide it. Absent (None) on a rubric with no brief criteria.
    by_origin = {}
    for o in ('brief', 'task', 'guard'):
        ids = [i for i in order if org[i] == o]
        by_origin[o] = wmean(ids) if ids else None
    return {'items': items, 'categories': cats, 'overall': overall,
            'by_origin': by_origin,
            'categories_observed': len(have),
            'coverage': sum(cats[c]['observed'] for c in CATS)
                        / max(sum(cats[c]['total'] for c in CATS), 1)}


def pair(per_demo: list[dict], rubric: dict) -> dict:
    """Both sides plus their differences. `Delta_proxy` is the headline.

    EVERY NUMBER THAT IS SUBTRACTED IS COMPUTED ON THE CRITERIA BOTH SIDES WERE
    SEEN ON. Each side used to be averaged over whatever its own recording
    happened to show, so the two means were over different sets of criteria and
    the difference moved when nothing about the game did: measured on
    puzzle-block-cascade R09 vs R19 (2026-09-15), one criterion scoring 0.5 was
    observed on the old side only, and the task half read -0.07 on one grading
    and +0.05 on another with no criterion changing. The per-side absolute
    means over everything each side showed are kept as `old_all` / `new_all`.
    """
    all_old = one_side(per_demo, rubric, 'old')
    all_new = one_side(per_demo, rubric, 'new')
    common = {i for i in all_new['items']
              if all_old['items'][i]['value'] is not None
              and all_new['items'][i]['value'] is not None}
    old = one_side(per_demo, rubric, 'old', only=common)
    new = one_side(per_demo, rubric, 'new', only=common)

    def d(a, b):
        return None if (a is None or b is None) else round(b - a, 4)

    # A CRITERION COMPARES ONLY WHERE BOTH SIDES WERE SEEN. One side observed
    # and the other not is `unobserved`, never a gain or a loss -- the most
    # common way a proxy invents movement is by reading a missing observation
    # as a zero.
    moves = []
    for i, it in new['items'].items():
        o, n = old['items'][i]['value'], it['value']
        if o is None or n is None:
            comp = 'unobserved'
        elif n > o:
            comp = 'improved'
        elif n < o:
            comp = 'regressed'
        else:
            comp = 'same'
        moves.append({'id': i, 'category': it['category'],
                      'old': o, 'new': n, 'delta': d(o, n),
                      'comparison': comp})

    cats = {}
    for c in CATS:
        cats[c] = {
            'name': CAT_NAME[c],
            'old': old['categories'][c]['value'],
            'new': new['categories'][c]['value'],
            'delta': d(old['categories'][c]['value'],
                       new['categories'][c]['value']),
            'headroom': new['categories'][c]['headroom'],
            'coverage': round(min(old['categories'][c]['coverage'],
                                  new['categories'][c]['coverage']), 3)}

    origins = {}
    for o in ('brief', 'task', 'guard'):
        if old['by_origin'].get(o) is None and new['by_origin'].get(o) is None:
            continue
        origins[o] = {'old': old['by_origin'][o], 'new': new['by_origin'][o],
                      'delta': d(old['by_origin'][o], new['by_origin'][o])}
    return {
        'proxy_value': {
            'overall': {'old': old['overall'], 'new': new['overall'],
                        'delta': d(old['overall'], new['overall']),
                        'old_all': all_old['overall'], 'new_all': all_new['overall'],
                        'n_common': len(common), 'n_items': len(all_new['items']),
                        'coverage': round(min(old['coverage'],
                                              new['coverage']), 3)},
            **({'by_origin': origins} if origins else {}),
            **{CAT_NAME[c]: cats[c] for c in CATS}},
        'items': moves,
        'largest_remaining_gaps': sorted(
            [m for m in moves if m['new'] is not None and m['new'] < 1.0],
            key=lambda m: m['new'])[:4],
        'largest_recent_gains': sorted(
            [m for m in moves if m['delta'] is not None and m['delta'] > 0],
            key=lambda m: -m['delta'])[:4],
        'n_demos': len(per_demo),
        'n_unobserved': sum(1 for m in moves if m['comparison'] == 'unobserved'),
    }
