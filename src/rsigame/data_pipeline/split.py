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
"""Split a corpus into train and holdout without leaking between them.

The danger here is not the obvious one. Splitting rows at random would be safe
if every row were an independent game, but this corpus has three ways for the
same content to appear twice, and each needs its own guard:

  1. SAME TASK, SEVERAL TRIALS. The Godot archives are timestamped retries:
     1,202 recordings across 1,004 tasks. `godot_trajectories` already keeps
     one trial per task, but a corpus assembled from several archives can
     reintroduce them. Split on TASK ID, never on row index.

  2. SAME BRIEF, DIFFERENT ENGINE. A brief named `puzzle-glacier-glyphs` can
     exist as both a web and a Godot task. Holding out the web one while
     training on the Godot one leaks the design. The unit of holdout is the
     BRIEF STEM -- the task id with its difficulty suffix removed -- so both
     go to the same side.

  3. THE EVALUATION BRIEFS. The briefs every published score is measured on
     must not be in the training corpus at all, or every number is
     contaminated. They are excluded outright and the count is REPORTED: a
     silent zero here is indistinguishable from a working check, which is a
     failure mode this project has already hit.

STRATIFICATION. The holdout is drawn per (engine, family) bucket, so a 10%
holdout is 10% of each archetype rather than 10% of whatever sorted first.

THE STOPPING RULE IS ON ROWS, NOT STEMS. Holding out a fixed fraction of stems
overshoots badly -- a web stem like `platformer-moving` carries many D1/D2/D3
rows while a Godot stem carries one, so 10% of stems came out as 16.7% of rows
on the pilot. The unit of holdout stays the stem, which is what prevents the
leak; only the stopping rule counts rows.

Nothing is written if the split does not hold. A leaking split that was
written is worse than no split, because the file looks finished.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from . import jsonl

# The briefs every published score is measured on. Override with --eval-briefs
# when the held-out set changes; do not quietly train on them.
EVAL_BRIEFS = {
    'platformer-thunder-valkyrie',
    'puzzle-circuit-wizard',
    'roguelike-breach-tactics',
}

DIFF_SUFFIX = re.compile(r'-D\d+(-\d+)?$')


def brief_stem(task: str) -> str:
    """`platformer-moving-D3-044` -> `platformer-moving`.

    A Godot task keeps its whole name. This is the unit that must not straddle
    the split.
    """
    return DIFF_SUFFIX.sub('', task or '').strip('-')


def family(task: str) -> str:
    m = re.match(r'^([a-z_]+)', task or '')
    return m.group(1) if m else 'unknown'


def choose_holdout(by_stem: dict[str, list[dict]], frac: float, rng: random.Random) -> set[str]:
    """-> the stems that go to the holdout side, stratified by (engine, family)."""
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for stem, rs in by_stem.items():
        engines = {x.get('engine', '?') for x in rs}
        eng = engines.pop() if len(engines) == 1 else 'mixed'
        buckets[(eng, family(stem))].append(stem)

    chosen: set[str] = set()
    for stems in buckets.values():
        stems = sorted(stems)
        rng.shuffle(stems)
        quota = sum(len(by_stem[s]) for s in stems) * frac
        if len(stems) < 2 or quota <= 0:
            continue                      # a single-stem bucket cannot be split
        taken = 0
        for s in stems:
            if taken >= quota:
                break
            n = len(by_stem[s])
            # do not blow past the quota by more than half a stem
            if taken and taken + n > quota + n / 2:
                continue
            chosen.add(s)
            taken += n
    return chosen


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog='rsigame.data_pipeline split', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--inp', required=True, help='the mixed corpus to split')
    ap.add_argument('--train', required=True)
    ap.add_argument('--test', required=True)
    ap.add_argument('--test-frac', type=float, default=0.10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--eval-briefs', default=','.join(sorted(EVAL_BRIEFS)),
                    help='comma-separated briefs excluded outright (the scored holdout)')
    ap.add_argument('--report', default=None, help='write a JSON report of the split')
    a = ap.parse_args(argv)

    rows = list(jsonl.read(a.inp))
    rng = random.Random(a.seed)
    evals = {b.strip() for b in a.eval_briefs.split(',') if b.strip()}

    # --- guard 3: the evaluation briefs must not be here at all -------------
    def is_eval(r: dict) -> bool:
        return brief_stem(r.get('task', '')) in evals or r.get('task') in evals

    contaminated = [r for r in rows if is_eval(r)]
    if contaminated:
        print(f'*** {len(contaminated)} row(s) match a held-out EVAL brief -- excluding ***')
        for r in contaminated:
            print(f"      {r.get('engine')}/{r.get('task')}")
    total_seen = len(rows)
    rows = [r for r in rows if not is_eval(r)]
    print(f'eval-brief contamination check: {len(contaminated)} removed '
          f'(checked {len(evals)} briefs against {total_seen} rows)')

    # --- guards 1 and 2: hold out whole brief stems -------------------------
    by_stem: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_stem[brief_stem(r.get('task', ''))].append(r)
    test_stems = choose_holdout(by_stem, a.test_frac, rng)

    train = [r for r in rows if brief_stem(r.get('task', '')) not in test_stems]
    test = [r for r in rows if brief_stem(r.get('task', '')) in test_stems]

    # --- verify the split actually holds ------------------------------------
    tr_stems = {brief_stem(r.get('task', '')) for r in train}
    te_stems = {brief_stem(r.get('task', '')) for r in test}
    overlap = tr_stems & te_stems
    task_overlap = {r.get('task') for r in train} & {r.get('task') for r in test}

    for name, rs in (('train', train), ('test', test)):
        eng = Counter(r.get('engine') for r in rs)
        stems = len({brief_stem(r.get('task', '')) for r in rs})
        print(f'{name:6s} {len(rs):5d} rows  engines={dict(eng)}  stems={stems}')
    print('\nLEAKAGE CHECK')
    print(f'  brief stems on both sides : {len(overlap)}   (must be 0)')
    print(f'  task ids on both sides    : {len(task_overlap)}   (must be 0)')
    print(f'  holdout fraction (rows)   : {len(test) / max(len(rows), 1):.1%}  '
          f'(target {a.test_frac:.0%})')
    if overlap or task_overlap:
        print('*** REFUSING TO WRITE: the split leaks ***')
        for s in sorted(overlap)[:10]:
            print('     stem:', s)
        return 1

    for path, rs in ((a.train, train), (a.test, test)):
        jsonl.write(path, rs)
        print(f'wrote {len(rs):5d} -> {path}')

    if a.report:
        rep = {
            'seed': a.seed,
            'test_frac_target': a.test_frac,
            'test_frac_actual': len(test) / max(len(rows), 1),
            'n_train': len(train),
            'n_test': len(test),
            'eval_briefs_removed': [r.get('task') for r in contaminated],
            'train_engines': dict(Counter(r.get('engine') for r in train)),
            'test_engines': dict(Counter(r.get('engine') for r in test)),
            'test_stems': sorted(te_stems),
            'leak_stems': sorted(overlap),
            'leak_tasks': sorted(task_overlap),
        }
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(rep, indent=1))
        print(f'report -> {a.report}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
