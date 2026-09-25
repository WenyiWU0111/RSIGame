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
"""Repair records -> stage-3 training rows: bug evidence in, a fix patch out.

RECORD SHAPE (one directory per repair round)

    records/<game>/r<NN>/
        prompt.txt     what the model was told was wrong, and what to fix
        patch.diff     the change it made                <- the training target
        verdict.json   goal_achieved: yes | no | unclear | null
        goal.json      the round's planned goal and its budget reasoning
        reading.json   the playthrough the verdict was drawn from
        cost.json      token and cost accounting

THE GATE
    Only `goal_achieved == "yes"` is trained on, and that verdict is measured
    in an INDEPENDENT play session -- never from the model's own report, because
    a model claiming success is not evidence of success. Measured over one
    collection arm: 1,032 rounds produced 179 passes (17%), 597 explicit
    failures and 194 unclear.

WHY "no" IS KEPT SEPARATELY RATHER THAN DISCARDED
    A failed repair whose verdict explains WHY it failed is exactly the shape a
    preference pass needs. `--negatives` writes those to their own file instead
    of deleting ~600 of them. They are NOT part of the SFT corpus.

ENGINE IS READ FROM THE PATCH, NOT FROM THE PROMPT
    The prompt opens "You are improving a small browser game", but these
    records patch `main.gd`, `project.godot` and `.tscn` scenes -- measured
    across 400 patches: 300 Godot markers, zero web. The prompt wording is
    stale boilerplate. Rows are tagged on the evidence of what the patch
    touches, because the same mistake (labelling an engine per directory
    rather than per artifact) had already put 42 Godot games in folders
    marked `web` earlier in this project.

THE SYSTEM PROMPT IS BYTE-IDENTICAL to the rest of the repair corpus. An
earlier version of this converter invented its own wording, which would have
split the corpus in two with no error raised anywhere.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from . import jsonl, patches

DIFF_FILE = re.compile(r'^diff -ruN a/(\S+)', re.M)
GODOT_EXT = {'gd', 'tscn', 'godot', 'tres'}
WEB_EXT = {'ts', 'tsx', 'js', 'jsx'}

SYSTEM = ('You are a senior game engineer. Given a bug report with play-session '
          'evidence, produce a unified diff that fixes it. Output only the patch.')


def detect_engine(patch: str) -> str:
    g = w = 0
    for m in DIFF_FILE.finditer(patch):
        e = m.group(1).rsplit('.', 1)[-1].lower()
        g += e in GODOT_EXT
        w += e in WEB_EXT
    if g and not w:
        return 'godot'
    if w and not g:
        return 'web'
    return 'mixed' if (g or w) else 'unknown'


def load_round(d: Path, *, minimize: bool = True) -> dict | None:
    try:
        verdict = json.loads((d / 'verdict.json').read_text(errors='replace'))
        prompt = (d / 'prompt.txt').read_text(errors='replace').strip()
        patch = (d / 'patch.diff').read_text(errors='replace')
    except (OSError, ValueError):
        return None
    if not prompt or not patch.strip():
        return None
    raw_bytes = len(patch)
    if minimize:
        small = patches.minimize(patch)
        # An empty minimisation means the keep-list matched nothing in this
        # patch. Training on '' teaches the empty fix, so keep the raw patch
        # and let the size stats show what happened.
        patch = small if small.strip() else patch
    return {
        'task': d.parent.name,
        'round': d.name,
        'goal_achieved': verdict.get('goal_achieved'),
        'why': verdict.get('why'),
        'broken_now': verdict.get('broken_now'),
        'prompt': prompt,
        'patch': patch,
        'engine': detect_engine(patch),
        'patch_bytes': len(patch),
        'raw_patch_bytes': raw_bytes,
    }


def to_row(r: dict) -> dict:
    return {
        'task': r['task'],
        'round': r['round'],
        'engine': r['engine'],
        'source': 'self_refine',
        'verdict': r['goal_achieved'],
        'messages': [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': r['prompt']},
            {'role': 'assistant', 'content': r['patch']},
        ],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog='rsigame.data_pipeline repair', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--roots', nargs='+', required=True, help='extracted record trees')
    ap.add_argument('--out', required=True, help='SFT rows (goal_achieved == yes)')
    ap.add_argument('--negatives', default=None, help='failed repairs, for a later preference pass')
    ap.add_argument('--raw-patch', dest='minimize', action='store_false',
                    help='keep the full-tree diff as the target (default: minimise it)')
    ap.add_argument('--max-patch-bytes', type=int, default=200_000,
                    help='drop absurd diffs; the median after minimisation is ~2.5KB')
    ap.add_argument('--one-per-game', action='store_true',
                    help='keep only the best passing round per game (dedup pressure)')
    a = ap.parse_args(argv)

    rounds = []
    for root in a.roots:
        for v in Path(root).rglob('records/*/r*/verdict.json'):
            r = load_round(v.parent, minimize=a.minimize)
            if r:
                rounds.append(r)

    stats: Counter = Counter()
    for r in rounds:
        stats[f"verdict:{r['goal_achieved']}"] += 1
        stats[f"engine:{r['engine']}"] += 1

    keep = [r for r in rounds
            if r['goal_achieved'] == 'yes' and r['patch_bytes'] <= a.max_patch_bytes]
    stats['dropped:oversize_patch'] = sum(
        1 for r in rounds
        if r['goal_achieved'] == 'yes' and r['patch_bytes'] > a.max_patch_bytes)

    if a.one_per_game:
        best: dict[str, dict] = {}
        for r in keep:
            cur = best.get(r['task'])
            # prefer the later round: it had more context and more prior fixes
            if cur is None or r['round'] > cur['round']:
                best[r['task']] = r
        stats['deduped_away'] = len(keep) - len(best)
        keep = list(best.values())

    keep.sort(key=lambda r: (r['task'], r['round']))
    jsonl.write(a.out, (to_row(r) for r in keep))

    if a.negatives:
        neg = []
        for r in rounds:
            if r['goal_achieved'] not in ('no', 'unclear'):
                continue
            row = to_row(r)
            row['why_failed'] = r['why']
            row['broken_now'] = r['broken_now']
            neg.append(row)
        jsonl.write(a.negatives, neg)
        print(f'negatives written : {len(neg):,} -> {a.negatives}  (NOT for SFT)')

    print(f'rounds found      : {len(rounds):,}')
    print(f'SFT rows written  : {len(keep):,} -> {a.out}')
    print(f"games covered     : {len({r['task'] for r in keep}):,}")
    for k in sorted(stats):
        print(f'  {k:28s} {stats[k]:,}')
    if keep:
        pb = sorted(r['patch_bytes'] for r in keep)
        raw = sorted(r['raw_patch_bytes'] for r in keep)
        mid = len(pb) // 2
        print(f'patch size        : median {pb[mid]:,}B  max {pb[-1]:,}B'
              + (f'  (raw median {raw[mid]:,}B, {raw[mid] / max(pb[mid], 1):.0f}x)'
                 if a.minimize else ''))
    return 0 if keep else 1


if __name__ == '__main__':
    raise SystemExit(main())
