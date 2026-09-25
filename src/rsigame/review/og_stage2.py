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
"""The stage after convergence: build the second round of review cases.

A branch is saturated for this stage once its champion has been unchanged for
three consecutive checkpoints. The next stage restarts from the build it
delivered: that build is the new P*, the director (a person or a model) plays
it again and writes a second brief, and the loop runs on with that brief.

This script only does the "make the cases" step, because that step needs no
judgement: which round is champion, where its tree is, and what happened during
the stage are all already written in the branch directory. The review itself
(a person playing in the page, or a model-director subagent) and the launch of
the next stage happen outside it.

Cases are written to the same directory and in the same format as the first
round, with ids like `pilot2s2-03-model` (the first round's number plus the
branch), so that both rounds appear in one list after the review service
restarts, and neither the page nor og_branch needs changing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

from . import cases as C


def stage_summary(recs: list[dict], rounds: list[dict]) -> str:
    """What happened during the stage: how each checkpoint was judged and what
    each round worked on. The first round's version of this field was computed
    from the Value Stop replay; here the branch produced it itself, so the
    record is read directly."""
    out = [f'Autonomous development rounds completed: {len(rounds)}', '',
           'Champion (best build so far), checked every 3 rounds:']
    for r in recs:
        if r['round'] == 0:
            continue
        ev = r.get('event')
        if ev == 'replaced':
            out.append(f"  R{r['round']:02d}: new champion")
        elif ev in ('build_failed', 'parse_error'):
            out.append(f"  R{r['round']:02d}: the build was broken, not adopted "
                       f"(champion still R{r['champion']:02d})")
        else:
            out.append(f"  R{r['round']:02d}: no update (champion still R{r['champion']:02d})")
    champ = recs[-1]['champion']
    out += ['', f'Build under review: R{champ:02d}', '',
            f'Development focus of rounds 1-{len(rounds)}:']
    kinds = {}
    for x in rounds:
        k = 'visual presentation' if x.get('kind') == 'art' else (
            'interaction feedback' if x.get('focus') == 'feedback' else 'fixing functional defects')
        kinds[k] = kinds.get(k, 0) + 1
    for k, n in sorted(kinds.items(), key=lambda t: -t[1]):
        out.append(f'  - {k}: {n} round(s)')
    return '\n'.join(out) + '\n'


def build(branch_root: Path, case_p: Path, out_dir: Path, branch: str) -> dict:
    case = json.loads(case_p.read_text())
    g = case['game_id']
    st_p = branch_root / 'og' / f'{g}.json'
    jl = branch_root / 'vm' / f'{g}.jsonl'
    if not st_p.is_file() or not jl.is_file():
        return {'game': g, 'branch': branch, 'skipped': 'no state yet'}
    st = json.loads(st_p.read_text())
    if st.get('state') != 'finished':
        return {'game': g, 'branch': branch, 'skipped': f"state {st.get('state')}"}
    recs = [json.loads(x) for x in jl.read_text().splitlines()]
    champ = recs[-1]['champion']
    rundir = branch_root / 'runs' / g
    rounds = json.loads((rundir / 'rounds.json').read_text())
    tree = (Path(case['tree_src']) if champ == 0
            else rundir / f'round_{champ:02d}' / 'tree')
    if not (tree / 'project.godot').is_file():
        return {'game': g, 'branch': branch, 'error': f'no tree at {tree}'}
    cid = f"pilot2s2-{case['case_id'].split('-')[-1]}-{branch}"
    out = out_dir / f'{cid}.json'
    if out.is_file():
        return {'game': g, 'branch': branch, 'case_id': cid, 'champion': champ, 'existing': True}
    new = dict(case)
    new.update({
        'case_id': cid,
        'run_id': 'pilot2s2',
        'stage': 2,
        'from': {'run_id': case['run_id'], 'case_id': case['case_id'], 'branch': branch,
                 'pstar': case['checkpoint_id'], 'brief': str(
                     json.loads((rundir / 'og_config.json').read_text()).get('brief') or '')},
        'checkpoint_id': f'R{champ:02d}' if champ else case['checkpoint_id'],
        'stop_round': len(rounds),
        'champion_updates': [r['round'] for r in recs if r.get('event') == 'replaced' and r['round']],
        'tree_src': str(tree.resolve()),
        'tree_hash': C.tree_hash(tree),
        'saturation_summary': stage_summary(recs, rounds),
    })
    out.write_text(json.dumps(new, indent=1, ensure_ascii=False))
    return {'game': g, 'branch': branch, 'case_id': cid, 'champion': champ,
            'rounds': len(rounds), 'tree': str(tree)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--branches', required=True)
    ap.add_argument('--cases', required=True)
    ap.add_argument('--branch', default='human,model')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    out_dir = Path(a.out) if a.out else Path(a.cases)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for br in a.branch.split(','):
        for cp in sorted(Path(a.cases).glob('*.json')):
            case = json.loads(cp.read_text())
            if case.get('run_id') != 'pilot1':
                continue                       # stage-2 cases do not spawn stage 3 here
            root = Path(a.branches) / br
            if not (root / 'runs' / case['game_id'] / 'og_config.json').is_file():
                continue
            rows.append(build(root, cp, out_dir, br))
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == '__main__':
    main()
