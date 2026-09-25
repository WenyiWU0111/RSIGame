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
"""Build the director-review case configs for one outer-guidance run.

    cases.py --run-id pilot1 --out review/cases/pilot1 \
             --runs RUN_ROOT --r0-runs R0_ROOT --vm RSIGAME_MONITOR_OUT --games g1,g2,...

For each game, the saturated checkpoint P* is what Value Stop would deliver:
walking the offline Monitor's checkpoints (vm_replay.py), the run stops at the
first checkpoint where the champion has gone unchanged for STOP_AFTER
consecutive checkpoints, and P* is the champion at that moment.

What a case carries is exactly what a director may see (design doc sections 10-11):
the public game specification (the player-facing part of the task instruction),
the playable P* build, and a minimal development context -- rounds completed,
champion history, how many rounds went to each development focus. Never the
official score, the rubric, source code or the repair agent's reasoning.
"""
import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from .. import paths

CONFIG_VERSION = 'outer-guidance-cases/1'
STOP_AFTER = 3           # consecutive unchanged checkpoints that stop the run
BUDGET_ACTIONS = 60
BUDGET_RESETS = 2
# rounds.json labels -> what the director is told. A round with no focus or
# direction label is a normal (defect-repair) round.
FOCUS_NAMES = {'presentation': 'visual presentation', 'feedback': 'interaction feedback',
               'functional': 'fixing functional defects', 'content': 'content / progression'}
# Player-facing sections of a gamecraft-bench instruction. The rest (assets,
# project layout, demo trace format) is harness engineering, not the design.
SPEC_SECTIONS = ('Core Vision', 'What the Player Experiences')


def tree_hash(tree: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(tree.rglob('*')):
        rel = p.relative_to(tree).as_posix()
        if not p.is_file() or rel.startswith(('.godot/', '.qwen/')) or '/.godot/' in rel:
            continue
        h.update(rel.encode() + b'\0')
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()[:16]


def public_spec(instruction: str) -> str:
    title = instruction.splitlines()[0].lstrip('# ').strip()
    parts = [f'# {title}']
    for sec in SPEC_SECTIONS:
        m = re.search(rf'^## {re.escape(sec)}\n(.*?)(?=^## |\Z)', instruction, re.S | re.M)
        if m:
            parts.append(f'## {sec}\n{m.group(1).strip()}')
    return '\n\n'.join(parts) + '\n'


def value_stop(recs: list[dict]) -> tuple[int, int, list[int]]:
    """(stop checkpoint, P*, rounds at which the champion changed up to the stop)."""
    recs = sorted(recs, key=lambda x: x['round'])
    streak, updates = 0, []
    for prev, cur in zip(recs, recs[1:]):
        if cur['champion'] != prev['champion']:
            streak = 0
            updates.append(cur['round'])
        else:
            streak += 1
        if streak == STOP_AFTER:
            return cur['round'], cur['champion'], updates
    last = recs[-1]
    return last['round'], last['champion'], updates


def summary_text(rounds: list[dict], stop: int, pstar: int, updates: list[int], every: int = 3) -> str:
    focus = Counter(FOCUS_NAMES.get(str(r.get('focus') or r.get('direction')
                                        or ('functional' if r.get('kind') == 'normal' else r.get('kind'))),
                                    'other')
                    for r in rounds if r.get('round', 0) <= stop)
    cks = list(range(every, stop + 1, every))
    lines = [f'Autonomous development rounds completed: {stop}', '',
             'Champion (best build so far), checked every 3 rounds:']
    champ = 0
    for r in cks:
        if r in updates:
            lines.append(f'  R{r:02d}: new champion')
            champ = r
        else:
            lines.append(f'  R{r:02d}: no update (champion still R{champ:02d})')
    lines += ['', f'Build under review: R{pstar:02d}'
              + (' (the original generated game; no round improved on it)' if pstar == 0 else ''),
              '', f'Development focus of rounds 1-{stop}:']
    lines += [f'  - {k}: {v} round(s)' for k, v in focus.most_common()]
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--runs', required=True, help='run root with runs/<game>/round_XX/tree')
    ap.add_argument('--r0-runs', required=True, help='run root whose runs/<game>/round_00/tree is the base game')
    ap.add_argument('--vm', required=True, help='vm_replay output with <game>.jsonl')
    ap.add_argument('--games', required=True)
    ap.add_argument('--bench-tasks', default=os.environ.get('GAMECRAFT_BENCH_TASKS')
                    or str(paths.bench_tasks()))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, g in enumerate(a.games.split(','), 1):
        recs = [json.loads(x) for x in (Path(a.vm) / f'{g}.jsonl').read_text().splitlines()]
        stop, pstar, updates = value_stop(recs)
        tree = (Path(a.r0_runs) / 'runs' / g / 'round_00' / 'tree' if pstar == 0
                else Path(a.runs) / 'runs' / g / f'round_{pstar:02d}' / 'tree')
        rounds = json.loads((Path(a.runs) / 'runs' / g / 'rounds.json').read_text())
        spec = public_spec((Path(a.bench_tasks) / g / 'instruction.md').read_text())
        case = {
            'config_version': CONFIG_VERSION,
            'case_id': f'{a.run_id}-{i:02d}',
            'run_id': a.run_id,
            'game_id': g,
            'title': spec.splitlines()[0].lstrip('# ').strip(),
            'checkpoint_id': f'R{pstar:02d}',
            'stop_round': stop,
            'champion_updates': updates,
            'tree_src': str(tree.resolve()),
            'tree_hash': tree_hash(tree),
            'engine': 'godot',
            'spec': spec,
            'spec_hash': hashlib.sha256(spec.encode()).hexdigest()[:16],
            'saturation_summary': summary_text(rounds, stop, pstar, updates),
            'budget': {'actions': BUDGET_ACTIONS, 'resets': BUDGET_RESETS},
        }
        (out / f"{case['case_id']}.json").write_text(json.dumps(case, indent=1, ensure_ascii=False))
        print(case['case_id'], g, 'stop', stop, 'P*', case['checkpoint_id'], case['tree_hash'])


if __name__ == '__main__':
    main()
