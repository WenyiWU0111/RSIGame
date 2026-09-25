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
"""Official scores for finished outer-guidance branches, and the pilot table.

    og_score.py --branches BRANCHES_DIR --cases review/cases/pilot1 [--jobs 2]

For every branch whose supervisor reports state=finished, the champion at the
stop is scored with the same judge and passes as the checkpoints P* was picked
from (local vLLM qwen38-27b, 3 passes) into runs/<game>/round_XX/score_official.
A champion of round 0 is P* itself and reuses P*'s own score. A pass whose
judge failed, or whose recordings are grey (>20 'Failed loading resource'), is
thrown away and scored again on the next invocation. Resumable; run it as often
as you like.

Writes BRANCHES_DIR/og_results.csv and og_results.md: per game, P*'s score and,
per branch, stop reason, rounds run, champion round, champion score, lift over P*.
"""
import sys
import argparse
import csv
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .. import paths

RUN = paths.run_root()
PY = os.environ.get('RSIGAME_PATHS_PYTHON') or sys.executable   # subprocesses use this same interpreter
BRANCHES = ('none', 'human', 'model')


def mean_rewards(score_dir: Path):
    try:
        rw = json.loads((score_dir / 'summary.json').read_text()).get('rewards') or []
        rw = [x for x in rw if x == x]
        return (sum(rw) / len(rw), len(rw)) if rw else (None, 0)
    except Exception:
        return None, 0


def judge_env() -> dict:
    e = dict(os.environ)
    e.update({'GAMECRAFT_BENCH': str(paths.bench()),
              'PYTHONPATH': str(paths.agent_repo()),
              'GAMECRAFT_BENCH_JUDGE_OPENAI_BASE_URL': 'http://127.0.0.1:8038/v1',
              'GAMECRAFT_BENCH_JUDGE_OPENAI_API_KEY': 'local'})
    return e


def score_official(tree: Path, game: str) -> tuple:
    out = tree.parent / 'score_official'
    m, n = mean_rewards(out)
    if n >= 3:
        return m, 'ok'
    if not (tree / '.godot').is_dir():
        subprocess.run(['godot', '--headless', '--path', str(tree), '--import', '--quit'],
                       capture_output=True, timeout=900)
    r = subprocess.run([PY, 'score_game.py', '--project', str(tree), '--game', game, '--output', str(out),
                        '--passes', '3', '--judge', 'openai', '--judge-model', 'qwen38-27b'],
                       cwd=str(RUN), env=judge_env(), capture_output=True, text=True, timeout=10800)
    (tree.parent / 'score_official.log').write_text(r.stdout[-20000:] + r.stderr[-20000:])
    bad = []
    for p in sorted(out.glob('p*')):
        errs = ' '.join(str(x) for x in (json.loads((p / 'breakdown.json').read_text()).get('errors') or [])) \
            if (p / 'breakdown.json').is_file() else 'no breakdown'
        fails = sum(f.read_text(errors='ignore').count('Failed loading resource')
                    for f in p.glob('demos/*/logs/godot.log'))
        if 'judge failed' in errs or 'API call failed' in errs or 'Connection error' in errs or fails > 20:
            bad.append(p.name)
    if bad:
        for b in bad:
            shutil.rmtree(out / b, ignore_errors=True)
        (out / 'summary.json').unlink(missing_ok=True)
        return None, f'rejected {bad}; rescore next run'
    m, n = mean_rewards(out)
    return m, 'ok' if n >= 3 else f'only {n} passes'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--branches', required=True)
    ap.add_argument('--cases', required=True)
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--checkpoints', action='store_true',
                    help='also score every Monitor checkpoint of every finished branch; writes og_ckpt_scores.csv')
    a = ap.parse_args()
    B = Path(a.branches)
    cases = [json.loads(p.read_text()) for p in sorted(Path(a.cases).glob('*.json'))]
    rows, jobs = [], []
    for c in cases:
        g, pr = c['game_id'], int(c['checkpoint_id'][1:])
        src = Path(c['tree_src']).parents[3] / 'runs' / g / f'round_{pr:02d}' / 'score'
        pstar, _ = mean_rewards(src)
        for br in BRANCHES:
            st_path = B / br / 'og' / f'{g}.json'
            st = json.loads(st_path.read_text()) if st_path.is_file() else {}
            row = {'case': c['case_id'], 'game': g, 'pstar': c['checkpoint_id'],
                   'pstar_score': round(100 * pstar, 1) if pstar is not None else None, 'branch': br,
                   'state': st.get('state'), 'rounds': st.get('rounds_done'),
                   'stop': (st.get('stop') or {}).get('reason'), 'stop_ckpt': (st.get('stop') or {}).get('checkpoint'),
                   'champion': st.get('champion'), 'champion_score': None, 'lift': None, 'note': ''}
            rows.append(row)
            if st.get('state') == 'finished' and st.get('champion') is not None:
                if st['champion'] == 0:
                    row['champion_score'], row['note'] = row['pstar_score'], 'champion is P*'
                else:
                    tree = B / br / 'runs' / g / f"round_{st['champion']:02d}" / 'tree'
                    jobs.append((row, tree, g))

    def run(job):
        row, tree, g = job
        m, note = score_official(tree, g)
        row['champion_score'] = round(100 * m, 1) if m is not None else None
        row['note'] = note

    ck_rows = []
    if a.checkpoints:
        for c in cases:
            g = c['game_id']
            for br in BRANCHES:
                vm = B / br / 'vm' / f'{g}.jsonl'
                if not vm.is_file():
                    continue
                for line in vm.read_text().splitlines():
                    rec = json.loads(line)
                    if rec['round'] == 0:
                        continue
                    tree = B / br / 'runs' / g / f"round_{rec['round']:02d}" / 'tree'
                    row = {'game': g, 'branch': br, 'round': rec['round'], 'monitor_event': rec['event'],
                           'proxy_old': rec.get('old'), 'proxy_new': rec.get('new'), 'proxy_delta': rec.get('delta'),
                           'champion_after': rec.get('champion'), 'official': None, 'note': ''}
                    ck_rows.append(row)
                    if (tree / 'project.godot').is_file():
                        jobs.append((row, tree, g))

    def run_any(job):
        row, tree, g = job
        m, note = score_official(tree, g)
        key = 'official' if 'official' in row else 'champion_score'
        row[key] = round(100 * m, 1) if m is not None else None
        row['note'] = note

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(run_any, jobs))
    if ck_rows:
        with open(B / 'og_ckpt_scores.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(ck_rows[0]))
            w.writeheader()
            w.writerows(ck_rows)
    for row in rows:
        if row['champion_score'] is not None and row['pstar_score'] is not None:
            row['lift'] = round(row['champion_score'] - row['pstar_score'], 1)
    with open(B / 'og_results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ['| game | P* | P* score | branch | state | rounds | stop | champion | champion score | lift |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['game']} | {r['pstar']} | {r['pstar_score']} | {r['branch']} | {r['state']} | "
                     f"{r['rounds']} | {r['stop']} | {r['champion']} | {r['champion_score']} | {r['lift']} |")
    (B / 'og_results.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
