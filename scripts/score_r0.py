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
"""Score frozen base games (R0), 3 passes each, with the hosted judge.

    score_r0.py --corpus godot/corpus_glm --tasks handoff/r0_glm_remote.txt \
                --out r0_scores/godot_glm [--jobs 3] [--local-judge URL]

Judge: Qwen3.8-27B on OpenRouter, provider pinned to Alibaba with no fallback,
thinking off, temperature 0 -- the same settings as the hosted-judge scorers on
the original machine (needs the provider-pin bench patch). --local-judge URL
uses an OpenAI-compatible vLLM endpoint serving `qwen38-27b` instead.

Tasks are scored IN FILE ORDER (the handoff lists are already reversed so the two
machines work from opposite ends). One CSV row per task in OUT.csv; per-task
output in OUT/<task>/p1..p3. Re-running skips tasks with a row, and score_game.py
skips passes already scored. A task with no project in the corpus scores 0.
A pass whose judge failed is moved aside and scored again on the next run.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LOOP = Path(os.environ.get('RSIGAME_LOOP_LOOP') or Path(__file__).resolve().parents[2])
PY = os.environ.get('RSIGAME_PATHS_PYTHON') or sys.executable


def dims(out_dir: Path) -> dict:
    """Dimension means over the 3 passes, plus 'Failed loading resource' lines:
    many of them mean the scoring copy did not import (grey screen) -- rescore."""
    acc = {k: [] for k in ('BUILD', 'M', 'D', 'V', 'A')}
    failed = 0
    for p in (1, 2, 3):
        try:
            v = json.loads((out_dir / f'p{p}' / 'breakdown.json').read_text()).get('variables') or {}
        except Exception:
            continue
        acc['BUILD'].append(v.get('BUILD', 0))
        for d in 'MDVA':
            ks = [k for k in v if k[0] == d and k[1:].isdigit()]
            if ks:
                acc[d].append(sum(v[k] for k in ks) / len(ks))
        for lg in (out_dir / f'p{p}' / 'demos').glob('*/logs/godot.log'):
            failed += lg.read_text(errors='ignore').count('Failed loading resource')
    row = {k: (round(sum(x) / len(x), 4) if x else None) for k, x in acc.items()}
    row['failed_load'] = failed
    return row


def judge_env(local: str) -> dict:
    e = dict(os.environ)
    if local:
        e['GAMECRAFT_BENCH_JUDGE_OPENAI_BASE_URL'] = local
        e['GAMECRAFT_BENCH_JUDGE_OPENAI_API_KEY'] = 'local'
    else:
        e['GAMECRAFT_BENCH_JUDGE_OPENAI_BASE_URL'] = 'https://openrouter.ai/api/v1'
        e['GAMECRAFT_BENCH_JUDGE_OPENAI_API_KEY'] = e.get('OPENROUTER_API_KEY', '')
        e['GAMECRAFT_BENCH_JUDGE_PROVIDER'] = e.get('RSIGAME_JUDGE_JUDGE_PROVIDER') or 'Alibaba'
    return e


def reject_failed_judge(out: Path):
    for p in (1, 2, 3):
        d = out / f'p{p}'
        try:
            errs = ' '.join(json.loads((d / 'breakdown.json').read_text()).get('errors') or [])
        except Exception:
            continue
        if 'judge failed' in errs or 'API call failed' in errs:
            d.rename(out / f'p{p}.rejected.{int(time.time())}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--tasks', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--local-judge', default='')
    a = ap.parse_args()
    corpus, out_root = Path(a.corpus).resolve(), Path(a.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path, log = Path(str(out_root) + '.csv'), Path(str(out_root) + '.log')
    tasks = Path(a.tasks).read_text().split()
    done = {r['task'] for r in csv.DictReader(csv_path.open())} if csv_path.exists() else set()
    todo = [t for t in tasks if t not in done]
    model = 'qwen38-27b' if a.local_judge else 'qwen/qwen3.8-27b'

    def say(msg):
        with log.open('a') as f:
            f.write(time.strftime('%m-%d %H:%M ') + msg + '\n')

    def write_row(row):
        new = not csv_path.exists()
        with csv_path.open('a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    say(f'start tasks={len(tasks)} todo={len(todo)} jobs={a.jobs} judge={a.local_judge or "openrouter/Alibaba"}')

    def one(t):
        t0, out = time.time(), out_root / t
        if not (corpus / t / 'project.godot').is_file():
            write_row({'task': t, 'reward': 0.0, 'passes': '[0.0, 0.0, 0.0]', 'BUILD': 0, 'M': 0, 'D': 0,
                       'V': 0, 'A': 0, 'failed_load': 0, 'minutes': 0, 'note': 'no generated project'})
            say(f'{t} = 0 (no project)')
            return
        r = subprocess.run([PY, str(LOOP / 'score_game.py'), '--project', str(corpus / t), '--game', t,
                            '--output', str(out), '--passes', '3', '--judge', 'openai', '--judge-model', model],
                           cwd=str(LOOP), env=judge_env(a.local_judge), capture_output=True, text=True)
        if out.exists():
            (out / 'score_game.log').write_text(r.stdout + r.stderr)
        reject_failed_judge(out)
        rewards = [float((out / f'p{p}' / 'reward.txt').read_text())
                   for p in (1, 2, 3) if (out / f'p{p}' / 'reward.txt').is_file()]
        if len(rewards) < 3:
            say(f'{t} INCOMPLETE rc={r.returncode} passes={rewards} (re-run to retry)')
            return
        row = {'task': t, 'reward': round(sum(rewards) / 3, 6), 'passes': json.dumps(rewards),
               **dims(out), 'minutes': round((time.time() - t0) / 60, 1), 'note': ''}
        write_row(row)
        say(f"{t} = {row['reward']:.4f} {rewards} failed_load={row['failed_load']} {row['minutes']}min")

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(one, todo))
    say('finished')


if __name__ == '__main__':
    main()
