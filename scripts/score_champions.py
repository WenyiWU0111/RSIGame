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
"""Official score of each game's Value-Monitor champion at the last checkpoint.

    score_champions.py --run RUN --vm RSIGAME_MONITOR_OUT --r0-csv r0_scores/godot_glm.csv \
                       --games handoff/main_glm120.txt --out champ_glm30.csv [--jobs 3] [--at 30]

The champion is read from RSIGAME_MONITOR_OUT/<game>.jsonl (vm_replay.py) at checkpoint --at.
Champion round 0 is the base game: its row copies the 3-pass base score from
--r0-csv. Otherwise RUN/runs/<game>/round_rr/tree is scored with 3 passes by the
hosted judge (score_r0.py's settings) into round_rr/score_official. A game whose
replay stops before --at is scored at its last checkpoint, noted in the row.
Resumable: games with a row are skipped.
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_r0 import LOOP, PY, dims, judge_env, reject_failed_judge  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--vm', required=True)
    ap.add_argument('--r0-csv', required=True)
    ap.add_argument('--games', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--at', type=int, default=30)
    a = ap.parse_args()
    run, vm, out_csv = Path(a.run).resolve(), Path(a.vm).resolve(), Path(a.out).resolve()
    r0 = {r['task']: r for r in csv.DictReader(open(a.r0_csv))}
    games = Path(a.games).read_text().split()
    done = {r['game'] for r in csv.DictReader(out_csv.open())} if out_csv.exists() else set()
    fields = ['game', 'champion_round', 'at', 'reward', 'passes', 'BUILD', 'M', 'D', 'V', 'A',
              'failed_load', 'minutes', 'note']

    def write_row(row):
        new = not out_csv.exists()
        with out_csv.open('a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, '') for k in fields})

    def say(msg):
        with Path(str(out_csv) + '.log').open('a') as f:
            f.write(time.strftime('%m-%d %H:%M ') + msg + '\n')

    def one(g):
        f = vm / f'{g}.jsonl'
        recs = [json.loads(x) for x in f.read_text().splitlines()] if f.is_file() else []
        recs = [x for x in recs if x['round'] <= a.at]
        if not recs:
            say(f'{g} no replay yet, skipped')
            return
        last = max(recs, key=lambda x: x['round'])
        c, note = last['champion'], ('' if last['round'] == a.at else f"replay ends at r{last['round']}")
        if c is None:
            say(f'{g} no champion (no recordings), skipped')
            return
        if c == 0:
            b = r0[g]
            write_row({**b, 'game': g, 'champion_round': 0, 'at': last['round'], 'minutes': 0,
                       'note': ('base game kept; ' + note).strip('; ')})
            say(f"{g} champion r0 = {float(b['reward']):.4f} (base score)")
            return
        t0, tree = time.time(), run / 'runs' / g / f'round_{c:02d}' / 'tree'
        out = tree.parent / 'score_official'
        r = subprocess.run([PY, str(LOOP / 'score_game.py'), '--project', str(tree), '--game', g,
                            '--output', str(out), '--passes', '3', '--judge', 'openai',
                            '--judge-model', 'qwen/qwen3.8-27b'],
                           cwd=str(LOOP), env=judge_env(''), capture_output=True, text=True)
        if out.exists():
            (out / 'score_game.log').write_text(r.stdout + r.stderr)
        reject_failed_judge(out)
        rewards = [float((out / f'p{p}' / 'reward.txt').read_text())
                   for p in (1, 2, 3) if (out / f'p{p}' / 'reward.txt').is_file()]
        if len(rewards) < 3:
            say(f'{g} r{c:02d} INCOMPLETE rc={r.returncode} passes={rewards} (re-run to retry)')
            return
        row = {'game': g, 'champion_round': c, 'at': last['round'], 'reward': round(sum(rewards) / 3, 6),
               'passes': json.dumps(rewards), **dims(out), 'minutes': round((time.time() - t0) / 60, 1),
               'note': note}
        write_row(row)
        say(f"{g} champion r{c:02d} = {row['reward']:.4f} {rewards} failed_load={row['failed_load']}")

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(one, [g for g in games if g not in done]))
    say('finished')


if __name__ == '__main__':
    main()
