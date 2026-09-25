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
"""Replay the ORIGINAL game's demos on both builds of every pair, with the
scoring system's own replay and frame sampler.

Per side: copy the round tree, replace its demo_outputs with the original
game's demos, import it once up front, then run score_game.py with the stub
judge and one pass. That is the official path end to end -- tail padding,
import, Xvfb replay, frames every 0.5 s -- minus the scoring. A replay whose
logs show the partial-import grey screen ("Failed loading resource" > 20) is
thrown away and redone, at most twice; it is never handed to the judge.

    replay.py MANIFEST OUT_DIR [--jobs 4] [--only game1,game2]
"""
import sys
import argparse
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from ... import paths

RUN = paths.run_root()
PY = os.environ.get('RSIGAME_PATHS_PYTHON') or sys.executable   # subprocesses use this same interpreter
GREY = 20


def env():
    e = dict(os.environ)
    e['GAMECRAFT_BENCH'] = str(paths.bench())
    e['PYTHONPATH'] = str(paths.agent_repo())
    return e


def failed_loads(score_dir: Path) -> int:
    return sum(p.read_text(errors='ignore').count('Failed loading resource')
               for p in score_dir.glob('p1/demos/*/logs/godot.log'))


def one_side(pair, arm, out_root: Path, log):
    game = pair['game']
    side = out_root / game / arm
    done = side / 'score' / 'p1' / 'reward.txt'
    if done.is_file() and failed_loads(side / 'score') <= GREY:
        return 'cached'
    for attempt in (1, 2, 3):
        shutil.rmtree(side, ignore_errors=True)
        tree = side / 'tree'
        shutil.copytree(pair['trees'][arm], tree, symlinks=True,
                        ignore=shutil.ignore_patterns('.godot'))
        shutil.rmtree(tree / 'demo_outputs', ignore_errors=True)
        (tree / 'demo_outputs').mkdir()
        for d in pair['demos']:
            shutil.copy2(d, tree / 'demo_outputs' / Path(d).name)
        subprocess.run(['godot', '--headless', '--path', str(tree), '--import', '--quit'],
                       capture_output=True, timeout=900)
        t0 = time.time()
        r = subprocess.run([PY, str(RUN / 'score_game.py'), '--project', str(tree), '--game', game,
                            '--output', str(side / 'score'), '--passes', '1',
                            '--judge', 'stub', '--judge-model', '1.0'],
                           cwd=str(RUN), env=env(), capture_output=True, text=True, timeout=7200)
        (side / 'replay.log').write_text(r.stdout + r.stderr)
        bad = failed_loads(side / 'score')
        frames = len(list((side / 'score').glob('p1/demos/*/frames/*')))
        if done.is_file() and bad <= GREY and frames:
            log(f'{game}/{arm}: {frames} frames, {time.time() - t0:.0f}s (attempt {attempt})')
            return 'ok'
        log(f'{game}/{arm}: attempt {attempt} rejected (rc={r.returncode}, failed_loads={bad}, frames={frames})')
    return 'failed'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('manifest')
    ap.add_argument('out')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = json.loads(Path(a.manifest).read_text())['pairs']
    if a.only:
        keep = set(a.only.split(','))
        pairs = [p for p in pairs if p['game'] in keep]
    logf = out / 'replay_progress.log'

    def log(msg):
        line = time.strftime('%H:%M ') + msg
        print(line, flush=True)
        with logf.open('a') as f:
            f.write(line + '\n')

    jobs = [(p, arm) for p in pairs for arm in ('self', 'robin')]
    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(lambda j: one_side(j[0], j[1], out, log), jobs))
    log(f'replay finished: {sum(r in ("ok", "cached") for r in res)}/{len(res)} sides ok')


if __name__ == '__main__':
    main()
