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
"""Score every Nth round of a Phaser run, off the snapshots. The web twin of
`score_checkpoints.py`.

The differences from the Godot scorer are the whole reason this file exists:

  * no import step and no load check. The web build gate
    (`tools/web_build_check.py`) serves the snapshot's own `dist/` and IS the
    load check. The loop deletes `dist/` when a round's build fails, so a
    snapshot of a broken round scores BUILD=0 rather than replaying the last
    build that worked.
  * `--replay-engine web`, and the rubric comes from `tasks-web/`, in a
    gamecraft-bench checkout built for the web tasks.
  * a tree is a Phaser project when it has `package.json`, not `project.godot`.

Everything else is deliberately identical to the Godot path -- three
independent passes, the same judge at temperature 0 with thinking off, the same
rule that a judge which did not cover every demo is not a score -- so a
checkpoint and its round-0 base game are comparable.

    RSIGAME_MONITOR_WEB_BENCH=/path/to/gamecraft-bench-web \\
    scripts/score_checkpoints_web.py RUN_ROOT [--once]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
WEB = Path(os.environ.get('RSIGAME_MONITOR_WEB_BENCH') or os.environ.get('GAMECRAFT_BENCH') or '')
PY = os.environ.get('RSIGAME_PATHS_PYTHON') or sys.executable
EVERY = int(os.environ.get('RSIGAME_MONITOR_CHECKPOINT_EVERY') or 3)
PASSES = int(os.environ.get('RSIGAME_MONITOR_PASSES') or 3)
JOBS = int(os.environ.get('RSIGAME_MONITOR_JOBS') or 2)
# A lock is only meaningful while its writer is alive. A scorer killed
# mid-checkpoint never reaches its `finally`, and the queue would then skip that
# round for good -- which happened twice, silently, for hundreds of checkpoints.
# Nothing here takes longer than an hour, so an older lock is abandoned.
STALE_LOCK_S = int(os.environ.get('RSIGAME_MONITOR_STALE_LOCK_S') or 5400)


def say(*bits) -> None:
    line = time.strftime('%H:%M ') + ' '.join(str(b) for b in bits)
    print(line, flush=True)


def judge_env() -> dict:
    """Miss one of these and every rubric item scores 0 while BUILD stays 1.0
    and the demos record normally -- a silent zero, not an error."""
    e = dict(os.environ)
    base = os.environ.get('RSIGAME_JUDGE_BASE_URL') or 'http://127.0.0.1:8038/v1'
    key = os.environ.get('RSIGAME_JUDGE_API_KEY') or 'local'
    e.update(GAMECRAFT_BENCH_JUDGE_NO_THINK='1', GAMECRAFT_BENCH_JUDGE_TEMPERATURE='0',
             GAMECRAFT_BENCH_JUDGE_OPENAI_BASE_URL=base,
             GAMECRAFT_BENCH_JUDGE_OPENAI_API_KEY=key, OPENAI_API_KEY=key)
    return e


def one_pass(tree: Path, game: str, out: Path, env: dict) -> float | None:
    """-> the reward, or None when the pass must not count."""
    rubric = WEB / 'tasks-web' / game / 'tests' / 'rubric.json'
    subprocess.run(['rm', '-rf', str(out)])
    p = subprocess.run(
        [PY, '-m', 'gamecraft_bench.verifier', '--project', str(tree), '--rubric', str(rubric),
         '--output', str(out), '--judge', 'openai',
         '--judge-model', os.environ.get('RSIGAME_JUDGE_MODEL') or 'qwen38-27b',
         '--replay-engine', 'web'],
        cwd=str(WEB), env=env, capture_output=True, text=True, timeout=5400)
    (out.parent / f'{out.name}.log').write_text(p.stdout + p.stderr)
    try:
        bd = json.loads((out / 'breakdown.json').read_text())
        reward = float((out / 'reward.txt').read_text().strip())
    except Exception:
        return None
    # A judge that did not reach every demo is not a lower score, it is no score.
    if bd.get('build_ok') and not (bd.get('judge_coverage') or {}).get('complete'):
        return None
    return reward


def score(game: str, r: int, tree: Path) -> float | None:
    out, lock = tree.parent / 'score', tree.parent / '.scoring'
    try:
        if not (tree / 'package.json').is_file() or (out / 'summary.json').is_file():
            return None
        with open(lock, 'x') as f:
            f.write(str(os.getpid()))
    except Exception as exc:
        say(f'  {game} r{r:02d} could not take the lock: {type(exc).__name__}: {str(exc)[:120]}')
        return None
    env = judge_env()
    try:
        say(f'{game} r{r:02d} scoring')
        t0 = time.time()
        out.mkdir(parents=True, exist_ok=True)
        rewards = []
        for k in range(1, PASSES + 1):
            reward = None
            for attempt in (1, 2):
                reward = one_pass(tree, game, out / f'p{k}', env)
                if reward is not None:
                    break
                say(f'  {game} r{r:02d} p{k} judge did not cover every demo, voided ({attempt})')
            if reward is None:
                say(f'  {game} r{r:02d} p{k} voided twice; this checkpoint stays unscored')
                return None
            rewards.append(reward)
        (out / 'summary.json').write_text(json.dumps(
            {'game': game, 'source': str(tree), 'passes': PASSES, 'engine': 'web',
             'rewards': rewards}, indent=1))
        mean = sum(rewards) / len(rewards)
        say(f'  {game} r{r:02d} = {mean:.4f}  {rewards}  {(time.time() - t0) / 60:.0f}min')
        return mean
    except Exception as exc:
        say(f'  {game} r{r:02d} scoring crashed: {type(exc).__name__}: {str(exc)[:160]}')
        return None
    finally:
        lock.unlink(missing_ok=True)


def pending() -> list[tuple[str, int, Path]]:
    out = []
    for gd in sorted((ROOT / 'runs').glob('*')):
        for rd in sorted(gd.glob('round_*')):
            try:
                r = int(rd.name.split('_')[1])
            except Exception:
                continue
            tree = rd / 'tree'
            if r % EVERY or not (tree / 'package.json').is_file():
                continue
            if (rd / 'score' / 'summary.json').is_file():
                continue
            lk = rd / '.scoring'
            if lk.exists():
                try:
                    if time.time() - lk.stat().st_mtime < STALE_LOCK_S:
                        continue
                    lk.unlink(missing_ok=True)
                    say(f'  {gd.name} r{r:02d} stale lock cleared, requeued')
                except OSError:
                    continue
            out.append((gd.name, r, tree))
    return out


def main() -> int:
    if ROOT is None or not WEB.is_dir():
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    once = '--once' in sys.argv
    say(f'web checkpoint scoring · root={ROOT} · {JOBS} at a time · {PASSES} passes · every {EVERY} rounds')
    idle = 0
    with ThreadPoolExecutor(max_workers=JOBS) as pool:
        while True:
            todo = pending()
            if todo:
                idle = 0
                say(f'{len(todo)} to score: ' + ', '.join(f'{g} r{r}' for g, r, _ in todo[:8]))
                list(pool.map(lambda a: score(*a), todo))
            elif once:
                return 0
            else:
                idle += 1
                if idle % 20 == 1:
                    say('nothing to score')
            time.sleep(60)


if __name__ == '__main__':
    raise SystemExit(main())
