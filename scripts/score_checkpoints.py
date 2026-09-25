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
"""Score every third round's tree, off the snapshots, while the run continues.

NOT INSIDE THE LOOP. The rubric is held out: nothing the loop reads may import
`score_game`. It is also the heavier half of the work -- three passes take
longer than the round that produced the tree -- so it runs beside the run, on
its own concurrency budget, and falls behind harmlessly.

Each checkpoint tree is imported and build-checked before it is scored. A
fresh copy imports from a deleted cache, and under parallel lanes that does not
always finish; the one time it did not, the tree scored with 2 entries in
`.godot/imported` where a finished import leaves 672, and every recording was
a blank sky.
"""
import json, os, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

S = Path(os.environ.get('RSIGAME_LOOP_SCORE_DIR') or Path(__file__).resolve().parents[2] / '_ckpt_scores')
S.mkdir(parents=True, exist_ok=True)
RUN = Path(os.environ.get('RSIGAME_PATHS_RUN') or Path(__file__).resolve().parents[2])
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else RUN / 'newloop_dispatch30'
# every sixth round rather than every third: two arms of twelve games is 120 checkpoints at ten-odd minutes each,
# which does not finish overnight at every third, while every sixth still shows the shape of the curve.
EVERY = int(os.environ.get('RSIGAME_MONITOR_CHECKPOINT_EVERY') or 3)
N = int(os.environ.get('RSIGAME_MONITOR_JOBS') or 2)
PASSES = int(os.environ.get('RSIGAME_MONITOR_PASSES') or 3)
PY = os.environ.get('RSIGAME_PATHS_PYTHON') or sys.executable
# RSIGAME_JUDGE_JUDGE_PROVIDER=Alibaba switches to the hosted judge; unset keeps the local vLLM one.
HOSTED = os.environ.get('RSIGAME_JUDGE_JUDGE_PROVIDER', '')
JUDGE_MODEL = 'qwen/qwen3.8-27b' if HOSTED else 'qwen38-27b'
# RSIGAME_MONITOR_JUDGE=stub: record only (RSIGAME_MONITOR_PASSES=1). The main-table runs need every
# checkpoint's pass-1 recordings for the offline Value Monitor, but an official
# score only for the champion (score_champions.py); the reward written is a stub.
STUB = os.environ.get('RSIGAME_MONITOR_JUDGE') == 'stub'
JUDGE_ARGS = ['--judge', 'stub', '--judge-model', '0.5'] if STUB else ['--judge', 'openai', '--judge-model', JUDGE_MODEL]
# SCOPED TO THE RUN ROOT. Two arms of one ablation carry the SAME ten game
# names, so a single log directory and a single csv would interleave their
# rows with nothing in the row saying which arm it came from. The original
# names are kept for `newloop_dispatch30` so the hundred rows already written
# stay where every other tool looks for them.
_TAG = '' if ROOT.name == 'newloop_dispatch30' else f'_{ROOT.name}'
LOG = S / f'ckpt_logs{_TAG}'
LOG.mkdir(exist_ok=True)
CSV = S / (f'{ROOT.name}_scores.csv' if _TAG else 'd30_scores.csv')


def env():
    e = dict(os.environ)
    for line in (Path(os.environ.get('RSIGAME_LOOP_ENV_FILE') or RUN / '.env')).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            e.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    e.setdefault('GAMECRAFT_BENCH', str(RUN.parent.parent / 'gamecraft-bench'))
    if HOSTED:
        # Hosted judge: the same Qwen3.8-27B on OpenRouter, provider pinned with
        # no fallback, thinking off (needs the provider-pin bench patch).
        e['GAMECRAFT_BENCH_JUDGE_OPENAI_BASE_URL'] = 'https://openrouter.ai/api/v1'
        e['GAMECRAFT_BENCH_JUDGE_OPENAI_API_KEY'] = e.get('OPENROUTER_API_KEY', '')
        e['GAMECRAFT_BENCH_JUDGE_PROVIDER'] = HOSTED
    else:
        e['GAMECRAFT_BENCH_JUDGE_OPENAI_BASE_URL'] = os.environ.get('RSIGAME_JUDGE_JUDGE_BASE_URL', 'http://127.0.0.1:8038/v1')
        e['GAMECRAFT_BENCH_JUDGE_OPENAI_API_KEY'] = 'local'
    e.setdefault('PYTHONPATH', os.environ.get('PYTHONPATH', ''))
    return e


def say(*bits):
    line = time.strftime('%H:%M ') + ' '.join(str(b) for b in bits)
    print(line, flush=True)
    with (LOG / 'daemon.log').open('a') as f:
        f.write(line + '\n')


def loads(tree, e):
    r = subprocess.run(
        [PY, '-c', "import sys; sys.path.insert(0,'.')\n"
         "import rsigame.project as PJ\nfrom pathlib import Path\n"
         f"raise SystemExit(0 if PJ.build(Path('{tree}')).get('ok') else 1)"],
        cwd=str(RUN), env=e, capture_output=True, timeout=900)
    return r.returncode == 0


def score(game, r, tree):
    out = tree.parent / 'score'
    lock = tree.parent / '.scoring'
    # THE ROUND MAY BE GONE BY NOW. A checkpoint is queued off one scan and
    # scored minutes later; in between, a round whose record turned out to be
    # a crash can be deleted. Taking the lock outside the try meant that
    # FileNotFoundError went straight through `pool.map` and took the whole
    # daemon down -- scoring stopped for twenty minutes and nothing said so.
    try:
        if not tree.parent.is_dir() or not (tree / 'project.godot').is_file():
            return None
        if (out / 'summary.json').is_file():
            return None
        # EXCLUSIVE CREATE, not check-then-write. Raising the concurrency means
        # two workers can see the same pending checkpoint in the same instant,
        # and two `score_game` runs writing into one `score/pN` would not fail
        # -- they would interleave and produce a number nobody can account for.
        with open(lock, 'x') as f:
            f.write(str(os.getpid()))
    except Exception as exc:
        say(f'  {game} r{r:02d} could not take the lock: {type(exc).__name__}: {str(exc)[:120]}')
        return None
    e = env()
    try:
        say(f'{game} r{r:02d} scoring')
        t0 = time.time()
        if not (tree / '.godot').is_dir():
            subprocess.run(['godot', '--headless', '--path', str(tree),
                            '--import', '--quit'], capture_output=True,
                           timeout=900)
        for attempt in (1, 2):
            if loads(tree, e):
                break
            say(f'  {game} r{r:02d} the copy will not load, attempt {attempt} re-importing')
            subprocess.run(['godot', '--headless', '--path', str(tree),
                            '--import', '--quit'], capture_output=True,
                           timeout=900)
        else:
            say(f'  {game} r{r:02d} still will not load after two imports, skipping')
            (tree.parent / 'score_failed.txt').write_text('tree does not load')
            return None
        p = subprocess.run(
            [PY, 'score_game.py', '--project', str(tree), '--game', game,
             '--output', str(out), '--passes', str(PASSES),
             *JUDGE_ARGS],
            cwd=str(RUN), env=e, capture_output=True, text=True, timeout=7200)
        (LOG / f'{game}.r{r:02d}.log').write_text(p.stdout + p.stderr)
        sm = out / 'summary.json'
        if not sm.is_file():
            say(f'  {game} r{r:02d} the scoring script produced no summary (rc={p.returncode})')
            return None
        # a judge that dropped out is not a score.
        #
        # the formula is reward = BUILD * (the weighted per-item readings)，every failed judge call returns
        # 0，so a perfectly good build is written down as 0.000 —— and once summary.json is written
        # that checkpoint is never rescored. A local vLLM outage ruined sixteen
        # checkpoints this way, several of which looked plausible (only some demos had failed judging),
        # and were nearly read as"B arm B scoring 0.163 below arm A".
        #
        # So a judging error voids the result and deletes the artifact for a rerun, instead of recording it.
        errs = []
        for bd in sorted(out.glob('p*/breakdown.json')):
            try:
                b = json.loads(bd.read_text())
            except Exception:
                continue
            errs += [str(x) for x in (b.get('errors') or [])
                     if 'API call failed' in str(x) or 'Connection error' in str(x)
                     or 'judge failed' in str(x)]
        if errs:
            say(f'  {game} r{r:02d} judge failures {len(errs)} , voided and rerun: {errs[0][:90]}')
            shutil.rmtree(out, ignore_errors=True)
            return None
        d = json.loads(sm.read_text())
        rw = d.get('rewards') or []
        mean = sum(rw) / len(rw) if rw else None
        say(f'  {game} r{r:02d} = {mean:.4f}  {rw}  '
            f'{(time.time() - t0) / 60:.0f}min')
        with CSV.open('a') as f:
            f.write(f'{game},{r},{mean},{rw},{time.strftime("%H:%M")}\n')
        return mean
    except Exception as exc:
        say(f'  {game} r{r:02d} scoring crashed: {type(exc).__name__}: {str(exc)[:160]}')
    finally:
        try:
            lock.unlink(missing_ok=True)
        except Exception:
            pass


def pending():
    out = []
    for gd in sorted((ROOT / 'runs').glob('*')):
        if not gd.is_dir():
            continue
        for rd in sorted(gd.glob('round_*')):
            try:
                r = int(rd.name.split('_')[1])
            except Exception:
                continue
            if r % EVERY:
                continue
            t = rd / 'tree'
            if not (t / 'project.godot').is_file():
                continue
            if (rd / 'score' / 'summary.json').is_file():
                continue
            if (rd / '.scoring').exists() or (rd / 'score_failed.txt').is_file():
                continue
            out.append((gd.name, r, t))
    return out


def main():
    say(f'overnight scoring started · root={ROOT} · concurrency {N} · {PASSES} passes · every {EVERY} rounds')
    idle = 0
    with ThreadPoolExecutor(max_workers=N) as pool:
        while True:
            todo = pending()
            if todo:
                idle = 0
                say(f'to score {len(todo)}: '
                    + ', '.join(f'{g} r{r}' for g, r, _ in todo[:8]))
                try:
                    list(pool.map(lambda a: score(*a), todo))
                except Exception as exc:
                    # Nothing a single checkpoint can do may end the daemon.
                    say(f'  this scoring pass had an error, continuing: '
                        f'{type(exc).__name__}: {str(exc)[:160]}')
            else:
                idle += 1
                if idle % 20 == 1:
                    say('no checkpoints left to score')
            time.sleep(120)


if __name__ == '__main__':
    main()
