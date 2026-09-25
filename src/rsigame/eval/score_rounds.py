#!/usr/bin/env python3
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
"""Score a Self-Refine arm at several points along its run.

  score_rounds.py <corpus> --points 0,2,4,6,8 --lines 4 [--games a,b,c]

One job is one (game, round) pair: rebuild the game as it stood after that
round, replay its shipped demo traces, and judge the frames. `breakdown.json`
lands under `scoring/<corpus>/<game>/r<k>/` and the job is skipped if that file
is already there, so a killed run resumes instead of restarting -- this host
takes long jobs out in clusters and a scoring sweep is a long job.

FRAMES ARE DELETED once judged. They are 58MB per job and this sweep is 200 of
them; the disk was at 93% when it was written. `judge_log.json` keeps every
pass's score, so nothing needed to re-examine a verdict is thrown away -- only
the pixels, which the rebuild can regenerate.
"""
from .. import paths
import argparse, json, os, pathlib, shutil, subprocess, sys, time
import concurrent.futures as cf
from pathlib import Path

RUN = Path(__file__).resolve().parent
BENCH = paths.lazy_bench()
PY_BIN = os.environ.get('RSIGAME_PATHS_PYTHON') or sys.executable   # subprocesses use this same interpreter
SCRATCH = Path(os.environ.get('RSIGAME_LOOP_SCORE_SCRATCH')
               or str(paths.run_root() / 'score'))

sys.path.insert(0, str(RUN))
from .rebuild_round import rebuild                      # noqa: E402


def job(corpus: str, game: str, k: int) -> tuple[str, int, str]:
    # The results root is settable so two arms can be scored into separate
    # trees in one sweep without one overwriting the other.
    root = pathlib.Path(os.environ.get('RSIGAME_LOOP_SCORE_OUT') or (RUN / 'scoring'))
    out = root / corpus / game / f'r{k}'
    if (out / 'breakdown.json').is_file():
        return game, k, 'skip'
    rubric = BENCH / 'tasks' / game / 'tests' / 'rubric.json'
    if not rubric.is_file():
        return game, k, 'no-rubric'
    work = SCRATCH / f'{corpus}__{game}__r{k}'
    work.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        rebuild(corpus, game, k, work)
        out.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            [PY_BIN, '-m', 'gamecraft_bench.verifier',
             '--project', str(work), '--rubric', str(rubric),
             '--output', str(out), '--judge', 'openai',
             '--judge-model', os.environ.get('RSIGAME_LOOP_JUDGE_MODEL', 'qwen38-27b')],
            cwd=BENCH, capture_output=True, text=True, timeout=3600)
        (out / 'verifier.log').write_text(r.stdout[-20000:] + r.stderr[-20000:])
        if not (out / 'breakdown.json').is_file():
            return game, k, f'FAILED rc={r.returncode}'
        rew = json.load(open(out / 'breakdown.json'))['reward']
        return game, k, f'{rew:.3f} ({time.time() - t0:.0f}s)'
    except Exception as e:
        return game, k, f'ERROR {type(e).__name__}: {str(e)[:120]}'
    finally:
        # The pixels are 58MB a job and regenerable; the verdicts are not.
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(out / 'demos', ignore_errors=True)


def main() -> int:
    from rsigame import config
    config.ensure()
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus')
    ap.add_argument('--points', default='0,2,4,6,8')
    ap.add_argument('--lines', type=int, default=4)
    ap.add_argument('--games', default='')
    a = ap.parse_args()

    pts = [int(x) for x in a.points.split(',') if x.strip()]
    if a.games:
        games = [g.strip() for g in a.games.split(',') if g.strip()]
    else:
        games = sorted(
            d.name for d in (RUN / f'self_refine_{a.corpus}' / 'runs').iterdir()
            if (d / 'result.json').is_file()
            and 'final_build' in json.load(open(d / 'result.json')))
    jobs = [(g, k) for g in games for k in pts]
    print(f'{a.corpus}: {len(games)} games x {len(pts)} points = {len(jobs)} '
          f'jobs on {a.lines} lines', flush=True)
    done = 0
    with cf.ThreadPoolExecutor(a.lines) as ex:
        futs = [ex.submit(job, a.corpus, g, k) for g, k in jobs]
        for f in cf.as_completed(futs):
            g, k, msg = f.result()
            done += 1
            print(f'[{done}/{len(jobs)}] {g} r{k}: {msg}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
