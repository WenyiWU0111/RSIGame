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
"""Score one game tree, N times, with demos long enough to show their outcome.

WHY THERE IS A WRAPPER AT ALL, rather than calling the verifier directly:

  1. ONE PASS IS NOT A MEASUREMENT. The judge is deterministic on identical
     frames -- five calls on one frame set gave 13/13 identical requirement
     scores -- but the REPLAY is not, and by how much depends on the game.
     shooter-sky-duel integrates `delta` in 39 places and two replays of one
     tree differ in 98.8-99.5% of their pixels; idle-spell-tower differs in
     0.4%. On a frozen tree nobody repaired, three passes gave 0.354 / 0.344 /
     0.335 against 0.310 / 0.237 / 0.300 -- a 0.062 spread from replay alone.
     Three passes is the floor.

  2. A DEMO THAT ENDS ON ITS LAST INPUT CANNOT SHOW WHAT THAT INPUT DID.
     See `demo_tail.py`. shooter-sky-duel's demos leave 0.1-0.3s of tail, and
     the rubric then asks whether a debrief screen appears. Padding is applied
     by the same rule to every tree, fires only on the condition, adds no
     input and moves no event -- and is recorded in `tail_padding.json` next
     to the results, because it IS a deviation from the task's 600-frame cap
     and a number produced under it must not be silently mixed with one that
     was not.

  3. `--max-demo-seconds` HAS TO FOLLOW THE PADDING. The sampler draws its
     window as min(duration, max_window); left at 20 while a demo runs 30, it
     takes a seeded RANDOM 20-second window out of a 30-second recording and
     can miss the beginning entirely. Raised to cover the padding, a demo that
     was NOT padded is unaffected -- its window is still its own duration --
     so the coarser sampling is paid only by the demos that needed the time.

Run:
    python score_game.py --project TREE --game NAME --output DIR [--passes 3]
                         [--no-pad] [--judge openai] [--judge-model M]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

RUN = Path(os.environ.get('RSIGAME_PATHS_RUN') or Path(__file__).resolve().parent)
BENCH = Path(os.environ.get('GAMECRAFT_BENCH')
             or RUN.parent / 'gamecraft-bench')
sys.path.insert(0, str(RUN))


def godot_import(tree: Path) -> bool:
    godot = os.environ.get('GAMECRAFT_BENCH_GODOT_BIN') or shutil.which('godot')
    if not godot or not (tree / 'project.godot').is_file():
        return False
    subprocess.run([godot, '--headless', '--path', str(tree), '--import',
                    '--quit'], capture_output=True, timeout=600)
    return (tree / '.godot').is_dir()


def main() -> int:
    from rsigame import config
    config.ensure()
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', required=True)
    ap.add_argument('--game', required=True, help='task name, for the rubric')
    ap.add_argument('--output', required=True)
    ap.add_argument('--passes', type=int, default=3)
    ap.add_argument('--max-demos', type=int, default=0,
                    help='0 = the bench default (10). Only raise it for a\n                          recording; every scored run must use the default.')
    ap.add_argument('--no-pad', action='store_true',
                    help='score the demos exactly as shipped')
    ap.add_argument('--judge', default='openai')
    ap.add_argument('--judge-model', default='qwen38-27b')
    a = ap.parse_args()

    import rsigame.verify.demo_tail as demo_tail
    src = Path(a.project).resolve()
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rubric = BENCH / 'tasks' / a.game / 'tests' / 'rubric.json'
    if not rubric.is_file():
        raise SystemExit(f'no rubric at {rubric}')

    tree, pad_report = src, {'padded': [], 'n_padded': 0,
                             'note': '--no-pad: demos as shipped'}
    if not a.no_pad:
        tree = out / '_tree'
        pad_report = demo_tail.padded_copy(src, tree)
        pad_report['source'] = str(src)
        if not godot_import(tree):
            print('  warning: the padded copy did not import; scoring anyway')
    (out / 'tail_padding.json').write_text(
        json.dumps(pad_report, indent=1, ensure_ascii=False))
    print(f"  {a.game}: {pad_report['n_padded']}/{pad_report.get('n_demos','?')} "
          f"demos padded")

    # The window has to cover the longest recording, or the sampler takes a
    # random slice of it. Demos that were not padded keep their own duration
    # as the window and are untouched by this.
    longest = 0
    for f in sorted((tree / 'demo_outputs').glob('*.json')):
        try:
            longest = max(longest, int(json.loads(f.read_text())
                                       .get('duration_frames', 0)))
        except Exception:
            pass
    window = max(20, -(-longest // 30))          # ceil to whole seconds

    env = dict(os.environ)
    env.setdefault('GAMECRAFT_BENCH_JUDGE_NO_THINK', '1')
    env.setdefault('GAMECRAFT_BENCH_JUDGE_TEMPERATURE', '0')
    env.setdefault('GAMECRAFT_BENCH_JUDGE_REPEATS', '1')

    rewards = []
    for p in range(1, a.passes + 1):
        d = out / f'p{p}'
        if (d / 'reward.txt').is_file():
            rewards.append(float((d / 'reward.txt').read_text()))
            print(f'  p{p}: already scored ({rewards[-1]:.4f})')
            continue
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, '-m', 'gamecraft_bench.verifier',
             '--project', str(tree), '--rubric', str(rubric),
             '--output', str(d), '--judge', a.judge,
             '--judge-model', a.judge_model,
             '--max-demo-seconds', str(window)]
            + (['--max-demos', str(a.max_demos)] if a.max_demos else []),
            cwd=str(BENCH), env=env, capture_output=True, text=True)
        (d.parent / f'p{p}.log').write_text(r.stdout + r.stderr)
        # A FAILED JUDGE IS NOT A ZERO.
        #
        # When the judge endpoint is down the verifier still finishes: every
        # requirement scores 0, BUILD stays 1, and reward.txt holds a
        # well-formed 0.0000. Thirty-seven trees were recorded that way during a
        # four-hour outage, some of them partially -- one demo unanswered out of
        # nine leaves a plausible 0.6692. Nothing downstream could tell those
        # from real scores, so the pass is refused here instead.
        bd = d / 'breakdown.json'
        judge_failed = ''
        if bd.is_file():
            try:
                errs = ' '.join(json.loads(bd.read_text()).get('errors') or [])
                if 'judge failed' in errs or 'API call failed' in errs:
                    judge_failed = errs[:160]
            except Exception:
                pass
        if judge_failed:
            print(f'  p{p}: JUDGE FAILED -- {judge_failed}')
            raise SystemExit(
                f'refusing to score {a.game}: the judge did not answer '
                f'({judge_failed}). Fix the endpoint and re-run; a partial '
                f'answer scores as zero and is indistinguishable from a real one.')
        got = (d / 'reward.txt')
        rewards.append(float(got.read_text()) if got.is_file() else float('nan'))
        print(f'  p{p}: reward={rewards[-1]:.4f}  window={window}s  '
              f'{time.time() - t0:.0f}s')

    (out / 'summary.json').write_text(json.dumps({
        'game': a.game, 'source': str(src), 'passes': a.passes,
        'padded': not a.no_pad, 'n_padded': pad_report['n_padded'],
        'max_demo_seconds': window, 'rewards': rewards,
    }, indent=1, ensure_ascii=False))
    ok = [x for x in rewards if x == x]
    if ok:
        print(f"  {a.game}: {sum(ok)/len(ok):.4f}  "
              f"({' / '.join(f'{x:.3f}' for x in ok)})  spread {max(ok)-min(ok):.3f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
