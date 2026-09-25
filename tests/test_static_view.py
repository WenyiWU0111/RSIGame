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
"""Preflight for the Phase 2 static block, per the handoff's section 12.

Runs `build_static_view` over real trees and asserts the four things that
would invalidate the experiment if they were wrong:

  1. the counts match what an independent shell command finds;
  2. no rubric or evaluator language reaches the prompt;
  3. the block stays inside its token budget;
  4. a tree with nothing extractable yields '' rather than an empty skeleton.

Run:  python tests_static_view.py
"""

import os
import pytest
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rsigame import paths
from rsigame.view.static_view import build_static_view, leak_hits   # noqa: E402


# These read real run directories(which the repo does not carry)。skip when unset,
# rather than passing green on an empty list.
pytestmark = pytest.mark.skipif(
    not os.environ.get('RSIGAME_TEST_GAMES'),
    reason='set RSIGAME_TEST_GAMES (and RSIGAME_PATHS_RUN) to run against real runs')


# These read real run directories, which the repo does not carry: skip when unset.
RUN = paths.run_root()
E2 = RUN / 'self_refine_godot_gpt_e2' / 'runs'
CORPUS = RUN / "godot/corpus_gpt"
TOKEN_CAP = 500

fails = []


def check(cond, msg):
    print(('  PASS  ' if cond else '  FAIL  ') + msg)
    if not cond:
        fails.append(msg)


def tokens(s: str) -> int:
    """Cheap upper bound. The block is ASCII counts and identifiers, which
    tokenize near 3.2 chars/token; 3.0 keeps the estimate pessimistic."""
    return int(len(s) / 3.0)


def shell_count(cmd: str) -> int:
    r = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True)
    return int((r.stdout or '0').strip() or 0)


def scenarios_from_result(game_dir: Path):
    """What the arm itself recorded, so the test does not re-derive them."""
    f = game_dir / 'result.json'
    if not f.is_file():
        return [], [], []
    d = json.loads(f.read_text())
    rounds = d.get('rounds') or []
    if not rounds:
        return [], [], []
    declared = rounds[-1].get('scenarios_declared') or []
    now = rounds[-1].get('scenarios_visited') or []
    hist = set()
    for r in rounds:
        hist |= set(r.get('scenarios_visited') or [])
    return declared, now, sorted(hist)


def main():
    games = sorted(p for p in E2.iterdir() if (p / 'work').is_dir())
    print(f'{len(games)} game trees under {E2}\n')

    for g in games:
        work = g / 'work'
        declared, now, hist = scenarios_from_result(g)
        block = build_static_view(work, declared=declared, visited_now=now,
                                  visited_history=hist)
        print('=' * 78)
        print(f'{g.name}   {len(block)} chars, ~{tokens(block)} tokens')
        print('=' * 78)
        print(block)
        print()

        # 2. leakage
        hits = leak_hits(block)
        check(not hits, f'{g.name}: no evaluator language (found {hits})')

        # 3. budget
        check(tokens(block) <= TOKEN_CAP,
              f'{g.name}: ~{tokens(block)} tokens <= {TOKEN_CAP}')

        # 1. counts, cross-checked against an independent shell scan
        pngs = shell_count(
            f'find {work}/assets -type f \\( -name "*.png" -o -name "*.jpg" '
            f'-o -name "*.svg" \\) 2>/dev/null | wc -l')
        m = re.search(r'images: (\d+) present', block)
        got = int(m.group(1)) if m else -1
        check(got >= pngs,
              f'{g.name}: images present {got} >= shell find under assets/ '
              f'{pngs} (module also scans outside assets/)')

        # Word boundaries, or the shell counts `_draw_archive()` as a
        # `draw_arc` call -- which is how this test first disagreed with the
        # module by exactly four on horror-tape-archive. The module was right.
        # Excludes the same harness-written directories the module skips.
        proc = shell_count(
            f'grep -rhoE "\\bdraw_rect\\b|\\bdraw_circle\\b|\\bdraw_line\\b|'
            f'\\bdraw_polygon\\b|\\bdraw_colored_polygon\\b|\\bdraw_arc\\b|'
            f'\\bdraw_multiline\\b|\\bColorRect\\b|\\bStyleBoxFlat\\b|'
            f'\\bPlaceholderTexture2D\\b" {work} --include=*.gd '
            f'--include=*.tscn --exclude-dir=_repair_evidence '
            f'--exclude-dir=demo_outputs --exclude-dir=.godot '
            f'2>/dev/null | wc -l')
        m = re.search(r'StyleBoxFlat\): (\d+)', block)
        got = int(m.group(1)) if m else -1
        check(got == proc,
              f'{g.name}: procedural sites {got} == shell {proc}')

        # scenario lines must agree with the arm's own record
        if declared:
            m = re.search(r'defined: (\d+)', block)
            check(m and int(m.group(1)) == len(declared),
                  f'{g.name}: declared {m.group(1) if m else "-"} == '
                  f'result.json {len(declared)}')
            m = re.search(r'entered at least once in this run: (\d+) of', block)
            check(m and int(m.group(1)) == len(set(hist) | set(now)),
                  f'{g.name}: cumulative {m.group(1) if m else "-"} == '
                  f'result.json {len(set(hist) | set(now))}')
        print()

    # 4. degenerate input
    check(build_static_view(Path('/nonexistent/tree')) == '',
          'a missing tree yields the empty string')

    # a frozen G0 tree, which has no run history at all
    g0 = next((p for p in CORPUS.iterdir()
               if (p / 'project.godot').is_file()), None) if CORPUS.is_dir() \
        else None
    if g0:
        b = build_static_view(g0)
        check('WHAT THE BUILD CONTAINS' in b and 'Scenarios' not in b,
              f'G0 tree {g0.name}: block builds with no scenario section')
        check(not leak_hits(b), f'G0 tree {g0.name}: no evaluator language')
        print(f'\n--- G0 sample ({g0.name}), ~{tokens(b)} tokens ---\n{b}\n')

    print(f'\n{"ALL PASS" if not fails else str(len(fails)) + " FAILED"}')
    for f in fails:
        print('  - ' + f)
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
