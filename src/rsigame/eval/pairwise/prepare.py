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
"""Build the pair manifest for the self-direction vs. round-robin comparison.

One entry per game run under both arms. Each side is that arm's round-30 tree
(round 18 for the one case-study game whose self run stopped there, on both
sides so the budgets match). The demos are the ORIGINAL game's own traces --
the same inputs replayed on both builds -- with test traces (names starting
with "_") left out. A game with no usable original demo is dropped and listed.

    prepare.py > $RSIGAME_RUNS/pairwise/manifest.json
"""
import json
import sys
from pathlib import Path
from ... import paths

RUN = paths.run_root()
BENCH = paths.lazy_bench_tasks()

SETS = [
    # (set name, self root, robin root, original corpus, games)
    ('bench40', 'newloop_self40', 'newloop_robin10', 'godot/corpus_gpt',
     None),      # None = this set's game list is scanned on use,not read from disk at import
    ('casestudy', 'newloop_self30', 'newloop_rot30', 'godot/corpus_casestudy',
     None),      # as above:scanned on use
]
ROUND_OVERRIDE = {('casestudy', 'platformer-plumber-kingdom'): 18}


def main():
    pairs, dropped = [], []
    for set_name, self_root, robin_root, corpus, games in SETS:
        for g in games:
            r = ROUND_OVERRIDE.get((set_name, g), 30)
            trees = {arm: RUN / root / 'runs' / g / f'round_{r:02d}' / 'tree'
                     for arm, root in (('self', self_root), ('robin', robin_root))}
            demos = sorted(p for p in (RUN / corpus / g / 'demo_outputs').glob('*.json')
                           if not p.name.startswith('_'))
            instr = BENCH / g / 'instruction.md'
            why = None
            if not demos:
                why = 'original game has no usable demo (test traces only)'
            elif not all((t / 'project.godot').is_file() for t in trees.values()):
                why = f'round {r} tree missing on one side (run not finished?)'
            elif not instr.is_file():
                why = 'no instruction.md'
            if why:
                dropped.append(dict(set=set_name, game=g, reason=why))
                continue
            pairs.append(dict(set=set_name, game=g, round=r,
                              trees={k: str(v) for k, v in trees.items()},
                              demos=[str(d) for d in demos], instruction=str(instr),
                              rubric=str(BENCH / g / 'tests' / 'rubric.json')))
    json.dump(dict(pairs=pairs, dropped=dropped), sys.stdout, indent=1)
    print(f'{len(pairs)} pairs, {sum(len(p["demos"]) for p in pairs)} demos; dropped {len(dropped)}: '
          + '; '.join(f'{d["game"]} ({d["reason"]})' for d in dropped), file=sys.stderr)


if __name__ == '__main__':
    main()
