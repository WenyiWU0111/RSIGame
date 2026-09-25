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
"""The extracted web plans and the distilled Godot plans -> one plan corpus.

WHY A MERGE STEP AT ALL
    The two halves arrive by different routes and neither can be swapped for
    the other. Web plans are EXTRACTED -- 147 of 150 web trajectories contain
    an explicit "Files to MODIFY / CREATE" block the agent wrote itself. Godot
    plans are DISTILLED, because 0 of 150 Godot trajectories contain one; that
    agent writes prose intent. Pooling them is legitimate only if they look
    like ONE task to the model, which is what this module checks.

THE SYSTEM PROMPT MUST MATCH BYTE FOR BYTE
    Three plan corpora pool into one training set only because their system
    turn is identical. One stray character gives the model two tasks that look
    the same and splits the corpus silently, with no error anywhere. So it is
    asserted here rather than trusted -- the same rule `repair_records`
    follows, for the same reason.

ARCHETYPE, AND WHY IT IS DERIVED DIFFERENTLY PER ENGINE
    `mix` subsamples plans stratified by `archetype`, so every game family
    survives the cut. Web rows already carry one: a compound name like
    `topdown-chase` (15 buckets, and a prefix of the task id in all 454 rows).
    Distilled Godot rows carry none, so the leading family of the task id is
    used -- `cardgame-autobattler` -> `cardgame`, 17 buckets. That is the same
    derivation `split` uses. Without it every Godot row would land in one "?"
    bucket and stratification would quietly become "take the first N".

PATHS ARE LEFT ALONE, DELIBERATELY
    Web plans name `src/gameConfig.json`; Godot plans name
    `/workspace/game/project.godot`. That is not an inconsistency to normalise:
    measured across 60 Godot generation trajectories, tool calls use the
    absolute workspace path 882 times against 155 relative paths, so each
    engine's plan matches the paths its own agent actually writes. Rewriting
    the Godot plans to relative paths would teach the model to emit paths that
    do not exist.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter

from . import jsonl

FAMILY = re.compile(r'^([a-z_]+)')


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog='rsigame.data_pipeline plans-merge', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--web', required=True, help='the extracted web plan corpus')
    ap.add_argument('--godot', required=True, help='the distilled Godot plan corpus')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)

    web, godot = list(jsonl.read(a.web)), list(jsonl.read(a.godot))

    prompts = {r['messages'][0]['content'] for r in web + godot
               if r['messages'][0]['role'] == 'system'}
    if len(prompts) != 1:
        raise SystemExit(f'REFUSING: {len(prompts)} distinct system prompts across the two '
                         'corpora -- they would train as two different tasks')

    out = []
    for r in web:
        m = FAMILY.match(r['task'])
        out.append({'task': r['task'], 'engine': 'web',
                    'archetype': r.get('archetype') or (m.group(1) if m else '?'),
                    'source': 'extracted', 'messages': r['messages']})
    for r in godot:
        m = FAMILY.match(r['task'])
        out.append({'task': r['task'], 'engine': 'godot',
                    'archetype': m.group(1) if m else '?',
                    'source': r.get('source', 'distilled'), 'messages': r['messages']})

    seen, dedup = set(), []
    for r in out:
        k = (r['task'], r['engine'])
        if k in seen:
            continue
        seen.add(k)
        dedup.append(r)

    bad = [r['task'] for r in dedup if r['archetype'] == '?']
    if bad:
        raise SystemExit(f'REFUSING: {len(bad)} rows have no derivable archetype, e.g. {bad[:3]}')

    dedup.sort(key=lambda r: (r['engine'], r['task']))
    jsonl.write(a.out, dedup)

    eng = Counter(r['engine'] for r in dedup)
    arch = Counter(r['archetype'] for r in dedup)
    sizes = sorted(arch.values())
    print(f'web plans    : {len(web):,} (extracted)')
    print(f'godot plans  : {len(godot):,} (distilled)')
    print(f'duplicates   : {len(out) - len(dedup):,}')
    print(f'written      : {len(dedup):,} -> {a.out}')
    print(f'  engines    : {dict(eng)}')
    print(f'  archetypes : {len(arch)} buckets, '
          f'min {sizes[0]} / median {sizes[len(sizes) // 2]} / max {sizes[-1]} rows')
    return 0 if dedup else 1


if __name__ == '__main__':
    raise SystemExit(main())
