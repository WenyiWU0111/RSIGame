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
"""One arm's training file: generation + plan + repair, balanced by STEPS.

WHY STEPS AND NOT TOKENS
    Training runs at batch size 1 (sequence parallelism, not data
    parallelism), so every row gets exactly one optimizer step no matter how
    long it is. Row share IS step share. Ignoring that once gave plan+repair
    59.4% of stage 3's updates while they carried 4.6% of its tokens -- a
    majority of training spent on single-turn, zero-tool data when the
    evaluation is a ~110-turn agentic loop. The rebuild rebalanced to 38.7%
    non-agentic, and that is the only stage-3 configuration whose behaviour
    has actually been observed.

WHY REPAIR IS SUBSAMPLED, HAVING JUST BEEN ENLARGED
    The repair pool grew from 145 to 1,430 rows as more collection arms
    landed. Dropped in whole next to 1,294 generation rows it would be
    1,430/2,837 = 50.4% of all updates by itself, and non-agentic would reach
    54.4% -- past the 59.4% that was diagnosed as the problem, not back toward
    the 38.7% that replaced it. More repair data is good; spending half the
    run's steps on single-turn patches is the failure this pipeline already
    paid for once. The subsample is stratified by game and round-diverse, so
    every game survives: breadth is kept, depth per game is what gets trimmed.

WHY THE PATCH HEADERS ARE REWRITTEN AND GENERATION'S PATHS ARE NOT
    For repair the patch IS the assistant target, so whatever the diff header
    says is what the model emits at inference -- and a diff header recorded on
    the collecting machine is an absolute path that exists nowhere else. Those
    are rewritten to repo-relative. In generation the paths appear only inside
    tool arguments the model re-derives, so rewriting them there is
    behaviourally inert -- and leaving generation untouched is what keeps the
    stage-1 adapter valid, which the whole ablation depends on.

SCHEMA
    Exactly (engine, messages, task). The three source corpora carry three
    different key sets; ms-swift infers one schema from the first rows and
    raises CastError on the first row that disagrees. This has already cost
    one job. Extra keys are dropped here rather than trusted to survive.

ORDERING
    The 4 longest rows first as an out-of-memory canary, then SHUFFLED.
    Sorting everything longest-first instead looks harmless and is not: the
    added corpora are the SHORT ones, so they all land at the tail. Measured
    on the first build of this file, all 116 plan rows sat at positions
    1294-1409, so a run cut off at 78% -- what a 30-hour budget buys at ~99
    s/iteration -- would have seen zero of them and produced a "stage 2"
    adapter trained purely on generation data. Length-sorting also correlates
    step order with row length, which is its own confound. Four canaries keep
    the fast-failure property the sort was for; the shuffle keeps composition
    uniform in time, so a partial run is still a valid, comparable arm.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re

from . import jsonl

DIFF_SUFFIX = re.compile(r'-D\d+(-\d+)?$')
HDR = re.compile(r'^(---|\+\+\+)[ \t]+(\S+)(.*)$', re.M)
SEED = 42

# Plan is the most redundant source -- roughly one plan per game that
# generation already covers -- so it is the one cut hardest.
PLAN_KEEP = 0.25
TARGET_NON_AGENTIC = 0.387


def brief_stem(task: str) -> str:
    """The same unit `split` holds out on: the task id with its difficulty
    suffix removed. A Godot task keeps its whole name."""
    return DIFF_SUFFIX.sub('', task or '').strip('-')


def rel_header(task: str, path: str) -> str:
    """An absolute diff header -> repo-relative, rooted at the game directory."""
    if not path.startswith('/'):
        return path
    marker = f'/{task}/'
    i = path.find(marker)
    if i == -1:
        return '/'.join(path.rstrip('/').split('/')[-3:])
    rest = path[i + len(marker):]
    return re.sub(r'^G\d+/branches/[^/]+/', '', rest)


def fix_patch(task: str, text: str) -> str:
    return HDR.sub(lambda m: f'{m.group(1)} {rel_header(task, m.group(2))}{m.group(3)}', text)


def norm(row: dict, engine: str | None = None) -> dict:
    return {
        'engine': row.get('engine') or engine or 'unknown',
        'messages': row['messages'],
        'task': row.get('task') or '?',
    }


def est(row: dict) -> int:
    return sum(len(m.get('content') or '') for m in row['messages']) // 4


def stratify(rows: list[dict], target: int, rng: random.Random) -> list[dict]:
    """Keep every game, trim depth per game.

    Round-robin across games, so the first pick from each game lands before
    any game's second pick -- which is what "breadth first" has to mean when
    the budget runs out partway through.
    """
    if target >= len(rows):
        return list(rows)
    by_game: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_game[r['task']].append(r)
    for g in by_game:
        by_game[g].sort(key=lambda r: r.get('round', ''))
        rng.shuffle(by_game[g])
    order = sorted(by_game)
    rng.shuffle(order)
    out, depth = [], 0
    while len(out) < target:
        added = False
        for g in order:
            if depth < len(by_game[g]):
                out.append(by_game[g][depth])
                added = True
                if len(out) >= target:
                    break
        if not added:
            break
        depth += 1
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog='rsigame.data_pipeline mix', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gen', required=True)
    ap.add_argument('--plan', required=True)
    ap.add_argument('--repair', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--holdout', default=None,
                    help='JSON {"tasks": [...]}; every source is filtered by brief stem')
    ap.add_argument('--non-agentic', type=float, default=TARGET_NON_AGENTIC,
                    help='target share of optimizer steps spent on plan+repair')
    ap.add_argument('--keep-all-repair', action='store_true',
                    help='skip the subsample (only sane when the generation corpus is large '
                         'enough in ROWS to absorb it)')
    a = ap.parse_args(argv)

    rng = random.Random(SEED)
    gen = [norm(r) for r in jsonl.read(a.gen)]
    plan_raw = list(jsonl.read(a.plan))
    repair_raw = list(jsonl.read(a.repair))

    # ---- holdout exclusion, applied to EVERY source ------------------------
    # The generation corpus was already split against the holdout, so it is
    # clean. plan and repair were NOT: they are separate corpora assembled from
    # their own sources, and nothing had ever filtered them. Measured before
    # this guard existed: 20 holdout tasks leaked into stage 2 and 41 into
    # stage 3. The unit is the BRIEF STEM, not the task id -- a brief can exist
    # as both a web and a Godot task, and training on one while evaluating the
    # other leaks the design just as surely.
    if a.holdout:
        with open(a.holdout) as fh:
            hs = {brief_stem(t) for t in json.load(fh)['tasks']}

        def keep(rows: list[dict], what: str) -> list[dict]:
            out = [r for r in rows if brief_stem(r.get('task', '')) not in hs]
            if len(out) != len(rows):
                print(f'holdout filter      : {what} {len(rows)} -> {len(out)} '
                      f'({len(rows) - len(out)} rows dropped)')
            return out

        gen = keep(gen, 'gen   ')
        plan_raw = keep(plan_raw, 'plan  ')
        repair_raw = keep(repair_raw, 'repair')

    # ---- repair: rewrite diff headers before anything else -----------------
    abs_hdr = re.compile(r'^(---|\+\+\+)\s+/', re.M)
    n_before = sum(1 for r in repair_raw if abs_hdr.search(r['messages'][-1]['content']))
    for r in repair_raw:
        r['messages'][-1]['content'] = fix_patch(r['task'], r['messages'][-1]['content'])
    n_after = sum(1 for r in repair_raw if abs_hdr.search(r['messages'][-1]['content']))
    if n_after:
        raise SystemExit(f'REFUSING: {n_after} repair rows still carry absolute diff headers')
    print(f'repair headers rewritten : {n_before} -> {n_after}')

    # ---- plan: stratified subsample by archetype ---------------------------
    by_arch: dict[str, list[dict]] = collections.defaultdict(list)
    for r in plan_raw:
        by_arch[r.get('archetype', '?')].append(r)
    plan_keep: list[dict] = []
    for arch in sorted(by_arch):
        rows = sorted(by_arch[arch], key=lambda r: r['task'])
        rng.shuffle(rows)
        plan_keep.extend(rows[:max(1, round(len(rows) * PLAN_KEEP))])
    plan = [norm(r, engine='web') for r in plan_keep]
    print(f'plan subsampled          : {len(plan_raw)} -> {len(plan)} '
          f'across {len(by_arch)} archetypes')

    # ---- repair: subsample to hit the step-share target --------------------
    if a.keep_all_repair:
        repair_sel = repair_raw
    else:
        budget = a.non_agentic / (1.0 - a.non_agentic) * len(gen)   # total non-agentic rows
        repair_sel = stratify(repair_raw, max(0, int(round(budget)) - len(plan)), rng)
    repair = [norm(r) for r in repair_sel]
    print(f'repair subsampled        : {len(repair_raw)} -> {len(repair)} '
          f"across {len({r['task'] for r in repair})} games "
          f"(pool covered {len({r['task'] for r in repair_raw})})")

    rows = gen + plan + repair
    random.Random(SEED).shuffle(rows)
    longest = set(sorted(range(len(rows)), key=lambda i: -est(rows[i]))[:4])
    rows = [rows[i] for i in sorted(longest)] + [r for i, r in enumerate(rows) if i not in longest]

    # ---- validate before writing -------------------------------------------
    keys = {tuple(sorted(r.keys())) for r in rows}
    if keys != {('engine', 'messages', 'task')}:
        raise SystemExit(f'REFUSING: schema not uniform: {keys}')
    dead = [r['task'] for r in rows
            if not any(m['role'] == 'assistant' for m in r['messages'])]
    if dead:
        raise SystemExit(f'REFUSING: {len(dead)} rows have no assistant turn, e.g. {dead[:3]}')
    # A row merely ENDING on a tool result is not a defect: every earlier
    # assistant turn still trains, and the trailing tool output is a recorded
    # call the session never got to answer. Measured at 1.9% of one generation
    # corpus. Reported, not refused -- refusing would reject the very corpus
    # stage 1 trains on.
    tail_tool = sum(1 for r in rows if r['messages'][-1]['role'] != 'assistant')

    jsonl.write(a.out, rows)

    n_non = len(plan) + len(repair)
    tok = sum(est(r) for r in rows)
    print(f'\nwritten {a.out}')
    print(f'  rows            {len(rows):,}  (gen {len(gen):,} + plan {len(plan):,} '
          f'+ repair {len(repair):,})')
    print(f'  tokens (est)    {tok / 1e6:.2f}M')
    print(f'  step share      non-agentic {n_non}/{len(rows)} = '
          f'{100 * n_non / len(rows):.1f}%  (the observed-good run was 38.7%, '
          'its predecessor 59.4%)')
    print(f'  token share     non-agentic '
          f'{100 * sum(est(r) for r in plan + repair) / max(tok, 1):.1f}%')
    print(f"  engines         {dict(collections.Counter(r['engine'] for r in rows))}")
    print(f'  ending on tool  {tail_tool} rows ({100 * tail_tool / len(rows):.1f}%) '
          '-- unanswered final call, still trains')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
