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
"""The three ablation arms, built from the published per-decision corpus.

The ablation asks one question -- does adding planning, and then repair, make
the generator better? -- so the three arms have to differ in exactly one thing:

    s1        generation only
    s1s2      generation + plan
    s1s2s3    generation + plan + repair

TWO PROPERTIES THE ARMS MUST HAVE, AND HOW THEY ARE ENFORCED HERE

  * THE GENERATION HALF IS IDENTICAL IN ALL THREE. Same seed, same budget, so
    the same rows are drawn. If stage 2 or stage 3 loses, it cannot be because
    that arm saw less generation data -- which is the confound that makes an
    ablation unreadable.

  * GENERATION STAYS DOMINANT. Stage 2 and stage 3 enter as a CAPPED FRACTION
    of the stage-1 budget (15% plan, 20% repair by default) rather than as
    everything available. There is far more plan and repair data than that;
    pouring it in would push one-shot generation down, which is the metric the
    whole system is judged on. `mix` states the same rule the other way round,
    as a target share of optimizer steps.

REPAIR IS DRAWN FROM FOUR POOLS, not one, in fixed proportion: agentic Godot
(45%), agentic web (25%), single-shot Godot (20%), single-shot web (10%).
Positives only. Agentic Godot is weighted highest because it is the closest
shape to what the development loop actually runs.

ROWS COME OUT LONGEST-FIRST, so an out-of-memory failure surfaces in the first
minutes rather than at hour six. (`mix`, which shuffles instead, explains when
that ordering is the wrong choice: it is safe here only because these arms
differ by ADD-ON rows that are interleaved by the sampler, not appended.)
"""
from __future__ import annotations

import argparse
import random

from . import filters, jsonl

ARMS = ('s1', 's1s2', 's1s2s3')

# (pattern, share of the repair budget, label, filter settings)
REPAIR_POOLS = (
    ('s3_repair_agent_godot/positive-*.parquet', 0.45, 's3_agent_godot',
     {'require_full_log': True}),
    ('s3_repair_agent_web/positive-*.parquet', 0.25, 's3_agent_web',
     {'soft_summary': True}),
    ('s3_repair_godot/positive-*.parquet', 0.20, 's3_single_godot', {}),
    ('s3_repair_web/positive-*.parquet', 0.10, 's3_single_web', {}),
)


def pick(pattern: str, n: int, rng: random.Random, label: str,
         stats: dict[str, int], **filt) -> list[dict]:
    """The n cleanest rows of one pool, sampled by index and then materialised."""
    if n <= 0:
        stats[label] = 0
        return []
    idx = filters.clean_indices(pattern, **filt)
    if n < len(idx):
        idx = rng.sample(list(idx), n)
    stats[label] = len(idx)
    return filters.materialise(pattern, idx)


def build(arm: str, *, gen: int, plan_frac: float, repair_frac: float,
          web_frac: float, seed: int) -> tuple[list[dict], dict[str, int]]:
    stats: dict[str, int] = {}
    rows: list[dict] = []

    # ---- stage 1: generation. Same seed and budget in every arm, so this
    #      half of each corpus is the identical set of rows. ----
    n_web = int(gen * web_frac)
    rows += pick('s1_generation/godot-*.parquet', gen - n_web,
                 random.Random(seed + 1), 's1_godot', stats)
    rows += pick('s1_generation/web-*.parquet', n_web,
                 random.Random(seed + 2), 's1_web', stats, soft_summary=True)

    # ---- stage 2: plan ----
    if arm in ('s1s2', 's1s2s3'):
        rows += pick('plan/train-*.parquet', int(gen * plan_frac),
                     random.Random(seed + 3), 's2_plan', stats)

    # ---- stage 3: repair, positives only, both engines and both shapes ----
    if arm == 's1s2s3':
        budget = int(gen * repair_frac)
        for i, (pat, share, label, filt) in enumerate(REPAIR_POOLS):
            rows += pick(pat, int(budget * share), random.Random(seed + 10 + i),
                         label, stats, **filt)

    rows.sort(key=lambda r: -(r.get('_n_tokens') or 0))
    for r in rows:
        r.pop('_n_tokens', None)
    return rows, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog='rsigame.data_pipeline arms', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arm', required=True, choices=ARMS)
    ap.add_argument('--out', required=True)
    ap.add_argument('--gen', type=int, default=3000,
                    help='stage-1 generation rows (identical in every arm)')
    ap.add_argument('--plan-frac', type=float, default=0.15,
                    help='plan rows as a fraction of --gen')
    ap.add_argument('--repair-frac', type=float, default=0.20,
                    help='repair rows as a fraction of --gen')
    ap.add_argument('--web-frac', type=float, default=0.35,
                    help='share of the stage-1 rows drawn from web')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(argv)

    rows, stats = build(a.arm, gen=a.gen, plan_frac=a.plan_frac,
                        repair_frac=a.repair_frac, web_frac=a.web_frac, seed=a.seed)
    jsonl.write(a.out, rows)

    gen_rows = stats.get('s1_godot', 0) + stats.get('s1_web', 0)
    print(f'arm={a.arm} rows={len(rows):,} -> {a.out}')
    for k, v in stats.items():
        print(f'   {k:16} {v:>6,}')
    print(f"   {'TOTAL':16} {sum(stats.values()):>6,}   gen={gen_rows:,}, "
          f"web share of gen={stats.get('s1_web', 0) / max(1, gen_rows):.0%}")
    return 0 if rows else 1


if __name__ == '__main__':
    raise SystemExit(main())
