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
"""The stage checklist: fold the brief into the proxy rubric, and decide which
of the original items survive based on what they already score on P*.

WHY THIS EXISTS. Outer guidance only reached the "what to work on" side of the
loop -- the planner, the defect ranking, the art goal, the repair prompt. The
monitor still decided whether to replace the champion using the proxy rubric
generated from the task document and frozen on its hash (rubric.py), where
nothing the brief asks for has a slot at all. Measured (pilot2, 2026-09-15):
more than half of all checkpoints had a proxy delta of exactly 0, and only 4
of 18 branches ever replaced their build. The repair follows the brief, the
monitor measures with the old ruler, and the change lands in the ruler's blank
space.

This is not the brief stapled to the end of the rubric. One model call, shown
three things:

  1. every item of the original rubric with its measured score on P* -- a 1.0
     means "this item is saturated, no edit can move it", and those items are
     exactly what makes the delta zero;
  2. the brief (stage objective, why now, priorities, what to preserve);
  3. what each of the four dimensions asks for (the same DIMENSIONS rubric.py
     uses).

Out comes a new checklist: the brief's items first and weighted higher, the
original items that are not yet saturated kept, and the saturated ones either
retired or left as low-weight "must not regress" guards.

Origin is recorded on `origin` (brief / task / guard), weight on `weight`, and
aggregation is weighted (aggregate.py). The official rubric is untouched -- it
lives in gamecraft-bench and is not read here.

    from rsigame.eval.estimator import stage_rubric as SR
    rb = SR.build(game, brief_path, base_rubric, pstar_values)

Frozen and cached on (task hash, brief hash), the same rule rubric.py follows:
a branch rerun gets the same ruler it had.

LANGUAGE. `_PROMPT` below is Chinese and stays that way: it is sent to the
model, and the runs that produced the paper's numbers used this wording.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
sys.path.insert(0, str(RUN))

from rsigame.eval.estimator.rubric import CATEGORIES, DIMENSIONS, ask  # noqa: E402

SCHEMA = 'stage-v1'
CACHE = HERE / 'rubrics_stage'

# The weight of a brief item. 3x is not a tuned number, it is a position: what
# this stage delivers is defined by the brief, and the task document's own
# requirements fall back to being the floor. Guards get 0.25 -- their only job
# is to raise an alarm on a regression.
W_BRIEF = 3.0
W_TASK = 1.0
W_GUARD = 0.25
MAX_ITEMS = 20

_PROMPT = """你要更新一份游戏质检清单。

这个游戏已经自主开发了很多轮，分数不再上涨。一位总监（人或模型）玩过当前这一
版，写了一份阶段简报，说下一阶段该往哪走。开发 loop 已经按简报在改了，但质检
清单还是旧的那份——所以简报要求的东西做到了也量不出来。

你的任务：结合"旧清单每一条在当前版本上的实测得分"和"简报"，写出这一阶段的
新清单。

====================
硬规则
====================

- 简报条目排在最前面，id 用 B1、B2……至少 3 条，最多 6 条。它们把简报的阶段
  目标和优先级拆成可以逐条检查的问题。
- 旧清单里**得分不满 1.0 的条目必须保留**（原样保留 id、措辞和三档锚点），它们
  是还没做到的要求。id 和措辞一个字都不要改。
- 旧清单里**得分是 1.0 的条目已经饱和**：可以退役（写进 retired），也可以留成
  守卫条目（source 写 "guard"），守卫条目只用来发现退步。和简报直接冲突的
  饱和条目应当退役，并在 retired 里说明理由。
- 每条只检查一件事，只用这个游戏里真实存在的叫法、按键、画面、对象、数值。
- 每条必须能只看静止画面回答，三档锚点也一样。要听声音、要看帧率、要连续观察
  几秒、要亲手点一下才知道的，一律不要写。
- 待在自己的维度里：F 问"做没做到"，I 问"有没有告诉玩家"，P 问"好不好看"，
  C 问"有多少东西真的做出来了"。
- 简报里写"要保留"的东西，写成守卫条目（source "guard"），不要写成新要求。
- 新写的条目必须有区分度：当前这一版应该在这条上失分，改好了才涨。如果一条
  现在就能拿满分，它没有价值，换一条。
- 总条数不超过 {max_items} 条。

只回严格 JSON：
{{"items": [{{"id": "B1", "source": "brief", "dimension": "F",
            "aggregation": "max", "criterion": "...",
            "anchors": {{"0.0": "...", "0.5": "...", "1.0": "..."}},
            "brief_ref": "简报里对应的那句话"}},
           {{"id": "F3", "source": "keep"}},
           {{"id": "P2", "source": "guard"}}],
 "retired": [{{"id": "F1", "why": "..."}}]}}

`source` 是 "keep" 或 "guard" 时只写 id，其余字段从旧清单原样取，不要重写。

====================
四个维度：要看什么
====================
{dims}

====================
旧清单，以及每一条在当前版本上的实测得分
====================
{base}

====================
这一阶段的简报
====================
{brief}
"""


def brief_text(brief: dict) -> str:
    out = [f"stage objective: {brief.get('stage_objective') or ''}",
           f"why now: {brief.get('why_now') or ''}"]
    for i, p in enumerate(brief.get('priorities') or [], 1):
        out.append(f'priority {i}: {p}')
    for i, p in enumerate(brief.get('preserve') or [], 1):
        out.append(f'preserve {i}: {p}')
    return '\n'.join(out)


def _base_text(base: dict, values: dict) -> str:
    rows = []
    for it in base['items']:
        v = values.get(it['id'])
        mark = 'not observed' if v is None else ('saturated 1.00' if v >= 1.0 else f'{v:.2f}')
        rows.append(f"{it['id']}  [{it.get('category') or it.get('dimension')}]  "
                    f"score {mark}  ({it.get('aggregation')})\n    {it['criterion']}")
    return '\n'.join(rows)


def _clean(out: dict, base: dict, values: dict) -> tuple[list, list, list]:
    """The model does not get to decide the structure; this does.

    -> (items, retired, complaints)
    """
    by = {it['id']: it for it in base['items']}
    bad = []
    items, seen = [], set()
    for it in (out.get('items') or []):
        if not isinstance(it, dict):
            continue
        i, src = it.get('id'), it.get('source')
        if not i or i in seen:
            bad.append(f'duplicate or missing id: {i!r}')
            continue
        if src in ('keep', 'guard'):
            if i not in by:
                # Invented ids for criteria it meant to keep (G1, G2 ... on
                # lawn-guardians). Dropping one is harmless -- every base
                # criterion that still matters is added back below -- so this
                # must not fail the whole checklist.
                bad.append(f'{i}: not in the previous checklist, dropped')
                continue
            row = dict(by[i])
            row['origin'] = 'task' if src == 'keep' else 'guard'
            row['weight'] = W_TASK if src == 'keep' else W_GUARD
        elif src == 'brief':
            dim = it.get('dimension')
            a = it.get('anchors')
            c = str(it.get('criterion') or '').strip()
            if dim not in CATEGORIES or not c or not isinstance(a, dict) \
                    or any(k not in a for k in ('0.0', '0.5', '1.0')):
                bad.append(f'{i}: dimension, wording or anchors not acceptable')
                continue
            row = {'id': i, 'category': dim, 'criterion': c, 'anchors': a,
                   'aggregation': it.get('aggregation') if it.get('aggregation') in ('max', 'mean') else 'max',
                   'origin': 'brief', 'weight': W_BRIEF,
                   'brief_ref': str(it.get('brief_ref') or '')[:300]}
        else:
            bad.append(f'{i}: unknown source {src!r}')
            continue
        seen.add(i)
        items.append(row)

    # Unsaturated items from the previous checklist are restored here rather
    # than trusted to the model: they are requirements that are not met yet, and
    # dropping one erases "this is still missing" from the ruler.
    for it in base['items']:
        v = values.get(it['id'])
        if it['id'] in seen or (v is not None and v >= 1.0):
            continue
        row = dict(it)
        row['origin'], row['weight'] = 'task', W_TASK
        items.append(row)
        bad.append(f"{it['id']}: dropped while unsaturated, restored")
        seen.add(it['id'])

    brief_items = [x for x in items if x['origin'] == 'brief']
    if len(brief_items) < 3:
        bad.append(f'only {len(brief_items)} brief items')
    # Brief items come first: this is the order the checklist is read in, and
    # the order the judge sees when scoring.
    order = {'brief': 0, 'task': 1, 'guard': 2}
    items.sort(key=lambda x: order[x['origin']])
    retired = [x for x in (out.get('retired') or []) if isinstance(x, dict) and x.get('id') in by]
    return items[:MAX_ITEMS], retired, bad


def build(game: str, brief: dict, base: dict, values: dict, *, force: bool = False) -> dict:
    """-> the stage checklist. Cached on (task hash, brief hash)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    bh = hashlib.sha256(json.dumps(brief, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
    th = base.get('task_hash') or ''
    out_p = CACHE / f'{game}.{SCHEMA}.{th}.{bh}.json'
    if out_p.is_file() and not force:
        return json.loads(out_p.read_text())
    body = _PROMPT.format(dims=DIMENSIONS, base=_base_text(base, values),
                          brief=brief_text(brief), max_items=MAX_ITEMS)
    last = None
    for _ in range(3):
        parsed, raw = ask([{'role': 'user', 'content': [{'type': 'text', 'text': body}]}],
                          max_tokens=8000)
        if not isinstance(parsed, dict):
            last = f'no parsable JSON returned: {str(raw)[:200]}'
            continue
        items, retired, bad = _clean(parsed, base, values)
        hard = [b for b in bad if 'restored' not in b and 'dropped' not in b]
        if items and len(hard) <= 2 and sum(1 for x in items if x['origin'] == 'brief') >= 3:
            rb = {'game': game, 'schema': SCHEMA, 'task_hash': th, 'brief_hash': bh,
                  'items': items, 'retired': retired, 'complaints': bad,
                  'weights': {'brief': W_BRIEF, 'task': W_TASK, 'guard': W_GUARD},
                  'pstar_values': values}
            out_p.write_text(json.dumps(rb, indent=1, ensure_ascii=False))
            n = {k: sum(1 for x in items if x['origin'] == k) for k in ('brief', 'task', 'guard')}
            print(f"[stage] {game}: brief {n['brief']} · kept {n['task']} · guards {n['guard']} · "
                  f"retired {len(retired)} -> {out_p.name}", flush=True)
            return rb
        last = '; '.join(bad[:4]) or 'no items'
    raise SystemExit(f'{game}: stage checklist not written: {last}')
