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
"""Whole sessions -> one row per agent decision: persistent prefix + sliding window.

WHY
    A stage-1 row is one complete agent session -- up to 81,061 tokens -- and
    that single number drove every infrastructure problem this project had:
    out-of-memory on four cards, an unusable eight-card path (the model's 4 KV
    heads cap Ulysses at sp=4, and ring attention is unimplemented for its
    linear-attention blocks), and a standing choice between truncating rows and
    deleting them.

    But a trajectory is not one decision. It is dozens. Training on the whole
    session as a single sample spends an 81k context window to supervise the
    same assistant turns a 32k window supervises individually.

THE SHAPE OF EACH SAMPLE

    [ persistent prefix ][ sliding window of recent history ][ TARGET turn ]

    persistent prefix   the brief (always), plus optionally the first assistant
                        turn, which in this corpus is where the agent lays out
                        its plan. Without the brief the model cannot know which
                        game it is building -- the engine is named there and
                        nowhere else.
    sliding window      as many immediately preceding messages as fit the token
                        budget, filled BACKWARDS from the target.
    target              exactly one assistant turn: the decision being taught.

TWO INVARIANTS THAT ARE EASY TO BREAK

  1. A `tool` message is meaningless without the assistant `tool_call` that
     produced it -- they pair on id. Filling backwards can strand a result
     whose call fell outside the window, which teaches a result arriving from
     nowhere. Stranded results are dropped.
  2. The window must never start mid-batch. A web assistant turn carries a
     MEDIAN OF 50 tool calls (max 96); half of one is not a state the agent
     ever occupies.

THE OPEN QUESTION THIS MODULE DOES NOT DECIDE
    ms-swift's `messages` format supervises EVERY assistant turn. History turns
    inside the window are therefore trained on again each time they appear in
    another sample's window -- the same tokens counted many times, which is not
    what "one sample per decision" means. `--history-as-context` rewrites
    history into a single user turn so only the target is supervised; the
    default leaves it alone. Measure both before trusting either. (The training
    side has a second lever for the same question: `sft.loss_scale`.)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from . import jsonl

CHARS_PER_TOKEN = 1.0 / 0.2575  # calibrated on the 409 rows carrying n_tokens


def est(obj) -> int:
    """Token estimate. Cheap on purpose: this is called once per message, and
    the calibration above is what makes an estimate good enough to budget with."""
    return int(len(json.dumps(obj, ensure_ascii=False)) / CHARS_PER_TOKEN)


def call_ids(m: dict) -> set[str]:
    return {c['id'] for c in (m.get('tool_calls') or [])}


def render_history(msgs: list[dict]) -> dict:
    """Collapse history into one user turn, so only the target is supervised."""
    parts = []
    for m in msgs:
        r = m['role']
        if r == 'assistant':
            if m.get('content'):
                parts.append(f"[assistant] {m['content']}")
            for c in (m.get('tool_calls') or []):
                parts.append(f"[assistant calls {c['function']['name']}] "
                             f"{c['function']['arguments']}")
        elif r == 'tool':
            parts.append(f"[result] {m.get('content', '')}")
        else:
            parts.append(f"[{r}] {m.get('content', '')}")
    return {'role': 'user', 'content': 'Recent history:\n' + '\n'.join(parts)}


def split_row(row: dict, budget: int, keep_plan: bool, hist_as_ctx: bool) -> list[dict]:
    ms = row['messages']
    # Size every message ONCE. The first version called est() inside the
    # backward-fill loop, re-serialising 76,000-character assistant turns
    # thousands of times per trajectory -- 300 rows took minutes instead of
    # seconds. Arithmetic on ints is the whole fix.
    sz = [est(m) for m in ms]
    if not ms or ms[0]['role'] != 'user':
        return []
    brief = ms[0]

    # indices of assistant turns -- each is one decision to supervise
    targets = [i for i, m in enumerate(ms) if m['role'] == 'assistant']
    if not targets:
        return []

    prefix = [brief]
    if keep_plan and len(targets) > 1:
        # the first assistant turn is where the plan lives; keep it as context
        # for every later decision, but never as its own prefix
        plan = ms[targets[0]]
        if sz[targets[0]] < budget // 4:      # only if it is not itself enormous
            prefix = [brief, plan]

    out = []
    for ti in targets:
        target = ms[ti]
        # The pinned plan turn IS targets[0]. Including it in the prefix of its
        # own sample shows the model the answer and then asks for it -- label
        # leakage that would read as a suspiciously good loss. Use the bare
        # brief for that one decision.
        pfx = [brief] if (len(prefix) > 1 and ti == targets[0]) else prefix
        base = sum(est(p) for p in pfx) + sz[ti]
        if base > budget:
            # a single web assistant turn can be 28k tokens on its own; it
            # cannot be shrunk without splitting a tool-call batch, so emit it
            # with the brief alone and let the caller decide
            out.append({'messages': [brief, target], 'oversize': True,
                        'tokens': est([brief, target])})
            continue

        # fill backwards, never starting mid-batch
        win: list[dict] = []
        used = base
        j = ti - 1
        while j > 0:
            if used + sz[j] > budget:
                break
            win.insert(0, ms[j])
            used += sz[j]
            j -= 1

        # drop tool results whose originating call fell outside the window
        available: set[str] = set()
        for m in pfx + win:
            available |= call_ids(m)
        win = [m for m in win
               if m['role'] != 'tool' or m.get('tool_call_id') in available]

        msgs = pfx + ([render_history(win)] if hist_as_ctx and win else win) + [target]
        out.append({'messages': msgs, 'oversize': False, 'tokens': est(msgs)})

    for k, s in enumerate(out):
        s.update(task=row.get('task'), engine=row.get('engine'),
                 parent_tokens=row.get('n_tokens'), decision_index=k,
                 n_decisions=len(out))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog='rsigame.data_pipeline decisions', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--inp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--budget', type=int, default=32768, help='token budget per sample')
    ap.add_argument('--no-plan', dest='keep_plan', action='store_false',
                    help='do not pin the first assistant turn into the prefix')
    ap.add_argument('--history-as-context', action='store_true',
                    help='collapse the window into one user turn, so ONLY the target is supervised')
    ap.add_argument('--limit', type=int, default=0, help='process only N rows (prototyping)')
    a = ap.parse_args(argv)

    rows = list(jsonl.read(a.inp))
    if a.limit:
        rows = rows[:a.limit]

    samples, stats = [], Counter()
    for r in rows:
        s = split_row(r, a.budget, a.keep_plan, a.history_as_context)
        stats[f"{r.get('engine')}_rows"] += 1
        stats[f"{r.get('engine')}_samples"] += len(s)
        stats['oversize'] += sum(1 for x in s if x['oversize'])
        samples += s

    jsonl.write(a.out, samples)

    toks = sorted(s['tokens'] for s in samples)
    n = len(toks)
    print(f'input rows      : {len(rows):,}')
    print(f'output samples  : {n:,}   ({n / max(len(rows), 1):.1f} per trajectory)')
    print(f'budget          : {a.budget:,}   history-as-context={a.history_as_context}')
    if n:
        print(f'tokens          : median {toks[n // 2]:,}  p90 {toks[int(.9 * n)]:,}  '
              f'p99 {toks[int(.99 * n)]:,}  max {toks[-1]:,}')
        print(f'over budget     : {sum(1 for t in toks if t > a.budget)}  '
              f"(oversize single turns: {stats['oversize']})")
    for k in sorted(stats):
        if k.endswith(('_rows', '_samples')):
            print(f'  {k:20s} {stats[k]:,}')
    return 0 if samples else 1


if __name__ == '__main__':
    raise SystemExit(main())
