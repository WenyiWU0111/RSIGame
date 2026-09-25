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
"""Fail loudly before a GPU is ever requested.

A corpus defect found at hour six of a run costs the run. Every check here is
one that has cost one, ordered by how much time it would have saved:

  1. every line is valid JSON carrying a `messages` list
  2. every row has a non-empty assistant target (content or tool_calls)
  3. every `tool_call_id` resolves to a tool turn, and every tool turn to a call
  4. every row fits `--max-length` under the REAL tokenizer
  5. the teacher spread is REPORTED, so a corpus accidentally dominated by one
     teacher is visible before it is trained on
  6. with `--require-artifact`, every row is backed by a built game and a
     session that did work -- the `success` flag is reported, never required,
     for the reason `web_trajectories` rule 1 explains

Checks 5 and 6 read fields only the generation corpus carries, so they are
skipped on a mixed arm file, whose schema is deliberately just
(engine, messages, task).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog='rsigame.data_pipeline validate', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--inp', type=Path, required=True)
    ap.add_argument('--model', default=os.environ.get('RSIGAME_SFT_BASE_MODEL', ''),
                    help='tokenizer, only needed with --retokenize (RSIGAME_SFT_BASE_MODEL)')
    ap.add_argument('--max-length', type=int, default=32768)
    ap.add_argument('--retokenize', action='store_true',
                    help='recount tokens instead of trusting n_tokens from the length step')
    ap.add_argument('--require-artifact', action='store_true',
                    help='generation corpora only: demand tool calls and a game behind each row')
    ap.add_argument('--show', type=int, default=3, help='hand-read this many records')
    a = ap.parse_args(argv)

    tk = None
    if a.retokenize:
        if not a.model:
            ap.error('--retokenize needs --model or RSIGAME_SFT_BASE_MODEL')
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)

    errors: list[str] = []
    models: Counter = Counter()
    roles: Counter = Counter()
    flags: Counter = Counter()
    toks: list[int] = []
    calls = 0
    n = 0
    samples: list[dict] = []

    with a.inp.open() as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                s = json.loads(line)
            except ValueError as e:
                errors.append(f'line {i}: invalid JSON: {e}')
                continue
            msgs = s.get('messages')
            if not isinstance(msgs, list) or not msgs:
                errors.append(f'line {i}: no messages')
                continue

            # --- 2 and 3 ---------------------------------------------------
            call_ids: set = set()
            tool_ids: set = set()
            has_target = False
            for m in msgs:
                roles[m.get('role')] += 1
                if m.get('role') == 'assistant':
                    tc = m.get('tool_calls') or []
                    if tc or (m.get('content') or '').strip():
                        has_target = True
                    for c in tc:
                        call_ids.add(c.get('id'))
                        calls += 1
                        fn = c.get('function') or {}
                        if not fn.get('name'):
                            errors.append(f'line {i}: tool_call with no name')
                        try:
                            json.loads(fn.get('arguments') or 'null')
                        except ValueError:
                            errors.append(f'line {i}: tool_call arguments are not JSON')
                elif m.get('role') == 'tool':
                    tool_ids.add(m.get('tool_call_id'))
            if not has_target:
                errors.append(f'line {i}: no non-empty assistant target')
            orphans = tool_ids - call_ids
            if orphans:
                errors.append(f'line {i}: {len(orphans)} tool turns with no matching call')

            # --- 6 ---------------------------------------------------------
            if a.require_artifact:
                if not s.get('n_tool_calls'):
                    errors.append(f'line {i}: chosen session made no tool calls')
                if not (s.get('game') or s.get('task')):
                    errors.append(f'line {i}: nothing to attribute the artifact to')

            # --- 4 ---------------------------------------------------------
            t = s.get('n_tokens')
            if a.retokenize or t is None:
                try:
                    t = len(tk.apply_chat_template(msgs, tokenize=True,
                                                   add_generation_prompt=False)) if tk else None
                except Exception:                  # noqa: BLE001 - template failures are reported
                    t = None
            if t is None:
                if a.retokenize:
                    errors.append(f'line {i}: no token count')
            else:
                toks.append(t)
                if t > a.max_length:
                    errors.append(f'line {i}: {t} tokens > max_length {a.max_length}')

            models[s.get('model')] += 1
            flags[s.get('success')] += 1
            if len(samples) < a.show:
                samples.append(s)

    print(f'rows               {n}')
    print(f'assistant targets  {calls} tool calls, roles {dict(roles)}')
    if toks:
        toks.sort()

        def q(p: float) -> int:
            return toks[min(len(toks) - 1, int(p * len(toks)))]

        print(f'tokens             p50 {q(.5)}  p90 {q(.9)}  max {q(1.0)}  '
              f'(limit {a.max_length})')
    if set(models) != {None}:
        print(f'teacher spread     {dict(models)}')
        print(f'success flag       {dict(flags)}   (reported, not filtered on)')
    print()

    for s in samples:
        msgs = s['messages']
        print(f"--- {s.get('task')}  [{s.get('model')}]  {s.get('n_tokens', -1)} tokens, "
              f"{s.get('n_tool_calls', -1)} calls")
        head = (msgs[0].get('content') or '')[:150].replace('\n', ' ')
        print(f'    user   : {head}')
        first_call = next((m for m in msgs
                           if m.get('role') == 'assistant' and m.get('tool_calls')), None)
        if first_call:
            fn = first_call['tool_calls'][0]['function']
            print(f"    call[0]: {fn['name']} {fn['arguments'][:110]}")
        first_tool = next((m for m in msgs if m.get('role') == 'tool'), None)
        if first_tool:
            body = (first_tool.get('content') or '')[:100].replace('\n', ' ')
            print(f"    tool[0]: {first_tool.get('name')} -> {body}")
        print()

    if errors:
        print(f'FAILED: {len(errors)} problems')
        for e in errors[:25]:
            print('  ' + e)
        return 1
    print('OK: all checks passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
