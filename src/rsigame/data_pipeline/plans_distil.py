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
"""Stage-2 plan rows for Godot, distilled from the brief by a teacher model.

WHY THIS EXISTS
    The web half of the plan corpus is EXTRACTED: 147 of 150 web trajectories
    contain an explicit "Files to MODIFY / CREATE" block the agent wrote before
    coding. The Godot half yields none -- 0 of 150 -- because that agent writes
    prose intent instead. Both numbers are measured. So Godot plans cannot be
    extracted at any price; they can only be distilled from the brief.

COST, MEASURED RATHER THAN GUESSED
    876 Godot games at ~1.59k input tokens each (42 system + 1,549 brief) and
    ~510 output: 1.39M in, 0.45M out. `--dry-run` reprints this from the actual
    corpus, with the real prompt, before anything is spent.

TENSE -- THE NON-OBVIOUS PART
    The extracted web plans are RETROSPECTIVE: "already merged the GDD fields;
    will read `difficultyConfig` in gameplay code". They were recorded after
    the archetype workflow had already scaffolded the project, so they mix past
    and future. A plan distilled cold from a brief is naturally PROSPECTIVE
    ("will create ..."), and mixing the two registers under one identical
    system prompt teaches two different behaviours for the same instruction.
    `--tense match` (the default) shows the model real web plans as few-shot
    examples so the distilled plans land in the same register; `--tense natural`
    skips the priming if you would rather keep the distillation unprimed.

TWO TRAPS THIS MODULE AVOIDS
  * The system prompt is READ FROM THE CORPUS, never retyped. Three plan
    corpora pool into one training set only because their system turn matches
    exactly; one stray character silently splits it into two tasks.
  * An accepted plan must name both sections. A teacher that answers with prose
    has not produced a plan, and a corpus that accepts it teaches the stage-2
    instruction to mean nothing in particular.

CREDENTIALS come from the environment through the same client the rest of the
project uses, and are never written to disk.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import jsonl

# a plan that names neither section is not a plan; this is the accept gate
NEEDS = (re.compile(r'Files to MODIFY', re.I), re.compile(r'CREATE', re.I))


def default_teacher() -> str:
    return os.environ.get('RSIGAME_CORPUS_PLAN_TEACHER', '')


def system_prompt(plan_rows: list[dict]) -> str:
    """The exact string the existing plan corpus uses. Refuse if it is not unique."""
    seen = {r['messages'][0]['content'] for r in plan_rows
            if r['messages'][0]['role'] == 'system'}
    if len(seen) != 1:
        raise SystemExit(f'REFUSING: plan corpus has {len(seen)} distinct system '
                         'prompts, expected exactly 1')
    return seen.pop()


def brief_of(row: dict) -> str | None:
    for m in row['messages']:
        if m['role'] == 'user':
            return m.get('content')
    return None


def call_retry(client, model: str, messages: list[dict], timeout: int, tries: int = 4) -> str | None:
    """One completion, retried. Returns None if the model never answered.

    Total backoff stays well inside 300s: past that the caller stops being able
    to tell a slow model from a silent one, and starts recording silence as a
    rejection.
    """
    for i in range(tries):
        try:
            r = client.chat.completions.create(model=model, messages=messages,
                                               temperature=0, timeout=timeout)
            txt = (r.choices[0].message.content or '').strip()
            if txt:
                return txt
        except Exception:                      # noqa: BLE001 - any transport error retries
            pass
        time.sleep(min(2 ** i, 20) + random.random())
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog='rsigame.data_pipeline plans-distil', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gen', required=True, help='stage-1 corpus (the source of Godot briefs)')
    ap.add_argument('--plans', required=True, help='the extracted web plans (system prompt + style)')
    ap.add_argument('--out', required=True, help='JSONL, appended as rows land (resumable)')
    ap.add_argument('--model', default=default_teacher(),
                    help='teacher model (RSIGAME_CORPUS_PLAN_TEACHER)')
    ap.add_argument('--lanes', type=int, default=6, help='6 measured clean, with no rate limiting')
    ap.add_argument('--limit', type=int, default=0, help='0 = all')
    ap.add_argument('--timeout', type=int, default=180)
    ap.add_argument('--tense', choices=('match', 'natural'), default='match')
    ap.add_argument('--shots', type=int, default=2)
    ap.add_argument('--dry-run', action='store_true', help='cost + prompt preview, no API calls')
    a = ap.parse_args(argv)
    if not a.model:
        ap.error('no teacher model: pass --model or set RSIGAME_CORPUS_PLAN_TEACHER')

    plans = list(jsonl.read(a.plans))
    system = system_prompt(plans)
    godot = [r for r in jsonl.read(a.gen) if r.get('engine') == 'godot']

    done: set[str] = set()
    if Path(a.out).exists():
        done = {r['task'] for r in jsonl.read(a.out)}
    todo = [r for r in godot if r['task'] not in done and brief_of(r)]
    if a.limit:
        todo = todo[:a.limit]

    shots: list[dict] = []
    if a.tense == 'match':
        for e in sorted(plans, key=lambda r: r['task'])[:a.shots]:
            shots.append({'role': 'user', 'content': e['messages'][1]['content']})
            shots.append({'role': 'assistant', 'content': e['messages'][2]['content']})

    shot_chars = sum(len(s['content']) for s in shots)
    est_in = sum((len(system) + len(brief_of(r) or '') + shot_chars) // 4 for r in todo)
    est_out = 510 * len(todo)
    print(f'godot games        : {len(godot):,}   already done: {len(done):,}   to do: {len(todo):,}')
    print(f'style priming      : {a.tense}' + (f' ({a.shots} shots)' if shots else ''))
    print(f'est input tokens   : {est_in / 1e6:.2f}M')
    print(f'est output tokens  : {est_out / 1e6:.2f}M')
    print(f'model              : {a.model}')
    if a.dry_run:
        print('\n--- DRY RUN, nothing sent ---')
        if todo:
            print(f"first task: {todo[0]['task']}")
            print(f'system    : {system}')
            print(f'brief head: {(brief_of(todo[0]) or "")[:300]}')
        return 0

    from ..agent.llm import _client        # the project's one client, not a second one
    client = _client(a.model)

    briefs = {r['task']: brief_of(r) for r in todo}
    ok = bad = 0

    def work(row: dict) -> tuple[str, str | None]:
        msgs = ([{'role': 'system', 'content': system}] + shots
                + [{'role': 'user', 'content': briefs[row['task']]}])
        txt = call_retry(client, a.model, msgs, a.timeout)
        if not txt or not all(p.search(txt) for p in NEEDS):
            return row['task'], None
        return row['task'], txt

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'a') as fh, ThreadPoolExecutor(max_workers=a.lanes) as pool:
        for i, (task, txt) in enumerate(pool.map(work, todo), 1):
            if txt is None:
                bad += 1
            else:
                fh.write(json.dumps({
                    'task': task, 'engine': 'godot', 'source': f'distil:{a.model}',
                    'messages': [{'role': 'system', 'content': system},
                                 {'role': 'user', 'content': briefs[task]},
                                 {'role': 'assistant', 'content': txt}],
                }, ensure_ascii=False) + '\n')
                fh.flush()   # resumable: a kill costs one row, not the batch
                ok += 1
            if i % 25 == 0:
                print(f'  {i}/{len(todo)}  ok={ok} rejected={bad}', flush=True)

    print(f'\nwritten {ok:,} plans -> {a.out}   rejected {bad:,} '
          '(no MODIFY/CREATE block, or the model never answered)')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
