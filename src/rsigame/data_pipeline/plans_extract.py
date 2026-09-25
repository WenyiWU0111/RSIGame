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
"""Stage-2 plan rows, extracted from web sessions that wrote a plan first.

Stage 2 teaches planning on top of generation: brief in, a file-level
implementation plan out. The web half of the corpus yields those plans by
EXTRACTION -- 147 of 150 web trajectories contain an explicit
"Files to MODIFY / CREATE" block the agent wrote before it started coding.
(The Godot half yields none, which is why `plans_distil` exists.)

THE BUG THIS MODULE EXISTS TO NOT HAVE, MEASURED
    The first version took the first text turn of the LAST session that made
    tool calls. 49.4% of games have more than one session, so that turn is a
    RESUMPTION's status recap rather than a plan -- and 437 of 454 targets
    (96.3%) contained the word "already": "already merged", "already set
    LEVEL_ORDER", "already added imports". Status reports, not plans. The
    shape filter passed them because it checks shape, not tense.

    The fix is the re-scan, not the regex: scan sessions IN ORDER and take the
    first text turn that is a plan and is not past-tense. Rejecting
    `\\balready\\b` alone would have left 17 rows. The regex is only the guard,
    and it is asserted at the end of the run.

THE SYSTEM PROMPT IS PART OF THE CORPUS
    Three plan corpora pool into one training set only because this exact
    string is their system turn. `plans_merge` checks that byte for byte.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from . import jsonl
from .web_trajectories import split_sessions

SYSTEM = ('You are a senior game engineer. Given a game brief, produce a '
          'file-level implementation plan: which files to MODIFY and which to '
          'CREATE, with a one-line reason for each.')

PLAN_SHAPE = re.compile(r'Files to (MODIFY|CREATE)', re.I)
PAST = re.compile(r'\balready\b', re.I)


def archetype_of(name: str) -> str:
    """`puzzle-glacier-glyphs-D3-07` -> `puzzle-glacier-glyphs`.

    Tasks are generated in families; the suffix is the variant. The archetype
    is what a split must hold out on, not the task name.
    """
    return re.sub(r'-D\d+-\d+$', '', name)


def find_plan(segs: list[dict]) -> tuple[str, str] | None:
    """-> (brief, plan) from the earliest session that wrote a forward plan."""
    for seg in segs:                       # IN ORDER -- the bug above was segs[-1]
        if not any(e.get('t') == 'tool_use' for e in seg['events']):
            continue
        brief = ((seg.get('session') or {}).get('prompt') or '').strip()
        if not brief:
            continue
        for e in seg['events']:
            if e.get('t') != 'text':
                continue
            txt = (e.get('text') or '').strip()
            if not txt:
                continue
            if not PLAN_SHAPE.search(txt):
                break                      # the first text is a summary, not a plan
            if PAST.search(txt):
                break                      # past-tense recap -> try the next session
            return brief, txt
    return None


def trajectories_under(root: Path) -> dict[str, Path]:
    """task name -> its trajectory, for either recording layout.

        layout A   <task>/.qwen/trajectory.jsonl
        layout B   trajectories/<task>/trajectory.jsonl
    """
    found: dict[str, Path] = {}
    for p in root.rglob('trajectory.jsonl'):
        task = p.parent.parent.name if p.parent.name == '.qwen' else p.parent.name
        found.setdefault(task, p)
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog='rsigame.data_pipeline plans-extract', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--roots', nargs='+', required=True, help='directories holding web recordings')
    ap.add_argument('--out', required=True)
    ap.add_argument('--tasks', help='optional .jsonl or newline list restricting which tasks are read; '
                                    'use it to reuse an artifact check already done elsewhere')
    a = ap.parse_args(argv)

    trajs: dict[str, Path] = {}
    for root in a.roots:
        trajs.update(trajectories_under(Path(root)))
    print(f'trajectories found on disk: {len(trajs)}')

    wanted = sorted(trajs)
    if a.tasks:
        names = set()
        for line in Path(a.tasks).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            names.add(json.loads(line)['task'] if line.startswith('{') else line)
        wanted = sorted(names)
        print(f'task list: {len(wanted)}')

    kept: list[dict] = []
    drops: dict[str, int] = {}

    def drop(why: str) -> None:
        drops[why] = drops.get(why, 0) + 1

    for task in wanted:
        p = trajs.get(task)
        if not p:
            drop('no trajectory on disk')
            continue
        try:
            segs = split_sessions(list(jsonl.read(p)))
        except OSError:
            drop('unreadable trajectory')
            continue
        found = find_plan(segs)
        if not found:
            drop('no forward plan in any session')
            continue
        brief, plan = found
        kept.append({
            'task': task,
            'archetype': archetype_of(task),
            'engine': 'web',
            'source': 'extracted',
            'n_chars': len(plan),
            'messages': [
                {'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': brief},
                {'role': 'assistant', 'content': plan},
            ],
        })

    jsonl.write(a.out, kept)
    print(f'\nkept {len(kept)} forward plans -> {a.out}')
    for k, v in sorted(drops.items(), key=lambda kv: -kv[1]):
        print(f'  dropped {v:4d}  {k}')

    # The guard, asserted rather than hoped for: this is the failure the whole
    # module is built around, so a run that reintroduces it must not look fine.
    bad = sum(1 for r in kept if PAST.search(r['messages'][-1]['content']))
    print(f'\nGUARD: targets still matching \\balready\\b = {bad} (must be 0)')
    if bad:
        return 1
    if kept:
        lens = sorted(r['n_chars'] for r in kept)
        print(f'plan chars: p50 {lens[len(lens) // 2]}  '
              f'p90 {lens[int(len(lens) * .9)]}  max {lens[-1]}')
    return 0 if kept else 1


if __name__ == '__main__':
    raise SystemExit(main())
