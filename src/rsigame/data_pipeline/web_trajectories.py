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
"""A web agent's recording -> one training row per session.

The web recorder writes a JSONL event stream (`.qwen/trajectory.jsonl`):
`session` headers, `tool_use` and `tool_result` pairs, free `text` turns, and
an `end` record. One row per game comes out, in ms-swift `messages` shape.

THREE RULES, EACH FROM A MEASUREMENT ON THE LIVE CORPUS, NOT FROM TASTE

  1. THE ARTIFACT DECIDES WHAT IS KEPT, NOT THE SUCCESS FLAG. Cross-tabulated
     over 428 games: 404 have both a built `dist/` and a session flagged
     success, but 5 have a built game and NO successful session -- three died
     on a 429 rate-limit AFTER 104-112 tool calls, two have no `end` record at
     all. Keying on `end.success` throws those built games away. So: keep the
     LAST session that made tool calls, and require `dist/index.html` on the
     game directory. Against the flag-based rule this is a strict superset
     (+4 games, +441 calls, nothing lost, and no case where the chosen session
     was superseded by a later one). Sessions that did no work still drop out
     by themselves -- one teacher variant contributed 162 sessions and zero
     tool calls between them.

  2. ONE SESSION PER ROW. An earlier version concatenated tool calls from
     EVERY session in the file into a single conversation. Measured: 47,333
     tool calls sit in sessions that succeeded and 7,735 in ones that failed,
     across 176 failed sessions carrying tools -- so that version trained on
     ~14% crashed-run actions and labelled 203 multi-session games with the
     failed attempt's model. Segment on `session`, emit one segment.

  3. THE CHECKLIST IS WRITTEN ONCE. The agent re-emits its whole todo list on
     every update with one flag changed; those arguments measure 14.9 M chars,
     27% of the training target. Keep the FIRST `todo_write` call -- that one
     is the plan -- and drop the rest.

Loss masking is ms-swift's default for `messages`: only assistant turns are
supervised, so every step is supervised exactly once.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from . import jsonl

MIN_TRAJ_BYTES = 10_240          # below this the recording is an empty shell
RESULT_CAP = 2_000               # default per-result ceiling
SHELL_HEAD, SHELL_TAIL = 1_200, 800
TODO_ACK = '[todo list updated]'


def default_roots() -> list[str]:
    """Where the recordings are. Machine-local, so it is configuration.

    `RSIGAME_CORPUS_WEB_ROOTS` is a comma-separated list; there is no built-in
    default, because a wrong default silently converts nothing and reports a
    clean zero.
    """
    v = os.environ.get('RSIGAME_CORPUS_WEB_ROOTS', '')
    return [r for r in (x.strip() for x in v.split(',')) if r]


def trim_result(name: str, text: str) -> str:
    """Shrink a tool result to what the next action could plausibly depend on.

    The ceilings are measured medians, not round numbers picked to look tidy:
    a shell result's first 1,200 and last 800 characters carry the command and
    its verdict, and the middle is build spam.
    """
    if not text:
        return ''
    if name == 'todo_write':
        return TODO_ACK
    if name == 'run_shell_command' and len(text) > SHELL_HEAD + SHELL_TAIL:
        cut = len(text) - SHELL_HEAD - SHELL_TAIL
        return (text[:SHELL_HEAD] + f'\n...[{cut} chars elided]...\n'
                + text[-SHELL_TAIL:])
    if len(text) > RESULT_CAP:
        return text[:RESULT_CAP] + f'\n...[{len(text) - RESULT_CAP} chars elided]...'
    return text


def split_sessions(recs: list[dict]) -> list[dict]:
    """-> [{session, end, events:[...]}].

    A segment runs from one `session` record to its `end`. Trailing events with
    no `end` form an unterminated segment (end=None), which rule 1 may still
    keep -- a run killed by a rate limit after 100 tool calls is a real
    recording of real work.
    """
    segs: list[dict] = []
    cur = None
    for r in recs:
        t = r.get('t')
        if t == 'session':
            if cur is not None:
                segs.append(cur)
            cur = {'session': r, 'end': None, 'events': []}
        elif cur is None:
            continue              # events before any session header
        elif t == 'end':
            cur['end'] = r
            segs.append(cur)
            cur = None
        else:
            cur['events'].append(r)
    if cur is not None:
        segs.append(cur)
    return segs


def build_messages(seg: dict, drop_late_todo: bool = True) -> tuple[list[dict], int, int, int]:
    """One segment -> messages. (messages, n_calls, n_todo_dropped, n_incomplete)."""
    prompt = (seg['session'].get('prompt') or '').strip()
    messages = [{'role': 'user', 'content': prompt}]

    # pair on id, never on position: a result can arrive after the next call starts
    results = {r.get('id'): r for r in seg['events'] if r.get('t') == 'tool_result'}

    pending: list[dict] = []
    n_calls = 0
    todo_seen = 0
    todo_dropped = 0
    dropped_incomplete = 0

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        messages.append({
            'role': 'assistant',
            'tool_calls': [{'id': c['id'], 'type': 'function',
                            'function': {'name': c['name'],
                                         'arguments': json.dumps(c.get('input') or {},
                                                                 ensure_ascii=False)}}
                           for c in pending]})
        for c in pending:
            res = results.get(c['id'])
            messages.append({'role': 'tool', 'tool_call_id': c['id'],
                             'name': c['name'],
                             'content': trim_result(c['name'], (res or {}).get('text') or '')})
        pending = []

    for r in seg['events']:
        t = r.get('t')
        if t == 'tool_use':
            # a run cut off mid-flight (rate limit, truncation) leaves a final
            # call with no result; emitting it would teach a call answered by
            # silence
            if r.get('id') not in results:
                dropped_incomplete += 1
                continue
            if r.get('name') == 'todo_write':
                todo_seen += 1
                if drop_late_todo and todo_seen > 1:
                    todo_dropped += 1
                    continue      # keep the first (the plan), drop status updates
            pending.append(r)
            n_calls += 1
        elif t == 'text' and (r.get('text') or '').strip():
            flush()
            messages.append({'role': 'assistant', 'content': r['text']})
    flush()
    return messages, n_calls, todo_dropped, dropped_incomplete


def convert(traj: Path, drop_late_todo: bool = True) -> tuple[dict | None, str]:
    """One recording -> one row, or (None, why it was dropped)."""
    recs = list(jsonl.read(traj))
    if not recs:
        return None, 'no parseable records'
    segs = split_sessions(recs)
    if not segs:
        return None, 'no session records'

    with_tools = [s for s in segs
                  if any(e.get('t') == 'tool_use' for e in s['events'])]
    if not with_tools:
        return None, 'no session made tool calls'
    seg = with_tools[-1]                           # never superseded, by construction
    if not (seg['session'].get('prompt') or '').strip():
        return None, 'chosen session has no prompt'

    messages, n_calls, todo_dropped, n_incomplete = build_messages(seg, drop_late_todo)
    if n_calls == 0:
        return None, 'chosen session has no usable tool calls'

    end = seg['end'] or {}
    return {
        'task': seg['session'].get('label'),
        'engine': 'web',
        'model': seg['session'].get('model') or end.get('resolvedModel'),
        'n_tool_calls': n_calls,
        'n_sessions_in_file': len(segs),
        'n_sessions_dropped': len(segs) - 1,
        'todo_calls_dropped': todo_dropped,
        'incomplete_calls_dropped': n_incomplete,
        # Recorded for reporting and NEVER used as a filter: False here means
        # the SDK session ended badly -- often a rate limit -- not that the
        # game is bad. Rule 1 is the whole reason this is not a filter.
        'success': end.get('success'),
        'session_terminated': seg['end'] is not None,
        'stopped_by': end.get('stoppedBy'),
        'wall_ms': end.get('wallMs'),
        'messages': messages,
    }, ''


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog='rsigame.data_pipeline web', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--roots', default=','.join(default_roots()),
                    help='comma-separated recording roots (RSIGAME_CORPUS_WEB_ROOTS)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--require-dist', action='store_true', default=True,
                    help='drop games with no dist/index.html (default: on)')
    ap.add_argument('--no-require-dist', dest='require_dist', action='store_false')
    ap.add_argument('--keep-late-todo', dest='drop_late_todo', action='store_false',
                    default=True, help='keep every todo_write call (default: first only)')
    a = ap.parse_args(argv)
    roots = [r for r in (x.strip() for x in a.roots.split(',')) if r]
    if not roots:
        ap.error('no recording roots: pass --roots or set RSIGAME_CORPUS_WEB_ROOTS')

    a.out.parent.mkdir(parents=True, exist_ok=True)
    skipped: Counter = Counter()
    seen: set[str] = set()
    kept = 0
    raw_chars = out_chars = 0
    calls_kept = todo_dropped_total = sessions_dropped_total = 0

    with a.out.open('w') as fh:
        for root in roots:
            p = Path(root)
            if not p.is_dir():
                skipped['root missing'] += 1
                continue
            for d in sorted(p.iterdir()):
                if not d.is_dir():
                    continue
                traj = d / '.qwen' / 'trajectory.jsonl'
                if not traj.is_file():
                    skipped['no trajectory'] += 1
                    continue
                if traj.stat().st_size < MIN_TRAJ_BYTES:
                    skipped['recording is an empty shell (<10KB)'] += 1
                    continue
                if a.require_dist and not (d / 'dist' / 'index.html').is_file():
                    skipped['no dist/index.html'] += 1
                    continue
                if d.name in seen:
                    skipped['duplicate name'] += 1
                    continue

                sample, why = convert(traj, a.drop_late_todo)
                if sample is None:
                    skipped[why] += 1
                    continue

                sample['game'] = d.name
                raw_chars += traj.stat().st_size
                line = json.dumps(sample, ensure_ascii=False)
                seen.add(d.name)
                fh.write(line + '\n')
                kept += 1
                out_chars += len(line)
                calls_kept += sample['n_tool_calls']
                todo_dropped_total += sample['todo_calls_dropped']
                sessions_dropped_total += sample['n_sessions_dropped']

    print(f'wrote {kept} samples -> {a.out}')
    if kept and raw_chars:
        print(f'  {out_chars/1e6:.1f} M chars out of {raw_chars/1e6:.1f} M raw '
              f'({100*out_chars/raw_chars:.0f}%)')
        print(f'  tool calls kept               {calls_kept}')
        print(f'  todo_write calls dropped      {todo_dropped_total}')
        print(f'  failed/extra sessions dropped {sessions_dropped_total}')
    print('  skipped:')
    for k, v in skipped.most_common():
        print(f'    {v:5d}  {k}')
    return 0 if kept else 1


if __name__ == '__main__':
    raise SystemExit(main())
