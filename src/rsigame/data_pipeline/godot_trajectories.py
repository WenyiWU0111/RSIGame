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
"""A Godot agent's recording -> one training row per task.

The Godot counterpart of `web_trajectories`. The two recorders are different
programs and the differences are not cosmetic:

    web     .jsonl event stream, `t=session|tool_use|tool_result|end`,
            a vocabulary of 14 named tools
    godot   a single `trajectory.json` document: `steps[]`, each carrying a
            message, its tool_calls and an observation holding the results.
            ONE generic tool (`exec`), whose argument is JavaScript wrapping
            the harness's own calls, and files written through a patch
            envelope rather than a write tool

Five decisions, each mirroring a rule the web converter already established:

  1. AN ARTIFACT GATE, NOT A SUCCESS FLAG. The web rule is "keep it if
     dist/index.html exists". Godot has no dist/; the equivalent artifact is a
     loadable project -- `project.godot` AND a `.tscn` AND a `.gd` under the
     trial's workspace. The recorded result carries no verdict (null on 96 of
     96 trials sampled), so keying on it would discard everything.

  2. ONE TRIAL PER TASK. These are timestamped retries: 1,202 recordings
     across 1,004 tasks. The web rule ("the last session that made tool
     calls") does not apply, because this recorder stores one session per
     file. Keep the trial with the most tool calls that also passes the gate.

  3. THE BRIEF IS THE USER TURN. This recorder emits an `<environment_context>`
     boilerplate user step first and the real brief second, so take the last
     user step that is not boilerplate.

  4. THE RESULT TRIMS MATCH THE WEB CORPUS EXACTLY -- shell head 1,200 + tail
     800, everything else 2,000 characters. Two corpora that pool into one
     training file must have been shrunk by the same rule, or the model sees
     two different conventions for what an observation looks like.

  5. `--normalize` REWRITES THE ACTION SPACE. `exec({input: "<JS calling
     exec_command>"})` becomes `run_shell_command({command})`, so the row
     shares the web corpus' vocabulary. Without it the row keeps the single
     generic `exec`. This is the A-vs-B switch: normalized rows can be pooled
     with the web corpus, raw rows should train a separate arm.

A NOTE ON ONE FIX MADE DURING THE PORT. When a step's second call had no
result, the old code popped a single message and broke out of the loop --
which removed the TOOL message emitted for the first call and left the
assistant turn asserting two tool_calls that nothing ever answered. Every
message emitted for the step is now removed together, which is what dropping
an unanswered call was always supposed to mean.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SHELL_HEAD, SHELL_TAIL, OTHER_CAP = 1200, 800, 2000
MIN_BYTES = 10_240
ENV_PREFIX = '<environment_context>'

# const r = await tools.exec_command({cmd:"...", workdir:"...", ...}); text(r.output);
EXEC_CMD = re.compile(r"""exec_command\(\s*\{.*?\bcmd\s*:\s*(?P<q>["'`])(?P<cmd>.*?)(?<!\\)(?P=q)""", re.S)

# This harness writes files through a patch envelope rather than a write tool:
#   const patch = "*** Begin Patch\n*** Add File: game/main.gd\n+line\n+line..."
# `Add File` carries the whole body on '+' lines, which is exactly the payload
# the web corpus' write_file(file_path, content) carries.
PATCH_FILE = re.compile(r'\*\*\*\s+(?:Add|Update)\s+File:\s*(?P<path>[^\\\n"]+)')


def js_unescape(s: str) -> str:
    """Decode the escapes this recorder actually emits, and nothing else.

    `bytes.decode('unicode_escape')` is the obvious call and is wrong here: it
    round-trips through latin-1, so every non-ASCII character in a game's
    source (accented names, box drawing, emoji in comments) comes out mangled,
    and a stray sequence like \\` raises a DeprecationWarning. An explicit
    table leaves UTF-8 untouched and keeps an unknown escape verbatim rather
    than silently eating the backslash.
    """
    simple = {'n': '\n', 't': '\t', 'r': '\r', '"': '"', "'": "'",
              '\\': '\\', '`': '`', '/': '/', '$': '$', 'b': '\b', 'f': '\f'}
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c != '\\' or i + 1 >= n:
            out.append(c)
            i += 1
            continue
        nxt = s[i + 1]
        if nxt in simple:
            out.append(simple[nxt])
            i += 2
        elif nxt == 'u' and i + 6 <= n:
            try:
                out.append(chr(int(s[i + 2:i + 6], 16)))
                i += 6
            except ValueError:
                out.append(c)
                i += 1
        else:
            out.append(c)      # unknown escape: keep it, do not guess
            i += 1
    return ''.join(out)


def parse_apply_patch(payload: str) -> tuple[str, str] | None:
    """-> (path, body) for a single-file Add/Update patch, else None.

    Only single-file patches are converted. A multi-file patch has no faithful
    write_file equivalent, and inventing one would put a call in the corpus
    that the serving harness never emits.
    """
    if 'Begin Patch' not in payload:
        return None
    decoded = js_unescape(payload)
    paths = PATCH_FILE.findall(decoded)
    if len(paths) != 1:
        return None
    path = paths[0].strip()
    body_lines = []
    started = False
    for line in decoded.splitlines():
        if PATCH_FILE.search(line):
            started = True
            continue
        if not started:
            continue
        if line.startswith('*** End Patch'):
            break
        if line.startswith('+'):
            body_lines.append(line[1:])
    if not body_lines:
        return None
    return path, '\n'.join(body_lines)


def clean_observation(raw) -> str:
    """Result content is stored as a *stringified Python list* of
    {'type': 'input_text', 'text': ...} dicts. Recover the text; if it does not
    parse, hand back the raw string rather than dropping the turn."""
    if not isinstance(raw, str):
        return str(raw)
    s = raw.strip()
    if not s.startswith('['):
        return s
    try:
        parts = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s
    out = []
    for p in parts:
        if isinstance(p, dict) and 'text' in p:
            out.append(str(p['text']))
        else:
            out.append(str(p))
    return '\n'.join(out)


def trim_result(text: str, shell: bool) -> str:
    if shell:
        if len(text) <= SHELL_HEAD + SHELL_TAIL:
            return text
        elided = len(text) - SHELL_HEAD - SHELL_TAIL
        return f'{text[:SHELL_HEAD]}\n... [{elided} characters elided] ...\n{text[-SHELL_TAIL:]}'
    if len(text) <= OTHER_CAP:
        return text
    return text[:OTHER_CAP] + f'\n... [{len(text) - OTHER_CAP} characters elided] ...'


# Tools with no web counterpart. Measured at scale these are ~4% of calls.
# They are dropped (call and result together) rather than mapped onto a web
# tool that behaves differently -- teaching read_file to open a PNG, or
# inventing a stdin tool the harness lacks, is worse than a short trajectory.
UNMAPPABLE = {'view_image', 'write_stdin', 'wait'}

# The 14 tools the web corpus actually uses, counted over all 518 rows -- not
# over a sample. An earlier version of this list was built from 40 rows and
# silently omitted web_fetch and read_many_files. Under --normalize, a call
# that does not land in this set is dropped with its result: measured on the
# pilot, 7% of calls are foreign but they are spread across 74% of rows, so
# dropping the CALL keeps 208 rows where dropping the ROW would keep 54.
WEB_TOOLS = {
    'run_shell_command', 'read_file', 'write_file', 'grep_search', 'edit',
    'generate_tilemap', 'generate_gdd', 'generate_game_assets',
    'classify_game_type', 'todo_write', 'list_directory', 'glob',
    'web_fetch', 'read_many_files',
}

# This harness can batch N operations behind ONE call that returns ONE result:
# a multi-file patch, a Promise.all over several tools, an array of commands.
# The web corpus pairs every call with its own result, so a batch cannot be
# split faithfully -- attaching the single outcome ('Script failed') to each of
# four file writes would assert something that did not happen. Batches are
# dropped with their result, and the assistant's reasoning text is kept.
BATCH_HINTS = ('Promise.all', 'const cmds', 'for (const')


def is_batch(payload) -> bool:
    if not isinstance(payload, str):
        return False
    if payload.count('*** Add File') + payload.count('*** Update File') > 1:
        return True
    return any(h in payload for h in BATCH_HINTS)


def normalize_call(name: str, args: dict) -> tuple[str, dict]:
    """One recorded call -> its web-corpus equivalent.

    Measured over 145 trajectories, this recorder emits seven distinct tools.
    Three map cleanly onto the web vocabulary, one (`exec`) is a JavaScript
    wrapper around the other two, and three have no counterpart:

        exec_command {cmd,...}        -> run_shell_command {command}
        apply_patch  {input: patch}   -> write_file {file_path, content}
        update_plan  {plan:[{step}]}  -> todo_write {todos}
        exec         {input: JS}      -> whichever of the first two it wraps
        view_image / write_stdin / wait  -> no equivalent, caller drops them

    Anything unrecognised is returned unchanged so the validator can catch it;
    a silent wrong rewrite is worse than a visible foreign tool name.
    """
    if name == 'exec_command':
        cmd = args.get('cmd')
        if isinstance(cmd, str):
            return 'run_shell_command', {'command': cmd}
        return name, args

    if name == 'apply_patch':
        payload = args.get('input')
        if isinstance(payload, str):
            patch = parse_apply_patch(payload)
            if patch:
                path, body = patch
                return 'write_file', {'file_path': path, 'content': body}
        return name, args

    if name == 'update_plan':
        plan = args.get('plan')
        if isinstance(plan, list):
            todos = [
                {'id': f'step{i + 1}',
                 'content': (p.get('step') if isinstance(p, dict) else str(p)),
                 'status': (p.get('status') if isinstance(p, dict) else 'pending')}
                for i, p in enumerate(plan)
            ]
            return 'todo_write', {'todos': todos}
        return name, args

    if name != 'exec':
        return name, args
    payload = args.get('input')
    if not isinstance(payload, str):
        return name, args
    m = EXEC_CMD.search(payload)
    if m:
        return 'run_shell_command', {'command': js_unescape(m.group('cmd'))}
    patch = parse_apply_patch(payload)
    if patch:
        path, body = patch
        return 'write_file', {'file_path': path, 'content': body}
    return name, args


def find_brief(steps: list[dict]) -> str | None:
    briefs = [
        (s.get('message') or '').strip()
        for s in steps
        if s.get('source') == 'user' and not (s.get('message') or '').lstrip().startswith(ENV_PREFIX)
    ]
    briefs = [b for b in briefs if b]
    return briefs[-1] if briefs else None


def workspace_of(trial: Path) -> Path:
    return trial / 'sandbox' / 'workspace' / 'game'


def has_artifact(trial: Path) -> bool:
    g = workspace_of(trial)
    if not (g / 'project.godot').is_file():
        return False
    return any(g.rglob('*.tscn')) and any(g.rglob('*.gd'))


def convert(traj: Path, normalize: bool) -> tuple[dict | None, str]:
    try:
        doc = json.loads(traj.read_text(errors='replace'))
    except (OSError, ValueError):
        return None, 'unreadable'
    steps = doc.get('steps') or []
    if not steps:
        return None, 'no steps'

    brief = find_brief(steps)
    if not brief:
        return None, 'no brief'

    messages: list[dict] = [{'role': 'user', 'content': brief}]
    n_calls = 0
    n_unanswered = 0
    n_unmappable = 0

    for s in steps:
        if s.get('source') != 'agent':
            continue
        calls = s.get('tool_calls') or []
        text = (s.get('message') or '').strip()
        results = {r.get('source_call_id'): r
                   for r in ((s.get('observation') or {}).get('results') or [])}

        if not calls:
            if text:
                messages.append({'role': 'assistant', 'content': text})
            continue

        tc = []
        kept_calls = []
        for c in calls:
            name = c.get('function_name') or 'exec'
            args = c.get('arguments')
            if not isinstance(args, dict):
                args = {'input': args} if args is not None else {}
            if normalize:
                payload = args.get('input')
                if name in UNMAPPABLE or is_batch(payload):
                    n_unmappable += 1
                    continue          # drop call and its result together
                name, args = normalize_call(name, args)
                if name not in WEB_TOOLS:
                    n_unmappable += 1
                    continue          # did not normalize; drop call + result
            tc.append({'id': c.get('tool_call_id'), 'type': 'function',
                       'function': {'name': name,
                                    'arguments': json.dumps(args, ensure_ascii=False)}})
            kept_calls.append(c)
        if not tc:
            # every call in this step was unmappable; keep any reasoning text
            if text:
                messages.append({'role': 'assistant', 'content': text})
            continue

        # Everything this step appends starts here, so an unanswered call can
        # remove the step whole rather than one message of it.
        mark = len(messages)
        msg: dict = {'role': 'assistant', 'tool_calls': tc}
        if text:
            msg['content'] = text
        messages.append(msg)

        answered = True
        for c, t in zip(kept_calls, tc):
            cid = c.get('tool_call_id')
            r = results.get(cid)
            if r is None:
                # an unanswered call teaches a call answered by silence -- the
                # web converter drops these, so drop the whole step here too
                answered = False
                n_unanswered += 1
                break
            body = clean_observation(r.get('content'))
            shell = t['function']['name'] in ('run_shell_command', 'exec')
            messages.append({'role': 'tool', 'tool_call_id': cid,
                             'content': trim_result(body, shell)})
        if answered:
            n_calls += len(tc)
        else:
            del messages[mark:]

    if n_calls == 0:
        return None, 'no tool calls'
    if not any(m['role'] == 'assistant' for m in messages):
        return None, 'no assistant turn'

    agent = doc.get('agent') or {}
    trial = traj.parent.parent
    return {
        'task': trial.name.split('__')[0],
        'trial': trial.name,
        'engine': 'godot',
        'model': agent.get('model_name'),
        'agent': agent.get('name'),
        'schema_version': doc.get('schema_version'),
        'n_tool_calls': n_calls,
        'n_unanswered_dropped': n_unanswered,
        'n_unmappable_dropped': n_unmappable,
        'normalized': normalize,
        'messages': messages,
    }, 'ok'


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog='rsigame.data_pipeline godot', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--roots', nargs='+', required=True,
                    help='directories containing extracted trial directories')
    ap.add_argument('--out', required=True)
    ap.add_argument('--normalize', action='store_true',
                    help='rewrite exec(JS) -> run_shell_command, so rows pool with the web corpus')
    ap.add_argument('--require-artifact', action='store_true', default=True)
    ap.add_argument('--no-require-artifact', dest='require_artifact', action='store_false')
    ap.add_argument('--min-bytes', type=int, default=MIN_BYTES)
    a = ap.parse_args(argv)

    trajs: list[Path] = []
    for root in a.roots:
        trajs.extend(Path(root).rglob('agent/trajectory.json'))

    skipped: Counter = Counter()
    by_task: dict[str, list[dict]] = defaultdict(list)

    for t in sorted(trajs):
        trial = t.parent.parent
        if t.stat().st_size < a.min_bytes:
            skipped['recording under min-bytes'] += 1
            continue
        if a.require_artifact and not has_artifact(trial):
            skipped['no project.godot + .tscn + .gd'] += 1
            continue
        row, why = convert(t, a.normalize)
        if row is None:
            skipped[why] += 1
            continue
        by_task[row['task']].append(row)

    # one trial per task: the one that got furthest (most tool calls)
    kept = []
    dupes = 0
    for task, rows in by_task.items():
        rows.sort(key=lambda r: r['n_tool_calls'], reverse=True)
        kept.append(rows[0])
        dupes += len(rows) - 1

    kept.sort(key=lambda r: r['task'])
    from . import jsonl
    jsonl.write(a.out, kept)

    print(f'trajectories found : {len(trajs)}')
    print(f'rows written       : {len(kept)}  -> {a.out}')
    print(f'extra trials merged: {dupes} (same task, lower tool-call count)')
    if kept:
        calls = sorted(r['n_tool_calls'] for r in kept)
        print(f'tool calls per row : min {calls[0]}  median {calls[len(calls) // 2]}  max {calls[-1]}')
        print(f'normalized         : {a.normalize}')
    if skipped:
        print('skipped:')
        for k, v in skipped.most_common():
            print(f'  {v:5d}  {k}')
    return 0 if kept else 1


if __name__ == '__main__':
    raise SystemExit(main())
