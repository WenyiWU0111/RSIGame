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
"""Check a model director's transcript against the prompt's rules.

    audit_director.py TRANSCRIPT.jsonl SESSION_DIR [--brief-dir DIR]

Allowed: shell commands that run director_cli.py, `mkdir` of the brief
directory, and writing a file inside it; image reads of frames under
SESSION_DIR/screenshots; writing the brief file. Anything else is a violation,
and a review with a violation is discarded (director_prompt.md, Rules).
Writes SESSION_DIR/audit.json and prints the verdict and the tool-call counts.
"""
import os
import sys
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from .. import paths



def tool_calls(transcript: Path):
    for line in transcript.read_text(errors='replace').splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        msg = d.get('message') or {}
        if msg.get('role') != 'assistant':
            continue
        for c in msg.get('content') or []:
            if isinstance(c, dict) and c.get('type') == 'tool_use':
                yield c.get('name'), c.get('input') or {}


def usage(transcript: Path) -> dict:
    tot = Counter()
    seen = set()
    for line in transcript.read_text(errors='replace').splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        msg = d.get('message') or {}
        u = msg.get('usage')
        if msg.get('role') == 'assistant' and u and msg.get('id') not in seen:
            seen.add(msg.get('id'))
            for k in ('input_tokens', 'output_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens'):
                tot[k] += int(u.get(k) or 0)
    return dict(tot)


ALLOWED_WORDS = {'export', 'cd', 'for', 'do', 'done', 'in', 'grep', 'head', 'tail', 'echo', 'printf',
                 'mkdir', 'cat', 'sleep', 'true', 'wc', 'jq', 'EOF', 'while', 'then',
                 'fi', 'if', 'seq', 'sort', 'tr', 'cut'}
PY_BIN = os.environ.get('RSIGAME_PATHS_PYTHON') or sys.executable   # subprocesses use this same interpreter
REVIEW_DIR = str(paths.agent_repo() / 'review')
FILTERS = {'grep', 'head', 'tail', 'wc', 'sort', 'tr', 'cut', 'jq'}
FILE_RE = re.compile(r'[\w./-]+\.(py|jsonl?|md|log|csv|txt|gd|tscn|godot|png|jpg|mp4|sh|cfg)\b')


def shell_ok(cmd: str, brief_dir: str, shots: str) -> tuple[bool, str]:
    """(ok, why). Every path must be the CLI, its python, the brief directory,
    this session's frames, or /tmp as a cd target; every command word must be
    the CLI (directly or through a variable) or plain shell plumbing; the only
    file names allowed are director_cli.py and a brief file in the brief dir."""
    body = re.sub(r"<<\s*'?(\w+)'?\n.*?\n\1\b", "<<HEREDOC", cmd, flags=re.S)
    body = re.sub(r'https?://[^\s;&|]+', 'URL', body)
    body = re.sub(r"\b(grep|jq|tr|cut)((?:\s+-[\w-]+)*)\s+('[^']*'|\"[^\"]*\")", r'\1\2 PATTERN', body)  # filters on CLI output
    body = re.sub(r'--note\s+("[^"]*"|\'[^\']*\')', '', body)     # the model's reasons are prose
    for m in re.finditer(r'(?<![\w:])/[\w./-]+', body):
        path = m.group(0).rstrip('.')
        if not (path in (PY_BIN, '/tmp', '/dev/null') or path.startswith((REVIEW_DIR + '/director_cli.py', brief_dir, shots))
                or path == REVIEW_DIR or path.startswith('/api/')):
            return False, f'path {path}'
    for m in FILE_RE.finditer(body):
        f = m.group(0)
        if f.endswith('director_cli.py') or (f.startswith(brief_dir) and f.endswith('.json')) \
                or f.startswith(shots) or f.startswith('brief'):
            continue
        return False, f'file {f}'
    env = {}
    for part in re.split(r'&&|;|\|\||\||\n|\$\(|\)|`', body):
        words = part.strip().split()
        while words and re.fullmatch(r'\w+=\S*', words[0]):
            k, v = words[0].split('=', 1)
            env[k] = v.strip('"\'')
            words = words[1:]
        if not words:
            continue
        w = words[0]
        for k, v in env.items():
            w = w.replace('${%s}' % k, v).replace('$' + k, v)
        w = w.strip('"\'')
        head = w.split()[0] if w.split() else w
        if head in FILTERS:
            # only as a filter on piped CLI output: no file or directory arguments
            rest = [x for x in words[1:] if not x.startswith('-') and not x.isdigit() and x != 'PATTERN']
            if head == 'grep' and rest and not any(x == 'PATTERN' for x in words[1:]):
                rest = rest[1:]                       # an unquoted pattern
            if rest or any(x in ('-r', '-R', '--recursive') for x in words[1:]):
                return False, f'{head} with arguments {rest[:3]}'
            continue
        if head == 'cat' and not re.search(r'>\s*' + re.escape(brief_dir), part):
            return False, 'cat not writing the brief'
        if head in (PY_BIN,) or head.endswith('director_cli.py') or head in ALLOWED_WORDS or head.startswith(('{', '}', '"', '[', ']')):
            continue
        return False, f'command {head}'
    return True, ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('transcript')
    ap.add_argument('session_dir')
    ap.add_argument('--brief-dir', default=None, help='where the briefs are (default: <runs>/director)')
    a = ap.parse_args()
    shots = str(Path(a.session_dir) / 'screenshots')
    counts, violations = Counter(), []
    for name, inp in tool_calls(Path(a.transcript)):
        counts[name] += 1
        if name == 'Bash':
            cmd = inp.get('command', '')
            okk, why = shell_ok(cmd, a.brief_dir, shots)
            if not okk:
                violations.append({'tool': name, 'why': why, 'command': cmd[:300]})
        elif name == 'Read':
            fp = inp.get('file_path', '')
            if not fp.startswith(shots):
                violations.append({'tool': name, 'file_path': fp})
        elif name == 'Write':
            if not inp.get('file_path', '').startswith(a.brief_dir):
                violations.append({'tool': name, 'file_path': inp.get('file_path')})
        else:
            violations.append({'tool': name, 'input': json.dumps(inp)[:300]})
    out = {'transcript': a.transcript, 'tool_calls': dict(counts), 'violations': violations,
           'clean': not violations, 'usage': usage(Path(a.transcript))}
    (Path(a.session_dir) / 'audit.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ('tool_calls', 'clean', 'usage')}, indent=1))
    for v in violations:
        print('VIOLATION', json.dumps(v)[:300])


if __name__ == '__main__':
    main()
