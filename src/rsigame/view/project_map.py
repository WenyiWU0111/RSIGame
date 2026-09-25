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
"""Where things are in this game, so the repair agent stops re-finding them.

PART 6. Measured over three arms' stored repair logs, roughly half of every
round is spent re-learning a repository the agent read last round:

    A0-strict   browsing 56%   editing 14%
    E2          browsing 51%   editing 22%
    EV1         browsing 45%   editing 28%

These are one-file games -- shooter-sky-duel is a single 1244-line `Main.gd`
with 82 functions -- and each round begins by grepping for where `_draw()` and
the input handling live, again. The budget is 26 tool calls; the median round
spends 9 or 10 of them on orientation and 6 on edits.

POSITIONS, NEVER CONCLUSIONS. The map says where a thing is. It does not say
whether it works, whether it was fixed, or what is wrong with it. Same reason
`scenario_ledger` records facts and not verdicts: a map that claims "this was
repaired in round 3" gets believed instead of looked at, and round 7 of
shooter-sky-duel is exactly why -- that repair session ended with
`files_edited: []` and `success: False`, having changed nothing at all.

MECHANICAL SKELETON, MODEL ANNOTATION. The names and line numbers come from a
regex over the source: they cannot be wrong and cannot be invented. What each
part is FOR is the agent's to write, once, and it is carried forward.

LINE NUMBERS ARE RE-READ EVERY ROUND, ANNOTATIONS ARE KEYED ON THE NAME. Code
moves -- a repair that inserts forty lines shifts everything below it -- so a
stored line number is stale by the time it is read. Storing the name instead
and re-scanning costs nothing and cannot drift. A function that is renamed or
deleted loses its annotation, and the map SAYS SO rather than dropping it in
silence: an annotation that vanished is a fact about the last repair.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

SRC_SUFFIX = ('.gd', '.ts', '.js')
SKIP = ('.godot', '.git', 'node_modules', 'dist', '_repair_evidence',
        'demo_outputs', '.qwen')
_FUNC = re.compile(
    r'^\s*(?:export\s+)?(?:async\s+)?'
    r'(?:func|function)\s+([A-Za-z_$][\w$]*)\s*\(', re.M)
_METHOD = re.compile(r'^\s{2,}(?:async\s+)?([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*\{',
                     re.M)


def _sources(work: Path) -> list:
    out = []
    for p in sorted(Path(work).rglob('*')):
        if p.suffix.lower() not in SRC_SUFFIX or not p.is_file():
            continue
        if any(part in SKIP for part in p.parts):
            continue
        out.append(p)
    return out


def scan(work: Path) -> dict:
    """Names and line numbers, read off the source. No judgement anywhere."""
    work = Path(work)
    files = {}
    for p in _sources(work):
        try:
            src = p.read_text(errors='replace')
        except OSError:
            continue
        hits = [(m.start(), m.group(1)) for m in _FUNC.finditer(src)]
        if p.suffix.lower() in ('.ts', '.js'):
            hits += [(m.start(), m.group(1)) for m in _METHOD.finditer(src)]
        hits.sort()
        n_lines = src.count('\n') + 1
        fns = []
        for i, (pos, name) in enumerate(hits):
            start = src[:pos].count('\n') + 1
            end = (src[:hits[i + 1][0]].count('\n')) if i + 1 < len(hits) \
                else n_lines
            fns.append({'name': name, 'line': start, 'end': max(start, end)})
        if fns:
            files[str(p.relative_to(work))] = {'lines': n_lines,
                                               'functions': fns}
    return {'files': files,
            'n_functions': sum(len(f['functions']) for f in files.values())}


def merge(skeleton: dict, notes: dict) -> dict:
    """Attach stored annotations to a fresh scan, and say what went missing.

    `notes` maps "file::function" -> one line. Keying on the name is the whole
    point: the line number in the note is never used, because it is stale.
    """
    live = {f'{f}::{fn["name"]}'
            for f, d in skeleton['files'].items() for fn in d['functions']}
    kept = {k: v for k, v in (notes or {}).items() if k in live}
    lost = sorted(set(notes or {}) - live)
    return {**skeleton, 'notes': kept, 'lost_notes': lost,
            'n_notes': len(kept)}


def render(m: dict, cap_per_file: int = 60) -> str:
    """The map as the repair agent reads it. Annotated parts first."""
    if not m.get('files'):
        return ''
    out = ['WHERE THINGS ARE IN THIS GAME', '',
           'Read off the source at the start of this round, so the line '
           'numbers are current. It says where code lives and nothing about '
           'whether it works.', '']
    notes = m.get('notes') or {}
    for f, d in m['files'].items():
        out.append(f"  {f}   {d['lines']} lines, {len(d['functions'])} functions")
        annotated = [fn for fn in d['functions'] if f'{f}::{fn["name"]}' in notes]
        plain = [fn for fn in d['functions'] if f'{f}::{fn["name"]}' not in notes]
        for fn in annotated:
            out.append(f"    {fn['line']:>5}-{fn['end']:<5} {fn['name']}"
                       f"   {notes[f'{f}::{fn['name']}']}")
        if plain:
            room = max(0, cap_per_file - len(annotated))
            out.append('    -- not yet described --')
            line = []
            for fn in plain[:room]:
                line.append(f"{fn['name']}:{fn['line']}")
                if len(line) == 4:
                    out.append('    ' + '  '.join(line))
                    line = []
            if line:
                out.append('    ' + '  '.join(line))
            if len(plain) > room:
                out.append(f'    ... and {len(plain) - room} more')
        out.append('')
    if m.get('lost_notes'):
        out.append('  Gone since last round (renamed or deleted): '
                   + ', '.join(x.split("::")[-1] for x in m['lost_notes'][:8]))
        out.append('')
    out.append('  If you learn what an undescribed part is for, say so in your '
               'final answer as `project_map: {"<file>::<function>": "<one '
               'line>"}` and it will be here next round.')
    return '\n'.join(out)


def path_for(gdir) -> Path:
    """One file per game, beside the run's other persistent objects.

    NOT in the work tree: a file there would show up in the diff that decides
    whether a round changed anything, and would travel into `files_vs_g0`.
    `gdir` is `<out>/runs/<game>`, so two games can never share one.
    """
    return Path(gdir) / 'project_map.json'


def load_notes(gdir) -> dict:
    p = path_for(gdir)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text()).get('notes') or {}
    except Exception:
        return {}


def save(m: dict, gdir) -> Path:
    """Store the annotations and a copy of the scan, atomically.

    Written through a temporary file in the same directory and renamed: a run
    killed mid-write would otherwise leave a truncated map, and the next round
    would silently start from nothing. This host kills long runs in clusters.
    """
    p = path_for(gdir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(m, ensure_ascii=False, indent=1))
    tmp.replace(p)
    return p


def take_notes(m: dict, answer) -> tuple:
    """Fold the agent's `project_map` block into the map. Returns (map, n_new).

    Only keys that name a function this scan actually found are kept -- an
    invented one would put a description on a line that is something else.
    """
    got = answer if isinstance(answer, dict) else {}
    live = {f'{f}::{fn["name"]}'
            for f, d in m['files'].items() for fn in d['functions']}
    new = 0
    notes = dict(m.get('notes') or {})
    for k, v in got.items():
        k = str(k).strip()
        if k in live and isinstance(v, str) and v.strip():
            if k not in notes:
                new += 1
            notes[k] = v.strip()[:160]
    m['notes'] = notes
    m['n_notes'] = len(notes)
    return m, new
