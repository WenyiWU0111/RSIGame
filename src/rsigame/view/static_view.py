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
"""WHAT THE BUILD CONTAINS -- a fact sheet the critic cannot get from playing.

Phase 2 of the verifier work. The diagnosis it answers is that the critique's
evidence is strong for two quality directions and structurally absent for the
other two:

    Mechanism        temporal   action -> response -> frame        HAVE IT
    Visual           temporal   HUD text, transitions, frames      HAVE IT
    Depth            enumerative  how many levels/enemies/endings  MISSING
    Art              comparative  asked-for look vs what is drawn  HALF

A fourteen-step trace can show what was reached. It cannot establish the total
content space, and twelve screenshots cannot establish whether the runtime is
still dominated by primitives. Both of those are project-level facts, cheap to
read off the tree, and the critic has never been shown either.

Everything here is mechanical: no model call, no inference, no judgement. The
block reports WHAT EXISTS. It never says what should exist, and it never
mentions the evaluator, its requirements, or its weights -- the rubric stays
held out. `_LEAK` below is the check that keeps that true, and the test asserts
on it rather than trusting the prose.

Where a structure cannot be anchored to something explicit in the source, the
category is omitted rather than guessed. That rule is not fastidiousness: the
scenario regex in `project.declared_scenarios` was wrong twice -- once
missing a whole game to tab indentation, once sweeping `armor`/`weapon` out of
an equipment dictionary and reporting eight uncovered scenarios where there was
one. A count that is quietly wrong is worse than a category that is absent,
because the critic has no way to tell.
"""

from __future__ import annotations

import re
from pathlib import Path

# Directories that are not the game.
#
# `.godot` is Godot's generated import cache. The other two are written INTO
# the work tree by our own harness and are the reason this list is not a
# formality: `_repair_evidence` holds the repair agent's screenshots and
# `demo_outputs` holds the bench recordings, and counting them made
# visualnovel-lastsignal report "186 images present, 3 named by code, 183 never
# named" for a game that ships three images. The critic would have read that as
# a project drowning in unused art. Preflight caught it; nothing else would
# have, because the number is plausible until you look at the filenames.
_SKIP_DIRS = {'.godot', '.git', 'node_modules', '.import', 'addons',
              '_repair_evidence', 'demo_outputs'}

_SRC_SUFFIXES = ('.gd', '.tscn', '.tres', '.cfg', '.godot')
_IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.webp', '.svg', '.bmp')
_AUDIO_SUFFIXES = ('.ogg', '.wav', '.mp3')

# Primitive drawing and default-widget construction. These are the sites that
# put a shape on screen without an authored asset behind it. Deliberately does
# NOT include draw_string: text is not a placeholder for art, it is text.
_PROCEDURAL = re.compile(
    r'\bdraw_rect\b|\bdraw_circle\b|\bdraw_line\b|\bdraw_polygon\b|'
    r'\bdraw_colored_polygon\b|\bdraw_arc\b|\bdraw_multiline\b|'
    r'\bColorRect\b|\bStyleBoxFlat\b|\bPlaceholderTexture2D\b')

# `enum Name { A, B, C }` -- the one content registry in these projects that is
# explicit enough to count. Most content (callers, sorties, endings) lives as
# literals inside function bodies, which is exactly the ambiguous case the
# module refuses to guess at.
_ENUM = re.compile(r'^[ \t]*enum[ \t]+([A-Za-z_][A-Za-z0-9_]*)[ \t]*\{([^}]*)\}',
                   re.M)

# Anything that would tell the loop how it is graded. Facts about the artifact
# are allowed; rules about how the benchmark rewards those facts are not.
_LEAK = re.compile(
    r'\brubric\b|\bscore[sd]?\b|\bscoring\b|\bpoints?\b|\bweight(s|ed|ing)?\b|'
    r'\bgrade[sd]?\b|\bgrading\b|\brequirement\b|\bcriteri|\bevaluat|'
    r'\bbenchmark\b|\bjudge\b|\breward\b|\bshould\b|\bought\b|\bmust\b|'
    r'\binsufficient\b|\btoo few\b|\bneeds? more\b|\bbetter\b|\bworse\b|'
    r'\bpenal|\bcapped?\b|\bat most\b|\bM[1-9]\b|\bD[1-9]\b|\bV[1-9]\b|'
    r'\bA[1-9]\b', re.I)


def _walk(work: Path, suffixes: tuple) -> list:
    """Files under `work` with one of these suffixes, skipping generated trees.

    Scoped to `work` and nothing above it. A scan that can escape the game
    directory is a scan that can walk the filesystem, and on this host that
    ends in the OOM killer taking someone else's training with it.
    """
    out = []
    for p in work.rglob('*'):
        if not p.is_file():
            continue
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() in suffixes:
            out.append(p)
    return out


def _source_text(work: Path) -> str:
    """Every source and scene file concatenated, for reference lookups."""
    buf = []
    for p in _walk(work, _SRC_SUFFIXES):
        try:
            buf.append(p.read_text(errors='replace'))
        except OSError:
            pass
    return '\n'.join(buf)


# See `polish_evidence.BYPRODUCTS`: the asset tool writes a contact sheet and a
# style reference beside what it made, and no game loads either.
_BYPRODUCTS = ('_contact_sheet.png', 'style_anchor.png')


def _assets(work: Path, src: str) -> dict:
    """Which asset files exist and which the code actually names.

    An asset counts as referenced if either its `res://` path or its bare
    filename appears anywhere in the source. That is deliberately generous:
    games build paths at runtime (`"res://assets/planes/plane%s%d.png" % ...`),
    and calling a used asset unused is the error that would mislead the critic.
    Under-reporting "unreferenced" is the safe direction.
    """
    def split(files):
        used, unused = [], []
        for p in files:
            rel = p.relative_to(work).as_posix()
            if rel in src or p.name in src or p.stem in src:
                used.append(rel)
            else:
                unused.append(rel)
        return used, unused

    imgs = [p for p in _walk(work, _IMAGE_SUFFIXES)
            if p.name not in _BYPRODUCTS]
    auds = _walk(work, _AUDIO_SUFFIXES)
    iu, ix = split(imgs)
    au, ax = split(auds)
    return {'img_n': len(imgs), 'img_used': len(iu), 'img_unused': ix,
            'aud_n': len(auds), 'aud_used': len(au), 'aud_unused': ax}


def _enums(work: Path) -> list:
    """Named enum registries, which are the one countable content structure."""
    out = []
    for p in _walk(work, ('.gd',)):
        try:
            text = p.read_text(errors='replace')
        except OSError:
            continue
        for name, body in _ENUM.findall(text):
            members = [m.strip().split('=')[0].strip()
                       for m in body.split(',') if m.strip()]
            if members:
                out.append((name, members))
    return out


def _procedural_sites(work: Path) -> int:
    n = 0
    for p in _walk(work, ('.gd', '.tscn')):
        try:
            n += len(_PROCEDURAL.findall(p.read_text(errors='replace')))
        except OSError:
            pass
    return n


def _names(xs, cap: int = 12) -> str:
    xs = sorted(xs)
    if len(xs) <= cap:
        return ', '.join(xs)
    return ', '.join(xs[:cap]) + f', and {len(xs) - cap} more'


# ---------------------------------------------------------------- Phaser (web)
#
# Every reader above looks for GDScript: `.gd`/`.tscn` sources, `draw_rect`,
# `enum Name {`. On a Phaser tree they read "24 images, 0 named, 24 never named"
# (the copies under `dist/` counted too) and no scenarios -- handed to the ranker
# and the checklist reader as fact. A tree without `project.godot` is read the
# way it is built instead; a Godot tree never reaches any of this.
_WEB_SKIP_DIRS = _SKIP_DIRS | {'dist'}
_WEB_SRC_SUFFIXES = ('.ts', '.tsx', '.js', '.mjs')
_WEB_PROCEDURAL = re.compile(
    r'\.add\.(?:rectangle|circle|ellipse|triangle|polygon|star|arc|line|grid)\(|'
    r'\.(?:fill|stroke)(?:Rect|RoundedRect|Circle|Ellipse|Triangle|Points|Path)\(')
_WEB_ENUM = re.compile(
    r'^[ \t]*(?:export[ \t]+)?(?:const[ \t]+)?enum[ \t]+([A-Za-z_][A-Za-z0-9_]*)'
    r'[ \t]*\{([^}]*)\}', re.M)


def is_web(work) -> bool:
    return not (Path(work) / 'project.godot').is_file()


def _web_walk(work: Path, suffixes: tuple) -> list:
    out = []
    for p in work.rglob('*'):
        if not p.is_file():
            continue
        if any(part in _WEB_SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() in suffixes:
            out.append(p)
    return out


def web_declared_scenarios(work) -> list:
    """The scenario ids a Phaser game's own source compares against.

    Only the two unambiguous shapes: `scenario === 'id'` (or `!==`, or `==`) on
    a variable whose name says scenario, and `case 'id':` inside a
    `switch (...scenario...)`. Route tables and lookups built at runtime are
    left out -- the same rule as the Godot reader: a category that is absent is
    better than a count that is quietly wrong.
    """
    work = Path(work)
    ids = set()
    for p in _web_walk(work, _WEB_SRC_SUFFIXES):
        try:
            t = p.read_text(errors='replace')
        except OSError:
            continue
        ids |= set(re.findall(
            r'\b\w*[sS]cenario\w*\s*[!=]==?\s*[\'"]([a-z0-9_]{2,40})[\'"]', t))
        ids |= set(re.findall(
            r'[\'"]([a-z0-9_]{2,40})[\'"]\s*[!=]==?\s*\w*[sS]cenario\w*\b', t))
        for m in re.finditer(r'switch\s*\(\s*[^)]*[sS]cenario[^)]*\)\s*\{', t):
            body = t[m.end():m.end() + 4000]
            depth, end = 1, len(body)
            for i, ch in enumerate(body):
                depth += (ch == '{') - (ch == '}')
                if depth == 0:
                    end = i
                    break
            ids |= set(re.findall(r'\bcase\s+[\'"]([a-z0-9_]{2,40})[\'"]\s*:',
                                  body[:end]))
    return sorted(ids)


def _web_assets(work: Path) -> dict:
    import rsigame.polish.polish_evidence as PE
    src = PE._web_sources(work)
    keys = PE._web_pack_keys(work)

    def split(files):
        used, unused = [], []
        for p in files:
            rel = p.relative_to(work).as_posix()
            (used if PE._web_is_loaded(rel, src, keys) else unused).append(rel)
        return used, unused

    imgs = [p for p in _web_walk(work, _IMAGE_SUFFIXES)
            if p.name not in _BYPRODUCTS]
    auds = _web_walk(work, _AUDIO_SUFFIXES)
    iu, ix = split(imgs)
    au, ax = split(auds)
    return {'img_n': len(imgs), 'img_used': len(iu), 'img_unused': ix,
            'aud_n': len(auds), 'aud_used': len(au), 'aud_unused': ax}


def _web_facts(work: Path) -> tuple:
    enums, proc = [], 0
    for p in _web_walk(work, _WEB_SRC_SUFFIXES):
        try:
            text = p.read_text(errors='replace')
        except OSError:
            continue
        proc += len(_WEB_PROCEDURAL.findall(text))
        for name, body in _WEB_ENUM.findall(text):
            members = [m.strip().split('=')[0].strip()
                       for m in body.split(',') if m.strip()]
            if members:
                enums.append((name, members))
    return enums, proc


def build_static_view(work, *, declared=None, visited_now=None,
                      visited_history=None) -> str:
    """The fact sheet, or '' when the tree yields nothing worth sending.

    `declared` / `visited_now` / `visited_history` come from the caller, which
    already computes them for the play side -- `declared_scenarios(work)`, the
    trace's own `boot_scenario` calls, and the union across earlier rounds.
    Passing them in rather than recomputing keeps one definition of each and
    means this module never has to import the arm.

    Returns a block whose every line is a count or a name. Nothing here is a
    verdict, and `_LEAK` is asserted against the output in the test.
    """
    work = Path(work)
    if not work.is_dir():
        return ''

    web = is_web(work)
    if web:
        a = _web_assets(work)
        enums, proc = _web_facts(work)
    else:
        src = _source_text(work)
        a = _assets(work, src)
        enums = _enums(work)
        proc = _procedural_sites(work)

    lines = ['WHAT THE BUILD CONTAINS', '',
             'Mechanically read from the current tree. Counts only -- what is '
             'there, not what it is worth.']

    declared = sorted(declared or [])
    if declared:
        now = sorted(visited_now or [])
        hist = sorted(set(visited_history or []) | set(now))
        never = [s for s in declared if s not in hist]
        lines += ['', 'Scenarios this game defines for itself',
                  f'  - defined: {len(declared)} ({_names(declared)})',
                  f'  - entered this session: {len(now)}'
                  + (f' ({_names(now)})' if now else ''),
                  f'  - entered at least once in this run: {len(hist)} '
                  f'of {len(declared)}']
        if never:
            lines.append(f'  - never entered in this run: {_names(never)}')

    if enums:
        lines += ['', 'Named registries in the source']
        for name, members in enums[:6]:
            lines.append(f'  - {name}: {len(members)} '
                         f'({_names(members, cap=10)})')

    lines += ['', 'Asset files present, and whether the source names them',
              f'  - images: {a["img_n"]} present, {a["img_used"]} named by '
              f'code or scenes, {len(a["img_unused"])} never named']
    if a['img_unused']:
        lines.append(f'      never named: {_names(a["img_unused"], cap=8)}')
    lines.append(f'  - audio: {a["aud_n"]} present, {a["aud_used"]} named by '
                 f'code or scenes, {len(a["aud_unused"])} never named')

    if web:
        lines += ['', 'How the screen is drawn',
                  f'  - calls that draw a primitive shape (add.rectangle, '
                  f'add.circle, graphics fillRect / fillCircle / '
                  f'fillRoundedRect, strokeRect): {proc}']
        return '\n'.join(lines)
    lines += ['', 'How the screen is drawn',
              f'  - calls that draw a primitive shape or a flat default widget '
              f'(draw_rect, draw_circle, draw_line, draw_polygon, ColorRect, '
              f'StyleBoxFlat): {proc}']

    return '\n'.join(lines)


def leak_hits(text: str) -> list:
    """Words in the block that would tell the loop how it is graded.

    The test asserts this is empty. Kept as a function rather than inlined so
    the same check can run over a generated block before an expensive arm
    starts, not only in the unit test.
    """
    return sorted(set(m.group(0) for m in _LEAK.finditer(text or '')))
