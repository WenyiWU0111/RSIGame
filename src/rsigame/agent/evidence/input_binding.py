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
"""The `input_binding` dimension: declared control <-> actual binding <-> runtime.

Chosen as the first Programmatic dimension because its ground truth is the
easiest for a human to check by eye. If evidence alignment cannot be trusted
here, it certainly cannot be trusted on resource economy.

Three fact_roles, three sources, deliberately kept apart:

    declared_intent      the HUD line the game shows, GAME_DESIGN controls,
                         result.json instruction
    implementation       registration sites actually present in source, each
                         with its SCOPE
    runtime_observation  what the Explorer dispatched, and what was (or was not)
                         seen to acknowledge it

Two things this module refuses to do, both learned from real output:

* It does not answer "does this key have a handler somewhere". That question is
  nearly worthless: field_commander registers `pointerdown` on BaseObstacle and
  on its gameplay scene, and only the second is what place_defender runs
  through. Bindings therefore carry scope, and a candidate can ask whether the
  advertised control resolves to a handler whose scope matches the mechanic.
* It does not write `acknowledged = False`. The harness watches a limited set of
  handlers; "our probe saw nothing" is `not_observed`, and when a downstream
  core effect DID occur it is upgraded to `inferred` -- the state changed, so
  something processed the input, whatever our probe managed to see.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import re
from pathlib import Path

# Lifted out of program_bundle.py, which was 473 lines of an evidence-fact
# framework -- entities, provenance, statuses, invariant audits -- of which
# this dataclass was the only thing anything still read.
@dataclass
class Binding:
    """A registration site, as a first-class thing with a SCOPE.

    "control:pointer has a handler somewhere in the tree" is nearly worthless
    and actively misleading: field_commander registers `pointerdown` on
    BaseObstacle AND on its gameplay scene, and only the second one is what
    place_defender runs through. A property worth discovering is not "does this
    key have a handler" but "does the advertised control resolve to a handler
    whose scope and semantics match the advertised mechanic", and that question
    cannot even be asked unless scope is carried.
    """
    binding_id: str
    control: str                       # canonical entity id
    handler_kind: str                  # scene_keydown / scene_pointer / object_pointer / dom
    scope_kind: str                    # scene / object / global
    scope_name: str                    # the enclosing class, when resolvable
    registered_event: str              # the LITERAL event name the engine receives
    pattern: str                       # the source form it was written in
    ref: str = ""
    callback: str = ""
    mechanic_relation: str = "unresolved"
    resolution_note: str = ""

    def to_dict(self):
        return asdict(self)

    def render(self) -> str:
        L = [f'  {self.binding_id}  {self.control}',
             f'      event registered : "{self.registered_event}"',
             f'      handler kind     : {self.handler_kind}',
             f'      scope            : {self.scope_kind} {self.scope_name or "(unresolved)"}',
             f'      written as       : {self.pattern}',
             f'      callback         : {self.callback or "(not resolved)"}',
             f'      mechanic relation: {self.mechanic_relation}',
             f'      ref              : {self.ref}']
        if self.resolution_note:
            L.append(f'      note             : {self.resolution_note}')
        return '\n'.join(L)


_BIND_PATTERNS = [
    (re.compile(r"""(?P<recv>this\.input|input)\s*\.\s*keyboard\s*\??!?\.\s*on\(\s*['"]keydown-(?P<key>[A-Za-z0-9_]+)['"]"""),
     "input.keyboard.on('keydown-<KEY>')", 'scene_keydown'),
    (re.compile(r"""(?P<recv>this\.input|input)\s*\.\s*keyboard\s*\??!?\.\s*on\(\s*['"]keyup-(?P<key>[A-Za-z0-9_]+)['"]"""),
     "input.keyboard.on('keyup-<KEY>')", 'scene_keyup'),
    (re.compile(r"""addKey\(\s*(?:Phaser\.Input\.Keyboard\.)?KeyCodes\.(?P<key>[A-Z0-9_]+)"""),
     "addKey(KeyCodes.<KEY>)", 'scene_addkey'),
    (re.compile(r"""addKey\(\s*['"](?P<key>[A-Za-z0-9_]+)['"]"""),
     "addKey('<KEY>')", 'scene_addkey'),
]
_DYN_BIND = re.compile(
    r"""(?P<recv>this\.input|input)\s*\.\s*keyboard\s*\??!?\.\s*on\(\s*[`'"]keydown-\$\{(?P<expr>[^}]+)\}""")
_CURSORS = re.compile(r"createCursorKeys\(")
_SCENE_POINTER = re.compile(
    r"""this\.input\s*\.\s*on\(\s*['"](?P<ev>pointerdown|pointerup|pointermove)['"]""")
_OBJ_POINTER = re.compile(
    r"""(?P<recv>[A-Za-z_][\w.]*)\s*\.\s*on\(\s*['"](?P<ev>pointerdown|pointerup)['"]""")
# `export abstract class BaseTDScene extends Phaser.Scene` is the shape the
# real scenes use. Missing `abstract` left every scene-level binding's scope
# as "(unresolved)" -- and scope is precisely the field that makes a binding
# worth having, since "some pointer handler exists somewhere" is the useless
# version of the question.
_CLASS = re.compile(r'''^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+|declare\s+)*class\s+([A-Za-z_]\w*)''', re.M)
_CALLBACK = re.compile(r"""=>\s*(?:this\.)?([A-Za-z_]\w*)\s*\(""")

_DIGIT_WORDS = {'ONE': '1', 'TWO': '2', 'THREE': '3', 'FOUR': '4', 'FIVE': '5',
                'SIX': '6', 'SEVEN': '7', 'EIGHT': '8', 'NINE': '9', 'ZERO': '0'}


def canonical_control(token: str) -> tuple[str, str]:
    t = str(token or '').strip().upper().replace('DIGIT', '')
    if t in _DIGIT_WORDS:
        t = _DIGIT_WORDS[t]
    if t in ('SPACEBAR', 'SPACE'):
        return 'control:space', 'SPACE'
    if t in ('ESC', 'ESCAPE'):
        return 'control:esc', 'ESC'
    if t in ('ENTER', 'RETURN'):
        return 'control:enter', 'ENTER'
    if re.fullmatch(r'[0-9]', t):
        return f'control:digit_{t}', t
    if re.fullmatch(r'[A-Z]', t):
        return f'control:key_{t.lower()}', t
    if t.startswith('ARROW') or t in ('UP', 'DOWN', 'LEFT', 'RIGHT'):
        return f'control:arrow_{t.replace("ARROW", "").lower()}', t
    return f'control:{t.lower()}', t




def _enclosing_class(text: str, offset: int) -> str:
    last = ''
    for m in _CLASS.finditer(text):
        if m.start() > offset:
            break
        last = m.group(1)
    return last


def _callback_near(text: str, end: int) -> str:
    m = _CALLBACK.search(text, end, end + 220)
    return m.group(1) if m else ''


def _resolve_dynamic_key_set(text: str, expr: str) -> tuple[list, str]:
    prop = expr.strip().split('.')[-1].strip()
    if not re.fullmatch(r'[A-Za-z_]\w*', prop):
        return [], f'expression {expr!r} is not a simple property path'
    vals = re.findall(rf"""\b{re.escape(prop)}\s*:\s*['"]([A-Za-z0-9_]+)['"]""", text)
    if not vals:
        return [], (f'no literal assignment to {prop!r} in the same file; '
                    f'the key set is unresolved')
    seen, out = set(), []
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out, f"resolved from {len(vals)} literal '{prop}' assignment(s) in the same file"


def scan_bindings(src_dir: Path) -> tuple[list, dict]:
    """-> ([Binding], coverage). Pure pattern parse, every key concrete."""
    src_dir = Path(src_dir)
    files = [p for p in sorted(src_dir.rglob('*'))
             if p.is_file() and p.suffix in ('.ts', '.js', '.tsx')
             and 'node_modules' not in str(p) and '/dist/' not in str(p)]
    out: list = []
    unparsed, unresolved_dynamic, unparsed_lines = 0, [], []
    n = [0]

    def bid():
        n[0] += 1
        return f'B{n[0]:02d}'

    for p in files:
        txt = p.read_text(errors='replace')
        rel = str(p.relative_to(src_dir))
        covered = set()

        def note(m):
            a = txt[:m.start()].count('\n') + 1
            b = txt[:m.end()].count('\n') + 1
            covered.update(range(a, b + 1))
            return a

        for rx, pattern, kind in _BIND_PATTERNS:
            for m in rx.finditer(txt):
                ln = note(m)
                key = m.group('key')
                eid, disp = canonical_control(key)
                literal = (f'keydown-{key}' if 'keydown' in pattern else
                           f'keyup-{key}' if 'keyup' in pattern else
                           f'Key({key})')
                out.append(Binding(
                    bid(), eid, kind, 'scene', _enclosing_class(txt, m.start()),
                    literal, pattern.replace('<KEY>', key), f'{rel}:{ln}',
                    _callback_near(txt, m.end()),
                    f'unresolved (callback: {_callback_near(txt, m.end()) or "n/a"})'))
        for m in _DYN_BIND.finditer(txt):
            ln = note(m)
            expr = m.group('expr')
            keys, how = _resolve_dynamic_key_set(txt, expr)
            if not keys:
                unresolved_dynamic.append((rel, ln, expr, how))
                continue
            cb = _callback_near(txt, m.end())
            cls = _enclosing_class(txt, m.start())
            for k in keys:
                eid, _ = canonical_control(k)
                out.append(Binding(
                    bid(), eid, 'scene_keydown', 'scene', cls,
                    f'keydown-{k}', f'input.keyboard.on(`keydown-${{{expr}}}`)',
                    f'{rel}:{ln}', cb,
                    f'unresolved (callback: {cb or "n/a"})', how))
        for m in _CURSORS.finditer(txt):
            ln = note(m)
            cls = _enclosing_class(txt, m.start())
            for d in ('UP', 'DOWN', 'LEFT', 'RIGHT'):
                eid, _ = canonical_control(d)
                out.append(Binding(bid(), eid, 'cursor_keys', 'scene', cls,
                                   f'cursors.{d.lower()}', 'createCursorKeys()',
                                   f'{rel}:{ln}'))
        for m in _SCENE_POINTER.finditer(txt):
            ln = note(m)
            cb = _callback_near(txt, m.end())
            out.append(Binding(bid(), 'control:pointer', 'scene_pointer', 'scene',
                               _enclosing_class(txt, m.start()), m.group('ev'),
                               "this.input.on('<EV>')".replace('<EV>', m.group('ev')),
                               f'{rel}:{ln}', cb,
                               f'unresolved (callback: {cb or "n/a"})'))
        for m in _OBJ_POINTER.finditer(txt):
            if m.group('recv').startswith('this.input'):
                continue                       # already counted as scene-level
            ln = note(m)
            cb = _callback_near(txt, m.end())
            recv = m.group('recv')
            out.append(Binding(
                bid(), 'control:pointer', 'object_pointer',
                'global' if recv.split('.')[0] in ('window', 'document') else 'object',
                _enclosing_class(txt, m.start()) or recv, m.group('ev'),
                f"{recv}.on('{m.group('ev')}')", f'{rel}:{ln}', cb,
                f'unresolved (callback: {cb or "n/a"})',
                f'receiver in source: {recv}'))

        for i, l in enumerate(txt.splitlines(), 1):
            if not re.search(r'\b(keyboard|keydown|keyup)\b', l, re.I):
                continue
            if i in covered:
                continue
            st = l.strip()
            if st.startswith(('//', '*', '/*')):
                continue
            if 'addEventListener' in st or 'removeEventListener' in st:
                continue
            if re.search(r'(private|public|let|const|var)\s+\w+\??\s*:', st):
                continue
            if re.search(r'KeyCodes\.[A-Z0-9_]+\s*,?\s*$', st):
                continue
            if re.search(r'if\s*\(.*keyboard|keyboard\s*\)', st, re.I):
                continue
            # A FIELD TYPE inside an object-type literal:
            #     private wasd!: { W: Phaser.Input.Keyboard.Key; A: ... }
            # The modifier sits on an earlier line, so the declaration filter
            # above misses it. Tower defense had none of these; platformers
            # declare a whole WASD block this way and it pushed their coverage
            # to `partial`, which would have downgraded every negative fact.
            # A TYPE POSITION, not a registration. Covers all of:
            #     public spaceKey!: Phaser.Input.Keyboard.Key;
            #     protected cursors!: Phaser.Types.Input.Keyboard.CursorKeys;
            #     override update(spaceKey: Phaser.Input.Keyboard.Key): void
            #     const mk = (primary: Phaser.Input.Keyboard.Key, ...) => ...
            #     ): Phaser.Input.Keyboard.Key {
            # The earlier filter demanded a leading modifier on the same line and
            # did not allow `!`, so a whole WASD type block counted as blind
            # spots and pushed platformer coverage to `partial` -- which would
            # have downgraded every negative fact on those games for no reason.
            if re.search(r':\s*Phaser\.(Types\.)?Input\.Keyboard\.', st) and \
                    not re.search(r'addKey|createCursorKeys|\.on\(', st):
                continue
            # POLLING an already-registered key is not registering one.
            # Verified before adding this: every polled key on these games traces
            # back to an addKey()/createCursorKeys() the parser did find. The one
            # apparent exception, `dashKey`, exists only inside a commented-out
            # line in _TemplatePlayer.ts.
            if re.search(r'\b(JustDown|JustUp|isDown|checkDown|addCapture)\b', st):
                continue
            unparsed += 1
            unparsed_lines.append((rel, i, st[:120]))

    return out, {'n_files': len(files), 'unparsed': unparsed,
                 'unparsed_lines': unparsed_lines,
                 'unresolved_dynamic': unresolved_dynamic}




