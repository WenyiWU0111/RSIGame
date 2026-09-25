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
"""What keys does this Godot game listen for?

WHY THIS EXISTS AT ALL. The TypeScript scanner next door globs `.ts/.js/.tsx`,
so on a Godot project it returned an empty list -- and an empty list, by the
contract that scanner's own docstring sets out, means "the parse ran and found
nothing", which the repair brief renders as *"This game may be driven by the
mouse."* For a keyboard-driven Godot game that is not a gap in the evidence, it
is a false statement in the brief, and the same file records that saying "no
controls" when nothing was measured has already produced two wrong verdicts.

So the choice was never between this parser and a perfect one. It was between
this parser and a confident lie.

THE SAME DISCIPLINE AS THE TS SCANNER: be right about the common case, silent
about the rest, never wrong. Every keycode below was read out of the engine on
this machine (`OS.find_keycode_from_string`), not recalled; anything this
cannot decode is reported as the raw number rather than guessed at.

GODOT SPELLS INPUT TWO WAYS and a scanner that knows only one is half blind:

    Input.is_action_pressed("move_left")     -- an ACTION, whose keys live in
                                                project.godot's [input] map
    event.keycode == KEY_A                   -- a key, named right there

Both are parsed. Actions are resolved through the input map when it names
them, so the brief can say `move_left -> A / Left` instead of an action name
the agent would have to go and look up.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# Off by default: it changes the opening brief, which changes what the play
# agent does, so it is an experimental variable like the other three. It must
# be ON for every arm of the next phase -- with it off, the brief keeps making
# a false statement about six of the corpus's games.
BINDING_RUNTIME = os.environ.get('RSIGAME_AGENT_BINDING_RUNTIME', '0') == '1'

# Measured with `OS.find_keycode_from_string` on the Godot in this image.
# The ASCII range decodes itself; only the special keys need a table.
_SPECIAL = {
    4194319: 'Left', 4194321: 'Right', 4194320: 'Up', 4194322: 'Down',
    4194309: 'Enter', 4194305: 'Escape', 4194306: 'Tab',
    4194325: 'Shift', 4194326: 'Ctrl', 4194328: 'Alt',
    4194308: 'Backspace', 4194312: 'Delete',
}

SKIP_DIRS = {'.godot', '.git', '__pycache__', 'node_modules'}

# `Input.is_action_pressed("x")`, `event.is_action_just_pressed("x")`, ...
_ACTION = re.compile(
    r'\b(?:Input|event|_event|ev)\s*\.\s*'
    r'is_action(?:_just)?_(?P<phase>pressed|released)\s*\(\s*'
    r'["\'](?P<action>[\w./-]+)["\']')
# `Input.is_key_pressed(KEY_W)` / `is_physical_key_pressed(KEY_W)`
_KEYFN = re.compile(
    r'\bInput\s*\.\s*is_(?:physical_)?key_pressed\s*\(\s*'
    r'KEY_(?P<key>\w+)\s*\)')
# `event.keycode == KEY_A` (either order), also physical_keycode
_KEYCMP = re.compile(
    r'(?:(?:keycode|physical_keycode)\s*==\s*KEY_(?P<key>\w+)'
    r'|KEY_(?P<key2>\w+)\s*==\s*(?:keycode|physical_keycode))')

_FUNC = re.compile(r'^\s*func\s+(\w+)', re.M)

# ACTIONS REGISTERED AT RUNTIME, WHICH project.godot CANNOT KNOW ABOUT.
#
# `parse_input_map` reads project.godot, and for a game that calls
# `InputMap.add_action` in its own setup that file is empty of the actions the
# code then reads. The brief said `move_left (not in this project's input map)`
# for platformer-ink-trail, whose Main.gd registers exactly that action three
# lines into `setup_input()`. That sentence is a claim about the GAME, it was
# false, and it cost eight rounds: the play agent reasonably declined to spend
# steps on keys it had been told did not exist, wrote "I did not test arrow
# keys or WASD" into `not_reached`, and the critique turned the non-test into
# "the player cannot move horizontally". Round 3 then repaired a working
# system and broke it.
#
# Six of the forty-five games in the gpt corpus register this way and declare
# nothing in project.godot, so every one of them was briefed with a false
# statement about its own controls.
#
# Two shapes are recognised. The direct call names the action; the wrapper --
# a helper taking (action, key) and calling add_action on its parameter -- is
# how these games actually do it, and resolving it recovers the KEY as well,
# so the brief can say `move_left -> A / Left` instead of nothing.
_ADD_ACTION_LIT = re.compile(
    r'InputMap\s*\.\s*add_action\s*\(\s*["\'](?P<action>[\w./-]+)["\']')
_ADD_ACTION_VAR = re.compile(
    r'InputMap\s*\.\s*add_action\s*\(\s*(?P<var>[a-z_]\w*)\s*\)')
_FUNC_SIG = re.compile(
    r'^\s*func\s+(?P<name>\w+)\s*\(\s*(?P<a1>\w+)[^)]*\)', re.M)


def runtime_input_map(texts: list[str]) -> dict[str, list[str]]:
    """action -> [key names] for actions the code registers as it starts.

    Conservative in the same way as everything else here: a wrapper only
    counts when its first parameter is the name `add_action` is called with,
    and a call only counts when the action is a literal. Anything it cannot
    resolve is left out rather than guessed, so this can add facts to the
    brief and never invent one.
    """
    out: dict[str, list[str]] = {}
    wrappers: dict[str, int] = {}
    for txt in texts:
        for m in _FUNC_SIG.finditer(txt):
            nxt = _FUNC_SIG.search(txt, m.end())
            body = txt[m.end():nxt.start() if nxt else len(txt)]
            for v in _ADD_ACTION_VAR.finditer(body):
                if v.group('var') == m.group('a1'):
                    wrappers[m.group('name')] = 1
    for txt in texts:
        for m in _ADD_ACTION_LIT.finditer(txt):
            out.setdefault(m.group('action'), [])
        for name in wrappers:
            call = re.compile(
                r'\b' + re.escape(name) +
                r'\s*\(\s*["\'](?P<action>[\w./-]+)["\']\s*'
                r'(?:,\s*KEY_(?P<key>\w+)\s*)?\)')
            for m in call.finditer(txt):
                keys = out.setdefault(m.group('action'), [])
                k = m.group('key')
                if k:
                    pretty = k.title() if len(k) > 1 else k
                    if pretty not in keys:
                        keys.append(pretty)
    return out


def _keyname(code: int) -> str:
    if code in _SPECIAL:
        return _SPECIAL[code]
    if code == 32:
        # chr(32) decodes correctly and prints as nothing: the brief said
        # `fire -> ` and an agent reading that learns less than from no line
        # at all. A key whose name is invisible needs to be spelled.
        return 'Space'
    if 33 <= code < 127:
        return chr(code).upper()
    return f'keycode {code}'


def parse_input_map(project_godot: Path) -> dict[str, list[str]]:
    """action -> [key names], from project.godot's `[input]` section.

    Godot writes each event as an `Object(InputEventKey, ... )` blob on one
    line, so the keycodes can be lifted without parsing the whole format --
    and an action whose events are joypad or mouse simply yields no keys,
    which is the truth about it rather than a failure.
    """
    out: dict[str, list[str]] = {}
    try:
        txt = project_godot.read_text(errors='replace')
    except OSError:
        return out
    if '[input]' not in txt:
        return out
    section = txt.split('[input]', 1)[1]
    # Stop at the next [section] header so a later block is not read as input.
    nxt = re.search(r'^\[', section[1:], re.M)
    if nxt:
        section = section[:nxt.start() + 1]
    for m in re.finditer(r'^(?P<name>[\w./-]+)\s*=\s*\{', section, re.M):
        start = m.end()
        depth, i = 1, start
        while i < len(section) and depth:
            if section[i] == '{':
                depth += 1
            elif section[i] == '}':
                depth -= 1
            i += 1
        body = section[start:i]
        keys = []
        for km in re.finditer(
                r'"(?:physical_)?keycode"\s*:\s*(\d+)', body):
            code = int(km.group(1))
            if code:
                name = _keyname(code)
                if name not in keys:
                    keys.append(name)
        out[m.group('name')] = keys
    return out


def _enclosing_func(txt: str, pos: int) -> str:
    last = ''
    for m in _FUNC.finditer(txt, 0, pos):
        last = m.group(1)
    return last or '(file scope)'


def scan_gdscript_bindings(src_dir) -> tuple[list[dict], dict]:
    """-> ([binding dict], coverage), shaped like scan_bindings' return.

    Plain dicts rather than the TS scanner's Binding: the consumers read named
    fields through a getattr/`.get` helper, and adding a second constructor
    call to a dataclass this module does not own is a way to break the web
    path while adding a Godot one.
    """
    src_dir = Path(src_dir)
    files = [p for p in sorted(src_dir.rglob('*.gd'))
             if not (set(p.parts) & SKIP_DIRS)]
    imap = parse_input_map(src_dir / 'project.godot')
    texts = {p: p.read_text(errors='replace') for p in files}
    if BINDING_RUNTIME:
        for action, keys in runtime_input_map(list(texts.values())).items():
            if keys:
                imap.setdefault(action, keys)
            else:
                imap.setdefault(action, [])
    out: list[dict] = []
    n = 0

    for p in files:
        txt = texts[p]
        rel = str(p.relative_to(src_dir))

        def add(control, event, pos, note=''):
            nonlocal n
            n += 1
            out.append({
                'id': f'G{n:02d}',
                'control': control,
                'event': event,
                'scope_name': _enclosing_func(txt, pos),
                'ref': f'{rel}:{txt[:pos].count(chr(10)) + 1}',
                'mechanic_relation': note,
            })

        for m in _ACTION.finditer(txt):
            action = m.group('action')
            keys = imap.get(action)
            if keys:
                shown = f'{action} -> {"/".join(keys)}'
            elif action in imap:
                shown = f'{action} (no keyboard event in the input map)'
            else:
                # Godot's own `ui_*` actions are built in and not written to
                # project.godot; saying "unmapped" about them would be wrong.
                # NOT "not in this project's input map". That is a claim
                # about the game and it can be false: the action may be
                # registered at runtime, which project.godot cannot show. What
                # is always true is what this parser did and did not find, so
                # that is what it says. See `runtime_input_map` for the cost of
                # the old wording.
                shown = (f'{action} (built-in action)'
                         if action.startswith('ui_')
                         else (f'{action} (no key for it in project.godot; '
                               f'the game may register it at runtime -- try it '
                               f'before concluding it does not exist)'
                               if BINDING_RUNTIME
                               else f'{action} (not in this project\'s input map)'))
            add(shown, f'action_{m.group("phase")}', m.start())

        for m in _KEYFN.finditer(txt):
            add(m.group('key'), 'key_pressed', m.start())

        for m in _KEYCMP.finditer(txt):
            add(m.group('key') or m.group('key2'), 'keycode_match', m.start())

    return out, {'n_files': len(files), 'actions_in_input_map': len(imap),
                 'engine': 'godot'}
