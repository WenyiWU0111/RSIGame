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
"""Play one build for a fixed budget and write down what happened.

THE ONE THING THIS DOES NOT DO IS JUDGE. It reaches for as many of the game's
declared mechanics as it can inside its step budget and records the readings
either side of every action. Whether any of it is a defect is decided later,
by a reader looking at the whole trace at once -- because an agent that is
asked to find bugs while it plays will find them, and the failure mode is
already documented in this repo: 140 clicks that placed no tower were written
up as "the core mechanic is broken", retracted the same day when runtime
introspection showed the clicks were landing on an invisible overlay.

WHY PER BUILD RATHER THAN PER PROPERTY. The verifier arm plays too, but it
plays wearing a blindfold: it is measuring ONE property and its
`blocking_defect` channel only opens when the game stops that measurement. So
a defect is found when a property happens to trip over it. Hajimi's "clicking a
cushion places no tower" surfaced only because an unrelated placement property
was being checked; had the scheduler drawn a HUD property that round, the same
broken click would have gone unmentioned.

WHAT IT REPORTS AT THE END is coverage, not verdicts: which declared mechanics
it exercised, and which it could not reach. `not_reached` is load-bearing --
"I never got there" and "I got there and it was broken" are opposite findings,
and the whole point of separating this from the judging step is that neither
gets rounded into the other.

WHAT THE HARNESS OWES THE AGENT, learned by watching it go without. Everything
the agent could work out for itself but would have to spend budget on, this
module hands over before step one: the controls, the mechanics the game claims,
the size of the window it is clicking into, and a frame after every action
whether it thought to ask or not. A step spent discovering the arrow keys is a
mechanic not reached.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .verifier_agent import (ACTIONS, _chat, _session_for, _tools_for,
                             _turn_content)
from ... import paths

# Twenty model turns. Enough to cross a title screen, a how-to card and reach
# live play with room to try several mechanics; short enough that one of these
# per node is affordable when every repaired child gets one too.
DEFAULT_STEPS = 20
# One frame per action. The verifier runs on six because it is answering one
# question and images dominate its request; here the picture sequence IS the
# output, so every step gets one.
MAX_FRAMES = 20

# ============ the second look after a discrete input ============
#
# A screenshot taken promptly after a key press catches the screen BEFORE a
# transition finishes, and "the screen did not change" is then recorded as
# "the key did nothing". That is not a hypothetical: bounce-arcade-D3-000
# fades the camera for 500 ms and waits another 500 ms before switching
# scenes, so pressing Enter on its title screen produced a frame still showing
# the title at 16.6% change. The reader wrote down "Enter does not start the
# game despite the title saying PRESS ENTER / SPACE"; the next step pressed
# Space sixteen seconds later, by which time the level was up, and the 73.4%
# change was credited to Space. A repair agent then spent seventeen minutes
# inventing a race-condition theory for a bug that did not exist, and the
# re-check reproduced the same artefact and called it unfixed. Enter works --
# measured afterwards in a browser, on BOTH builds: the level is up 400 ms
# after the press.
#
# So: when a discrete input barely moves the screen, look again. If the second
# frame differs, the first one caught a transition and the settled frame is
# the honest record. If both agree, the input really did nothing, and now that
# is a measurement instead of an assumption.
#
# The prompt already warned readers about transitions caught early. It did not
# help, and could not: the reader's evidence genuinely said "no change", and
# telling it to doubt its evidence gives it nothing to put in place of it.
# THE THRESHOLDS ARE MEASURED, and the first guess at them was wrong in a way
# worth recording. "Barely moved" was set to 3%, on the assumption that a
# missed input looks like a still screen. It does not: pressing Enter on this
# title screen moves 7.3%, 9.2% and 16.6% on three runs -- the background
# animates -- so a 3% trigger would never have fired on the very case it was
# written for. The mistaken reading was never "nothing changed"; it was "16.6%
# changed and it is still the title screen".
#
# Measured on bounce-arcade-D3-000, same press, same build:
#     immediately after Enter        7.3%   (transition begun, title still up)
#     1500 ms later                 84.9%   (Level 1)
# So a scene change is 70-85% and background animation is 10-20%. The trigger
# sits above the animation band and the confirmation sits between the two.
SETTLE_RECHECK_ACTIONS = ('press_key', 'click', 'hold_key')
SETTLE_RECHECK_BELOW_PCT = 40.0    # look again unless the screen already turned over
SETTLE_RECHECK_MS = 1500           # the fade+delay in this case totals 1000 ms
SETTLE_RECHECK_MIN_PCT = 35.0      # above animation (10-20%), below a scene change (70-85%)
# How many of those frames stay attached to the conversation. The agent keeps
# the FULL text history either way -- every action, every reading, in order --
# but at twenty steps twenty images is about 2 MB re-sent every turn, and what
# decides the next move is the recent ones. Older turns keep a line saying a
# frame was there.
KEEP_IMAGES = 6

# A click that misses and a click that lands on a broken button are the same
# two lines in the trace. Off by default: it changes the frame the agent is
# shown, so it is an experimental variable, not a silent upgrade.
CLICK_EVIDENCE = os.environ.get('RSIGAME_AGENT_CLICK_EVIDENCE', '0') == '1'

# This agent PLAYS. It does not read the source, and it does not ask for
# screenshots.
#
# The source tools belong to the verifier, which has to quote a file to justify
# a verdict on a programmatic property. Handing the whole verifier menu to an
# explorer was thoughtless, and it cost exactly what you would expect: on the
# first live hajimi run six of nineteen steps went to `list_source`,
# `read_source` x4 and `read_bindings`, four files came back, none of it turned
# into an action, and the session ended with "tower selection and tower
# placement were not exercised" -- the game's central mechanic, and the one an
# earlier repair had actually fixed.
#
# `screenshot` goes for a different reason: the loop already takes one after
# every action, so an agent asking for one buys a picture it was getting free
# and pays a step for it. It did exactly that on the same run.
# `set_scene_fields` is here for a different reason from the rest. The others
# are tools this arm does not need; that one is a tool it must not have. The
# session refuses it anyway unless the caller enabled writes, and this arm
# never does -- but a refusal BY NAME tells the model what it did wrong, where
# the session's generic refusal reads as the game being broken.
_REFUSED = ('read_source', 'list_source', 'read_bindings', 'screenshot',
            'set_scene_fields')
PLAY_ACTIONS = ('boot', 'boot_scenario', 'press_key', 'hold_key', 'click',
                'wait', 'read_state', 'read_text', 'read_scene_fields')


@dataclass
class Exploration:
    game_id: str = ''
    artifact_ref: str = ''
    steps: list = field(default_factory=list)
    frames: list = field(default_factory=list)
    reached: list = field(default_factory=list)
    not_reached: list = field(default_factory=list)
    closing_note: str = ''
    n_steps: int = 0
    n_model_calls: int = 0
    stopped_by: str = ''
    wall_s: float = 0.0
    # Seconds spent opening the build before step 1: server start, settle, page
    # load, and any retries. Recorded because a layer's wall clock had ~19% in
    # a bucket nothing accounted for, and this is the biggest candidate -- once
    # per session here, plus once per play_cli call a repair agent makes.
    boot_s: float = 0.0
    viewport: dict = field(default_factory=dict)
    brief: str = ''
    errors: list = field(default_factory=list)
    # WHAT THE CLICK MARKER LOOKED LIKE, recorded because whoever reads this
    # trace has to describe it and cannot ask the environment. `CLICK_EVIDENCE`
    # is read from the env at import; a reader re-reading a stored trace months
    # later, or under a different arm's env, would read the wrong value and
    # tell the model the wrong thing. It already did: every arm's critique was
    # told "a pink crosshair", including EV1, which draws a red one with a 3x
    # inset -- the inset that exists specifically to separate "I aimed wrong"
    # from "the button is broken", and the critique was never told it was there.
    click_evidence: bool = False

    def to_dict(self):
        return asdict(self)


_SYSTEM = """You are playing ONE build of a small browser game to find out what it actually does.

You are NOT deciding whether anything is broken. Someone else reads this trace
afterwards and decides that. Your job is to operate as much of the game as you
can and record what you saw, so that reader has something real to work from.

Every step is for playing. You cannot read the game's source and you do not
need to -- whoever reads this trace is given the source separately. You also do
not need to ask for a screenshot: one is taken automatically after every action
and attached to your next turn, with a marker drawn on it wherever you clicked.

Spend your budget on BREADTH. Reaching six mechanics once each is worth more
than reaching one six times. If a mechanic needs a short sequence -- select,
then place; start a wave, then wait -- do the sequence. Do not re-read the HUD
after every input; act, and read when the reading will change what you do next.

THREE RULES ABOUT WHAT YOU WRITE DOWN.

Report readings, never conclusions. "I clicked (432,324), waited 500ms, and
towersGroup stayed at 0" is a reading. "Tower placement is broken" is a
conclusion, and it is not yours to draw here. This matters: 140 clicks that
placed no tower were once written up as a broken mechanic and retracted the
same day -- the clicks were being eaten by an invisible overlay, so the game
was fine and the automation was not.

Not reaching something is a real result. If you could not find how to place a
tower, say so in `not_reached`. Saying nothing, or implying it does not exist,
turns your own failure into a claim about the game.

A number that moves on its own is not an effect you caused. Before attributing
a change to your action, do nothing for the same interval and read again --
these games run enemies, timers and spawns whether you act or not.

Reply with JSON only, one action per turn:

{"thought": "<one line: what you are trying to reach and why>",
 "action": "<name>", "args": {...}}

or, to finish early:

{"action": "done",
 "args": {"reached": ["<mechanic you actually exercised, with what you saw>"],
          "not_reached": ["<what you could not get to, and what stopped you>"],
          "note": "<anything a reader of this trace should know>"}}

You will be stopped at the step budget whether or not you call `done`. Calling
it early is fine when the game is small and you have covered it.
"""


# --------------------------------------------------------------- the opening
#
# THE OPENING IS RE-BILLED ON EVERY TURN. The whole message list goes back to
# the model each call, so a 21-step session pays for it 21 times. Measured
# before any of this: 18,500 characters, 43% of everything a session spent --
# more than the images, and images were already capped at six.
#
# Two thirds of that was waste of two different kinds.
#
# The bindings were a dataclass `repr` per line, eleven fields deep, because
# the renderer tested `isinstance(b, dict)` and `scan_bindings` returns
# dataclasses. 8,591 characters where the same facts render in 646 -- and read
# better grouped by scene, since "every input on the Game Over screen calls
# restartGame" is invisible in twenty-six separate lines.
#
# The design-document sections are the other kind: genuinely long, genuinely
# read once. `Technical Architecture`, `Visual Style & Asset Registry`,
# `Game Configuration` describe how the game was built, not how it is played,
# and nothing in a play loop refers back to them.
#
# An earlier attempt truncated them AFTER the first turn. That was worse than
# it looks: the first turn still paid in full, and a model that finds a section
# replaced by a pointer may conclude the information was lost. This distils
# them ONCE PER GAME instead, cached on disk, so every turn including the first
# gets the short version and nothing is ever removed mid-conversation.

_DIGEST_SECTIONS = ('visual style', 'asset registry', 'technical architecture',
                    'architecture', 'game configuration', 'config',
                    'entities and scenes', 'entity / scene', 'authored config',
                    'scene flow', 'scene keys')
_DIGEST_KEEP = ('controls', 'the actions you have', 'what the game says it is')


def _digest_cache_dir() -> Path:
    d = Path(os.environ.get('RSIGAME_AGENT_BRIEF_CACHE')
             or str(paths.run_root() / '.brief_cache'))
    d.mkdir(parents=True, exist_ok=True)
    return d


_DIGEST_PROMPT = """Below are sections of a generated game's own design
document. A play agent is about to drive this game looking for defects. It
cannot read source and it will re-read this text on every turn of its session,
so it needs the few facts that change how it plays -- not a summary of the
document.

Reply with JSON and nothing else:

{"digest": "..."}

At most 120 words. Keep only:
  - named scenes and the order they come in, if stated
  - what ends a run, what wins one
  - resources or counters that change on their own, and roughly how fast
  - anything the game says an input does that the control list does not show

Drop everything about assets, colours, file layout, class hierarchies and build
configuration. If a section says nothing a player could act on, say nothing
about it. An empty digest is a valid answer.

THE SECTIONS
{body}
"""


def _digest_design(src: Path, sections: str, model: str | None) -> str:
    """The design document's long half, compressed once and cached.

    Keyed on the CONTENT, not the game name: a rebuilt game with the same
    document reuses the digest, and an edited one does not silently keep a
    stale summary.

    Never cached inside the game directory. Anything under a game's own path
    is part of the tree diff that decides whether a repair changed something,
    and a cache file there would read as a repair.
    """
    import hashlib
    if not sections.strip():
        return ''
    key = hashlib.sha256(sections.encode()).hexdigest()[:24]
    f = _digest_cache_dir() / f'{key}.json'
    if f.is_file():
        try:
            return json.loads(f.read_text()).get('digest') or ''
        except Exception:
            pass
    try:
        from .verifier_agent import _chat
        parsed, _raw = _chat(
            [{'role': 'user',
              'content': _DIGEST_PROMPT.replace('{body}', sections[:12000])}],
            model=model or os.environ.get('RSIGAME_MODELS_LOOP_MODEL') or 'gpt-5.5')
        text = (parsed or {}).get('digest') if isinstance(parsed, dict) else ''
    except Exception:
        text = ''
    text = str(text or '').strip()
    # A failed digest caches as empty rather than retrying every session: the
    # sections are already dropped either way, and the play agent has the
    # controls, which is the half that matters.
    try:
        f.write_text(json.dumps({'digest': text, 'src': src.name},
                                ensure_ascii=False))
    except OSError:
        pass
    return text


def _bindings_by_model(src: Path) -> str:
    """Ask a model for the input registrations when the parser found none.

    Only ever called on an empty parse. Reads the scene files rather than the
    whole tree: input is registered in a scene's create/setup, and handing over
    every file would spend the call on entity classes and asset loaders.
    """
    files = []
    for pat in ('src/scenes/*.ts', 'src/*.ts'):
        files += sorted(src.glob(pat))
    if not files:
        return ''
    body = []
    for f in files[:12]:
        try:
            body.append(f'--- {f.relative_to(src)} ---\n'
                        + f.read_text(errors='replace')[:6000])
        except OSError:
            continue
    if not body:
        return ''
    prompt = ("These are scene files from a generated browser game. List every "
              "keyboard key, pointer event or gamepad input the code registers "
              "a handler for, grouped by the class it is registered in.\n\n"
              "Reply with JSON and nothing else:\n"
              '{"bindings": [{"scene": "...", "keys": ["W -> movePlayer", '
              '"SPACE"]}]}\n\n'
              "Name the callback after `->` only where the code makes it "
              "plain. List nothing you did not see registered.\n\n"
              + '\n\n'.join(body)[:40000])
    try:
        from .verifier_agent import _chat
        parsed, _ = _chat([{'role': 'user', 'content': prompt}],
                          model=os.environ.get('RSIGAME_MODELS_LOOP_MODEL') or 'gpt-5.5')
    except Exception:
        return ''
    rows = (parsed or {}).get('bindings') if isinstance(parsed, dict) else None
    if not isinstance(rows, list):
        return ''
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        keys = [str(k) for k in (r.get('keys') or []) if k]
        if keys:
            out.append(f"  {r.get('scene') or 'unscoped'}: " + ', '.join(keys))
    return '\n'.join(out)


def _brief(src: Path) -> str:
    """The controls and mechanics, handed over rather than discovered.

    Every line here is something the agent could find on its own by spending
    steps -- reading the how-to card, pressing keys to see what they do, or
    (before they were taken away) reading the source. Steps spent on that are
    steps not spent on a mechanic, so the harness pays for it instead. It costs
    nothing: the bindings come from a static parse of the game's own
    TypeScript, run here in-process, and the mechanics from the design document
    the game shipped with.

    AN EMPTY BINDING LIST IS NOT AN ABSENCE OF CONTROLS, and this repo has
    already called FAIL twice on that confusion. So the two cases say different
    things: a parse that threw says the parse threw, and a parse that ran and
    found nothing says the game may be mouse-driven.
    """
    parts = []
    gd = src / 'GAME_DESIGN.md'
    if gd.exists():
        import re as _re
        raw = gd.read_text(errors='replace')[:12000]
        keep, drop = [], []
        for sec in _re.split(r'(?m)^(?=#+ )', raw):
            h = sec.split('\n', 1)[0].strip('# ').strip().lower()
            if len(sec) <= 400 or any(k in h for k in _DIGEST_KEEP):
                keep.append(sec)
            elif any(d in h for d in _DIGEST_SECTIONS):
                drop.append(sec)
            else:
                keep.append(sec)
        parts += ['## WHAT THE GAME SAYS IT IS (its own design document)', '',
                  ''.join(keep)[:2500], '']
        if drop:
            dg = _digest_design(src, ''.join(drop), None)
            if dg:
                parts += ['## WHAT ELSE ITS DOCUMENT SAYS, in brief', '',
                          dg, '']

    binds, failed = [], None
    try:
        # The TS scanner does not read GDScript. On a Godot project it returns
        # [] WITHOUT raising, which lands in the `else` below and tells the
        # player "probably mouse-driven" about a game that reads the keyboard.
        # The empty list has to mean "looked and found nothing" for that
        # sentence to be true, and only the matching scanner can make it so.
        if (src / 'project.godot').is_file():
            from ..evidence.gdscript_binding import scan_gdscript_bindings
            binds, _meta = scan_gdscript_bindings(src)
        else:
            from ..evidence.input_binding import scan_bindings
            binds, _meta = scan_bindings(src)
    except Exception as exc:
        failed = f'{type(exc).__name__}: {str(exc)[:120]}'

    if failed:
        parts += ['## CONTROLS', '',
                  f'The control parse did not run ({failed}). That says '
                  f'nothing about whether this game has controls -- find them '
                  f'on the how-to screen and by trying keys.', '']
    elif binds:
        # GROUPED BY SCENE, AND ONLY THE THREE FIELDS THAT ARE ACTIONABLE.
        #
        # `scan_bindings` returns dataclasses, so the `isinstance(b, dict)`
        # test below was always false and every binding fell through to
        # `f'  {b}'` -- its full repr, eleven fields of it:
        #
        #   Binding(binding_id='B01', control='control:key_w',
        #     handler_kind='scene_addkey', scope_kind='scene',
        #     scope_name='BaseGridScene', registered_event='Key(W)',
        #     pattern='addKey(KeyCodes.W)', ref='src/scenes/BaseGridScene.ts:816',
        #     callback='', mechanic_relation='unresolved (callback: n/a)',
        #     resolution_note='')
        #
        # Measured over eight games: 73,590 characters of that, where the same
        # information renders in 4,088. It is 94% waste, it is re-billed on
        # every turn of the session, and half of what it spends on is for a
        # human reading a parse dump -- `binding_id`, `handler_kind`,
        # `pattern`, `resolution_note`, and `ref`, a source line number the
        # play agent is not allowed to open.
        #
        # What is left is what a player can act on: which key, in which scene,
        # calling what.
        def _field(b, *names):
            d = b if isinstance(b, dict) else getattr(b, '__dict__', {})
            for n in names:
                v = d.get(n)
                if v:
                    return str(v)
            return ''

        by_scene = {}
        for b in binds[:40]:
            # Field names from BOTH sides. Godot registers input under
            # `control` / `event` where Phaser uses `registered_event`, and the
            # adapter added those to the dict branch upstream; the compact
            # rendering here has to look for them too or a Godot game's
            # bindings all come out as '?'.
            key = (_field(b, 'registered_event', 'control', 'key', 'code',
                          'binding', 'event')
                   .replace('Key(', '').replace('keydown-', '')
                   .replace(')', '').replace('control:', ''))
            cb = _field(b, 'callback', 'action', 'handler', 'purpose', 'note')
            scene = _field(b, 'scope_name', 'scope', 'node', 'script') or 'unscoped'
            by_scene.setdefault(scene, []).append(
                key + (f' -> {cb}' if cb else ''))
        lines = []
        for scene, keys in by_scene.items():
            seen, uniq = set(), []
            for k in keys:
                if k not in seen:
                    seen.add(k)
                    uniq.append(k)
            lines.append(f'  {scene}: ' + ', '.join(uniq))
        parts += ['## CONTROLS, parsed from this build\'s own source', '',
                  '\n'.join(lines), '',
                  'A binding that exists in the code and does nothing on '
                  'screen is worth reporting as exactly that: what you '
                  'pressed, and what did not change.', '']
    else:
        # CONDITIONAL, AND THE CONDITION MATTERS. Over 159 games in the corpus
        # the regex parse succeeded on every one -- no empty result, no
        # exception -- so a fallback that ran unconditionally would be a model
        # call per game for a case that has never happened. It runs only when
        # the parse comes back with nothing, which is either a game with no
        # keyboard bindings at all or a registration style the patterns do not
        # know yet, and those two are worth telling apart.
        #
        # AN EMPTY BINDING LIST IS NOT AN ABSENCE OF CONTROLS -- this repo has
        # called FAIL twice on that confusion -- so what the model returns is
        # labelled as a reading of the source, not as a fact about the game.
        found = _bindings_by_model(src)
        if found:
            parts += ["## CONTROLS, read out of this build's source by a model "
                      "because the parser found none", '',
                      found, '',
                      'The static parse found nothing here, which means either '
                      'this game registers input in a way the parser does not '
                      'know or it has no keyboard controls. Treat the list '
                      'above as a lead to check by pressing, not as a '
                      'guarantee.', '']
        else:
            parts += ['## CONTROLS', '',
                      'The parse ran and found no keyboard bindings, so this is '
                      'probably mouse-driven. Find out by playing.', '']

    cfg = src / 'src' / 'gameConfig.json'
    if cfg.exists():
        # 2,500 characters of authored numbers, re-sent every turn. What a
        # player can act on is a handful of them -- what drains, what a run
        # costs, how many levels there are -- so the same digest runs over it.
        dg = _digest_design(src, cfg.read_text(errors='replace')[:8000], None)
        if dg:
            parts += ['## ITS AUTHORED CONFIG, in brief', '', dg, '']
    return '\n'.join(parts)


_VIEWPORT_JS = """() => {
  const c = document.querySelector('canvas');
  const r = c ? c.getBoundingClientRect() : null;
  const g = window.__rlenv && window.__rlenv.game;
  return {
    view_w: window.innerWidth, view_h: window.innerHeight,
    canvas: r ? {x: Math.round(r.x), y: Math.round(r.y),
                 w: Math.round(r.width), h: Math.round(r.height)} : null,
    internal: g && g.scale ? {w: g.scale.width, h: g.scale.height} : null
  };
}"""


def _viewport(sess) -> dict:
    """Where the game actually is on screen, in the units clicks are in.

    MEASURED, and the reason this exists: hajimi is authored at 1152x768 and
    served into a 960x640 window. The agent, reasoning in the game's own
    coordinates, clicked x=1070 -- off the right edge of a 960-wide window. The
    click landed nowhere, the tower count stayed at zero, and that zero was
    very nearly written down as evidence that placement is broken. It was
    evidence about our coordinate space.
    """
    # A Godot session has no `.page`, and without this it fell into the except
    # below and reported "the window could not be measured" -- losing exactly
    # the mapping this docstring says a near-miss depended on. Godot needs no
    # measuring: it runs at a fixed `--resolution` with no letterboxing, so
    # the window IS the canvas and the session already knows its size.
    if not hasattr(sess, 'page') and getattr(sess, 'w', None):
        # SAME SHAPE AS _VIEWPORT_JS RETURNS, including `canvas.x/y`. The first
        # version of this invented its own keys and the consumer died on
        # `canvas['x']` -- a KeyError that surfaced as `session_error` and
        # ended the exploration at step 1, which reads like the game crashed.
        w, h = sess.w, sess.h
        return {'view_w': w, 'view_h': h,
                'canvas': {'x': 0, 'y': 0, 'w': w, 'h': h},
                'internal': {'w': w, 'h': h}}
    try:
        return sess.page.evaluate(_VIEWPORT_JS) or {}
    except Exception as exc:
        return {'error': f'{type(exc).__name__}: {str(exc)[:120]}'}


def _coords_note(vp: dict) -> str:
    if not vp or vp.get('error'):
        return ('The window could not be measured, so keep your clicks inside '
                'what you can see in the frames, and say so if you are unsure '
                'where the play area ends.')
    w, h = vp.get('view_w'), vp.get('view_h')
    c, i = vp.get('canvas'), vp.get('internal')
    out = [f'The window you are clicking into is {w}x{h} pixels. Every x,y you '
           f'give is in THESE coordinates, from its top-left corner.']
    if c:
        out.append(f'The game canvas covers x {c["x"]}..{c["x"] + c["w"]}, '
                   f'y {c["y"]}..{c["y"] + c["h"]} inside that window. A click '
                   f'outside that box does not reach the game.')
    if i and c and (i.get('w') != c.get('w') or i.get('h') != c.get('h')):
        sx, sy = c['w'] / float(i['w']), c['h'] / float(i['h'])
        out.append(f'CAREFUL: internally the game thinks it is {i["w"]}x'
                   f'{i["h"]} and it is drawn scaled into {c["w"]}x{c["h"]}. '
                   f'Any coordinate you read out of the game state or see in '
                   f'its config is in that {i["w"]}x{i["h"]} space and is NOT '
                   f'what to click. Convert first: click '
                   f'({c["x"]} + gx*{sx:.3f}, {c["y"]} + gy*{sy:.3f}).')
    return ' '.join(out)


def _scenario_kw(session_cls, scenarios) -> dict:
    """Pass the scenario list only to a session that has somewhere to put it.

    The web session takes no such argument and must keep working untouched, so
    this is asked of the signature rather than assumed from the engine.
    """
    if not scenarios:
        return {}
    import inspect
    try:
        params = inspect.signature(session_cls).parameters
    except (TypeError, ValueError):
        return {}
    return ({'available_scenarios': tuple(scenarios)}
            if 'available_scenarios' in params else {})


def _scenario_ledger(ledger) -> str:
    """What states this game has, and what past sessions found in each.

    THE PROBLEM THIS ADDRESSES. A session starts at the title screen and has
    fourteen steps. Late states -- an ending, a boss phase, a resource crisis --
    take most of that to reach by playing, so round after round examines the
    opening and reports on the opening, and half of a game is never looked at.
    The spec already gives a way out: a game launched with `--scenario <id>`
    sets that state up on demand. Nothing was using it.

    THE LEDGER IS FACTS, NOT VERDICTS. Each line says whether a state has been
    visited, what was reported there, and whether the round that followed
    touched any file. It does NOT say whether the defect was fixed -- no one has
    checked, and a ledger that guesses would be worse than one that admits it,
    because the agent would trust it instead of going to look. Going to look is
    the whole point of being able to launch into the state.
    """
    if not ledger:
        return ''
    rows = []
    for e in ledger:
        sid = e.get('id', '')
        seen = e.get('rounds') or []
        if not seen:
            rows.append(f'  {sid:<22} never examined')
            continue
        last = e.get('last_finding') or ''
        edited = e.get('edited_after')
        note = f'round {seen[-1]}: {last[:90]}' if last else \
               f'round {seen[-1]}: nothing reported'
        if last and edited:
            note += ' -- files were edited after; unverified'
        elif last:
            note += ' -- no file was edited after'
        rows.append(f'  {sid:<22} {note}')
    return '\n'.join([
        '## THE STATES THIS GAME CAN BE LAUNCHED INTO', '',
        *rows, '',
        'Prefer a state nobody has examined. Where something was reported and '
        'files were edited afterwards, nobody has confirmed the fix -- going '
        'back and looking is worth more than a report repeated from memory. '
        'These are records of what was SAID, not of what is true now.'])


# What the menu says about `click` when the target form is live. The default
# description asks for a coordinate, which is the thing the loop is bad at;
# this asks for a name and keeps the coordinate as the escape hatch that
# dragging and aiming still need.
_CLICK_TARGET_DOC = (
    'click something you can see. NAME IT: {"target": "the PLAY button"} -- '
    'a separate step looks at the frame and resolves that to a pixel, which '
    'is more reliable than a coordinate you estimate here, and it puts what '
    'you were aiming at into the record. If the target is not on screen you '
    'are told so and NOTHING is clicked, which is a real answer. '
    '{"target": "the PLAY button", "times": 5} for a mechanic that needs '
    'repeated hits, up to 10. Use {"x": 480, "y": 320} instead only for a '
    'point with no name -- somewhere in the world, a spot on a canvas.')


def _opening(src: Path, steps: int, supports, ledger=None) -> str:
    playable = tuple(a for a in supports if a in PLAY_ACTIONS)
    parts = [f'You have {steps} steps. Boot the game first.', '',
             _brief(src), '',
             '## THE ACTIONS YOU HAVE', '',
             (_tools_for(playable).replace(
                 ACTIONS['click'], _CLICK_TARGET_DOC)
              if CLICK_GROUNDING else _tools_for(playable)), '']
    led = _scenario_ledger(ledger)
    if led:
        parts += [led, '']
    parts.append('Reach for what the description above promises. What you '
                 'cannot reach goes in `not_reached` with the reason.')
    return '\n'.join(parts)


# ============ NAMING WHAT TO CLICK, INSTEAD OF WHERE ============
#
# Asked plainly, in a short prompt with one frame, this model puts the centre
# of a button within two pixels: seventeen tries across three games, every one
# inside the control. Asked for the same coordinate from inside the play
# loop -- the same model, the same frame, the opening prepended -- it lands
# outside the button most of the time, and greedy decoding makes it worse
# rather than better (0/12 at temperature 0, against 9/11 at the default). The
# ability is there; the long prompt is what loses it.
#
# So the agent stops emitting coordinates for controls it can name. It says
# WHAT it wants to click, and one short call resolves that to a pixel.
#
# The accuracy is the smaller half of why. The bigger half is that the trace
# stops being ambiguous. "clicked (875,467), 1.2% change" cannot distinguish a
# click that missed from a control that is dead, and the critique has been
# reading the first as the second -- four rounds of horror-tape-archive went
# into moving the hit area of a PLAY button that works. "aimed at the PLAY
# button, resolved to (640,540), 1.2% change" is a claim about the game, and a
# checkable one.
#
# `found: false` is a real observation and is recorded as one. A grounding
# call that cannot see the target must not fall back to the centre of the
# screen, because that manufactures exactly the evidence this is here to stop.
CLICK_GROUNDING = os.environ.get('RSIGAME_AGENT_CLICK_GROUNDING', '0') == '1'

# WHICH MODEL RESOLVES THE TARGET. Empty means the one that is playing.
#
# Worth setting, because the two calls want opposite things. Play needs the
# long context and the provider that serves it quickly; grounding needs one
# short question answered the same way twice. On OpenRouter temperature 0 is
# not reproducible -- the same box request for the CONTINUE button of
# visualnovel-lastsignal came back (997,603)-(1197,661) once and
# (418,605)-(618,660) the next time, the right size in the wrong place. The
# same request to our own vLLM returned (997,603)-(1197,660) three times out
# of three, against a true (996,605)-(1198,660), and the ASSIGN CASTERS box
# to within a pixel three times out of three.
#
# So: reasoning on the API, locating on the GPU we own.
#   RSIGAME_AGENT_GROUND_MODEL=qwen38-27b
#   RSIGAME_MODELS_LOCAL_MODELS=qwen38-27b=http://127.0.0.1:8038/v1
GROUND_MODEL = (os.environ.get('RSIGAME_AGENT_GROUND_MODEL') or '').strip()

# A BOX, NOT A POINT. Asked for the centre directly this model put the
# CONTINUE button of visualnovel-lastsignal at (1055,585) -- 47 px above a box
# that starts at y=605, so the click would have missed and the miss would have
# been written up as a dead button. Asked for the bounding box instead it
# returned (997,603)-(1197,661) against a true (996,605)-(1198,660), and the
# centre falls out exact. Four for four across the four controls that produced
# the artifacts in this corpus, against three for four asking for the point.
#
# The box is also worth keeping for its own sake: it goes in the trace, so a
# reader can see what was aimed at and how big it was, and the harness can
# refuse a box that covers half the screen.
_GROUND_PROMPT = """This is a {w}x{h} screenshot of a game.

Find: {target}

Reply with JSON and nothing else.
  {{"found": true, "x1": <int>, "y1": <int>, "x2": <int>, "y2": <int>, "what": "<what you found, briefly>"}}
  {{"found": false, "why": "<what you looked for and what is there instead>"}}

x1,y1 is the TOP-LEFT corner and x2,y2 the BOTTOM-RIGHT corner of the thing's
visible bounding box, in pixels of THIS {w}x{h} image, from its top-left
corner. Answer `found: false` rather than guessing: a wrong box puts the click
somewhere real, and whatever happens next gets blamed on the game."""


def ground(frame, target: str, w: int, h: int, model: str | None) -> dict:
    """Where is `target` in this frame? One short call, one image.

    Deliberately not given the session history, the spec, the action menu or
    the step budget. Those are what the play loop carries and what costs it
    the coordinate; this call exists precisely to ask the question in the form
    the model answers well.
    """
    from .verifier_agent import _chat, _data_uri
    W, H = w or 1280, h or 720
    if not frame or not Path(frame).is_file():
        return {'found': False, 'why': 'no frame to look at'}
    try:
        parsed, _ = _chat([{'role': 'user', 'content': [
            {'type': 'text',
             'text': _GROUND_PROMPT.format(w=W, h=H,
                                           target=str(target)[:200])},
            {'type': 'image_url', 'image_url': {'url': _data_uri(Path(frame))}},
        ]}], model=model)
    except Exception as exc:
        return {'found': False, 'why': f'the grounding call failed: '
                                       f'{type(exc).__name__}'}
    if not isinstance(parsed, dict):
        return {'found': False, 'why': 'the grounding call returned no JSON'}
    box = [parsed.get(k) for k in ('x1', 'y1', 'x2', 'y2')]
    if not parsed.get('found') or not all(
            isinstance(v, (int, float)) for v in box):
        return {'found': False,
                'why': str(parsed.get('why') or 'not found')[:300]}
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    # A box covering most of the screen is the model describing the scene, not
    # locating a control, and its centre is just the middle of the window --
    # the exact reading this exists to avoid manufacturing.
    if (x2 - x1) * (y2 - y1) > 0.4 * W * H:
        return {'found': False,
                'why': f'the box returned for that target, '
                       f'({x1},{y1})-({x2},{y2}), covers most of the screen, '
                       f'so it locates nothing'}
    return {'found': True, 'x': (x1 + x2) // 2, 'y': (y1 + y2) // 2,
            'box': [x1, y1, x2, y2],
            'what': str(parsed.get('what') or '')[:120]}


def _mark_click(path, x: int, y: int, n: int, times: int = 1):
    """Draw where the click went -- on a copy.

    On a copy deliberately. The marker is drawn by the harness and is not
    something the game rendered, so a reader judging whether the game LOOKS
    right must be able to get the untouched frame, while the agent looking back
    at its own history wants the marked one. Both exist and the trace names
    each.
    """
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None
    src = Path(path)
    if not src.is_file():
        return None
    out = src.with_suffix('.marked.jpg')
    try:
        im = Image.open(src).convert('RGB')
        if not CLICK_EVIDENCE:
            d = ImageDraw.Draw(im)
            r, arm, pink = 13, 22, (255, 0, 170)
            for off in (-1, 0, 1):        # thicker ring without needing a font
                d.ellipse([x - r + off, y - r, x + r + off, y + r], outline=pink)
            d.line([x - arm, y, x - 4, y], fill=pink, width=2)
            d.line([x + 4, y, x + arm, y], fill=pink, width=2)
            d.line([x, y - arm, x, y - 4], fill=pink, width=2)
            d.line([x, y + 4, x, y + arm], fill=pink, width=2)
            label = f'click {n}' if times <= 1 else f'click {n} x{times}'
            d.text((min(x + r + 5, im.width - 58), max(y - r - 12, 2)),
                   label, fill=pink)
            im.save(out, quality=72)
            return str(out)

        # A MARKER THE AGENT CANNOT MISS, AND A LOOK AT WHAT WAS UNDER IT.
        #
        # The 13px pink ring survives a 1280x720 frame going into a
        # multimodal context about as well as a button does, which is to say
        # barely -- and a click that misses is the single most misread
        # observation in this loop. On horror-tape-archive the agent clicked
        # (640,348), (640,350) and (640,400) at a PLAY button whose visual
        # centre is (640,475); the critique turned three misses into "the hit
        # area is offset ~+127px below its visual centre" and four rounds went
        # into moving hitboxes on a button that works. 348+127 = 475: what it
        # measured was the agent's own aim.
        #
        # The inset is the fix for that specific confusion. A magnified view
        # of what sits under the cursor lets the reader see a subtitle rather
        # than a button, which is the difference between "I aimed wrong" and
        # "the game is broken" -- and nothing else in the trace carries it.
        # THE INSET IS CUT BEFORE THE MARKER IS DRAWN.
        #
        # Cut afterwards it contains the marker, magnified three times, sitting
        # over the pixels it was supposed to reveal -- which is the one thing
        # the inset must not do.
        cw, ch, zoom = 120, 80, 3
        box = (max(0, x - cw // 2), max(0, y - ch // 2),
               min(im.width, x + cw // 2), min(im.height, y + ch // 2))
        clean = im.crop(box)

        d = ImageDraw.Draw(im)
        red, dark = (255, 26, 26), (24, 0, 0)
        r, arm = 26, 48
        for off in (-1, 0, 1, 2):
            d.ellipse([x - r + off, y - r, x + r + off, y + r], outline=dark)
        for off in (0, 1):                # dark halo first, so red reads on
            d.ellipse([x - r + off, y - r, x + r + off, y + r], outline=red)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            a, b = (x + dx * 8, y + dy * 8), (x + dx * arm, y + dy * arm)
            d.line([a, b], fill=dark, width=6)
            d.line([a, b], fill=red, width=3)

        crop = clean.resize(
            ((box[2] - box[0]) * zoom, (box[3] - box[1]) * zoom),
            Image.NEAREST)
        cd = ImageDraw.Draw(crop)
        cx, cy = (x - box[0]) * zoom, (y - box[1]) * zoom
        # Ticks with a gap, not a cross. A cross through the centre hides the
        # pixel the inset exists to show, which would make the inset useless
        # for the one question it is here to answer.
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            cd.line([(cx + dx * 13, cy + dy * 13), (cx + dx * 34, cy + dy * 34)],
                    fill=red, width=3)
        # Bottom-right unless the click is there, in which case bottom-left --
        # an inset covering the thing it is meant to explain helps nobody.
        px = 12 if x > im.width * 0.55 else im.width - crop.width - 12
        py = im.height - crop.height - 12
        d.rectangle([px - 3, py - 3, px + crop.width + 2, py + crop.height + 2],
                    fill=dark)
        im.paste(crop, (px, py))
        d.rectangle([px - 3, py - 3, px + crop.width + 2, py + crop.height + 2],
                    outline=red, width=3)
        label = f'click {n}' if times <= 1 else f'click {n} x{times}'
        d.text((px + 6, py + 5), f'{label}  ({x},{y})  {zoom}x', fill=red)
        im.save(out, quality=78)
        return str(out)
    except Exception:
        return None


def _frame_delta(prev, cur):
    """How much of the picture changed, as a plain number.

    The agent cannot otherwise tell a click that did something from a click
    that did nothing: it gets two images and has to eyeball them, and a small
    sprite appearing in a busy scene is exactly what eyeballing misses.

    IT IS A READING, NOT A VERDICT, and the wording attached to it says so.
    These games run enemies, timers and spawns whether anyone acts or not, so
    a large delta is not proof the agent caused anything. The informative
    direction is the other one: 0.0 means nothing on screen moved at all,
    which is worth having when the question is whether a control is wired to
    anything.
    """
    try:
        from PIL import Image, ImageChops
    except Exception:
        return None
    try:
        a = Image.open(prev).convert('L').resize((160, 120))
        b = Image.open(cur).convert('L').resize((160, 120))
    except Exception:
        return None
    diff = ImageChops.difference(a, b)
    # 12 grey levels: below that is JPEG ringing, not the game.
    changed = sum(1 for px in diff.getdata() if px > 12)
    return round(100.0 * changed / (160 * 120), 1)




def _trim_images(messages: list, keep: int) -> None:
    """Keep the last `keep` frames attached; leave every word in place.

    The agent needs its whole history -- what it tried, in order, and what came
    back -- or it repeats itself. It does not need twenty images re-uploaded
    every turn to know that. So older image blocks become a line of text saying
    the frame is in the trace, and the reasoning history is untouched.
    """
    seen = 0
    for m in reversed(messages):
        c = m.get('content')
        if not isinstance(c, list):
            continue
        imgs = [b for b in c
                if isinstance(b, dict) and b.get('type') == 'image_url']
        if not imgs:
            continue
        seen += len(imgs)
        if seen <= keep:
            continue
        kept = [b for b in c
                if not (isinstance(b, dict) and b.get('type') == 'image_url')]
        kept.append({'type': 'text',
                     'text': '(the frame from this step is no longer attached; '
                             'it is in the trace)'})
        m['content'] = kept


_CLOSING = """Your steps are used up, so this turn is not an action -- it is the
part of the trace someone else has to read.

Reply with the `done` object and nothing else:

{"action": "done",
 "args": {"reached": ["<a mechanic you actually exercised, and what you saw>"],
          "not_reached": ["<what you could not get to, and what stopped you>"],
          "note": "<anything a reader of this trace should know>"}}

Say what you SAW, not what you concluded. And put everything you did not reach
in `not_reached` with the reason -- "I never got there" and "I got there and it
did nothing" are opposite findings, and only you know which one this was.
"""


def _ask_for_the_summary(messages, res, chat, model) -> None:
    """Collect the coverage report from an agent that ran out of steps.

    MEASURED across six games: two of them spent all twenty steps playing and
    never called `done`, so `reached` and `not_reached` came back EMPTY -- on
    the two runs with the MOST inputs, 13 and 16 of 20. The budget was working
    exactly as intended and the most valuable half of the output was being
    thrown away at the buzzer.

    This is not a step. The agent has already stopped playing; this asks what
    it learned, and an agent that cannot answer leaves the fields empty rather
    than having something invented for it.
    """
    messages.append({'role': 'user', 'content': _CLOSING})
    try:
        parsed, _raw = chat(messages, model=model)
    except Exception as exc:
        res.errors.append(f'closing summary: {type(exc).__name__}: '
                          f'{str(exc)[:120]}')
        return
    res.n_model_calls += 1
    if not isinstance(parsed, dict):
        res.errors.append('closing summary: nothing usable came back; '
                          'coverage is unrecorded, not empty')
        return
    args = parsed.get('args') or parsed
    res.reached = list(args.get('reached') or [])
    res.not_reached = list(args.get('not_reached') or [])
    res.closing_note = str(args.get('note') or '')
    res.stopped_by = 'step_budget (summary asked for afterwards)'


def explore(src, *, game_id: str = '', steps: int = DEFAULT_STEPS,
            model: str = 'gpt-5.5', port: int = 7960, out_dir=None,
            seeds: list | None = None, checks: list | None = None,
            chat=None, scenarios: list | None = None,
            scenario_ledger: list | None = None,
            goal: dict | None = None) -> Exploration:
    """Drive one build for `steps` turns and return the trace.

    `seeds` are defects observed on the PREVIOUS build. They are handed over as
    things to re-check first, not as things to confirm: a child explored freely
    may simply never revisit the place the defect lived, and "I did not look"
    would then read as "it is gone".

    `checks` are the checklist -- faults found on OTHER games built by this
    pipeline. They are a separate argument and a separate section of the
    prompt, because the seeds section states they were seen on this game's
    previous build, and for a check that is not true.

    `goal` is what this round was sent to find out: `{'ask': str, 'items':
    [...], 'scenarios': [...]}`. It turns the session from "explore this game"
    into "answer this question", which is the whole of Part 4. It does NOT take
    the session over -- the steps left after the question is answered are the
    agent's own, deliberately, because the regressions this project has caught
    were caught by free wandering: shooter-sky-duel's firing broke between
    rounds 5 and 6 while the item was `satisfied`, so nobody would have probed
    it, and only an unprompted look found it.

    `scenarios` are the named states this game's own source declares it can be
    launched into; passing them lets the session refuse an id the game does not
    have, instead of relaunching into a silent fallback. `scenario_ledger` is
    what earlier rounds found in each -- see `_scenario_ledger`. Both default to
    nothing, and with nothing this behaves exactly as it did before.
    """
    src = Path(src)
    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    chat = chat or _chat
    res = Exploration(game_id=game_id, artifact_ref=str(src),
                      click_evidence=CLICK_EVIDENCE)
    t0 = time.time()

    # A build that does not exist and a build that will not boot are different
    # findings, and without this they are the same 25-second timeout.
    #
    # MEASURED: grid-toggle-D3-000 was picked for a sweep, spent 144 seconds
    # timing out, and came back as `session_error`. Its own result.json had
    # said `build: no dist/index.html` all along. Two and a half minutes and a
    # slot in the sweep went on rediscovering something already written down,
    # and the outcome read as "this game could not be played" when the truth is
    # that this game was never made.
    #
    # ENGINE-AWARE, because `dist/index.html` is the web answer to the question
    # and not the question itself. A Godot project runs from source and has no
    # dist at all, so the web check called every Godot game "never made" and
    # stopped before `_session_for` below could ever pick the right session --
    # the whole adapter sat behind a door this line held shut.
    #
    # The Godot equivalent of "it was never made" is that the project does not
    # load, which `godot_build.check_project` answers in a few seconds and
    # which is the same trade the web check makes: pay a little to avoid the
    # 144-second timeout.
    never_built_why = None
    if (src / 'project.godot').is_file():
        from . import godot_build
        r = godot_build.check_project(src)
        if not r['builds']:
            never_built_why = (
                f'{src} is a Godot project that does not load, so there is '
                f'nothing to play: {r["why"]}. This is a fact about the '
                f'BUILD, not about whether the game works.')
    elif not (src / 'dist' / 'index.html').is_file():
        never_built_why = (
            f'{src} has no dist/index.html, so there is nothing to play. This '
            f'is a fact about the BUILD, not about whether the game works -- '
            f'it was never made.')
    if never_built_why:
        res.stopped_by = 'never_built'
        res.errors.append(never_built_why)
        res.wall_s = round(time.time() - t0, 1)
        if out:
            (out / 'exploration.json').write_text(
                json.dumps(res.to_dict(), indent=1, default=str))
        return res

    session_cls = _session_for(src)
    supports = getattr(session_cls, 'SUPPORTS', tuple(ACTIONS))
    res.brief = _brief(src)
    opening = _opening(src, steps, supports, ledger=scenario_ledger)
    if goal and (goal.get('ask') or '').strip():
        where = ', '.join(goal.get('scenarios') or [])
        opening += (
            '\n\n## WHAT THIS SESSION WAS SENT TO FIND OUT\n\n'
            + goal['ask'].strip()
            + (f'\n\nStart in: {where}.' if where else '')
            + '\n\nAnswer it either way -- "I did X and Y happened" settles '
              'it as well as finding a fault does. If you cannot get there, '
              'say which: you ran out of steps elsewhere, or you arrived and '
              'something stopped you (then say what).\n\n'
              'Steps left after that are yours. Look at whatever else you '
              'want.')

    if seeds:
        opening += (
            '\n\n## SEEN ON THE PREVIOUS BUILD OF THIS GAME -- GO AND CHECK '
            'THESE FIRST\n\n'
            'Someone changed the code since these were seen. Spend your early '
            'steps getting to each one and finding out what it does NOW. This '
            'is the job; free exploration comes after.\n\n'
            'Report what you SEE, not whether it looks fixed -- somebody else '
            'decides that from your trace.\n\n'
            'AND IF YOU CANNOT GET TO ONE, SAY WHICH OF THESE TWO IT WAS, '
            'because they mean opposite things to whoever reads this:\n'
            '  - you never went there, and ran out of steps elsewhere\n'
            '  - you went there and something stopped you -- then say WHAT '
            'stopped you. "I tried to select a cat to place and there is no '
            'cat card on the screen to click" is a finding. "Not reached" on '
            'its own is not.\n\n'
            + '\n'.join(f'  - {s}' for s in seeds))

    # NOT folded into `seeds`. That section says "seen on the PREVIOUS BUILD OF
    # THIS GAME", which is true of a carried defect and false of a checklist
    # item -- those were seen on OTHER games. Putting them under that heading
    # would be a lie in the prompt, and the agent would report a defect it was
    # told had already been observed here.
    if checks:
        opening += (
            '\n\n## THINGS OTHER GAMES FROM THIS PIPELINE GOT WRONG\n\n'
            'These have NOT been observed on this game. Each was found broken '
            'on at least two other games built the same way, which makes them '
            'worth a few steps here.\n\n'
            'Exercise them if you get the chance, and say what you saw. If you '
            'never reach one, that is fine and is not a finding -- say you did '
            'not reach it.\n\n'
            + '\n'.join(f'  - {c}' for c in checks))

    messages = [{'role': 'system', 'content': _SYSTEM},
                {'role': 'user', 'content': opening}]

    try:
        # Asked of the signature, not by catching TypeError: a substrate that
        # cannot take the argument and a substrate that raises TypeError for
        # its own reasons must not look the same, and the second one would
        # otherwise be retried silently on the default six-frame budget.
        import inspect
        if 'max_frames' in inspect.signature(session_cls).parameters:
            sess_cm = session_cls(src, port, out, max_frames=MAX_FRAMES,
                                  **_scenario_kw(session_cls, scenarios))
        else:
            res.errors.append(f'{session_cls.__name__} takes no frame budget; '
                              f'the trace will hold fewer frames than steps')
            sess_cm = session_cls(src, port, out,
                                  **_scenario_kw(session_cls, scenarios))
        n_clicks = 0
        measured = False
        prev_frame = None
        last_step_at = t0

        with sess_cm as sess:
            res.boot_s = getattr(sess, 'boot_s', 0.0)
            if res.boot_s:
                print(f'    [{game_id or src.name}] build opened in '
                      f'{res.boot_s}s', flush=True)
            for n in range(1, steps + 1):
                try:
                    # `_chat` hands back (parsed, raw): it has already done the
                    # extraction and its own three retries, so a None here is a
                    # turn that genuinely failed, not one needing re-parsing.
                    parsed, raw = chat(messages, model=model)
                except Exception as exc:
                    res.errors.append(f'model call {n}: {type(exc).__name__}: '
                                      f'{str(exc)[:160]}')
                    res.stopped_by = 'model_error'
                    break
                res.n_model_calls += 1
                if not isinstance(parsed, dict):
                    res.errors.append(f'step {n}: no usable action came back')
                    messages.append({'role': 'user', 'content':
                                     'That was not JSON. One action, JSON '
                                     'only.'})
                    continue
                messages.append({'role': 'assistant', 'content': raw})

                action = str(parsed.get('action') or '')
                args = parsed.get('args') or {}
                if action == 'done':
                    res.reached = list(args.get('reached') or [])
                    res.not_reached = list(args.get('not_reached') or [])
                    res.closing_note = str(args.get('note') or '')
                    res.stopped_by = 'done'
                    break

                rec = None
                rec_ground = None
                if action in _REFUSED:
                    # Not in the menu, so this is the model reaching for a tool
                    # it remembers rather than one it was offered. Refused, and
                    # the step is still spent -- otherwise it could sit here
                    # being told no for the whole budget.
                    rec = {'ok': False, 'action': action, 'refused': True,
                           'error': (
                               'A frame is taken automatically after every '
                               'action and attached to your next turn, so '
                               'asking for one spends a step on something you '
                               'already get.' if action == 'screenshot' else
                               'You cannot read the source here. You are '
                               'playing. Whoever reads this trace is given the '
                               'source separately, so a file you read buys '
                               'nothing, while a mechanic you never touched is '
                               'gone.')}
                elif action == 'click':
                    # RESOLVE A NAMED TARGET BEFORE ANYTHING ELSE, so the
                    # window check below sees the pixel that will actually be
                    # clicked rather than the absence of one.
                    if CLICK_GROUNDING and args.get('target') is not None:
                        vp0 = res.viewport or {}
                        g = ground(prev_frame, args.get('target'),
                                   vp0.get('view_w'), vp0.get('view_h'),
                                   GROUND_MODEL or model)
                        rec_ground = dict(g)
                        if g.get('found'):
                            args = dict(args)
                            args['x'], args['y'] = g['x'], g['y']
                        else:
                            # Not a click. Saying so is the whole point: a
                            # target that is not on screen is a fact about the
                            # frame, and clicking the middle of it instead
                            # would manufacture a reading about the game.
                            rec = {'ok': False, 'action': action,
                                   'refused': True, 'grounding': rec_ground,
                                   'error': (
                                       f'could not find '
                                       f'{str(args.get("target"))[:80]!r} in '
                                       f'this frame: {g.get("why")}. Nothing '
                                       f'was clicked. Look again, name '
                                       f'something you can see, or give x, y '
                                       f'directly.')}
                    vp, x, y = res.viewport, args.get('x'), args.get('y')
                    w, h = vp.get('view_w'), vp.get('view_h')
                    if (isinstance(x, (int, float))
                            and isinstance(y, (int, float)) and w and h
                            and not (0 <= x <= w and 0 <= y <= h)):
                        rec = {'ok': False, 'action': action, 'refused': True,
                               'error': (f'({x},{y}) is outside the {w}x{h} '
                                         f'window, so that click cannot reach '
                                         f'the game and whatever you read '
                                         f'afterwards would say nothing about '
                                         f'it. {_coords_note(vp)}')}

                if rec is None:
                    rec = sess.do(action, args)
                    if rec_ground is not None:
                        rec['grounding'] = rec_ground
                rec['n'] = n
                rec['args'] = args
                rec['thought'] = str(parsed.get('thought') or '')[:300]
                # When this happened, and how long since the previous step.
                # Without these the only duration in the trace is the action's
                # own -- 900 ms for a held key -- and a reader with nothing
                # else will use it as the interval between two readings. That
                # produced "the timer drains far too fast" on four games when
                # the real gap was a model call of about a minute.
                _now = time.time()
                rec['at_s'] = round(_now - t0, 1)
                rec['since_prev_s'] = round(_now - last_step_at, 1)
                last_step_at = _now

                if not measured and not rec.get('refused'):
                    res.viewport = _viewport(sess)
                    measured = True
                    rec['viewport'] = res.viewport

                new_frames = []
                if not rec.get('refused') and len(res.frames) < MAX_FRAMES:
                    shot = sess.do('screenshot', {})
                    fp = shot.get('frame')
                    if fp:
                        fp = str((out / fp) if out and not Path(fp).is_absolute()
                                 else fp)
                        res.frames.append(fp)
                        rec['frame'] = fp
                        show = fp
                        if prev_frame:
                            d = _frame_delta(prev_frame, fp)
                            if d is not None:
                                rec['pixels_changed_pct'] = d

                        # THE SECOND LOOK. Only when a discrete input appears
                        # to have done nothing -- that is the reading that
                        # gets written down as a defect, and the one a
                        # half-finished transition imitates exactly.
                        if (action in SETTLE_RECHECK_ACTIONS
                                and rec.get('pixels_changed_pct') is not None
                                and rec['pixels_changed_pct']
                                < SETTLE_RECHECK_BELOW_PCT):
                            sess.do('wait', {'ms': SETTLE_RECHECK_MS})
                            shot2 = sess.do('screenshot', {})
                            fp2 = shot2.get('frame')
                            if fp2:
                                fp2 = str((out / fp2) if out and not
                                          Path(fp2).is_absolute() else fp2)
                                d2 = _frame_delta(fp, fp2)
                                rec['settle_recheck'] = {
                                    'waited_ms': SETTLE_RECHECK_MS,
                                    'then_changed_pct': d2,
                                    'immediate_frame': fp,
                                }
                                if (d2 is not None
                                        and d2 >= SETTLE_RECHECK_MIN_PCT):
                                    # The input DID land; the first frame was
                                    # the transition. The settled frame is the
                                    # record, and the reported change is
                                    # measured against the previous step so it
                                    # answers "what did this input do".
                                    rec['settle_recheck']['verdict'] = (
                                        'the input landed; the first frame '
                                        'caught the transition')
                                    total = _frame_delta(prev_frame, fp2)
                                    if total is not None:
                                        rec['pixels_changed_pct'] = total
                                    res.frames.append(fp2)
                                    rec['frame'] = fp2
                                    show = fp2
                                    fp = fp2
                                else:
                                    rec['settle_recheck']['verdict'] = (
                                        'still nothing after waiting; the '
                                        'input really did not land')
                        prev_frame = fp
                        if action == 'click':
                            n_clicks += 1
                            marked = _mark_click(fp, int(args.get('x') or 0),
                                                 int(args.get('y') or 0),
                                                 n_clicks,
                                                 int(args.get('times') or 1))
                            if marked:
                                rec['marked_frame'] = marked
                                show = marked
                        new_frames.append(show)

                res.steps.append(rec)

                extra = []
                if rec.get('viewport'):
                    extra.append(_coords_note(rec['viewport']))
                if rec.get('marked_frame'):
                    extra.append(
                        'The red crosshair and the magnified inset on this '
                        'frame are drawn by the harness, not by the game. '
                        'The inset is what sits under the cursor at 3x. Read '
                        'it before concluding anything about hit areas: a '
                        'click that landed on a caption or on empty space is '
                        'a miss, not a broken button.'
                        if CLICK_EVIDENCE else
                        'The pink crosshair on this frame is drawn by '
                        'the harness to show where your click landed. '
                        'It is not part of the game.')
                sr = rec.get('settle_recheck')
                if sr:
                    if sr.get('then_changed_pct') is not None and \
                            sr['then_changed_pct'] >= SETTLE_RECHECK_MIN_PCT:
                        extra.append(
                            f"The screen barely moved right after this input, "
                            f"so the harness waited "
                            f"{sr['waited_ms']}ms and looked again -- and it "
                            f"HAD changed ({sr['then_changed_pct']}%). The "
                            f"frame you are being shown is the settled one. "
                            f"This input landed; the game just takes a moment "
                            f"(a fade, a delayed scene change). Do NOT record "
                            f"it as unresponsive.")
                    else:
                        extra.append(
                            f"The screen did not move after this input, and "
                            f"the harness waited {sr['waited_ms']}ms and "
                            f"looked again to be sure. It still had not "
                            f"moved. This is a measured non-response, not a "
                            f"transition caught early.")
                if 'pixels_changed_pct' in rec and n <= 2:
                    extra.append('`pixels_changed_pct` is how much of the '
                                 'screen differs from the previous frame. A '
                                 'big number is NOT proof you caused it -- '
                                 'enemies and timers move on their own. A 0.0 '
                                 'is the informative direction: nothing on '
                                 'screen moved at all.')
                content = _turn_content(rec, new_frames)
                if extra:
                    note = {'type': 'text', 'text': ' '.join(extra)}
                    content = ([note] + content if isinstance(content, list)
                               else [note, {'type': 'text',
                                            'text': str(content)}])
                messages.append({'role': 'user', 'content': content})
                _trim_images(messages, KEEP_IMAGES)
            else:
                res.stopped_by = 'step_budget'
                _ask_for_the_summary(messages, res, chat, model)
    except BaseException as exc:
        # A boot that never happened is an infrastructure failure, not an
        # observation about the game, and must not come back looking like an
        # exploration that found nothing.
        res.errors.append(f'session: {type(exc).__name__}: {str(exc)[:200]}')
        res.stopped_by = res.stopped_by or 'session_error'

    res.n_steps = len(res.steps)
    res.wall_s = round(time.time() - t0, 1)
    if out:
        (out / 'exploration.json').write_text(
            json.dumps(res.to_dict(), indent=1, default=str))
    return res
