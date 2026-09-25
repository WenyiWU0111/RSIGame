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
"""Decide whether one property holds on one build, by playing the game.

WHY THIS REPLACES THE COMPILED CONTRACT. The old path turned a property into a
fixed script before the game was ever seen, which meant the script had to know
in advance everything that could stand between it and the state under test. It
never could, and the record of that is four proxies deep:

    hasPlayer                  true underneath an onboarding card
    hasPlayer AND no prompt    true while the card is hidden but the level's
                               own `howToStarted` gate is still false, so
                               `update()` returns on its first line
    the world is frozen        so every reading is identical and the oracle
                               reports, correctly and uselessly, that the
                               input changed nothing
    "no numeric field moved"   the proxy that was going to be added next, and
                               false for turn-based games, tower defence
                               before a wave, and any puzzle at rest

Each fix replaced game semantics with one more harness assumption. An agent
that looks does not need the assumption: it presses ENTER again because it can
see nothing moved, and it can read `howToStarted` out of the scene because the
source is right there.

Measured, on the contracts this replaces: 24 archived contracts used 4 oracle
kinds, 22 of them were two fixed recipes, and the only genuinely per-property
values were a key, a field, and a floor that was 2.0 in eight cases out of
nine. The compiler's free choices -- `wait_ms` of 3000, 800, 300, 150, 50,
30000 -- were invented pacing, and 80 of 106 properties could not be expressed
at all.

WHAT KEEPS THIS FROM BEING "THE MODEL SAYS SO". Three things, and none is a
new oracle.

  1  the requirement must be quoted from the game's own text, and the quote is
     looked up mechanically. A verifier that cannot source what the game
     promised has not established a violation, it has stated a preference.
  2  a verdict must carry the actions taken and the observations they
     produced. A bare PASS/FAIL is downgraded to INCONCLUSIVE here, not
     argued with.
  3  the whole trace -- every action, every state read, every frame -- is
     written to disk before the verdict is returned, so a reader can disagree.

WHAT "THE SAME RULER" MEANS NOW. Not the same keystrokes. The same CRITERION:
the property, the requirement it rests on, and the tool budget, hashed and
frozen before either build is touched. The parent may need three ENTERs to
reach play and the child two; forcing identical scripts on two different builds
is what made a UI change indistinguishable from a repair.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..llm import _data_uri

PASS, FAIL, INCONCLUSIVE = 'PASS', 'FAIL', 'INCONCLUSIVE'

MAX_FRAMES = 6

ACTIONS = {
    'boot': 'load the page and wait for the runtime; takes no arguments',
    'boot_scenario': 'RESTART the game in a named state it declares, e.g. {"id": "boss"}. This is a fresh launch: nothing you did before survives. Use it to examine a state that would take most of a session to reach by playing.',
    'press_key': 'press and release once. {"key": "Enter"}',
    'hold_key': 'hold for a while. {"key": "d", "ms": 800}',
    'click': 'click a viewport point, once or several times in a row. '
             '{"x": 480, "y": 320} or {"x": 480, "y": 320, '
             '"times": 5} when a mechanic needs repeated hits. Up to 10; '
             'a repeated click reports the state before and after.',
    'wait': 'let time pass, touching nothing. {"ms": 600}',
    'read_state': 'the runtime state object the game exposes',
    'read_text': 'the HUD and canvas text currently on screen',
    'read_scene_fields': 'named fields off the live scene objects, for gates '
                         'the state object does not expose. '
                         '{"fields": ["howToStarted", "started"]}',
    'read_bindings': 'the control bindings parsed from the game SOURCE',
    # Without this the agent guesses paths. Its first real run spent two of
    # its twenty actions on src/scenes/Level.ts and src/main.ts, neither of
    # which existed in that game.
    'list_source': 'the source files that exist, as paths',
    'read_source': 'one source file. {"path": "src/scenes/Level.ts"}',
    'screenshot': 'capture a frame; it is attached to your next turn',
    'verdict': 'finish. see the reply schema',
}














def _session_for(src):
    """Which substrate drives this build, decided by what is in the directory.

    Both classes present the same surface -- `do(action, args)`, `.actions`,
    `.observations`, `.frames`, `closing_snapshot()` -- so the verifier loop
    above cannot tell which it got, and neither can the protected verifier.
    What differs is which of the eight actions are answerable, and each
    session refuses the rest BY NAME rather than returning nothing: on Godot
    `read_state` says no bridge exists, which is not the same claim as the
    game having no state.
    """
    if (Path(src) / 'project.godot').is_file():
        from .godot_session import GodotSession
        return GodotSession
    return _Session






_SCENE_FIELDS_JS = """(names) => {
  const g = window.__rlenv && window.__rlenv.game;
  if (!g) return {error: 'no runtime'};
  // Describe, never return the object. A scene field is often a live Phaser
  // GameObject whose parent points back at it, Playwright's serialisation
  // carries cycles, and the Python client rebuilds them faithfully -- at which
  // point json.dumps raises "Circular reference detected" and the whole
  // verification dies. Measured on hajimi/H0066.
  //
  // A cycle is also not what the agent asked for. It wanted to know whether a
  // gate is set, how many things are in a group, what a counter reads.
  const describe = (v) => {
    if (v === null || v === undefined) return v === null ? null : '(undefined)';
    const t = typeof v;
    if (t === 'boolean' || t === 'number' || t === 'string') return v;
    if (t === 'function') return '(function)';
    if (Array.isArray(v)) return `(array, length ${v.length})`;
    if (t === 'object') {
      const parts = [];
      if ('length' in v && typeof v.length === 'number')
        parts.push(`length=${v.length}`);
      for (const k of ['x', 'y', 'visible', 'active', 'alpha', 'isDown',
                       'value', 'size'])
        if (k in v && ['number', 'boolean', 'string'].includes(typeof v[k]))
          parts.push(`${k}=${v[k]}`);
      if (v.getLength instanceof Function) {
        try { parts.push(`getLength()=${v.getLength()}`); } catch (e) {}
      }
      const name = (v.constructor && v.constructor.name) || 'object';
      return `(${name}${parts.length ? ' ' + parts.join(' ') : ''})`;
    }
    return String(v);
  };
  const out = {scenes: [], fields: {}};
  for (const s of g.scene.getScenes(true)) {
    out.scenes.push(s.scene.key);
    for (const n of names) {
      if (n in s) out.fields[s.scene.key + '.' + n] = describe(s[n]);
    }
  }
  return out;
}"""


# THE ONLY WAY THIS SESSION CAN WRITE, and it is deliberately a narrow one.
#
# WHY IT EXISTS. Verifying "the score is not reset after GAME OVER" means
# reaching GAME OVER, and reaching it means sitting out a 90-second level
# timer. Measured on bounce-arcade b0: 11.5 of the 18.7 minutes that fixed it
# were inside play_cli, about 90% of that was `wait`, and the branch paid the
# timer fifteen times over three attempts. The agent had already read the
# source and knew exactly which field held the clock. It simply had no way to
# say so.
#
# WHAT IT REFUSES, and why each refusal matters more than the speed-up.
#
#   A FIELD THAT DOES NOT ALREADY EXIST is not created. Setting an unknown
#   name would let a session invent state the game never had and then verify
#   against it -- a fix confirmed against a world the player cannot reach.
#   Unknown names come back named, in `missing`, rather than silently ignored.
#
#   ONLY SCALARS. No objects, no functions, no arrays. The agent may move a
#   counter or flip a flag; it may not replace a scene's collaborators or
#   graft behaviour onto a live object.
#
#   BEFORE AND AFTER ARE BOTH RETURNED, always. A self-check run under
#   manufactured state has to be legible as one afterwards -- the record is
#   the product here, and "what state was this fix confirmed in" is part of
#   the label, not an implementation detail.
#
# WHAT IT IS NOT FOR. This is an iteration accelerant, not a verification.
# Jumping the clock skips whatever the real path to that state would have run,
# so a pass under manufactured state is a hint, not a result. The branch's own
# recheck is a separate session that plays honestly and knows nothing of this,
# which is what keeps the shortcut from reaching the label.
_SET_SCENE_FIELDS_JS = """(fields) => {
  const g = window.__rlenv && window.__rlenv.game;
  if (!g) return {error: 'no runtime'};
  const scalar = (v) => ['number', 'boolean', 'string'].includes(typeof v);
  const out = {scenes: [], set: {}, missing: [], refused: {}};
  const names = Object.keys(fields || {});
  const seen = new Set();
  for (const s of g.scene.getScenes(true)) {
    out.scenes.push(s.scene.key);
    for (const n of names) {
      const v = fields[n];
      if (!scalar(v)) { out.refused[n] = 'not a scalar'; seen.add(n); continue; }
      // Existing fields only. `in` rather than a truthiness test, so a field
      // that is legitimately 0, '' or false still counts as present.
      if (!(n in s)) continue;
      seen.add(n);
      const key = s.scene.key + '.' + n;
      const before = s[n];
      if (!scalar(before)) { out.refused[key] = 'existing value is not a scalar'; continue; }
      try {
        s[n] = v;
        out.set[key] = {before: before, after: s[n]};
      } catch (e) {
        out.refused[key] = String(e).slice(0, 120);
      }
    }
  }
  for (const n of names) if (!seen.has(n)) out.missing.push(n);
  return out;
}"""


class _Session:
    """One live build, driven a step at a time, recording everything.

    `boot_to_gameplay` is deliberately NOT used. That step is the accumulated
    guesswork this module exists to remove; entering play is the agent's job
    and it has the tools to see whether it worked.
    """

    # A Phaser build answers all of them.
    SUPPORTS = tuple(k for k in ACTIONS if k != 'verdict')

    def __init__(self, src, port, out_dir, max_frames: int = None):
        from .probe_session import Probe

        self.src = Path(src)
        self._probe = Probe(self.src, port=port, out_dir=out_dir)
        self.out_dir = Path(out_dir) if out_dir else None
        self.actions, self.observations, self.frames = [], [], []
        self._n_frames = 0
        # Verification wants few frames -- it is answering one question and
        # images dominate the request size. Exploration wants one per action,
        # because its output IS the picture sequence someone reads afterwards.
        self.max_frames = MAX_FRAMES if max_frames is None else max_frames

    def __enter__(self):
        """Open the build, patiently.

        `Probe.__enter__` starts a local server, waits 1.5s, and gives the page
        25s to load. On a box also running six generation workers that is
        tight for a 24 MB bundle, and a `Page.goto` timeout there loses the
        whole verification before it has done anything -- measured on hajimi,
        twice. Three tries, backing off. A boot that still will not happen is
        raised unchanged: it is an infrastructure failure, not a verdict.
        """
        # TIMED, because nobody knows what this costs. A layer's wall clock
        # breaks down as 37% exploring, 25% repair agents, 19% verification
        # sessions and 19% "everything else" -- and of that last slice, the
        # npm build is ~3 minutes and the retry backoff ~10, leaving about
        # two thirds unexplained across a day of running. Opening a build is
        # the biggest unmeasured thing in the loop: a server start, a 1.5s
        # settle and up to 25s for the page, once per verification session and
        # again for every play_cli call a repair agent makes.
        last = None
        t0 = time.time()
        for attempt in range(3):
            try:
                self._probe.__enter__()
                self.boot_s = round(time.time() - t0, 1)
                self._boot_attempts = attempt + 1
                return self
            except Exception as exc:
                last = exc
                self._boot_attempts = attempt + 1
                time.sleep(2 + 3 * attempt)
        self.boot_s = round(time.time() - t0, 1)
        raise last

    def closing_snapshot(self) -> dict:
        """One frame and the readable state, taken after the verdict.

        Best effort by design: a build that is already broken may refuse a
        screenshot or have no runtime to read, and that is worth recording
        rather than raising -- the failure to capture is itself a fact about
        the build. Every field is independently optional.
        """
        snap: dict = {'taken': 'after the verdict; nothing measured depends '
                               'on it'}
        try:
            path = (self.out_dir or Path('.')) / 'closing_frame.jpg'
            path.parent.mkdir(parents=True, exist_ok=True)
            self.page.screenshot(path=str(path), type='jpeg', quality=80)
            snap['frame'] = str(path)
            self.frames.append(str(path))
        except Exception as exc:
            snap['frame_error'] = f'{type(exc).__name__}: {str(exc)[:120]}'
        for key, action in (('hud', 'read_text'), ('state', 'read_state'),
                            ('scenes', 'read_scene_fields')):
            try:
                snap[key] = self.do(action, {})
            except Exception as exc:
                snap[f'{key}_error'] = f'{type(exc).__name__}: {str(exc)[:120]}'
        return snap

    def __exit__(self, *exc):
        self._probe.__exit__(*exc)

    @property
    def page(self):
        return self._probe.page

    def do(self, action: str, args: dict) -> dict:
        """Run one action and record it with what it produced."""
        rec = {'action': action, 'args': args, 'n': len(self.actions) + 1}
        try:
            rec.update(self._dispatch(action, args))
        except Exception as e:
            # A tool that failed is reported to the agent, not raised: the
            # agent may well have a different route to the same answer, and a
            # crashed verification is less useful than a refused one.
            rec['error'] = f'{type(e).__name__}: {str(e)[:160]}'
        self.actions.append(rec)
        if 'observed' in rec:
            self.observations.append({'after_action': rec['n'],
                                      'action': action,
                                      'value': rec['observed']})
        return rec

    def _dispatch(self, action: str, args: dict) -> dict:
        p = self.page
        if action == 'boot':
            # THE FIRST INPUT AFTER BOOT WAS NOT LANDING, and it cost a real
            # repair.
            #
            # Measured on two different games. On hajimi the child arm pressed
            # Enter on the title screen immediately after boot and the screen
            # did not change; the parent arm did the same and it did. The
            # protected check -- which ran ZERO probes of its own -- read those
            # two incidental observations, concluded Enter-to-start had
            # regressed, and rejected a repair that four purpose-built trials
            # afterwards could not fault: Enter advanced 4/4 on both builds
            # when pressed after a wait. On grid-sokoban the same immediate
            # Enter did nothing and the verifier fell back to Space.
            #
            # So boot no longer returns the instant the page is up. It waits
            # for the runtime to become readable and SAYS what it saw: a build
            # that settles in 200ms and one that is still unreadable at the cap
            # are different facts, and a blind sleep would report them the
            # same. Unreadable is not an error -- a title screen legitimately
            # has no scene state yet -- so the cap is short and the action
            # still succeeds.
            waited, ready = 0, False
            while waited < BOOT_SETTLE_MS:
                try:
                    if self._probe.read('scene_state'):
                        ready = True
                        break
                except Exception:
                    pass
                p.wait_for_timeout(BOOT_POLL_MS)
                waited += BOOT_POLL_MS
            return {'ok': True, 'settled_ms': waited,
                    'runtime_readable': ready,
                    'note': ('the page is loaded and the runtime is up; play '
                             'may not have started. Input issued before this '
                             'settle used to be dropped by the page.')}
        if action == 'press_key':
            from .probe_session import normalise_key
            k = normalise_key(str(args.get('key', '')))
            p.keyboard.press(k)
            p.wait_for_timeout(250)
            return {'ok': True, 'key': k}
        if action == 'hold_key':
            from .probe_session import normalise_key
            k = normalise_key(str(args.get('key', '')))
            ms = max(50, min(int(args.get('ms', 800)), 5000))
            before = self._probe.read('scene_state')
            p.keyboard.down(k)
            samples = []
            for _ in range(6):
                p.wait_for_timeout(max(1, ms // 6))
                samples.append(self._probe.read('scene_state'))
            p.keyboard.up(k)
            p.wait_for_timeout(150)
            after = self._probe.read('scene_state')
            return {'ok': True, 'key': k, 'ms': ms,
                    'observed': {'before': before, 'during': samples,
                                 'after': after}}
        if action == 'click':
            x, y = int(args.get('x', 480)), int(args.get('y', 320))
            # Separate clicks, not Playwright's click_count: that is a
            # double-click gesture, and a game counting hits wants distinct
            # pointerdown events with time between them.
            times = max(1, min(int(args.get('times', 1) or 1), 10))
            before = self._probe.read('scene_state') if times > 1 else None
            for i in range(times):
                p.mouse.click(x, y)
                p.wait_for_timeout(300 if i == times - 1 else 140)
            if times == 1:
                return {'ok': True}
            return {'ok': True, 'times': times,
                    'observed': {'before': before,
                                 'after': self._probe.read('scene_state')}}
        if action == 'wait':
            ms = max(50, min(int(args.get('ms', 500)), 5000))
            before = self._probe.read('scene_state')
            p.wait_for_timeout(ms)
            return {'ok': True, 'ms': ms,
                    'observed': {'before': before,
                                 'after': self._probe.read('scene_state')}}
        if action == 'read_state':
            st = self._probe.read('scene_state')
            if st is None:
                # Same distinction as read_bindings: the runtime did not
                # answer, which is a fact about the harness, not about the
                # game's state.
                return {'error': 'the runtime exposed no state object. It may '
                                 'not have finished constructing; this says '
                                 'nothing about the game yet.'}
            return {'ok': True, 'observed': st}
        if action == 'read_text':
            return {'ok': True, 'observed': {
                'hud_text': self._probe.read('hud_text'),
                'canvas_text': self._probe.read('canvas_text')}}
        if action == 'read_scene_fields':
            names = [str(n) for n in (args.get('fields') or [])][:12]
            return {'ok': True,
                    'observed': p.evaluate(_SCENE_FIELDS_JS, names)}
        if action == 'set_scene_fields':
            # GATED AT THE SESSION, not in the prompt. The exploring and
            # verifying agents are the measurement; one that can rewrite the
            # game's state is not measuring the game. Keeping this out of their
            # menus would only stop the ones that do not guess the name, and
            # models guess. So the capability is off unless the caller turned
            # it on, and only play_cli -- the repair agent's own door -- does.
            if not getattr(self, 'allow_state_writes', False):
                return {'ok': False, 'refused': True,
                        'error': 'this session may observe the game, not '
                                 'rewrite it'}
            fields = args.get('fields') or {}
            if not isinstance(fields, dict):
                return {'ok': False, 'error':
                        'fields must be an object of name -> scalar'}
            fields = {str(k): v for k, v in list(fields.items())[:12]}
            observed = p.evaluate(_SET_SCENE_FIELDS_JS, fields)
            # Never quiet about it. Every downstream reader of this trace --
            # a person, a training set -- has to be able to see that the
            # readings after this point were taken in a state nobody played to.
            return {'ok': True, 'observed': observed,
                    'manufactured_state': True}
        if action == 'read_bindings':
            # AN EMPTY RESULT MUST NOT LOOK LIKE AN ABSENCE. `scan_bindings`
            # parses the game's TypeScript, so an artifact staged without its
            # sources yields `[]` -- indistinguishable, to a reader, from a
            # game that registers no controls at all.
            #
            # That is not hypothetical. Two verifications called FAIL on
            # exactly this: `read_bindings` returned `[]`, the agent had
            # ALREADY been told by `list_source` that the artifact has no src/
            # directory, and it reported the empty list as evidence that no
            # control was bound. The reasoning was wrong and the tool made it
            # easy. A tool that refuses cannot be misread, so this one refuses.
            if not (self.src / 'src').is_dir():
                return {'error': 'this artifact has no src/ directory, so the '
                                 'bindings CANNOT BE SCANNED. This is not the '
                                 'same as the game registering no bindings, '
                                 'and it is not evidence about the game.'}
            bs = self._probe.read('bindings')
            if isinstance(bs, list) and not bs:
                return {'ok': True, 'observed': [],
                        'note': 'the source was scanned and no binding pattern '
                                'matched. Check the parser coverage before '
                                'reading this as the game binding nothing.'}
            return {'ok': True, 'observed': bs}
        if action == 'list_source':
            root = self.src / 'src'
            if not root.is_dir():
                return {'error': 'this artifact has no src/ directory'}
            paths = sorted(
                str(p.relative_to(self.src)) for p in root.rglob('*')
                if p.is_file() and p.suffix in ('.ts', '.js', '.json')
                and 'node_modules' not in p.parts)
            return {'ok': True, 'observed': paths[:120]}
        if action == 'read_source':
            rel = str(args.get('path', '')).lstrip('/')
            f = (self.src / rel).resolve()
            # Stay inside the artifact. A path that escapes it is a bug or a
            # prompt injection, and either way is not this game's source.
            if not str(f).startswith(str(self.src.resolve())):
                return {'error': f'{rel!r} is outside the artifact'}
            if not f.is_file():
                return {'error': f'{rel!r} does not exist'}
            return {'ok': True, 'observed': f.read_text(errors='replace')[:6000]}
        if action == 'screenshot':
            if self._n_frames >= self.max_frames:
                return {'error': f'the {self.max_frames}-frame budget is '
                                 f'used up'}
            self._n_frames += 1
            name = f'frame{self._n_frames:02d}.jpg'
            path = (self.out_dir or Path('.')) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            # Stop the render loop first: a canvas driving rAF on software
            # rendering never yields the composited frame `screenshot` waits
            # for, and the call sits until it times out.
            try:
                p.evaluate("""() => {
                  const g = window.__rlenv && window.__rlenv.game;
                  if (g && g.loop && g.loop.sleep) g.loop.sleep();
                }""")
                p.wait_for_timeout(200)
                # Two tries. On a loaded box the first `screenshot` can pass
                # 'fonts loaded' and then wait out its timeout for a
                # composited frame that the canvas never yields -- measured on
                # hajimi/H0066, where the miss cost the verification its only
                # visual evidence and it correctly answered INCONCLUSIVE. A
                # lost observation is worse than a slow one.
                for attempt in (1, 2):
                    try:
                        p.screenshot(path=str(path), type='jpeg', quality=70,
                                     animations='disabled', timeout=60000)
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        p.wait_for_timeout(1500)
            finally:
                try:
                    p.evaluate("""() => {
                      const g = window.__rlenv && window.__rlenv.game;
                      if (g && g.loop && g.loop.wake) g.loop.wake();
                    }""")
                except Exception:
                    pass
            self.frames.append(str(path))
            return {'ok': True, 'frame': name, 'attached_next_turn': True}
        return {'error': f'unknown action {action!r}; the tools are '
                         f'{sorted(ACTIONS)}'}


def _turn_content(rec: dict, frames: list) -> list:
    """One tool result as message content, with any new frame attached.

    Frames go in as real `image_url` blocks using the same data-URI convention
    `agent/llm.py` defines -- imported, not reimplemented, so the
    convention lives in exactly one place. A screenshot the agent asked for and
    could not see would make the tool a lie.
    """
    body = {k: v for k, v in rec.items() if k != 'args'}
    try:
        text = json.dumps(body, default=str)[:4000]
    except ValueError as exc:
        # A tool result that cannot be serialised must not take the whole
        # verification with it. The agent is told what happened so it can try
        # a different read, and the run continues.
        text = json.dumps({'action': rec.get('action'),
                           'error': f'this result could not be serialised '
                                    f'({exc}); try a narrower read'})
    content = [{'type': 'text', 'text': text}]
    for p in frames:
        f = Path(p)
        if not f.exists():
            continue
        content.append({'type': 'image_url',
                        'image_url': {'url': _data_uri(f)}})
        content.append({'type': 'text', 'text': f'(frame {f.name})'})
    return content


# How long `boot` waits for the runtime before handing control to the agent.
# 1200ms is where the measurement landed: Enter pressed immediately after boot
# failed to register on two different games, and pressed after ~1.2s advanced
# the title screen 4 times out of 4 on both. Polled rather than slept, so a
# game that is ready in 200ms costs 200ms.
BOOT_SETTLE_MS = 1200
BOOT_POLL_MS = 200


def _tools_for(supports) -> str:
    """The menu, listing only what this session can serve.

    Same rule as `_system_for`'s source enum, and it was re-learned the same
    way. Measured on the first Godot repair run: of twelve actions the agent
    spent, two went on `read_text` and `list_source`, neither of which a Godot
    build can ever answer. The model does not pick a tool because it is
    listed as possible -- it picks the one that best names what it wants, and
    then pays an action to discover the refusal. Offer what can be served.

    The refusals stay in the sessions regardless: this narrows what is
    suggested, it is not a substitute for saying no by name.
    """
    keys = set(supports or ACTIONS) | {'verdict'}
    return '\n'.join(f'  {k:<20} {v}' for k, v in sorted(ACTIONS.items())
                     if k in keys)






# THE CALL SENT NO SAMPLING PARAMETERS AT ALL, AND EVERY MODEL CALL IN THE LOOP
# COMES THROUGH HERE -- play, critique, and the spec digest.
#
# With none set, the endpoint picks: vLLM falls back to the model's own
# generation_config (temperature 1.0, top_p 0.95, top_k 20) and OpenRouter to
# the provider default, typically 1.0. The same omission in the benchmark judge
# produced a spread of 0.138 across three passes of one game's rubric and was
# read as the model being weak; pinned to 0 the three passes agreed exactly.
#
# Here it costs accuracy in the one place accuracy is a coordinate. Twelve
# samples of the SAME opening turn and the SAME frame, asking the play agent to
# click a PLAY button whose centre is (640,540) and whose box is x 483..798,
# y 505..575:
#
#     9 inside the button, clustered on (640,540)
#     2 nowhere near it -- (1157,448) and (1000,535)
#     1 malformed -- {"x": [640, 480]}, a list where a number belongs
#     x sd 168 px, range [639,1157];  y sd 27 px
#
# The mode is exactly right, so this was never a localisation problem: asked
# the same question outside the loop the model answers within 2 px. It is a
# heavy right tail on an otherwise sharp distribution, and that tail is what
# the sessions clicked -- (875,467), (995,483), (960,483) all have this shape.
# The critique then read "clicked, nothing happened" and reported a broken
# button, and repair spent rounds moving hitboxes that were never wrong.
#
# A coordinate is not creative writing. Overridable for a deliberate
# temperature sweep, and set to 0 everywhere else.
def _sampling() -> dict:
    t = os.environ.get('RSIGAME_AGENT_CHAT_TEMPERATURE')
    if t is not None and t.strip() == '':
        return {}                      # explicitly asked for the old behaviour
    return {'temperature': float(t) if t else 0.0, 'top_p': 1}


_SAMPLING = _sampling()


def _chat(messages, *, model: str):
    """One multimodal turn. Three tries, then give up on this turn.

    Three unrelated failures used to leave the identical trace, `(None, '')`:
    the request threw, the reply was not JSON, and the model spent its whole
    token budget reasoning without ever beginning the answer. The caller can
    only report that it got nothing back, so a game whose trace was never read
    is filed the same way whichever happened.

    The third one is also the one a retry cannot fix on its own -- the same
    ceiling produces the same result. Measured on glm-5.3-flash reading the
    12-frame reader payload: 8000/0, 7996/4 and 7350/650 reasoning-to-answer
    tokens, `content` empty all three times. So a length failure now raises the
    ceiling AND bounds the reasoning, and each try says which failure it was.

    stage1 was given this fix once already -- its `_MAX_TOKENS` note promises a
    reasoning-heavy call "will fail loudly with finish_reason=length now, not
    vanish" -- but that landed on `_chat_with_retry`, and this path is a
    different one that never received it.
    """
    from ..llm import (_BUDGET_CEILING, _MAX_TOKENS, _REASONING_CAP, _client,
                         _extract_json)

    client = _client(model)
    # Learning it the expensive way, once per call, is the price of not knowing
    # what endpoint we are on. When the caller DOES know -- an overnight run
    # pinned to OpenRouter, say -- RSIGAME_MODELS_CHAT_REASONING_CAP=1 skips the tuition: two
    # doomed requests and about five minutes per reading, every reading.
    extra = dict(_REASONING_CAP) if os.environ.get('RSIGAME_MODELS_CHAT_REASONING_CAP') else {}
    budget = _MAX_TOKENS
    for i in range(3):
        if i:
            time.sleep(3 * i)
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, max_completion_tokens=budget,
                **_SAMPLING, **({'extra_body': extra} if extra else {}))
        except Exception as exc:
            if extra:
                # The cap is an OpenRouter extension. An endpoint that refuses
                # it must cost us the cap, not the try.
                print(f'[chat] the reasoning cap was refused '
                      f'({type(exc).__name__}); retrying without it',
                      flush=True)
                extra = {}
            else:
                print(f'[chat] try {i + 1}: the request failed: '
                      f'{type(exc).__name__}: {str(exc)[:160]}', flush=True)
            continue
        choice = resp.choices[0]
        raw = choice.message.content or ''
        parsed = _extract_json(raw)
        if isinstance(parsed, dict):
            return parsed, raw
        rt = getattr(getattr(getattr(resp, 'usage', None),
                             'completion_tokens_details', None),
                     'reasoning_tokens', None)
        if choice.finish_reason == 'length' and not raw:
            budget = min(budget * 2, _BUDGET_CEILING)
            extra = dict(_REASONING_CAP)
            print(f'[chat] try {i + 1}: the whole budget went to reasoning '
                  f'({rt} tokens) and the answer never started; retrying with '
                  f'{budget} tokens and the reasoning capped', flush=True)
            continue
        print(f'[chat] try {i + 1}: the reply was not JSON ({len(raw)} chars, '
              f'finish_reason={choice.finish_reason}, reasoning={rt})',
              flush=True)
    return None, ''




