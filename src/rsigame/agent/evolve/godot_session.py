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
"""The verifier's session contract, over a live Godot process instead of a page.

`verifier_agent._Session` drives a Phaser build through Playwright and reads
its runtime through `window.__rlenv`. Godot exports no such bridge, so the
question this module answers is not "how do we fake one" but "which of the
eight primitives survive, and what do the others say instead".

WHAT CARRIES OVER unchanged in meaning:

    boot press_key hold_key click wait screenshot

driven by xdotool against a Godot process on a private Xvfb display, the same
substrate gamecraft-bench's own replay uses. Verified on hajimi: a click
spent the right amount of Kibble, a cat appeared on the slot it was aimed at,
SPACE started the wave, and the enemy queue advanced.

WHAT REFUSES, and why refusing is the point:

    read_state          no runtime bridge exists here
    read_scene_fields   likewise
    read_text           the HUD is DRAWN, not text in a DOM

Each returns an `error` naming what is unavailable, in the shape
`read_bindings` already uses in the Phaser session -- and for the same
recorded reason. An empty result must not look like an absence: a
`read_state` that answered `{}` would let an agent conclude the game exposes
no score, when what happened is that nobody asked the game.

FRAMES REPLACE STATE, AND THE MATCHED INTERVAL SURVIVES. The comparison the
verifier leans on hardest -- do nothing for N ms, then do something for the
same N ms, and see what differs -- does not need a state dict. It needs two
readings of the same kind, taken the same way, N ms apart. So `wait` and
`hold_key` capture a frame before and after and hand back both. hajimi's HUD
renders SANITY, KIBBLE, WAVE and the enemy count as large pixel text, so a
multimodal read of those two frames answers exactly what `scene_state` used
to, for the quantities that matter.

WHAT IS GENUINELY LOST: a numeric field nobody drew. Player velocity, an
internal flag, a scene name -- these were readable through `__rlenv` and are
not readable here. Properties resting on them are unmeasurable on Godot
through this adapter, and the honest form of that is an INCONCLUSIVE with a
reason, which is what the refusals above produce.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from ... import paths

# Off by default: turning it on changes what the agent is shown, so it is an
# experimental variable and not a silent upgrade. Phase 3 turns it on for both
# arms at once.
HOLD_DURING_FRAME = os.environ.get('RSIGAME_AGENT_HOLD_DURING_FRAME', '0') == '1'

# gamecraft-bench owns the awkward parts of this substrate -- locking a free
# display number, finding the window by pid, focusing it with no window
# manager running -- and each has a comment there explaining a failure that
# is not obvious. Imported rather than reimplemented.
# WHERE THE BENCH'S REPLAY MODULE LIVES. This used to be a hardcoded absolute path,
# an absolute path from the machine it was written on, and on any other box the
# session dies with `ModuleNotFoundError: No module named 'gamecraft_bench'` --
# which `explore` records as `session_error`, i.e. as a fact about the GAME
# rather than about this constant. Measured here: every Godot play session
# returned 0 steps and 0 frames until it was found.
#
# environment only:this used to probe three candidate directories(one of them another machine's absolute path),
# falling back to the last one, so a wrong path raised nothing, it just replayed 0 frames. config fills it now.
GAMECRAFT_BENCH = os.environ.get('GAMECRAFT_BENCH', '')

# Actions a Godot build can actually answer. Anything else is refused by
# name rather than silently returning nothing.
# `boot_scenario` belongs here and not only in PLAY_ACTIONS: `_opening` offers
# the intersection of the two, so an action missing from this tuple is never
# shown to the agent no matter what else is switched on. Measured on the first
# E2 launch -- seven rounds, the switches all on, the ledger rendering correctly
# in a unit test, and not one `boot_scenario` call, because the menu the agent
# actually saw never contained it.
DRIVABLE = ('boot', 'boot_scenario', 'press_key', 'hold_key', 'click', 'wait',
            'screenshot')

# The verifier's key vocabulary is Playwright's. gamecraft-bench's
# `_normalize_keycode` speaks the trace format's instead, and the two differ
# on exactly the keys a game uses most: Playwright names the space bar `' '`
# and the arrows `ArrowLeft`. Measured on the first Godot verification -- the
# agent sent `' '`, got `unknown keycode`, and spent one of its twelve
# actions rediscovering that `'Space'` works. Translating here costs nothing
# and lets the agent speak one language across both engines.
_KEY_ALIASES = {
    'SPACEBAR': 'SPACE',
    'ARROWLEFT': 'LEFT', 'ARROWRIGHT': 'RIGHT',
    'ARROWUP': 'UP', 'ARROWDOWN': 'DOWN',
    'ESC': 'ESCAPE', 'RETURN': 'ENTER',
}


def _key(raw) -> str:
    """A Playwright key name, as gamecraft-bench's table expects it."""
    k = str(raw)
    if k == ' ':
        return 'SPACE'
    return _KEY_ALIASES.get(k.strip().upper(), k.strip())


# Said on every bracketed action. The warning is the load-bearing half: a
# verdict was lost to attributing a delta to a click when a wave bonus had
# landed in the same window, so the pair is offered together with the reason
# it can still mislead.
_BRACKET_NOTE = (
    'the HUD is legible in both frames; compare them. A change between them '
    'is not necessarily CAUSED by this action -- the game may move the same '
    'quantity on its own (a timer, a wave bonus, a kill reward). If a number '
    'must be attributed to the action, check the frames for any other event '
    'in the same moment before concluding.')

UNAVAILABLE = {
    'read_state':
        'this runtime exposes no state object -- Godot has no `__rlenv` '
        'bridge. This says NOTHING about whether the game tracks the value '
        'you asked for; it means nobody can ask it here. Read the HUD from a '
        'screenshot instead.',
    'read_scene_fields':
        'scene fields cannot be read from a Godot process here. Use '
        'screenshots and what the game draws.',
    'read_text':
        'this game DRAWS its text rather than exposing it -- there is no DOM '
        'to read. Take a screenshot: the HUD is legible in the frame.',
    'read_bindings':
        'source-level binding scan is not implemented for GDScript. Absence '
        'here is not evidence that the game binds no controls.',
}


class GodotSession:
    """Same surface as `verifier_agent._Session`, different substrate.

    `port` is accepted and ignored: it names a local HTTP server in the
    Phaser session, and there is none here. Kept in the signature so the
    session can be substituted without the caller knowing which engine it
    got.
    """

    # What the verifier is OFFERED, as opposed to what it is refused after
    # asking. Measured on the first Godot repair run: the agent spent 2 of
    # its 12 actions on `read_text` and `list_source`, both of which can only
    # ever refuse here. A menu item that cannot be served is not a harmless
    # extra -- the model picks whatever best names what it wants, and pays an
    # action to find out. UNAVAILABLE stays as the safety net for anything
    # asked anyway.
    SUPPORTS = DRIVABLE

    def __init__(self, src, port=None, out_dir=None, *,
                 viewport=(1280, 720), scenario='', settle=2.0,
                 available_scenarios=None):
        self.src = Path(src)
        self.out_dir = Path(out_dir) if out_dir else None
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.w, self.h = viewport
        self.scenario = scenario
        # Set by the caller when it knows what the game declares; empty means
        # "unknown", and then any id is passed through to the game rather than
        # refused here -- a session must not decide a state does not exist.
        self.available_scenarios = tuple(available_scenarios or ())
        self.settle = settle
        self.actions, self.observations, self.frames = [], [], []
        self.evidence_pathway = ''
        self._procs = []
        self._n = 0
        self.env = None
        self.window = None
        self.display = None
        self.godot = None

    # ------------------------------------------------------------ lifecycle
    def __enter__(self):
        import sys
        if GAMECRAFT_BENCH not in sys.path:
            sys.path.insert(0, GAMECRAFT_BENCH)
        from gamecraft_bench.verifier import replay as R
        self._R = R

        log = open((self.out_dir or Path('/tmp')) / 'xvfb.log', 'wb')
        xvfb, n = R._start_xvfb(log, viewport=(self.w, self.h))
        self._procs.append(xvfb)
        self.display = f':{n}'
        self.env = {**os.environ, 'DISPLAY': self.display,
                    'GODOT_SILENCE_ROOT_WARNING': '1', 'LP_NUM_THREADS': '1'}
        self._start_game()
        return self

    def _start_game(self):
        """Launch the project on the display this session already owns.

        Separate from `__enter__` because a scenario switch relaunches the GAME
        and nothing else: the Xvfb, the display and the frame counter survive,
        so a session that jumps between states keeps one continuous trace rather
        than becoming several.
        """
        R = self._R
        cmd = [R.cfg.GODOT_BIN or 'godot', '--path', str(self.src),
               '--display-driver', 'x11', '--rendering-driver', 'opengl3',
               '--audio-driver', 'Dummy',
               '--resolution', f'{self.w}x{self.h}', '--single-window']
        if self.scenario:
            cmd += ['--', '--scenario', self.scenario]
        glog = open((self.out_dir or Path('/tmp')) / 'godot.log', 'ab')
        self.godot = subprocess.Popen(cmd, env=self.env,
                                      stdout=glog, stderr=glog)
        self._procs.append(self.godot)

        time.sleep(self.settle)
        if self.godot.poll() is not None:
            raise RuntimeError(
                f'godot exited before a window appeared (rc='
                f'{self.godot.returncode}); this is an infrastructure '
                f'failure, not a verdict')
        self.window = R._find_godot_window(self.env, pid=self.godot.pid,
                                           timeout=15.0)
        # No window manager runs on this display, so nobody brokers focus and
        # xdotool's --window flag on key events silently does nothing without
        # this. replay.py carries the same call and the same explanation.
        R._xdotool(self.env, 'windowfocus', '--sync', self.window)

    def __exit__(self, *exc):
        for p in reversed(self._procs):
            try:
                p.terminate()
                p.wait(timeout=5)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        return False

    # -------------------------------------------------------------- frames
    def _grab(self, label=''):
        self._n += 1
        d = self.out_dir or Path('/tmp')
        path = d / f'godot-{self._n:03d}{("-" + label) if label else ""}.png'
        subprocess.run(
            ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'x11grab',
             '-video_size', f'{self.w}x{self.h}', '-i', self.display,
             '-frames:v', '1', str(path)],
            env=self.env, check=False,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if path.exists():
            self.frames.append(str(path))
            return str(path)
        return ''

    def closing_snapshot(self) -> dict:
        """One last look, taken after the verdict so neither affects the other."""
        try:
            return {'frame': self._grab('closing'),
                    'note': 'Godot exposes no readable state; the frame is '
                            'the whole of the closing snapshot'}
        except Exception as exc:
            return {'error': f'{type(exc).__name__}: {str(exc)[:150]}'}

    # ------------------------------------------------------------- the loop
    def do(self, action: str, args: dict) -> dict:
        rec = {'action': action, 'args': args, 'n': len(self.actions) + 1}
        try:
            rec.update(self._dispatch(action, args or {}))
        except Exception as e:
            # Reported to the agent, never raised: it may have another route
            # to the same answer, and a crashed verification is worth less
            # than a refused one.
            rec['error'] = f'{type(e).__name__}: {str(e)[:160]}'
        self.actions.append(rec)
        if 'observed' in rec:
            self.observations.append({'after_action': rec['n'],
                                      'action': action,
                                      'value': rec['observed']})
        return rec

    def _refocus(self):
        """Re-assert focus before every input, not once per session.

        No window manager runs on this display, so nobody brokers focus and
        nobody restores it. `windowfocus --sync` at session start is enough
        right up until something takes focus away -- and from that moment
        EVERY key and button silently does nothing while `mousemove` keeps
        working, because pointer position is an X-server fact and delivery is
        not. That failure is invisible in a screenshot: the cursor sits on the
        right pixel and the game ignores it.

        Which is the exact shape of one unexplained observation: a placement
        click on an eligible slot that moved the cursor there and produced no
        defender, no popup and no deduction, on a build where reading
        `handle_click` and `add_tower` line by line shows no branch that can
        refuse it. Nineteen out of nineteen scripted clicks placed correctly
        afterwards -- across a quiet state and across the whole wave window --
        so it is NOT reproduced and the cause is NOT established. This is
        insurance against the symptom, not a fix for a diagnosed bug, and it
        is worth taking on that basis alone: a dropped input reads as "the
        game did not respond", which is a FAIL, and a verifier that can be
        talked into false FAILs by its own substrate is worse than one that
        crashes.
        """
        try:
            R = self._R
            R._xdotool(self.env, 'windowfocus', '--sync', self.window)
        except Exception:
            # Best-effort. A refocus that fails is not itself a finding, and
            # raising here would turn a plain action into an INFRA error.
            pass

    def _dispatch(self, action: str, args: dict) -> dict:
        R = self._R
        if action in UNAVAILABLE:
            return {'error': UNAVAILABLE[action]}
        if action in ('press_key', 'hold_key', 'click'):
            self._refocus()

        if action == 'boot':
            return {'ok': True, 'frame': self._grab('boot'),
                    'note': 'the process is up and its window has focus; '
                            'play may not have started'}

        # RESTART THE GAME IN A NAMED STATE.
        #
        # The spec requires a game launched with `-- --scenario <id>` to skip
        # menus, set that state up deterministically and start taking input. So
        # the states a game declares are reachable on demand, and a short
        # session no longer has to arrive at a late one by playing there -- the
        # usual reason a whole half of a game goes unexamined round after round.
        #
        # This is a RELAUNCH, not a jump: whatever the session did before is
        # gone, exactly as if the game had just been started. Said plainly in
        # the note, because an agent that thinks its earlier progress survived
        # will misread the frame it gets back.
        if action == 'boot_scenario':
            sid = str((args or {}).get('id') or '').strip()
            if not sid:
                return {'ok': False, 'error': 'boot_scenario needs an `id`'}
            if self.available_scenarios and sid not in self.available_scenarios:
                return {'ok': False,
                        'error': f'this game declares no scenario {sid!r}; it '
                                 f'has {", ".join(self.available_scenarios)}'}
            prev = self.godot
            try:
                prev.terminate(); prev.wait(timeout=5)
            except Exception:
                try:
                    prev.kill()
                except Exception:
                    pass
            if prev in self._procs:
                self._procs.remove(prev)
            self.scenario = sid
            self._start_game()
            return {'ok': True, 'scenario': sid,
                    'frame': self._grab(f'scenario-{sid}'),
                    'note': f'the game was RESTARTED in scenario {sid!r}. '
                            f'Nothing you did before this survives -- this is a '
                            f'fresh launch that begins in that state.'}

        if action == 'screenshot':
            return {'ok': True, 'frame': self._grab('shot')}

        # `press_key` and `click` also bracket themselves, and the reason is a
        # measured wrong verdict rather than symmetry.
        #
        # hajimi, 2026-08-25, on a build byte-identical to one that had PASSed
        # the same property: the agent placed a 35-cost defender, read KIBBLE
        # 125 in an earlier frame and 110 in the frame after the click, and
        # FAILed the game for deducting 15. The arithmetic was 125 - 35 + 20:
        # wave 1 ended inside the gap between those two readings and paid its
        # +20 bonus (Main.gd, `kibble += reward`). Every step of that was
        # sound except the gap, which was several seconds long and contained
        # an event the agent had no way to know about.
        #
        # A delta is only attributable to an action if nothing else can move
        # the quantity in between. Bracketing does not make that true -- a
        # wave could still turn inside 0.35s -- but it shrinks the window from
        # seconds to sub-second and, more to the point, hands over two
        # readings TAKEN AROUND THE ACTION instead of one reading and a
        # memory. Same reason `wait` returns a pair.
        if action == 'press_key':
            k = R._normalize_keycode(_key(args.get('key', '')))
            before = self._grab('key_before')
            R._xdotool(self.env, 'key', '--window', self.window, k)
            time.sleep(0.3)
            after = self._grab(f'key_{k}')
            return {'ok': True, 'key': k, 'frame': after,
                    'observed': {'before_frame': before, 'after_frame': after,
                                 'read_how': _BRACKET_NOTE}}

        if action == 'click':
            x, y = int(args.get('x', self.w // 2)), int(args.get('y', self.h // 2))
            b = {'left': '1', 'right': '3'}.get(
                str(args.get('button', 'left')).lower(), '1')
            before = self._grab('click_before')
            # One invocation, chaining move + down + up. Split into two calls
            # every click is swallowed -- measured on hajimi, where the cursor
            # reached the right pixel and the title screen never advanced.
            R._xdotool(self.env, 'mousemove', '--window', self.window,
                       str(x), str(y), 'mousedown', b, 'mouseup', b)
            time.sleep(0.35)
            after = self._grab(f'click_{x}_{y}')
            return {'ok': True, 'x': x, 'y': y, 'frame': after,
                    'observed': {'before_frame': before, 'after_frame': after,
                                 'read_how': _BRACKET_NOTE}}

        # `wait` and `hold_key` return a BEFORE and an AFTER frame. That pair
        # is what the matched-interval comparison actually needs -- two
        # readings of the same kind, the same interval apart -- and it does
        # not need a state dict to be one.
        if action == 'wait':
            ms = max(50, min(int(args.get('ms', 500)), 5000))
            before = self._grab('wait_before')
            time.sleep(ms / 1000)
            after = self._grab('wait_after')
            return {'ok': True, 'ms': ms,
                    'observed': {'before_frame': before, 'after_frame': after,
                                 'read_how': _BRACKET_NOTE}}

        if action == 'hold_key':
            k = R._normalize_keycode(_key(args.get('key', '')))
            ms = max(50, min(int(args.get('ms', 800)), 5000))
            before = self._grab('hold_before')
            R._xdotool(self.env, 'keydown', '--window', self.window, k)
            time.sleep(ms / 1000)
            # A FRAME WHILE THE KEY IS STILL DOWN.
            #
            # Without it every frame this action has ever returned was taken
            # AFTER `keyup`, so anything that only exists while a key is held
            # -- a held-state readout, a pressed highlight, sustained
            # acceleration -- had already reset by the time the agent saw it.
            # Measured on platformer-ink-trail: the game draws
            # `D=... actR=...` on screen, the after-frame reported both false
            # for eight consecutive rounds, and the loop concluded the key
            # never registered. Sending the same keydown by hand and grabbing
            # before the release shows `D=true actR=true` -- and `vel=0`, which
            # is the real defect and is one layer further in. Three rounds of
            # repair went into the input map instead.
            during = self._grab('hold_during') if HOLD_DURING_FRAME else None
            R._xdotool(self.env, 'keyup', '--window', self.window, k)
            time.sleep(0.2)
            after = self._grab('hold_after')
            obs = {'before_frame': before, 'after_frame': after,
                   'read_how': _BRACKET_NOTE}
            if during:
                obs['during_frame'] = during
                obs['read_how'] = (
                    'THREE frames: before the key went down, WHILE it was '
                    'held, and after it was released. Anything that only '
                    'shows while a key is down appears in the middle frame '
                    'and nowhere else -- do not conclude from the last frame '
                    'that an input never registered. ' + _BRACKET_NOTE)
            return {'ok': True, 'key': k, 'ms': ms, 'observed': obs}

        return {'error': f'{action!r} is not available on a Godot build. '
                         f'Drivable here: {sorted(DRIVABLE)}'}
