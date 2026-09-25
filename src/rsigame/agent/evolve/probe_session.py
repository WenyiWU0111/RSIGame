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
"""A live session against a running build: start it, look at it, press keys.

`Probe` opens the game in a browser (or a Godot window), reads the display list
and the canvas text, and sends input. The verifier agent drives it; `explore()`
is what drives the verifier agent.

It used to live in `contract_jit.py`, a 1780-line module whose subject was a
just-in-time contract compiler with an oracle evaluator -- a design nothing in
this repo runs. This is the part of it that does.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

# Read every text object the scene graph is actually showing, with the geometry
# needed to tell a HUD label from an on-field callout. `visible` and `alpha` are
# both checked: a faded-out callout is still in the display list.
_CANVAS_TEXT_JS = r"""
() => {
  const g = window.__rlenv && window.__rlenv.game;
  if (!g || !g.scene) return {error: 'no game'};
  const out = [];
  const walk = (obj, scene, depth) => {
    if (!obj || depth > 6) return;
    const list = obj.list || obj.children?.list;
    if (list) { for (const c of list) walk(c, scene, depth + 1); return; }
    const t = obj.text;
    if (typeof t === 'string' && t.trim()) {
      const vis = obj.visible !== false && (obj.alpha === undefined || obj.alpha > 0.05);
      out.push({scene: scene, text: t.trim().slice(0, 120), visible: vis,
                alpha: obj.alpha === undefined ? 1 : Math.round(obj.alpha * 100) / 100,
                x: Math.round(obj.x || 0), y: Math.round(obj.y || 0),
                depth: obj.depth || 0,
                scrollFactor: obj.scrollFactorX === undefined ? 1 : obj.scrollFactorX});
    }
  };
  for (const s of g.scene.getScenes(true)) walk(s.children, s.scene.key, 0);
  return {texts: out, n_scenes: g.scene.getScenes(true).length};
}
"""


# The same walk as _CANVAS_TEXT_JS with the text filter taken off. Two of the
# three modalities that the fixed vocabulary could not express turn out to want
# the same thing: what objects are on screen, of what kind, where, and how big.
# Counting placed defenders and checking an overlay sits on its actor are the
# same read with different questions asked of it.
_DISPLAY_OBJECTS_JS = r"""
() => {
  const g = window.__rlenv && window.__rlenv.game;
  if (!g || !g.scene) return {error: 'no game'};
  const out = [];
  // ox/oy accumulate the container chain. A child of a Container carries
  // coordinates RELATIVE to it -- field_commander's three build buttons sit at
  // x = -220, 0, 220 inside a centred container, and clicking those numbers as
  // if they were world coordinates lands on the wrong tile without erroring.
  const walk = (obj, scene, depth, ox, oy) => {
    if (!obj || depth > 6) return;
    const list = obj.list || obj.children?.list;
    if (list) {
      const nx = ox + (obj.x || 0), ny = oy + (obj.y || 0);
      for (const c of list) walk(c, scene, depth + 1, nx, ny);
      return;
    }
    const vis = obj.visible !== false &&
                (obj.alpha === undefined || obj.alpha > 0.05);
    if (!vis) return;
    let w = obj.displayWidth, h = obj.displayHeight;
    if (w === undefined && obj.width !== undefined) { w = obj.width; h = obj.height; }
    // a Graphics circle carries no width; its radius lives on the command list
    let radius = null;
    const cmds = obj.commandBuffer;
    if (Array.isArray(cmds)) {
      for (let i = 0; i < cmds.length - 3; i++) {
        if (cmds[i] === 18 /* ARC */) { radius = cmds[i + 3]; break; }
      }
    }
    if (radius === null && obj.radius !== undefined) radius = obj.radius;
    // constructor.name is MANGLED by the production build -- every object in
    // field_commander came back as `initialize`. Duck-type instead, which
    // survives minification.
    let kind = 'object';
    if (typeof obj.text === 'string') kind = 'text';
    else if (Array.isArray(obj.commandBuffer)) kind = 'graphics';
    else if (obj.texture && obj.texture.key) kind = 'image';
    else if (obj.width !== undefined && obj.height !== undefined) kind = 'shape';
    out.push({
      scene: scene,
      type: kind,
      name: obj.name || '',
      texture: (obj.texture && obj.texture.key) || '',
      frame: (obj.frame && obj.frame.name) || '',
      x: Math.round((ox + (obj.x || 0)) * 10) / 10,
      y: Math.round((oy + (obj.y || 0)) * 10) / 10,
      local_x: Math.round((obj.x || 0) * 10) / 10,
      local_y: Math.round((obj.y || 0) * 10) / 10,
      w: w === undefined ? null : Math.round(w * 10) / 10,
      h: h === undefined ? null : Math.round(h * 10) / 10,
      radius: radius === null ? null : Math.round(radius * 10) / 10,
      alpha: obj.alpha === undefined ? 1 : Math.round(obj.alpha * 100) / 100,
      depth: obj.depth || 0,
      text: typeof obj.text === 'string' ? obj.text.trim().slice(0, 60) : ''
    });
  };
  for (const s of g.scene.getScenes(true)) walk(s.children, s.scene.key, 0, 0, 0);
  return {objects: out, n_scenes: g.scene.getScenes(true).length};
}
"""


_STATE_JS = r"""
() => {
  const g = window.__rlenv && window.__rlenv.game;
  if (!g) return null;
  try { return window.__rlenv.observe ? window.__rlenv.observe() : null; }
  catch (e) { return {error: String(e).slice(0, 100)}; }
}
"""


# Playwright's key vocabulary is exact: " " or "Space", never "SPACE". The
# compiler should not have to know that, and a contract should not fail because
# a model wrote a plausible synonym -- so the names are normalised here, once.
_KEY_ALIASES = {
    'space': ' ', 'spacebar': ' ', 'space_bar': ' ',
    'enter': 'Enter', 'return': 'Enter', 'esc': 'Escape', 'escape': 'Escape',
    'up': 'ArrowUp', 'down': 'ArrowDown', 'left': 'ArrowLeft',
    'right': 'ArrowRight', 'arrowup': 'ArrowUp', 'arrowdown': 'ArrowDown',
    'arrowleft': 'ArrowLeft', 'arrowright': 'ArrowRight',
    'shift': 'Shift', 'tab': 'Tab', 'ctrl': 'Control', 'control': 'Control',
    # the arrow GLYPHS -- the mechanic declarations in the fact sheets
    # spell arrow keys this way, and Playwright does not know them
    '\u2191': 'ArrowUp', '\u2193': 'ArrowDown',
    '\u2190': 'ArrowLeft', '\u2192': 'ArrowRight',
}


def normalise_key(k: str) -> str:
    if not k:
        return k
    if not k.strip():
        # A literal space IS the spacebar, and the compiler prompt shows it
        # that way. Stripping it to '' made Playwright reject the key and the
        # whole probe abort.
        return ' '
    low = k.strip().lower()
    if low in _KEY_ALIASES:
        return _KEY_ALIASES[low]
    if len(k.strip()) == 1:
        return k.strip().upper() if k.strip().isalpha() else k.strip()
    return k.strip()


def _claim_port(preferred: int) -> int:
    """Return a port nothing else is listening on, preferring `preferred`.

    A bind test rather than a connect test: a port with no listener refuses a
    connection, but so does one whose listener is momentarily busy, and those
    two must not be confused. If the preferred port cannot be bound the OS
    picks one -- the number carries no meaning, and a verification that
    refuses to run because a stale server holds an address is a worse failure
    than one that moves.
    """
    import socket
    for candidate in (preferred, 0):
        try:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', candidate))
                return sock.getsockname()[1]
        except OSError:
            continue
    return preferred


class Probe:
    """One live game, driven by contract steps. Records everything it reads."""

    def __init__(self, game_dir, port=7700, out_dir=None):
        self.game_dir = Path(game_dir)
        self.port = port
        self.out_dir = Path(out_dir) if out_dir else None
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.observations = []
        self._bindings = None            # static source read, parsed once
        self._binding_coverage = None
        self._srv = self._pw = self._browser = self.page = None

    def __enter__(self):
        """Everything after the first resource is acquired must be protected.

        Python calls `__exit__` only if `__enter__` RETURNED. Anything that
        raises partway through leaks whatever was already started -- and here
        that includes the Playwright driver, whose sync API keeps a
        process-wide loop. A leaked instance leaves that loop running, and the
        NEXT `sync_playwright().start()` in the same process dies with
        "Playwright Sync API inside the asyncio loop".

        That is not hypothetical. Shard 1 of the Track A batch hit
        `Page.goto: Timeout 25000ms exceeded` on its 20th edge, and its 23rd
        edge -- a different game, a different property -- died with the asyncio
        error. Two crashes, one cause: the timeout leaked and the next edge to
        reach a browser inherited it. Read as two separate faults, the second
        one is unexplainable.
        """
        from playwright.sync_api import sync_playwright
        try:
            dist = self.game_dir / 'dist'
            # THE PORT MUST BE OURS, AND THE SERVER MUST BE ALIVE ON IT.
            #
            # This was a deterministic port and a 1.5s sleep. When the port
            # was already held -- and these servers leak, six were up at once,
            # the oldest for seven hours -- `http.server` failed to bind, wrote
            # to a stderr pointed at DEVNULL, and exited. The sleep then
            # finished, the page loaded from WHOEVER WAS ALREADY THERE, and
            # `wait_for_function('!!window.__rlenv')` passed, because every
            # game in this corpus exposes that object.
            #
            # Nothing raised. The verifier played a different game end to end
            # and returned a verdict about this one. Measured on a hajimi edge
            # whose parent arm read "OXYGEN - CORAL - CHOICE | Bubbleblade
            # Reef", pressed keys, saw hp 3/3 and O2 59%, and concluded that
            # hajimi's defender-placement property did not apply -- while
            # :7802 served a platformer from a run that had ended seven hours
            # earlier.
            #
            # So the port is claimed here first, and one that cannot be
            # claimed is replaced rather than trusted. The liveness check
            # closes the remaining race: if something took it between the test
            # bind and the server's, `http.server` dies and this says so
            # instead of measuring a stranger.
            self.port = _claim_port(self.port)
            # LOOPBACK ONLY. `http.server` binds 0.0.0.0 when told nothing,
            # and this was the one server in the repo not saying otherwise --
            # every other one (d3_probe/*, demo_trace) passes --bind. The page
            # is fetched over 127.0.0.1 four lines down, so a public bind buys
            # the run nothing and costs it a game build exposed on every
            # interface. These servers also leak: six were up at once, the
            # oldest for seven hours, which is how long each one would have
            # been reachable. It also restores the port claim's meaning --
            # `_claim_port` tests 127.0.0.1, so a server on 0.0.0.0 was not
            # claiming quite the same thing it had tested.
            self._srv = subprocess.Popen(
                [sys.executable, '-m', 'http.server', str(self.port),
                 '--bind', '127.0.0.1', '-d', str(dist)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1.5)
            if self._srv.poll() is not None:
                raise RuntimeError(
                    f'the file server for {self.game_dir} exited immediately '
                    f'on port {self.port} (rc={self._srv.returncode}). '
                    f'Something else is on that port, and continuing would '
                    f'measure whatever game it is serving')
            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch(headless=True, args=[
                '--use-gl=angle', '--use-angle=swiftshader',
                '--enable-unsafe-swiftshader'])
            ctx = self._browser.new_context(
                viewport={'width': 960, 'height': 640})
            self.page = ctx.new_page()
            self.errors = []
            self.page.on('pageerror',
                         lambda e: self.errors.append(str(e)[:200]))
            self.page.goto(f'http://127.0.0.1:{self.port}/',
                           wait_until='load', timeout=25000)
            self.page.wait_for_function('() => !!window.__rlenv', timeout=25000)
        except BaseException:
            # release in reverse order, then re-raise unchanged: a boot failure
            # is still a boot failure and the caller must see it.
            self._release()
            raise
        return self

    def _release(self):
        """Close what was opened, tolerating anything that was never opened."""
        for closer in (lambda: self._browser.close(),
                       lambda: self._pw.stop(),
                       lambda: self._srv.kill()):
            try:
                closer()
            except Exception:
                pass
        self._browser = self._pw = self._srv = self.page = None

    def __exit__(self, *exc):
        self._release()

    # ------------------------------------------------------------ reads
    def read(self, what: str):
        if what == 'hud_text':
            try:
                return self.page.inner_text('body')[:1200]
            except Exception as e:
                return f'(hud read failed: {e})'
        if what == 'canvas_text':
            try:
                r = self.page.evaluate(_CANVAS_TEXT_JS) or {}
                return [t for t in (r.get('texts') or []) if t['visible']]
            except Exception as e:
                return [{'error': str(e)[:120]}]
        if what == 'display_objects':
            try:
                r = self.page.evaluate(_DISPLAY_OBJECTS_JS) or {}
                return r.get('objects') or []
            except Exception as e:
                return [{'error': str(e)[:120]}]
        if what == 'hud_numbers':
            # The HUD's numbers, keyed by the label in front of them. Reused
            # from the behavioural probe rather than re-parsed here, so a
            # contract and a probe never disagree about what GOLD means.
            try:
                from ..evidence.mechanic_inventory import hud_numbers
                return hud_numbers(self.page.inner_text('body')[:1200])
            except Exception as e:
                return {'error': str(e)[:120]}
        if what == 'bindings':
            # Static, and cached: the source does not change mid-run, and a
            # contract that reads it twice must see the same thing both times
            # or a before/after comparison would be measuring the parser.
            if self._bindings is None:
                try:
                    from ..evidence.input_binding import scan_bindings
                    bs, cov = scan_bindings(self.game_dir)
                    self._bindings = [
                        {'control': b.control, 'handler_kind': b.handler_kind,
                         'scope_kind': b.scope_kind, 'scope_name': b.scope_name,
                         'callback': getattr(b, 'callback', '') or '',
                         'registered_event': getattr(b, 'registered_event', '')}
                        for b in bs]
                    self._binding_coverage = cov
                except Exception as e:
                    self._bindings = [{'error': str(e)[:140]}]
            return self._bindings
        if what == 'scene_state':
            try:
                return self.page.evaluate(_STATE_JS)
            except Exception as e:
                return {'error': str(e)[:120]}
        return None

    def observe(self, label: str, sources: list) -> dict:
        o = {'label': label, 'at_ms': int(time.time() * 1000) % 10 ** 8}
        for src in sources:
            o[src] = self.read(src)
        if self.out_dir:
            try:
                p = self.out_dir / f'{label}.jpg'
                self.page.screenshot(path=str(p), type='jpeg', quality=72)
                o['frame'] = p.name
            except Exception:
                pass
        self.observations.append(o)
        return o

    # ------------------------------------------------------------ steps
    # Phrases that are a CALL TO ACTION to begin. Deliberately not "how to
    # play": shadow_courier and others keep that heading on screen beside a
    # live HUD after play has started, so treating it as a gate would refuse
    # to verify games that are running perfectly well. What must disappear is
    # the instruction to press something to start.
    _START_PROMPT = re.compile(
        r'(press|click|tap|hit)[^.\n]{0,24}\b(to\s+)?(start|begin|play|continue)\b'
        r'|(start|begin)\s+game\b|press\s+any\s+key',
        re.I)

    def _start_prompt(self) -> str:
        """The start prompt still on screen, or '' -- with the text that
        matched, so a wrong heuristic is visible in the trace rather than
        silently gating every game."""
        seen = []
        try:
            seen.append(self.page.inner_text('body')[:2000])
        except Exception:
            pass
        try:
            r = self.page.evaluate(_CANVAS_TEXT_JS) or {}
            seen.extend(t.get('text', '') for t in (r.get('texts') or [])
                        if t.get('visible'))
        except Exception:
            pass
        for chunk in seen:
            m = self._START_PROMPT.search(chunk or '')
            if m:
                return m.group(0)[:80]
        return ''

    def step(self, s: dict) -> dict:
        op = s.get('op')
        rec = {'op': op, 'arg': {k: v for k, v in s.items() if k != 'op'}}
        try:
            if op == 'boot_to_gameplay':
                # Pressing through a title screen and then ASSUMING you are in
                # play is how a game that never left its menu gets reported as
                # a game whose controls are dead. Confirm arrival by reading
                # the runtime, and say so when it cannot be confirmed.
                rec['attempts'] = []
                # Pressing through a title screen before the bundle has even
                # constructed the game is a race: the keypress lands on
                # nothing. Measured here -- one run failed all four attempts
                # while an identical run reached gameplay on the first.
                for _ in range(30):
                    try:
                        if self.page.evaluate(
                                '() => !!(window.__rlenv && window.__rlenv.game)'):
                            rec['runtime_ready'] = True
                            break
                    except Exception:
                        pass
                    self.page.wait_for_timeout(500)
                else:
                    rec['runtime_ready'] = False
                # `hasPlayer` ALONE IS NOT ARRIVAL. It goes true the moment the
                # level scene constructs the player -- which happens
                # UNDERNEATH an onboarding card that is intentional design and
                # is dismissed with ENTER. Confirming on it made every game
                # with a how-to overlay report `gameplay_confirmed: true` while
                # still showing "Press ENTER to start", so holding a movement
                # key moved nothing and the game was blamed for a dead control.
                #
                # Measured: of 20 edges that reached a repair, 12 had the agent
                # remove or bypass a start gate, and ALL FIVE promotions did.
                # Five repairs deleted a feature the game was supposed to have,
                # and each one was scored a success. The harness's assumption
                # became a requirement on the game -- the exact leak this
                # project catalogues.
                #
                # So arrival needs the player AND the start prompt gone. The
                # prompt is a call to action ("press ... to start"), not the
                # words HOW TO PLAY, which several games keep on screen beside
                # a live HUD after play has begun.
                for attempt in range(6):
                    for k in (' ', 'Enter'):
                        self.page.keyboard.press(k)
                        self.page.wait_for_timeout(400)
                    try:
                        self.page.mouse.click(480, 320)
                    except Exception:
                        pass
                    self.page.wait_for_timeout(900)
                    st = self.read('scene_state')
                    has_player = isinstance(st, dict) and (
                        st.get('hasPlayer') in (1, True) or
                        st.get('inPlay') in (1, True))
                    prompt = self._start_prompt()
                    rec['attempts'].append(
                        {'n': attempt,
                         'hasPlayer': (st or {}).get('hasPlayer')
                         if isinstance(st, dict) else None,
                         'start_prompt_still_shown': prompt})
                    if has_player and not prompt:
                        rec['gameplay_confirmed'] = True
                        break
                else:
                    rec['gameplay_confirmed'] = False
                    last = rec['attempts'][-1] if rec['attempts'] else {}
                    rec['why_not_confirmed'] = (
                        f'start prompt still shown: {last.get("start_prompt_still_shown")!r}'
                        if last.get('start_prompt_still_shown')
                        else 'no player object in the runtime state')
                rec['ok'] = True
                rec['boot_attempts_used'] = len(rec['attempts'])
            elif op == 'wait_ms':
                self.page.wait_for_timeout(int(s.get('ms', 1000)))
                rec['ok'] = True
            elif op == 'press_key':
                rec['normalised_key'] = normalise_key(s['key'])
                self.page.keyboard.press(rec['normalised_key'])
                self.page.wait_for_timeout(200)
                rec['ok'] = True
            elif op == 'hold_key':
                rec['normalised_key'] = normalise_key(s['key'])
                rec['ms'] = int(s.get('ms', 800))
                # Sample DURING the hold and keep the frame furthest from
                # where we started. A jump that takes off and lands inside the
                # window puts the actor back on the ground, so reading only
                # after the release records a completed jump and a dead key
                # identically -- measured on this exact game.
                before = self.read('scene_state')
                before = before if isinstance(before, dict) else {}
                # PER FIELD, not one frame. Picking the single frame that
                # deviates most overall picks it by whichever field happens to
                # swing hardest -- measured here: the frame chosen was the
                # launch instant, extreme in vy (-430) while py had not moved
                # yet, so an oracle reading py saw a working jump as no jump.
                envelope, peak_frame, peak_d = dict(before), None, -1.0
                self.page.keyboard.down(rec['normalised_key'])
                try:
                    for _ in range(10):
                        self.page.wait_for_timeout(max(1, rec['ms'] // 10))
                        now = self.read('scene_state')
                        if not isinstance(now, dict):
                            continue
                        worst = 0.0
                        for k, v in now.items():
                            b = before.get(k)
                            if not (isinstance(v, (int, float))
                                    and isinstance(b, (int, float))):
                                envelope.setdefault(k, v)
                                continue
                            d = abs(float(v) - float(b))
                            worst = max(worst, d)
                            cur = envelope.get(k, b)
                            if not isinstance(cur, (int, float)) or \
                                    d > abs(float(cur) - float(b)):
                                envelope[k] = v
                        if worst > peak_d:
                            peak_frame, peak_d = now, worst
                finally:
                    # Release even if the loop blew up; a stuck key would
                    # contaminate every observation after this one.
                    self.page.keyboard.up(rec['normalised_key'])
                self.page.wait_for_timeout(120)
                rec['peak_state'] = envelope
                rec['peak_state_is_per_field_envelope'] = True
                rec['peak_frame'] = peak_frame
                rec['peak_distance'] = round(peak_d, 3)
                rec['state_before_hold'] = before
                rec['ok'] = True
            elif op == 'click_centre':
                self.page.mouse.click(480, 320)
                self.page.wait_for_timeout(300)
                rec['ok'] = True
            elif op == 'click_object':
                objs = self.read('display_objects')
                pat = re.compile(s.get('match', '.'), re.I)
                hits = [o for o in objs if isinstance(o, dict) and pat.search(
                    ' '.join(str(o.get(f, '')) for f in
                             ('name', 'texture', 'type', 'text')))]
                rec['matched'] = len(hits)
                rec['candidates'] = [{k: o.get(k) for k in
                                      ('name', 'texture', 'type', 'x', 'y')}
                                     for o in hits[:4]]
                nth = int(s.get('nth', 0))
                if not hits or nth >= len(hits):
                    # Not an error in the probe: the thing to click was not on
                    # screen. Say which, so the recompile has something to use.
                    rec['ok'] = False
                    rec['error'] = (f'no visible object matches '
                                    f'{s.get("match")!r}'
                                    if not hits else
                                    f'only {len(hits)} match {s.get("match")!r}, '
                                    f'asked for index {nth}')
                else:
                    o = hits[nth]
                    gx = float(o.get('x') or 0) + float(s.get('dx', 0))
                    gy = float(o.get('y') or 0) + float(s.get('dy', 0))
                    # Game coordinates are not viewport coordinates: the canvas
                    # is letterboxed and scaled. Convert through the live rect
                    # rather than assuming they coincide.
                    vp = self.page.evaluate(
                        """() => {const c = document.querySelector('canvas');
                           if (!c) return null; const r = c.getBoundingClientRect();
                           return {x: r.x, y: r.y, w: r.width, h: r.height,
                                   cw: c.width, ch: c.height};}""")
                    if vp and vp.get('cw'):
                        vx = vp['x'] + gx * vp['w'] / vp['cw']
                        vy = vp['y'] + gy * vp['h'] / vp['ch']
                    else:
                        vx, vy = gx, gy
                    rec['game_xy'] = [round(gx, 1), round(gy, 1)]
                    rec['viewport_xy'] = [round(vx, 1), round(vy, 1)]
                    self.page.mouse.click(vx, vy)
                    self.page.wait_for_timeout(int(s.get('settle_ms', 400)))
                    rec['ok'] = True
            elif op == 'wait_for_hud_text':
                pat = re.compile(s['pattern'], re.I)
                deadline = time.time() + int(s.get('timeout_ms', 60000)) / 1000
                seen = False
                while time.time() < deadline:
                    if pat.search(self.read('hud_text') or ''):
                        seen = True
                        break
                    self.page.wait_for_timeout(1000)
                rec['ok'] = seen
                rec['note'] = ('pattern appeared' if seen else
                               'pattern never appeared within the timeout -- the '
                               'precondition was not reached, which is not the '
                               'same as the property holding')
            else:
                rec['ok'] = False
                rec['note'] = f'unknown primitive {op!r}'
        except Exception as e:
            rec['ok'] = False
            rec['error'] = str(e)[:160]
        return rec


