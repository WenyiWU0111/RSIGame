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
"""One director review of one case: counting, logging, drafts, the brief.

Used by the web backend for a person and by the model driver for a model, so
the two are counted the same way and write the same files:

    <root>/<run_id>/<game_id>/<checkpoint_id>/<human|model>/<director_id>/<session_id>/
        session.json              identity, hashes, budget, counts, times (atomic rewrite)
        interaction_events.jsonl  one line per event, appended and fsynced
        notes_draft.json          latest draft (atomic rewrite)
        draft_history.jsonl       every autosave
        development_brief.json    the submitted brief
        screenshots/              a frame before and after every counted action
        recording/                mp4 of the whole session, one file per launch
        work/                     the game copy and process logs

COUNTING (one policy for both directors):
    key press, from down to up however long it is held   1 action
    mouse click                                          1 action
    looking, waiting, moving the pointer                  0
    reset (relaunch the game)                             counts against the reset budget only
A key-down for a key already down (browser auto-repeat) is not a new press.
Once the budget is spent, presses and clicks are refused and logged as such;
releases always go through so no key is left held.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from .live import LiveGame, keysym

UI_VERSION = 'director-review/1'
# Free observations (wait / look) a model may take in one session. A person
# watches for free too; this only stops a runaway loop.
MODEL_STEP_CAP = 300
BRIEF_LIMITS = {'priorities': (1, 3), 'preserve': (0, 3)}


def _atomic_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def now_iso(t=None):
    return time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(t or time.time()))


class ReviewSession:
    def __init__(self, case: dict, root: Path, kind: str, director_id: str, session_id: str):
        self.case = case
        self.kind, self.director_id, self.session_id = kind, director_id, session_id
        self.dir = (Path(root) / case['run_id'] / case['game_id'] / case['checkpoint_id']
                    / kind / director_id / session_id)
        (self.dir / 'screenshots').mkdir(parents=True, exist_ok=True)
        self.events_path = self.dir / 'interaction_events.jsonl'
        self.lock = threading.RLock()
        self.down: set[str] = set()
        self.game: LiveGame | None = None
        meta_path = self.dir / 'session.json'
        if meta_path.is_file():
            self.meta = json.loads(meta_path.read_text())
        else:
            self.meta = {
                'session_id': session_id, 'director_type': kind, 'director_id': director_id,
                'case_id': case['case_id'], 'run_id': case['run_id'], 'game_id': case['game_id'],
                'checkpoint_id': case['checkpoint_id'], 'tree_src': case['tree_src'],
                'tree_hash': case['tree_hash'], 'spec_hash': case['spec_hash'],
                'saturation_summary': case['saturation_summary'],
                'budget': case['budget'], 'ui_version': UI_VERSION,
                'config_version': case['config_version'],
                'started_at': now_iso(), 'ended_at': None, 'submitted': False,
            }
        self._recount()
        self._save_meta()

    # ------------------------------------------------------------- records
    def _recount(self):
        """Counts come from the event log, so a restarted backend resumes exactly."""
        acts = resets = 0
        if self.events_path.is_file():
            for line in self.events_path.read_text().splitlines():
                try:
                    e = json.loads(line)
                except Exception:
                    continue              # a torn last line from a crash
                acts += e.get('counted', 0)
                resets += e['type'] == 'reset' and e.get('ok', False)
        self.meta['actions_used'], self.meta['resets_used'] = acts, resets

    def _save_meta(self):
        _atomic_json(self.dir / 'session.json', self.meta)

    def log(self, **e) -> dict:
        e = {'t': round(time.time(), 3), 'iso': now_iso(), **e}
        with self.lock, open(self.events_path, 'a') as f:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        return e

    def state(self) -> dict:
        m = self.meta
        return {'session_id': self.session_id, 'actions_used': m['actions_used'],
                'actions_max': m['budget']['actions'], 'resets_used': m['resets_used'],
                'resets_max': m['budget']['resets'], 'submitted': m['submitted'],
                'game_alive': bool(self.game and self.game.alive()),
                'held': sorted(self.down)}

    # ---------------------------------------------------------------- game
    def open_game(self):
        with self.lock:
            if self.game and self.game.alive():
                return
            self.game = LiveGame(self.case['tree_src'], self.dir / 'work', self.dir / 'recording',
                                 freezable=self.kind == 'model')
            self.game.start()
            self.down.clear()
            self.log(type='game_start', launch=self.game.launches,
                     resumed=self.meta['actions_used'] > 0 or self.meta['resets_used'] > 0)

    def close_game(self, reason: str):
        with self.lock:
            if self.game:
                for sym in list(self.down):
                    self.game.key(sym, False)
                self.game.stop()
                self.log(type='game_stop', reason=reason)
                self.game = None
            self.down.clear()

    def _shots(self, n: int, t: float, before: bytes):
        d = self.dir / 'screenshots'
        (d / f'a{n:03d}_before.jpg').write_bytes(before)

        def later():
            (d / f'a{n:03d}_after.jpg').write_bytes(self.game.frame_after(t + 0.4) if self.game else b'')
        threading.Timer(0.45, later).start()

    def _spend(self) -> tuple[bool, int]:
        m = self.meta
        if m['submitted'] or m['actions_used'] >= m['budget']['actions']:
            return False, m['actions_used']
        m['actions_used'] += 1
        self._save_meta()
        return True, m['actions_used']

    def key(self, code: str, down: bool, client_t=None) -> dict:
        with self.lock:
            sym = keysym(code)
            if sym is None or not (self.game and self.game.alive()):
                self.log(type='key_down' if down else 'key_up', code=code, ok=False, counted=0,
                         reason='unmapped key' if sym is None else 'game not running')
                return self.state()
            if not down:
                if sym in self.down:
                    self.down.discard(sym)
                    self.game.key(sym, False)
                    self.log(type='key_up', code=code, keysym=sym, ok=True, counted=0, client_t=client_t)
                return self.state()
            if sym in self.down:                   # auto-repeat
                return self.state()
            ok, n = self._spend()
            if not ok:
                self.log(type='key_down', code=code, keysym=sym, ok=False, counted=0,
                         reason='budget spent', client_t=client_t)
                return self.state()
            t, before = time.time(), self.game.latest
            self.down.add(sym)
            self.game.key(sym, True)
            self.log(type='key_down', code=code, keysym=sym, ok=True, counted=1, action_n=n,
                     client_t=client_t, shot=f'a{n:03d}')
            self._shots(n, t, before)
            return self.state()

    def click(self, x: int, y: int, button: str = 'left', client_t=None) -> dict:
        with self.lock:
            if not (self.game and self.game.alive()):
                self.log(type='click', x=x, y=y, ok=False, counted=0, reason='game not running')
                return self.state()
            ok, n = self._spend()
            if not ok:
                self.log(type='click', x=x, y=y, button=button, ok=False, counted=0,
                         reason='budget spent', client_t=client_t)
                return self.state()
            t, before = time.time(), self.game.latest
            self.game.click(x, y, {'left': '1', 'middle': '2', 'right': '3'}.get(button, '1'))
            self.log(type='click', x=x, y=y, button=button, ok=True, counted=1, action_n=n,
                     client_t=client_t, shot=f'a{n:03d}')
            self._shots(n, t, before)
            return self.state()

    # ------------------------------------------------------ model director
    def grab(self, name: str, t: float | None = None) -> str:
        """Save the frame on screen now (or the first one after t); return its path."""
        if not self.game:
            return ''
        if t:
            time.sleep(max(0.0, t - time.time()))
        img = self.game.frame_after(t) if t else self.game.latest
        path = self.dir / 'screenshots' / f'{name}.jpg'
        path.write_bytes(img)
        return str(path)

    def model_act(self, a: dict) -> dict:
        """One step of the model director, through the same counting as a person.

        press / hold / click cost one action; wait and look are free, as watching
        is for a person. A model cannot watch continuously, so every step returns
        frames: after an action, one at +0.3 s and one at +1.0 s (and one while a
        held key is still down); a wait returns the frame at its end.
        """
        from .live import keysym
        kind = a.get('action')
        note = str(a.get('note') or '')[:500]
        with open(self.dir / 'raw_model_outputs.jsonl', 'a') as f:
            f.write(json.dumps({'t': round(time.time(), 3), 'request': a}, ensure_ascii=False) + '\n')
        self.meta['model_steps'] = self.meta.get('model_steps', 0) + 1
        if self.meta['model_steps'] > MODEL_STEP_CAP:
            return {'error': f'step cap {MODEL_STEP_CAP} reached: submit the brief now', **self.state()}
        k = self.meta['model_steps']
        frames, err = [], None
        if kind != 'look' and self.game:
            self.game.resume()
            self.log(type='resume', step=k, counted=0)
        before_n = self.meta['actions_used']
        if kind in ('press', 'hold'):
            code = str(a.get('key') or '')
            if keysym(code) is None:
                err = f'unknown key {code!r}: use a browser KeyboardEvent.code such as Enter, Space, ArrowLeft, KeyA, Digit1'
            else:
                self.key(code, True)
                if self.meta['actions_used'] == before_n:
                    err = 'not sent: action budget spent or game not running'
                else:
                    t0 = time.time()
                    ms = 90 if kind == 'press' else max(100, min(int(a.get('ms') or 800), 5000))
                    if kind == 'hold':
                        time.sleep(ms / 2000)
                        frames.append(self.grab(f'm{k:03d}_during', t0 + ms / 2000))
                        time.sleep(ms / 2000)
                    else:
                        time.sleep(ms / 1000)
                    self.key(code, False)
                    t1 = time.time()
                    frames += [self.grab(f'm{k:03d}_t03', t1 + 0.3), self.grab(f'm{k:03d}_t10', t1 + 1.0)]
        elif kind == 'click':
            x, y = int(a.get('x', 0)), int(a.get('y', 0))
            if not (0 <= x < 1280 and 0 <= y < 720):
                err = 'x must be in [0,1280) and y in [0,720)'
            else:
                self.click(x, y, str(a.get('button') or 'left'))
                if self.meta['actions_used'] == before_n:
                    err = 'not sent: action budget spent or game not running'
                else:
                    t1 = time.time()
                    frames += [self.grab(f'm{k:03d}_t03', t1 + 0.3), self.grab(f'm{k:03d}_t10', t1 + 1.0)]
        elif kind == 'wait':
            ms = max(100, min(int(a.get('ms') or 1000), 5000))
            t1 = time.time() + ms / 1000
            frames.append(self.grab(f'm{k:03d}_wait', t1))
        elif kind == 'look':
            frames.append(self.grab(f'm{k:03d}_look'))
        else:
            err = f'unknown action {kind!r}: press, hold, click, wait, look'
        if self.game and not self.game.paused:
            self.game.pause()                 # frozen while the model thinks
            self.log(type='pause', step=k, counted=0)
        self._save_meta()
        self.log(type='model_step', step=k, action=kind, args={x: a[x] for x in ('key', 'ms', 'x', 'y', 'button') if x in a},
                 note=note, frames=[Path(f).name for f in frames if f], error=err, counted=0)
        return {'step': k, 'frames': [f for f in frames if f], **({'error': err} if err else {}), **self.state()}

    def move(self, x: int, y: int):
        if self.game and self.game.alive():
            self.game.move(x, y)

    def reset(self, reason: str = 'reviewer') -> dict:
        with self.lock:
            m = self.meta
            if m['submitted'] or m['resets_used'] >= m['budget']['resets']:
                self.log(type='reset', ok=False, reason='reset budget spent')
                return self.state()
            for sym in list(self.down):
                self.game.key(sym, False)
            self.down.clear()
            self.game.restart_game()
            m['resets_used'] += 1
            self._save_meta()
            self.log(type='reset', ok=True, reason=reason, resets_used=m['resets_used'])
            return self.state()

    def relaunch_after_crash(self) -> dict:
        """The game died on its own. Relaunching is not the director's reset."""
        with self.lock:
            self.down.clear()
            self.game.restart_game()
            self.log(type='game_crash_relaunch', ok=True, counted=0)
            return self.state()

    # --------------------------------------------------------------- brief
    def load_draft(self) -> dict:
        p = self.dir / 'notes_draft.json'
        return json.loads(p.read_text()) if p.is_file() else {}

    def save_draft(self, fields: dict) -> dict:
        d = {'case_id': self.case['case_id'], 'director_id': self.director_id,
             'session_id': self.session_id, 'last_edit': now_iso(), 'fields': fields}
        _atomic_json(self.dir / 'notes_draft.json', d)
        with open(self.dir / 'draft_history.jsonl', 'a') as f:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
        return d

    @staticmethod
    def validate(fields: dict) -> list[str]:
        errs = []
        if not str(fields.get('stage_objective') or '').strip():
            errs.append('Stage objective is required.')
        if not str(fields.get('why_now') or '').strip():
            errs.append('Why now is required.')
        for k, (lo, hi) in BRIEF_LIMITS.items():
            items = [x for x in (fields.get(k) or []) if str(x).strip()]
            if len(items) < lo:
                errs.append(f'At least {lo} {k[:-1] if k.endswith("s") else k} item is required.')
            if len(items) > hi:
                errs.append(f'At most {hi} {k} items.')
        return errs

    def submit(self, fields: dict, extra: dict | None = None) -> dict:
        errs = self.validate(fields)
        if errs:
            return {'ok': False, 'errors': errs}
        with self.lock:
            m = self.meta
            brief = {
                'director_type': self.kind, 'director_id': self.director_id,
                'session_id': self.session_id, 'case_id': self.case['case_id'],
                'game_id': self.case['game_id'], 'checkpoint_id': self.case['checkpoint_id'],
                'tree_hash': self.case['tree_hash'],
                'stage_objective': fields['stage_objective'].strip(),
                'why_now': fields['why_now'].strip(),
                'priorities': [x.strip() for x in fields.get('priorities') or [] if str(x).strip()],
                'preserve': [x.strip() for x in fields.get('preserve') or [] if str(x).strip()],
                'interaction_actions': m['actions_used'], 'resets': m['resets_used'],
                'submitted_at': now_iso(), **(extra or {}),
            }
            _atomic_json(self.dir / 'development_brief.json', brief)
            m['submitted'], m['ended_at'] = True, now_iso()
            self._save_meta()
            self.log(type='submit', ok=True)
        self.close_game('submitted')
        return {'ok': True, 'brief': brief}
