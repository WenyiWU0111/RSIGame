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
"""A Godot build running on a private Xvfb display, played live.

The same substrate as gamedoctor's GodotSession and gamecraft-bench's replay
(Xvfb + xdotool, their helpers for the display lock, window lookup and focus),
but driven by a person rather than a scripted action list: key-down and key-up
arrive separately, so a held key stays held for as long as it is held.

One ffmpeg process grabs the display and writes two outputs: the session
recording (mp4, one file per game launch) and a stream of JPEG frames kept in
memory, which the browser views and the event log snapshots around each action.

The build is COPIED before it is launched. Godot writes its import cache into
the project directory, and the checkpoint tree is experiment data.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from .. import paths

def _bench_dir() -> str:
    """Resolved on use: resolving at import would break --help and the tests."""
    return os.environ.get('GAMECRAFT_BENCH') or str(paths.bench())
FPS = 12
SETTLE_S = 2.0

# Browser KeyboardEvent.code -> X keysym, for the keys games use.
_CODE_KEYSYM = {
    'Space': 'space', 'Enter': 'Return', 'NumpadEnter': 'KP_Enter', 'Escape': 'Escape',
    'Tab': 'Tab', 'Backspace': 'BackSpace', 'Delete': 'Delete',
    'ArrowLeft': 'Left', 'ArrowRight': 'Right', 'ArrowUp': 'Up', 'ArrowDown': 'Down',
    'ShiftLeft': 'Shift_L', 'ShiftRight': 'Shift_R', 'ControlLeft': 'Control_L',
    'ControlRight': 'Control_R', 'AltLeft': 'Alt_L', 'AltRight': 'Alt_R',
    'Minus': 'minus', 'Equal': 'equal', 'Comma': 'comma', 'Period': 'period',
    'Slash': 'slash', 'Backslash': 'backslash', 'Semicolon': 'semicolon',
    'Quote': 'apostrophe', 'BracketLeft': 'bracketleft', 'BracketRight': 'bracketright',
    'Backquote': 'grave', 'Home': 'Home', 'End': 'End', 'PageUp': 'Prior', 'PageDown': 'Next',
}


def keysym(code: str) -> str | None:
    if code in _CODE_KEYSYM:
        return _CODE_KEYSYM[code]
    if code.startswith('Key') and len(code) == 4:
        return code[3].lower()
    if code.startswith('Digit') and len(code) == 6:
        return code[5]
    if code.startswith('Numpad') and code[6:].isdigit():
        return f'KP_{code[6:]}'
    if code.startswith('F') and code[1:].isdigit():
        return code
    return None


class LiveGame:
    def __init__(self, tree_src: str | Path, work: str | Path, rec_dir: str | Path,
                 viewport=(1280, 720), freezable: bool = False):
        self.tree_src = Path(tree_src)
        self.work = Path(work)
        self.game_dir = self.work / 'game'
        self.rec_dir = Path(rec_dir)
        self.w, self.h = viewport
        self.xvfb = self.godot = self.ffmpeg = None
        self.display = self.window = None
        self.env: dict = {}
        self.latest: bytes = b''
        self.latest_t = 0.0
        self.frame_cond = threading.Condition()
        self.launches = 0
        # FREEZABLE (model directors): the game advances by frames, not by the
        # wall clock (--fixed-fps 60, capped at 60 per real second), so stopping
        # the process while the model thinks and continuing it afterwards leaves
        # no jump in game time. A person plays unfrozen, on the wall clock.
        self.freezable = freezable
        self.paused = False
        self._lock = threading.RLock()
        bench_dir = _bench_dir()
        if bench_dir not in sys.path:
            sys.path.insert(0, bench_dir)
        from gamecraft_bench.verifier import replay as R
        self._R = R

    # ------------------------------------------------------------ lifecycle
    def prepare(self):
        """Copy the checkpoint tree and import it once, headless."""
        if not (self.game_dir / 'project.godot').is_file():
            self.work.mkdir(parents=True, exist_ok=True)
            shutil.copytree(self.tree_src, self.game_dir, symlinks=False,
                            ignore=shutil.ignore_patterns('.godot', '.qwen'))
        if not (self.game_dir / '.godot').is_dir():
            subprocess.run([self._R.cfg.GODOT_BIN or 'godot', '--headless', '--path',
                            str(self.game_dir), '--import', '--quit'],
                           capture_output=True, timeout=900)

    def start(self):
        with self._lock:
            self.prepare()
            self.rec_dir.mkdir(parents=True, exist_ok=True)
            log = open(self.work / 'xvfb.log', 'ab')
            self.xvfb, n = self._R._start_xvfb(log, viewport=(self.w, self.h))
            self.display = f':{n}'
            self.env = {**os.environ, 'DISPLAY': self.display,
                        'GODOT_SILENCE_ROOT_WARNING': '1', 'LP_NUM_THREADS': '1'}
            self._launch_game()
            self._start_capture()

    def _launch_game(self):
        cmd = [self._R.cfg.GODOT_BIN or 'godot', '--path', str(self.game_dir),
               '--display-driver', 'x11', '--rendering-driver', 'opengl3',
               '--audio-driver', 'Dummy', '--resolution', f'{self.w}x{self.h}',
               '--single-window']
        if self.freezable:
            cmd += ['--fixed-fps', '60', '--max-fps', '60']
        glog = open(self.work / 'godot.log', 'ab')
        self.godot = subprocess.Popen(cmd, env=self.env, stdout=glog, stderr=glog)
        time.sleep(SETTLE_S)
        if self.godot.poll() is not None:
            raise RuntimeError(f'godot exited before a window appeared (rc={self.godot.returncode})')
        self.window = self._R._find_godot_window(self.env, pid=self.godot.pid, timeout=15.0)
        self._R._xdotool(self.env, 'windowfocus', '--sync', self.window)
        self.launches += 1

    def _start_capture(self):
        seg = len(list(self.rec_dir.glob('session_*.mp4')))
        mp4 = self.rec_dir / f'session_{seg:02d}.mp4'
        cmd = ['ffmpeg', '-loglevel', 'error', '-f', 'x11grab', '-framerate', str(FPS),
               '-video_size', f'{self.w}x{self.h}', '-i', self.display,
               '-map', '0', '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p',
               str(mp4),
               '-map', '0', '-f', 'image2pipe', '-c:v', 'mjpeg', '-q:v', '5', 'pipe:1']
        self.ffmpeg = subprocess.Popen(cmd, env=self.env, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=open(self.work / 'ffmpeg.log', 'ab'))
        threading.Thread(target=self._read_frames, args=(self.ffmpeg,), daemon=True).start()

    def _read_frames(self, proc):
        buf = b''
        while True:
            chunk = proc.stdout.read(65536)
            if not chunk:
                return
            buf += chunk
            while True:
                s = buf.find(b'\xff\xd8')
                e = buf.find(b'\xff\xd9', s + 2) if s >= 0 else -1
                if s < 0 or e < 0:
                    break
                frame, buf = buf[s:e + 2], buf[e + 2:]
                with self.frame_cond:
                    self.latest, self.latest_t = frame, time.time()
                    self.frame_cond.notify_all()

    def pause(self):
        if self.freezable and self.alive() and not self.paused:
            os.kill(self.godot.pid, signal.SIGSTOP)
            self.paused = True

    def resume(self):
        if self.paused and self.godot and self.godot.poll() is None:
            os.kill(self.godot.pid, signal.SIGCONT)
        self.paused = False

    def restart_game(self):
        """A reset: the game process only. Display and recording keep running."""
        with self._lock:
            self.resume()
            self._stop_proc(self.godot)
            self._launch_game()

    def alive(self) -> bool:
        return bool(self.godot and self.godot.poll() is None)

    def stop(self):
        with self._lock:
            self.resume()
            if self.ffmpeg and self.ffmpeg.poll() is None:
                try:
                    self.ffmpeg.stdin.write(b'q')       # let it finish the mp4
                    self.ffmpeg.stdin.flush()
                    self.ffmpeg.wait(timeout=10)
                except Exception:
                    pass
            for p in (self.ffmpeg, self.godot, self.xvfb):
                self._stop_proc(p)

    @staticmethod
    def _stop_proc(p):
        if p is None or p.poll() is not None:
            return
        try:
            p.send_signal(signal.SIGTERM)
            p.wait(timeout=5)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass

    # --------------------------------------------------------------- input
    def _focus(self):
        self._R._xdotool(self.env, 'windowfocus', '--sync', self.window)

    def key(self, sym: str, down: bool):
        self._focus()
        self._R._xdotool(self.env, 'keydown' if down else 'keyup', '--window', self.window, sym)

    def click(self, x: int, y: int, button: str = '1'):
        self._focus()
        self._R._xdotool(self.env, 'mousemove', '--window', self.window, str(x), str(y),
                         'mousedown', button, 'mouseup', button)

    def move(self, x: int, y: int):
        self._R._xdotool(self.env, 'mousemove', '--window', self.window, str(x), str(y))

    def frame_after(self, t: float, timeout: float = 2.0) -> bytes:
        """The first frame captured after time t."""
        end = time.time() + timeout
        with self.frame_cond:
            while self.latest_t <= t and time.time() < end:
                self.frame_cond.wait(timeout=0.2)
            return self.latest
