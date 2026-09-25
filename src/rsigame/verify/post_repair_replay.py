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
"""Replay the shipped demos on a tree and package what happened, for a reader.

WHY THIS EXISTS. After the repair agent runs, the loop only asks "does it
still compile". Three regressions survived fifteen rounds that way: a
projectile drawn at 10368px that buried the whole screen, a guard flag set in
one of nine scenario branches that swallowed every key press, and a marking
control that grew a two-step confirmation the demo could not perform. None is
visible in a diff. All three are obvious in one replay.

WHY THERE ARE NO NUMBERS HERE. The first design computed per-event response
magnitudes and screen statistics and compared them against thresholds. Two
measurements killed it. First, the replay is not deterministic -- the same
tree replayed twice shares no identical frame -- so any statistic inherits
that noise. Second, and decisively: on horror-tape-archive's G0, a MARK
ANOMALY click that genuinely works (it prints CONFIRMED and scores a point)
moves the whole-screen mean by 1.2 against an ambient 0.8. A threshold would
call that working click dead. The change is small because only a caption
changed -- which a reader sees instantly and an average cannot.

So this module measures nothing. It replays, picks the frames that bracket
each input event, labels them, and hands them to a verifier that can look.

FRAME ALIGNMENT IS MEASURED, NOT ASSUMED. `replay_trace` anchors its event
clock on ffmpeg's capture start, so a trace frame does not land on the mp4
frame of the same index. Measured across five games and fifteen events
spanning one recording from frame 24 to frame 506, the effect of an event at
trace frame N first appears at mp4 frame N+7 to N+10, with no drift over the
recording. `LAG` is that measurement; `BEFORE`/`AFTER` sit clear of both ends
of its range.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get('GAMECRAFT_BENCH')
                or str(RUN.parent / 'gamecraft-bench'))

# Measured: an event at trace frame N shows up at mp4 frame N+7..N+10.
LAG = 9
BEFORE = 3        # N+3 is before the earliest observed response (N+7)
AFTER = (14, 30)  # first is past the latest observed response (N+10)

INPUT_TYPES = ('mouse_click', 'key_press', 'key_down', 'key_up', 'mouse_move')

# A SECOND, UNIFORM SAMPLE OF THE SAME RECORDING.
#
# The event-anchored frames above answer "did this input get a response". They
# cannot answer "what did this build look like over the run", because they are
# all clustered a few frames either side of an input and say nothing about the
# gaps -- and because the anchors are trace frames, two builds sample the same
# MOMENTS whether or not they are in the same PLACE. Measured on
# battle-critter-clash, 29% of r12's recording is black: the build had not
# started drawing, so every anchored frame of that stretch shows a black
# rectangle, and a reader comparing it against a lit r15 frame at the same
# anchor learns only that one of them was asleep.
#
# So take the whole recording evenly as well. It costs one extra handful of
# PNGs in the ffmpeg pass that was already running -- `prune()` deletes the
# mp4 straight afterwards, and re-cutting it later is not possible -- and it
# is what the value estimator and the official evaluator both read.
N_UNIFORM = int(os.environ.get('RSIGAME_REPLAY_N_UNIFORM') or 16)


def _describe(ev: dict) -> str:
    """The event as a reader needs it, not as JSON."""
    t = ev.get('type')
    if t == 'mouse_click':
        return f"click {ev.get('button', 'left')} at ({ev.get('x')}, {ev.get('y')})"
    if t in ('key_press', 'key_down', 'key_up'):
        return f"{t.replace('_', ' ')} {ev.get('keycode')}"
    if t == 'mouse_move':
        return f"move to ({ev.get('x')}, {ev.get('y')})"
    return t or 'unknown'


def _evenly(n: int, k: int) -> list[int]:
    """k frame indices spread from the first to the last."""
    if n <= 0 or k <= 0:
        return []
    if k >= n:
        return list(range(n))
    if k == 1:
        return [n // 2]
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def _grab(mp4: Path, wanted: list[int], out_dir: Path, tag: str) -> dict:
    """Pull just the frames asked for, in one ffmpeg pass.

    Extracting every frame of a 20-second recording costs ~120 MB per demo and
    all but a handful are never looked at.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = sorted({n for n in wanted if n >= 0})
    if not wanted:
        return {}
    expr = '+'.join(f'eq(n\\,{n})' for n in wanted)
    tmp = out_dir / f'_{tag}_%04d.png'
    subprocess.run(
        ['ffmpeg', '-y', '-loglevel', 'error', '-i', str(mp4),
         '-vf', f"select='{expr}'", '-vsync', '0', str(tmp)],
        check=False, timeout=180)
    got = sorted(out_dir.glob(f'_{tag}_*.png'))
    # select emits the kept frames in order, so the k-th file is wanted[k].
    # A frame past the end of the recording simply does not arrive, which is
    # itself worth knowing -- a demo that ended early has fewer files.
    out = {}
    for n, p in zip(wanted, got):
        dst = out_dir / f'{tag}_f{n:05d}.png'
        p.replace(dst)
        out[n] = dst
    for leftover in out_dir.glob(f'_{tag}_*.png'):
        leftover.unlink(missing_ok=True)
    return out


# ---------------------------------------------------------------- Phaser (web)
#
# A PHASER BUILD IS REPLAYED IN A BROWSER, and two things about its recording
# differ from Godot's, both measured:
#
#   1. Playwright records at 25 fps, not 30. `LAG`/`BEFORE`/`AFTER` above are
#      mp4 frame counts on a 30 fps Godot capture, so used on a web recording
#      every anchor drifts later by a sixth of its position -- three seconds
#      by the end of a 20 s demo -- and `last` points past the end.
#   2. The browser's input-to-screen delay is its own (see WEB_* below).
#
# So a web replay picks frames BY TIME and still NAMES them in trace frames:
# `<demo>_f<trace frame + offset>.png`, the exact names the Godot path writes.
# Everything that reads frames back off disk (`monitor.replay_pair`,
# `action_frames`, the estimator) keys on those names and needs no change.
#
# Chosen by the tree: a project with `project.godot` never reaches this code.
WEB_FPS_TRACE = 30.0
# Seconds after the event's trace time at which each named slot is sampled. The
# names stay `f + BEFORE` and `f + a for a in AFTER`; only where they are cut
# from moves.
#
# Measured on 2026-09-13 over 40 inputs that visibly answered, across five
# BENCH40 web games (echo-climb, potion-craft, pipe-crisis, dj-arena,
# music-label), first changed frame minus trace time:
#   15 inputs  -0.09 .. -0.03 s   the recording's lead-in cut lands ~0.1 s late,
#                                 so an immediate answer shows slightly EARLY
#   10 inputs  +0.03 .. +0.19 s   ordinary answers
#   15 inputs  +0.38 .. +1.09 s   tweens and scene changes
# So the before-frame sits clear of the earliest (-0.09) and the first
# after-frame clear of the ordinary answers (+0.19), as Godot's do; the second
# after-frame keeps Godot's one second, which catches all but the slowest tween.
WEB_BEFORE_S = float(os.environ.get('RSIGAME_REPLAY_WEB_BEFORE_S') or -0.25)
WEB_AFTER_S = tuple(float(x) for x in
                    (os.environ.get('RSIGAME_REPLAY_WEB_AFTER_S') or '0.35,1.0').split(','))


def _mp4_frames(mp4: Path) -> tuple[float, int]:
    """(fps, decoded frame count). Counted, not trusted from the header."""
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
         '-show_entries', 'stream=avg_frame_rate,nb_read_frames',
         '-of', 'json', str(mp4)],
        capture_output=True, text=True, timeout=300).stdout
    st = (json.loads(out or '{}').get('streams') or [{}])[0]
    num, _, den = str(st.get('avg_frame_rate') or '25/1').partition('/')
    try:
        fps = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        fps = 25.0
    try:
        n = int(st.get('nb_read_frames') or 0)
    except ValueError:
        n = 0
    return (fps if fps > 0 else 25.0), n


def _grab_named(mp4: Path, slots: dict, out_dir: Path, tag: str) -> dict:
    """slots: name (trace frame) -> mp4 frame index. One ffmpeg pass.

    Several names may map to one index; each gets its own file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = sorted({i for i in slots.values() if i >= 0})
    if not idx:
        return {}
    got = _grab(mp4, idx, out_dir, f'_w{tag}')     # -> {index: path}
    out = {}
    for name, i in sorted(slots.items()):
        src = got.get(i)
        if src is None:
            continue
        dst = out_dir / f'{tag}_f{name:05d}.png'
        if dst != src:
            shutil.copyfile(src, dst)
        out[name] = dst
    for p in got.values():
        if p.name.startswith(f'_w{tag}_f'):
            p.unlink(missing_ok=True)
    return out


def _replay_one_web(work: Path, trace_path: Path, out_dir: Path) -> dict:
    try:
        from gamecraft_bench.verifier.replay_web import replay_trace
        from gamecraft_bench.verifier.replay import ReplayError
    except ImportError as exc:
        raise RuntimeError(
            'this is a web (Phaser) tree and the web replay could not be '
            'imported: set GAMECRAFT_BENCH to the gamecraft-bench-web checkout '
            f'({exc})') from exc
    import rsigame.verify.demo_tail as demo_tail

    demo_id = trace_path.stem
    trace = json.loads(trace_path.read_text())
    mp4 = out_dir / f'{demo_id}.mp4'
    rec: dict = {'demo_id': demo_id, 'scenario': trace.get('scenario'),
                 'duration_frames': trace.get('duration_frames'),
                 'tail_s': round(demo_tail.tail_of(trace) / 30.0, 2),
                 'engine': 'web'}
    played = trace_path
    padded, did = demo_tail.pad(trace)
    if did:
        played = out_dir / f'{demo_id}.padded.json'
        played.parent.mkdir(parents=True, exist_ok=True)
        played.write_text(json.dumps(padded, indent=2))
        rec['padded_to'] = padded['duration_frames']
    if not (Path(work) / 'dist' / 'index.html').is_file():
        rec['replay_error'] = 'ReplayError: the project has no build (dist/index.html)'
        return rec
    try:
        rr = replay_trace(project_dir=work, trace_path=played,
                          output_mp4=mp4, log_dir=out_dir / f'log_{demo_id}')
    except ReplayError as e:
        rec['replay_error'] = f'{type(e).__name__}: {str(e)[:200]}'
        return rec
    rec['duration_seconds'] = round(rr.duration_seconds, 2)
    fps, n = _mp4_frames(mp4)
    rec['mp4_fps'], rec['mp4_frames'] = round(fps, 3), n
    last_i = max(0, n - 1)

    def at(seconds: float) -> int:
        return min(last_i, max(0, int(round(seconds * fps))))

    events = [e for e in (trace.get('events') or [])
              if e.get('type') in INPUT_TYPES]
    slots: dict = {}
    for ev in events:
        f = int(ev['frame'])
        te = f / WEB_FPS_TRACE
        slots[f + BEFORE] = at(te + WEB_BEFORE_S)
        for a, s in zip(AFTER, WEB_AFTER_S):
            slots[f + a] = at(te + s)
    last = max(0, int(round(rr.duration_seconds * 30)) - 2)
    slots[last] = last_i
    uniform = _evenly(last + 1, N_UNIFORM)
    for u in uniform:
        slots.setdefault(u, at(u / WEB_FPS_TRACE))
    frames = _grab_named(mp4, slots, out_dir / f'frames_{demo_id}', demo_id)
    rec['uniform'] = [{'frame': u, 'pos': round(u / max(last, 1), 4),
                       'path': str(frames[u])}
                      for u in uniform if u in frames]
    shots = []
    for i, ev in enumerate(events, start=1):
        f = int(ev['frame'])
        shots.append({
            'n': i,
            'trace_frame': f,
            'what': _describe(ev),
            'before': str(frames.get(f + BEFORE, '')) or None,
            'after': [str(frames[f + a]) for a in AFTER if f + a in frames],
        })
    rec['events'] = shots
    rec['n_events'] = len(shots)
    rec['final_frame'] = str(frames.get(last, '')) or None
    return rec


def replay_one(work: Path, trace_path: Path, out_dir: Path) -> dict:
    """Replay one demo and bracket every input event with labelled frames."""
    if not (Path(work) / 'project.godot').is_file():
        return _replay_one_web(Path(work), Path(trace_path), Path(out_dir))
    from gamecraft_bench.verifier.replay import replay_trace, ReplayError

    import rsigame.verify.demo_tail as demo_tail

    demo_id = trace_path.stem
    trace = json.loads(trace_path.read_text())
    mp4 = out_dir / f'{demo_id}.mp4'
    rec: dict = {'demo_id': demo_id, 'scenario': trace.get('scenario'),
                 'duration_frames': trace.get('duration_frames'),
                 'tail_s': round(demo_tail.tail_of(trace) / 30.0, 2)}

    # LET THE LAST INPUT FINISH. Measured across the pilot games, a demo whose
    # last input lands within three seconds of the end cannot show what that
    # input started -- shooter-sky-duel's five demos leave between 0.1 and 0.3
    # seconds -- and the reader below then sees a recording that stops
    # mid-flow and says so, which is true of the recording and not of the game.
    # The file on disk is never touched: `demo_outputs/` is the deliverable and
    # comes out of a round byte-identical to how it went in.
    played = trace_path
    padded, did = demo_tail.pad(trace)
    if did:
        played = out_dir / f'{demo_id}.padded.json'
        played.parent.mkdir(parents=True, exist_ok=True)
        played.write_text(json.dumps(padded, indent=2))
        rec['padded_to'] = padded['duration_frames']
    try:
        rr = replay_trace(project_dir=work, trace_path=played,
                          output_mp4=mp4, log_dir=out_dir / f'log_{demo_id}')
    except ReplayError as e:
        # A demo that will not replay at all is the strongest possible signal
        # and must reach the verifier as such, not as a missing entry.
        rec['replay_error'] = f'{type(e).__name__}: {str(e)[:200]}'
        return rec
    rec['duration_seconds'] = round(rr.duration_seconds, 2)

    events = [e for e in (trace.get('events') or [])
              if e.get('type') in INPUT_TYPES]
    # The events are the original ones either way -- only the recording length
    # differs -- so the frame arithmetic below is unchanged.
    wanted: list[int] = []
    for ev in events:
        f = int(ev['frame'])
        wanted += [f + BEFORE] + [f + a for a in AFTER]
    last = max(0, int(round(rr.duration_seconds * 30)) - 2)
    wanted.append(last)

    uniform = _evenly(last + 1, N_UNIFORM)
    # Same pass, same directory, same names: `_grab` sorts and de-duplicates,
    # so an index that happens to coincide with an anchor costs nothing.
    frames = _grab(mp4, wanted + uniform, out_dir / f'frames_{demo_id}',
                   demo_id)
    rec['uniform'] = [{'frame': n, 'pos': round(n / max(last, 1), 4),
                       'path': str(frames[n])}
                      for n in uniform if n in frames]

    shots = []
    for i, ev in enumerate(events, start=1):
        f = int(ev['frame'])
        shots.append({
            'n': i,
            'trace_frame': f,
            'what': _describe(ev),
            'before': str(frames.get(f + BEFORE, '')) or None,
            'after': [str(frames[f + a]) for a in AFTER if f + a in frames],
        })
    rec['events'] = shots
    rec['n_events'] = len(shots)
    rec['final_frame'] = str(frames.get(last, '')) or None
    return rec


def replay_demos(work: Path, out_dir: Path,
                 only: list[str] | None = None) -> dict:
    """Replay every shipped demo. The demos are read, never written."""
    work, out_dir = Path(work), Path(out_dir)
    demo_dir = work / 'demo_outputs'
    if not demo_dir.is_dir():
        return {'error': f'no demo_outputs/ under {work}', 'demos': []}
    traces = sorted(p for p in demo_dir.glob('*.json')
                    if only is None or p.stem in only)
    out_dir.mkdir(parents=True, exist_ok=True)
    demos = [replay_one(work, t, out_dir) for t in traces]
    res = {
        'work': str(work),
        'n_demos': len(demos),
        'n_failed': sum(1 for d in demos if d.get('replay_error')),
        'demos': demos,
    }
    # WRITTEN DOWN, because the only caller throws the return value away and
    # the directory then cannot say which of its PNGs are the uniform sample
    # and which bracket an input. Both live in one folder under one naming
    # scheme, so the distinction exists only here.
    try:
        (out_dir / 'replay.json').write_text(
            json.dumps(res, indent=1, ensure_ascii=False))
    except OSError:
        pass
    return res


def prune(out_dir: Path) -> None:
    """Drop the mp4s and logs, keep the labelled frames.

    A round keeps ~30 PNGs; the recordings are ~4 MB a demo and are only
    needed while the frames are being cut.
    """
    out_dir = Path(out_dir)
    for p in out_dir.glob('*.mp4'):
        p.unlink(missing_ok=True)
    for p in out_dir.glob('log_*'):
        shutil.rmtree(p, ignore_errors=True)


if __name__ == '__main__':
    w, o = Path(sys.argv[1]), Path(sys.argv[2])
    res = replay_demos(w, o)
    print(json.dumps(res, indent=1, ensure_ascii=False))
