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
"""Let the repair agent run and watch its own candidate build.

    python -m rsigame.agent.evolve.play_cli <game_dir> <action> [--k v ...]

WHY THIS EXISTS. The repair agent could edit code and run `npm run build`, and
nothing else. For a game that is close to useless: a build that compiles tells
you nothing about whether the HUD appeared or whether the player moves, which
is what most of these repairs are about. So the agent was doing

    inspect -> edit -> build -> stop

when the natural loop for a coding agent, and the one it is good at, is

    inspect -> edit -> build -> RUN -> OBSERVE -> revise

The primitives already exist -- `_Session` drives a Phaser build through
Playwright, `GodotSession` drives a Godot process through xdotool, and both
present the same eight actions. What was missing was a way to reach them from
the agent, which runs in TypeScript with a shell. Hence a CLI: one process per
call, JSON on stdout, no daemon to leak.

THE LINE THIS MUST NOT CROSS. The agent may observe the GAME. It may not
observe the JUDGE. Nothing here can reach the verifier's verdict, the
protected verifier's verdict, or the promotion decision -- those run after the
session ends, and if the agent could see them it would start optimising
against the thing that is supposed to be measuring it. Environment feedback is
allowed; judge feedback is withheld. That distinction is the whole design, and
this file is where it would be easiest to break by accident, so: this module
imports the session and nothing else from the evolve package.

STATELESS BY CHOICE. Each invocation boots, acts, reports and exits. A
persistent session would be faster and would need the agent to manage a
handle, a lifetime and a cleanup path; a repair that leaves a browser running
in job scratch is a worse failure than a slow one. Sequences are expressed by
passing several actions to one call.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

# The one import from the verifier side, and deliberately the only one: the
# session drives a build. Nothing that judges is reachable from here.
from .verifier_agent import _session_for


def _run(game_dir: Path, steps: list, out_dir: Path, port: int) -> dict:
    session_cls = _session_for(game_dir)
    out = {'game': str(game_dir), 'session': session_cls.__name__,
           'steps': []}
    session = session_cls(game_dir, port, out_dir)
    # THE ONE DOOR THIS CAPABILITY COMES THROUGH. Set as an attribute rather
    # than a constructor argument so the Godot session, which shares this
    # surface and has no JS runtime to write into, needs no change and simply
    # goes on refusing the action by name.
    session.allow_state_writes = True
    with session as sess:
        for st in steps:
            action = st.get('action')
            args = {k: v for k, v in st.items() if k != 'action'}
            rec = sess.do(action, args)
            # Frames are reported as paths. The agent opens them with the file
            # tool it already has; handing back base64 would blow up its
            # context for images it may not need.
            step = {
                'action': action, 'args': args,
                'ok': rec.get('ok', not rec.get('error')),
                'error': rec.get('error'),
                'frame': rec.get('frame'),
                'observed': rec.get('observed'),
            }
            # Anything else the action chose to report comes through too. This
            # projection used to be a fixed six keys, which silently dropped
            # every new field: `boot` began reporting `settled_ms` and
            # `runtime_readable` and they were invisible here -- which looks
            # exactly like the code not running, and cost a round of debugging
            # to tell apart.
            for k, v in rec.items():
                if k not in step and k not in ('n', 'note'):
                    step[k] = v
            out['steps'].append(step)
        out['frames'] = list(getattr(sess, 'frames', []) or [])
    return out


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog='play_cli',
        description='Run a game build and observe it. This reports what the '
                    'GAME did; it never reports what any verifier concluded.')
    ap.add_argument('game_dir')
    ap.add_argument('--steps', required=True,
                    help='JSON list of steps, e.g. '
                         '\'[{"action":"boot"},'
                         '{"action":"press_key","key":"Enter"},'
                         '{"action":"screenshot"}]\'')
    ap.add_argument('--out', default='', help='where frames are written')
    # A constant here is a silent wrong answer, not a crash: under any
    # concurrency the second agent either fails to bind or -- worse --
    # talks to a server another game already put on that port and reads
    # its state as if it were its own.
    ap.add_argument('--port', type=int, default=0,
                    help='0 (default) asks the OS for a free one')
    a = ap.parse_args(argv)

    game = Path(a.game_dir).resolve()
    if not game.is_dir():
        print(json.dumps({'error': f'no such directory: {game}'}))
        return 2
    try:
        steps = json.loads(a.steps)
        assert isinstance(steps, list) and steps
    except Exception as exc:
        print(json.dumps({'error': f'--steps is not a non-empty JSON list: '
                                   f'{type(exc).__name__}'}))
        return 2

    out_dir = Path(a.out) if a.out else game / '_repair_evidence' / 'play'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Each run stamps its own elapsed seconds into a log beside the game. The
    # caller sums that file and subtracts it from the session's wall clock, so
    # booting a browser is not charged against the time the agent has to
    # think. Measured here rather than in TypeScript because the SDK does not
    # report per-tool duration, and guessing from the call count would price a
    # 2-second screenshot the same as a 40-second boot.
    t0 = time.time()

    def _stamp():
        try:
            log = game / '_repair_evidence' / 'play_time.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('a') as fh:
                fh.write(f'{time.time() - t0:.2f}\n')
        except Exception:
            pass  # timing is bookkeeping; never fail a run over it

    try:
        result = _run(game, steps, out_dir, a.port or _free_port())
        _stamp()
        print(json.dumps(result, default=str))
        return 0
    except Exception as exc:
        _stamp()
        # Reported as JSON, not raised: the agent parses stdout, and a
        # traceback on stderr reads to it as the game being broken rather
        # than the harness failing to start.
        print(json.dumps({'error': f'{type(exc).__name__}: {str(exc)[:300]}',
                          'note': 'the harness failed to run the game; this '
                                  'says nothing about your repair'}))
        return 1


if __name__ == '__main__':
    sys.exit(main())
