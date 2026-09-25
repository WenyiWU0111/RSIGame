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
"""One entry point, so the loop is not started by remembering a shell script.

    rsigame config [-c experiment]     show what a run would actually use
    rsigame check                      validate configuration, paths and keys, without running anything
    rsigame develop <game> [-c ...]    run the development loop on one game
    rsigame score <run_dir>            score the checkpoints of a finished run

Everything reads `configs/default.toml` first, then the `-c` overlay, then the
environment -- see `rsigame.config`.
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog='rsigame', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-c', '--config', help='experiment overlay (name under configs/experiment or a path)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('config', help='print the resolved configuration')
    sub.add_parser('check', help='validate configuration, paths and keys')
    d = sub.add_parser('develop', help='run the development loop on one game')
    d.add_argument('game')
    d.add_argument('--rounds', type=int, default=None)
    s = sub.add_parser('score', help='score the checkpoints of a finished run')
    s.add_argument('run_dir')
    a, rest = ap.parse_known_args(argv)

    from . import config
    if a.cmd == 'config':
        return config.main(['-c', a.config] if a.config else [])
    cfg = config.load(a.config)          # raises ConfigError with what is missing
    if a.cmd == 'check':
        print('configuration ok:', ' + '.join(cfg.sources))
        return 0
    if a.cmd == 'develop':
        from . import loop
        argv2 = [a.game] + (['--rounds', str(a.rounds)] if a.rounds else []) + rest
        return int(loop.main(argv2) or 0) if hasattr(loop, 'main') else _no_main('loop')
    if a.cmd == 'score':
        from .eval import score_rounds
        return int(score_rounds.main([a.run_dir] + rest) or 0) if hasattr(score_rounds, 'main') else _no_main('score_rounds')
    return 2


def _no_main(mod: str) -> int:
    print(f'{mod} has no main(); run it as `python -m rsigame.{mod}` for now', file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
