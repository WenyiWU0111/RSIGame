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
"""When development has stopped paying: the rule, and what it delivers.

THE RULE. Stop at the first checkpoint whose champion -- the best build seen so
far -- has been the same for K consecutive checkpoints (K=3 in the paper: three
checkpoints, nine rounds, during which nothing beat the held build). Deliver
that champion.

TWO PLACES USE IT, AND THE DIFFERENCE IS WHAT A RUN COSTS:

  live     `[monitor] live = true` runs the monitor during the run. Each
           checkpoint is judged as it is written and the run ENDS when this
           rule fires -- rounds after that are never spent. See monitor/live.py.

  offline  the default. Every round runs, `scripts/vm_replay.py` replays the
           checkpoints afterwards, and this rule is read over the finished
           champion trace: "if we had stopped when the monitor said so, what
           would we have shipped and how many rounds would it have cost". That
           is what the paper reports, and it is why the paper's arms all spent
           the same budget.

The arithmetic is identical in both; only the moment differs. `value_stop()`
answers for a finished trace, so a live caller must ignore its "never settled"
fallback -- a run that has not saturated yet has not decided anything.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


def stop_k() -> int:
    return int(os.environ.get('RSIGAME_MONITOR_STOP_AFTER_K') or 3)


@dataclass
class Stop:
    """Where the saturation stop would have ended one game."""
    game: str
    checkpoint: int       # index into the checkpoint list
    round: int            # the round that checkpoint corresponds to
    champion: int         # the round whose build would be delivered
    saturated: bool       # False = never settled; the full budget was used


def read_trace(path: str | Path) -> list[dict]:
    """A champion trace as `vm_replay.py` writes it: one json object per line,
    each carrying `round` and the `champion` held after it."""
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return sorted(rows, key=lambda r: int(r['round']))


def value_stop(trace: list[dict], k: int | None = None, game: str = '') -> Stop:
    k = stop_k() if k is None else k
    champs = [(int(r['round']), r.get('champion')) for r in trace]
    for i in range(k, len(champs)):
        window = [c for _, c in champs[i - k:i + 1]]
        if window[0] is not None and all(c == window[0] for c in window):
            return Stop(game, i, champs[i][0], int(window[0]), True)
    last = champs[-1] if champs else (0, 0)
    return Stop(game, len(champs) - 1, last[0], int(last[1] or 0), False)


def main(argv: list[str] | None = None) -> int:
    from rsigame import config
    config.ensure()
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('traces', nargs='+', help='champion traces (<game>.jsonl from vm_replay)')
    ap.add_argument('-k', type=int, default=None, help=f'checkpoints unchanged before stopping (default {stop_k()})')
    a = ap.parse_args(argv)
    used, saturated = [], 0
    for t in a.traces:
        p = Path(t)
        s = value_stop(read_trace(p), a.k, p.stem)
        saturated += s.saturated
        used.append(s.round)
        print(f'{s.game:<40} stop at round {s.round:>3}  deliver R{s.champion:02d}'
              f'{"" if s.saturated else "   (never settled)"}')
    if used:
        print(f'\n{len(used)} games · {saturated} settled · mean rounds used '
              f'{sum(used) / len(used):.1f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
