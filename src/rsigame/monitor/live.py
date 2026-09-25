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
"""The monitor, running during the run, with the authority to end it.

OFF BY DEFAULT, and the default matters. Every number in the paper comes from
runs where all thirty rounds happened and the champion was picked afterwards;
turning this on changes what a run *is*, because two arms no longer spend the
same budget. Set `[monitor] live = true` when that is what you want to study.

WHAT IT DOES. At each checkpoint boundary it makes the same decision the
offline sweep makes -- this checkpoint against the held champion, adopt or keep
-- appends it to the same champion trace, and applies Value Stop: once the
champion has survived K consecutive checkpoints, the run ends at the next round
boundary and the champion is what it delivers.

WHAT IT DOES NOT DO. It never kills a round mid-flight, never rolls the tree
back, and never edits anything: the loop keeps developing from the newest
build, exactly as before, and the champion only decides what is *delivered*.
A checkpoint it cannot grade (no recording, a failed build, a judge error) is
recorded as such and never ends a run -- a stop has to be earned by K
comparisons that actually happened.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .value_stop import Stop, value_stop


def enabled() -> bool:
    return (os.environ.get('RSIGAME_MONITOR_LIVE') or '0').lower() in ('1', 'true', 'yes')


class LiveMonitor:
    """One game's champion, kept while the game is still being developed."""

    def __init__(self, game: str, run_root: Path, out_dir: Path | None = None,
                 r0_dir: Path | None = None, k: int | None = None):
        from . import vm_replay as VM
        self.VM = VM
        self.game = game
        self.k = int(os.environ.get('RSIGAME_MONITOR_STOP_AFTER_K') or 3) if k is None else k
        self.out = Path(out_dir or run_root / 'vm')
        self.out.mkdir(parents=True, exist_ok=True)
        VM.setup(run=str(run_root), r0_dir=str(r0_dir or run_root / 'r0'), out=str(self.out))
        self.trace_file = self.out / f'{game}.jsonl'
        self.trace: list[dict] = []
        if self.trace_file.is_file():
            self.trace = [json.loads(l) for l in self.trace_file.read_text().splitlines() if l.strip()]
        self.rubric = VM.load_rubric(game)

    @property
    def champion(self) -> int | None:
        return self.trace[-1]['champion'] if self.trace else None

    def every(self) -> int:
        return int(os.environ.get('RSIGAME_MONITOR_CHECKPOINT_EVERY') or 3)

    def due(self, r: int) -> bool:
        return r % self.every() == 0

    def consider(self, r: int) -> dict:
        """Judge round r's checkpoint. Returns the trace record."""
        rec = self.VM.decide_checkpoint(self.game, r, self.champion, self.rubric)
        self.trace.append(rec)
        with self.trace_file.open('a') as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
        return rec

    def stop(self) -> Stop | None:
        """Value Stop over the trace so far, or None while it is still paying.

        `value_stop` answers for a finished trace, so a mid-run call has to
        ignore its "never settled" fallback: a run that has not saturated yet
        is not a run that has decided to stop.
        """
        if len(self.trace) <= self.k:
            return None
        s = value_stop(self.trace, self.k, self.game)
        return s if s.saturated and s.checkpoint == len(self.trace) - 1 else None
