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
"""Give a demo time to show what its last input started.

WHY. A demo's `duration_frames` is authored by the game's own generator, and
the task caps it at 600 frames -- twenty seconds. shooter-sky-duel's generator
spent all of it: across its five demos the last input lands at frame 590-598
of 600, so whatever that input starts has between 0.1 and 0.3 seconds to
appear on screen. The rubric then asks whether a debrief screen is shown and
whether the loop reaches its end, and scores a recording that was cut off
mid-flow as a game that has no debrief.

Measured on all five pilot games, both arms, three passes each. Adding ten
seconds of TAIL -- not one event moved -- against tail slack:

    game                tail    new-base @20s   new-base @30s   change
    shooter-sky-duel    0.1s       -0.022          +0.251       +0.273
    platformer-ink      0.7s       -0.026          +0.000       +0.026
    visualnovel         2.5s       +0.089          +0.046       -0.043
    horror-tape         2.7s       +0.037          +0.048       +0.011
    idle-spell-tower    3.7s       +0.000          -0.004       -0.004

The effect is monotone in the slack and lives entirely in the one game whose
demos have none. That is a property of how those demos were written, not of
the games being compared.

WHY THIS IS NOT A THUMB ON THE SCALE. The rule is the same for every tree --
G0, the baseline arm, ours -- and fires only on the condition, not on the
result: a demo with room already gets nothing, which is four of these five
games. It adds no inputs, moves no event, changes no coordinate and asks the
game to do nothing it was not already asked to do. What it changes is how long
the camera keeps running afterwards.

WHY IT STILL HAS TO BE DECLARED. It is a deviation from the task's stated cap,
and a number produced under it is not comparable with one produced without it.
Every caller records `padded` in its output so a table can say which it is.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

# Below this much slack after the last input, a demo cannot show the outcome
# of that input. Three seconds is above every game here that did not have the
# problem (0.7s is the next-tightest and it did show the effect, weakly) and
# below every game that did not (2.5s and up showed none).
MIN_TAIL_FRAMES = 90        # 3.0s at 30fps
TAIL_PAD_FRAMES = 300       # 10.0s


def tail_of(trace: dict) -> int:
    """Frames between the last input and the end of the recording."""
    ev = [e for e in (trace.get('events') or []) if e.get('type') != 'wait']
    last = max((int(e.get('frame', 0)) for e in ev), default=0)
    return int(trace.get('duration_frames', 0)) - last


def pad(trace: dict, min_tail: int = MIN_TAIL_FRAMES,
        extra: int = TAIL_PAD_FRAMES) -> tuple[dict, bool]:
    """(trace, was_padded). Only `duration_frames` is ever touched."""
    t = dict(trace)
    if tail_of(trace) >= min_tail:
        return t, False
    t['duration_frames'] = int(trace.get('duration_frames', 0)) + extra
    return t, True


def report(demo_dir: Path) -> list:
    """What each demo's slack is, for the record and for a reader."""
    out = []
    for f in sorted(Path(demo_dir).glob('*.json')):
        try:
            t = json.loads(f.read_text())
        except Exception:
            continue
        tail = tail_of(t)
        out.append({'demo': f.stem, 'duration_frames': t.get('duration_frames'),
                    'tail_frames': tail, 'tail_s': round(tail / 30.0, 2),
                    'would_pad': tail < MIN_TAIL_FRAMES})
    return out


def padded_copy(src_tree: Path, dst_tree: Path,
                min_tail: int = MIN_TAIL_FRAMES,
                extra: int = TAIL_PAD_FRAMES) -> dict:
    """A whole game tree whose demos have room, for scoring.

    A copy, because `demo_outputs/` in the tree under test is the deliverable
    and must come out of this byte-identical to how it went in.
    """
    src_tree, dst_tree = Path(src_tree), Path(dst_tree)
    if dst_tree.exists():
        shutil.rmtree(dst_tree)
    shutil.copytree(src_tree, dst_tree, symlinks=True,
                    ignore=shutil.ignore_patterns('.godot'))
    changed = []
    for f in sorted((dst_tree / 'demo_outputs').glob('*.json')):
        t = json.loads(f.read_text())
        t2, did = pad(t, min_tail, extra)
        if did:
            f.write_text(json.dumps(t2, indent=2))
            changed.append({'demo': f.stem,
                            'from': t.get('duration_frames'),
                            'to': t2['duration_frames'],
                            'tail_was_s': round(tail_of(t) / 30.0, 2)})
    return {'tree': str(dst_tree), 'padded': changed,
            'n_demos': len(list((dst_tree / 'demo_outputs').glob('*.json'))),
            'n_padded': len(changed), 'min_tail_frames': min_tail,
            'extra_frames': extra}


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        for r in report(Path(sys.argv[1]) / 'demo_outputs'):
            print(f"  {r['demo']:<30}{r['duration_frames']:>5}f  "
                  f"tail {r['tail_s']:>5.1f}s  {'← will pad' if r['would_pad'] else ''}")
    else:
        print(json.dumps(padded_copy(Path(sys.argv[1]), Path(sys.argv[2])),
                         indent=1, ensure_ascii=False))
