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
"""Which frames a reader gets, and what each one is called.

A label is half the evidence. "frame 12" tells a reader nothing they can use;
"after step 4, pressed ENTER" tells them what the picture is supposed to show,
and lets them say an input did nothing rather than guessing at two similar
screens. Every frame here carries what produced it.

The budget is small on purpose. post_verify batches at 40 images, but a reader
given forty screens narrates forty screens; the grounding stage is deciding
whether a handful of things are defects, and it does that better on the moments
that bracket an input than on everything the recorder kept.
"""
from __future__ import annotations

import json
from pathlib import Path

MAX_FRAMES = 16


def from_exploration(trace_dir: str | Path, limit: int = MAX_FRAMES) -> list[tuple]:
    """Labelled before/after pairs from a play-agent session."""
    p = Path(trace_dir) / 'exploration.json'
    if not p.is_file():
        return []
    t = json.loads(p.read_text())
    out = []
    for st in (t.get('steps') or []):
        obs = st.get('observed') or {}
        what = st.get('action') or '?'
        args = st.get('args') or {}
        if args.get('keycode'):
            what += f" {args['keycode']}"
        elif args.get('x') is not None:
            what += f" ({args['x']}, {args['y']})"
        n = st.get('n')
        if obs.get('before_frame'):
            out.append((f'step {n} before: {what}', obs['before_frame']))
        if obs.get('after_frame'):
            out.append((f'step {n} after: {what}', obs['after_frame']))
    return _trim(out, limit)


def from_replay(rec: dict, limit: int = MAX_FRAMES) -> list[tuple]:
    """Labelled frames from one replayed demo."""
    out = []
    for e in (rec.get('events') or []):
        if e.get('before'):
            out.append((f"input {e['n']} before: {e['what']}", e['before']))
        # Both after-frames. With one, an input that renders a frame late
        # reads as an input that did nothing, and a reader confirmed exactly
        # that as a defect -- on a build that was never edited, so the second
        # replay showed the same press working.
        for j, f in enumerate((e.get('after') or [])[:2], start=1):
            out.append((f"input {e['n']} after {j}: {e['what']}", f))
    if rec.get('final_frame'):
        out.append(('final frame', rec['final_frame']))
    return _trim(out, limit)


def _trim(pairs: list[tuple], limit: int) -> list[tuple]:
    """Keep the ends and spread the rest, in order.

    Dropping the tail would hide how a session finished, which is often the
    whole answer -- a demo that never reaches its state looks fine until the
    last frame.
    """
    pairs = [(l, f) for l, f in pairs if f and Path(f).is_file()]
    if len(pairs) <= limit:
        return pairs
    step = len(pairs) / (limit - 1)
    idx = sorted({int(i * step) for i in range(limit - 1)} | {len(pairs) - 1})
    return [pairs[i] for i in idx[:limit]]
