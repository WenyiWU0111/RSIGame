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
"""M1 -- pair two checkpoints on one fixed demo, event by event.

The evidence a comparator can use is not "ten frames of the old build and ten
of the new". It is "at step 6 the script pressed Attack; here is the old build
just before and just after, and the new build just before and just after".
Producing that alignment is this module's whole job.

Alignment is possible because the demo is a fixed script: every input event
carries a `frame`, and `post_repair_replay` cuts frames at `frame + BEFORE` and
`frame + AFTER` around each one, with those offsets measured rather than
guessed. Replaying the same script on two trees therefore yields two frame sets
indexed by the same event numbers.

It is also the thing most likely to break silently, so the pairing is checked
rather than assumed: same event count, same trace frames, same descriptions. A
pair that fails those checks is returned with `aligned=False` and the reason,
not quietly used.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUN))

import rsigame.verify.post_repair_replay as PRR          # noqa: E402


@dataclass
class EventPair:
    """One scripted input, seen on both builds."""
    n: int
    trace_frame: int
    what: str
    old_before: str | None = None
    old_after: list[str] = field(default_factory=list)
    new_before: str | None = None
    new_after: list[str] = field(default_factory=list)


@dataclass
class DemoPair:
    demo_id: str
    scenario: str | None
    old_label: str
    new_label: str
    events: list[EventPair] = field(default_factory=list)
    old_final: str | None = None
    new_final: str | None = None
    aligned: bool = True
    problems: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        d = asdict(self)
        d['n_events'] = len(self.events)
        return d


def _rec_from_recorded(tree_dir: Path, demo_id: str,
                       frozen_trace: Path | None = None) -> dict | None:
    """Rebuild the event record from frames a previous replay already wrote.

    `post_repair_replay` computes this record in memory and hands it to the
    verifier; it is not persisted. The frames are, and their names carry the
    frame number, so the record can be reconstructed from the demo script plus
    the files on disk -- which avoids re-running a replay that has already been
    run once.
    """
    frames_dir = tree_dir / f'frames_{demo_id}'
    if not frames_dir.is_dir():
        return None
    # `demo_tail.pad` writes a padded copy only when it actually padded; a demo
    # that needed no tail leaves frames and no `.padded.json`. Fall back to the
    # frozen script, which is the same events either way -- padding changes the
    # recording length, not the inputs. Demos are frozen inputs (the refine
    # prompt forbids editing `demo_outputs/`, and `result.json` records
    # `demo_outputs_rewritten` when one is touched), so reading the corpus copy
    # is reading the script that was replayed.
    trace = tree_dir / f'{demo_id}.padded.json'
    if trace.is_file():
        t = json.loads(trace.read_text())
    elif frozen_trace is not None and Path(frozen_trace).is_file():
        t = json.loads(Path(frozen_trace).read_text())
    else:
        return None
    have = {}
    for p in frames_dir.glob('*.png'):
        stem = p.stem
        if '_f' not in stem:
            continue
        try:
            have[int(stem.rsplit('_f', 1)[1])] = str(p)
        except ValueError:
            continue
    if not have:
        return None
    events = [e for e in (t.get('events') or [])
              if e.get('type') in PRR.INPUT_TYPES]
    shots = []
    for i, ev in enumerate(events, start=1):
        f = int(ev['frame'])
        shots.append({'n': i, 'trace_frame': f, 'what': PRR._describe(ev),
                      'before': have.get(f + PRR.BEFORE),
                      'after': [have[f + a] for a in PRR.AFTER if f + a in have]})
    return {'demo_id': demo_id, 'scenario': t.get('scenario'),
            'events': shots, 'n_events': len(shots),
            'final_frame': have.get(max(have)) if have else None}


def _rec_from_replay(work: Path, demo_id: str, out_dir: Path) -> dict:
    trace = work / 'demo_outputs' / f'{demo_id}.json'
    if not trace.is_file():
        raise FileNotFoundError(f'no demo script {trace}')
    out_dir.mkdir(parents=True, exist_ok=True)
    return PRR.replay_one(work, trace, out_dir)


def load_side(source: Path, demo_id: str, *, work: Path | None = None,
              record_to: Path | None = None,
              frozen_trace: Path | None = None) -> dict:
    """One side of a pair: reuse recorded frames if present, else replay.

    `source` is a directory that already holds `frames_<demo>/`; `work` is a
    game tree to replay when it does not.
    """
    rec = _rec_from_recorded(Path(source), demo_id, frozen_trace)
    if rec is not None:
        rec['from'] = 'recorded'
        return rec
    if work is None:
        raise FileNotFoundError(
            f'{source} has no frames_{demo_id}/ and no tree was given to replay')
    rec = _rec_from_replay(Path(work), demo_id, Path(record_to or source))
    rec['from'] = 'replayed'
    return rec


def pair(old_rec: dict, new_rec: dict, old_label: str, new_label: str) -> DemoPair:
    """Match two records event by event, and say so when they do not match."""
    dp = DemoPair(demo_id=old_rec.get('demo_id') or new_rec.get('demo_id'),
                  scenario=old_rec.get('scenario'),
                  old_label=old_label, new_label=new_label,
                  old_final=old_rec.get('final_frame'),
                  new_final=new_rec.get('final_frame'))
    oe, ne = old_rec.get('events') or [], new_rec.get('events') or []
    if len(oe) != len(ne):
        dp.aligned = False
        dp.problems.append(f'event count differs: {len(oe)} vs {len(ne)}')
    for a, b in zip(oe, ne):
        if a['trace_frame'] != b['trace_frame']:
            dp.aligned = False
            dp.problems.append(
                f"event {a['n']}: trace frame {a['trace_frame']} vs {b['trace_frame']}")
        if a['what'] != b['what']:
            dp.aligned = False
            dp.problems.append(f"event {a['n']}: '{a['what']}' vs '{b['what']}'")
        dp.events.append(EventPair(
            n=a['n'], trace_frame=a['trace_frame'], what=a['what'],
            old_before=a.get('before'), old_after=list(a.get('after') or []),
            new_before=b.get('before'), new_after=list(b.get('after') or [])))
    missing = [e.n for e in dp.events
               if not e.old_before or not e.new_before]
    if missing:
        dp.problems.append(f'events missing a before frame: {missing}')
    return dp


def pair_checkpoints(old_src: Path, new_src: Path, demo_ids: list[str],
                     old_label: str, new_label: str,
                     old_work: Path | None = None,
                     new_work: Path | None = None,
                     frozen_corpus: Path | None = None) -> list[DemoPair]:
    out = []
    for d in demo_ids:
        try:
            ft = (Path(frozen_corpus) / 'demo_outputs' / f'{d}.json'
                  if frozen_corpus else None)
            o = load_side(old_src, d, work=old_work, frozen_trace=ft)
            n = load_side(new_src, d, work=new_work, frozen_trace=ft)
        except FileNotFoundError as e:
            dp = DemoPair(demo_id=d, scenario=None, old_label=old_label,
                          new_label=new_label, aligned=False,
                          problems=[str(e)])
            out.append(dp)
            continue
        out.append(pair(o, n, old_label, new_label))
    return out
