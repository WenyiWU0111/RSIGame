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
"""Build Controller-Plan's input from a finished run, at one round.

Everything here reads fields that already exist. The one thing worth writing
down is where the four macro directions actually live in the current data,
because it is not one place:

    correctness    checklist items with stage `correct`
    completeness   checklist items with stage `complete`
    feedback       polish_state rounds with focus `feedback`, and its headroom
    presentation   polish_state rounds with focus `visual`, and its headroom

The checklist's stage vocabulary only ever takes the two values, so half the
direction taxonomy has no checklist representation at all and has to come from
the polish ledger. A build that read only the checklist would report feedback
and presentation as empty rather than as unmeasured.

Two rules the assembly follows, both from the design doc and both easy to
violate by accident:

  * evidence from an older build is never presented as evidence about this one
    (S5). Anything older than the requested round is placed under a heading
    that says so.
  * demo descriptors state what a demo DOES, never what it covers (S6). A
    script's intent is not runtime coverage, and a static coverage map would
    read as authority.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

DIRECTIONS = ('correctness', 'completeness', 'feedback', 'presentation')
STAGE_TO_DIRECTION = {'correct': 'correctness', 'complete': 'completeness'}
POLISH_TO_DIRECTION = {'visual': 'presentation', 'feedback': 'feedback'}

# `_describe`-style naming for the deterministic half of a demo descriptor
_ACTION = {'key_press': 'key', 'mouse_click': 'click', 'mouse_move': 'move',
           'wait': 'wait', 'key_hold': 'hold'}


def _read(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


class Run:
    """One finished run, read once."""

    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir)
        self.game = self.dir.name
        self.corpus = self.dir.parent.parent.name
        self.state = _read(self.dir / 'checklist_state.json') or {'items': []}
        self.result = _read(self.dir / 'result.json') or {}
        self.pending = _read(self.dir / 'pending_repair.json') or []
        self.polish = (_read(self.dir / 'polish_state.json') or {}).get('rounds') or []
        self.rounds = [r for r in (self.result.get('rounds') or []) if isinstance(r, dict)]

    # -- state ----------------------------------------------------------
    def items(self, direction: str | None = None) -> list[dict]:
        out = []
        for it in self.state.get('items') or []:
            d = STAGE_TO_DIRECTION.get(it.get('stage'))
            if direction and d != direction:
                continue
            out.append(it)
        return out

    def demos(self) -> list[dict]:
        """Descriptors: what each demo does, from the script itself."""
        out = []
        for p in sorted((self.dir / 'work' / 'demo_outputs').glob('*.json')):
            t = _read(p)
            if not t:
                continue
            evs = t.get('events') or []
            seq, waits = [], 0
            for e in evs:
                kind = _ACTION.get(e.get('type'), e.get('type') or '?')
                if kind == 'wait':
                    waits += 1
                    continue
                if kind == 'key':
                    seq.append(f"key {e.get('keycode')}")
                elif kind == 'click':
                    seq.append('click')
                else:
                    seq.append(kind)
            out.append({
                'demo_id': p.stem,
                'scenario_flag': t.get('scenario'),
                # Input events only, waits counted separately. The two blocks
                # that used to describe a demo disagreed about its size -- one
                # counted waits as steps, one did not -- and a prompt that
                # gives two numbers for one thing invites a look at the
                # discrepancy rather than at the game.
                'num_inputs': len(seq),
                'duration_frames': t.get('duration_frames'),
                'actions': seq,
                'waits': waits,
            })
        return out

    # -- evidence -------------------------------------------------------
    def trace(self, r: int) -> dict:
        return _read(self.dir / f'round_{r:02d}' / 'trace' / 'exploration.json') or {}

    def demo_evidence(self, r: int) -> list[dict]:
        """Round `r`'s demo replays, as compact views.

        This is the evidence rule 6 asks for: a post-repair replay already on
        disk, handed to the next controller instead of being re-earned by
        playing the game again. Nothing is replayed here.
        """
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from rsigame.evidence.artifact import from_round, probes_at
        from rsigame.evidence.store import compact
        out = []
        for pid in probes_at(self.dir, r):
            a = from_round(self.dir, r, pid)
            if a:
                out.append(compact(a))
        return out

    def replayed_demos(self, r: int) -> list[str]:
        d = self.dir / f'round_{r:02d}' / 'post_replay'
        return sorted(p.name[len('frames_'):] for p in d.glob('frames_*')) if d.is_dir() else []


def _known_broken_by_demo(pending: list, demo_ids: set[str], upto: int) -> dict:
    """Attach outstanding post-repair problems to the demo that showed them.

    `where` reads like "05_totem_brand input 7", so demo ids are matched as
    whole tokens against the known vocabulary -- the same closed-vocabulary
    match the consolidation layer uses, not a semantic read of the sentence.
    """
    out = defaultdict(list)
    for entry in pending:
        if (entry.get('round') or 0) > upto:
            continue
        for b in entry.get('broken') or []:
            where = str(b.get('where') or '')
            hit = [d for d in demo_ids if re.search(rf'(?<![\w-]){re.escape(d)}(?![\w-])', where)]
            for d in hit or ['(no demo named)']:
                out[d].append({'what': b.get('what'), 'where': where,
                               'round': entry.get('round')})
    return dict(out)


def _merge_demos(demos: list[dict], replays: dict, broken: dict,
                 descriptions: dict, replayed_round: int) -> list[dict]:
    """Everything known about one demo, in one entry.

    A demo used to appear twice -- once as a script and once as a recording --
    and the recording half said "no written account of this replay" while the
    block below it carried, for three demos out of five, an exact account of
    what broke and at which input. The claim was false about the very prompt it
    sat in, and five repetitions of it read as five places worth looking.
    """
    out = []
    for d in demos:
        pid = d['demo_id']
        rep = replays.get(pid)
        out.append({**d,
                    'description': descriptions.get(pid),
                    'replayed_on_this_build': bool(rep),
                    'replayed_round': replayed_round if rep else None,
                    'outcome': (rep or {}).get('observation'),
                    'still_broken': [b['what'] for b in (broken.get(pid) or [])]})
    return out


def build(run: Run, r: int, direction: str, *, budget_total: int = 15,
          outer_reason: str = '', descriptions: dict | None = None) -> dict:
    """Assemble the five input blocks for the round-`r` decision."""
    items = run.items(direction)
    demo_ids = {d['demo_id'] for d in run.demos()}
    tr = run.trace(r)

    # state, trimmed to the active direction, plus anything currently broken
    state_rows = [{'id': i['id'], 'requirement': i['requirement'],
                   'status': i['status'], 'last_change_round': i.get('last_change_round'),
                   'stale': bool(i.get('stale'))} for i in items]
    broken = _known_broken_by_demo(run.pending, demo_ids, r)

    # evidence: this build first, older builds only as marked context.
    #
    # The build the controller is looking at is the one this round came in
    # with, and the round BEFORE produced it -- round r-1 repaired, then
    # replayed its demos against the result. So round r-1's post_replay is
    # evidence about the current build, and it is the cheapest evidence there
    # is because it already exists. Reading round r's own post_replay here
    # would be reading the output of a repair that has not happened yet.
    current = {'round': r,
               'closing_note': tr.get('closing_note') or '',
               'reached': tr.get('reached') or [],
               'not_reached': tr.get('not_reached') or [],
               'replayed_demos': run.replayed_demos(r - 1) if r > 1 else []}
    replays = {a['probe_id']: a
               for a in (run.demo_evidence(r - 1) if r > 1 else [])}
    history = []
    for prev in range(max(1, r - 2), r):
        t = run.trace(prev)
        if t.get('closing_note'):
            history.append({'round': prev, 'closing_note': t['closing_note']})

    # which items each earlier round aimed a probe at, by id
    probed = defaultdict(list)
    for i, row in enumerate(run.rounds[:r], start=1):
        for x in ((row.get('probe_goal') or {}).get('items') or []):
            probed[x].append(i)

    # the polish ledger carries the two directions the checklist has no
    # vocabulary for; report it as its own block rather than merging it in
    polish = [{'round': p.get('round'), 'focus': POLISH_TO_DIRECTION.get(
        p.get('focus'), p.get('focus')), 'goal': p.get('goal'),
        'headroom_after': p.get('headroom_after')}
        for p in run.polish if (p.get('round') or 0) <= r]

    # THE READER'S LATEST LOOK AT THE BUILD.
    #
    # Only assembled when nobody set a direction -- the arm where the planner
    # chooses the kind of round as well as the question. Under a direction the
    # monitor is the one that reads this and the planner executes; without one
    # the planner is being asked "does this game look unfinished?" and has to
    # be able to answer it. Measured: without this block the planner chose
    # `repair` 112 times out of 112, because the only evidence in front of it
    # was requirement statuses and replay outcomes. That was a missing input,
    # not a judgement.
    #
    # Read from `rounds.json` rather than from `Run`, whose `polish_state.json`
    # and `result.json` are both written when the run ENDS.
    reader = None
    if direction is None:
        try:
            rows = json.loads((run.dir / 'rounds.json').read_text())
            pol = [x['polish'] for x in rows
                   if isinstance(x, dict) and (x.get('polish') or {}).get('headroom')
                   and (x.get('round') or 0) <= r]
            reader = pol[-1] if pol else None
        except Exception:
            reader = None

    return {
        'game': run.game, 'corpus': run.corpus, 'round': r, 'reader': reader,
        # `next_direction` is None when nobody decided one -- the arm where the
        # planner chooses for itself. `render` branches on exactly this.
        'outer_directive': {'action': 'continue', 'next_direction': direction,
                            'remaining_headroom': 'high', 'reason': outer_reason,
                            'frozen': direction is not None},
        'state': {'direction': direction, 'items': state_rows,
                  # The WHOLE frozen checklist, for whoever decides whether
                  # something is a defect rather than what to investigate
                  # next. Filtering by direction is right for the plan, which
                  # works inside one direction, and wrong for grounding:
                  # horror-floor-13 REQUIRES that the elevator's buttons
                  # rearrange as it malfunctions, that requirement sits under
                  # `completeness`, and a grounder given only the
                  # `correctness` slice reported the rearranging as a defect
                  # and had it chosen as the round's repair target.
                  'all_items': [{'id': i['id'], 'requirement': i['requirement'],
                                 'status': i['status'],
                                 'direction': STAGE_TO_DIRECTION.get(i.get('stage'))}
                                for i in (run.state.get('items') or [])],
                  'probed_rounds': {k: v for k, v in probed.items()
                                    if k in {i['id'] for i in items}},
                  'known_broken': broken},
        'evidence': {'current': current, 'historical': history,
                     'replays': replays},
        'demos': _merge_demos(run.demos(), replays, broken,
                              descriptions or {}, r - 1),
        'polish_history': polish,
        'budget': {'total': budget_total, 'used': r, 'remaining': budget_total - r},
    }
