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
"""The target ledger: what is being worked on, and how often it has been tried.

The ledger exists for one reason -- so the controller can see that it has asked
something three rounds running. It is not a control input: nothing forces a
redirect off a target, and no rule caps attempts. Doc S29 is explicit that
repeated effort is a negative prior, not an unconditional stop, and the numbers
here are context for that judgment.

That is also why an occasionally wrong count is survivable. The controller sees
the target STATEMENTS, not just the counts, so a target that got split across
three entries shows up as three near-identical statements -- which reads as
repetition just as plainly as `attempts: 3` would.

What is NOT survivable is a ledger that only grows. Fifteen rounds of
accumulated targets, most of them long since fixed, crowd the real ones out of
the prompt. So targets close, and the rule is deterministic:

    a target names the requirement ids it is about; when all of them read
    `satisfied`, it is resolved.

They also REOPEN. In the corpus, block-cascade's G2 reads confirmed_gap at
round 9, satisfied at 10 and 11, and confirmed_gap again at 12 -- a repair that
did not hold. A ledger that could only close would have lost the target exactly
when it mattered, so `sync` moves both ways and records the reopen.

A target with no requirement ids -- a general quality question -- cannot be
closed this way. It goes dormant after some rounds without being picked, which
keeps it out of the prompt without claiming it was fixed, and it comes back if
the controller raises it again.
"""
from __future__ import annotations

import json
from pathlib import Path

DORMANT_AFTER = 3


def status_at(item: dict, r: int) -> str:
    """What the checklist said about this item as of round `r`."""
    refs = [e for e in (item.get('evidence_refs') or [])
            if (e.get('round') or 0) <= r and e.get('status')]
    if refs:
        return max(refs, key=lambda e: e['round'])['status']
    # No evidence yet as of this round. The item's current status describes a
    # later state and must not be read backwards onto this one.
    return 'unverified'


class Ledger:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.targets: list[dict] = []
        if self.path and self.path.is_file():
            self.targets = json.loads(self.path.read_text())

    # -- reading --------------------------------------------------------
    def open(self) -> list[dict]:
        return [t for t in self.targets if t['status'] == 'open']

    def save(self):
        if self.path:
            self.path.write_text(json.dumps(self.targets, indent=1, ensure_ascii=False))

    # -- writing --------------------------------------------------------
    def note(self, target_ref: str | None, question: str,
             related_items: list[str], r: int) -> str | None:
        """Record that round `r` asked this. Returns the target id, or None.

        `target_ref` is whatever survived validation: an open target's id, or
        the literal `new`. Anything else was already dropped, and a dropped
        reference means the round is not attributed rather than attributed to
        a target the model named freely.
        """
        if target_ref == 'new':
            tid = f'TGT{len(self.targets) + 1:02d}'
            self.targets.append({
                'id': tid, 'statement': question[:160], 'items': related_items,
                'attempts': 1, 'verified_successes': 0, 'reopens': 0,
                'status': 'open', 'first_round': r, 'last_round': r})
            return tid
        for t in self.targets:
            if t['id'] == target_ref:
                t['attempts'] += 1
                t['last_round'] = r
                if t['status'] != 'open':
                    t['status'] = 'open'
                    t['reopens'] += 1
                # A target picked again may have been described better this
                # time; keep the newer wording, which is what the next round
                # reads.
                t['statement'] = question[:160] or t['statement']
                for i in related_items:
                    if i not in t['items']:
                        t['items'].append(i)
                return t['id']
        return None

    def sync(self, state: dict, r: int) -> list[dict]:
        """Close what the checklist says is fixed, reopen what came back."""
        by_id = {i['id']: i for i in (state.get('items') or [])}
        moved = []
        for t in self.targets:
            if not t['items']:
                if t['status'] == 'open' and r - t['last_round'] >= DORMANT_AFTER:
                    t['status'] = 'dormant'
                    moved.append({'id': t['id'], 'to': 'dormant',
                                  'why': f'not picked for {r - t["last_round"]} rounds, '
                                         f'and it names no requirement'})
                continue
            sts = [status_at(by_id[i], r) for i in t['items'] if i in by_id]
            if not sts:
                continue
            if all(s == 'satisfied' for s in sts):
                if t['status'] == 'open':
                    t['status'] = 'resolved'
                    t['verified_successes'] += 1
                    moved.append({'id': t['id'], 'to': 'resolved',
                                  'why': f"all of {', '.join(t['items'])} read satisfied"})
            elif any(s == 'confirmed_gap' for s in sts):
                if t['status'] == 'resolved':
                    t['status'] = 'open'
                    t['reopens'] += 1
                    moved.append({'id': t['id'], 'to': 'open',
                                  'why': 'a requirement it closed is a confirmed gap again'})
        return moved
