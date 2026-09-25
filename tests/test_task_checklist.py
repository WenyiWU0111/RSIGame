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
"""Acceptance for Part 1, on real tasks.

The stop condition in the handoff is "extraction works on a few stored tasks".
What "works" has to mean here is narrow and checkable, because the one failure
that would poison everything downstream -- a requirement the task never stated
-- looks exactly like a real item.

    1. every kept item's quote is verbatim in the document
    2. nothing survives that the document does not say
    3. numbers in an item appear in its own quote (no inference)
    4. build/engine/demo-format instructions are left out
    5. granularity lands near the rubric's, without having seen the rubric

Run:  python tests_task_checklist.py [game ...]
"""

import pytest
import os
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rsigame.checklist.task_checklist as TC   # noqa: E402


# These read real run directories(which the repo does not carry)。skip when unset,
# rather than passing green on an empty list.
pytestmark = pytest.mark.skipif(
    not os.environ.get('RSIGAME_TEST_GAMES'),
    reason='set RSIGAME_TEST_GAMES (and RSIGAME_PATHS_RUN) to run against real runs')


RUN = Path(__file__).resolve().parent
# which games to run comes from --games or RSIGAME_TEST_GAMES;the repo carries no experiment lists.
_LIST = os.environ.get('RSIGAME_TEST_GAMES', '')
DEFAULT = [l.strip() for l in _LIST.split()
           if l.strip()]

# Things the task says to the BUILDER, not about the game. An item quoting one
# of these is a leak of build instructions into the checklist.
_BUILD_TALK = re.compile(
    r'--headless|--quit-after|project\.godot|demo_outputs|/workspace/|'
    r'\.json|duration_frames|keycode|screenshot\.sh|godot --path|'
    r'trace file|mouse_click|OS\.get_cmdline', re.I)


# SPELLED-OUT NUMBERS COUNT AS THE NUMBER.
#
# The first version of this check knew `one` through `ten` and failed
# platformer-ink-trail on "The game contains 36 levels", whose quote is
# "Thirty-six levels across six worlds". The document states the number; the
# extraction copied it and wrote it as a digit, which is what we want. The
# check was what was wrong, and it was wrong in the expensive direction --
# it accused a correct extraction of inventing a figure.
#
# Specs say things like "six worlds", "Thirty-six levels", "at least four".
# Zero to ninety-nine, hyphenated compounds included, covers every number word
# in the five pilot documents and anything a game task is likely to spell out.
_UNITS = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
          'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
          'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
          'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
          'nineteen': 19}
_TENS = {'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60,
         'seventy': 70, 'eighty': 80, 'ninety': 90}


def _numbers_in(text: str) -> set:
    """Every number the text states, as digit strings -- digits and words."""
    t = (text or '').lower()
    out = set(re.findall(r'\b\d+\b', t))
    for m in re.finditer(r'\b([a-z]+)(?:[- ]([a-z]+))?\b', t):
        a, b = m.group(1), m.group(2)
        if a in _TENS:
            out.add(str(_TENS[a] + (_UNITS[b] if b in _UNITS else 0)))
        elif a in _UNITS:
            out.add(str(_UNITS[a]))
    return out


fails = []


def check(cond, msg):
    print(('  PASS  ' if cond else '  FAIL  ') + msg)
    if not cond:
        fails.append(msg)


def rubric_count(game: str):
    """How many lines the graded rubric has, for scale only.

    Researcher-side. The number is never shown to the extractor and never
    reaches the loop -- it is here so we can say whether the checklist is in
    the right ballpark, not to tune it toward the rubric.
    """
    for arm, tree in (('P2-E0', 'godot_gpt_e2'), ('A0', 'godot_gpt_strict')):
        p = RUN / 'scoring_t0' / arm / tree / game / 'r0' / 'breakdown.json'
        if p.is_file():
            return len(json.loads(p.read_text()).get('requirements') or [])
    return None


def main():
    games = sys.argv[1:] or DEFAULT
    print(f'{len(games)} tasks\n')
    totals = []
    for g in games:
        res = TC.extract(g)
        if res.get('error'):
            check(False, f'{g}: {res["error"]}')
            continue
        items, dropped = res['items'], res['dropped']
        raw = TC.task_document(g)[0]
        n_rub = rubric_count(g)
        totals.append((g, len(items), n_rub))

        print('=' * 78)
        print(f'{g}   {len(items)} items, {len(dropped)} dropped'
              f'   (rubric has {n_rub})'
              f'{"   [cached]" if res.get("from_cache") else ""}')
        print('=' * 78)
        print(TC.render(res))
        print()

        # 1 + 2. every kept quote is really in the document
        kept, bad = TC.verify_quotes(items, raw)
        check(len(bad) == 0,
              f'{g}: all {len(items)} kept quotes verbatim in the source')

        # 3. a number in the requirement must appear in its own quote
        invented = []
        for it in items:
            rn = set(re.findall(r'\b\d+\b', it['requirement']))
            qn = _numbers_in(it['quote'])
            if rn - qn:
                invented.append((it['id'], sorted(rn - qn), it['requirement'][:70]))
        check(not invented,
              f'{g}: no number appears that its own quote does not have'
              + (f' (offenders: {invented})' if invented else ''))

        # 4b. the launch contract is out of the list, and recorded as such
        leaked_launch = [it['id'] for it in items
                         if TC._LAUNCH_CONTRACT.search(
                             f"{it['requirement']} {it['quote']}")]
        check(not leaked_launch,
              f'{g}: no --scenario launch-contract item in the checklist'
              + (f' ({leaked_launch})' if leaked_launch else '')
              + f'; {len(res.get("harness") or [])} filed under harness')

        # 4. build instructions stayed out
        leaked = [it['id'] for it in items
                  if _BUILD_TALK.search(it['requirement'])]
        check(not leaked,
              f'{g}: no build/engine/demo-format instruction became a '
              f'requirement' + (f' ({leaked})' if leaked else ''))

        # 5. scale
        if n_rub:
            ok = 0.6 * n_rub <= len(items) <= 2.5 * n_rub
            check(ok, f'{g}: {len(items)} items is within 0.6x-2.5x of the '
                      f'{n_rub} rubric lines')

        # ids are unique and dense
        ids = [it['id'] for it in items]
        check(len(set(ids)) == len(ids), f'{g}: item ids unique')
        print()

    print('=' * 78)
    print(f'  {"game":<26}{"items":>7}{"rubric":>8}{"ratio":>8}')
    for g, n, r in totals:
        print(f'  {g:<26}{n:>7}{str(r):>8}{(n / r if r else 0):>8.2f}')
    print(f'\n{"ALL PASS" if not fails else str(len(fails)) + " FAILED"}')
    for f in fails:
        print('  - ' + f)
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
