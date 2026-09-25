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
"""Acceptance for Part 2, offline, on sessions already on disk.

The stop condition in the handoff is "run offline / stored-trace checks on
several games, inspect the resulting initial state, report and stop". Nothing
here calls the game; it reads EV1's round-1 traces, which are the best evidence
any arm has produced, and asks what the bootstrap makes of them.

What "works" means here is not "it decided a lot". It is the opposite:

  1. every decided item names steps that exist
  2. no defect rests on a place the session never reached
  3. the share left `unverified` is neither near zero nor near everything
  4. general checks come from the fixed library, 3 to 5 of them
  5. the gates actually fire, or they are decoration

Run:  python tests_bootstrap_verifier.py [game ...]
"""

import pytest
import json
import os
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN))
import rsigame.checklist.bootstrap_verifier as BV   # noqa: E402
import rsigame.checklist.task_checklist as TC       # noqa: E402


# These read real run directories(which the repo does not carry)。skip when unset,
# rather than passing green on an empty list.
pytestmark = pytest.mark.skipif(
    not os.environ.get('RSIGAME_TEST_GAMES'),
    reason='set RSIGAME_TEST_GAMES (and RSIGAME_PATHS_RUN) to run against real runs')


ARM = Path(os.environ.get('RSIGAME_TEST_ARM')
           or RUN / 'self_refine_godot_gpt_ev1' / 'runs')
CORPUS = RUN / 'godot' / 'corpus_gpt'
ROUND = os.environ.get('RSIGAME_TEST_ROUND') or 'round_01'
# which games to run comes from --games or RSIGAME_TEST_GAMES;the repo carries no experiment lists.
_LIST = os.environ.get('RSIGAME_TEST_GAMES', '')
DEFAULT = [l.strip() for l in _LIST.split()
           if l.strip()]

fails = []


def check(cond, msg):
    print(('  PASS  ' if cond else '  FAIL  ') + msg)
    if not cond:
        fails.append(msg)


def static_for(game: str, exp: dict) -> str:
    """The same fact sheet E2 sends, built off the round-1 tree (a frozen G0)."""
    try:
        from rsigame.view.static_view import build_static_view
        from rsigame.project import declared_scenarios
        work = CORPUS / game
        if not work.is_dir():
            return ''
        visited = [((s.get('args') or {}).get('scenario') or '')
                   for s in (exp.get('steps') or [])
                   if s.get('action') == 'boot_scenario']
        return build_static_view(work, declared=declared_scenarios(work),
                                 visited_now=[v for v in visited if v],
                                 visited_history=[])
    except Exception as exc:
        print(f'    (no static view: {type(exc).__name__}: {exc})')
        return ''


def main():
    games = sys.argv[1:] or DEFAULT
    print(f'{len(games)} games, {ARM}/<game>/{ROUND}\n')
    rows = []
    for g in games:
        p = ARM / g / ROUND / 'trace' / 'exploration.json'
        if not p.is_file():
            check(False, f'{g}: no stored trace at {p}')
            continue
        exp = json.loads(p.read_text())
        res = BV.bootstrap(g, p, static_view=static_for(g, exp),
                           click_evidence=True)
        if res.get('error'):
            check(False, f'{g}: {res["error"]}')
            continue
        out = RUN / '_scratch' / 'bootstrap' / f'{g}.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1))

        print('=' * 78)
        print(BV.render(res))
        print()
        t = res['tally']
        rows.append((g, res['n_items'], t['satisfied'], t['confirmed_gap'],
                     t['unverified'], t['unverified_pct'], t['demoted'],
                     len(res['general'])))

        real = {s.get('n') for s in exp.get('steps') or []}
        decided = [s for s in res['statuses'] if s['status'] != 'unverified']

        # 1. everything decided names steps that exist
        bad = [s['id'] for s in decided
               if not s['cites'] or set(s['cites']) - real]
        check(not bad, f'{g}: all {len(decided)} decided items cite real steps'
                       + (f' (offenders: {bad})' if bad else ''))

        # 2. no defect rests on a place the session never reached
        text, spans = BV.trace_text(exp)
        hay = TC._tokens(text)
        nr = spans.get('not_reached', (0, 0))
        leaks = []
        for s in decided:
            if s['status'] == 'confirmed_gap' and s['basis'] == 'trace':
                score, at = TC.quote_locate(s['quote'], hay)
                if score >= TC._QUOTE_MATCH and nr[0] <= at < nr[1]:
                    leaks.append(s['id'])
        check(not leaks, f'{g}: no confirmed_gap grounded in not_reached'
                         + (f' ({leaks})' if leaks else ''))

        # 3. the share left open is neither near zero nor near everything
        check(30 <= t['unverified_pct'] <= 80,
              f'{g}: {t["unverified_pct"]}% unverified is inside 30-80%')

        # 4. general checks are library entries, 3 to 5 of them
        ids = [x['id'] for x in res['general']]
        check(all(i in BV.GENERAL_CHECKS for i in ids) and len(ids) == len(set(ids)),
              f'{g}: general checks are distinct library ids {ids}')
        check(3 <= len(ids) <= 5, f'{g}: {len(ids)} general checks, wanted 3-5')
        print()

    print('=' * 78)
    print(f'  {"game":<24}{"items":>6}{"sat":>5}{"gap":>5}{"unv":>5}'
          f'{"unv%":>6}{"demoted":>9}{"gen":>5}')
    for r in rows:
        print(f'  {r[0]:<24}{r[1]:>6}{r[2]:>5}{r[3]:>5}{r[4]:>5}'
              f'{r[5]:>6}{r[6]:>9}{r[7]:>5}')
    dem = sum(r[6] for r in rows)
    print(f'\n  gates demoted {dem} claim(s) across {len(rows)} games'
          + ('  -- if this is 0 the gates are decoration, say so'
             if dem == 0 else ''))
    print(f'\n{"ALL PASS" if not fails else str(len(fails)) + " FAILED"}')
    for f in fails:
        print('  - ' + f)
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
