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
"""How much the task checklist moves when the task does not. Offline analysis.

Part 3 wants the checklist to be a persistent object carried across rounds, so
how repeatable it is decides whether that is even possible.

MEASURED BY REQUIREMENT, NOT BY QUOTE. The first pass keyed identity on the
quote and reported a Jaccard of 0.00 for idle-spell-tower -- no quote at all
common to three runs. That is almost certainly the wrong question: the same
requirement can honestly be sourced to a different sentence each time, and the
document says most things more than once. Keying on the quote makes rewording
the citation look like changing the requirement, which is the same over-strict
identity mistake that dropped a true item from horror-tape-archive.

So this stores every run and matches requirement text across them, greedily,
at the same 0.70 the quote check uses. It reports both numbers, because if they
disagree the disagreement is the finding.
"""

from .. import paths
import difflib
import json
import sys
import time
from pathlib import Path

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN))
sys.path.insert(0, str(paths.agent_repo()))
import rsigame.checklist.task_checklist as TC   # noqa: E402

OUT = RUN / '_scratch' / 'stability'
K = 3
GAMES: list[str] = []      # passed with --games; no experiment list is read at import
MATCH = 0.70


def sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, TC._tokens(a), TC._tokens(b),
                                   autojunk=False).ratio()


def pair(xs: list, ys: list) -> list:
    """Greedy best-first matching between two runs' requirement lists."""
    cand = sorted(((sim(x, y), i, j) for i, x in enumerate(xs)
                   for j, y in enumerate(ys)), reverse=True)
    ui, uj, out = set(), set(), []
    for s, i, j in cand:
        if s < MATCH or i in ui or j in uj:
            continue
        ui.add(i); uj.add(j); out.append((i, j, s))
    return out


def main():
    from rsigame import config
    config.ensure()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f'{len(GAMES)} games, {K} uncached extractions each\n')
    print(f'  {"game":<24}{"items":>16}{"in all 3":>10}{"by quote":>10}'
          f'{"core %":>9}')
    for g in GAMES:
        runs = []
        for k in range(K):
            f = OUT / f'{g}__{k}.json'
            if f.is_file():
                runs.append(json.loads(f.read_text()))
                continue
            r = TC.extract(g, use_cache=False)
            if r.get('error'):
                print(f'  {g}: {r["error"]}')
                continue
            f.write_text(json.dumps(r, ensure_ascii=False, indent=1))
            runs.append(r)
            time.sleep(2)
        if len(runs) < 2:
            continue
        reqs = [[i['requirement'] for i in r['items']] for r in runs]

        # an item of run 0 survives if it has a counterpart in every other run
        keep = 0
        maps = [dict((i, j) for i, j, _ in pair(reqs[0], reqs[n]))
                for n in range(1, len(runs))]
        for i in range(len(reqs[0])):
            if all(i in m for m in maps):
                keep += 1
        qs = [{TC._norm(i['quote']) for i in r['items']} for r in runs]
        byq = len(set.intersection(*qs))
        n0 = max(1, len(reqs[0]))
        print(f'  {g:<24}{str([len(x) for x in reqs]):>16}{keep:>10}'
              f'{byq:>10}{100.0 * keep / n0:>8.0f}%')
    print('\n  "in all 3" = requirements from run 1 with a >=0.70 counterpart '
          'in BOTH other runs\n  "by quote" = the old measure, identical quote '
          'text in all three')


if __name__ == '__main__':
    sys.exit(main())
