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
"""Acceptance for Part 6's map. Offline, no API, no repair session.

Two properties were asked for by name, and both have a way to fail silently:

  line drift   the map is regenerated every round from source, so a stored
               annotation has to survive code moving underneath it. If it does
               not, the map is worse than nothing: it points at the wrong line
               with full confidence.
  isolation    one file per game, and a run this host kills mid-write must not
               leave the next round starting from an empty map.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN))
import rsigame.view.project_map as PM   # noqa: E402

G0 = RUN / 'godot' / 'corpus_gpt'
fails = []


def check(cond, msg):
    print(('  PASS  ' if cond else '  FAIL  ') + msg)
    if not cond:
        fails.append(msg)


def main():
    tmp = Path(tempfile.mkdtemp(prefix='pmtest-'))
    try:
        work = tmp / 'work'
        shutil.copytree(G0 / 'shooter-sky-duel', work, symlinks=True,
                        ignore=shutil.ignore_patterns('.godot'))
        gd = work / 'scripts' / 'Main.gd'

        s0 = PM.scan(work)
        f = 'scripts/Main.gd'
        check(s0['n_functions'] > 50,
              f"scanned {s0['n_functions']} functions in {len(s0['files'])} file(s)")

        # pick three functions spread through the file and describe them
        fns = s0['files'][f]['functions']
        picks = [fns[2], fns[len(fns) // 2], fns[-2]]
        notes = {f"{f}::{p['name']}": f"described {p['name']}" for p in picks}
        before = {p['name']: p['line'] for p in picks}
        print(f"    described: " + ', '.join(f"{k}@{v}" for k, v in before.items()))

        # 1. LINE DRIFT. Insert 40 lines near the top, which moves everything
        #    below it, then rescan and re-merge.
        src = gd.read_text()
        i = src.index('\nfunc ')
        gd.write_text(src[:i] + '\n' + '\n'.join(f'# filler {n}'
                                                 for n in range(40)) + src[i:])
        s1 = PM.scan(work)
        m1 = PM.merge(s1, notes)
        after = {fn['name']: fn['line'] for fn in s1['files'][f]['functions']
                 if fn['name'] in before}
        moved = [n for n in before if after.get(n, 0) == before[n] + 40]
        print(f"    after inserting 40 lines: "
              + ', '.join(f"{k}@{after.get(k)}" for k in before))
        check(len(moved) == len(before),
              f'all {len(before)} line numbers moved by exactly +40')
        check(m1['n_notes'] == len(notes) and not m1['lost_notes'],
              f"all {len(notes)} annotations survived the shift "
              f"(lost: {m1['lost_notes']})")
        rendered = PM.render(m1)
        shown = [n for n in before
                 if f"{after[n]}" in rendered and f'described {n}' in rendered]
        check(len(shown) == len(before),
              f'the rendered map carries all {len(before)} annotations at '
              f'their NEW line numbers')

        # 2. A RENAMED FUNCTION LOSES ITS NOTE, AND SAYS SO.
        victim = picks[0]['name']
        gd.write_text(gd.read_text().replace(f'func {victim}(',
                                             f'func {victim}_renamed('))
        m2 = PM.merge(PM.scan(work), notes)
        check(m2['n_notes'] == len(notes) - 1
              and any(victim in x for x in m2['lost_notes']),
              f'renaming {victim} drops its note and reports it as lost')

        # 3. ISOLATION. Two games, two directories, no bleed.
        a, b = tmp / 'runs' / 'game-a', tmp / 'runs' / 'game-b'
        PM.save({**s0, 'notes': {'x::y': 'A'}}, a)
        PM.save({**s0, 'notes': {'x::y': 'B'}}, b)
        check(PM.load_notes(a) == {'x::y': 'A'}
              and PM.load_notes(b) == {'x::y': 'B'},
              'two games keep separate maps at separate paths')
        check(PM.path_for(a) != PM.path_for(b)
              and PM.path_for(a).name == 'project_map.json',
              f'path is <run>/<game>/project_map.json')

        # 4. NOT IN THE WORK TREE -- it must not enter the round's diff.
        check(not (work / 'project_map.json').exists()
              and 'work' not in PM.path_for(a).parts,
              'the map is not written into the work tree')

        # 5. A KILLED WRITE LEAVES THE OLD MAP, NOT A TRUNCATED ONE.
        PM.save({**s0, 'notes': {'x::y': 'first'}}, a)
        (PM.path_for(a).with_suffix('.json.tmp')).write_text('{"notes": {"x')
        check(PM.load_notes(a) == {'x::y': 'first'},
              'a half-written temp file does not become the map')

        # 6. AN INVENTED KEY IS REFUSED.
        m3, n_new = PM.take_notes(PM.merge(PM.scan(work), {}),
                                  {f'{f}::no_such_function': 'nonsense',
                                   f'{f}::{fns[5]["name"]}': 'real one'})
        check(n_new == 1 and f'{f}::no_such_function' not in m3['notes'],
              'a description for a function that does not exist is dropped')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f'\n{"ALL PASS" if not fails else str(len(fails)) + " FAILED"}')
    for x in fails:
        print('  - ' + x)
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
