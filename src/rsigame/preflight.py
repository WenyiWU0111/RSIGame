#!/usr/bin/env python
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
"""Refuse to start a batch that cannot finish.

Every check here corresponds to something that has actually cost a run on the
machine this was written on, and each failed in the same expensive shape: the
batch started, looked healthy, and produced nothing usable hours later.

  - the wrong interpreter          every play session died on its first model
                                   call, once per game, and every round then
                                   repaired against a critique that saw nothing
  - node v12 on PATH               the agent CLI is ESM; the failure is a
                                   SyntaxError inside a subprocess
  - a stale CLI bundle             the fix was in packages/core and the agent
                                   kept spawning a bundle that predated it
  - no Godot binary                every replay raises, every round scores the
                                   build as unplayable
  - missing model names            the run gets to its first call before
                                   finding out
  - a corpus that is not one       the arm copies an empty tree and repairs it

Run:  python preflight.py [--corpus NAME_OR_PATH] [--list FILE]
Exit: 0 all good, 1 something would have broken the batch.
"""

from __future__ import annotations
from . import paths

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

RUN = Path(os.environ.get('RSIGAME_PATHS_RUN') or Path(__file__).resolve().parent)
fails: list[str] = []
warns: list[str] = []


def ok(cond, msg, *, warn_only=False):
    print(('  ok    ' if cond else ('  warn  ' if warn_only else '  FAIL  ')) + msg)
    if not cond:
        (warns if warn_only else fails).append(msg)
    return cond


def main() -> int:
    from . import config
    config.ensure()
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default='godot_gpt')
    ap.add_argument('--list', default=None, help='games list to check against '
                                                 'the corpus')
    a = ap.parse_args()

    print(f'preflight  RSIGAME_PATHS_RUN={RUN}\n')

    # 1. the interpreter the batch will actually use -------------------------
    py = os.environ.get('RSIGAME_AGENT_PYTHON') or sys.executable
    ok(Path(py).is_file(), f'RSIGAME_AGENT_PYTHON exists: {py}')
    for mod in ('openai', 'playwright', 'PIL'):
        r = subprocess.run([py, '-c', f'import {mod}'], capture_output=True)
        ok(r.returncode == 0, f'{py} can import {mod}')

    # 2. the trees ----------------------------------------------------------
    repo = paths.agent_repo()
    bench = Path(os.environ.get('GAMECRAFT_BENCH')
                 or RUN.parent / 'gamecraft-bench')
    ok((repo / 'agent').is_dir(), f'the agent package is importable: {repo}')
    ok((bench / 'tasks').is_dir(),
       f'bench has tasks/: {bench}   (GAMECRAFT_BENCH)')
    n_tasks = len(list((bench / 'tasks').iterdir())) if (bench / 'tasks').is_dir() else 0
    ok(n_tasks > 0, f'bench holds {n_tasks} task directories')

    # 3. the corpus ---------------------------------------------------------
    named = {'web_gpt': RUN / 'corpus/gamecraft141_phaser_gpt',
             'godot_gpt': RUN / 'godot/corpus_gpt',
             'godot_qwen': RUN / 'godot/corpus_qwen'}
    cdir = named.get(a.corpus) or Path(a.corpus).expanduser()
    if ok(cdir.is_dir(), f'corpus is a directory: {cdir}'):
        games = sorted(p for p in cdir.iterdir() if p.is_dir())
        ok(len(games) > 0, f'corpus holds {len(games)} games')
        # A game the arm can start from: a project file and the demos the
        # evaluator replays. A corpus of empty directories passes every other
        # check and produces fifteen rounds of nothing.
        bad = [g.name for g in games[:200]
               if not ((g / 'project.godot').is_file() or (g / 'index.html').is_file()
                       or (g / 'package.json').is_file())]
        ok(not bad, 'every game has a project file'
                    + (f' (missing in {len(bad)}: {bad[:5]})' if bad else ''))
        nodemo = [g.name for g in games[:200] if not (g / 'demo_outputs').is_dir()]
        ok(not nodemo, 'every game ships demo_outputs/'
                       + (f' (missing in {len(nodemo)}: {nodemo[:5]})' if nodemo else ''))
        # The task document is what the checklist is extracted FROM; a game
        # with no task in the bench cannot be scored or read.
        notask = [g.name for g in games[:200]
                  if not (bench / 'tasks' / g.name / 'instruction.md').is_file()]
        ok(not notask, 'every game has instruction.md in the bench'
                       + (f' (missing for {len(notask)}: {notask[:5]})' if notask else ''))
        if a.list:
            want = [l.strip() for l in Path(a.list).read_text().split() if l.strip()]
            have = {g.name for g in games}
            miss = [w for w in want if w not in have]
            ok(not miss, f'all {len(want)} games in {a.list} are in the corpus'
                         + (f' (missing {len(miss)}: {miss[:5]})' if miss else ''))

    # 4. the engine ---------------------------------------------------------
    godot = os.environ.get('GAMECRAFT_BENCH_GODOT_BIN') or shutil.which('godot')
    if ok(bool(godot), 'a Godot binary is on PATH or in '
                       'GAMECRAFT_BENCH_GODOT_BIN'):
        r = subprocess.run([godot, '--version'], capture_output=True, text=True)
        ok(r.returncode == 0, f'godot --version -> {(r.stdout or r.stderr).strip()[:60]}')
    for tool in ('Xvfb', 'xdotool', 'ffmpeg'):
        ok(shutil.which(tool) is not None, f'{tool} is on PATH (replay needs it)')

    # 5. node and the agent CLI --------------------------------------------
    nb = os.environ.get('RSIGAME_JUDGE_NODE_BIN')
    node = (Path(nb) / 'node') if nb else (shutil.which('node') or '')
    if ok(bool(node) and Path(node).exists(), f'node found: {node}'):
        v = subprocess.run([str(node), '--version'], capture_output=True,
                           text=True).stdout.strip()
        major = int(v.lstrip('v').split('.')[0] or 0)
        ok(major >= 18, f'node is {v} (>= 18 required; the CLI is ESM)')
    cli = repo.parent / 'dist' / 'cli.js'
    ok(cli.is_file(), f'the agent CLI bundle exists: {cli}'
                      '  -- build it with packages/sdk-typescript/scripts/build.js')

    # 6. models and keys ----------------------------------------------------
    for k in ('RSIGAME_MODELS_PLAY_MODEL', 'RSIGAME_MODELS_LOOP_MODEL', 'RSIGAME_REPAIR_MODEL'):
        ok(bool(os.environ.get(k)), f'{k} is set ({os.environ.get(k) or "unset"})')
    key = any(os.environ.get(k) for k in
              ('OPENROUTER_API_KEY', 'OPENAI_API_KEY', 'OPENAI_COMPAT_API_KEY'))
    ok(key, 'a model API key is in the environment')

    # 6b. image generation -------------------------------------------------
    # EvoGame's art rounds call generate_game_assets (unlike the repair line,
    # which never does). Read from the environment, falling back to
    # agent-test/.env, which the node agent loads with dotenv.
    envf = {}
    for f in (repo / '.env', Path(__file__).resolve().parent / '.env'):
        if f.is_file():
            for line in f.read_text().splitlines():
                if '=' in line and not line.lstrip().startswith('#'):
                    k, v = line.split('=', 1)
                    envf.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    img = lambda k: os.environ.get(k) or envf.get(k, '')
    ok(img('OPENGAME_ASSET_MODE') == 'generate', f"OPENGAME_ASSET_MODE=generate ({img('OPENGAME_ASSET_MODE') or 'unset'})")
    ok(img('OPENGAME_IMAGE_PROVIDER') == 'openai-compat',
       f"OPENGAME_IMAGE_PROVIDER=openai-compat ({img('OPENGAME_IMAGE_PROVIDER') or 'unset'})")
    ok(img('OPENGAME_IMAGE_BASE_URL').rstrip('/') == 'https://openrouter.ai/api/v1',
       f"OPENGAME_IMAGE_BASE_URL is OpenRouter itself, not a local proxy ({img('OPENGAME_IMAGE_BASE_URL') or 'unset'})")
    want_img = os.environ.get('RSIGAME_ASSETS_EXPECT_IMAGE_MODEL', '')
    if want_img:      # checked only when an experiment pins it; no model name is hardcoded here any more
        ok(img('OPENGAME_IMAGE_MODEL') == want_img,
           f"OPENGAME_IMAGE_MODEL={want_img} ({img('OPENGAME_IMAGE_MODEL') or 'unset'})")
    mp = img('OPENGAME_IMAGE_MIN_PIXELS')
    ok(mp.isdigit() and int(mp) >= 3686400,
       f'OPENGAME_IMAGE_MIN_PIXELS >= 3686400 ({mp or "unset"}; seedream refuses smaller requests and every sprite call fails)')
    ok(bool(img('OPENGAME_IMAGE_API_KEY')), 'OPENGAME_IMAGE_API_KEY is set')

    # 7. disk ---------------------------------------------------------------
    free = shutil.disk_usage(RUN).free / 2**30
    ok(free > 50, f'{free:.0f} GB free under {RUN} '
                  f'(a 141-game arm writes tens of GB of round diffs)',
       warn_only=free > 20)

    print()
    if fails:
        print(f'{len(fails)} check(s) would have broken the batch:')
        for f in fails:
            print('  - ' + f)
    if warns:
        print(f'{len(warns)} warning(s):')
        for w in warns:
            print('  - ' + w)
    if not fails:
        print('preflight passed' + (' (with warnings)' if warns else ''))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
