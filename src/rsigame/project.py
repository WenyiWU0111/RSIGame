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
"""The game project: where a corpus lives, and how one tree is handled.

Copying a frozen game out of its corpus, deciding which engine it is, building
it, importing it under Godot, reading back what its demos declare, and counting
what a repair attempt cost. None of this is specific to how development is
driven, which is why it sits apart from the loop that drives it.

Extracted from `self_refine.py`, where it had grown up alongside a baseline arm
that no longer ships. The loop imported that file for these thirteen names and
for nothing else, so the file's departure is a move, not a rewrite.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from rsigame import paths

# WHERE THINGS ARE, ASKED RATHER THAN ASSUMED.
#
# These were three absolute paths under one user's home. That is fine while
# the arm only ever runs on the machine it was written on, and it is the first
# thing that fails on any other -- with an import error a long way from the
# line that caused it. Each now has an environment variable and a default that
# reproduces the original layout, so this host is unchanged and a second host
# needs no edits, only exports. `preflight.py` checks all of them before a
# batch starts rather than after the first game dies.
RUN = Path(os.environ.get('RSIGAME_PATHS_RUN')
           or Path(__file__).resolve().parent)


# The agent's code, which the repair prompt tells the agent where to find.
# It is this package: there is no second checkout to point at any more.
REPO = paths.agent_repo()


# Which corpus this run repairs. One arm per corpus, selected on the command
# line, because they differ in engine and generator and a run that mixed them
# would produce a table row nobody could attribute.
#
# A corpus may also be given as a PATH, which is what a second machine needs:
# its G0 trees arrive as a directory and giving it a name here would mean
# editing this file to run them.
CORPORA = {
    'web_gpt':   RUN / 'corpus/gamecraft141_phaser_gpt',
    'web_qwen':  RUN / 'corpus/gamecraft141_phaser_qwen',
    'godot_gpt': RUN / 'godot/corpus_gpt',
    'godot_qwen': RUN / 'godot/corpus_qwen',
    # Kimi-k2.6 and GLM-5.3-flash base games, lifted out of the
    # gamecraft-bench job trials.  Those jobs ran with the stub judge,
    # so every reward is 0.0 and cannot break ties between reruns; the
    # most recent non-aborted trial per game was taken instead.
    'godot_kimi': RUN / 'godot/corpus_kimi',
    'godot_glm':  RUN / 'godot/corpus_glm',
}


CORPUS = CORPORA['web_gpt']          # rebound in main() from --corpus


REPAIR_TOOLS = int(os.environ.get('RSIGAME_LOOP_REPAIR_TOOLS') or 26)


BUILD_TIMEOUT_S = int(os.environ.get('RSIGAME_LOOP_BUILD_TIMEOUT_S') or 600)


def declared_scenarios(work: Path) -> list:
    """The named states this game's own source dispatches on.

    Read from the game rather than from a list kept beside it: a game is the
    only authority on what it will do with an id, and a stale inventory would
    send play into a silent fallback that looks like the state it asked for.
    """
    import re
    src = ''.join(f.read_text(errors='replace') for f in work.rglob('*.gd'))
    out = set()
    for m in re.finditer(
            r'(?:func\s+_?(?:start_from|load|apply|setup)_scenario'
            r'[\s\S]{0,6000}?)(?=\nfunc |\Z)', src):
        out |= set(re.findall(r'^[ \t]+"([a-z0-9_]{2,40})"\s*:\s*$',
                              m.group(0), re.M))
    return sorted(out)


RSIGAME_MODELS_LOOP_MODEL = os.environ.get('RSIGAME_MODELS_LOOP_MODEL')       # reads the trace, self-critique


RSIGAME_REPAIR_MODEL = os.environ.get('RSIGAME_REPAIR_MODEL')   # the coding half


# The directory holding a node new enough for the agent CLI (>= 18). Defaults
# to whatever `node` is on PATH so a second machine needs no export; this
# host's own node is v12, which is why the original default named an env.
NODE_BIN = os.environ.get('RSIGAME_JUDGE_NODE_BIN') or (
    str(Path(shutil.which('node')).parent) if shutil.which('node') else '')


def repair_cost(round_dir_tag: str, work: Path, proxy=None,
                before: dict | None = None) -> dict:
    """Every attempt this round made, from the logs `agent_repair.ts` wrote.

    `_run_repair_agent` returns only its LAST attempt, and a round that
    retried spent the earlier attempts' tokens and tool calls just as really.
    Reading the directory instead of the return value is what makes "actual
    spend" actual.
    """
    base = work.parent / f'_repairs_{work.name}'
    tot = {'attempts': 0, 'n_tools': 0, 'input': 0, 'output': 0,
           'cache_read': 0, 'cache_creation': 0, 'n_play_calls': 0,
           'usage_missing': 0}
    tot_sdk_in = tot_sdk_out = 0
    for lg in sorted(base.glob(f'*-{round_dir_tag}/log.json')):
        try:
            d = json.load(open(lg))
        except Exception:
            continue
        tot['attempts'] += 1
        tot['n_tools'] += d.get('n_tools') or 0
        tot['n_play_calls'] += d.get('n_play_calls') or 0
        c = d.get('cost') or {}
        if not c:
            tot['usage_missing'] += 1
            continue
        tot_sdk_in += c.get('inputTokens') or 0
        tot_sdk_out += c.get('outputTokens') or 0
        tot['input'] += c.get('inputTokens') or 0
        tot['output'] += c.get('outputTokens') or 0
        tot['cache_read'] += c.get('cacheReadInputTokens') or 0
        tot['cache_creation'] += c.get('cacheCreationInputTokens') or 0

    # THE TOKENS COME FROM THE PROXY, NOT FROM `log.json`.
    #
    # `cost` is null on this endpoint -- measured on a real session, with no
    # token field in `.qwen/trajectory.jsonl` either. The `cost`-derived
    # numbers above are kept because a future endpoint may populate them and
    # because a disagreement between the two is worth being able to see, but
    # the proxy's count is what this round SPENT and it is what is reported.
    if proxy is not None:
        now = proxy.snapshot()
        b = before or {}
        d = {k: now.get(k, 0) - b.get(k, 0) for k in now}
        tot['input'] = d.get('input', 0)
        tot['output'] = d.get('output', 0)
        tot['cache_read'] = d.get('cached', 0)
        tot['reasoning'] = d.get('reasoning', 0)
        tot['cost_usd'] = round(d.get('cost_usd', 0.0), 6)
        tot['proxy_calls'] = d.get('calls', 0)
        tot['proxy_calls_without_usage'] = d.get('calls_without_usage', 0)
        tot['proxy_errors'] = d.get('errors', 0)
        tot['from_sdk_cost'] = {'input': tot_sdk_in, 'output': tot_sdk_out}
    return tot


def fresh_copy(game: str, work: Path, corpus: Path | None = None) -> Path:
    """A byte-identical G0, never the corpus original.

    `corpus` is explicit for callers outside main(). ALL FORTY GAMES EXIST IN
    BOTH CORPORA UNDER THE SAME NAME, and `CORPUS` is a module global that
    main() rebinds from --corpus -- so a programmatic caller that just imports
    this module gets the WEB tree for a Godot game, silently, and everything
    downstream looks fine until the engine matters. Cost of learning that: one
    17-minute repair session that handed a Phaser game a page of Godot
    instructions and edited nothing.

    `node_modules` is symlinked rather than copied: 141 games share 28
    dependency trees through `.store/<closure>`, and copying one is 414 MB for
    a tree that must not differ anyway. `dist` is left out because every round
    builds it.
    """
    src = (corpus or CORPUS) / game
    if not src.is_dir():
        raise SystemExit(f'no such game in the frozen corpus: {game}')
    if work.exists():
        shutil.rmtree(work)
    work.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, work, symlinks=True,
                    ignore=shutil.ignore_patterns('node_modules', 'dist',
                                                  '.git', '.qwen', '.godot'))
    # Web only: the dependency tree is shared through `.store/<closure>` and
    # symlinked, never copied. A Godot project has no such tree -- its
    # equivalent, `.godot`, is an import cache the engine rebuilds on open and
    # is excluded above rather than linked.
    if (src / 'node_modules').exists() and not (work / 'node_modules').exists():
        os.symlink(src / 'node_modules', work / 'node_modules')

    # GODOT NEEDS ONE IMPORT PASS, and without it the arm repairs a phantom.
    #
    # `.godot` is the import cache and is excluded above, so a fresh copy has
    # none. `godot_build.check_project` does not import -- it runs
    # `--headless --quit` directly -- so on an unimported tree that first run
    # emits resource errors while it imports, the FATAL regex matches them,
    # and the round opens with `build does not compile`. Measured on
    # cardgame-inscription-dark: round 1's critique was "this build does not
    # compile, so it cannot be played at all" for a project that loads fine.
    # The arm would then spend its whole budget fixing a build that was never
    # broken.
    #
    # One pass costs 3 seconds and is setup, not repair: it is this engine's
    # `npm install`, and the corpus originals already ship the cache it
    # produces, so it puts the work tree in the SAME state as G0 rather than a
    # different one.
    if (work / 'project.godot').is_file():
        try:
            subprocess.run(['godot', '--headless', '--path', str(work),
                            '--import', '--quit'],
                           capture_output=True, text=True, timeout=300)
        except Exception:
            pass          # a failed import shows up as `build does not
                          # compile`, which is then a real finding
    return src


def engine_of(work: Path) -> str:
    """What kind of project this is, decided by what is in the directory.

    The same test `verifier_agent._session_for` uses, so the build check and
    the play session can never disagree about which engine they are looking
    at -- a disagreement that would show up as a game that builds and cannot
    be played, or the reverse, with nothing saying why.
    """
    return 'godot' if (work / 'project.godot').is_file() else 'web'


def build(work: Path) -> dict:
    """Does this tree build? `npm run build` on web, a headless project load
    on Godot.

    GODOT EXITS 0 ON A SCRIPT THAT FAILS TO PARSE, which is why this defers to
    `godot_build.check_project` rather than running the binary here: that
    helper already reads the OUTPUT rather than the exit code, and the repair
    runbook records the failure mode it exists to catch.
    """
    if engine_of(work) == 'godot':
        t0 = time.time()
        try:
            from rsigame.agent.evolve import godot_build
            r = godot_build.check_project(work, timeout_s=BUILD_TIMEOUT_S)
            # THE KEY IS `builds`, NOT `ok`. Read wrong, this returns False for
            # a project that loads perfectly, every round, for every Godot
            # game -- and the arm would spend its whole budget "fixing" a build
            # that was never broken, with the log saying only `build=False`.
            ok = bool(r.get('builds') if isinstance(r, dict) else r)
            tail = '' if ok else json.dumps(r, ensure_ascii=False)[-1500:]
        except Exception as exc:
            ok, tail = False, f'{type(exc).__name__}: {str(exc)[:400]}'
        return {'ok': ok, 'wall_s': round(time.time() - t0, 1), 'tail': tail}
    env = dict(os.environ)
    env['PATH'] = f"{NODE_BIN}:{env.get('PATH', '')}"
    t0 = time.time()
    try:
        p = subprocess.run(['npm', 'run', 'build'], cwd=str(work), env=env,
                           capture_output=True, text=True,
                           timeout=BUILD_TIMEOUT_S)
        ok = p.returncode == 0 and (work / 'dist' / 'index.html').is_file()
        tail = ((p.stderr or '') + (p.stdout or ''))[-1500:] if not ok else ''
    except subprocess.TimeoutExpired:
        ok, tail = False, f'the build did not finish in {BUILD_TIMEOUT_S}s'
    return {'ok': ok, 'wall_s': round(time.time() - t0, 1), 'tail': tail}


def unimported(work: Path) -> list:
    """Resource files with no `.import` beside them."""
    out = []
    for p in Path(work).rglob('*'):
        if not p.is_file() or p.suffix.lower() not in _IMPORTABLE:
            continue
        if any(part in ('.godot', '.git', 'node_modules', '_repair_evidence',
                        'demo_outputs') for part in p.parts):
            continue
        if not p.with_suffix(p.suffix + '.import').is_file():
            out.append(str(p.relative_to(work)))
    return out


def godot_import(work: Path, timeout_s: int = 300,
                 force: bool = False) -> dict:
    """Import anything new, so `load()` can find it. No-op when nothing is new.

    Returns what it did, for the round record: a round that generated art and
    a round that did not must be distinguishable months later.
    """
    work = Path(work).resolve()

    # NEVER ON THE FROZEN CORPUS. An import pass WRITES: it creates a `.import`
    # beside every resource, a `.godot/` cache, and a `.uid` beside each script.
    # Run against the corpus it silently rewrites the G0 every arm starts from,
    # and every future `fresh_copy` would carry the difference. Done by accident
    # while testing this very function -- 62 `.import` files, a `.godot/` and a
    # `Main.gd.uid` appeared in shooter-sky-duel's G0 and had to be removed by
    # hand. A read-only-looking helper that writes needs the guard at the point
    # of writing, not in the caller's discipline.
    # EVERY corpus, not the one this run happens to have selected: `CORPUS` is
    # rebound in main() from --corpus, so a guard reading it would protect only
    # the arm currently running and leave the other two open to exactly the
    # accident it exists to prevent.
    for root in CORPORA.values():
        root = root.resolve()
        if work == root or root in work.parents:
            raise SystemExit(
                f'refusing to import inside the frozen corpus: {work}')

    if not (work / 'project.godot').is_file():
        return {'ran': False, 'why': 'not a godot project'}
    pending = unimported(work)

    # A MISSING CACHE IS ALSO SOMETHING TO IMPORT.
    #
    # `unimported` asks whether each resource has an `.import` file beside it,
    # and a corpus game ships those -- `fresh_copy` copies them. What it does
    # NOT copy is `.godot/`, deliberately, because that is a cache one machine
    # built. But `.import` only NAMES the cache entry: it points at
    # `res://.godot/imported/<hash>.ctex`, and that is the file `load()` opens.
    # So a game with every `.import` present and no cache reports nothing to
    # import, while none of its textures can be loaded.
    #
    # The engine does rebuild the cache when it opens a project, which is why
    # this hid: on a quiet machine the rebuild finishes inside the build
    # check's own short window and everything looks fine. Under five parallel
    # lines it does not, and the round sees a project that will not load.
    # puzzle-circuit-wizard spent fifteen rounds that way. Importing here waits
    # for it properly instead of racing it.
    cache = work / '.godot' / 'imported'
    has_imports = any(work.rglob('*.import'))
    cold = has_imports and not (cache.is_dir() and any(cache.iterdir()))
    if not pending and not cold and not force:
        return {'ran': False, 'why': 'every resource already imported'}
    t0 = time.time()
    try:
        subprocess.run(['godot', '--headless', '--path', str(work),
                        '--import', '--quit'],
                       capture_output=True, timeout=timeout_s)
    except Exception as exc:
        return {'ran': True, 'ok': False, 'pending': pending,
                'error': f'{type(exc).__name__}: {str(exc)[:200]}'}
    left = unimported(work)
    warm = cache.is_dir() and any(cache.iterdir())
    return {'ran': True, 'ok': (not left) and (warm or not has_imports),
            'imported': len(pending) - len(left), 'cold_cache': cold,
            'cache_built': warm, 'pending': pending[:20],
            'still_missing': left[:20],
            'wall_s': round(time.time() - t0, 1)}


# What "it must build" means, per engine. Godot exits 0 on a script that fails
# to parse, so the agent is told to read the OUTPUT -- the same trap
# `godot_build.check_project` exists to avoid on our side.
BUILD_CMD = {
    'web': 'npm run build',
    'godot': 'godot --headless --path . --quit  (read its OUTPUT, not its exit '
             'code: Godot exits 0 on a script that fails to parse)',
}


def _map_notes_from(rr: dict) -> dict:
    """The `project_map: {...}` block from the agent's closing message.

    Read out of its prose rather than asked for as a separate call: the agent
    already writes a summary of what it changed, and one more request is one
    more chance to fail. A message with no block is the normal case and returns
    nothing -- describing the code is optional, and a round that skipped it
    still gets the map it was given.
    """
    msgs = [m for m in (rr.get('messages') or []) if isinstance(m, str)]
    for text in reversed(msgs):
        i = text.lower().rfind('project_map')
        if i < 0:
            continue
        j = text.find('{', i)
        if j < 0:
            continue
        depth = 0
        for k in range(j, len(text)):
            if text[k] == '{':
                depth += 1
            elif text[k] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        got = json.loads(text[j:k + 1])
                        if isinstance(got, dict):
                            return got
                    except Exception:
                        pass
                    break
    return {}
