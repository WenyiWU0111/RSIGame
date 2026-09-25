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
"""Value Monitor, replayed offline over one run's checkpoints.

For every game the champion starts at round 0 (the base game). At each
checkpoint r = 3, 6, ..., 30 the estimator (estimator/score_pair, proxy rubric,
OpenRouter qwen3.8-flash -- never the official rubric or score) compares the
champion against the new checkpoint in one call per demo. The checkpoint
becomes the champion when its proxy overall beats the champion's by at least
gain(). Never adopted: a checkpoint whose build failed in the loop, or whose own
recordings show GDScript parse errors. A champion whose footage grades nothing
(old None) loses to a candidate whose footage grades (champion_unscored).

Phaser (web) trees -- a `package.json` instead of a `project.godot` -- take the
same path with three engine substitutions: a checkpoint counts as recorded when
its tree is scored, "does not parse" becomes "fails the web start-up check"
(`tools/web_build_check.py` on the checkpoint's own `dist/`, the check the web
loop now runs after every build), and a champion is re-recorded with the web
verifier (stub judge). Set GAMECRAFT_BENCH to the gamecraft-bench-web checkout.

Frames: the scorer's pass-1 recording of each demo -- round 0 from the base-game
scores (RSIGAME_MONITOR_R0_DIR/<game>/p1), round r from RSIGAME_MONITOR_RUN/runs/<game>/round_rr/score/p1
(score_checkpoints.py; RSIGAME_MONITOR_JUDGE=stub records without judging). When champion
and candidate share no demo name, the champion is re-recorded on the
candidate's demo inputs (stub judge, one pass).

    source scripts/evogame/env.sh
    RSIGAME_MONITOR_RUN=runs/evogame_glm30 RSIGAME_MONITOR_R0_DIR=r0_scores/godot_glm RSIGAME_MONITOR_R0_TREES=basegames/godot_glm \
    RSIGAME_MONITOR_OUT=vm/glm30 python scripts/evogame/vm_replay.py --games handoff/main_glm120.txt [--jobs 6]

RSIGAME_MONITOR_R0_TREES/<game> (or <game>/game) is the base game's project. Writes
RSIGAME_MONITOR_OUT/<game>.jsonl (one line per checkpoint, resumable) and RSIGAME_MONITOR_OUT/champ.json.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..eval.estimator import pairs as P, score_pair, aggregate
from ..eval.estimator.rubric import load as load_rubric

# WHERE THE CHECKPOINTS ARE. Resolved on first use, not at import: the live
# monitor imports this module inside a running loop, where RSIGAME_MONITOR_RUN and friends
# are not set, and an import that reads them would take the whole loop down.
SELF = R0 = OUT = ROOT = R0_TREES = None


def setup(run=None, r0_dir=None, out=None, r0_trees=None) -> None:
    """Point the module at one experiment's directories. Arguments win over
    the RSIGAME_MONITOR_* environment variables, which is how the live monitor
    passes a run that has none of them set."""
    global SELF, R0, OUT, ROOT, R0_TREES
    SELF = Path(run or os.environ['RSIGAME_MONITOR_RUN']).resolve() / 'runs'
    R0 = Path(r0_dir or os.environ['RSIGAME_MONITOR_R0_DIR']).resolve()
    OUT = Path(out or os.environ['RSIGAME_MONITOR_OUT']).resolve()
    ROOT = OUT / 'runs'
    R0_TREES = Path(r0_trees or os.environ.get('RSIGAME_MONITOR_R0_TREES') or R0).resolve()
    os.environ['RSIGAME_MONITOR_RUNS'] = str(ROOT)
    os.environ.setdefault('RSIGAME_MONITOR_FRAME_CACHE', str(OUT / 'frames_cache'))

# The margin a candidate must beat the held champion by, and which rounds are
# replayed. Both come from [monitor] in the config; see monitor/value_stop.py.
def gain() -> float:
    return float(os.environ.get('RSIGAME_MONITOR_CHAMPION_GAIN') or 0.05)


def checkpoints(total: int = 30) -> list[int]:
    every = int(os.environ.get('RSIGAME_MONITOR_CHECKPOINT_EVERY') or 3)
    return list(range(0, total + 1, every))


def score_dir(g, r):
    if r == 0:
        d = R0 / g
    else:
        d = SELF / g / f'round_{r:02d}' / 'score'
    return d if (d / 'p1' / 'demos').is_dir() else None


def stage(g, r):
    """Mirror one checkpoint into the run root. -> True if it has recordings."""
    src = score_dir(g, r)
    if src is None:
        return False
    rd = ROOT / g / f'round_{r:02d}'
    (rd / 'post_replay').mkdir(parents=True, exist_ok=True)
    link = rd / 'score'
    if not link.exists():
        link.symlink_to(src)
    for demo in (src / 'p1' / 'demos').iterdir():
        if (demo / f'{demo.name}.mp4').is_file():
            (rd / 'post_replay' / f'frames_{demo.name}').mkdir(exist_ok=True)
    return any((rd / 'post_replay').glob('frames_*'))


def is_web(tree):
    return (Path(tree) / 'package.json').is_file() and not (Path(tree) / 'project.godot').is_file()


def web_crash(g, r):
    """Errors when checkpoint r's built page fails the web start-up check, else [].

    The web counterpart of `parse_errors`: `npm run build` passes a game that
    throws on start-up or on the first input, and the SR20 Phaser loops ran
    before the loop checked for that. A check that could not run at all (no
    script, the browser failed) says nothing about the game and returns [].
    """
    import subprocess
    import tempfile
    tree = tree_of(g, r)
    tool = Path(os.environ.get('GAMECRAFT_BENCH') or '') / 'tools' / 'web_build_check.py'
    if not r or not is_web(tree) or not tool.is_file():
        return []
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as fh:
        out = Path(fh.name)
    try:
        subprocess.run([sys.executable, str(tool), '--project', str(tree), '--json-out', str(out)],
                       capture_output=True, text=True, timeout=180)
        rep = json.loads(out.read_text())
    except Exception:
        return []
    finally:
        out.unlink(missing_ok=True)
    errs = [str(e) for e in (rep.get('errors') or [])]
    if rep.get('pass') is not False or any(e.startswith('probe failed') for e in errs):
        return []
    return errs[:6] or ['page failed the start-up check']


def build_ok(g, r):
    if r == 0:
        return True
    rr = json.loads((SELF / g / 'rounds.json').read_text())
    return bool(rr[r - 1].get('build_ok'))


def parse_errors(g, r):
    """GDScript parse or runtime errors in the candidate's own recording logs.

    Godot's build check passes a project whose scripts do not parse; the game
    then runs with those scripts missing (puzzle-pipe-crisis r27: a grey field,
    which the estimator still rated 1.0). A candidate that does not parse is
    treated like a failed build and never adopted.
    """
    src = score_dir(g, r)
    if src is None:
        return 0
    # SCRIPT ERROR too: a runtime type error parses and builds fine, so the
    # only place it shows is the recording. battle-critter-clash R03 scored
    # 95.1 and was adopted while the battle never started for a real player.
    n = 0
    for p in (src / 'p1' / 'demos').glob('*/logs/godot.log'):
        t = p.read_text(errors='ignore')
        n += t.count('Parse Error') + t.count('SCRIPT ERROR')
    return n


# The base game's project, for re-recording it.
SCORE_ENV = dict(os.environ)


def tree_of(g, r):
    if r:
        return SELF / g / f'round_{r:02d}' / 'tree'
    return R0_TREES / g / 'game' if (R0_TREES / g / 'game').is_dir() else R0_TREES / g


def rerecord(g, champ, r):
    """The champion recorded on checkpoint r's demo inputs. -> pseudo round id or None.

    Needed when the two share no demo name (the agent replaced the base game's
    demos with its own), so there is nothing to pair. The live Monitor plays
    one demo set on both builds; offline that means replaying the champion's
    tree with r's input scripts. Recording only: stub judge, one pass, the
    scorer's own padding and pre-import, so the footage matches r's.
    """
    import shutil
    import subprocess
    pid = 10000 + champ * 100 + r
    work = OUT / 'rerec' / g / f'c{champ:02d}_d{r:02d}'
    score = work / 'score'
    if not (score / 'p1' / 'demos').is_dir():
        src, demos = tree_of(g, champ), tree_of(g, r) / 'demo_outputs'
        if is_web(src) and demos.is_dir():
            return rerecord_web(g, champ, r, pid, src, demos, work, score)
        if not (src / 'project.godot').is_file() or not demos.is_dir():
            return None
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        # A real copy, never a link into the source tree: the import writes into it.
        def top_only(d, names, root=str(src)):
            return [n for n in names if d == root and (n == '.godot' or n.startswith('score')
                                                         or n == 'post_replay')]
        shutil.copytree(src, work / 'tree', symlinks=False, ignore=top_only)
        shutil.rmtree(work / 'tree' / 'demo_outputs', ignore_errors=True)
        (work / 'tree' / 'demo_outputs').mkdir()
        for f in demos.glob('*.json'):
            shutil.copy2(f, work / 'tree' / 'demo_outputs' / f.name)
        for attempt in (1, 2):
            shutil.rmtree(score, ignore_errors=True)
            subprocess.run([sys.executable, str(LOOP / 'score_game.py'), '--project', str(work / 'tree'),
                            '--game', g, '--output', str(score), '--passes', '1',
                            '--judge', 'stub', '--judge-model', '0.5'],
                           cwd=str(LOOP), env=SCORE_ENV, capture_output=True, text=True, timeout=5400)
            fails = sum(p.read_text(errors='ignore').count('Failed loading resource')
                        for p in score.glob('p1/demos/*/logs/godot.log'))
            if (score / 'p1' / 'demos').is_dir() and fails <= 20:
                break
        if not (score / 'p1' / 'demos').is_dir():
            return None
    rd = ROOT / g / f'round_{pid}'
    (rd / 'post_replay').mkdir(parents=True, exist_ok=True)
    if not (rd / 'score').exists():
        (rd / 'score').symlink_to(score)
    for demo in (score / 'p1' / 'demos').iterdir():
        if (demo / f'{demo.name}.mp4').is_file():
            (rd / 'post_replay' / f'frames_{demo.name}').mkdir(exist_ok=True)
    return pid


def rerecord_web(g, champ, r, pid, src, demos, work, score):
    """`rerecord` for a Phaser tree: its built `dist/` replayed on r's inputs.

    No build and no import: the web verifier serves `dist/` as it is, so the
    copy needs the page, the traces and nothing from `node_modules`.
    """
    import shutil
    import subprocess
    web = Path(os.environ.get('GAMECRAFT_BENCH') or '')
    rubric = web / 'tasks-web' / g / 'tests' / 'rubric.json'
    if not (src / 'dist' / 'index.html').is_file() or not rubric.is_file():
        return None
    shutil.rmtree(work, ignore_errors=True)
    (work / 'tree' / 'demo_outputs').mkdir(parents=True)
    shutil.copytree(src / 'dist', work / 'tree' / 'dist')
    for name in ('package.json', 'index.html'):
        if (src / name).is_file():
            shutil.copy2(src / name, work / 'tree' / name)
    for f in demos.glob('*.json'):
        shutil.copy2(f, work / 'tree' / 'demo_outputs' / f.name)
    for attempt in (1, 2):
        shutil.rmtree(score, ignore_errors=True)
        score.mkdir(parents=True)
        subprocess.run([sys.executable, '-m', 'gamecraft_bench.verifier', '--project', str(work / 'tree'),
                        '--rubric', str(rubric), '--output', str(score / 'p1'), '--judge', 'stub',
                        '--replay-engine', 'web'],
                       cwd=str(web), env=SCORE_ENV, capture_output=True, text=True, timeout=5400)
        if any((score / 'p1' / 'demos').glob('*/*.mp4')):
            break
    if not any((score / 'p1' / 'demos').glob('*/*.mp4')):
        return None
    rd = ROOT / g / f'round_{pid}'
    (rd / 'post_replay').mkdir(parents=True, exist_ok=True)
    if not (rd / 'score').exists():
        (rd / 'score').symlink_to(score)
    for demo in (score / 'p1' / 'demos').iterdir():
        if (demo / f'{demo.name}.mp4').is_file():
            (rd / 'post_replay' / f'frames_{demo.name}').mkdir(exist_ok=True)
    return pid


def ready(g):
    """All 30 rounds done and every checkpoint recorded (or marked unscorable).

    A checkpoint missing only because its recording has not run yet would be
    written down as no_recording for good, so a game is replayed only when
    nothing is still pending.
    """
    try:
        if len(json.loads((SELF / g / 'rounds.json').read_text())) < checkpoints()[-1]:
            return False
    except Exception:
        return False
    for r in checkpoints()[1:]:
        rd = SELF / g / f'round_{r:02d}'
        if ((rd / 'tree' / 'project.godot').is_file() or (rd / 'tree' / 'package.json').is_file()) and not (
                (rd / 'score' / 'summary.json').is_file() or (rd / 'score_failed.txt').is_file()):
            return False
    return True


def _passes() -> int:
    return int(os.environ.get('RSIGAME_MONITOR_PASSES') or 1)


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    return None if not xs else (xs[len(xs) // 2] if len(xs) % 2
                                else round((xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2, 4))


def compare(g, old, new, rb, passes: int | None = None):
    """RSIGAME_MONITOR_PASSES>1: the same pair graded that many times, medians reported.

    One call decides whether a checkpoint is adopted, and the estimator is a
    model: measured on 2026-09-15, two runs over the same recordings with the
    same checklist put the champion at R6 and at R12, because one checkpoint's
    brief half came out -0.10 once and +0.17 the next time. Temperature is
    already 0; the variance is the provider's. A median of three is the cheapest
    thing that does not turn a coin flip into a delivered build.
    """
    n = passes or _passes()
    if n <= 1:
        return _compare_once(g, old, new, rb)
    runs = []
    for _ in range(n):
        c = _compare_once(g, old, new, rb)
        if c.get('error'):
            return c
        runs.append(c)
    out = dict(runs[0])
    pv = {k: dict(v) for k, v in runs[0]['proxy'].items() if isinstance(v, dict)}
    for k in list(pv):
        if k == 'by_origin':
            for o in list(pv[k]):
                for f in ('old', 'new', 'delta'):
                    # A pass can miss an origin outright: if none of that
                    # origin's items were observed on both sides in that
                    # replay, by_origin has no key for it. Take the median of
                    # the passes that did observe it -- indexing blind killed
                    # the maze supervisor mid-checkpoint (KeyError: 'task').
                    pv[k][o][f] = _median([((r['proxy'].get(k) or {}).get(o) or {}).get(f)
                                           for r in runs])
            continue
        for f in ('old', 'new', 'delta'):
            if f in pv[k]:
                pv[k][f] = _median([r['proxy'][k].get(f) for r in runs])
    out['proxy'] = pv
    out['passes'] = n
    out['pass_deltas'] = [r['proxy']['overall']['delta'] for r in runs]
    if 'by_origin' in pv:
        out['pass_brief_deltas'] = [(r['proxy'].get('by_origin') or {}).get('brief', {}).get('delta')
                                    for r in runs]
    out['seconds'] = round(sum(r.get('seconds') or 0 for r in runs), 1)
    return out


def _compare_once(g, old, new, rb):
    t0 = time.time()
    d = P.build(g, old, new)
    if d.get('error'):
        return {'error': d['error']}
    demos = [x for x in d['demos'] if x.get('content')]
    if not demos:
        return {'error': 'no usable demo', 'problems': [x.get('problems') for x in d['demos']]}
    with ThreadPoolExecutor(len(demos)) as ex:
        per = list(ex.map(lambda x: score_pair.score(x, rb), demos))
    bad = [p.get('error') for p in per if p.get('error')]
    per = [p for p in per if not p.get('error')]
    if not per:
        return {'error': f'all demos failed: {bad[:1]}'}
    pv = aggregate.pair(per, rb)['proxy_value']
    return {'proxy': pv, 'n_demos': len(per), 'failed': bad, 'demo_ids': [x['demo_id'] for x in demos],
            'seconds': round(time.time() - t0, 1)}


def decide_checkpoint(g, r, champ, rb, *, rerecord_ok=True):
    """One checkpoint against the held champion: adopt it, or keep the champion.

    Split out of `run_game` so the same decision can be made during a run (the
    live monitor) and after it (the offline sweep). It reads only what the
    scorer already wrote for that round, and returns the record that goes into
    the champion trace -- it writes nothing itself.
    """
    rec = {'game': g, 'round': r}
    # A broken web build is never recorded by the scorer, so it is named
    # before the recording is looked for, not reported as no_recording.
    web = bool(r) and is_web(tree_of(g, r))
    if web and not build_ok(g, r):
        rec.update(event='build_failed')
    elif web and (errs := web_crash(g, r)):
        rec.update(event='crashed', vs=champ, errors=errs)
    elif not stage(g, r):
        rec.update(event='no_recording')
    elif not build_ok(g, r):
        rec.update(event='build_failed')
    elif champ is None:
        rec.update(event='first')
        champ = r
    elif parse_errors(g, r):
        rec.update(event='parse_error', vs=champ, n_parse_errors=parse_errors(g, r))
    else:
        old = champ
        if not set(P.demo_ids(g, champ)) & set(P.demo_ids(g, r)):
            old = rerecord(g, champ, r) if rerecord_ok else None
            if old is None:
                old = champ          # compare() then reports the mismatch
            else:
                rec['rerecorded_champion'] = old
        c = None
        for attempt in (1, 2):
            c = compare(g, old, r, rb)
            if not c.get('error'):
                break
        rec['vs'] = champ
        if c.get('error'):
            rec.update(event='compare_failed', error=c['error'])
        else:
            ov = c['proxy']['overall']
            rec.update(comparison=c, old=ov['old'], new=ov['new'], delta=ov['delta'])
            if ov['delta'] is not None and ov['delta'] >= gain():
                rec['event'] = 'replaced'
                champ = r
            elif ov['old'] is None and ov['new'] is not None:
                # Nothing in the champion's footage could be graded (a base game
                # that does not start) while the candidate's could: no delta
                # exists, and keeping the champion would mean never leaving a
                # build that shows nothing.
                rec['event'] = 'replaced'
                rec['reason'] = 'champion_unscored'
                champ = r
            else:
                rec['event'] = 'kept'
    rec['champion'] = champ
    return rec


def run_game(g, limit, log):
    f = OUT / f'{g}.jsonl'
    done = {}
    if f.is_file():
        for line in f.read_text().splitlines():
            rec = json.loads(line)
            done[rec['round']] = rec
    rb = load_rubric(g)
    champ = done[max(done)]['champion'] if done else None
    n_pairs = 0
    for r in checkpoints():
        if r in done:
            continue
        pairing = champ is not None and stage(g, r) and build_ok(g, r) and not parse_errors(g, r)
        if pairing and limit and n_pairs >= limit:
            break
        n_pairs += pairing
        rec = decide_checkpoint(g, r, champ, rb)
        champ = rec['champion']
        with f.open('a') as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
        log(f"{g} r{r:02d} {rec['event']}" + (f" vs r{rec['vs']:02d} delta={rec.get('delta')}" if 'vs' in rec else '')
            + f" -> champ r{champ}" + (f" ({rec['comparison']['seconds']}s)" if 'comparison' in rec else ''))
    return g


def main():
    from rsigame import config
    config.ensure()
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--only', default='')
    ap.add_argument('--limit-pairs', type=int, default=0)
    ap.add_argument('--games', required=True, help='file with one game per line')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    games = Path(a.games).read_text().split()
    waiting = [g for g in games if not ready(g)]
    games = [g for g in games if g not in waiting]
    if a.only:
        games = [g for g in games if g in set(a.only.split(','))]
    logf = OUT / 'progress.log'

    def log(msg):
        line = time.strftime('%m-%d %H:%M ') + msg
        print(line, flush=True)
        with logf.open('a') as fh:
            fh.write(line + '\n')

    def safe(g):
        try:
            return run_game(g, a.limit_pairs, log)
        except BaseException as exc:          # rubric SystemExit included
            log(f'{g} CRASH {type(exc).__name__}: {str(exc)[:200]}')

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(safe, games))
    champ = {}
    for f in sorted(OUT.glob('*.jsonl')):
        recs = [json.loads(x) for x in f.read_text().splitlines()]
        champ[f.stem] = {r['round']: r['champion'] for r in recs}
    (OUT / 'champ.json').write_text(json.dumps(champ, indent=1))
    log(f'finished: replayed {len(games)} ready games, {len(waiting)} still running; champ.json has {len(champ)}')


if __name__ == '__main__':
    main()
