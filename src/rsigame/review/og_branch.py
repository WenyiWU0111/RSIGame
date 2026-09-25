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
"""One outer-guidance branch of one game: the next stage, run until the Monitor converges.

    og_branch.py --case review/cases/pilot1/pilot1-01.json --branch human \
                 --brief .../development_brief.json --root BRANCH_ROOT --port 9601

The stage starts FRESH from the case's saturated checkpoint P* (a new run whose
round 0 is P*), with the same settings as the runs that produced P* plus lean
repair sessions and full metering. The three branches of a game differ only in
RSIGAME_CONTROLLER_BRIEF: none, the human's brief, the model's brief.

The Monitor runs beside it, offline-style but live: when round r (every third,
and the last) has finished, its tree is recorded (score_game, stub judge, one
pass) and compared with the champion by the estimator (vm_replay's rules). The
stage has CONVERGED when the champion stays unchanged for STOP_AFTER checkpoints
in a row; the supervisor then writes the loop's stop file and the loop ends at
the next round boundary. MAX_ROUNDS caps the stage either way.

Writes BRANCH_ROOT/runs/<game>/... (the loop), BRANCH_ROOT/vm/<game>.jsonl (one
line per checkpoint) and BRANCH_ROOT/og/<game>.json (status, stop reason, champion).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from .. import paths

STOP_AFTER = 3
TASK_FLOOR = -0.05          # how far the task document's half may fall on an adoption
MAX_ROUNDS = 20
EVERY = 3
RUN = paths.run_root()          # the loop code the case runs were produced with
REPO = Path(__file__).resolve().parents[1]
PY = os.environ.get('RSIGAME_PATHS_PYTHON') or sys.executable   # subprocesses use this same interpreter


def load_env(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--branch', required=True, choices=['none', 'human', 'model'])
    ap.add_argument('--brief', default='')
    ap.add_argument('--root', required=True)
    ap.add_argument('--port', type=int, required=True)
    # MAIN-TABLE RUNS START AT THE BASE GAME, NOT AT A P* CHECKPOINT. The
    # supervisor is the only thing that stops a run at saturation, so a run
    # that begins at P0 needs it just as much as an outer-guidance branch --
    # it just cannot derive round 0's recordings from a previous run's tree.
    # Point it at the R0 scoring directory instead.
    ap.add_argument('--r0-score', default='',
                    help='pass-1 scoring dir for round 0 (base-game runs)')
    a = ap.parse_args()
    case = json.loads(Path(a.case).read_text())
    g = case['game_id']
    if (a.branch == 'none') != (not a.brief):
        raise SystemExit('--brief is required for human/model branches and forbidden for none')
    root = Path(a.root).resolve()
    og = root / 'og'
    og.mkdir(parents=True, exist_ok=True)
    status_path = og / f'{g}.json'
    stop_file = og / f'{g}.STOP'

    def status(**kw):
        st = json.loads(status_path.read_text()) if status_path.is_file() else {}
        st.update(kw, updated=time.strftime('%Y-%m-%dT%H:%M:%S'))
        status_path.write_text(json.dumps(st, indent=1, ensure_ascii=False))
        return st

    # ---- inputs: P* as the corpus game, and as the Monitor's round 0 ----------
    corpus = root.parent / 'corpus'
    if not (corpus / g / 'project.godot').is_file():
        corpus.mkdir(parents=True, exist_ok=True)
        shutil.copytree(case['tree_src'], corpus / g, symlinks=True,
                        ignore=shutil.ignore_patterns('.godot', '.qwen'))
    if a.r0_score:
        r0_score = Path(a.r0_score).resolve()
    else:
        pstar_round = int(case['checkpoint_id'][1:])
        src_run = Path(case['tree_src']).parents[3]       # .../runs/<g>/round_XX/tree -> run root
        r0_score = (src_run / 'runs' / g / f'round_{pstar_round:02d}' / 'score').resolve()
    vmin = root / 'vm_inputs'
    for d, target in ((vmin / 'r0scores' / g, r0_score), (vmin / 'r0trees' / g / 'game', corpus / g)):
        d.parent.mkdir(parents=True, exist_ok=True)
        if not d.exists():
            d.symlink_to(target)
    if not (r0_score / 'p1' / 'demos').is_dir():
        raise SystemExit(f'no pass-1 recordings for P* at {r0_score}')

    # ---- the loop ----------------------------------------------------------------
    env = dict(os.environ)
    env.update(load_env(paths.env_file()))
    env.update({
        'PYTHONPATH': str(paths.agent_repo()),
        'RSIGAME_LOOP_LOOP': str(RUN), 'RSIGAME_PATHS_RUN': str(RUN), 'RSIGAME_PATHS_PYTHON': PY,
        'GAMECRAFT_BENCH': str(paths.bench()),
        'RSIGAME_MODELS_LOOP_MODEL': 'z-ai/glm-5.3-flash', 'RSIGAME_MODELS_PLAY_MODEL': 'qwen/qwen3.8-27b',
        'RSIGAME_REPAIR_MODEL': 'z-ai/glm-5.3-flash', 'RSIGAME_CONTROLLER_PROBE_STEPS': '8',
        'RSIGAME_CONTROLLER_DISPATCH': 'inner', 'RSIGAME_CONTROLLER_ART_ROUNDS': '0', 'RSIGAME_CONTROLLER_SNAPSHOT': '1',
        'RSIGAME_CONTROLLER_ASSET_CAP': '20', 'REPLAY_WAIT_FOR_DRAW': '1', 'RSIGAME_LOOP_REPAIR_BUDGET_S': '1800',
        'RSIGAME_LOOP_LEAN_REPAIR': '0', 'RSIGAME_CONTROLLER_STOP_FILE': str(stop_file),
        'RSIGAME_CONTROLLER_KEEP_REPLAY_EVERY': str(EVERY),
    })
    env.pop('RSIGAME_CONTROLLER_ESTIMATOR', None)
    if a.brief:
        env['RSIGAME_CONTROLLER_BRIEF'] = str(Path(a.brief).resolve())
    else:
        env.pop('RSIGAME_CONTROLLER_BRIEF', None)
    rundir = root / 'runs' / g
    rundir.mkdir(parents=True, exist_ok=True)

    # ---- recordings: the loop's own replays, laid out the way the Monitor reads them ----
    # The loop already replays every demo after each round (post_replay/<demo>.mp4,
    # log_<demo>/godot.log). With RSIGAME_CONTROLLER_KEEP_REPLAY_EVERY it keeps those at checkpoint
    # rounds, so a checkpoint is linked from them instead of being replayed a second
    # time by score_game. P* is replayed once by the same code, so both sides of every
    # comparison come from one recording pipeline.
    def link_replay(replay_dir: Path, out: Path) -> bool:
        mp4s = sorted(replay_dir.glob('*.mp4')) if replay_dir.is_dir() else []
        if not mp4s:
            return False
        for m in mp4s:
            dd = out / 'p1' / 'demos' / m.stem
            (dd / 'logs').mkdir(parents=True, exist_ok=True)
            if not (dd / m.name).exists():
                (dd / m.name).symlink_to(m.resolve())
            lg = replay_dir / f'log_{m.stem}' / 'godot.log'
            if lg.is_file() and not (dd / 'logs' / 'godot.log').exists():
                (dd / 'logs' / 'godot.log').symlink_to(lg.resolve())
        return True

    r0loop = vmin / 'r0loop' / g
    if not (r0loop / 'p1' / 'demos').is_dir():
        pw = vmin / 'pstar_work' / g
        if not (pw / 'project.godot').is_file():
            shutil.copytree(corpus / g, pw, symlinks=True, ignore=shutil.ignore_patterns('.godot', '.qwen'))
            subprocess.run(['godot', '--headless', '--path', str(pw), '--import', '--quit'],
                           capture_output=True, timeout=900, env=env)
        prd = vmin / 'pstar_replay' / g
        subprocess.run([PY, 'post_repair_replay.py', str(pw), str(prd)], cwd=str(RUN), env=env,
                       capture_output=True, text=True, timeout=3600)
        link_replay(prd, r0loop)
    (rundir / 'og_config.json').write_text(json.dumps({
        'case': case['case_id'], 'branch': a.branch, 'brief': env.get('RSIGAME_CONTROLLER_BRIEF'),
        'pstar': case['checkpoint_id'], 'tree_hash': case['tree_hash'],
        'brief_injection': 'planner, rank stage B, polish goal, repair prompt, art round prompt' if a.brief else None,
        'stop_after': STOP_AFTER, 'max_rounds': MAX_ROUNDS, 'every': EVERY,
        'env': {k: env[k] for k in ('RSIGAME_MODELS_LOOP_MODEL', 'RSIGAME_MODELS_PLAY_MODEL', 'RSIGAME_REPAIR_MODEL', 'RSIGAME_CONTROLLER_DISPATCH',
                                    'RSIGAME_CONTROLLER_ART_ROUNDS', 'RSIGAME_CONTROLLER_ASSET_CAP', 'RSIGAME_LOOP_LEAN_REPAIR')}},
        indent=1))
    loop_log = open(root / 'og' / f'{g}.loop.out', 'ab')
    loop = subprocess.Popen([PY, '-u', str(REPO / 'scripts/evogame/metered_loop.py'), g,
                             '--rounds', str(MAX_ROUNDS), '--root', str(root), '--corpus', str(corpus),
                             '--port', str(a.port)],
                            cwd=str(RUN), env=env, stdout=loop_log, stderr=subprocess.STDOUT)
    status(game=g, branch=a.branch, loop_pid=loop.pid, state='running', stop=None)

    # ---- the Monitor -------------------------------------------------------------
    r0_dir = vmin / 'r0loop' if (r0loop / 'p1' / 'demos').is_dir() else vmin / 'r0scores'
    status(pstar_recordings='loop replay' if r0_dir.name == 'r0loop' else 'original scorer recording')
    os.environ.update({'RSIGAME_LOOP_LOOP': str(RUN), 'RSIGAME_MONITOR_RUN': str(root), 'RSIGAME_MONITOR_R0_DIR': str(r0_dir),
                       'RSIGAME_MONITOR_R0_TREES': str(vmin / 'r0trees'), 'RSIGAME_MONITOR_OUT': str(root / 'vm'),
                       'GAMECRAFT_BENCH': env['GAMECRAFT_BENCH'], 'PYTHONPATH': env['PYTHONPATH'],
                       'OPENROUTER_API_KEY': env.get('OPENROUTER_API_KEY', ''),
                       # Three gradings per comparison, median reported: one
                       # grading flipped a champion between two identical runs.
                       'RSIGAME_MONITOR_PASSES': os.environ.get('RSIGAME_MONITOR_PASSES', '3')})
    sys.path.insert(0, str(REPO / 'scripts/evogame'))
    import rsigame.monitor.vm_replay as VM                                  # reads the env above at import
    (root / 'vm').mkdir(parents=True, exist_ok=True)
    rb = VM.load_rubric(g)
    # THE BRIEF HAS TO REACH THE MONITOR TOO. The task document's checklist is
    # what the stage already saturated on, so a build that does what the brief
    # asked for moves nothing on it: measured on pilot2, most checkpoint deltas
    # were exactly 0. The stage checklist puts the brief's own criteria first
    # (weight 3), keeps the base criteria P* has not maxed out, and retires or
    # demotes the saturated ones -- see estimator/stage_rubric.py.
    stage_rb = None
    if a.brief:
        try:
            from rsigame.eval.estimator import pairs as P, score_pair, aggregate, stage_rubric as SRB
            VM.stage(g, 0)
            d = P.build(g, 0, 0)
            per = [score_pair.score(x, rb) for x in d['demos'] if x.get('content')]
            per = [x for x in per if not x.get('error')]
            vals = {i: v['value'] for i, v in aggregate.one_side(per, rb, 'new')['items'].items()}
            stage_rb = SRB.build(g, json.loads(Path(a.brief).read_text()), rb, vals)
            status(stage_checklist={'items': len(stage_rb['items']),
                                    'brief_items': sum(1 for x in stage_rb['items'] if x['origin'] == 'brief'),
                                    'retired': len(stage_rb.get('retired') or [])})
        except Exception as exc:
            print(f'stage checklist failed, falling back to the task rubric: '
                  f'{type(exc).__name__}: {str(exc)[:200]}', flush=True)
    if stage_rb is not None:
        rb = stage_rb
    jl = root / 'vm' / f'{g}.jsonl'
    recs = {json.loads(x)['round']: json.loads(x) for x in jl.read_text().splitlines()} if jl.is_file() else {}
    if 0 not in recs:
        VM.stage(g, 0)
        recs[0] = {'game': g, 'round': 0, 'event': 'first', 'champion': 0}
        with jl.open('a') as f:
            f.write(json.dumps(recs[0]) + '\n')

    # ALREADY CONVERGED BEFORE THIS PROCESS STARTED. A resumed branch recomputes
    # the streak from the records, but the convergence test used to run only
    # after a NEW checkpoint, so a branch whose last checkpoints already showed
    # three without a change went on running rounds nobody would deliver
    # (human fighting-alley-brawlers, 2026-09-15). Decided here too.
    def check_converged(streak: int, r: int, champ: int) -> None:
        if streak >= STOP_AFTER and not stop_file.exists():
            stop_file.write_text(f'converged at R{r:02d}: champion R{champ:02d} '
                                 f'unchanged for {streak} checkpoints')
            status(stop={'reason': 'converged', 'checkpoint': r, 'champion': champ})

    def record(r: int) -> bool:
        tree = rundir / f'round_{r:02d}' / 'tree'
        out = tree.parent / 'score'
        if (out / 'p1' / 'demos').is_dir():
            return True
        if link_replay(tree.parent / 'post_replay', out):
            return True
        if not (tree / 'project.godot').is_file():
            return False
        for attempt in (1, 2):
            if attempt == 2 or not (tree / '.godot').is_dir():
                subprocess.run(['godot', '--headless', '--path', str(tree), '--import', '--quit'],
                               capture_output=True, timeout=900, env=env)
            shutil.rmtree(out, ignore_errors=True)
            subprocess.run([PY, 'score_game.py', '--project', str(tree), '--game', g, '--output', str(out),
                            '--passes', '1', '--judge', 'stub', '--judge-model', '0.5'],
                           cwd=str(RUN), env=env, capture_output=True, text=True, timeout=5400)
            fails = sum(p.read_text(errors='ignore').count('Failed loading resource')
                        for p in out.glob('p1/demos/*/logs/godot.log'))
            if (out / 'p1' / 'demos').is_dir() and fails <= 20:
                return True
        return (out / 'p1' / 'demos').is_dir()

    def rounds_done() -> int:
        try:
            return len(json.loads((rundir / 'rounds.json').read_text()))
        except Exception:
            return 0

    def checkpoints(done: int, alive: bool):
        # every third round, and the last round once the stage is over
        last = [done] if done % EVERY and (not alive or done >= MAX_ROUNDS) else []
        return list(range(EVERY, done + 1, EVERY)) + last

    champ = recs[max(recs)]['champion']
    streak = 0
    for r in sorted(recs)[1:]:
        # A checkpoint that could not be judged -- no recording, a failed
        # comparison -- is not evidence that the champion still wins, so it
        # neither resets the streak nor counts toward convergence.
        ev = recs[r].get('event')
        if ev == 'replaced':
            streak = 0
        elif ev not in ('no_recording', 'compare_failed'):
            streak += 1
    check_converged(streak, max(recs), champ)
    while True:
        done = rounds_done()
        alive = loop.poll() is None
        todo = [r for r in checkpoints(min(done, MAX_ROUNDS), alive) if r not in recs]
        if not alive and not todo:
            break
        if not todo:
            time.sleep(60)
            continue
        r = todo[0]
        rec = {'game': g, 'round': r}
        if not record(r) or not VM.stage(g, r):
            rec['event'] = 'no_recording'
        elif not VM.build_ok(g, r):
            rec['event'] = 'build_failed'
        elif VM.parse_errors(g, r):
            rec.update(event='parse_error', n_parse_errors=VM.parse_errors(g, r))
        else:
            old = champ
            if not set(VM.P.demo_ids(g, champ)) & set(VM.P.demo_ids(g, r)):
                rr = VM.rerecord(g, champ, r)
                if rr is not None:
                    old = rr
                    rec['rerecorded_champion'] = rr
            c = None
            for attempt in (1, 2):
                c = VM.compare(g, old, r, rb)
                if not c.get('error'):
                    break
            rec['vs'] = champ
            if c.get('error'):
                rec.update(event='compare_failed', error=c['error'])
            else:
                ov = c['proxy']['overall']
                bo = c['proxy'].get('by_origin') or {}
                bd = (bo.get('brief') or {}).get('delta')
                td = (bo.get('task') or {}).get('delta')
                rec.update(comparison=c, old=ov['old'], new=ov['new'], delta=ov['delta'],
                           brief_delta=bd, task_delta=td)
                # On a stage checklist the decision is the brief's half: the
                # task half is a floor, not the target, so it only has to not
                # fall. Without brief criteria this is the old rule exactly.
                if bd is not None:
                    adopt = bd >= VM.GAIN and (td is None or td >= TASK_FLOOR)
                else:
                    adopt = ov['delta'] is not None and ov['delta'] >= VM.GAIN
                if adopt:
                    rec['event'] = 'replaced'
                elif ov['old'] is None and ov['new'] is not None:
                    rec.update(event='replaced', reason='champion_unscored')
                else:
                    rec['event'] = 'kept'
        if rec['event'] == 'replaced':
            champ, streak = r, 0
        else:
            streak += 1
        rec['champion'], rec['streak'] = champ, streak
        recs[r] = rec
        with jl.open('a') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        st = status(last_checkpoint=r, champion=champ, streak=streak, rounds_done=rounds_done())
        if streak >= STOP_AFTER and not stop_file.exists():
            stop_file.write_text(f'converged at R{r:02d}: champion R{champ:02d} unchanged for {streak} checkpoints')
            status(stop={'reason': 'converged', 'checkpoint': r, 'champion': champ})
            if st.get('state') == 'running':
                status(state='stopping')
            break
    loop.wait()
    st = json.loads(status_path.read_text())
    if not st.get('stop'):
        status(stop={'reason': 'max_rounds' if rounds_done() >= MAX_ROUNDS else 'loop_ended',
                     'checkpoint': max(recs), 'champion': champ})
    status(state='finished', rounds_done=rounds_done(), loop_rc=loop.returncode)


if __name__ == '__main__':
    main()
