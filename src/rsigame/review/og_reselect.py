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
"""Re-run the monitor's decision on existing recordings, with a stage rubric.

No game is replayed. Every checkpoint's footage is still in the branch
directory (score/p1/demos) and the monitor's decision was offline to begin
with, so changing the ruler is one more call to the proxy estimator.

THE RULER is the stage checklist `estimator/stage_rubric.py` produces: the
brief's own items come first and weigh more; items the original rubric did not
yet score full marks for on P* stay; items already at full marks retire, or
drop to a "must not regress" guard.

THE REPLACEMENT RULE changes with the ruler: the build is replaced when the
brief half gains more than GAIN and the task half does not clearly regress
(>= -0.05). With no brief items it falls back to the original rule (the
overall gain must exceed GAIN).

NOTHING OVERWRITES THE ORIGINAL RECORD. The new decision is written to
<branch>/vm_stage/<game>.jsonl; the original stays at <branch>/vm/<game>.jsonl.
Keeping both is what makes "did the choice change when the ruler changed"
answerable at all.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
REPO = RUN.parent

TASK_GAIN = -0.05          # how far the task half may regress
CKS = list(range(0, 31, 3))


def one(branch_root: Path, case: Path, jsonl_out: Path) -> dict:
    """One game of one branch, in a subprocess: vm_replay reads its environment at import."""
    case_d = json.loads(case.read_text())
    g = case_d['game_id']
    cfg = json.loads((branch_root / 'runs' / g / 'og_config.json').read_text())
    brief_p = cfg.get('brief')
    if not brief_p or not Path(brief_p).is_file():
        return {'game': g, 'skipped': 'no brief on this branch'}
    vmin = branch_root / 'vm_inputs'
    r0 = vmin / 'r0loop' if (vmin / 'r0loop' / g / 'p1' / 'demos').is_dir() else vmin / 'r0scores'
    env = dict(os.environ)
    env.setdefault('GAMECRAFT_BENCH', str(paths.bench()))
    env.setdefault('RSIGAME_MONITOR_PASSES', '3')            # median of three; one grading is a coin flip
    env.update(RSIGAME_LOOP_LOOP=str(RUN), RSIGAME_MONITOR_RUN=str(branch_root), RSIGAME_MONITOR_R0_DIR=str(r0),
               RSIGAME_MONITOR_R0_TREES=str(vmin / 'r0trees'), RSIGAME_MONITOR_OUT=str(branch_root / 'vm'),
               PYTHONPATH=f"{REPO / 'agent-test'}:{RUN}")
    rounds_json = str(branch_root / 'runs' / g / 'rounds.json')
    code = f'''
import json, sys, time
sys.path.insert(0, {str(RUN / "scripts")!r})
sys.path.insert(0, {str(RUN)!r})
import rsigame.monitor.vm_replay as VM
from rsigame.eval.estimator import pairs as P, score_pair, aggregate, stage_rubric as SR
from rsigame.eval.estimator.rubric import load as load_rubric
from .. import paths

g = {g!r}
brief = json.loads(open({brief_p!r}).read())
base = load_rubric(g)

# P*'s own per-item scores: pair P* with itself once. The frames and the cache are already there.
VM.stage(g, 0)
d = P.build(g, 0, 0)
demos = [x for x in d["demos"] if x.get("content")]
per = [score_pair.score(x, base) for x in demos]
per = [x for x in per if not x.get("error")]
side = aggregate.one_side(per, base, "new")
values = {{i: v["value"] for i, v in side["items"].items()}}
rb = SR.build(g, brief, base, values)

# ONLY THE CHECKPOINTS THIS BRANCH ACTUALLY REACHED. Walking all ten wrote a
# no_recording record for every round the branch never ran, and a resumed
# supervisor counts those as "the champion did not change" -- three branches
# declared convergence the moment they came back instead of carrying on.
n_rounds = len(json.loads(open({rounds_json!r}).read()))
recs, champ, streak = [], None, 0
for r in [x for x in VM.CKS if x <= n_rounds]:
    rec = {{"game": g, "round": r}}
    if not VM.stage(g, r):
        rec["event"] = "no_recording"
    elif not VM.build_ok(g, r):
        rec["event"] = "build_failed"
    elif champ is None:
        rec["event"] = "first"; champ = r
    elif VM.parse_errors(g, r):
        rec.update(event="parse_error", vs=champ)
    else:
        old = champ
        if not set(P.demo_ids(g, champ)) & set(P.demo_ids(g, r)):
            old = VM.rerecord(g, champ, r) or champ
        c = VM.compare(g, old, r, rb)
        rec["vs"] = champ
        if c.get("error"):
            rec.update(event="compare_failed", error=c["error"])
        else:
            pv = c["proxy"]
            ov = pv["overall"]
            bo = (pv.get("by_origin") or {{}})
            bd = (bo.get("brief") or {{}}).get("delta")
            td = (bo.get("task") or {{}}).get("delta")
            rec.update(comparison=c, old=ov["old"], new=ov["new"], delta=ov["delta"],
                       brief_delta=bd, task_delta=td)
            if bd is not None:
                adopt = bd >= VM.GAIN and (td is None or td >= {TASK_GAIN})
            else:
                adopt = ov["delta"] is not None and ov["delta"] >= VM.GAIN
            if adopt:
                rec["event"] = "replaced"; champ = r
            elif ov["old"] is None and ov["new"] is not None:
                rec.update(event="replaced", reason="champion_unscored"); champ = r
            else:
                rec["event"] = "kept"
    rec["champion"] = champ
    streak = 0 if rec["event"] == "replaced" or rec["event"] == "first" else streak + 1
    rec["streak"] = streak
    recs.append(rec)
    print(f"  r{{r:02d}} {{rec['event']}} brief={{rec.get('brief_delta')}} task={{rec.get('task_delta')}} -> champ r{{champ}}", flush=True)

out = {str(jsonl_out)!r}
with open(out, "w") as f:
    for rec in recs:
        f.write(json.dumps(rec, ensure_ascii=False) + "\\n")
print(json.dumps({{"game": g, "champion": champ, "rubric_items": len(rb["items"]),
                  "brief_items": sum(1 for x in rb["items"] if x["origin"] == "brief"),
                  "retired": len(rb.get("retired") or [])}}, ensure_ascii=False))
'''
    p = subprocess.run([sys.executable, '-c', code], cwd=str(RUN), env=env,
                       capture_output=True, text=True, timeout=7200)
    sys.stdout.write(p.stdout)
    if p.returncode != 0:
        return {'game': g, 'error': (p.stderr or '')[-400:]}
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {'game': g, 'error': f'{type(exc).__name__}: {(p.stdout or "")[-200:]}'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--branches', required=True)
    ap.add_argument('--cases', required=True)
    ap.add_argument('--branch', default='human,model')
    ap.add_argument('--only', default='', help='comma-separated game ids')
    a = ap.parse_args()
    B = Path(a.branches)
    rows = []
    for br in a.branch.split(','):
        for cp in sorted(Path(a.cases).glob('*.json')):
            case = json.loads(cp.read_text())
            g = case['game_id']
            if a.only and g not in a.only.split(','):
                continue
            root = B / br
            if not (root / 'runs' / g / 'og_config.json').is_file():
                continue
            out_dir = root / 'vm_stage'
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f'== {br} {g}', flush=True)
            res = one(root, cp, out_dir / f'{g}.jsonl')
            old = [json.loads(x) for x in (root / 'vm' / f'{g}.jsonl').read_text().splitlines()] \
                if (root / 'vm' / f'{g}.jsonl').is_file() else []
            res.update(branch=br, old_champion=(old[-1]['champion'] if old else None))
            rows.append(res)
            print(json.dumps(res, ensure_ascii=False), flush=True)
    print('\n| branch | game | old champion | stage champion | brief items | retired |')
    print('|---|---|---|---|---|---|')
    for r in rows:
        print(f"| {r.get('branch')} | {r.get('game')} | R{r.get('old_champion')} | "
              f"R{r.get('champion')} | {r.get('brief_items', '-')} | {r.get('retired', '-')} |")


if __name__ == '__main__':
    main()
