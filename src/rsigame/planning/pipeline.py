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
"""The redesigned inner loop, end to end, in shadow mode.

Runs the whole chain against one real build and writes down what each stage
decided. Nothing is repaired: the round stops at the packet, which is the last
artefact that can be judged by reading it. The repair agent is the first stage
whose output can only be judged by running the game again, and there is no
point paying for that until the four stages before it are producing sense.

The build is a copy. The corpora are frozen and the replay writes into the tree
it plays, so the tree is copied to a scratch directory and imported there --
Godot without its script class cache parses `class_name` as an error, renders
grey, and records a clean-looking replay of nothing.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rsigame.controller.inputs import Run, build            # noqa: E402
from rsigame.controller.plan import plan                     # noqa: E402
from rsigame.controller import execute, rank as RANK         # noqa: E402
from rsigame.controller.targets import Ledger                # noqa: E402
from rsigame.evidence.fingerprint import project_hash        # noqa: E402
from rsigame.evidence.observe import describe, observe_round  # noqa: E402


def prepare(run_dir: Path, scratch: Path) -> Path:
    work = scratch / 'work'
    if not work.is_dir():
        shutil.copytree(run_dir / 'work', work)
        subprocess.run(['godot', '--headless', '--path', str(work),
                        '--import', '--quit'], capture_output=True, timeout=600)
    return work


def polish_for(run_dir: Path, r: int) -> dict | None:
    """What the polish reader said about the build this round came in with.

    It already runs every round, and its finding goes to the planner beside the
    open requirements rather than queued behind them -- the note in
    `_polish_read` says why: as a stage of its own it was eligible for 35% of
    the rounds and got none of them.
    """
    f = run_dir / 'polish_state.json'
    if not f.is_file():
        return None
    rows = (json.loads(f.read_text()) or {}).get('rounds') or []
    prev = [x for x in rows if (x.get('round') or 0) <= r - 1]
    return prev[-1] if prev else None


def keyframes_for(ex: dict, probe_dir: Path, run_dir: Path, r: int) -> list:
    """Whatever the probe left behind, labelled by the input that made it.

    A probe that ran gives its own frames. A round that reused existing
    evidence has none of its own, so it falls back to the replay the previous
    round left -- which is the evidence it decided was sufficient, and is
    exactly what should be looked at.
    """
    from rsigame.evidence.keyframes import from_exploration, from_replay
    from rsigame.evidence.artifact import from_round, probes_at
    if ex.get('ran') and (probe_dir / 'trace').is_dir():
        return from_exploration(probe_dir / 'trace')
    if ex.get('ran'):
        rec = {'events': [], 'final_frame': ex.get('final_frame')}
        return from_replay(rec)
    out = []
    for pid in probes_at(run_dir, r - 1)[:3]:
        a = from_round(run_dir, r - 1, pid)
        if a:
            out += [(f'{pid} {l}', f) for l, f in from_replay(a, limit=5)]
    return out[:16]


def main():
    from rsigame import config
    config.ensure()
    run_dir = Path(sys.argv[1])
    r = int(sys.argv[2])
    direction = sys.argv[3] if len(sys.argv) > 3 else 'correctness'
    scratch = Path(sys.argv[4]) if len(sys.argv) > 4 else Path('/tmp/x')
    scratch.mkdir(parents=True, exist_ok=True)
    out = scratch / 'pipeline.json'
    log: dict = {'run': str(run_dir), 'round': r, 'direction': direction}

    def say(stage, *bits):
        print(f'\n=== {stage} ' + '=' * (56 - len(stage)))
        for b in bits:
            print(b)

    work = prepare(run_dir, scratch)
    ph = project_hash(work)
    say('build', f'  {run_dir.name}  round {r}  project_hash {ph}')

    run = Run(run_dir)
    cache = scratch / 'cache'
    desc = describe(work, cache / 'demos.json')
    obs = observe_round(run_dir, r - 1, cache=cache / f'r{r-1:02d}.json') if r > 1 else {}
    say('evidence summary', f'  demo descriptions {len(desc)}  · previous round results {len(obs)} ')

    inp = build(run, r, direction, budget_total=15,
                outer_reason='(frozen for this run)', descriptions=desc)
    for d in inp['demos']:
        if d['replayed_on_this_build'] and not d['outcome']:
            d['outcome'] = obs.get(d['demo_id'])

    ledger = Ledger(scratch / 'targets.json')
    ledger.sync(run.state, r)
    t0 = time.time()
    p = plan(inp, ledger.open())
    log['plan'] = p
    say('Controller-Plan', f"  action={p.get('action')}  mode={p.get('mode')}  "
        f"demo={p.get('demo_id') or '—'}  ({time.time()-t0:.0f}s)",
        f"  Q: {p.get('question')}", f"  why now: {p.get('why_now')}")
    if p.get('schema_problems'):
        print(f"  schema: {p['schema_problems']}")
    tid = ledger.note(p.get('target_ref'), p.get('question', ''),
                      p.get('related_state_items') or [], r)
    ledger.save()
    log['target'] = tid

    t0 = time.time()
    ex = execute.run({'mode': p.get('mode'), 'demo_id': p.get('demo_id'),
                      'extension': p.get('extension')},
                     work=work, out_dir=scratch / f'probe_r{r:02d}',
                     question=p.get('question', ''),
                     items=p.get('related_state_items') or [], port=7973)
    log['execute'] = ex
    say('probe run', f"  ran={ex.get('ran')}  {ex.get('error') or ''}  "
        f"({time.time()-t0:.0f}s)")
    if ex.get('observation'):
        print(f"  saw: {ex['observation'][:400]}")
    if ex.get('n_events') is not None:
        print(f"  recorded {ex['n_events']} events")

    ev = ex.get('observation') or ''
    if not ev:
        parts = [f"{d['demo_id']}: {d['outcome']}" for d in inp['demos']
                 if d.get('outcome')]
        ev = ('The probe produced no written account. What is known about this '
              'build comes from the previous round:\n' + '\n'.join(parts))
    t0 = time.time()
    frames = keyframes_for(ex, scratch / f'probe_r{r:02d}', run_dir, r)
    g = RANK.ground(ev, inp['state']['all_items'], direction,
                    p.get('question', ''), frames=frames)
    log['issues'] = g
    say('Stage A evidence', f"  {len(g['issues'])} problems · looked at {len(frames)} frames  "
        f"({time.time()-t0:.0f}s)")
    for i in g['issues']:
        print(f"  {i['issue_id']} [{i['status']}/{i['importance']}] {i['issue'][:96]}")

    hist = '\n'.join(f"  {t['id']} ({t['attempts']} attempt(s)) {t['statement']}"
                     for t in ledger.open())
    t0 = time.time()
    sel = RANK.rank(g['issues'], direction, p.get('question', ''), hist,
                    15 - r)
    log['selection'] = sel
    say('Stage B ranking', f"  primary={sel.get('primary_target')}  "
        f"concurrent={sel.get('concurrent_minor_fixes')}  "
        f"deferred={len(sel.get('deferred') or [])}  "
        f"gated (unverified)={len(sel.get('blocked_unverified') or [])}  "
        f"({time.time()-t0:.0f}s)",
        f"  why this one: {sel.get('why_primary')}")
    if sel.get('schema_problems'):
        print(f"  schema: {sel['schema_problems']}")

    art = polish_for(run_dir, r)
    if art:
        say('art opportunity (read by polish, not gated)',
            f"  focus={art.get('focus')}  headroom={art.get('headroom')}",
            f"  {(art.get('goal') or '')[:220]}")
    log['art'] = art
    pkt = RANK.packet(sel, g['issues'], art=art)
    log['packet'] = pkt
    say('repair packet (not executed)', pkt)
    out.write_text(json.dumps(log, indent=1, ensure_ascii=False))
    print(f'\nwritten to {out}')


if __name__ == '__main__':
    main()
