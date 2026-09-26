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
"""The redesigned loop, running for real on one game.

One round, in order:

    global monitor (every third round)   where should effort go
    Controller-Plan                      what should we find out
    probe                                go and find it out
    Stage A                              what does the evidence support
    Stage B                              what gets this round's budget
    repair agent                         change the code
    verification                         did those things change

What closes the loop is the last step feeding the next one. The verdicts go
into the round record; the record supplies the history block the next repair
agent reads and the attempt counts the next controller reads; and the target
ledger opens, closes and reopens off the checklist. Without that the chain is a
line, not a loop, and every round starts from nothing.

The probe is frozen into a script whether the play agent drove it or not, and
verification replays that same script. That is the design doc's rule -- the
probe that found a problem is the acceptance test for its repair -- and it is
also the only way the two readings can be compared frame by frame.

Nothing here writes to a corpus. Every game runs in its own copy.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rsigame.project as PJ                                  # noqa: E402
import rsigame.verify.post_repair_replay as PRR                          # noqa: E402
from rsigame.controller.inputs import Run, build                  # noqa: E402
from rsigame.controller.plan import plan                          # noqa: E402
from rsigame.controller import execute, rank as RANK, verify as VERIFY  # noqa: E402
from rsigame.controller.repair_prompt import render as render_repair, parse_report  # noqa: E402
from rsigame.controller.targets import Ledger                     # noqa: E402
from rsigame.evidence.freeze import freeze                        # noqa: E402
from rsigame.evidence.keyframes import from_exploration, from_replay  # noqa: E402
from rsigame.evidence.artifact import from_round  # noqa: E402
from rsigame.evidence.observe import describe, observe_round, observe_one  # noqa: E402
from rsigame.evidence.fingerprint import project_hash              # noqa: E402

MONITOR_EVERY = 3

# THE LAST ROUNDS ARE ART ROUNDS.
#
# By round twelve the requirement work has had its budget and what holds these
# games back is not what they do. Measured over the ten case studies, twelve
# rounds wired 27 more images onto the screen between them, and the reader that
# looks at every build never once called either of its dimensions finished.
#
# An art round replaces the whole per-round chain -- plan, probe, ground, rank,
# polish read, polish ask, five calls -- with one reading of the frames the
# round before already produced. It can, because an art round does not have to
# decide what to look into: the evidence is the demo library replayed and the
# question is the same every time.
ART_ROUNDS = int(os.environ.get('RSIGAME_CONTROLLER_ART_ROUNDS') or 3)

# How many past rounds the repair packet and its ledgers carry. Measured on
# newloop_self40 (2026-09-16): a round's input grows 359k -> 34.6M tokens
# between R01 and R30 while the packet itself stays ~13KB, so this window
# bounds the ledgers rather than the bulk -- the bulk is the agent's own
# tool conversation inside one round. Override with RSIGAME_LOOP_HISTORY_ROUNDS.
HISTORY_ROUNDS = int(os.environ.get('RSIGAME_LOOP_HISTORY_ROUNDS', '5'))

# An art round has more to do than a repair round: generate a set, wire every
# one of them in, delete what they replace, import, and look. Twenty-six calls
# is the number a single-defect round was sized for.
#
# IT RAN AT 40 UNTIL 2026-09-16, AND THAT IS WHERE THE MONEY WENT. Cost is
# essentially (tool calls) x (context resent per call): measured on a live
# branch, each call carries ~40k tokens, so an art round at 40 calls costs
# about as much as one and a half repair rounds, and art rounds were 63.5% of
# a run's bill on newloop_self40. Nothing in the measurement said the extra
# fourteen calls bought anything; the art list is capped at six assets either
# way. Back to the repair budget.
ART_TOOLS = int(os.environ.get('RSIGAME_CONTROLLER_ART_TOOLS') or PJ.REPAIR_TOOLS)

# A SESSION THAT PRODUCED NOTHING IS RETRIED; ONE THAT SPENT ITS BUDGET IS NOT.
#
# `_run_repair_agent` has a retry ladder written for exactly this: a dropped
# platform session, a silent start, a model that answered with a malformed
# tool call. Six silent starts measured on glm, none recovered inside the
# session, and every retry that followed did the work.
#
# This loop inherited `max_attempts=1` from the Self-Refine arm, where it is
# correct: that arm has to reach its total in EIGHT sessions of 26 tools, so a
# retry hands back an eighth of the game's budget and a budget-matched
# comparison stops being one. Neither half of that holds here -- thirty rounds,
# and `_session_died` only fires when the attempt made NO tool call, so the
# retry re-spends nothing. Measured tonight: two consecutive art rounds on one
# game died at 7.1s and 8.0s with zero tools and zero change to the tree,
# because the model emitted a raw shell command where a structured call
# belonged. Both were recorded as rounds that did nothing.
REPAIR_ATTEMPTS = int(os.environ.get('RSIGAME_CONTROLLER_REPAIR_ATTEMPTS') or 3)

# A CAP ON IMAGES PER ROUND, and the balance put in front of the agent.
#
# Measured over 260 art rounds of the three-arm run: median 5 images a round,
# 90th percentile 21, maximum 63 -- and that 63 was three attempts of one
# platformer round, each cut off at its 30 minutes before wiring anything in,
# each read as a dead session and retried. 0 (the default) leaves the tool and
# the retry ladder exactly as they were. The ledger is per round and shared by
# every attempt at that round; `generate_game_assets` writes it and
# `_run_repair_agent` reads it (to tell each attempt what is left, and to stop
# retrying a session that generated images).
ASSET_CAP = int(os.environ.get('RSIGAME_CONTROLLER_ASSET_CAP') or 0)


def asset_budget(rdir: Path) -> Path | None:
    """Point this round's repair at its own ledger. None when the cap is off."""
    if ASSET_CAP <= 0:
        os.environ.pop('OPENGAME_ASSET_ROUND_CAP', None)
        os.environ.pop('OPENGAME_ASSET_LEDGER', None)
        return None
    led = rdir / 'asset_ledger.json'
    os.environ['OPENGAME_ASSET_ROUND_CAP'] = str(ASSET_CAP)
    os.environ['OPENGAME_ASSET_LEDGER'] = str(led)
    return led


def asset_budget_record(led: Path | None, rr: dict) -> dict | None:
    if led is None:
        return None
    try:
        d = json.loads(led.read_text())
    except Exception:
        d = {}
    return {'cap': ASSET_CAP, 'used': d.get('used', 0),
            'calls': len(d.get('events') or []),
            'refused': sorted({k for e in (d.get('events') or [])
                               for k in (e.get('refused') or [])}),
            'last_attempt_images': (rr or {}).get('images_generated'),
            'cut_off_holding_assets': (rr or {}).get('cut_off_holding_assets')}


# WHAT THE APPENDED ROUNDS ARE FOR. `presentation` reads how the game looks
# off representative frames; `feedback` reads whether acting on it lands, off
# the frames either side of each input. Same chain, same gates, different
# question and different pictures.
ART_FOCUS = os.environ.get('RSIGAME_CONTROLLER_ART_FOCUS') or 'presentation'


def is_art_round(r: int, total: int) -> bool:
    return ART_ROUNDS > 0 and r > total - ART_ROUNDS


# WHO DECIDES THAT A ROUND IS AN ART ROUND.
#
# `fixed` is what the first runs used: the last `ART_ROUNDS` rounds are art
# rounds and `RSIGAME_CONTROLLER_ART_FOCUS` says which question they ask for the whole run.
# That schedule proved the mechanism works and proved nothing about when to
# use it, because a human wrote it.
#
# `monitor` hands the choice to the global monitor. Two of the four directions
# it already votes on -- `presentation` and `feedback` -- are exactly the two
# questions the one-call chain was built to answer, so a round that comes back
# with one of those runs that chain instead of plan/probe/ground/rank/repair.
# Nothing about the monitor's vocabulary changes. What changes is that its vote
# now reaches the round.
DISPATCH = (os.environ.get('RSIGAME_CONTROLLER_DISPATCH') or 'fixed').lower()

# `correctness` and `completeness` keep the full chain: a defect has to be
# found before it can be repaired, and the art round has no finder in it.
LEAN_DIRECTIONS = ('presentation', 'feedback')


def round_kind(r: int, total: int, direction: str) -> str:
    # `rotate` executes a direction exactly as `monitor` would: the arms differ
    # in which direction is chosen, never in how it is carried out.
    if DISPATCH in ('monitor', 'rotate'):
        return 'art' if direction in LEAN_DIRECTIONS else 'normal'
    return 'art' if is_art_round(r, total) else 'normal'


# EVERY ROUND'S TREE, KEPT. A round's record says what it changed; the tree
# itself is the only thing that can answer a question nobody thought to record
# in advance, and reconstructing the missing ones afterwards cost two days and
# still left one game unrecoverable. `.godot` is left out -- it is the import
# cache, it is a third of the bytes, and `godot --import` rebuilds it.
SNAPSHOT = (os.environ.get('RSIGAME_CONTROLLER_SNAPSHOT') or '') not in ('', '0', 'no')
# Two-call design: the estimator measures, the monitor decides. Off by
# default so the existing ten-game run stays the control arm.
ESTIMATOR = os.environ.get('RSIGAME_CONTROLLER_ESTIMATOR', '') not in ('', '0')


def snapshot_tree(g, r: int):
    if not SNAPSHOT:
        return None
    import shutil
    dst = g.dir / f'round_{r:02d}' / 'tree'
    if dst.exists():
        return str(dst)
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(g.work, dst, symlinks=True,
                        ignore=shutil.ignore_patterns('.godot'))
        # The checklist is state too, and it is the half of the monitor's
        # input that is not in the tree.
        st = g.dir / 'checklist_state.json'
        if st.is_file():
            shutil.copy2(st, dst.parent / 'checklist_state.json')
        return str(dst)
    except Exception as exc:
        log(g.dir, f'  snapshot failed r{r}: {type(exc).__name__}: {str(exc)[:120]}')
        return None
DIRECTIONS = ('correctness', 'completeness', 'feedback', 'presentation')


def log(gdir: Path, *bits):
    line = ' '.join(str(b) for b in bits)
    print(line, flush=True)
    with (gdir / 'loop.log').open('a') as f:
        f.write(line + '\n')


def engine(work: Path) -> str:
    """'godot' or 'web', from the tree itself -- the rule `project` uses."""
    return PJ.engine_of(Path(work))


def build_tree(work: Path) -> dict:
    """`PJ.build`, plus one thing a web tree needs and a Godot tree does not.

    A FAILED WEB BUILD LEAVES THE LAST GOOD ONE BEHIND. `npm run build` runs
    `tsc` before `vite`, so a type error stops it before `dist/` is replaced,
    and everything that plays the game -- the probe, every replay -- would
    then play the previous round's build and report on it as this one. A
    Godot project that does not compile simply does not load, so nothing is
    removed there. Here the stale `dist/` is removed, which makes a broken web
    build look the way a broken Godot build does: not there.
    """
    b = PJ.build(work)
    if engine(work) != 'web':
        return b
    if not b.get('ok'):
        import shutil
        shutil.rmtree(Path(work) / 'dist', ignore_errors=True)
        return b
    gate = web_gate(work)
    if gate and gate.get('pass') is False:
        errs = [str(e) for e in (gate.get('errors') or [])][:6]
        b = dict(b, ok=False, runtime_gate=gate,
                 tail=('`npm run build` succeeded, but the built page fails the '
                       'benchmark\'s start-up check (loaded='
                       f"{gate.get('loaded')}, canvas={gate.get('canvas')}, "
                       f"dead on first input={gate.get('dead_on_interaction')}):\n"
                       + '\n'.join(errs)))
    return b


def web_gate(work: Path) -> dict | None:
    """The benchmark's own build gate, run on the build just made.

    `npm run build` passing says the TypeScript compiles; it does not say the
    game starts. The scorer gates every web checkpoint on
    `tools/web_build_check.py` -- page loads, a canvas exists, no uncaught error
    at start-up, and not an error plus a frozen canvas after one click -- and a
    build that fails it scores 0. Measured on the Phaser SR20 runs (2026-09-15):
    Qwen checkpoints scored 0 for exactly this in rounds the loop had recorded
    as built, e.g. `Cannot read properties of undefined (reading 'duration')`
    on four consecutive checkpoints of one game. Running the same script here
    makes the loop's "built" mean what the score means.

    None when the check could not run at all (no script, the browser itself
    failed): that says nothing about the game, so the build is left as it was.
    """
    tool = Path(os.environ.get('GAMECRAFT_BENCH') or '') / 'tools' / 'web_build_check.py'
    if not tool.is_file():
        return None
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as fh:
        out = Path(fh.name)
    try:
        subprocess.run([sys.executable, str(tool), '--project', str(work),
                        '--json-out', str(out)],
                       capture_output=True, text=True, timeout=180)
        rep = json.loads(out.read_text())
    except Exception:
        return None
    finally:
        out.unlink(missing_ok=True)
    if any(str(e).startswith('probe failed') for e in (rep.get('errors') or [])):
        return None
    return rep


class Game:
    """One game, its tree, and everything the rounds have written down."""

    def __init__(self, game: str, root: Path, corpus: Path, port: int):
        self.game, self.port = game, port
        self.dir = root / 'runs' / game
        self.dir.mkdir(parents=True, exist_ok=True)
        self.work = self.dir / 'work'
        if not self.work.is_dir():
            PJ.fresh_copy(game, self.work, corpus=corpus)
            self.import0 = PJ.godot_import(self.work, force=True)
            if engine(self.work) == 'web':
                # `fresh_copy` leaves `dist/` behind, and round 1's probe plays
                # `dist/`. Without this it has nothing to play.
                self.build0 = build_tree(self.work)
        self.ledger = Ledger(self.dir / 'targets.json')
        self.rounds: list[dict] = []
        f = self.dir / 'rounds.json'
        if f.is_file():
            self.rounds = json.loads(f.read_text())
        self.demo_desc = describe(self.work, self.dir / 'demo_descriptions.json')
        self.start_proxy()
        self.write_settings()

    def write_settings(self):
        """Which rules this run is under. Nothing else records them.

        Every mechanism inherited from the old arm is an env flag defaulting
        to '0', and neither framework has ever written down which were set.
        Two runs of the "same" arm with different flags are two different
        experiments, and months later there is no way to tell them apart from
        the result file. Written before round 1 so a crashed run still says
        what it was.
        """
        keys = [k for k in os.environ
                if k.startswith('RSIGAME_')]
        (self.dir / 'settings.json').write_text(json.dumps(
            {'flags': {k: os.environ[k] for k in sorted(keys)},
             'models': {k: os.environ.get(k) for k in
                        ('RSIGAME_MODELS_LOOP_MODEL', 'RSIGAME_MODELS_PLAY_MODEL', 'RSIGAME_REPAIR_MODEL')},
             'argv': sys.argv, 'started': time.time()},
            ensure_ascii=False, indent=1))

    # Which variables name an endpoint this loop talks to. The repair
    # subprocess reads its own pair and is handled separately, because it is a
    # subprocess and inherits the environment we hand it.
    _PROXIED = ('RSIGAME_JUDGE_QUALITY_JUDGE_BASE_URL', 'RSIGAME_GENERATOR_BASE_URL',
                'OPENAI_BASE_URL', 'RSIGAME_MONITOR_BASE_URL')
    _UPSTREAM_DEFAULT = 'https://openrouter.ai/api/v1'

    def start_proxy(self):
        """Meter every model call this run makes, not only the repair subprocess.

        The repair agent reports nothing of its own -- `log.json`'s `cost` is
        null on this endpoint -- so a round that went silent leaves no record of
        how long its context had grown. That was the original reason for the
        proxy. But metering only the repair half answers "what did the coding
        agent cost", while the question a cost column is read for is "what did
        the run cost", and the Controller, Explorer, Verifier and Monitor calls
        were missing from it entirely.

        ONLY ENDPOINTS THAT ALREADY MATCH. The proxy forwards to one upstream
        and signs with one key, so pointing a variable at it that named a
        different host would send those calls somewhere they do not belong --
        the Monitor in particular defaults to its own provider. A variable is
        therefore redirected only when it already resolves to this upstream,
        and the ones left alone are named in the log rather than passed over in
        silence: a cost column that quietly omits a component is worse than one
        that says what it omits.
        """
        try:
            from rsigame.eval.token_proxy import TokenProxy
            up = (os.environ.get('RSIGAME_JUDGE_QUALITY_JUDGE_BASE_URL')
                  or os.environ.get('RSIGAME_GENERATOR_BASE_URL')
                  or os.environ.get('OPENAI_BASE_URL')
                  or self._UPSTREAM_DEFAULT)
            key = (os.environ.get('OPENROUTER_API_KEY')
                   or os.environ.get('OPENAI_API_KEY') or '')
            self.proxy = TokenProxy(up, self.dir / 'all_token_calls.jsonl', key)
            url = self.proxy.start()
            os.environ['RSIGAME_REPAIR_OPENAI_BASE_URL'] = url
            os.environ['RSIGAME_REPAIR_API_KEY'] = key
            on, off = [], []
            for var in self._PROXIED:
                cur = (os.environ.get(var) or self._UPSTREAM_DEFAULT).rstrip('/')
                if cur == up.rstrip('/'):
                    os.environ[var] = url
                    on.append(var)
                else:
                    off.append(f'{var}={cur}')
            log(self.dir, f'  token proxy metering {len(on) + 1} endpoint(s)'
                          + (f'; NOT metered: {", ".join(off)}' if off else ''))
        except Exception as exc:
            self.proxy = None
            log(self.dir, f'  token proxy did not start: {type(exc).__name__}: {str(exc)[:120]}')

    def stop_proxy(self):
        """Totals beside the call log, so a reader needs no aggregation step."""
        if not getattr(self, 'proxy', None):
            return
        try:
            (self.dir / 'all_token_totals.json').write_text(
                json.dumps(self.proxy.snapshot(), indent=1))
        except Exception as exc:
            log(self.dir, f'  token totals not written: {type(exc).__name__}')
        finally:
            try:
                self.proxy.stop()
            except Exception:
                pass

    def save(self):
        (self.dir / 'rounds.json').write_text(
            json.dumps(self.rounds, indent=1, ensure_ascii=False))
        self.ledger.save()

    # -- the state the controller reads -------------------------------
    def state_run(self) -> Run:
        """A Run view over this game's own directory."""
        self.ensure_checklist()
        return Run(self.dir)

    def ensure_checklist(self) -> dict:
        """The frozen requirement list, built once and kept.

        Extraction is cached by the spec's own hash, so this costs nothing
        after the first game that reads a given spec.
        """
        f = self.dir / 'checklist_state.json'
        if f.is_file():
            return json.loads(f.read_text())
        import rsigame.checklist.task_checklist as TC, rsigame.checklist.bootstrap_verifier as BV
        import rsigame.checklist.checklist_state as CS
        cl = TC.extract(self.game, self.work)
        items = cl.get('items') or []
        if not items:
            return {'items': []}
        general = [{'id': k, 'check': v} for k, v in BV.GENERAL_CHECKS.items()]
        stages = BV.classify_stages(
            items + [{'id': g['id'], 'requirement': g['check']} for g in general])
        # No bootstrap reading: nothing has been observed yet, and inventing a
        # status would be a claim about a build nobody has looked at.
        state = CS.init_state(cl, {'statuses': [], 'general': general},
                              stages=stages)
        (self.dir / 'checklist_meta.json').write_text(
            json.dumps({'stages': stages, 'scenarios': {}},
                       ensure_ascii=False, indent=1))
        f.write_text(json.dumps(state, indent=1, ensure_ascii=False))
        log(self.dir, f'  checklist: {len(state["items"])} requirements '
                      f'({len(items)} task + {len(general)} general)')
        return state

    def history_for_prompt(self, limit: int = 0) -> list[dict]:
        """The per-round history block, last HISTORY_ROUNDS rounds only.

        It used to hand over every round the run had ever done. The packet
        itself stayed small (the renderer already sliced it), but everything
        downstream that walks this list -- the art ledger, the direction table
        -- grew for the whole run. A window keeps every consumer bounded, and
        the rounds that matter to the next edit are the recent ones: what the
        run did twenty rounds ago is in the code, not in the log.
        """
        n = limit or HISTORY_ROUNDS
        out = []
        for rec in (self.rounds[-n:] if n > 0 else self.rounds):
            v = rec.get('verification') or {}
            art = (rec.get('repair_report') or {}).get('report') or {}
            files = [Path(f).name for f in
                     ((rec.get('repair') or {}).get('files_edited') or [])]
            files += [f for f in ((art.get('art') or {}).get('files') or [])
                      if Path(f).name not in files]
            out.append({'round': rec['round'],
                        'goal': rec.get('primary_issue') or rec.get('question'),
                        'verdict': v.get('primary_verdict'),
                        'files': files,
                        'note': (art.get('note') or '')[:300],
                        'todo': (art.get('todo') or '')[:300],
                        # What the round was asked for about how it looks, and
                        # what came of it. Without this the next round's ask is
                        # written against nothing: four rounds of one game were
                        # handed the same sentence about thirty shipped images.
                        'art_ask': ((rec.get('art') or {}).get('goal') or '')[:200],
                        'art_verdict': next(
                            (i.get('verdict') for i in (v.get('issues') or [])
                             if i.get('issue_id') == 'ART'), None),
                        'build_ok': rec.get('build_ok'),
                        'skipped': [x for x in (v.get('issues') or [])
                                    if x.get('verdict') == 'not_attempted']})
        return out

    def direction_rows(self) -> list[dict]:
        return [{'direction': rec.get('direction'),
                 'accepted': (rec.get('verification') or {}).get(
                     'primary_verdict') == 'fixed'}
                for rec in self.rounds if rec.get('direction')]


def backfill_verdict(g: Game, r: int, a: dict) -> None:
    """This round's opening read IS the previous art round's verification.

    An art round asks for specific things; the round after it looks at the
    frames those asks produced, and `landed` is one line per ask saying `yes`,
    `no` or `cannot_tell`. That is step one of `controller.verify` performed by
    a different reader, and it was being thrown away -- art rounds carried no
    `primary_verdict`, so `direction_rows` counted every one of them as not
    accepted, and `presentation` would have read as permanently stalling the
    moment the monitor could see those rounds at all.

    Step two is the same arithmetic, with the same rule underneath it:
    `cannot_tell` is `unobserved`, never `failed`. A reader that could not tell
    has not shown the ask did not land.
    """
    prev = next((x for x in reversed(g.rounds)
                 if x.get('round') == r - 1 and x.get('kind') == 'art'), None)
    if prev is None or prev.get('verification'):
        return
    rows = [x for x in (a.get('landed') or []) if isinstance(x, dict)]
    if not rows:
        return
    yes = sum(1 for x in rows if x.get('seen') == 'yes')
    no = sum(1 for x in rows if x.get('seen') == 'no')
    # A ROUND THAT CHANGED NOTHING CANNOT HAVE LANDED ANYTHING.
    #
    # The reader ticks off what it can see, and what it can see may have been
    # put there two rounds ago. Measured tonight: ten art rounds edited no file
    # and moved no counted fact -- a session that died at seven seconds, one
    # that spent sixteen tool calls reading -- and NINE of them were written
    # down as `fixed_incidentally`, crediting them with someone else's work.
    # The counted facts settle it: no file, no change on the tree, nothing was
    # attempted here.
    a0 = prev.get('visual_before') or {}
    a1 = prev.get('visual_after') or {}
    moved = any(a0.get(k) != a1.get(k) for k in
                ('images_present', 'images_used', 'texture_draws',
                 'primitive_draws'))
    if not moved and not (prev.get('repair') or {}).get('files_edited'):
        prev['verification'] = {
            'issues': [], 'primary_verdict': 'not_attempted',
            'source': f'art_list.landed read at round {r}, overridden: '
                      f'this round changed no file and no counted fact',
            'counts': {'yes': yes, 'no': no,
                       'cannot_tell': len(rows) - yes - no}}
        log(g.dir, f'  backfill r{r - 1}  verdict: not_attempted '
                   f'(the tree did not change this round; the {yes} the reader saw came from earlier rounds)')
        return
    # The agent's own account of whether it tried, taken from what it edited
    # rather than from what it said: an art round that touched no file did not
    # attempt anything, whatever its closing block claims.
    tried = bool((prev.get('repair') or {}).get('files_edited'))
    if yes:
        v = 'fixed' if tried else 'fixed_incidentally'
    elif no:
        v = 'failed' if tried else 'not_attempted'
    else:
        v = 'unobserved'
    prev['verification'] = {
        # `verdict` ON EVERY ISSUE, not only at the top. `history_for_prompt`
        # reads each issue's own verdict to tell the next round what became of
        # the last art ask, and a row without the key took down every round
        # after the first art round -- including the normal ones, which read
        # the same list.
        'issues': [{'issue_id': 'ART', 'role': 'primary', 'verdict': v,
                    'observed': x.get('seen'), 'asked': x.get('asked', ''),
                    'note': x.get('note', '')} for x in rows],
        'primary_verdict': v,
        'source': f'art_list.landed read at round {r}',
        'counts': {'yes': yes, 'no': no,
                   'cannot_tell': len(rows) - yes - no}}
    log(g.dir, f'  backfill r{r - 1}  verdict: {v} · landed {yes}/{len(rows)}')


def art_verify(g: Game, r: int, frames: list) -> dict | None:
    """The previous art round's asks, re-checked by a reader that was not told.

    WHY THIS EXISTS ALONGSIDE `backfill_verdict`. That one asks the round's own
    reader whether the things IT asked for are there, which is the question a
    requester should never be the one to answer -- and it collapses a list of
    eight asks into one label the moment any single one lands. Measured over
    one night: 109 of 152 art rounds were recorded `fixed`, and `fixed` is the
    only verdict `direction_rows` counts, so the monitor compared that number
    against a `completeness` column scored by blind verification on one named
    defect. Two different rulers, read as one.

    This is the same two steps every other round gets, on the same code:

      1. `VERIFY.observe` is handed each ask's `now` -- what the reader said
         was DRAWN THERE AT THE TIME, not what was requested -- as a prior
         observation to re-check, and is told nothing about anyone repairing
         anything. It answers still_there / gone / not_exercised.
      2. `VERIFY.verdicts` combines that with the agent's own account of what
         it generated and wired. The agent knows what it tried; only the
         frames know what happened; neither knows both.

    The primary is the ask the reader itself ranked 1, so a round is `fixed`
    when the thing it led with landed -- not when any of eight did.
    """
    from rsigame.controller import verify as VERIFY
    from rsigame.controller.art_list import _same_thing

    prev = next((x for x in reversed(g.rounds)
                 if x.get('round') == r - 1 and x.get('kind') == 'art'), None)
    if prev is None:
        return None
    pl = prev.get('art_list') or {}
    issues = []
    for tag, rows in (('asset', pl.get('assets') or []),
                      ('fx', pl.get('effects') or []),
                      ('ui', pl.get('layout') or [])):
        for x in rows:
            if not isinstance(x, dict):
                continue
            name = x.get('key') or x.get('when') or x.get('what') or ''
            now = str(x.get('now') or '').strip()
            if not name or not now:
                continue
            issues.append({'issue_id': f'{tag}:{name}'[:70], 'issue': now,
                           'evidence_refs': [str(x.get('seen_at') or '')],
                           'rank': int(x.get('rank') or 99), 'name': name})
    if not issues:
        return None
    issues.sort(key=lambda i: i['rank'])

    # WHAT THE AGENT SAYS IT TOUCHED. Keys drift between the ask and the
    # delivery -- `battle_platforms` comes back as `battle_platform_art` --
    # so the same token match the history table uses is used here.
    rep = ((prev.get('repair_report') or {}).get('report') or {})
    claimed = [str(x) for x in
               ((rep.get('generated') or []) + (rep.get('wired') or [])
                + (rep.get('effects_done') or []) + (rep.get('layout_done') or []))]
    touched = [str(f) for f in ((prev.get('repair') or {}).get('files_edited') or [])]
    # ABSENT FROM A LIST MEANS "NOT ATTEMPTED" ONLY IF THERE IS A LIST.
    #
    # Half the art rounds are cut at the time budget and write no closing
    # block, so `rep` is empty -- and an empty list is not the agent saying it
    # skipped everything, it is the agent saying nothing. `controller.verify`
    # already draws that distinction: `attempted` is None for an ask the agent
    # gave no account of, and None is NOT False. Reporting False here would
    # turn every landed ask in a truncated round into `fixed_incidentally`,
    # which reads as "someone else did it".
    has_account = bool(claimed)

    def attempted(i):
        if any(_same_thing(i['name'], c) for c in claimed):
            return True
        if any(_same_thing(i['name'], Path(f).stem) for f in touched):
            return True
        return False if has_account else None

    items = [{'id': x['id'], 'requirement': x['requirement']}
             for x in (g.state_run().state.get('items') or [])][:40]
    obs = VERIFY.observe([{k: v for k, v in i.items() if k != 'rank'}
                          for i in issues], items, frames,
                         model=PJ.RSIGAME_MODELS_LOOP_MODEL)
    if obs.get('error'):
        return {'error': obs['error']}

    primary = issues[0]['issue_id']
    # An ask whose `attempted` is unknown is left OUT of the report, which is
    # how `verify.verdicts` is told "nobody supplied this" -- it keys on
    # presence, so an entry carrying None would be read as False.
    pa = attempted(issues[0])
    report = {'primary': ({'issue_id': primary, 'attempted': pa}
                          if pa is not None else {}),
              'concurrent': [{'issue_id': i['issue_id'], 'attempted': a}
                             for i, a in ((i, attempted(i)) for i in issues[1:])
                             if a is not None]}
    sel = {'primary_target': primary,
           'concurrent_minor_fixes': [i['issue_id'] for i in issues[1:]]}
    out = VERIFY.verdicts(obs.get('observations') or [], report, sel)
    out['source'] = f'blind re-read at round {r} (controller.verify)'
    out['schema_problems'] = obs.get('schema_problems') or []
    return out


def replay_as_ex(g: Game, rdir: Path) -> dict:
    """The round's own replay, in the shape the polish reader wants.

    `polish_read` was written against `execute.run`'s output and an art round
    never calls it. The events it needs are already on disk -- the replay this
    round just made -- so this reads them back rather than playing anything.
    """
    from rsigame.monitor.replay_pair import _rec_from_recorded
    src = rdir / 'post_replay'
    out = []
    if not src.is_dir():
        return {'events': []}
    for p in sorted(src.glob('frames_*')):
        did = p.name[len('frames_'):]
        rec = _rec_from_recorded(src, did,
                                 g.work / 'demo_outputs' / f'{did}.json')
        out += [e for e in ((rec or {}).get('events') or []) if e.get('before')]
    return {'events': out[:40]}


# THE ROTATION ARM'S ORDER, FIXED IN ADVANCE.
#
# The control this arm provides is "spend the budget evenly without looking",
# so the order must not be tuned: it is the four directions in the sequence the
# checklist itself names them, cycled. Anything picked to look good would make
# the comparison a contest between two policies rather than a measurement of
# what looking is worth.
ROTATION = ('correctness', 'completeness', 'feedback', 'presentation')


def choose_direction(g: Game, r: int) -> tuple[str, dict | None]:
    """The global monitor, every third round. Advisory: it is asked, not obeyed.

    Rounds between monitor calls keep the last direction. Asking every round
    would spend a reading on a question whose answer cannot have changed much,
    and the monitor is built to look at a checkpoint, not at a single edit.

    UNDER `RSIGAME_CONTROLLER_DISPATCH=rotate` the monitor is still asked and still recorded
    -- the two arms must differ by what drives the direction and by nothing
    else, and its reading is the thing the comparison is about -- but the
    direction comes from the block number instead of from its answer.
    """
    if DISPATCH == 'rotate':
        return ROTATION[((r - 1) // MONITOR_EVERY) % len(ROTATION)], None
    if DISPATCH == 'inner':
        # Nobody decides a direction for this arm. The planner sees the whole
        # checklist and picks both its question and the kind of round, from the
        # build in front of it rather than from a checkpoint comparison three
        # rounds wide. Same actions as the other arms; different place for the
        # decision.
        return None, None
    last = next((rec.get('direction') for rec in reversed(g.rounds)
                 if rec.get('direction')), 'correctness')
    # The monitor speaks at the end of a checkpoint round, on that round's own
    # comparison, and what it says sets the direction from the next round on.
    # So `choose_direction` only reads what it already said.
    said = next((rec.get('monitor_said') for rec in reversed(g.rounds)
                 if rec.get('monitor_said')), None)
    if said and said.get('next_direction') in DIRECTIONS:
        return said['next_direction'], None
    return last, None


def est_summary(g: Game, r: int) -> dict | None:
    """Call 1 of the two-call design: what the proxy rubric makes of the block
    just finished. Returns `aggregate.pair()['proxy_value']`, or None.

    OFF UNLESS ASKED. `RSIGAME_CONTROLLER_ESTIMATOR=1` turns it on, because the ten-game run
    that exists was made without it and is the control arm of the ablation --
    a default-on estimator would silently make the two arms differ by more
    than the thing under test.

    NEVER FATAL. Every failure path returns a dict with `error`, which the
    caller records and does not pass to the monitor. A measurement that did
    not come back must cost its own reading and nothing else; the monitor has
    decided without this input for the whole of the existing run and can go on
    doing so.

    THE PAIR IS THIS CHECKPOINT AGAINST THE LAST ONE, three rounds back, which
    is the block the monitor is about to judge.

    AT THE FIRST CHECKPOINT THAT IS ROUND 0 -- the untouched G0, replayed once
    before the run starts and left in `round_00/post_replay`. The loop never
    replays G0 on its own, so for the whole of the first ten-game run the
    opening block was the only one with no measurement at all, and it is the
    block where the largest moves happen: plumber gained 0.225 across it, and
    towerdefense's redirect at its first checkpoint set the direction of the
    following twenty-seven rounds. If that baseline is missing the function
    still returns None rather than inventing a comparison.
    """
    if not ESTIMATOR:
        return None
    prev = r - MONITOR_EVERY
    if prev < 0:
        return None
    if not any((g.dir / f'round_{prev:02d}' / 'post_replay').glob('frames_*')):
        return {'error': f'round {prev} has no comparable post_replay'}
    t0 = time.time()
    try:
        from concurrent.futures import ThreadPoolExecutor
        from rsigame.eval.estimator import pairs as P, score_pair, aggregate
        from rsigame.eval.estimator.rubric import load as load_rubric
        # The estimator addresses runs through a module-level root. Each game
        # is its own process, so pointing it at this game's tree is safe and
        # keeps the estimator usable against any run root, not just the one
        # its default names.
        P.RUNS = g.dir.parent
        rb = load_rubric(g.dir.name)
        d = P.build(g.dir.name, prev, r)
        if d.get('error'):
            return {'error': d['error']}
        demos = [x for x in d['demos'] if x.get('content')]
        if not demos:
            return {'error': 'no usable demo',
                    'problems': [x.get('problems') for x in d['demos']]}
        with ThreadPoolExecutor(max_workers=len(demos)) as ex:
            per = list(ex.map(lambda x: score_pair.score(x, rb), demos))
        bad = [p.get('error') for p in per if p.get('error')]
        per = [p for p in per if not p.get('error')]
        if not per:
            return {'error': f'every demo failed to score: {bad[:1]}'}
        out = aggregate.pair(per, rb)['proxy_value']
        out['_meta'] = {'r_old': prev, 'r_new': r, 'n_demos': len(per),
                        'failed': len(bad), 'seconds': round(time.time() - t0, 1)}
        return out
    except Exception as exc:
        return {'error': f'{type(exc).__name__}: {str(exc)[:200]}'}


def ask_monitor(g: Game, r: int, summary: dict, last: str) -> dict | None:
    """One reading, on the checkpoint just measured. Advisory: it is recorded."""
    try:
        from rsigame.monitor import monitor as GM, consolidate as CONS
        if not summary or summary.get('error'):
            return None
        items = [{'id': i['id'], 'requirement': i['requirement'],
                  'status': i.get('status'), 'stage': i.get('stage'),
                  'stale': i.get('stale')}
                 for i in (g.state_run().state.get('items') or [])]
        # The presentation reader runs every round and was never shown to the
        # monitor. `presentation` was in its vocabulary and its history table
        # said `untested`, but nothing it could see spoke to it, so it chose
        # the direction zero times in forty checkpoints.
        pol = [x.get('polish') for x in g.rounds
               if x.get('polish') and not x['polish'].get('error')]
        cur = pol[-1] if pol else None
        prev = next((x for x in reversed(pol[:-1])
                     if x.get('round') is not None
                     and x['round'] <= r - MONITOR_EVERY), None)
        est = est_summary(g, r)
        said = GM.assess(summary, items,
                         CONS.direction_history(g.direction_rows()[-HISTORY_ROUNDS:]),
                         budget=g.total - r, total_budget=g.total,
                         estimator=(est if est and not est.get('error')
                                    else None),
                         current_direction=last, polish=cur, polish_prev=prev,
                         prompt_out=str(g.dir / f'round_{r:02d}' / 'monitor_prompt.txt'))
        # RECORDED EVEN WHEN IT FAILED, and recorded whether or not it was
        # passed on: the ablation has to be able to tell a checkpoint the
        # monitor decided WITH this input from one where the estimator errored
        # and it decided without, and after the fact the prompt file alone
        # cannot say which happened.
        if isinstance(said, dict) and est is not None:
            said['estimator'] = est
        return said
    except Exception as exc:
        return {'error': f'{type(exc).__name__}: {str(exc)[:200]}'}


def move_statuses(g: Game, summary: dict, r: int) -> None:
    """Let the checkpoint comparison move requirement statuses.

    It is the only reading here that talks about requirements by id, so it is
    the only one that can. A criterion it saw and did not fault reads
    `satisfied`; one it faulted reads `confirmed_gap`; one it never exercised
    is left alone, because not looking is not a finding.
    """
    f = g.dir / 'checklist_state.json'
    if not f.is_file():
        return
    import rsigame.checklist.checklist_state as CS
    from rsigame.controller import checklist_read as CR
    bs = CR.from_comparison(summary, r)
    if not bs['statuses']:
        return
    st = json.loads(f.read_text())
    moves = CS.update(st, bs, r)['moves']
    CS.save(st, f)
    log(g.dir, f"  checklist(monitor): {len(moves)} status changes · "
               f"observed {len(bs['statuses'])}")


def compare_checkpoints(g: Game, r: int) -> dict | None:
    """Pair this build against the last monitored one, for the global stage."""
    try:
        from rsigame.monitor.replay_pair import pair_checkpoints
        from rsigame.monitor import comparator, consolidate as CONS
        prev = [rec for rec in g.rounds if rec.get('demos_replayed_at')]
        if not prev:
            return None
        old = g.dir / f"round_{prev[-1]['demos_replayed_at']:02d}" / 'post_replay'
        new = g.dir / f'round_{r:02d}' / 'post_replay'
        ids = sorted(p.name[len('frames_'):] for p in new.glob('frames_*'))
        if not ids:
            return None
        pairs = pair_checkpoints(old, new, ids, f"round {prev[-1]['demos_replayed_at']}",
                                 f'round {r}',
                                 frozen_corpus=g.work)
        items = [i for i in (g.state_run().state.get('items') or [])
                 if i.get('source') == 'task']
        recs = [comparator.compare(p, items) for p in pairs if p.aligned]
        return CONS.consolidate(recs, items, old_label=f"round {prev[-1]['demos_replayed_at']}",
                                new_label=f'round {r}')
    except Exception as exc:
        return {'error': f'{type(exc).__name__}: {str(exc)[:200]}'}


def fix_fake_pngs(work: Path) -> list:
    """Re-encode generated files whose bytes are JPEG and whose name says PNG.

    THE ASSET TOOL WRITES BACKGROUNDS AS JPEG AND NAMES THEM `.png`. A browser
    sniffs the content type and does not care, which is why this never showed
    on the Phaser games the tool was written for. Godot picks its importer by
    EXTENSION: it hands JPEG bytes to the texture importer, the import fails,
    and the `.import` file it leaves behind says `valid=false` with no path to
    a cached texture. `load()` then returns null and the whole build check
    fails on `Failed loading resource`.

    Measured the first time an art round generated a background: three of the
    twelve files it made were JPEG -- `title_bg`, `battlefield_bg` and the
    generator's own `style_anchor` -- and the project stopped loading. Running
    the import again does not help; the bytes are simply wrong for the name.
    Re-encoding them fixes it, which is what this does.

    Not the agent's problem to notice: it asked for a background, one arrived,
    and nothing it can see says the file is the wrong format.
    """
    work = Path(work)
    if engine(work) == 'web':
        # A browser sniffs the bytes, so a JPEG named `.png` loads; and the
        # rglob below would rewrite the copies under `dist/` too.
        return []
    for root in PJ.CORPORA.values():
        root = root.resolve()
        if work.resolve() == root or root in work.resolve().parents:
            return []
    from PIL import Image
    fixed = []
    for p in work.rglob('*.png'):
        if any(x in p.parts for x in ('.godot', 'demo_outputs', '_repair_evidence')):
            continue
        try:
            if p.open('rb').read(4) == b'\x89PNG':
                continue
            Image.open(p).convert('RGBA').save(p, 'PNG')
            fixed.append(str(p.relative_to(work)))
        except Exception:
            continue

    # A FAILED IMPORT IS STICKY, and that is the half that actually bites.
    # Godot writes an `.import` even when the import failed -- `valid=false`,
    # no path to a cached texture -- and every later pass sees the file and
    # skips the resource, `--force` included. So a background that was JPEG on
    # Monday and is a real PNG today still does not load: the byte fix lands
    # and the game looks exactly as broken as it did. Measured on
    # battle-critter-clash, where `title_bg.png` was correct on disk, the
    # transcode found nothing to do, and the build still failed on it.
    for imp in work.rglob('*.import'):
        if any(x in imp.parts for x in ('.godot', 'demo_outputs', '_repair_evidence')):
            continue
        try:
            if 'valid=false' in imp.read_text():
                imp.unlink()
                fixed.append(str(imp.relative_to(work)) + ' (failed import record)')
        except Exception:
            continue
    return fixed


def visual_facts(work: Path) -> dict:
    """What the tree contains, counted. Never a verdict."""
    try:
        import rsigame.polish.polish_evidence as PE
        return PE.visual_facts(work)
    except Exception as exc:
        return {'error': f'{type(exc).__name__}: {str(exc)[:120]}'}


def polish_read(g: Game, r: int, ex: dict,
                probe_dir: Path | None = None) -> dict | None:
    """The polish reader, on whatever this round looked at.

    EITHER SHAPE OF EVIDENCE. It wants an exploration record. A fixed-demo
    replay is not one -- no action names, no pixel-change figures -- and both
    are recoverable: the action is in the event's description, and the change
    is the thing the pairing code already measures. A play session IS one and
    was being thrown away: the reader was built off `ex['events']`, which only
    a replay returns, so the 52 of 120 rounds where the agent explored got no
    polish read at all and no art ask reached their packet. Those are the
    rounds with the best frames in them.
    """
    import numpy as np
    from PIL import Image
    import rsigame.polish.polish_evidence as PE, rsigame.polish.polish_review as PR

    def pct(a, b):
        if not a or not b or not Path(a).is_file() or not Path(b).is_file():
            return None
        x = np.asarray(Image.open(a).convert('L').resize((128, 72)), dtype=np.float32)
        y = np.asarray(Image.open(b).convert('L').resize((128, 72)), dtype=np.float32)
        return round(float((np.abs(x - y) > 12).mean()) * 100, 1)

    # A real session, if this round ran one. Its steps already carry frames and
    # action names; `polish_evidence` wants nothing else.
    trace = (probe_dir / 'trace' / 'exploration.json') if probe_dir else None
    if trace and trace.is_file():
        try:
            exp = json.loads(trace.read_text())
            if exp.get('steps'):
                return _polish_on(g, r, exp)
        except Exception as exc:
            log(g.dir, f'  polish could not read the session: {type(exc).__name__}: {str(exc)[:90]}')

    steps = []
    for e in (ex.get('events') or []):
        what = e.get('what') or ''
        act = ('click' if what.startswith('click')
               else 'press_key' if 'key' in what else 'wait')
        after = (e.get('after') or [None])[0]
        steps.append({'n': e.get('n'), 'action': act, 'args': {},
                      'frame': after or e.get('before'),
                      'pixels_changed_pct': pct(e.get('before'), after),
                      'thought': what})
    if ex.get('final_frame'):
        steps.append({'n': len(steps) + 1, 'action': 'shot', 'args': {},
                      'frame': ex['final_frame'], 'pixels_changed_pct': None,
                      'thought': 'last frame'})
    if not steps:
        return None
    return _polish_on(g, r, {'steps': steps})


def _polish_on(g: Game, r: int, exp: dict) -> dict:
    """The review itself, on an exploration-shaped record."""
    import rsigame.polish.polish_evidence as PE, rsigame.polish.polish_review as PR
    hist = [x['polish'] for x in g.rounds if x.get('polish')
            and not x['polish'].get('error')]
    try:
        pack = PE.build(g.game, g.work, exp)
        rev = PR.review(pack, history=hist, model=PJ.RSIGAME_MODELS_LOOP_MODEL)
    except Exception as exc:
        return {'error': f'{type(exc).__name__}: {str(exc)[:200]}'}
    if rev.get('error'):
        return {'error': rev['error']}
    # `next_goal`, not `goal` -- the reader's own key. Read under the wrong
    # name this was empty on all 120 rounds and nothing noticed, because the
    # ask that reaches the packet is written downstream from `main_issue`.
    # It is the reader's own words for what it would do, and the ask is better
    # for having them.
    return {'round': r, 'focus': rev.get('focus') or '',
            'headroom': {d: rev[d]['headroom'] for d in PR.DIMENSIONS},
            'goal': rev.get('next_goal') or '',
            'why': rev.get('why_this_goal') or '',
            'main_issue': {d: rev[d]['main_issue'][:300] for d in PR.DIMENSIONS}}


def carried_damage(g: Game, r: int) -> list:
    """Regressions from earlier rounds that nothing has repaired yet."""
    fixed = set()
    for rec in g.rounds:
        for i in ((rec.get('verification') or {}).get('issues') or []):
            if i.get('verdict') in ('fixed', 'fixed_incidentally'):
                fixed.add(i['issue_id'])
    out = []
    for rec in g.rounds[-3:]:
        for dmg in (rec.get('regressions') or []):
            if dmg.get('what') and not dmg['what'].startswith('('):
                out.append(dmg)
    return out[:3]


def look_for_damage(g: Game, before: dict, after: dict, r: int) -> list:
    """What this round's repair broke, from the pair it already produced.

    The verification replays the same script against the repaired build, so
    before and after are one script on two builds -- the comparator's own input
    shape. Asking it costs one call and answers a question the verifier is not
    asked: the verifier re-checks the issues it was handed, and a repair that
    breaks something else breaks something nobody named.
    """
    try:
        from rsigame.monitor.replay_pair import pair
        from rsigame.monitor import comparator
        p = pair(before, after, f'round {r} before repair', f'round {r} after')
        if not p.aligned:
            return []
        items = [i for i in (g.state_run().state.get('items') or [])
                 if i.get('source') == 'task']
        out = comparator.compare(p, items)
        return [{'what': x.get('what', ''), 'evidence': x.get('evidence', ''),
                 'severity': x.get('severity'), 'round': r}
                for x in (out.get('regressions') or [])]
    except Exception as exc:
        return [{'what': f'(the comparison failed: {type(exc).__name__})',
                 'evidence': '', 'severity': None, 'round': r}]


def style_anchor(g: Game) -> str:
    """The phrase this game's art was asked for under, pinned on first use.

    `generate_game_assets` takes a `style_anchor`, and it is what makes a batch
    of images look like one game. Nothing was carrying it between rounds, so
    the agent invented one per call: six rounds of battle-critter-clash used
    six, swinging between "soft painted storybook cartoon" and "16-bit pixel
    art". No two batches of that game's art belong to the same picture.

    Read back out of the repair agent's own tool log rather than asked for
    separately -- the anchor that matters is the one actually passed.
    """
    f = g.dir / 'style_anchor.txt'
    if f.is_file():
        return f.read_text().strip()
    import re
    # `repair.py` puts them beside the tree, not inside it:
    #     base = child_src.parent / f"_repairs_{child_src.name}"
    logs = sorted((g.work.parent / f'_repairs_{g.work.name}').glob('*/log.json'),
                  key=lambda p: p.stat().st_mtime)
    for lg in logs:                      # oldest first: pin what was used first
        try:
            m = re.search(r'"style_anchor"\s*:\s*"([^"]{8,400})"',
                          lg.read_text(errors='replace'))
        except OSError:
            continue
        if m:
            f.write_text(m.group(1))
            return m.group(1)
    return ''


def new_images(work: Path, since: float) -> dict:
    """What the round put on disk, and how long the tool took to make it.

    Generation is the only part of an art round with a real wait in it and
    nothing has ever recorded how long it takes. Files from one call land
    within seconds of each other, so the clusters are the calls.
    """
    # On a web tree the build copies every new image into `dist/` a moment
    # later; counting those would double every call.
    skip = (('.godot', 'demo_outputs', '_repair_evidence')
            + (('dist', 'node_modules') if engine(work) == 'web' else ()))
    ts = sorted((p.stat().st_mtime, p) for p in Path(work).rglob('*.png')
                if p.stat().st_mtime > since
                and not any(x in p.parts for x in skip))
    if not ts:
        return {'n': 0, 'calls': []}
    calls, cur = [], [ts[0]]
    for t, p in ts[1:]:
        if t - cur[-1][0] < 120:
            cur.append((t, p))
        else:
            calls.append(cur)
            cur = [(t, p)]
    calls.append(cur)
    return {'n': len(ts),
            'calls': [{'files': len(c), 'span_s': round(c[-1][0] - c[0][0], 1),
                       'names': [p.name for _, p in c][:8]} for c in calls]}


def art_frames(g: Game, r: int, cap: int = 24) -> list[tuple]:
    """Frames from the last demo replay: each demo's opening, middle and end.

    Read off the directory rather than off the events, because the two frames
    that matter most here are the two the event picker has no reason to keep.
    The first frame of a demo is the title screen and the last is usually the
    game-over panel -- the two panels the layout question is mostly about --
    and neither sits beside an input.
    """
    src = None
    for k in range(r - 1, 0, -1):
        d = g.dir / f'round_{k:02d}' / 'post_replay'
        if d.is_dir() and any(d.glob('frames_*')):
            src, at = d, k
            break
    if src is None:
        return []
    dirs = sorted(src.glob('frames_*'))
    per = max(2, cap // max(1, len(dirs)))
    out = []
    for fd in dirs:
        pngs = sorted(fd.glob('*.png'))
        if not pngs:
            continue
        demo = fd.name[len('frames_'):]
        picks = [(0, 'opening frame'), (len(pngs) - 1, 'last frame')]
        for j in range(1, per - 1):
            picks.append((len(pngs) * j // max(1, per - 1), f'partway through ({j})'))
        seen = set()
        for i, label in sorted(picks):
            i = min(max(i, 0), len(pngs) - 1)
            if i in seen:
                continue
            seen.add(i)
            out.append((f'demo {demo} -- {label}', pngs[i]))
    return out[:cap]


def art_round(g: Game, r: int, rec: dict) -> dict:
    """One look, three lists, one repair session. No planner, no ranker."""
    from rsigame.controller import art_list as AL
    from rsigame.controller.repair_prompt import render_art, parse_art_report

    rdir = g.dir / f'round_{r:02d}'
    rdir.mkdir(exist_ok=True)
    # THE ROUND'S OWN QUESTION, taken from the direction it was dispatched
    # with. Under the fixed schedule this was one env var for a whole run;
    # under monitor dispatch two art rounds in the same run can ask different
    # questions, and the frames they read differ accordingly.
    focus = (rec.get('direction') if rec.get('direction') in LEAN_DIRECTIONS
             else ART_FOCUS)
    rec.update(round=r, started=time.time(), kind='art', focus=focus,
               direction=focus, project_hash=project_hash(g.work))
    rec['visual_before'] = visual_facts(g.work)
    log(g.dir, f'\n=== {g.game} round {r} · '
               + ('art round' if focus == 'presentation' else 'feedback round'))

    # -- one reading -----------------------------------------------------
    # WHETHER IT COMPILES AT ALL, before anyone reads a frame. Nothing rolls
    # back here by decision, so an art round can begin on a tree the round
    # before left broken, and the frames it reads are from the build BEFORE
    # that -- they look fine and say nothing about it. Checked here so the
    # brief can lead with it.
    b0 = build_tree(g.work)
    rec['build_before'] = bool(b0.get('ok'))
    if not rec['build_before']:
        rec['build_before_tail'] = (b0.get('tail') or '')[-800:]
        log(g.dir, f"  the build was already broken at the start: {rec['build_before_tail'][-260:]}")
    elif carried_breakage(g, r):
        rec['build_before'] = False
        rec['build_before_tail'] = carried_breakage(g, r)[-800:]
        log(g.dir, f"  broken state left by the previous round: {rec['build_before_tail'][-260:]}")

    frames = (action_frames(g, r) if focus == 'feedback'
              else art_frames(g, r))
    # WHAT WAS ASKED BEFORE AND WHAT IS TRUE ABOUT IT NOW, read off the tree.
    # The bare list this replaces was the whole reason one game asked for the
    # same character sprite three rounds running: told only that it was still
    # not visible, the reader could do nothing but ask again.
    try:
        asked = AL.history(g.work, g.rounds[-HISTORY_ROUNDS:])
    except Exception as exc:
        asked = ''
        rec['history_error'] = f'{type(exc).__name__}: {str(exc)[:140]}'
    (rdir / 'art_history.txt').write_text(asked or '')
    a = AL.read(g.game, g.work, frames, facts=rec['visual_before'],
                asked=asked, focus=focus, model=PJ.RSIGAME_MODELS_LOOP_MODEL)
    rec['art_list'] = a
    rec['n_frames_read'] = len(frames)
    if a.get('error'):
        log(g.dir, f"  art reading failed: {a['error'][:140]}")
        rec['wall_s'] = round(time.time() - rec['started'], 1)
        return rec
    landed = sum(1 for x in a.get('landed') or [] if x['seen'] == 'yes')
    # THE BLIND READING FIRST, the requester's own tick-list only if it fails.
    # One extra model call on the frames this round already read.
    try:
        av = art_verify(g, r, frames)
    except Exception as exc:
        av = {'error': f'{type(exc).__name__}: {str(exc)[:160]}'}
    prev = next((x for x in reversed(g.rounds)
                 if x.get('round') == r - 1 and x.get('kind') == 'art'), None)
    if av and not av.get('error') and prev is not None:
        prev['verification'] = av
        log(g.dir, f"  blind verify r{r - 1}: {av.get('primary_verdict')} · "
                   f"{av.get('counts')}")
    else:
        if av and av.get('error'):
            log(g.dir, f"  blind verify failed, falling back to the reading: {av['error'][:110]}")
        backfill_verdict(g, r, a)
    log(g.dir, f"  reading: still playable={a['playable']} · landed last round {landed}/"
               f"{len(a.get('landed') or [])} · assets {len(a['assets'])} "
               f"effects {len(a['effects'])} layout {len(a['layout'])}"
               + (f" · dropped {len(a['dropped'])}" if a.get('dropped') else ''))
    if a['playable'] == 'broken':
        # Said, not acted on. Nothing rolls back here by decision, and the
        # round still has a repair session -- the brief simply now leads with
        # a game that cannot be played, which is what it should work on.
        log(g.dir, f"  the screen collapsed: {a['playable_why'][:160]}")

    brief = AL.brief(a, focus=focus)
    if not rec['build_before']:
        brief = ('THIS PROJECT DOES NOT LOAD RIGHT NOW. The build check says:\n\n'
                 + rec['build_before_tail'] + '\n\nFix that FIRST and only '
                 'that until it passes. The frames below are from the last '
                 'build that ran, so they do not show this.\n\n' + brief)
    elif a['playable'] == 'broken':
        brief = ('THIS BUILD IS NOT PLAYABLE RIGHT NOW. A reader looking at '
                 'the frames says: ' + a['playable_why'] + '\n\nPut that '
                 'right first; the rest of this round comes after it.\n\n'
                 + brief)
    (rdir / 'art_brief.txt').write_text(brief)

    # -- repair ----------------------------------------------------------
    pmap = None
    try:
        import rsigame.view.project_map as PM
        pmap = PM.merge(PM.scan(g.work), PM.load_notes(g.dir))
        rendered = PM.render(pmap)
    except Exception as exc:
        rendered = ''
        rec['project_map_error'] = f'{type(exc).__name__}: {str(exc)[:120]}'
    # WHAT THE TASK ASKED FOR, and the style this game is already in. Neither
    # reached the agent before: it wrote every art description with no sight
    # of the task's own words about how the game should look, and picked a new
    # style phrase every round.
    try:
        import rsigame.polish.polish_evidence as PE
        intent = PE.design_intent(g.game, g.work)
    except Exception:
        intent = ''
    style = style_anchor(g)
    rec['style_anchor'] = style
    prompt = render_art(brief, r=r, total=g.total,
                        n_art=0 if DISPATCH == 'monitor' else ART_ROUNDS,
                        build_cmd=PJ.BUILD_CMD.get(engine(g.work), ''),
                        py=sys.executable, repo=str(PJ.REPO),
                        rounds=g.history_for_prompt(), project_map=rendered,
                        intent=intent, style=style, engine=engine(g.work))
    (rdir / 'repair_prompt.txt').write_text(prompt)

    t0 = time.time()
    try:
        from rsigame.agent.generator.repair import _run_repair_agent
        led = asset_budget(rdir)
        rr = _run_repair_agent(g.work, prompt, max_attempts=REPAIR_ATTEMPTS,
                               model=PJ.RSIGAME_REPAIR_MODEL, max_tools=ART_TOOLS,
                               tag=f'r{r:02d}')
    except Exception as exc:
        led = None
        rr = {'success': False, 'error': f'{type(exc).__name__}: {str(exc)[:200]}'}
    rec['asset_budget'] = asset_budget_record(led, rr)
    tail = ''
    for m in reversed(rr.get('messages') or []):
        c = m if isinstance(m, str) else (m.get('content') if isinstance(m, dict) else '')
        if isinstance(c, list):
            c = ' '.join(x.get('text', '') for x in c if isinstance(x, dict))
        if isinstance(c, str) and c.strip():
            tail = c
            break
    rec['repair'] = {'success': rr.get('success'), 'error': rr.get('error'),
                     'files_edited': rr.get('files_edited') or [],
                     'stopped_by': rr.get('stopped_by'),
                     'n_tools': rr.get('n_tools'),
                     # What the subprocess said on the way out. Dropped, a
                     # session that dies before it starts is indistinguishable
                     # from a model that answered nothing.
                     'stderr_tail': (rr.get('stderr_tail') or '')[-400:] or None,
                     'wall_s': round(time.time() - t0, 1)}
    if not rr.get('success'):
        log(g.dir, f"  repair did not run: {str(rr.get('error'))[:200]}")
    rec['repair_report'] = parse_art_report(tail)
    if pmap is not None:
        try:
            import rsigame.view.project_map as PM
            pmap, n_new = PM.take_notes(pmap, PJ._map_notes_from(rr))
            PM.save(pmap, g.dir)
            rec['map_notes_added'] = n_new
        except Exception:
            pass
    if g.proxy is not None:
        try:
            rec['tokens'] = PJ.repair_cost(f'r{r:02d}', g.work, proxy=g.proxy)
        except Exception:
            pass

    rec['generated_files'] = new_images(g.work, rec['started'])
    # PINNED, BUT NOT FROZEN. The first anchor a game ever used may be the
    # wrong one -- the agent now sees the task's own words about how the game
    # should look and may decide the style itself is the problem. Its report
    # says what it passed; if that differs, the pin moves and the change is
    # recorded, so the next round is consistent with what is on screen rather
    # than with a phrase nobody stands behind any more.
    used = str(((rec.get('repair_report') or {}).get('report')
                or {}).get('style_anchor') or '').strip()
    if used and used != style:
        (g.dir / 'style_anchor.txt').write_text(used)
        rec['style_anchor_changed'] = {'from': style, 'to': used}
        log(g.dir, f'  style anchor changed: {used[:100]}')
    rec['style_anchor'] = style_anchor(g)
    rec['transcoded'] = fix_fake_pngs(g.work)
    if rec['transcoded']:
        log(g.dir, f"  transcoded {len(rec['transcoded'])} fake pngs: "
                   f"{rec['transcoded'][:3]}")
    rec['import'] = PJ.godot_import(g.work)
    b = build_tree(g.work)
    rec['build_ok'] = bool(b.get('ok'))
    if not rec['build_ok']:
        rec['build_tail'] = (b.get('tail') or '')[-600:]
        log(g.dir, f"  build failed: {rec['build_tail'][-300:]}")
    rec['visual_after'] = visual_facts(g.work)
    a0, a1 = rec['visual_before'], rec['visual_after']
    moved = [f"{k} {a0.get(k)}->{a1.get(k)}" for k in
             ('images_present', 'images_used', 'texture_draws', 'primitive_draws')
             if a0.get(k) != a1.get(k)]
    art = (rec['repair_report'].get('report') or {})
    gen, wired = art.get('generated') or [], art.get('wired') or []
    log(g.dir, f"  repair: {rec['repair']['success']} "
               f"{len(rec['repair']['files_edited'])} files "
               f"{rec['repair']['wall_s']}s · generated {len(gen)} wired {len(wired)}")
    log(g.dir, f"  build: {rec['build_ok']}"
               + (f"  art: {' · '.join(moved)}" if moved else '  art: no change')
               + (f"  still will not open: {rec['import']['still_missing'][:3]}"
                  if rec['import'].get('still_missing') else ''))

    # -- the frames the next round reads, and the final checkpoint --------
    try:
        PRR.replay_demos(g.work, rdir / 'post_replay')
        runtime_parse_errors(g, rec, rdir / 'post_replay')
        if not keep_replay(g, r):
            PRR.prune(rdir / 'post_replay')
        rec['demos_replayed_at'] = r
        if r % MONITOR_EVERY == 0 or r == g.total:
            rec['comparison'] = compare_checkpoints(g, r)
            move_statuses(g, rec.get('comparison') or {}, r)
            crit = ((rec.get('comparison') or {}).get('regressions') or {}).get('critical')
            log(g.dir, f'  checkpoint: demos replayed · critical regression={crit}')
            # THE PRESENTATION CHANNEL, ON AN ART ROUND'S OWN BUILD. The
            # monitor's presentation snapshot comes from `polish_review`, and
            # an art round ran no polish read -- so the one kind of round that
            # works on presentation was the one kind the monitor could see
            # nothing about. Read off the replay this round just made.
            try:
                rec['polish'] = polish_read(g, r, replay_as_ex(g, rdir))
            except Exception as exc:
                rec['polish_error'] = f'{type(exc).__name__}: {str(exc)[:160]}'
            # NOT IN THE `inner` ARM. That arm's whole claim is that no global
            # reader decides anything; asking one and recording its answer would
            # put its reason into `outer_directive` and from there into prompts
            # downstream of the plan.
            said = (None if DISPATCH == 'inner' else
                    ask_monitor(g, r, rec.get('comparison') or {}, focus))
            rec['monitor_said'] = said
            if said and not said.get('error'):
                log(g.dir, f"  monitor: progress={said.get('progress')} "
                           f"coverage={said.get('evidence_coverage')} "
                           f"headroom={said.get('overall_headroom')} "
                           f"-> {said.get('action')}/{said.get('next_direction')}")
                log(g.dir, f"           {str(said.get('reason'))[:170]}")
    except Exception as exc:
        rec['checkpoint_error'] = f'{type(exc).__name__}: {str(exc)[:200]}'

    rec['wall_s'] = round(time.time() - rec['started'], 1)
    return rec


def action_frames(g: Game, r: int, cap: int = 24) -> list[tuple]:
    """Frames either side of an input, for judging whether acting lands.

    PRESENTATION AND FEEDBACK NEED DIFFERENT PICTURES. How a game looks can be
    read off one frame; whether pressing a key did anything cannot be read off
    any single frame at all. `art_frames` takes the opening, the middle and the
    end of each demo, which is right for the first question and useless for the
    second.

    The pairing already exists: a replay records, per input, the frame before
    it and the two after. Nothing is replayed here -- the last checkpoint's
    recording is read back off disk.

    TWO AFTER-FRAMES, NOT ONE. With one, an input whose response renders a
    frame late reads as an input that did nothing, and a reader confirmed
    exactly that as a defect when the comparator was being built.
    """
    src = at = None
    for k in range(r - 1, 0, -1):
        d = g.dir / f'round_{k:02d}' / 'post_replay'
        if d.is_dir() and any(d.glob('frames_*')):
            src, at = d, k
            break
    if src is None:
        return []
    from rsigame.monitor.replay_pair import _rec_from_recorded
    ids = sorted(p.name[len('frames_'):] for p in src.glob('frames_*'))
    per = max(2, cap // max(1, len(ids)))
    out = []
    for did in ids:
        rec = _rec_from_recorded(src, did,
                                 g.work / 'demo_outputs' / f'{did}.json')
        evs = [e for e in ((rec or {}).get('events') or []) if e.get('before')]
        if not evs:
            continue
        # Spread across the demo rather than taking the first few: the opening
        # inputs of a run are usually menu presses, and what is being judged is
        # whether the game's own verbs answer.
        step = max(1, len(evs) // per)
        for e in evs[::step][:per]:
            out.append((f"demo {did} -- before: {e['what']}", e['before']))
            for j, f in enumerate((e.get('after') or [])[:2], start=1):
                out.append((f"demo {did} -- after {j}: {e['what']}", f))
    return out[:cap * 3]


def art_close(g: Game) -> dict:
    """One more look, at the last art round's own output.

    EVERY ART ROUND'S OPENING READ IS THE PREVIOUS ROUND'S VERIFICATION -- it
    looks at the frames the round before produced. The last one has no
    successor, so its own work is the one change in the run that nothing ever
    looks at. Round fifteen edited eight files on one game and the run ended
    there.

    Two triggers, one mechanical and one observed. A project that does not
    compile needs no model to say so, and a project that compiles can still
    have had the line that drew the world deleted -- which is what the reader
    catches. Where that happened mid-run it cost 70 seconds to put right: the
    round after saw a flat grey screen, led its brief with it, and fixed it.

    ONE ATTEMPT, then the result is recorded whatever it is. A second would be
    an unbounded loop, and an honest "it ended broken" is worth more than a
    tree that was struggled over and left in an unknown state.
    """
    from rsigame.controller import art_list as AL
    from rsigame.controller.repair_prompt import render_art, parse_art_report

    f = g.dir / 'art_close.json'
    if f.is_file():
        return json.loads(f.read_text())
    cdir = g.dir / 'art_close'
    cdir.mkdir(exist_ok=True)
    # The last art round's question, not the run's env var: under monitor
    # dispatch they need not be the same.
    focus = next((x.get('focus') for x in reversed(g.rounds)
                  if x.get('kind') == 'art' and x.get('focus')), ART_FOCUS)
    rec = {'kind': 'art_close', 'started': time.time(), 'focus': focus}
    log(g.dir, f'\n=== {g.game} final reading')

    def look(tag: str) -> dict:
        b = build_tree(g.work)
        out = {'build_ok': bool(b.get('ok')),
               'build_tail': (b.get('tail') or '')[-800:] if not b.get('ok') else ''}
        try:
            asked = AL.history(g.work, g.rounds[-HISTORY_ROUNDS:])
        except Exception:
            asked = ''
        fr = (action_frames(g, g.total + 1) if focus == 'feedback'
              else art_frames(g, g.total + 1))
        a = AL.read(g.game, g.work, fr, facts=visual_facts(g.work),
                    asked=asked, focus=focus, model=PJ.RSIGAME_MODELS_LOOP_MODEL)
        out['art_list'] = a
        out['playable'] = a.get('playable')
        log(g.dir, f"  {tag}: build={out['build_ok']} · still playable={out['playable']}"
                   + (f" · {str(a.get('playable_why'))[:140]}"
                      if a.get('playable') != 'playable' else ''))
        return out

    rec['before'] = look('final reading')
    # The last art round is the one nothing else verifies. This reading is its
    # verification, by the same rule as every other art round's.
    try:
        backfill_verdict(g, g.total + 1,
                         (rec['before'].get('art_list') or {}))
        g.save()
    except Exception as exc:
        log(g.dir, f'  final backfill failed: {type(exc).__name__}: {str(exc)[:120]}')
    broken = (not rec['before']['build_ok']
              or rec['before']['playable'] == 'broken')
    rec['repaired'] = False
    if not broken:
        rec['visual_final'] = visual_facts(g.work)
        rec['wall_s'] = round(time.time() - rec['started'], 1)
        f.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        return rec

    # -- one repair, on that and nothing else ---------------------------
    why = (rec['before']['build_tail'] if not rec['before']['build_ok']
           else (rec['before']['art_list'].get('playable_why') or ''))
    brief = ('====================\nTHIS BUILD DOES NOT WORK\n'
             '====================\n\n'
             + ('The build check says:\n\n' + why
                if not rec['before']['build_ok'] else
                'A reader looked at the frames of this build and says: ' + why)
             + '\n\nThis is the last round of the run. Put that right and '
               'nothing else -- no new art, no layout work. The change that '
               'broke it is in what the previous round edited.')
    (cdir / 'brief.txt').write_text(brief)
    prompt = render_art(brief, r=g.total, total=g.total, n_art=ART_ROUNDS,
                        build_cmd=PJ.BUILD_CMD.get(engine(g.work), ''),
                        py=sys.executable, repo=str(PJ.REPO),
                        rounds=g.history_for_prompt(), engine=engine(g.work))
    (cdir / 'repair_prompt.txt').write_text(prompt)
    t0 = time.time()
    try:
        from rsigame.agent.generator.repair import _run_repair_agent
        led = asset_budget(cdir)
        rr = _run_repair_agent(g.work, prompt, max_attempts=REPAIR_ATTEMPTS,
                               model=PJ.RSIGAME_REPAIR_MODEL, max_tools=ART_TOOLS,
                               tag='close')
    except Exception as exc:
        led = None
        rr = {'success': False, 'error': f'{type(exc).__name__}: {str(exc)[:200]}'}
    rec['asset_budget'] = asset_budget_record(led, rr)
    rec['repaired'] = True
    rec['repair'] = {'success': rr.get('success'), 'error': rr.get('error'),
                     'files_edited': rr.get('files_edited') or [],
                     'n_tools': rr.get('n_tools'),
                     'wall_s': round(time.time() - t0, 1)}
    log(g.dir, f"  final repair: {rr.get('success')} "
               f"{len(rr.get('files_edited') or [])} files "
               f"{rec['repair']['wall_s']}s")
    fix_fake_pngs(g.work)
    PJ.godot_import(g.work)
    try:
        PRR.replay_demos(g.work, cdir / 'post_replay')
        PRR.prune(cdir / 'post_replay')
    except Exception as exc:
        rec['replay_error'] = f'{type(exc).__name__}: {str(exc)[:160]}'
    rec['after'] = look('reading after the fix')
    rec['visual_final'] = visual_facts(g.work)
    rec['wall_s'] = round(time.time() - rec['started'], 1)
    f.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    return rec




def carried_breakage(g, r: int) -> str:
    """What the previous round left broken, in its own words; '' when it built.

    A normal round used to learn this only as one line of history ("the build
    did not compile"), with no error text and no instruction to put it right
    first -- and a round that broke a script loaded behind the title screen was
    not known to be broken at all.

    Measured twice, once per engine, and both numbers are the reason this exists:
    on the Godot outer-guidance pilot one branch stayed broken for four normal
    rounds and was repaired only by the art round, whose prompt already opened
    with the error; on the Phaser SR20 runs (2026-09-15) 77 of 600 GPT rounds
    and 65 of 597 Qwen rounds ended on a build that did not compile, in streaks
    of up to nine rounds, and every such checkpoint scored 0.

    Every round now opens the same way.
    """
    prev = next((x for x in reversed(g.rounds) if x.get('round') == r - 1), None)
    if not prev or prev.get('build_ok') is not False:
        return ''
    tail = str(prev.get('build_tail') or '').strip()
    return tail or 'the previous round left the build broken (no error text was recorded)'


BROKEN_BLOCK_GODOT = ('THIS PROJECT IS BROKEN RIGHT NOW. The previous round left it in a state '
                'that does not load or whose scripts do not parse:\n\n{tail}\n\n'
                'Fix that FIRST, and only that until the project loads and every script '
                'parses. The evidence below was recorded on this broken build, so it '
                'mostly shows the breakage.\n\n')


def keep_replay(g, r: int) -> bool:
    """RSIGAME_CONTROLLER_KEEP_REPLAY_EVERY=N keeps round r's replay mp4s and logs when r % N == 0
    or r is the last round. An outer-guidance supervisor then reads the Monitor's
    checkpoint recordings from them instead of replaying the same build again."""
    n = int(os.environ.get('RSIGAME_CONTROLLER_KEEP_REPLAY_EVERY') or 0)
    return bool(n) and (r % n == 0 or r == g.total)

def runtime_parse_errors(g, rec: dict, replay_dir: Path) -> None:
    """A build that loads but throws while it runs is a failed build.

    `PJ.build` loads the project headless and stops at the main scene, so a
    script first loaded behind the title screen is never parsed there -- and
    Godot's own checks cannot parse such scripts in isolation either, because
    autoload singletons are not registered outside the running game. The demo
    replay just run IS the running game: its logs carry every error the player
    would hit. Measured on the outer-guidance pilot (2026-09-15): 10 of 45
    checkpoints built "ok" while their recordings logged parse errors, and one
    branch edited a game whose play scene did not parse for nine rounds.

    `SCRIPT ERROR` is here for the same reason and it is the worse case: a
    RUNTIME type error parses fine and builds fine. battle-critter-clash R03
    scored 95.1 on the official rubric and was adopted as champion while a
    fresh player reaching the battle got 1751 script errors and an empty
    screen -- `make_critter` was being handed a Dictionary where it wanted a
    String. Neither the build check nor the demo replay's score noticed;
    only playing it did. The count and the distinct messages go into
    `build_tail`, which the next round's packet opens with.

    Read before `prune` deletes the logs.
    """
    RUNTIME = ('Parse Error', 'SCRIPT ERROR')
    lines, n = [], 0
    for f in sorted(Path(replay_dir).glob('log_*/**/*.log')):
        for ln in f.read_text(errors='ignore').splitlines():
            if any(k in ln for k in RUNTIME) or ('Failed to load script' in ln and 'Parse error' in ln):
                n += 1
                lines.append(ln.strip()[:200])
    if not lines:
        return
    uniq = list(dict.fromkeys(lines))
    rec['runtime_parse_errors'] = uniq[:8]
    rec['runtime_error_count'] = n
    if rec.get('build_ok'):
        rec['build_ok'] = False
        rec['build_tail'] = (f'the project loads and the build check passes, but playing it throws: '
                             f'{n} runtime script errors were logged during the demo replay. '
                             f'The distinct messages were:\n' + '\n'.join(uniq[:6]))
        log(g.dir, f"  runtime script errors in the replay ({n} ), counted as a failed build: {uniq[0][:160]}")
BROKEN_BLOCK_WEB = ('THIS GAME IS BROKEN RIGHT NOW. The previous round left it in a state '
                'that does not compile, or that crashes when the page starts or on the '
                'first input:\n\n{tail}\n\n'
                'Fix that FIRST, and only that until `npm run build` succeeds and the '
                'game starts without an uncaught error. The evidence below was recorded '
                'on this broken build, so it mostly shows the breakage.\n\n')


def plan_first(g: Game, r: int, rec: dict) -> dict:
    """The planner speaks before the round branches -- `inner` arm only.

    `plan` is the FIRST model call inside `one_round`; everything before it is
    local (`visual_facts`, `observe_round`, `build`). So hoisting it above the
    dispatch costs nothing: the round that follows reuses this plan instead of
    asking again.
    """
    rdir = g.dir / f'round_{r:02d}'
    rdir.mkdir(parents=True, exist_ok=True)
    obs = {}
    prev = [x for x in g.rounds if x.get('demos_replayed_at')]
    if prev:
        try:
            obs = observe_round(g.dir, prev[-1]['demos_replayed_at'],
                                cache=rdir / 'demo_outcomes.json')
        except Exception:
            obs = {}
    inp = build(g.state_run(), r, None, budget_total=g.total,
                descriptions=g.demo_desc)
    for d in inp['demos']:
        if d['replayed_on_this_build'] and not d.get('outcome'):
            d['outcome'] = obs.get(d['demo_id'])
    g.ledger.sync(g.state_run().state, r)
    return plan(inp, g.ledger.open())


def one_round(g: Game, r: int, rec: dict) -> dict:
    """`rec` is filled in as the round goes, so a crash keeps what it cost."""
    rdir = g.dir / f'round_{r:02d}'
    rdir.mkdir(exist_ok=True)
    rec.update(round=r, started=time.time(), project_hash=project_hash(g.work))
    # Counted off the source, before and after the repair, so a round says what
    # it changed about the art without anyone having to be believed about it.
    # `images_used` is the one that matters: art that is generated and never
    # loaded costs money and changes nothing on screen -- one game went from 30
    # images to 150 while the code went from loading 3 to loading 5.
    rec['visual_before'] = visual_facts(g.work)

    # Decided by the caller now, so that the dispatch and the round agree on
    # it. Recomputed only when something calls `one_round` directly.
    direction = rec.get('direction') or choose_direction(g, r)[0]
    rec['direction'] = direction
    log(g.dir, f'\n=== {g.game} round {r} · {direction or "self-chosen"}')

    # -- plan ----------------------------------------------------------
    obs = {}
    prev = [x for x in g.rounds if x.get('demos_replayed_at')]
    if prev:
        try:
            obs = observe_round(g.dir, prev[-1]['demos_replayed_at'],
                                cache=rdir / 'demo_outcomes.json')
        except Exception:
            obs = {}
    said = next((x.get('monitor_said') for x in reversed(g.rounds)
                 if x.get('monitor_said')), None) or {}
    inp = build(g.state_run(), r, direction, budget_total=g.total,
                outer_reason=said.get('reason', ''),
                descriptions=g.demo_desc)
    for d in inp['demos']:
        if d['replayed_on_this_build'] and not d.get('outcome'):
            d['outcome'] = obs.get(d['demo_id'])
    g.ledger.sync(g.state_run().state, r)
    # Already asked, if the dispatch had to know the answer before it could
    # choose the round's kind. Asking twice would cost a call and could return
    # a different question than the one the dispatch acted on.
    p = rec.get('plan') or plan(inp, g.ledger.open())
    # REUSE NEEDS SOMETHING TO REUSE. A plan that says the evidence is
    # sufficient, in a run that has not recorded a single replay on this build,
    # is citing something else as evidence -- measured on the outer-guidance
    # pilot: the planner cited the director's brief ("holding LEFT moved one
    # cell") as sufficient, skipped observing, grounding found nothing
    # confirmed, and the round ended in 13 s with no repair, round after round.
    # Observe first: replay the demo the plan named, or the first one.
    if (p.get('action') == 'reuse' and not prev
            and not any(d.get('replayed_on_this_build') for d in inp['demos'])):
        demo = p.get('demo_id') or next((d['demo_id'] for d in inp['demos'] if d.get('num_inputs')),
                                        inp['demos'][0]['demo_id'] if inp['demos'] else '')
        if demo:
            rec['plan_overridden'] = {'from': {k: p.get(k) for k in ('action', 'mode', 'demo_id')},
                                      'why': 'reuse with no evidence recorded on this build'}
            p = dict(p, action='explore', mode='replay_existing_demo', demo_id=demo)
            log(g.dir, f"  plan rewritten: reuse but this run has no replay yet -> replay {demo}")
    rec['plan'] = p
    rec['question'] = p.get('question')
    log(g.dir, f"  plan: {p.get('action')}/{p.get('mode')} "
               f"demo={p.get('demo_id') or '-'}  {str(p.get('question'))[:110]}")
    rec['target'] = g.ledger.note(p.get('target_ref'), p.get('question', ''),
                                  p.get('related_state_items') or [], r)

    # -- probe ---------------------------------------------------------
    pdir = rdir / 'probe'
    ex = execute.run({'mode': p.get('mode'), 'demo_id': p.get('demo_id'),
                      'extension': p.get('extension')},
                     work=g.work, out_dir=pdir, question=p.get('question', ''),
                     items=p.get('related_state_items') or [], port=g.port)
    rec['probe'] = ex
    log(g.dir, f"  probe: ran={ex.get('ran')} {ex.get('error') or ''}")

    # The script that answers this round's question, frozen so the same
    # question can be asked again after the repair.
    script = None
    if (pdir / 'trace' / 'exploration.json').is_file():
        info = freeze(pdir / 'trace' / 'exploration.json', rdir / 'probe_script.json',
                      question=p.get('question', ''))
        script, rec['frozen'] = rdir / 'probe_script.json', info
    elif p.get('mode') == 'replay_existing_demo' and p.get('demo_id'):
        script = g.work / 'demo_outputs' / f"{p['demo_id']}.json"

    frames = (from_exploration(pdir / 'trace') if (pdir / 'trace').is_dir()
              else from_replay(ex))
    # Whatever else was recorded on this same build. Labelled as incidental so
    # the reader knows which frames answer this round's question and which it
    # is merely being shown.
    for pid in list(obs)[:3]:
        a = from_round(g.dir, prev[-1]['demos_replayed_at'], pid) if prev else None
        if a:
            frames += [(f'{pid} (incidental) {l}', f)
                       for l, f in from_replay(a, limit=4)]
    frames = frames[:26]
    ev = ex.get('observation') or ''
    if not ev and ex.get('events'):
        # A fixed demo writes no account of itself, so one is made here from
        # the frames it just produced. Without it the grounding stage reads
        # pictures with nothing telling it what the demo was trying to do.
        try:
            ev = observe_one(ex, p.get('demo_id') or 'probe',
                             g.demo_desc.get(p.get('demo_id') or ''),
                             cache=rdir / 'probe_outcome.json')
            ex['observation'] = ev
        except Exception as exc:
            ev = ''
            rec['observe_error'] = f'{type(exc).__name__}: {str(exc)[:160]}'
    if not ev:
        ev = ('No new observation was written this round. What is known about '
              'this build comes from the previous one:\n'
              + '\n'.join(f"{k}: {v}" for k, v in obs.items()))

    # -- what the tree contains, and what the round just showed -----------
    #
    # Both are about the requirement list rather than about this round's
    # question, and both were in the old loop. The fact sheet is free; the
    # reading is one call and is what keeps the board from sitting at
    # `unverified` for twelve rounds while the ranker finds nothing eligible.
    sview = ''
    try:
        from rsigame.view.static_view import build_static_view
        seen = []
        trace = pdir / 'trace' / 'exploration.json'
        if trace.is_file():
            seen = sorted({st.get('args', {}).get('scenario')
                           for st in (json.loads(trace.read_text()).get('steps') or [])
                           if st.get('action') == 'boot_scenario'} - {None})
        past = sorted(set(seen) | {x for h in g.rounds
                                   for x in (h.get('scenarios_visited') or [])})
        rec['scenarios_visited'] = seen
        from rsigame.view.static_view import web_declared_scenarios
        declared = (web_declared_scenarios(g.work) if engine(g.work) == 'web'
                    else PJ.declared_scenarios(g.work))
        sview = build_static_view(g.work, declared=declared,
                                  visited_now=seen, visited_history=past)
        (rdir / 'static_view.txt').write_text(sview)
    except Exception as exc:
        rec['static_view_error'] = f'{type(exc).__name__}: {str(exc)[:120]}'

    try:
        from rsigame.controller import checklist_read as CR
        cexp = CR.as_exploration(ex, pdir, ev)
        if cexp is not None:
            prev_edit = bool(g.rounds and
                             ((g.rounds[-1].get('repair') or {}).get('files_edited')))
            cr = CR.read(g.game, exp=cexp, exp_path=rdir / 'checklist_exp.json',
                         state_path=g.dir / 'checklist_state.json',
                         static_view=sview, round_n=r, edited=prev_edit)
            rec['checklist_read'] = cr
            if cr.get('error'):
                log(g.dir, f"  checklist reading failed: {cr['error'][:120]}")
            else:
                t = cr['tally']
                log(g.dir, f"  checklist: {len(cr['moves'])} changes · "
                           f"satisfied {t['satisfied']} / gaps {t['confirmed_gap']} "
                           f"/ unverified {t['unverified']}"
                           + (f" · gated {len(cr['gated'])}"
                              if cr['gated'] else ''))
    except Exception as exc:
        rec['checklist_read'] = {'error': f'{type(exc).__name__}: {str(exc)[:160]}'}

    # -- ground and rank ------------------------------------------------
    gr = RANK.ground(ev, inp['state']['all_items'], direction,
                     p.get('question', ''), frames=frames, static_view=sview)
    rec['issues'] = gr
    # Damage from the last repair is already grounded -- it was seen in frames
    # of this build -- so it enters as a confirmed gap rather than being put to
    # the reader again. Anything the ranker leaves alone stays open and comes
    # back next round; that is the point of carrying it.
    for k, dmg in enumerate(carried_damage(g, r), start=1):
        gr['issues'].append({
            'issue_id': f'R{k}', 'issue': dmg['what'], 'status': 'confirmed_gap',
            'direction': direction, 'importance': dmg.get('severity') or 'medium',
            'evidence_refs': [dmg.get('evidence') or ''],
            'note': f"broken by the round-{dmg['round']} repair", 'closes': []})

    sel = RANK.rank(gr['issues'], direction, p.get('question', ''),
                    '\n'.join(f"  {t['id']} ({t['attempts']}x) {t['statement']}"
                              for t in g.ledger.open()), g.total - r)
    rec['selection'] = sel
    by = {i['issue_id']: i for i in gr['issues']}
    rec['primary_issue'] = (by.get(sel.get('primary_target')) or {}).get('issue')
    log(g.dir, f"  issues: {len(gr['issues'])} "
               f"({sum(1 for i in gr['issues'] if i['status']=='confirmed_gap')} confirmed) "
               f"-> primary={sel.get('primary_target')} "
               f"concurrent={sel.get('concurrent_minor_fixes')}")

    # -- repair ---------------------------------------------------------
    reader = polish_read(g, r, ex, probe_dir=pdir)
    rec['polish'] = reader
    art = None
    if reader and not reader.get('error'):
        # The reader says what is weak; this decides what to ask for about it,
        # having seen the counted facts and every earlier ask verbatim. Without
        # it four rounds of one game were handed the same sentence.
        from rsigame.evidence.polish_goal import ask as polish_ask
        facts = rec.get('visual_before') or {}
        past = [(x['round'], (x.get('art') or {}).get('goal'))
                for x in g.rounds if (x.get('art') or {}).get('goal')]
        art = polish_ask(reader, facts, past)
        art['facts'] = {k: facts.get(k) for k in
                        ('images_present', 'images_used', 'authored_share')}
        rec['art'] = art
        if not art.get('error'):
            log(g.dir, f"  polish: {reader.get('headroom')} · "
                       f"images used {facts.get('images_used')}/{facts.get('images_present')} · "
                       f"{'repeated · ' if art.get('repeats_earlier') else ''}"
                       f"{str(art.get('goal'))[:120]}")
    pkt = RANK.packet(sel, gr['issues'], art=art)
    broken = carried_breakage(g, r)
    if broken:
        rec['build_before'] = False
        rec['build_before_tail'] = broken[-800:]
        # each line wrote its own wording for its own failure shape(Godot:does not load / scripts do not parse;
        # Phaser:npm run build fails / the page crashes on start). After the merge the tree decides which one applies.
        block = BROKEN_BLOCK_WEB if engine(g.work) == 'web' else BROKEN_BLOCK_GODOT
        pkt = block.format(tail=broken[-800:]) + pkt
        log(g.dir, f"  broken state left by the previous round, fixed first: {broken[-200:]}")
    (rdir / 'packet.txt').write_text(pkt)
    # Where things are, carried between rounds. The agent annotates functions
    # as it reads them and the notes persist, so the round after does not have
    # to find the same file again.
    pmap = None
    try:
        import rsigame.view.project_map as PM
        pmap = PM.merge(PM.scan(g.work), PM.load_notes(g.dir))
        rendered = PM.render(pmap)
    except Exception as exc:
        rendered = ''
        rec['project_map_error'] = f'{type(exc).__name__}: {str(exc)[:120]}'
    prompt = render_repair(pkt, r=r, total=g.total,
                           build_cmd=PJ.BUILD_CMD.get(engine(g.work), ''),
                           py=sys.executable, repo=str(PJ.REPO),
                           rounds=g.history_for_prompt(), engine=engine(g.work),
                           project_map=rendered)
    (rdir / 'repair_prompt.txt').write_text(prompt)

    if not sel.get('primary_target') and not art and not broken:
        rec['repair'] = {'skipped': 'nothing to repair and no art work offered'}
    else:
        t0 = time.time()
        try:
            from rsigame.agent.generator.repair import _run_repair_agent
            led = asset_budget(rdir)
            rr = _run_repair_agent(g.work, prompt,
                                   max_attempts=REPAIR_ATTEMPTS,
                                   model=PJ.RSIGAME_REPAIR_MODEL,
                                   max_tools=PJ.REPAIR_TOOLS, tag=f'r{r:02d}')
        except Exception as exc:
            led = None
            rr = {'success': False, 'error': f'{type(exc).__name__}: {str(exc)[:200]}'}
        rec['asset_budget'] = asset_budget_record(led, rr)
        # The agent's transcript holds plain strings in some places and
        # message objects in others; the closing JSON can be in either.
        tail = ''
        for m in reversed(rr.get('messages') or []):
            c = m if isinstance(m, str) else (m.get('content') if isinstance(m, dict) else '')
            if isinstance(c, list):
                c = ' '.join(x.get('text', '') for x in c if isinstance(x, dict))
            if isinstance(c, str) and c.strip():
                tail = c
                break
        # How long it looked before it changed anything. This is the number
        # that says whether a grounded evidence packet actually saves the
        # agent its diagnosis: under the old framework the median was 14 calls
        # out of 26, and the rounds that never got there are the ones that
        # edited nothing.
        tools = rr.get('tools') or []
        first_edit = next((i for i, t in enumerate(tools, 1)
                           if isinstance(t, dict)
                           and t.get('name') in ('edit', 'write_file')), None)
        rec['repair'] = {'success': rr.get('success'), 'error': rr.get('error'),
                         'files_edited': rr.get('files_edited') or [],
                         'stopped_by': rr.get('stopped_by'),
                         'n_tools': len(tools),
                         'calls_to_first_edit': first_edit,
                         'replayed_before_edit': sum(
                             1 for t in tools[:(first_edit or len(tools))]
                             if isinstance(t, dict)
                             and 'play_cli' in str(t.get('target') or '')),
                         'budget_policy': 'no-rediagnosis-v1',
                         'wall_s': round(time.time() - t0, 1)}
        rec['repair_report'] = parse_report(tail)
        if pmap is not None:
            try:
                import rsigame.view.project_map as PM
                pmap, n_new = PM.take_notes(pmap, PJ._map_notes_from(rr))
                PM.save(pmap, g.dir)
                rec['map_notes_added'] = n_new
            except Exception as exc:
                rec['project_map_error'] = f'{type(exc).__name__}: {str(exc)[:120]}'
        if g.proxy is not None:
            try:
                rec['tokens'] = PJ.repair_cost(f'r{r:02d}', g.work, proxy=g.proxy)
            except Exception:
                pass
        log(g.dir, f"  repair: {rr.get('success')} "
                   f"{len(rr.get('files_edited') or [])} files "
                   f"{rec['repair']['wall_s']}s "
                   f"first_edit_at={first_edit or '-'}/{len(tools)} "
                   f"replays_first={rec['repair']['replayed_before_edit']} "
                   f"report={'yes' if rec['repair_report']['ok'] else 'MISSING'}")

    # KEEP WHAT THE IMPORT SAID. `still_missing` is the list of resources that
    # are on disk and that `load()` cannot open -- the exact failure mode
    # behind "the round generated art and the screen looks identical". Thrown
    # away, that round is indistinguishable from one that generated nothing.
    rec['transcoded'] = fix_fake_pngs(g.work)
    if rec['transcoded']:
        log(g.dir, f"  transcoded {len(rec['transcoded'])} fake pngs: "
                   f"{rec['transcoded'][:3]}")
    rec['import'] = PJ.godot_import(g.work)
    if rec['import'].get('still_missing'):
        log(g.dir, f"  still will not open after import: {rec['import']['still_missing'][:4]}")
    b = build_tree(g.work)
    rec['build_ok'] = bool(b.get('ok'))
    if not rec['build_ok']:
        rec['build_tail'] = (b.get('tail') or '')[-800:]
        log(g.dir, f"  build failed: {rec['build_tail'][-300:]}")
    rec['visual_after'] = visual_facts(g.work)
    a0, a1 = rec.get('visual_before') or {}, rec['visual_after']
    moved = [f"{k} {a0.get(k)}->{a1.get(k)}" for k in
             ('images_present', 'images_used', 'texture_draws', 'primitive_draws')
             if a0.get(k) != a1.get(k)]
    log(g.dir, f"  build: {rec['build_ok']}"
               + (f"  art: {' · '.join(moved)}" if moved else ''))

    # -- verify ----------------------------------------------------------
    edited = (rec.get('repair') or {}).get('files_edited') or []
    if script and Path(script).is_file() and sel.get('primary_target') and edited:
        vdir = rdir / 'verify'
        try:
            after = PRR.replay_one(g.work, Path(script), vdir)
        except Exception as exc:
            after = {'replay_error': f'{type(exc).__name__}: {str(exc)[:160]}'}
        vframes = from_replay(after)
        asked = [i for i in gr['issues'] if i['issue_id'] in
                 ([sel.get('primary_target')] + list(sel.get('concurrent_minor_fixes') or []))]
        # THE ART ASK IS ASKED FOR, SO IT IS CHECKED. It was in the packet as
        # co-equal work and then left out of the only reading that says whether
        # anything happened -- so no verdict, nothing in the history, and the
        # next round could not tell an ask that was done from one that was
        # ignored. Phrased as the absence, because `observe` answers whether
        # the thing it is given is still there.
        if art and art.get('goal') and not art.get('error'):
            asked = asked + [{'issue_id': 'ART',
                              'issue': 'Not done yet: ' + art['goal'][:400],
                              'evidence_refs': [
                                  f"the polish reader, round {r}"]}]
        o = VERIFY.observe(asked, inp['state']['all_items'], vframes)
        rec['verification'] = VERIFY.verdicts(
            o['observations'], (rec.get('repair_report') or {}).get('report'), sel)
        rec['verification']['observe_problems'] = o.get('schema_problems')
        rec['regressions'] = look_for_damage(g, ex, after, r)
        if rec['regressions']:
            log(g.dir, f"  regressions: {len(rec['regressions'])}  -- "
                       + ' / '.join(x['what'][:60] for x in rec['regressions'][:2]))
        log(g.dir, f"  verify: primary={rec['verification'].get('primary_verdict')} "
                   f"{rec['verification'].get('counts')}")
    elif sel.get('primary_target') and not edited:
        # Verifying an unedited tree replays one build against itself, and the
        # differences are the recorder's. The first run called two issues
        # fixed that way, in a round that changed no files at all.
        rec['verification'] = {'skipped': 'the repair edited nothing, so there '
                                          'is no new build to look at',
                               'primary_verdict': 'not_attempted'}
    else:
        rec['verification'] = {'skipped': 'no probe script, or nothing selected'}

    # -- the checkpoint the global monitor reads --------------------------
    if r % MONITOR_EVERY == 0 or r == g.total:
        try:
            PRR.replay_demos(g.work, rdir / 'post_replay')
            runtime_parse_errors(g, rec, rdir / 'post_replay')
            if not keep_replay(g, r):
                PRR.prune(rdir / 'post_replay')
            rec['demos_replayed_at'] = r
            rec['comparison'] = compare_checkpoints(g, r)
            move_statuses(g, rec.get('comparison') or {}, r)
            crit = ((rec.get('comparison') or {}).get('regressions') or {}).get('critical')
            log(g.dir, f'  checkpoint: demos replayed · critical regression={crit}')
            # NOT IN THE `inner` ARM. That arm's whole claim is that no global
            # reader decides anything; asking one and recording its answer would
            # put its reason into `outer_directive` and from there into prompts
            # downstream of the plan.
            said = (None if DISPATCH == 'inner' else
                    ask_monitor(g, r, rec.get('comparison') or {}, direction))
            rec['monitor_said'] = said
            if said and not said.get('error'):
                log(g.dir, f"  monitor: progress={said.get('progress')} "
                           f"coverage={said.get('evidence_coverage')} "
                           f"headroom={said.get('overall_headroom')} "
                           f"-> {said.get('action')}/{said.get('next_direction')}")
                log(g.dir, f"           {str(said.get('reason'))[:170]}")
        except Exception as exc:
            rec['checkpoint_error'] = f'{type(exc).__name__}: {str(exc)[:200]}'

    rec['wall_s'] = round(time.time() - rec['started'], 1)
    return rec


def main():
    from . import config
    config.ensure()
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('game')
    ap.add_argument('--rounds', type=int, default=15)
    ap.add_argument('--root', default='newloop_cs10')
    ap.add_argument('--corpus', default='godot/corpus_casestudy')
    ap.add_argument('--port', type=int, default=7980)
    a = ap.parse_args()

    # ABSOLUTE, ALWAYS. The repair agent runs as a subprocess with its own cwd
    # (the agent repo), and it is handed the prompt file by the path this
    # process built. A relative --root makes that path relative to the LOOP's
    # cwd, so the agent opens it from the wrong directory and dies before it
    # starts -- `ENOENT ... _repairs_work/13-r13/prompt.txt`, 1.3 seconds, no
    # tool calls, and a round that looks like a silent model failure. Cost
    # three games one art round each to find, because the default was relative
    # and every pool script so far happened to pass an absolute path.
    g = Game(a.game, Path(a.root).resolve(), Path(a.corpus).resolve(), a.port)
    g.total = a.rounds
    start = len(g.rounds) + 1
    # THE MONITOR, LIVE. Off unless [monitor] live = true. When it is on it
    # judges each checkpoint against the held champion as the run goes and ends
    # the run once the saturation stop fires -- see monitor/live.py for what that costs.
    from .monitor import live as LIVE
    mon = None
    if LIVE.enabled():
        try:
            mon = LIVE.LiveMonitor(g.game, Path(a.root).resolve())
            log(g.dir, f'monitor: live, K={mon.k}, every {mon.every()} rounds')
        except Exception as exc:
            # A monitor that cannot start must not take the run with it: the run
            # is still valid, it just gets no early stop.
            log(g.dir, f'monitor: live requested but did not start '
                       f'({type(exc).__name__}: {str(exc)[:120]}); continuing without an early stop')

    for r in range(start, a.rounds + 1):
        # STOP FILE (outer-guidance branches): a supervisor that watches the
        # Monitor writes it when the stage has converged; the loop ends at the
        # next round boundary instead of being killed mid-round.
        stop = os.environ.get('RSIGAME_CONTROLLER_STOP_FILE')
        if stop and Path(stop).exists():
            log(g.dir, f'\nstop file present before round {r}: {Path(stop).read_text()[:200]}')
            break
        rec: dict = {'round': r}
        g.rounds.append(rec)
        # THE DISPATCH. Chosen out here rather than inside the round, because
        # which round shape runs is now a consequence of the direction and only
        # one of the two shapes ever asked for it.
        direction, _ = choose_direction(g, r)
        rec['direction'] = direction
        if DISPATCH == 'inner':
            # The planner speaks before the branch, and its answer IS the
            # dispatch. `feedback`/`visual` become the round's direction so
            # that `art_round` reads the same frames it would under the
            # monitor arm -- the arms differ in who decides, never in what a
            # decision makes happen.
            try:
                pre = plan_first(g, r, rec)
            except Exception as exc:
                pre = {'error': f'{type(exc).__name__}: {str(exc)[:160]}'}
            rec['plan'] = pre
            if pre.get('round_kind') == 'art':
                kind = 'art'
                direction = ('feedback' if pre.get('art_focus') == 'feedback'
                             else 'presentation')
            else:
                kind = 'normal'
            rec['direction'] = direction
            log(g.dir, f"  self-chosen: {kind}"
                       + (f"/{direction}" if kind == 'art' else '')
                       + (f" · {str(pre.get('question'))[:80]}"
                          if kind != 'art' else ''))
        else:
            kind = round_kind(r, a.rounds, direction)
        # AN ART ROUND READS THE PREVIOUS ROUND'S REPLAY AND HAS NO OTHER
        # EVIDENCE. Before any replay exists there is nothing for it to look
        # at, so the full chain runs instead -- which is also the chain that
        # produces the first replay.
        # NO REPLAY YET: RECORD ONE, RATHER THAN DOWNGRADE THE ROUND.
        #
        # Downgrading used to run the normal chain with the planner's ART plan
        # already attached (action reuse, no question), so the chain selected
        # nothing and the round did nothing: measured 3 of 6 such rounds in the
        # self arm skipped in ~10 s with "nothing to repair". From round 2 on
        # there is a previous build to replay; replaying it is a few minutes
        # and gives the art round the frames it reads. Round 1 has no previous
        # round directory to hold that replay, so it still runs the full chain.
        if kind == 'art' and not any(
                (g.dir / f'round_{k:02d}' / 'post_replay').is_dir()
                for k in range(1, r)):
            prev_rec = next((x for x in g.rounds if x.get('round') == r - 1), None)
            if r >= 2 and prev_rec is not None:
                prev_dir = g.dir / f'round_{r - 1:02d}' / 'post_replay'
                try:
                    PRR.replay_demos(g.work, prev_dir)
                    PRR.prune(prev_dir)
                    # on the round the replay belongs to; this round's own
                    # record is already appended and is not that round
                    prev_rec['demos_replayed_at'] = r - 1
                    log(g.dir, f'  r{r} art round with no replay yet: replaying the current build into round_{r - 1:02d}')
                except Exception as exc:
                    log(g.dir, f'  r{r} catch-up replay failed ({type(exc).__name__}), running the full round')
            if not any((g.dir / f'round_{k:02d}' / 'post_replay').glob('frames_*')
                       for k in range(1, r)):
                log(g.dir, f'  r{r} direction is {direction}, but there is no replay to read yet; running the full round')
                kind = 'normal'
                if DISPATCH == 'inner':
                    # the plan attached above was an ART plan; let the normal
                    # chain plan for itself
                    rec.pop('plan', None)
        rec['kind'] = kind
        try:
            (art_round if kind == 'art' else one_round)(g, r, rec)
        except Exception:
            rec['crash'] = traceback.format_exc()[-1500:]
            log(g.dir, f'  CRASH in round {r}:\n{traceback.format_exc()[-700:]}')
        rec['tree_snapshot'] = snapshot_tree(g, r)
        g.save()

        # THE MONITOR'S TURN. After the round is on disk, so the checkpoint it
        # judges is the one that was actually written. A monitor that throws is
        # logged and dropped for the rest of the run: an estimator outage is not
        # a reason to end a run early, and silently stopping on one would look
        # exactly like saturation.
        if mon is not None and mon.due(r):
            try:
                mrec = mon.consider(r)
                rec['monitor'] = {k: mrec.get(k) for k in ('event', 'champion', 'delta')}
                log(g.dir, f"  monitor: r{r:02d} {mrec['event']} -> champion r{mrec['champion']}")
                s = mon.stop()
                if s is not None:
                    rec['monitor']['stop'] = True
                    log(g.dir, f'\nvalue stop: the champion has been r{s.champion:02d} for '
                               f'{mon.k} checkpoints; ending at round {r} of {a.rounds} '
                               f'and delivering r{s.champion:02d}')
                    break
            except Exception as exc:
                log(g.dir, f'  monitor: failed at r{r} ({type(exc).__name__}: {str(exc)[:140]}); '
                           'no early stop for the rest of this run')
                mon = None
    # The last art round's own output, which nothing else looks at. Under
    # monitor dispatch whether the run ends on one is not known in advance.
    if len(g.rounds) >= a.rounds and g.rounds[-1].get('kind') == 'art':
        try:
            art_close(g)
        except Exception:
            log(g.dir, f'  the final reading crashed:\n{traceback.format_exc()[-600:]}')
    # After every exit path, including the value stop's `break`: a run that
    # ended early spent what it spent.
    g.stop_proxy()
    log(g.dir, f'\n{a.game}: {len(g.rounds)} rounds written to {g.dir}')


if __name__ == '__main__':
    main()
