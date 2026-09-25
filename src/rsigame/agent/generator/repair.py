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
"""
G.repair(): turns one node's Diagnoses into 1-3 child nodes, each a real
coding-agent repair round on a cloned copy of the parent's source.

Four pieces:
  branch_diagnoses(diagnoses)   -> list[RepairAttempt]     pure, agent-free
  render_repair_prompt(attempt) -> str                     pure
  _clone_src / _diff_summary / _next_child_id               pure, disk-only

The agent invocation itself reuses scripts/agent_repair.ts (already existed
for the legacy eval/quality/repair_loop — same CLI contract we need:
--game-dir/--prompt-file/--model, JSON on stdout), which as of this change
imports its query() call from scripts/lib/agent_run.ts — the same shared
runAgent() batch_generate.ts uses, so the success-detection fix (a run whose
entire transcript is a synthetic error message is not a success) covers
repair too instead of drifting across a second copy.

branch_diagnoses's every branch has a distinct PURPOSE, not just an arbitrary split:
  - one dimension involved, all diagnoses high confidence -> 1 child,
    prescriptive (trust the diagnosed location/patch_target; combine if more
    than one diagnosis in that dimension — they're one coherent fault area).
  - one dimension involved, any diagnosis below confidence threshold -> 2
    children: prescriptive (as diagnosed) + exploratory (only the symptom, no
    location hint — hedges against the diagnosed location itself being
    wrong). NOT further isolated-by-dimension, since with one dimension
    there's nothing to isolate FROM — that would just rerun an identical
    prompt twice.
  - multiple dimensions involved -> up to 3: one "fix everything together"
    attempt plus isolated attempts for the top-2 dimensions by diagnosis
    count, so a bad fix in one area can't be blamed on / entangled with a fix
    in another. P scores each independently; only genuinely-improved,
    non-regressed children get promoted.

Capped at 3 children always (budget discipline — see core/budget.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


AGENT_TEST = Path(__file__).resolve().parents[2]
NODE_BIN = os.environ.get("RSIGAME_JUDGE_NODE_BIN", "node")   # PATH node from PATH;set RSIGAME_JUDGE_NODE_BIN to pin one
_SKIP_DIRS = {"node_modules", "dist", ".git", ".qwen"}

CONFIDENCE_HIGH = 0.8
MAX_CHILDREN = 3






_SKELETON_FILES = (Path("src/chrome/chrome.js"),)  # relative to a game's src root





def detect_engine(src) -> str:
    """Which kind of project this is, read off the directory itself.

    `project.godot` at the root is the whole test -- it is what Godot itself
    uses to decide a directory is a project. Deliberately not a parameter
    threaded down from the caller: every call site already holds the
    directory, and a mislabelled config would hand a Godot project the Phaser
    system prompt and the `dist/index.html` build check, which fails looking
    like a bad agent rather than like a bad label.
    """
    return "godot" if (Path(src) / "project.godot").is_file() else "web"


# ONE autonomous repair session, bounded two ways.
#
# `AGENT_BUDGET_S` is the ceiling on REASONING time -- wall clock minus the
# seconds `play_cli` spent driving the game. Ten minutes is well above the 2.6
# min median measured over the corpus run and well below the 30 min worst
# case, which is the one worth stopping.
#
# `timeout_s` is the ABSOLUTE backstop and exists only because the first cap
# excuses play time: without it an agent looping on screenshots would never
# hit any bound. It is not the budget and should never be reached.
#
# Ten minutes is right for a model whose thirty tool calls take 2.6 minutes. It
# is the WRONG number for a slower one, and wrong in a way that hides: the
# agent is told it has thirty actions and the clock takes it away at nineteen.
# Measured on glm-5.3-flash, grid-keys b0: 19 tool calls in 631s, with the time
# sitting in single 54s-161s gaps BEFORE each call -- the model reasoning about
# what to do next, not the tools running.
#
# What makes that fatal rather than merely slow: a retry does not resume. It
# re-reads the same files from scratch, so an agent cut off during
# investigation loses all of it and gets cut off again in the same place.
# Three attempts on that branch produced ZERO edits, where one uninterrupted
# session would have reached the edit. Attempts accumulate FILES, not
# understanding.
#
# So the ceiling is env-settable and the default is unchanged. Raise it only
# with a measured per-turn cost in hand -- and note that a slow model is not
# the only reason a session runs long. bounce-arcade b0 spent five minutes
# reading the HARNESS's source to decide whether to trust its own negative
# result, which is a session doing something worthwhile and a session the
# ceiling should still stop.
AGENT_BUDGET_S = int(os.environ.get("RSIGAME_REPAIR_AGENT_BUDGET_S") or 600)
MAX_TOOLS = 30


def _run_repair_agent_once(child_src: Path, prompt: str, *, model: str | None = None,
                      timeout_s: int | None = None, engine: str | None = None,
                      agent_budget_s: int = AGENT_BUDGET_S,
                      max_tools: int = MAX_TOOLS,
                      tag: str = "property") -> dict:
    # ONE DIRECTORY PER REPAIR, because these two filenames were fixed and the
    # directory is shared.
    #
    # `child_src.parent` is the same path for the property repair (…/child) and
    # for every blocking repair (…/parent) on one edge, so a single edge wrote
    # `_repair_prompt.txt` and `_repair_log.json` four times over:
    #
    #     blocking layer 1  ->  layer 2  ->  build repair  ->  property repair
    #
    # and only the last survived. The tool trace was the smaller loss. The
    # PROMPT went with it, and for training data that is the input half -- what
    # the agent was actually shown, which cannot be reconstructed from the
    # diff afterwards.
    #
    # Sequence AND tag: the sequence guarantees uniqueness (two blocking layers
    # on one edge would collide on the tag alone) and the tag says which kind
    # of repair it was, so neither a person nor a script has to infer it from
    # ordering. A retry gets its own directory too -- a session the platform
    # dropped and the one that replaced it are two events.
    # Named FOR the target, so two targets cannot share a counter.
    #
    # This was `child_src.parent / "repairs"`, and a layer repairing several
    # bugs in parallel hands over branches/b0, branches/b1, branches/b2 --
    # three targets, one parent directory. Three agents then raced on one
    # sequence: a live run produced 01-bug0, 03-bug1 and 01-bug2, and only the
    # tag kept two of those from being the same path.
    base = child_src.parent / f"_repairs_{child_src.name}"
    base.mkdir(parents=True, exist_ok=True)
    seq = sum(1 for x in base.iterdir() if x.is_dir()) + 1
    rdir = base / f"{seq:02d}-{tag}"
    rdir.mkdir(parents=True, exist_ok=True)
    prompt_file = rdir / "prompt.txt"
    prompt_file.write_text(prompt)
    log_file = rdir / "log.json"
    engine = engine or detect_engine(child_src)

    # THE ABSOLUTE BACKSTOP HAS TO MOVE WITH THE BUDGET IT BACKS.
    #
    # It was a fixed 1800s with a comment saying it "should never be reached".
    # Raising `agent_budget_s` to 1200 tonight made it reachable at once: the
    # budget charges only reasoning time, so 1200s of thinking plus a handful
    # of 45s play_cli calls clears 1800s of wall clock easily. Measured within
    # the hour: attempts spanning 1482s, 1523s, 1580s, 1660s and 1789s -- the
    # last one hitting the wall exactly.
    #
    # Worse than the kill was its disguise. A `TimeoutExpired` kill produces no
    # final JSON, so it arrived as `no agent output` with no `stopped_by`,
    # which is what a crashed subprocess looks like.
    if timeout_s is None:
        timeout_s = max(1800, agent_budget_s * 2)

    env = dict(os.environ)
    env["PATH"] = f"{NODE_BIN}:{env.get('PATH', '')}"
    # Read by agent_repair.ts on the OpenGame side, so the name is theirs:
    # exit as soon as the result line is flushed.
    env.setdefault("REPAIR_EXIT_WHEN_DONE", "1")

    # THE REPAIR AGENT MAY RUN SOMEWHERE ELSE FROM THE REST OF THE LOOP.
    #
    # Repair is text-only and by far the largest token consumer -- one game
    # three layers deep measured 45M input tokens, where the play agent and
    # the reader together are a fraction of that. But the play agent needs a
    # MULTIMODAL model (it reads frames) and the reader does too, so they
    # cannot follow repair onto a cheap text-only provider.
    #
    # Hence a separate pair. Set RSIGAME_REPAIR_OPENAI_BASE_URL and RSIGAME_REPAIR_API_KEY
    # to send only this subprocess elsewhere; leave them unset and it inherits
    # the same endpoint as everything else, which is what it did before.
    if os.environ.get("RSIGAME_REPAIR_OPENAI_BASE_URL"):
        env["OPENAI_BASE_URL"] = os.environ["RSIGAME_REPAIR_OPENAI_BASE_URL"]
        if os.environ.get("RSIGAME_REPAIR_API_KEY"):
            env["OPENAI_API_KEY"] = os.environ["RSIGAME_REPAIR_API_KEY"]
    cmd = ["npx", "tsx", "scripts/agent_repair.ts", "--game-dir", str(child_src),
           "--prompt-file", str(prompt_file), "--log-file", str(log_file),
           "--engine", engine, "--max-tools", str(max_tools)]
    if model:
        cmd += ["--model", model]

    start = time.time()
    # Process-group timeout handling: npx/tsx
    # spawns a node child which spawns further agent-process children, and a
    # plain subprocess.run(timeout=...) only SIGKILLs the top `npx` process,
    # orphaning the rest on timeout.
    proc = subprocess.Popen(cmd, cwd=str(AGENT_TEST), env=env, start_new_session=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # THE TIME CAP IS ON REASONING, so it cannot be a plain `communicate`
    # timeout: that measures the wall clock, and one `play_cli` call on a real
    # Phaser build was measured at 45 SECONDS. Thirteen of those would exhaust
    # a ten-minute wall budget before the agent had read a line of code --
    # which would make "you may run the game" mean "and it comes out of your
    # thinking time".
    #
    # So a watchdog polls the log `play_cli` writes and charges only
    # `elapsed - play_s`. `timeout_s` stays as the ABSOLUTE backstop: with all
    # play time excused there is otherwise no bound at all, and an agent stuck
    # in a screenshot loop would never hit the cap it is supposed to hit.
    play_log = child_src / "_repair_evidence" / "play_time.log"
    stop_watch = threading.Event()
    over_budget = {"hit": False}

    # A SESSION THAT NEVER ACTS IS NOT A SLOW SESSION, AND WAITING OUT ITS
    # WHOLE BUDGET IS PURE LOSS.
    #
    # Measured on glm-5.3-flash: six sessions spent their full ten minutes
    # having emitted NOTHING -- no tool call, no text, not even an error. The
    # model reasons without ever producing an action and the SDK simply waits.
    # None recovered; the retry that followed did the work. So the ten minutes
    # bought nothing at all, and raising the ceiling to 1200s would buy twenty.
    #
    # A first tool call normally lands in about 50s (measured: 49, 51, 54), so
    # 150s is three times the observed cost of getting started.
    #
    # The trajectory is APPENDED across attempts on one branch, so "has it
    # acted" has to mean "since THIS attempt began" -- hence the offset. Asking
    # whether the file contains a tool call at all would see the previous
    # attempt's work and never fire.
    traj = child_src / ".qwen" / "trajectory.jsonl"
    try:
        traj_at_start = traj.stat().st_size if traj.is_file() else 0
    except OSError:
        traj_at_start = 0
    silent_start_s = int(os.environ.get("RSIGAME_REPAIR_SILENT_START_S") or 150)
    acted = {"yet": False}
    # A SESSION THAT HAS WRITTEN ITS LOG HAS FINISHED. If its process is still
    # alive this long afterwards it is hanging on shutdown, not working, and is
    # stopped here -- recorded as finished, not as a death (see below).
    finished_grace_s = float(os.environ.get("RSIGAME_REPAIR_FINISHED_GRACE_S") or 30)

    def _watch():
        while not stop_watch.wait(5.0):
            if proc.poll() is not None:
                return
            played = 0.0
            try:
                if play_log.exists():
                    played = sum(float(x) for x in play_log.read_text().split()
                                 if x.strip())
            except Exception:
                played = 0.0
            charged = (time.time() - start) - played

            try:
                if log_file.is_file():
                    done_for = time.time() - log_file.stat().st_mtime
                    if done_for > finished_grace_s:
                        over_budget["hit"] = "finished"
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        except (ProcessLookupError, PermissionError):
                            pass
                        return
            except OSError:
                pass

            if not acted["yet"]:
                try:
                    if traj.is_file() and traj.stat().st_size > traj_at_start:
                        with traj.open("rb") as fh:
                            fh.seek(traj_at_start)
                            if b'"tool_use"' in fh.read():
                                acted["yet"] = True
                except OSError:
                    pass
                if not acted["yet"] and charged > silent_start_s:
                    over_budget["hit"] = "silent_start"
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                    return

            if charged > agent_budget_s:
                over_budget["hit"] = True
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                return

    watcher = threading.Thread(target=_watch, daemon=True)
    watcher.start()
    hit_wall = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        hit_wall = True
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()
    finally:
        stop_watch.set()
    wall_s = round(time.time() - start, 1)

    out_lines = [l for l in stdout.strip().splitlines() if l.strip()]
    try:
        result = json.loads(out_lines[-1]) if out_lines else {
            "success": False, "error": "no agent output",
            # WHY THE SESSION SAID NOTHING, instead of only that it said nothing.
            #
            # stderr was read and dropped. A session that dies before it can
            # print its JSON -- an unresolvable import, a missing binary, a
            # node version the SDK refuses -- puts the entire reason on stderr
            # and NOTHING on stdout, so `no agent output` was every one of
            # those failures wearing the same face.
            #
            # It is also the face of a wrong endpoint, and the runbook's
            # troubleshooting table maps it that way ("repair.n_tools is 0
            # everywhere -> wrong endpoint or key for RSIGAME_REPAIR_MODEL"). Measured
            # here: three branches, four attempts each, 203s apiece, all
            # reported as dead sessions, and the real cause -- workspace
            # packages never built, so `@opengame/sdk/dist/index.mjs` did not
            # exist -- appeared in no log the loop wrote. Twelve sessions of
            # evidence pointing at the model, and the model was fine.
            "stderr_tail": (stderr or "").strip()[-1200:] or None}
    except json.JSONDecodeError:
        result = {"success": False,
                  "error": f"agent output unparsable: {stdout[:200]}",
                  "stderr_tail": (stderr or "").strip()[-1200:] or None}
    result["wall_s"] = wall_s

    # THE AGENT'S OWN SCRATCH FILES ARE NOT A REPAIR.
    #
    # `files_edited` comes from the SDK and counts every write, including the
    # probe scripts the agent writes to look at the game with. Measured twice:
    # eighteen of the 45 branches whose verdict was `fixed` with an empty
    # source diff had in fact "edited files" -- `_repair_evidence/paddle_check5.py`
    # and its four siblings -- and the checklist run wrote `/tmp/checkboot.py`
    # and a copy of it into the game directory.
    #
    # Any consumer filtering on "did this repair change anything" is fooled by
    # those, and that filter is what separates a real fix from a verdict with
    # nothing behind it. Split rather than dropped: what the agent wrote to
    # look at the game is worth keeping, it is simply not a repair.
    # THE TEST WAS WEB-ONLY, AND SILENTLY EMPTIED THE FIELD ON GODOT.
    #
    # It kept a path only if it started `src/` and ended `.ts/.tsx/.js`. A
    # Godot game keeps its code in `scripts/*.gd` beside `project.godot`, so
    # every real source edit was filed as a scratch probe and `files_edited`
    # came back `[]` -- for all eighty rounds of the Godot arms, while the
    # session logs beside them named the file that changed. Any analysis
    # asking "did this round edit anything" had to fall back on the tree diff
    # against G0, which on these projects is mostly `.import` sidecars Godot
    # regenerates on its own.
    #
    # Recognise the source of both engines, and decide scratch by WHERE a file
    # is rather than by what it is not: the agent's probes go under
    # `_repair_evidence/`, and `demo_outputs/` is the evaluator's input, which
    # a repair must never be credited with touching.
    _SRC_SUFFIX = ('.ts', '.tsx', '.js', '.jsx', '.mjs',
                   '.gd', '.tscn', '.tres', '.godot', '.cfg', '.json')
    _NOT_A_REPAIR = ('_repair_evidence/', 'demo_outputs/', 'node_modules/',
                     'dist/', '.godot/')
    edited = result.get("files_edited") or []
    src, scratch = [], []
    for f in edited:
        rel = str(f).split(f"/{child_src.name}/")[-1]
        is_src = (rel.endswith(_SRC_SUFFIX)
                  and not any(part in rel for part in _NOT_A_REPAIR)
                  and not rel.startswith('/'))     # outside the game tree
        (src if is_src else scratch).append(f)
    result["files_edited"] = src
    if scratch:
        result["scratch_files"] = scratch

    # THE TOOL TRACE, read back off disk. `agent_repair.ts` writes the full
    # `tools` array to the log file and deliberately strips it from stdout --
    # stdout is the compact channel this function parses. So every caller that
    # reached for `result['tools']` got nothing, and the episode's
    # `trace_ref` came out empty on all 53 episodes of the corpus run: what
    # the agent read, edited and ran was captured and then never picked up.
    #
    # That trace is the only record of which source the agent actually
    # inspected, which is exactly what plan distillation reads. Merged here so
    # every caller gets it, rather than at one call site.
    try:
        if log_file.exists():
            log = json.loads(log_file.read_text())
            if isinstance(log, dict):
                result.setdefault("tools", log.get("tools") or [])
                result["log_file"] = str(log_file)
                # `messages` is the agent's own narration. Picked up here
                # for the same reason as the rest: a field the TypeScript side
                # writes and the Python side never reads is a field that
                # exists everywhere except the record that matters.
                # `cost` is the SDK's own accounting for this session:
                # input/output/cache tokens, `numTurns`, `durationApiMs`, and a
                # per-model breakdown. It is the ONLY place a model-call count
                # is measured rather than approximated -- `n_tools` counts tool
                # calls, which over-counts a turn carrying two and misses a turn
                # that answers with none. A budget-matched comparison rests on
                # that difference.
                #
                # Absent stays absent. `setdefault` never invents a zero here,
                # and no reader downstream may either.
                for k in ("images_read", "files_edited", "messages", "cost"):
                    if k in log:
                        result.setdefault(k, log[k])
    except Exception as exc:
        result["trace_read_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"

    # THE BUDGET IS ON REASONING, NOT ON USING THE TOOLS. `play_cli` stamps its
    # own elapsed seconds into a log beside the game; summing that and taking
    # it off the wall clock is what stops "you may run the game" from silently
    # meaning "and every second you spend watching it comes out of your
    # thinking time". Booting a Phaser build is several seconds and a Godot
    # process more, so on a session that drives the game a dozen times the
    # difference is most of the budget.
    play_s = 0.0
    try:
        log = child_src / "_repair_evidence" / "play_time.log"
        if log.exists():
            play_s = sum(float(x) for x in log.read_text().split() if x.strip())
    except Exception:
        play_s = 0.0
    result["play_s"] = round(play_s, 1)
    result["agent_s"] = round(max(0.0, wall_s - play_s), 1)
    result["agent_budget_s"] = agent_budget_s

    # HOW MANY TOOL CALLS THIS ATTEMPT ACTUALLY MADE, read off the trajectory
    # rather than off `n_tools`.
    #
    # `n_tools` comes from the final JSON, and a killed session never writes
    # one -- so `r.get('n_tools') or 0` turned ABSENT into ZERO, and the log
    # said "stopped by its own time budget after 0 tool calls" about a session
    # that had made thirty-nine. Two hours of my own analysis rested on that
    # number before a trajectory contradicted it.
    #
    # The trajectory is appended as the session runs, so it survives the kill
    # and it is the only honest source. Counted from this attempt's starting
    # offset, since the file spans every attempt on the branch.
    try:
        if traj.is_file():
            with traj.open("rb") as fh:
                fh.seek(traj_at_start)
                result["tools_seen"] = fh.read().count(b'"tool_use"')
        else:
            result["tools_seen"] = 0
    except OSError:
        result["tools_seen"] = None
    # Where the prompt and the trace for THIS attempt live. Without it a reader
    # holding the record has no way back to what the agent was shown.
    result["repair_dir"] = str(rdir)
    result["repair_tag"] = tag
    # KILLED AFTER IT HAD ALREADY FINISHED is not a death. `agent_repair.ts`
    # writes its log and prints its result as its last acts, so either one
    # being present means the session ended on its own terms -- and its own
    # `stopped_by` (tool_budget, or none) is the true one. Recorded instead as
    # `killed_after_finishing`. Before this, a session that used all 40 of its
    # tool calls and then hung on exit was overwritten to `time_budget` and
    # retried from scratch.
    agent_wrote_log = False
    try:
        agent_wrote_log = log_file.is_file() and 'synthesized_by' not in json.loads(
            log_file.read_text())
    except (OSError, ValueError):
        agent_wrote_log = False
    finished = agent_wrote_log or result.get("engine") is not None
    if (over_budget["hit"] or hit_wall) and finished:
        result["killed_after_finishing"] = (
            "finished" if over_budget["hit"] == "finished"
            else "wall_clock" if hit_wall and not over_budget["hit"]
            else "time_budget")
        if result.get("engine") is None and agent_wrote_log:
            # stdout was lost in the kill; the log holds the same result
            try:
                lg = json.loads(log_file.read_text())
                for k in ("success", "error", "n_tools", "stopped_by", "engine"):
                    if k in lg:
                        result[k] = lg[k]
            except (OSError, ValueError):
                pass
    elif over_budget["hit"]:
        # Said plainly, and not folded into `error`. A session stopped by its
        # own clock is a truncation, not a failure of the repair, and the two
        # must not read the same when the episode is used as training data.
        #
        # Two clocks, and they mean different things. `time_budget` is a
        # session that worked and ran out. `silent_start` is one that never
        # began -- no tool call in the first 150s -- which is not a truncated
        # repair at all and should not be read as one.
        result["stopped_by"] = ("silent_start"
                                if over_budget["hit"] == "silent_start"
                                else "time_budget")
    elif hit_wall:
        # The third kill, and the one that used to be invisible. A
        # `TimeoutExpired` writes no final JSON either, so it arrived as
        # `no agent output` with no `stopped_by` -- the shape of a crash. It
        # is not a crash and it is not the reasoning budget; it is play time,
        # which the budget deliberately does not charge for.
        result["stopped_by"] = "wall_clock"
        result["timeout_s"] = timeout_s

    # A KILLED SESSION MUST STILL LEAVE A LOG, because everything downstream
    # counts by reading the directory rather than this return value.
    #
    # `agent_repair.ts` writes `log.json` when it finishes, so a session cut
    # off at its budget writes none -- and a caller totalling `n_tools` across
    # `*/log.json` then scores that attempt at zero. Measured across the
    # Self-Refine runs, 14-17% of attempts had no log, and they were not a
    # random 14%: they are exactly the attempts that used the whole budget, so
    # the totals understated spend precisely where spend was highest. One round
    # recorded 0 tool calls for a session that had made 22 and cost $0.022.
    #
    # `tools_seen` is counted off the trajectory, which is appended as the
    # session runs and survives the kill, so the number here is real. The file
    # is marked `synthesized` so no reader mistakes it for the agent's own.
    try:
        lg = rdir / "log.json"
        if not lg.is_file():
            json.dump({"synthesized_by": "repair.py: session died before "
                                          "agent_repair.ts wrote its log",
                       "n_tools": result.get("tools_seen") or 0,
                       "n_play_calls": 0,
                       "stopped_by": result.get("stopped_by"),
                       "success": False,
                       "agent_s": result.get("agent_s"),
                       "play_s": result.get("play_s"),
                       "cost": {}},
                      lg.open("w"), indent=1)
    except OSError:
        pass
    return result


# Up to three retries when the platform drops the session.
#
# The first version of this retried ONLY a session with zero tool calls, on the
# reasoning that one which had already edited files would restart from a
# half-modified tree and become a different repair. That reasoning was wrong in
# a way worth writing down: the verifier judges the ARTIFACT, not the path
# taken to it. It does not know or care how many attempts produced the tree in
# front of it, and the four gates apply the same either way. Meanwhile an edge
# abandoned to a 400 is pure loss.
#
# Measured on the first three-game run: 3 of 9 repair sessions were killed by
# the platform -- `[API Error: 429 qpm limit]` at zero tools, and
# `[API Error: 400 Request body must be valid JSON]` after six and after
# twelve. The generation line shares the same quota and was seeing four 400s a
# cycle at the same time.
#
# After the last attempt the loop stops and hands whatever is on disk to the
# verifier. It does not decide for itself that the repair failed: if six tools
# of work landed before the session died, that IS a child artifact and the
# gates are what say whether it is any good.
MAX_ATTEMPTS = 4                       # the first try plus three retries
RETRY_BACKOFF_S = (20, 60, 120)        # 429s need time, not immediacy


def _session_died(r: dict) -> str:
    """Why this attempt was not a session, or '' if it was one.

    Deliberately the same question `close_edge.agent_never_ran` asks, and for
    the same reason: `success` has been wrong about a platform error before,
    so the tool count is checked too.
    """
    if not isinstance(r, dict):
        return 'the repair returned nothing at all'
    if r.get('replay'):
        return ''
    if r.get('stopped_by') == 'wall_clock':
        # Distinct from the reasoning budget on purpose. This session was not
        # thinking too long -- it was PLAYING, which the budget excuses and the
        # wall clock does not. A reader who sees these should raise the
        # backstop, not the budget.
        n = r.get('tools_seen')
        return (f'ran past the {r.get("timeout_s")}s wall clock '
                f'({r.get("agent_s")}s reasoning, {r.get("play_s")}s playing'
                + (f', {n} tool calls' if n is not None else '') + ')')
    if r.get('stopped_by') == 'silent_start':
        # Not a truncated repair. The session never started one: no tool call
        # in its first 150s. Six of these were measured on glm, none recovered
        # within the session, and every retry that followed did the work -- so
        # this reads as "go again", not as "it tried and failed".
        return 'never made a tool call; stopped early rather than waiting out '\
               'the budget'
    if r.get('stopped_by') == 'time_budget':
        # Still a death, and still retried -- but not the same death. A
        # session killed by its own clock produced no final JSON, so it
        # arrives here as `success: False, error: "no agent output"`, which is
        # exactly what a crashed subprocess looks like. `_run_repair_agent_once`
        # already refuses to fold this into `error` for that reason; saying
        # "no agent output" here undoes that a line later.
        #
        # Measured on bounce-arcade b0: the session edited a file, built it,
        # self-checked with play_cli, disliked the answer, and then spent five
        # minutes reading the HARNESS's source. That is what the ceiling is
        # for. Reported as a crash, it looks like a platform fault and invites
        # raising the ceiling, which would buy the wandering more time.
        # `tools_seen` is counted off the trajectory, which survives the kill.
        # `n_tools` comes from a final JSON the killed session never wrote, so
        # reading it here reported "0 tool calls" for a session that had made
        # thirty-nine -- and that false zero is what a reader then uses to
        # decide whether the ceiling or the model is at fault.
        n = r.get('tools_seen')
        n = r.get('n_tools') or 0 if n is None else n
        return (f'stopped by its own time budget after {n} tool calls '
                f'({r.get("agent_s")}s of agent time)')
    if r.get('success') is False:
        why = str(r.get('error') or 'the session reported failure')[:160]
        # The retry line is the only place most readers ever look, so the
        # cause travels WITH the death rather than sitting in a JSON field
        # nobody opens. Last non-empty stderr line: for a node crash that is
        # the exception, which is the whole diagnosis.
        tail = [l for l in (r.get('stderr_tail') or '').splitlines() if l.strip()]
        return f'{why} -- {tail[-1].strip()[:120]}' if tail else why
    if not r.get('n_tools'):
        msgs = r.get('messages') or []
        return (str(msgs[-1])[:160] if msgs
                else 'the session made no tool calls and said nothing')
    return ''


def _asset_budget():
    """(cap, ledger path) when the per-round image budget is on, else None.

    Set by the caller for one round; read by generate_game_assets, which writes
    the ledger. Off unless both are set, so every other caller is unchanged.
    """
    try:
        cap = int(os.environ.get('OPENGAME_ASSET_ROUND_CAP') or 0)
    except ValueError:
        cap = 0
    path = (os.environ.get('OPENGAME_ASSET_LEDGER') or '').strip()
    return (cap, Path(path)) if cap > 0 and path else None


def _ledger_used(ledger: Path) -> int:
    try:
        return int(json.loads(ledger.read_text()).get('used') or 0)
    except Exception:
        return 0


def _budget_block(cap: int, used: int) -> str:
    return ('\n\n====================\nIMAGE BUDGET FOR THIS ROUND\n'
            '====================\n'
            f'generate_game_assets may produce at most {cap} new images in this '
            f'round, counted across every attempt at it. {used} have already been '
            f'generated this round; {max(0, cap - used)} remain. Requests beyond '
            'that are refused, and every call reports the balance. Generate only '
            'what you will wire into the game in this session.\n')


_IMG = ('.png', '.jpg', '.jpeg', '.webp')
_REF_EXT = ('.gd', '.tscn', '.tres', '.json', '.cfg', '.ts', '.js', '.html')


def _trajectory_events(child_src, offset: int) -> list:
    """The events one attempt appended to the agent's trajectory.

    The file is appended across attempts and survives a killed session, which
    is why it and not `log.json` is read: a killed attempt never writes its log.
    """
    traj = Path(child_src) / '.qwen' / 'trajectory.jsonl'
    out = []
    try:
        with traj.open('rb') as f:
            f.seek(offset)
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def _images_since(root, since: float) -> list:
    """Image files under `root` written at or after `since`, relative paths.
    Hidden and underscore directories hold the harness's own files."""
    root = Path(root)
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if not d.startswith(('.', '_')) and d != 'node_modules']
        for fn in filenames:
            if fn.lower().endswith(_IMG) and not fn.startswith('_'):
                p = Path(dirpath) / fn
                try:
                    if p.stat().st_mtime >= since:
                        out.append(str(p.relative_to(root)))
                except OSError:
                    pass
    return sorted(out)


def _unreferenced(root, images: list) -> list:
    """Which of `images` no scene, script or data file names yet."""
    if not images:
        return []
    names = {Path(i).name: i for i in images}
    seen = set()
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if not d.startswith(('.', '_')) and d != 'node_modules']
        for fn in filenames:
            if not fn.endswith(_REF_EXT) or fn.endswith('.import'):
                continue
            try:
                text = (Path(dirpath) / fn).read_text(errors='ignore')
            except OSError:
                continue
            for n in names:
                if n not in seen and n in text:
                    seen.add(n)
    return [names[n] for n in names if n not in seen]


def _short(v, n=90) -> str:
    v = ' '.join(str(v).split())
    return v if len(v) <= n else v[:n - 3] + '...'


def _tool_target(inp) -> str:
    if not isinstance(inp, dict):
        return ''
    for k in ('file_path', 'absolute_path', 'path', 'command', 'pattern', 'query'):
        if inp.get(k):
            return str(inp[k])
    if isinstance(inp.get('assets'), list):
        return 'assets: ' + ', '.join(str(a.get('key')) for a in inp['assets']
                                      if isinstance(a, dict))
    return ''


def _attempt_summary(n: int, why: str, events: list, images: list,
                     unreferenced: list, root=None) -> str:
    """What an attempt that died left behind, for the attempt that follows it.

    A retry is a fresh session on the same prompt: without this it does not
    know that the edits and images on disk are its predecessor's, and the
    measured result was a second and third batch of images and no wiring.
    """
    uses = [e for e in events if e.get('t') == 'tool_use']
    end = next((e for e in reversed(events) if e.get('t') == 'end'), None)
    note = next((e.get('text') for e in reversed(events)
                 if e.get('t') == 'text' and str(e.get('text') or '').strip()), '')
    rel = (lambda p: str(Path(p).relative_to(root))
           if root and str(p).startswith(str(root)) else str(p))
    edited = []
    for e in uses:
        if re_edit(e.get('name', '')):
            t = rel(_tool_target(e.get('input')))
            # The harness's own files (.qwen/, _repair_evidence/) are not
            # edits to the game.
            if t and not t.startswith(('.', '_')) and t not in edited:
                edited.append(t)
    gen_calls = [e for e in uses if e.get('name') == 'generate_game_assets']
    lines = [f'Attempt {n} of this repair was cut off: {why}.']
    if end is not None and end.get('stoppedBy') == 'tool_budget':
        lines.append(f'(It had already used all {end.get("nTools")} of its tool '
                     'calls and finished; it was cut off while shutting down.)')
    lines.append(f'It made {len(uses)} tool calls. Its last actions:')
    for e in uses[-5:]:
        lines.append(f'  - {e.get("name")} {_short(rel(_tool_target(e.get("input"))))}'.rstrip())
    if note:
        lines.append(f'Its last note: {_short(note, 240)}')
    lines.append('Files it edited (still on disk, not reverted): '
                 + (', '.join(edited) if edited else 'none'))
    if images:
        shown = ', '.join(images[:25]) + (f' ... and {len(images) - 25} more'
                                           if len(images) > 25 else '')
        lines.append(f'Images it generated ({len(images)}, on disk): {shown}')
        lines.append(f'{len(unreferenced)} of those images are not referenced by '
                     'any scene, script or data file yet.')
    elif gen_calls:
        lines.append(f'Images it generated: none. It called generate_game_assets '
                     f'{len(gen_calls)} times, but no image reached the disk.')
    else:
        lines.append('Images it generated: none')
    todo = []
    if edited:
        todo.append('check those edits')
    if unreferenced:
        todo.append('wire in the images that are not used yet')
    lines.append('Do not redo that work. Start from what is on disk'
                 + (': ' + ', '.join(todo) if todo else '')
                 + ('; generate new images only for something still missing.'
                    if images or gen_calls else '.'))
    return '\n'.join(lines)


def re_edit(name: str) -> bool:
    n = (name or '').lower()
    return any(k in n for k in ('write', 'edit', 'replace'))


def _earlier_block(summaries: list) -> str:
    if not summaries:
        return ''
    return ('\n\n====================\nEARLIER ATTEMPTS AT THIS SAME REPAIR\n'
            '====================\n' + '\n\n'.join(summaries) + '\n')


def _run_repair_agent(child_src, prompt, max_attempts=None, **kw):
    """Run the repair agent, retrying a session that died before it acted.

    `max_attempts` overrides MAX_ATTEMPTS for ONE caller. It exists because a
    retry does not resume: it starts over with a fresh `max_tools`, so a round
    that retried three times spent four times the tool budget it was given.

    THE LOOP CAN AFFORD THAT AND THE SERIAL ARM CANNOT, which is why this is a
    parameter rather than a new constant. The loop spreads a game's ~212 tool
    calls over roughly fifteen short branch sessions, so one death costs a
    fifteenth of the budget and the retry is what stops a platform fault from
    being recorded as a failed repair. The Self-Refine arm has to reach the
    same total in EIGHT sessions of 26, so one retry is an eighth of the game's
    whole budget handed back -- measured over the forty completed A0 runs, 75
    of 320 rounds retried and spent a mean of 61.8 tool calls against a stated
    26, up to 156 in one round and 434 against a stated 208 across one game.
    A budget-matched comparison that is out by 2x on one side is not one.

    Note what this does NOT change: the retry still fires on a session that
    never acted, which is the case it was written for. A session that used its
    tools and landed no edit was never retried -- `_why_not_a_session` tests
    `n_tools`, not the diff -- and still is not.
    """
    history = []
    # EVERY ATTEMPT'S EDITS, not just the surviving attempt's.
    #
    # The branch directory is NOT reset between attempts, so an attempt that
    # dies after editing leaves that edit in place and the next attempt
    # inherits it. Measured on bounce-arcade b0: attempt 1 made the one edit
    # that fixed the defect and was then killed at its budget; attempt 3 read
    # the code, found the fix already there, and returned success having
    # touched nothing. The branch record takes `files_edited` from the last
    # attempt, so it recorded a branch that changed a file as having changed
    # none -- while composition merged that very file. The evidence survived
    # only inside a superseded attempt's log.
    edited = []
    r = {}
    budget = _asset_budget()
    summaries = []
    for attempt in range(max_attempts or MAX_ATTEMPTS):
        if attempt:
            wait = RETRY_BACKOFF_S[min(attempt - 1, len(RETRY_BACKOFF_S) - 1)]
            print(f'[repair] attempt {attempt} died ({history[-1][:80]}); '
                  f'retrying in {wait}s', flush=True)
            time.sleep(wait)
        # THE BALANCE IS RE-READ FOR EVERY ATTEMPT. An attempt that died after
        # generating images left them on disk and in the ledger; the next one
        # has to be told how much is left, not the number the round began with.
        used_before = _ledger_used(budget[1]) if budget else 0
        attempt_prompt = (prompt + _earlier_block(summaries)
                          + (_budget_block(budget[0], used_before) if budget else ''))
        traj = Path(child_src) / '.qwen' / 'trajectory.jsonl'
        try:
            traj_off = traj.stat().st_size if traj.is_file() else 0
        except OSError:
            traj_off = 0
        # Two seconds of slack: file mtimes come from a coarser clock than
        # time.time() and were measured landing a few ms BEFORE it.
        t_start = time.time() - 2.0
        r = _run_repair_agent_once(child_src, attempt_prompt, **kw)
        made = (_ledger_used(budget[1]) - used_before) if budget else 0
        if budget and isinstance(r, dict):
            r['images_generated'] = made
        for f in (r.get('files_edited') or []):
            if f not in edited:
                edited.append(f)
        why = _session_died(r)
        if not why:
            break
        # BEING CUT OFF IS NOT THE SAME AS HAVING FAILED, once the session has
        # edited something.
        #
        # The branch directory is never reset, so a session killed at its
        # budget leaves its edits behind. Whether those edits FIXED the defect
        # is a question this loop cannot answer and the caller can: `branch_for`
        # builds the branch and runs a verification session against it, three
        # lines after this returns. Retrying first spends another full budget
        # to find out something the verifier was about to measure anyway.
        #
        # Measured on bounce-arcade b0. Attempt 1 was cut off after making the
        # one edit that fixed the defect. Attempt 2 was cut off after nine
        # more writes. Attempt 3 then ran 21 tools, edited NOTHING, and
        # returned success -- it had read the code, found the fix already
        # present, and said so. Nineteen minutes to re-derive "already fixed",
        # and 45 minutes on the branch against the 17 its sibling took.
        #
        # The retry ladder is still right for what it was built for: a session
        # that produced NOTHING. A dropped platform session, a silent start, a
        # model that reasoned without acting -- those leave no candidate, and
        # going again is the only move. This narrows the ladder to that case
        # rather than removing it, which is why the condition is on the edits
        # and not on `stopped_by` alone: a budget kill during investigation,
        # before any edit, still retries, and the comment on AGENT_BUDGET_S is
        # still describing that case correctly.
        if r.get('stopped_by') == 'time_budget' and (
                r.get('files_edited') or edited):
            # Recorded, and NOT as a death -- the difference is the whole
            # point. A reader has to be able to tell "cut off holding a
            # candidate" from "came back with nothing", because only one of
            # them says anything about the model.
            r['cut_off_holding_edits'] = why
            break
        history.append(why)
        # WHAT THIS ATTEMPT LEFT BEHIND, for the next one. Measured on one
        # platformer round without it: three attempts, 63 images, zero files
        # edited, 96 minutes -- each retry started over and generated again.
        try:
            imgs = _images_since(child_src, t_start)
            summaries.append(_attempt_summary(
                attempt + 1, why, _trajectory_events(child_src, traj_off),
                imgs, _unreferenced(child_src, imgs), root=child_src))
        except Exception as exc:
            summaries.append(f'Attempt {attempt + 1} of this repair was cut off: '
                             f'{why}. (Its summary could not be built: '
                             f'{type(exc).__name__}.)')
    # Union, in first-touched order. The final attempt's own list stays a
    # subset of it, so a reader who wants "what did the winning session do"
    # can still get that from its log.
    if edited:
        r['files_edited'] = edited
    if history:
        # Never silent. Two events collapsed into one record hide how often the
        # platform is dropping sessions, and that number is what says whether
        # the quota shared with the generation line needs attention.
        r['dead_session_attempts'] = history
        r['attempts_made'] = len(history) + (0 if _session_died(r) else 1)
    return r


