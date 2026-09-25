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
"""Did this round's repair earn its commit? One reader, one call.

WHERE THIS SITS. The loop spends three model calls before repair deciding
what to change, and after repair it asks only "does it still compile".
Fifteen rounds on five games shipped three regressions through that gap: a
projectile drawn at 10368px that buried the playfield, a guard flag set in one
of nine scenario branches that swallowed every key press, and -- not a
regression at all -- a marking control that grew a two-step confirmation the
fixed demo could not perform. Compiling told us nothing about any of them.

WHAT THIS IS NOT. It is not a rollback gate. A finding here does not revert
the round; it becomes the next round's work, at the front of the queue. Two
reasons. Reverting has to decide whether assets count (`code_snapshot`
deliberately excludes them, so an asset-only break cannot be reverted anyway),
and reverting throws away whatever else the round did right. The build-failure
rollback already in the loop stays -- a tree that will not load cannot be
played at all, and "fix it next round" does not apply to it.

WHAT IT IS NOT, SECOND. It is not a comparison against the previous round.
An earlier design replayed both trees and diffed them. It is not needed: the
question a checklist framework asks is "is this true of the game now", not
"did this round break it". Attribution is a question for the write-up, and
it can be answered offline from the stored rounds.

WHAT THIS DELIBERATELY DOES NOT DECIDE. A fixed demo failing is not proof the
game broke. On horror-tape-archive the repair made marking require a pause and
a confirm -- a better design -- and the demo, which clicks once, stopped
working. Reporting that as a regression would punish the improvement and send
the next round to undo it.

A `trace_stale` field was built to separate the two and then removed, because
it did not work: on that exact case the reader answered "yes" once and "no" in
five further readings, including two given four times the thinking budget. The
one right answer had quoted the MARK MODE ON prompt it saw in the after-frame,
so the evidence was there and the reading simply did not hold on to it -- and a
field that is right one time in six is worse than no field, because a wrong
"the script is stale" waves a real regression through.

So the prompt draws the line more cheaply instead: an input that visibly does
ANYTHING has been received and is not reported. That costs us the contract
change -- horror's case now yields nothing at all -- and a human notices it
from the score, which is what happened anyway. Splitting it into its own
smaller call, on one demo and one event, is the obvious next thing to try.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

RUN = Path(__file__).resolve().parent

_PROMPT = """A repair agent has just edited a game. Your job is to look at what
the game does now and answer two questions. You are the only check on this
edit: after you, it ships.

WHAT THE ROUND WAS TRYING TO DO
{goal}

WHAT THE TASK REQUIRES (the items this goal was drawn from)
{items}

WHAT YOU ARE LOOKING AT
Every one of the game's shipped demo recordings was replayed. A demo is a
FIXED script of clicks and key presses at fixed times -- it cannot adapt. For
every input you get the frame just BEFORE it and one or two frames AFTER,
plus the last frame of the recording. The frames are labelled.

One thing in every frame is not part of the game: the mouse pointer, drawn by
the recorder as a small cross or arrow. It shows you where a click landed,
which is useful, but it is never a defect. Do not report it as covering,
overlapping or garbling anything it happens to sit on.
{batch_note}
{demos}

ANSWER THESE TWO, IN THIS ORDER.

1. goal_achieved -- yes / no / unclear.
   Is the thing the round set out to do visibly true now? "unclear" is a real
   answer: say it when the demos never reach the part of the game the goal is
   about, rather than guessing.

2. broken_now -- a list, possibly empty.
   Anything the game does that is plainly wrong to a player watching these
   frames. Be concrete about what you SAW and in which frame. The kinds that
   matter most, because they are invisible in source code:
     - an input produces no change at all -- the before and after frames are
       the same, again and again, so the game is not listening
     - something covers the playfield: one sprite, one effect, one panel that
       hides the characters, the HUD or the action
     - the screen is frozen: the last frame equals the first and nothing
       between them moved
     - text is unreadable, clipped, or drawn on top of itself
   Do not list a missing feature here. A feature the game never had is a task
   requirement, not something that is broken. This list is for a game that is
   behaving wrongly.

AND ONE THING NOT TO REPORT. An input that visibly DOES something -- a mode
label appears, a prompt is shown, a message is printed -- has been received,
even when the demo never goes on to reach the outcome it was written for. That
is the game asking for more steps than this fixed script performs, and it is
not a fault in the game. Only an input that changes NOTHING belongs in
broken_now.

Then, only if broken_now is not empty, write next_goal: one sentence telling
the repair agent what to fix first. Name the file or the on-screen element if
the frames make it obvious. Fix the worst thing, not all of them.

Reply with strict JSON and nothing else:

{{
  "goal_achieved": "yes" | "no" | "unclear",
  "why": "<one sentence, citing a frame label>",
  "broken_now": [
    {{"what": "<one sentence>", "where": "<demo id and frame label>"}}
  ],
  "next_goal": "<one sentence, or empty when broken_now is empty>"
}}
"""

_MAX_FRAMES = 40


# HOW MUCH THINKING THIS READING IS ALLOWED.
#
# `_chat` caps reasoning with OpenRouter's `effort: "low"`, which is a
# preference, not a ceiling: measured on this module's own 28-frame payload
# against glm-5.3-flash it came back with 1232 reasoning tokens and 67.6s, and
# in two of the three acceptance cases it ran to the full 8000-token budget
# without ever beginning the answer, costing a doubled-budget retry -- 504s for
# two readings on horror-tape-archive. `reasoning: {"max_tokens": N}` is a
# ceiling: the same payload came back in 26.8s with 186 reasoning tokens and
# the same valid JSON. (Turning reasoning off entirely is not available; the
# endpoint answers 400, "Reasoning is mandatory for this endpoint".)
#
# THE CAP IS LOCAL TO THIS MODULE ON PURPOSE. `_REASONING_CAP` in stage1 is
# shared with the checklist read, which weighs twenty-six requirements against
# a session and has an argument for thinking longer. What this module returns
# is four short fields about frames in front of it. Cutting the shared constant
# would change that reading's quality to fix this one's latency.
# 512 WAS TOO TIGHT, MEASURED. It cut horror-tape-archive's reading from 504s
# to 209s and lost the answer that mattered: `trace_stale` flipped from yes to
# no, i.e. the reader stopped noticing that the click DID respond -- with a
# MARK MODE ON prompt -- and only the confirmation was missing. That judgement
# is the whole reason this module distinguishes a stale script from a broken
# game. The uncapped reading spent 1232 reasoning tokens on the same payload,
# so the ceiling has to sit above what the work costs, not at the point where
# the latency looks best.
_REASONING_TOKENS = int(os.environ.get('RSIGAME_LOOP_POST_VERIFY_REASONING') or 2048)
# Above the ceiling plus a full answer, so a first try is not spent discovering
# that it was not: at 4000 the reading retried once before it could finish.
_TOKENS = 8000
_CEILING = 16000


def _ask(messages, model: str, *, timeout: float | None = None):
    """One reading. Same three-try shape as `_chat`, its own reasoning ceiling.

    Not `_chat` because the cap has to differ, and not a parameter on `_chat`
    because that file is shared with three other callers who did not ask for
    this change.
    """
    from rsigame.agent.llm import _client, _extract_json
    from rsigame.agent.evolve.verifier_agent import _SAMPLING

    client = _client(model)
    budget, cap = _TOKENS, _REASONING_TOKENS
    for i in range(3):
        if i:
            time.sleep(3 * i)
        try:
            # An endpoint that stops answering must cost one try, not the
            # run: a stalled read has no deadline of its own and blocked a
            # shadow evaluation for ten minutes on a socket that was never
            # going to reply. Default stays None so existing callers are
            # unchanged.
            # DROPPING THE CEILING MEANS SENDING NO `reasoning` KEY, not
            # sending a ceiling of zero. Some endpoints read
            # `{'max_tokens': 0}` as "turn reasoning off" and refuse it
            # outright -- "Reasoning is mandatory for this endpoint and cannot
            # be disabled" -- so the fallback that was meant to save the
            # reading killed it instead, on all three tries. One game spent
            # five consecutive art rounds this way, 35 seconds each, and the
            # record of them is an art round with no repair in it at all.
            resp = client.chat.completions.create(
                model=model, messages=messages, max_completion_tokens=budget,
                **({'extra_body': {'reasoning': {'max_tokens': cap}}}
                   if cap else {}),
                **({'timeout': timeout} if timeout else {}), **_SAMPLING)
        except Exception as exc:
            # An endpoint that will not take the ceiling costs us the ceiling,
            # not the try -- the reading still has to happen.
            if cap:
                print(f'[post_verify] the reasoning ceiling was refused '
                      f'({type(exc).__name__}); retrying without it', flush=True)
                cap = 0
                continue
            print(f'[post_verify] try {i + 1}: {type(exc).__name__}: '
                  f'{str(exc)[:160]}', flush=True)
            continue
        choice = resp.choices[0]
        raw = choice.message.content or ''
        parsed = _extract_json(raw)
        if isinstance(parsed, dict):
            return parsed, raw
        if choice.finish_reason == 'length' and not raw:
            budget = min(budget * 2, _CEILING)
            print(f'[post_verify] try {i + 1}: the answer never started; '
                  f'retrying with {budget} tokens', flush=True)
            continue
        print(f'[post_verify] try {i + 1}: the reply was not JSON '
              f'({len(raw)} chars, finish_reason={choice.finish_reason})',
              flush=True)
    return None, ''


def _demo_text(demos: list) -> str:
    """The script, in words, so the reader knows what each frame answers."""
    out = []
    for d in demos:
        if d.get('replay_error'):
            out.append(f"  {d['demo_id']}: THE REPLAY FAILED -- "
                       f"{d['replay_error']}")
            continue
        head = f"  {d['demo_id']}"
        if d.get('scenario'):
            head += f" (launched with --scenario {d['scenario']})"
        out.append(head)
        for ev in d.get('events') or []:
            out.append(f"      input {ev['n']}: {ev['what']}")
        if not (d.get('events') or []):
            out.append('      (no inputs; this demo only watches)')
    return '\n'.join(out)


def _frames_for_demo(d: dict, after_n: int = 2) -> list:
    """(label, path) for one demo, every input bracketed.

    A before/after PAIR is the unit. A lone frame invites a judgement with
    nothing to compare against, which is the one mistake this whole module
    exists to avoid, so a pair is kept whole or dropped whole.
    """
    out = []
    for ev in d.get('events') or []:
        pair = []
        if ev.get('before'):
            pair.append((f"{d['demo_id']} input {ev['n']} "
                         f"({ev['what']}) -- BEFORE", ev['before']))
        for k, p in enumerate((ev.get('after') or [])[:after_n]):
            pair.append((f"{d['demo_id']} input {ev['n']} "
                         f"({ev['what']}) -- AFTER {k + 1}", p))
        if len(pair) >= 2:
            out.append(pair)
    return out


def _demo_frames(d: dict, cap: int) -> list:
    """Flat (label, path) for one demo, inside `cap` frames.

    Tries three frames an input, then two. Only if one demo's inputs still do
    not fit -- shooter-sky-duel ships a demo with seventeen -- are inputs
    sampled evenly across the recording, so what survives still spans it.
    """
    final = ([(f"{d['demo_id']} -- LAST FRAME", d['final_frame'])]
             if d.get('final_frame') else [])
    room = max(0, cap - len(final))
    for after_n in (2, 1):
        groups = _frames_for_demo(d, after_n)
        flat = [f for g in groups for f in g]
        if len(flat) <= room:
            return flat + final
    groups = _frames_for_demo(d, 1)
    per = len(groups[0]) if groups else 2
    keep = max(1, room // max(1, per))
    step = len(groups) / float(keep)
    picked = [groups[min(int(i * step), len(groups) - 1)] for i in range(keep)]
    return [f for g in picked for f in g] + final


def batches(demos: list, *, max_demos: int | None = None,
            max_frames: int = 30) -> list:
    """Split the demos across calls rather than dropping their frames.

    Forty native 854x480 PNGs are 7.5 MB of base64; the benchmark's own judge
    measured 36 frames at 4.18 MB working and 38 at 4.46 MB rejected. Shrinking
    them was the other option and was rejected: the evidence that separates a
    swallowed input from a stricter one is often a single caption -- MARK MODE
    ON -- and that is exactly what a downscale destroys. So resolution is kept
    and the demos are read a few at a time.
    """
    out, cur, n = [], [], 0
    for d in demos:
        if d.get('replay_error'):
            fr = []
        else:
            fr = _demo_frames(d, max_frames)
        if cur and ((max_demos is not None and len(cur) >= max_demos)
                    or n + len(fr) > max_frames):
            out.append(cur)
            cur, n = [], 0
        cur.append((d, fr))
        n += len(fr)
    if cur:
        out.append(cur)
    return out


def _one_call(goal_text: str, item_text: str, batch: list, k: int, n: int,
              model: str | None) -> dict:
    """One batch of demos -> one reading."""
    from rsigame.agent.evolve.verifier_agent import _data_uri

    demos = [d for d, _ in batch]
    frames = [f for _, fr in batch for f in fr]
    note = ('' if n == 1 else
            f"\nThis is part {k} of {n}. It carries {len(demos)} of the "
            f"game's demos; the others are being read separately. Judge only "
            f"what these frames show, and do not assume the rest are fine.\n")
    body = _PROMPT.format(goal=goal_text, items=item_text,
                          batch_note=note, demos=_demo_text(demos))
    content = [{'type': 'text', 'text': body}]
    for label, fp in frames:
        f = Path(fp)
        if not f.is_file():
            continue
        content.append({'type': 'image_url',
                        'image_url': {'url': _data_uri(f)}})
        content.append({'type': 'text', 'text': f'({label})'})

    try:
        parsed, _ = _ask([{'role': 'user', 'content': content}],
                         model=model or os.environ.get('RSIGAME_MODELS_LOOP_MODEL'))
    except Exception as exc:
        return {'error': f'{type(exc).__name__}: {str(exc)[:200]}',
                'demos': [d['demo_id'] for d in demos], 'n_frames': len(frames)}
    if parsed is None:
        return {'error': 'the reading returned nothing after three tries',
                'demos': [d['demo_id'] for d in demos], 'n_frames': len(frames)}
    out = _clean(parsed)
    out.update({'demos': [d['demo_id'] for d in demos],
                'n_frames': len(frames)})
    return out


def _merge(parts: list) -> dict:
    """Fold the per-batch readings into one answer.

    `goal_achieved` is optimistic on purpose: a goal usually shows up in one
    demo, so one batch seeing it is enough, while every other batch honestly
    reporting `unclear` about demos that never touch it must not outvote that.
    Everything else is pessimistic -- one batch finding the screen covered is
    enough for the screen to be covered.
    """
    ok = [p for p in parts if not p.get('error')]
    verdicts = [p['goal_achieved'] for p in ok]
    achieved = ('yes' if 'yes' in verdicts
                else 'no' if 'no' in verdicts else 'unclear')
    why = next((p['why'] for p in ok
                if p['goal_achieved'] == achieved and p['why']), '')
    broken = [b for p in ok for b in p['broken_now']]
    goals = [p['next_goal'] for p in ok if p['next_goal']]
    return {
        'goal_achieved': achieved,
        'why': why,
        'broken_now': broken,
        # One goal goes to the repair agent. The rest are kept rather than
        # discarded: which one was taken is a choice made mechanically here,
        # and the record has to show what it chose between.
        'next_goal': goals[0] if goals else '',
        'next_goal_all': goals,
        'errors': [p['error'] for p in parts if p.get('error')],
    }


def verify(goal: dict | str, items: list, replay: dict, *,
           model: str | None = None, max_demos: int | None = None,
           max_frames: int = 30) -> dict:
    """Read the round's own replay. `replay` is `replay_demos`'s return."""
    from rsigame.agent.llm import _load_env
    _load_env()

    demos = replay.get('demos') or []
    if not demos:
        return {'error': replay.get('error') or 'no demos were replayed'}

    goal_text = goal if isinstance(goal, str) else (
        (goal or {}).get('goal') or (goal or {}).get('ask') or '')
    if not goal_text:
        goal_text = '(no goal was recorded for this round)'
    item_text = '\n'.join(
        f"  {it.get('id')}  {it.get('requirement', '')}" for it in items
    ) or '  (none recorded)'

    bs = batches(demos, max_demos=max_demos, max_frames=max_frames)
    parts = [_one_call(goal_text, item_text, b, k, len(bs), model)
             for k, b in enumerate(bs, start=1)]
    out = _merge(parts)
    out.update({'calls': len(bs), 'parts': parts, 'n_demos': len(demos),
                'n_frames': sum(p.get('n_frames', 0) for p in parts),
                'n_failed_replays': replay.get('n_failed', 0), 'model': model})
    return out


def _clean(raw: dict) -> dict:
    """Keep the shape fixed whatever the model returned.

    A missing field must not read as a clean bill of health: `goal_achieved`
    falls back to `unclear`, never to `yes`.
    """
    def word(v, allowed, default):
        s = str(v or '').strip().lower()
        return s if s in allowed else default

    broken = []
    for b in (raw.get('broken_now') or []):
        if isinstance(b, str):
            broken.append({'what': b, 'where': ''})
        elif isinstance(b, dict) and (b.get('what') or '').strip():
            broken.append({'what': str(b['what']).strip(),
                           'where': str(b.get('where') or '').strip()})
    nxt = str(raw.get('next_goal') or '').strip()
    return {
        'goal_achieved': word(raw.get('goal_achieved'),
                              {'yes', 'no', 'unclear'}, 'unclear'),
        'why': str(raw.get('why') or '').strip(),
        'broken_now': broken,
        # A next_goal with nothing broken is noise; nothing broken and a goal
        # would put unowned work at the front of the queue.
        'next_goal': nxt if broken else '',
    }
