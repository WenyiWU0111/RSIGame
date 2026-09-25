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
"""Stage A grounds the evidence into issues; Stage B decides which get budget.

Kept apart because they answer different questions and one call would blur
them: whether an issue exists is a reading of the frames, whether it deserves
this round is a reading of the budget and the direction, and a single call that
did both would let "this matters to me" leak into "I saw this".

The gate between them is the reliability rule the whole framework rests on:

    confirmed_gap   may be repaired
    unverified      may NOT be repaired

Not observed is not broken. A round that repairs an `unverified` item is
repairing something nobody has seen fail, and if it then verifies clean, the
loop has learned nothing except that it can spend a round. The gate is enforced
in code after Stage B answers, so a selection that reaches for an unverified
issue is dropped and recorded rather than argued with.

Neither stage may say how to fix anything. Stage A reports what is visible;
Stage B reports what is worth doing; the repair agent, which is the only one
that will have read the code, decides how.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUN))

STATUS = ('confirmed_gap', 'unverified')
IMPORTANCE = ('major', 'medium', 'minor', 'unknown')
DIRECTIONS = ('correctness', 'completeness', 'feedback', 'presentation')
ROLES = ('primary', 'concurrent_minor', 'deferred')

# Not a tuned number, a starting one, and low on purpose: a packet the repair
# agent cannot finish teaches us nothing about batching, while a packet it
# finishes early costs one round. Raise it once real rounds say it is safe.
MAX_CONCURRENT = 3


_GROUND = """You are reading what a game did when it was played, and turning it
into a list of problems that the evidence actually supports.

You do not say how to fix anything. Someone else reads the code and decides
that; they need your account to weigh on its own.

Say what is observable. "The score is partly covered by the objective panel" is
an observation. "Reduce the panel width by 20 pixels" is an instruction, and it
assumes a layout you have not seen the code for.

====================
WHAT THE TASK REQUIRES
====================
{items}

====================
DIRECTION BEING WORKED ON
====================
{direction}

====================
THE QUESTION THIS ROUND WENT TO ANSWER
====================
{question}

====================
WHAT WAS SEEN
====================
{evidence}

{frames_note}
{static_note}====================
RULES
====================
- The written account and the frames are one body of evidence, not two. Where
  they disagree, the frames are what happened and the account is someone's
  reading of it -- say so rather than choosing silently.
- The recording is not frame-exact. Replaying one build twice cuts the same
  moment up to a frame or two apart, so a single frame showing no change after
  an input is not evidence that the input is dead. You get two frames after
  each input: if the change is there by the second, or the moment after undoes
  it, that is the recorder. Only something still there across the frames you
  were given is `confirmed_gap`; the rest is `unverified`.
- A behaviour the task requires is not a defect, however strange it looks.
  Read the requirements above before calling something wrong: a game that is
  meant to misbehave is working when it misbehaves.
- Every issue must point at something in the evidence above.
- The question is what this round set out to check, not what was found. A claim
  written into the question -- including someone's earlier report of what they
  saw while playing -- is not evidence and cannot make an issue
  `confirmed_gap`; only WHAT WAS SEEN can.
- `confirmed_gap` means the evidence SHOWS the problem. `unverified` means the
  evidence never exercised it. Not seeing a behaviour is not seeing it broken,
  and an issue you cannot point at belongs in `unverified` with empty evidence.
- Issues outside the direction above are still worth recording if the evidence
  shows them. Say which direction each belongs to.
- Do not repeat one problem as several issues.

Reply with strict JSON and nothing else:

{{"issues": [{{"issue_id": "I1",
             "issue": "<one sentence, what is wrong and where it is visible>",
             "status": "confirmed_gap|unverified",
             "direction": "correctness|completeness|feedback|presentation",
             "importance": "major|medium|minor|unknown",
             "evidence_refs": ["<which demo and which moment>"],
             "note": "<one sentence of what the frames showed>"}}]}}
"""


_RANK = """You decide which of this round's confirmed problems get its repair
budget. You do not decide how to fix them.

Exactly one is the PRIMARY TARGET: the thing this round is for, and the thing
the round is judged on. It should sit in the direction being worked, and it
should normally be the problem the round's question was about.

Some of the rest may be CONCURRENT MINOR FIXES -- small, evidence-backed
problems that someone already editing this code can reasonably put right in the
same sitting. They are secondary in priority, not optional. A good candidate:

  - is visibly small next to the primary target -- a clipped label, an overlap,
    a wrong scale -- and not another feature;
  - lives in the same screen or interaction, so the same replay shows both;
  - needs no new investigation to confirm.

Everything else is DEFERRED. Prefer deferring when repairing or checking
something would widen what this round is about. Do not build a multi-feature
plan; a round that does one thing properly beats a round that half-does four.

Do not say a fix will be small in the code. You have not read the code.

====================
DIRECTION AND QUESTION
====================
{direction}
{question}

====================
CONFIRMED PROBLEMS
====================
{issues}

====================
ALREADY TRIED
====================
{history}

Budget: {budget} round(s) left.

The ids you answer with are the issue ids above -- I1, I2 and so on. The
TGT.. ids under ALREADY TRIED are a different thing entirely: they name work
that has been going on across rounds, not problems in this evidence. Answering
with one of those names nothing on this list.

Reply with strict JSON and nothing else:

{{"primary_issue": "<issue_id, e.g. I1>",
 "concurrent_minor_fixes": ["<issue_id>", "..."],
 "deferred": ["<issue_id>", "..."],
 "why_primary": "<one sentence>",
 "why_these_concurrent": "<one sentence, or \\"\\" if none>"}}
"""


def _items_text(items: list[dict]) -> str:
    """The whole frozen checklist, each item with its standing.

    A behaviour the task asks for is not a defect, however odd it looks.
    This game requires the elevator's buttons to rearrange as it
    malfunctions, and a grounder that could not see that requirement
    reported the rearranging as a problem to repair.
    """
    out = []
    for i in items:
        st = f" [{i['status']}]" if i.get('status') else ''
        out.append(f"  {i['id']}{st}  {i['requirement']}")
    return '\n'.join(out) or '  (none)'


def ground(evidence: str, items: list[dict], direction: str, question: str,
           *, frames: list | None = None, static_view: str = '',
           model: str | None = None) -> dict:
    """Stage A. Evidence in, structured issues out.

    `frames` are (label, path) pairs and they matter more than their cost
    suggests. This stage decides whether something is a defect, and a whole
    class of defects -- clipped text, an overlap, a wrong scale -- cannot be
    established from a play agent's sentence about the session. A grounder that
    never sees a pixel does not report those badly; it never reports them at
    all, silently, and the round that would have batched them as small
    concurrent fixes finds nothing to batch.
    """
    import rsigame.verify.post_verify as PV
    from rsigame.agent.evolve.verifier_agent import _data_uri

    frames = [(l, f) for l, f in (frames or []) if Path(f).is_file()]
    note = (f'You also have {len(frames)} frames from the same session, each '
            f'labelled with the input that produced it. They are the record; '
            f'the account above is a reading of it.'
            if frames else
            'No frames were kept for this session, so the account above is all '
            'there is. Judge only what it supports.')
    # COUNTS OFF THE TREE, not a reading of it. Two whole classes of defect
    # are invisible to a play session however long: how much content exists at
    # all, and whether the screen is drawn from authored art or from
    # primitives. A session shows what was reached; it cannot establish what
    # there was to reach.
    sv = (f'====================\nWHAT THE BUILD CONTAINS\n'
          f'====================\n{static_view}\n\nThese are counts read '
          f'mechanically off the source, not something anyone watched. They '
          f'are evidence about content and about how the screen is drawn, and '
          f'nothing else -- do not read a verdict into a number.\n\n'
          if static_view else '')
    body = _GROUND.format(items=_items_text(items), direction=direction,
                          question=question or '(none recorded)',
                          evidence=evidence or '(no evidence was collected)',
                          frames_note=note, static_note=sv)
    content = [{'type': 'text', 'text': body}]
    for label, fp in frames:
        content.append({'type': 'image_url', 'image_url': {'url': _data_uri(Path(fp))}})
        content.append({'type': 'text', 'text': f'({label})'})
    parsed, _ = PV._ask([{'role': 'user', 'content': content}],
                        model=model or os.environ.get('RSIGAME_MODELS_LOOP_MODEL'), timeout=180.0)
    if not isinstance(parsed, dict):
        return {'issues': [], 'error': 'the reading returned nothing after three tries'}

    known = {i['id'] for i in items}
    out, bad = [], []
    for n, it in enumerate(parsed.get('issues') or [], start=1):
        if not isinstance(it, dict):
            bad.append(f'issues[{n}] not an object')
            continue
        st = it.get('status') if it.get('status') in STATUS else None
        if st is None:
            bad.append(f'issues[{n}].status={it.get("status")!r}')
            continue
        refs = [str(r)[:120] for r in (it.get('evidence_refs') or [])]
        if st == 'confirmed_gap' and not refs:
            # A confirmed gap with nothing to point at is the failure this
            # stage exists to prevent, so it is demoted rather than dropped:
            # the observation may still be worth carrying as unverified.
            bad.append(f'issues[{n}] claimed confirmed_gap with no evidence ref')
            st = 'unverified'
        out.append({
            'issue_id': it.get('issue_id') or f'I{n}',
            'issue': str(it.get('issue') or '')[:300],
            'status': st,
            'direction': it.get('direction') if it.get('direction') in DIRECTIONS else None,
            'importance': it.get('importance') if it.get('importance') in IMPORTANCE else 'unknown',
            'evidence_refs': refs,
            'note': str(it.get('note') or '')[:300],
            'closes': [x for x in (it.get('closes') or []) if x in known],
        })
    return {'issues': out, 'schema_problems': bad}


def rank(issues: list[dict], direction: str, question: str, history: str,
         budget: int, *, model: str | None = None) -> dict:
    """Stage B. Only confirmed gaps are offered, and only they can be chosen."""
    import rsigame.verify.post_verify as PV

    eligible = [i for i in issues if i['status'] == 'confirmed_gap']
    blocked = [i for i in issues if i['status'] != 'confirmed_gap']
    if not eligible:
        return {'primary_target': None, 'concurrent_minor_fixes': [],
                'deferred': [i['issue_id'] for i in issues],
                'why_primary': 'no issue is supported by evidence',
                'blocked_unverified': [i['issue_id'] for i in blocked],
                'schema_problems': []}

    lines = '\n'.join(
        f"  {i['issue_id']}  [{i['importance']}/{i['direction'] or '?'}] {i['issue']}\n"
        f"        seen: {i['note']}   ({', '.join(i['evidence_refs'])})"
        for i in eligible)
    body = _RANK.format(direction=direction, question=question or '(none recorded)',
                        issues=lines, history=history or '  (nothing recorded)',
                        budget=budget)
    from rsigame.controller.stage_brief import text as _stage_brief
    sb = _stage_brief()
    if sb:
        # Stage B only. Whether a problem exists (Stage A) never sees the brief:
        # a director's opinion must not turn into something the frames confirmed.
        # The code gate below still refuses anything that is not a confirmed gap.
        body = body.replace('====================\nCONFIRMED PROBLEMS', sb + (
            '\nUse it to choose AMONG the confirmed problems below: when more than one could '
            'be the primary target, prefer the one that serves the stage objective. It '
            'cannot make a problem eligible that is not listed.\n\n'
            '====================\nCONFIRMED PROBLEMS'), 1)
    parsed, _ = PV._ask([{'role': 'user', 'content': body}],
                        model=model or os.environ.get('RSIGAME_MODELS_LOOP_MODEL'), timeout=120.0)
    if not isinstance(parsed, dict):
        return {'primary_target': None, 'concurrent_minor_fixes': [], 'deferred': [],
                'error': 'the reading returned nothing after three tries'}

    ok = {i['issue_id'] for i in eligible}
    bad = []

    def keep(v):
        if v in ok:
            return v
        if v is not None:
            bad.append(f'{v!r} is not an eligible confirmed gap')
        return None

    primary = keep(parsed.get('primary_issue') or parsed.get('primary_target'))
    if primary is None and eligible:
        # A confirmed problem that reaches this stage and is not chosen is a
        # problem dropped. The first run did exactly that: it answered with a
        # ledger id, validation refused it, and the round's one real defect
        # left the packet entirely while the round spent itself on art. If the
        # answer names nothing usable, the most important eligible issue is
        # the primary.
        order = {'major': 0, 'medium': 1, 'minor': 2, 'unknown': 3}
        primary = sorted(eligible, key=lambda i: order.get(i['importance'], 3))[0]['issue_id']
        bad.append(f'no usable primary was named; took {primary}')
    conc, seen = [], {primary}
    for v in (parsed.get('concurrent_minor_fixes') or []):
        v = keep(v)
        if v and v not in seen:
            conc.append(v)
            seen.add(v)
    if len(conc) > MAX_CONCURRENT:
        bad.append(f'{len(conc)} concurrent fixes proposed; kept {MAX_CONCURRENT}')
        conc = conc[:MAX_CONCURRENT]
    deferred = [i['issue_id'] for i in eligible if i['issue_id'] not in seen]
    return {'primary_target': primary, 'concurrent_minor_fixes': conc,
            'deferred': deferred,
            'why_primary': str(parsed.get('why_primary') or '')[:300],
            'why_these_concurrent': str(parsed.get('why_these_concurrent') or '')[:300],
            'blocked_unverified': [i['issue_id'] for i in blocked],
            'schema_problems': bad}


def packet(selection: dict, issues: list[dict], budget_tools: int = 26,
           art: dict | None = None) -> str:
    """What the repair agent is told.

    The wording matters more than it looks. The existing planner tells the
    agent that anything after the first goal "keeps its place and comes back
    next round", which is an instruction to drop it, and rounds duly dropped
    it. Concurrent fixes are secondary in priority and still expected: the only
    grounds for skipping one are that doing it would widen the round or put the
    primary at risk, and a skip has to be said out loud so the next round knows
    it was a decision and not an oversight.

    `art` is the polish reader's standing finding and it does NOT come through
    the issue gate. That gate exists to stop the loop repairing what nobody has
    seen fail; art that is merely plain has not failed, and there is no
    checklist item for it to be a gap against, so it would be marked
    `unverified` and blocked -- the gate working correctly on a question it was
    not built for. It sits in its own section: ungated, uncapped, and not
    conditional on the primary being finished, because the polish reader has
    never once called this work finished in thirty games and a round that only
    does it after everything else never does it.
    """
    by = {i['issue_id']: i for i in issues}

    def block(iid: str) -> str:
        i = by.get(iid)
        if not i:
            return ''
        refs = ', '.join(i['evidence_refs']) or 'no reference recorded'
        return f"{iid}: {i['issue']}\n     seen: {i['note']}\n     evidence: {refs}"

    p = selection.get('primary_target')
    conc = selection.get('concurrent_minor_fixes') or []
    art_block = _art_text(art)

    if not p:
        head = ('NOTHING IS ELIGIBLE FOR REPAIR THIS ROUND\n\n'
                'No problem in the evidence is confirmed. Repairing something '
                'nobody has seen fail teaches nothing, so this round does not '
                'chase one.')
        if not art_block:
            return head + ' It collects evidence instead.'
        return (head + '\n\n' + art_block + '\n\nREPAIR POLICY\n\n'
                'Nothing in the evidence is a confirmed defect this round, so '
                'the work above is what there is to do. It is not a '
                'consolation prize: it is the other half of what this game is '
                'judged on. Say which files you added or replaced.\n\n'
                f'You have about {budget_tools} tool calls.')

    out = ['WHAT THE EVIDENCE SAYS IS BROKEN', block(p)]
    if conc:
        out += ['', 'ALSO BROKEN, SMALLER'] + [block(c) for c in conc]
    if art_block:
        out += ['', art_block]

    two = bool(art_block)
    policy = []
    if two:
        policy.append(
            '1. The sections above come from two different questions -- does '
            'this game work, and does it look and feel finished -- and they '
            'are worth the same. One is not the round and the other its '
            'leftovers. A game that works and looks unmade is as unfinished '
            'as a beautiful one that does not respond.')
        policy.append(
            '2. Do as much of both as the budget carries. If it will only '
            'carry one, choose, and say which you chose and why -- that is a '
            'decision the next round can read. Doing one of them properly '
            'beats half-doing everything.')
        n = 3
    else:
        policy.append('1. Repair what is above. Doing one thing properly '
                      'beats half-doing several.')
        n = 2
    if conc:
        policy.append(
            f'{n}. The smaller defects are grounded in the same evidence and '
            'are not suggestions. Repair them in the same sitting wherever '
            'that can be done locally and safely.')
        n += 1
    if two:
        policy.append(
            f'{n}. On the art: there is no cap on it, and how far to take it '
            'is yours to judge with the code in front of you. Prefer not to '
            'disturb what the defect above will be re-checked against, and '
            'make sure the project still builds and every new asset imports -- '
            'a texture that fails to load leaves the build passing and the '
            'screen empty.')
        n += 1
    policy.append(
        f'{n}. If you skip any of it, say which and why, and say which files '
        'you added or replaced. A skip that is written down is a decision; a '
        'silent one cannot be told from an oversight. The next round compares '
        'this build against this one frame by frame, and it should be able to '
        'see that a screenful of differences was your work.')

    out += ['', 'REPAIR POLICY', '', '\n\n'.join(policy), '',
            f'You have about {budget_tools} tool calls for all of it.']
    return '\n'.join(out)


def _art_text(art: dict | None) -> str:
    """The polish finding, phrased as work available rather than as a fault."""
    if not art or art.get('error'):
        return ''
    goal = str(art.get('goal') or '').strip()
    if not goal:
        goal = ' '.join(str(v) for v in (art.get('main_issue') or {}).values()
                        if v).strip()
    if not goal:
        return ''
    hr = art.get('headroom') or {}
    lines = ['WHAT THE POLISH READER SAYS IS UNFINISHED', '',
             'A second reader looks at every build and asks a different',
             'question: are the things on screen drawn, or are they shapes',
             'standing in for drawings, and does an action land like an event.',
             'It has not yet called this work finished on any build.', '',
             'What it says is worth doing now:', f'  {goal}']
    if hr:
        lines.append('  headroom: ' + ', '.join(f'{k} {v}' for k, v in hr.items()))
    return '\n'.join(lines)
