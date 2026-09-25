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
"""Controller-Plan: what concrete question is worth resolving next.

The controller is told the macro direction as a conclusion, never as an
argument, because its job is the question INSIDE that direction and a monitor's
reasoning would invite it to relitigate the choice.

Two things it must not produce, and both are easy to slip into:

  * a repair instruction. "Improve level progression" is a decision about what
    to change; the controller's output is a decision about what to find out.
    The evidence has not been collected yet, so anything phrased as a fix is a
    guess dressed as a plan.
  * a restatement of the rubric. "Check completeness" cannot be answered by any
    particular observation, so it cannot direct an exploration.

Target identity is a selection, never a generation: the controller is handed
the open targets by id and either picks one or says `new`, and the harness
allocates. An id outside the list is a schema violation, dropped and logged,
so the ledger cannot be corrupted by the model inventing a name.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUN))

ACTIONS = ('reuse', 'explore', 'escalate')
MODES = ('replay_existing_demo', 'extend_existing_demo',
         'custom_targeted_probe', 'none')
STATUS = ('sufficient', 'insufficient')

# Measured: a healthy call answers in 6-30 seconds. One in the M2 run took 422,
# and one here never answered at all. Three tries at 120s each bounds a case at
# six minutes instead of forever.
TIMEOUT_S = 120.0

_PROMPT = """You plan what to investigate next in a game that is being
developed one round at a time.

You do not write code, propose patches, or say what should be changed. Another
agent does that, after the evidence you ask for has been collected. Your output
is a decision about what to FIND OUT.

{direction_block}

====================
{items_header}
====================
{items}

{probed}
====================
WHAT THIS BUILD DID WHEN PLAYED
====================
{current}

{historical}====================
DEMOS AVAILABLE
====================
Each demo is a fixed input script. What it attempts is listed; what it covers
is not, because a script that intends to reach a state does not always reach
it. Where a demo has already been run against the build you are looking at,
what happened is listed too.

{demos}
====================
OPEN TARGETS
====================
{targets}

====================
BUDGET
====================
Round {used} of {total} spent, {remaining} remaining.
Prefer existing evidence when it already settles the matter. Otherwise prefer
an existing demo. Open-ended exploration is the expensive option.

====================
WHAT TO DECIDE
====================
One primary question. It must be:
  - inside the direction above;
  - grounded in the state and evidence above, not in what a game like this
    usually has;
  - answerable by watching the game: someone replays it and looks. Not by
    reading the source, which nobody will do for you;
  - concrete enough that you can say what that observation would be;
  - useful for deciding whether something needs repairing.

Not a question: "Can the game be improved?" -- nothing observable answers it.
Not a question: "Improve the HUD layout." -- that is an instruction to change
something, and nothing has been observed yet.
Not a question: "Why does LEFT fail, and what does the input path look like?"
-- the second half asks about source code. Nobody is going to read the code to
answer you; someone is going to play the game and watch. Ask what a viewer
would see, not why the program does it.

Then choose one action:
  reuse      the evidence above already settles what the problem is. No new
             observation is needed before deciding what to repair.
  explore    the evidence is missing or ambiguous, and a specific observation
             would settle it.
  escalate   there is no grounded unresolved question left in this direction.
             Say so rather than inventing one.

Say what the extension should achieve, not which keys to press. Someone else
plays the game and works that out with the screen in front of them; you are
deciding what needs finding out, and a keystroke you picked without seeing the
screen would be a guess they then have to work around.

If exploring, choose the cheapest mode that can answer the question:
  replay_existing_demo    a demo already exercises what you need to see
  extend_existing_demo    a demo reaches the state but stops before the answer
  custom_targeted_probe   no demo can reach it. The expensive option.

On targets: a target is a thing being repaired across rounds, and its ids are
the TGT.. ones listed above -- NOT the requirement ids (T07, G3) from the
checklist, which name something else entirely. If this question continues work
on one of the open targets, give that target's id. If it is new work, write
"new" and an id will be assigned to it. Never write an id yourself.

If a question has already been asked several rounds running without being
settled, asking it again the same way is not a plan. Either say what would be
different this time, or pick something else.

Reply with strict JSON and nothing else:

{{"round_kind": "repair|art",
 "art_focus": "feedback|visual|none",
 "action": "reuse|explore|escalate",
 "development_question": {{"question": "<one question>",
   "why_now": "<what in the evidence above makes this the thing to ask>",
   "related_state_items": ["<requirement ids>"]}},
 "target_ref": "<a TGT.. id from OPEN TARGETS, or \\"new\\">",
 "evidence_assessment": {{"status": "sufficient|insufficient",
   "relevant_evidence": "<what above you are relying on>",
   "missing_information": "<what you still do not know, or \\"\\" if none>"}},
 "exploration_plan": {{"mode": "replay_existing_demo|extend_existing_demo|custom_targeted_probe|none",
   "demo_id": "<a demo id above, or \\"\\">",
   "focus": "<what to watch for>",
   "extension": "<what to do beyond the existing script, or \\"\\">",
   "stop_when": "<what ends the observation>"}}}}
"""


def _items_text(rows: list[dict]) -> str:
    if not rows:
        return '  (no requirement in this direction)'
    out = []
    for i in rows:
        tail = ''
        if i.get('last_change_round'):
            tail = f"  (last changed round {i['last_change_round']})"
        out.append(f"  {i['id']:5s} [{i['status']}] {i['requirement']}{tail}")
    return '\n'.join(out)


def _demos_text(demos: list[dict]) -> str:
    """One entry per demo: what it attempts, what it sends, what it did."""
    out = []
    for d in demos:
        flag = (f" (launched with --scenario {d['scenario_flag']})"
                if d.get('scenario_flag') else '')
        out.append(f"  {d['demo_id']}{flag}")
        if d.get('description'):
            out.append(f"    {d['description']}")
        acts = ', '.join(d['actions'][:12]) + (' ...' if len(d['actions']) > 12 else '')
        out.append(f"    {d['num_inputs']} input(s), {d['waits']} wait(s)"
                   f"{': ' + acts if acts else ''}")
        if d.get('replayed_on_this_build'):
            out.append(f"    ALREADY RUN against this build (round "
                       f"{d['replayed_round']}). Running it again records the "
                       f"same behaviour a second time.")
            if d.get('outcome'):
                out.append(f"    What happened: {d['outcome']}")
            else:
                out.append("    No account of that run was written down.")
        for b in (d.get('still_broken') or [])[:2]:
            out.append(f"    STILL BROKEN after an earlier repair: {b}")
        out.append('')
    return '\n'.join(out).rstrip() or '  (none)'


def render(inp: dict, open_targets: list[dict] | None = None) -> str:
    cur = inp['evidence']['current']
    cur_txt = (f"round {cur['round']}, demos replayed: "
               f"{', '.join(cur['replayed_demos']) or '(none)'}\n\n"
               f"{cur['closing_note'] or '(no play report for this round)'}")
    if cur['reached']:
        cur_txt += '\n\nStates reached:\n' + '\n'.join(f'  - {x}' for x in cur['reached'])
    if cur['not_reached']:
        cur_txt += '\n\nStates NOT reached:\n' + '\n'.join(f'  - {x}' for x in cur['not_reached'])

    hist = inp['evidence']['historical']
    hist_txt = ''
    if hist:
        hist_txt = ('====================\nOLDER BUILDS -- CONTEXT ONLY\n'
                    'This describes builds that no longer exist. It is not '
                    'evidence about the current one.\n====================\n'
                    + '\n'.join(f"round {h['round']}: {h['closing_note'][:400]}"
                                for h in hist) + '\n\n')

    pr = inp['state'].get('probed_rounds') or {}
    probed = ''
    if pr:
        probed = ('ALREADY PROBED BY NAME\n'
                  + '\n'.join(f'  {k}: rounds {v}' for k, v in sorted(pr.items()))
                  + '\n\n')

    tg = open_targets or []
    targets = ('\n'.join(f"  {t['id']}  ({t['attempts']} attempt(s), "
                         f"{t['verified_successes']} verified) {t['statement']}"
                         for t in tg) or '  (none open -- any target is new)')

    d = inp['outer_directive']
    if d.get('next_direction'):
        block = ('The macro direction below was decided elsewhere and is not '
                 'yours to revisit.\nYour question must sit inside it.\n\n'
                 '====================\nDIRECTION FOR THIS ROUND\n'
                 '====================\n'
                 + f"{d['next_direction']}   (action: {d['action']}, headroom: "
                   f"{d['remaining_headroom']})"
                 + (f"\n{d['reason']}" if d.get('reason') else ''))
    else:
        # NOBODY DECIDED A DIRECTION THIS ROUND. The requirements below are the
        # whole checklist rather than one direction's slice, and the choice of
        # what kind of round this is belongs here too. This is the arm that
        # moves that decision from a global reader looking at a checkpoint
        # every third round to a local one looking at the build in front of it
        # every round; the two must have the SAME actions available or the
        # comparison is between capabilities rather than between where the
        # decision lives.
        block = ('NO DIRECTION WAS SET FOR THIS ROUND. Decide for yourself '
                 'what this build most needs.\n\n'
                 'You also choose what KIND of round this is:\n\n'
                 '  repair  the full chain -- investigate a named defect, then '
                 'fix it. Use it when\n'
                 '          something the task asks for is broken, missing, or '
                 'unverified, and a\n'
                 '          question about the frames would settle it.\n'
                 '  art     one pass that generates and wires assets and '
                 'effects. Use it when the\n'
                 '          game works but looks or feels unfinished -- '
                 'placeholder shapes, no\n'
                 '          response to input, unreadable text. Say whether it '
                 'is about `feedback`\n'
                 '          (the game answering the player) or `visual` (how it '
                 'looks).\n\n'
                 'An `art` round does not investigate: it does not get a '
                 'development question or an\n'
                 'exploration plan, so leave those fields as their emptiest '
                 'legal value if you\n'
                 'choose it. Choose on the evidence below, not on a rotation.')
        rd = inp.get('reader')
        if rd:
            hr = rd.get('headroom') or {}
            mi = rd.get('main_issue') or {}
            lines = []
            for k, lab in (('visual', 'visual   '), ('feedback', 'feedback')):
                lines.append(f"  {lab}  headroom {hr.get(k, '?')}")
                if mi.get(k):
                    lines.append(f"      {str(mi[k])[:420]}")
            block += ('\n\n====================\nHOW FINISHED IT LOOKS AND '
                      'FEELS, on the current build\n====================\n'
                      'A reader looked at this build\'s frames and judged two '
                      'things it cannot\nread off the requirements. This is the '
                      'evidence for an `art` round;\nthe requirements below are '
                      'the evidence for a `repair` one.\n\n'
                      + '\n'.join(lines))
        else:
            block += ('\n\nNo presentation reading is available for this build, '
                      'so treat how it\nlooks and feels as UNKNOWN rather than '
                      'as fine.')

    b = inp['budget']
    from rsigame.controller.stage_brief import text as _stage_brief
    sb = _stage_brief()
    if sb:
        block = sb + '\n' + block
    return _PROMPT.format(
        direction_block=block,
        items_header=('REQUIREMENTS IN THIS DIRECTION' if d.get('next_direction')
                      else 'ALL REQUIREMENTS'),
        items=_items_text(inp['state']['items']),
        probed=probed, current=cur_txt, historical=hist_txt,
        demos=_demos_text(inp['demos']), targets=targets,
        used=b['used'], total=b['total'], remaining=b['remaining'])


def plan(inp: dict, open_targets: list[dict] | None = None, *,
         model: str | None = None) -> dict:
    import rsigame.verify.post_verify as PV
    body = render(inp, open_targets)
    out = {'game': inp['game'], 'corpus': inp['corpus'], 'round': inp['round'],
           'direction': inp['outer_directive'].get('next_direction'),
           'prompt_chars': len(body)}
    parsed, _ = PV._ask([{'role': 'user', 'content': body}],
                        model=model or os.environ.get('RSIGAME_MODELS_LOOP_MODEL'),
                        timeout=TIMEOUT_S)
    if not isinstance(parsed, dict):
        out['error'] = 'the reading returned nothing after three tries'
        return out
    out.update(validate(parsed, inp, open_targets))
    rk = parsed.get('round_kind')
    out['round_kind'] = rk if rk in ('repair', 'art') else 'repair'
    af = parsed.get('art_focus')
    out['art_focus'] = af if af in ('feedback', 'visual') else 'visual'
    return out


def validate(d: dict, inp: dict, open_targets: list[dict] | None) -> dict:
    bad = []

    def enum(v, allowed, where):
        if v in allowed:
            return v
        bad.append(f'{where}={v!r}')
        return None

    q = d.get('development_question') or {}
    ea = d.get('evidence_assessment') or {}
    ep = d.get('exploration_plan') or {}

    known_items = {i['id'] for i in inp['state']['items']}
    rel = [x for x in (q.get('related_state_items') or []) if x in known_items]
    unknown_items = [x for x in (q.get('related_state_items') or [])
                     if x not in known_items]
    if unknown_items:
        bad.append(f'related_state_items not in this direction: {unknown_items}')

    demo_ids = {x['demo_id'] for x in inp['demos']}
    demo = ep.get('demo_id') or ''
    if demo and demo not in demo_ids:
        bad.append(f'demo_id={demo!r} is not one of the demos')
        demo = ''

    # target_ref is a selection, never a generation
    open_ids = {t['id'] for t in (open_targets or [])}
    tr = d.get('target_ref')
    if tr not in open_ids and tr != 'new':
        bad.append(f'target_ref={tr!r} is neither an open target nor "new"')
        tr = None

    return {
        'action': enum(d.get('action'), ACTIONS, 'action'),
        'question': str(q.get('question') or '')[:400],
        'why_now': str(q.get('why_now') or '')[:400],
        'related_state_items': rel,
        'target_ref': tr,
        'evidence_status': enum(ea.get('status'), STATUS, 'evidence.status'),
        'relevant_evidence': str(ea.get('relevant_evidence') or '')[:300],
        'missing_information': str(ea.get('missing_information') or '')[:300],
        'mode': enum(ep.get('mode'), MODES, 'mode'),
        'demo_id': demo,
        'focus': str(ep.get('focus') or '')[:300],
        'extension': str(ep.get('extension') or '')[:300],
        'stop_when': str(ep.get('stop_when') or '')[:300],
        'schema_problems': bad,
    }
