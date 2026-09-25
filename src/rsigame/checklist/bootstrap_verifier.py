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
"""What one session actually shows about the checklist. PART 2, and only that.

Part 1 says what the task asked for. This says, for each of those items, what
a single broad play session lets you conclude -- and, much more often, what it
does not. It produces the initial `checklist_state` and a handful of general
checks. It does not track state across rounds, does not run stages, does not
probe, and is not wired into the loop.

WHY THIS IS A SEPARATE CALL AND NOT A BIGGER `self_critique`. `self_critique`
is the behaviour of all four arms on disk; changing it makes the next arm
incomparable to every result already scored. And the two questions pull in
opposite directions -- "what is wrong with this game" wants a model ranging
freely, "is this one item present" wants it pinned to one line at a time. We
have measured what happens when those are merged: adding the long `_opening()`
preamble to the play prompt cost it click accuracy outright.

THE ONLY HONEST ANSWER IS USUALLY `unverified`. Measured on the five EV1 r0
traces: 9 to 13 steps, 5 to 9 things the agent said it reached, against a
15-to-24-item checklist. A session that size cannot decide twenty requirements.
If this module comes back having decided most of them, it is not perceptive,
it is guessing -- so the share of `unverified` is the first thing the
acceptance run looks at, and a low one fails.

GROUNDING IS CHECKED, NOT REQUESTED. Every `satisfied` and every
`confirmed_gap` has to name the steps it rests on and say whether it read the
claim in the record or off a frame; a claim from the record carries the words,
and those words are matched back against the record with the same check Part 1
uses on quotes. Ungrounded does not mean rejected -- it means `unverified`,
which is the failure direction that costs nothing.

AND "I NEVER GOT THERE" IS NOT "IT IS NOT THERE". This is the specific error
that cost horror-tape-archive four rounds: the agent's own misses became "the
hit area is offset ~+127px" and repair went to work on a button that was fine.
The session's `not_reached` list is part of the record, so a defect whose
evidence traces back into that list is caught exactly -- by where the matched
words sit -- rather than by guessing at a similarity threshold. That guess is
what broke the first evidence-validity matcher and it is not repeated here.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rsigame.checklist.task_checklist as TC   # noqa: E402

RUN = Path(__file__).resolve().parent
MAX_FRAMES = int(os.environ.get('RSIGAME_EVIDENCE_MAX_FRAMES') or 12)
STATUSES = ('satisfied', 'confirmed_gap', 'unverified')


# A SMALL FIXED LIBRARY, NOT FREE INVENTION.
#
# Handoff 4.2 asks for three to five general checks. Letting the model write
# its own drifts straight back into what `self_critique` already produces --
# "the game should feel more responsive" -- and, worse, produces a different
# set every round and every arm, when Part 3 needs these to be persistent
# objects that can be carried forward and compared.
#
# So the model picks, it does not write. Eight entries: the four the handoff
# names, plus four that are true of any small game and are answerable from a
# play trace. None of these came from the rubric -- the rubric is held out and
# has never been read into this file -- they came from the handoff's examples
# and from what the traces on disk actually stumble over.
GENERAL_CHECKS = {
    'G1': 'core navigation must not dead-end: from any screen the player can '
          'get back to play or to a menu',
    'G2': 'the controls the game advertises must respond',
    'G3': 'a failure state must allow recovery or restart without relaunching',
    'G4': 'an important state change must be visible when it happens',
    'G5': 'the build must reach a playable state on launch without manual '
          'intervention',
    'G6': 'progression the game shows the player must be attainable, not '
          'permanently locked',
    'G7': 'on-screen text must be readable: not clipped, not overlapping, not '
          'off-screen',
    'G8': 'the player must be able to tell what to do next without outside '
          'instructions',
    # WHY THIS ONE IS HERE AND NOT IN THE CHECKLIST.
    #
    # The checklist is extracted from the task document, so it can only hold
    # what the task says. Measured across the bench: 73 graded requirements in
    # 71 of the 141 games ask for exactly this -- "the HUD clearly shows fuel
    # or heat, velocity or thrust state, active contract, salvage value,
    # credits", "HUD shows essential information: goo supply count by type,
    # blobs collected toward quota" -- and the task documents for those same
    # games say nothing about it. Checked on three of them: 0, 0 and 1 lines
    # mentioning a readout, against rubric requirements listing six values
    # apiece. So it is not something extraction is missing; it is not there to
    # miss.
    #
    # That is what this library is for. Every other entry is the same shape: a
    # property any playable game has that no task bothers to state. G7 is the
    # nearest and is not the same question -- it asks whether text that IS on
    # screen can be read, not whether the numbers the game runs on are on
    # screen at all.
    'G9': 'the numbers the game runs on must be on screen while they matter: '
          'whatever the player is spending, earning, losing or racing against '
          'should be readable without opening a menu',
}


_PROMPT = """Below is a checklist of what a game's task asked for, and the
record of ONE session somebody played of the build. For each checklist item,
say what this session actually shows.

Three answers, and no others:

  satisfied      the session shows this working or present
  confirmed_gap  the session shows this missing, broken, or materially short
  unverified     the session does not show enough to say

`unverified` is the right answer most of the time and you should expect to use
it for more than half the list. This session is about a dozen steps long and
the checklist has {n_items} items; it cannot have touched them all. NOT SEEING
SOMETHING IS NOT SEEING IT MISSING.

In particular: the section "WHAT THEY SAID THEY COULD NOT REACH" is a list of
places the player never got to. Nothing in it is evidence that anything is
missing from the game. An item that lives behind one of those is `unverified`.

GROUND IT OR DOWNGRADE IT. For `satisfied` and `confirmed_gap` you must give:

  cites   the step numbers it rests on, e.g. [7, 9] -- real steps from below
  basis   "trace" if the record states it in words, "frame" if you read it off
          one of the frames
  quote   basis "trace": the words from the record, COPIED EXACTLY. They are
          checked against the record.
          basis "frame": leave it empty.
  note    one sentence, what you saw

If you cannot fill those in, the answer is `unverified`. That is not a failure;
it is the correct answer, and it costs nothing. A wrong `confirmed_gap` costs a
whole repair round on something that was never broken.

THEN PICK 3 TO 5 GENERAL CHECKS from this fixed list -- by id, do not invent
new ones -- that are clearly worth asking about THIS game, and answer each the
same way with the same grounding.

{library}

Reply with JSON and nothing else:

{{"statuses": [{{"id": "T01", "status": "...", "cites": [], "basis": "",
                "quote": "", "note": ""}}],
  "general": [{{"id": "G2", "status": "...", "cites": [], "basis": "",
                "quote": "", "note": ""}}]}}

THE CHECKLIST

{checklist}

THE SESSION

{session}
"""


# WHAT THE MERGED READING ADDS.
#
# The loop used to spend three calls on one decision: `PROBE.pick` chose what
# PLAY should find out, this reading judged the checklist, and
# `PROBE.repair_goal` turned the open gaps into an instruction. They ran at
# three different moments on three different inputs, so they were never
# duplicated work -- but they were three views of the same question, and the
# state moved between the second and the third, which is why the loop had to
# call `SC.plan` twice and rewrite the round's own label afterwards.
#
# Folded in here, the reader answers all three at the moment it has the most
# evidence: it has just read the session. `next_probe` is for the NEXT round,
# which is what removes the pre-PLAY call entirely.
_MERGE_TAIL = """

THEN TWO DECISIONS, both from what you just read.

REPAIR GOAL. Of the items you marked `confirmed_gap` above, and only those,
write ONE improvement goal for this round: the single coherent change worth
making now. If several gaps are one fault, say so and treat them as one. If
they are unrelated, take the one that matters most and leave the rest -- they
keep their place and come back. Do not restate the list; say what should
become true and where the evidence points. The person acting on this can read
the code but cannot re-run the session.

{stage_line}

If you marked nothing `confirmed_gap` in that set, leave the goal empty. An
empty goal is a real answer and costs nothing; an invented one costs a whole
repair round on something that was never broken.

NEXT PROBE. Write one instruction for the NEXT session, aimed at what you
still could not answer. Name the input to make or the screen to reach and what
to watch: "Check whether the plane rotates" is weak, "in a sortie, press Left
then Right for about a second each and report whether the plane's heading
changes" can be carried out. Prefer items that share a place to look; a
session that has to boot two states spends half its steps travelling.

{scenario_line}

Add these two keys to the SAME JSON object, alongside `statuses` and
`general`:

  "repair_goal": {{"goal": "", "items": [], "why": "one sentence"}},
  "next_probe":  {{"ask": "", "items": [], "scenario": "", "why": "one sentence"}}
"""

def trace_text(exp: dict) -> tuple[str, dict]:
    """The session as text, plus where each section sits in it.

    Returns the text and, per section, the half-open span of token indices it
    occupies. The spans are the point: a claimed defect whose evidence matches
    inside `not_reached` is the agent citing its own failure to arrive as proof
    of absence, and that is decidable exactly here instead of approximately by
    comparing item text to excuse text.
    """
    parts, spans, cur = [], {}, 0

    def add(name, body):
        nonlocal cur
        parts.append(body)
        n = len(TC._tokens(body))
        spans[name] = (cur, cur + n)
        cur += n

    lines = []
    for s in exp.get('steps') or []:
        a = s.get('action') or ''
        args = ' '.join(f'{k}={v}' for k, v in (s.get('args') or {}).items())
        bits = [f"step {s.get('n')}: {a} {args}".rstrip()]
        for k in ('thought', 'note'):
            if s.get(k):
                bits.append(f'    {k}: {s[k]}')
        if s.get('ok') is False:
            bits.append('    this step did not succeed')
        lines.append('\n'.join(bits))
    add('steps', 'WHAT THE SESSION DID\n\n' + '\n'.join(lines))

    add('reached', '\n\nWHAT THEY SAID THEY REACHED\n'
        + '\n'.join(f'  - {x}' for x in (exp.get('reached') or [])))
    add('not_reached', '\n\nWHAT THEY SAID THEY COULD NOT REACH\n'
        + '\n'.join(f'  - {x}' for x in (exp.get('not_reached') or [])))
    if exp.get('closing_note'):
        add('closing', f"\n\nTHEIR CLOSING NOTE\n  {exp['closing_note']}")
    return ''.join(parts), spans


def _library() -> str:
    return 'THE GENERAL CHECKS\n\n' + '\n'.join(
        f'  {k}  {v}' for k, v in GENERAL_CHECKS.items())


def apply_gates(raw: dict, items: list, exp: dict) -> dict:
    """Keep only what the session grounds; everything else is `unverified`.

    Nothing is dropped here. Handoff 5 is explicit that `unverified` is not
    `confirmed_gap` -- it is an evidence-collection task -- so a claim that
    fails a gate is demoted to it and the reason is recorded on the item. That
    makes the demotions countable, which is how we find out whether the gates
    are doing anything or just decorating the output.
    """
    text, spans = trace_text(exp)
    hay = TC._tokens(text)
    nr = spans.get('not_reached', (0, 0))
    real_steps = {s.get('n') for s in (exp.get('steps') or [])}
    shown = {n for n, _ in _frames(exp)}

    def gate(rec: dict, known: set) -> dict:
        out = {'id': str(rec.get('id') or '').strip(),
               'status': 'unverified',
               'cites': [c for c in (rec.get('cites') or [])
                         if isinstance(c, int)],
               'basis': str(rec.get('basis') or '').strip().lower(),
               'quote': str(rec.get('quote') or '').strip()[:400],
               'note': str(rec.get('note') or '').strip()[:300],
               'gate': ''}
        want = str(rec.get('status') or '').strip().lower()
        if out['id'] not in known:
            out['gate'] = 'no such id'
            return out
        if want not in STATUSES:
            out['gate'] = f'unknown status {want!r}'
            return out
        if want == 'unverified':
            out['status'] = 'unverified'
            return out

        bad = [c for c in out['cites'] if c not in real_steps]
        if not out['cites']:
            out['gate'] = 'cites nothing'
        elif bad:
            out['gate'] = f'cites steps that do not exist: {bad}'
        elif out['basis'] == 'trace':
            score, at = TC.quote_locate(out['quote'], hay)
            out['quote_match'] = round(score, 3)
            if score < TC._QUOTE_MATCH:
                out['gate'] = f'quote is not in the record ({score:.2f})'
            elif want == 'confirmed_gap' and nr[0] <= at < nr[1]:
                # THE ONE THAT MATTERS. The evidence for this defect is a line
                # from the session's own list of places it never got to.
                out['gate'] = 'evidence is a place the session never reached'
        elif out['basis'] == 'frame':
            unseen = [c for c in out['cites'] if c not in shown]
            if unseen:
                out['gate'] = f'no frame from step(s) {unseen} was shown'
        else:
            out['gate'] = f'unknown basis {out["basis"]!r}'

        if not out['gate']:
            out['status'] = want
        return out

    known_t = {it['id'] for it in items}
    seen, statuses = set(), []
    for rec in (raw.get('statuses') or []):
        if not isinstance(rec, dict):
            continue
        g = gate(rec, known_t)
        if g['id'] in known_t and g['id'] not in seen:
            seen.add(g['id'])
            statuses.append(g)
    for it in items:                      # anything the model skipped
        if it['id'] not in seen:
            statuses.append({'id': it['id'], 'status': 'unverified',
                             'cites': [], 'basis': '', 'quote': '', 'note': '',
                             'gate': 'not answered'})
    statuses.sort(key=lambda x: x['id'])

    general, seen_g = [], set()
    for rec in (raw.get('general') or []):
        if not isinstance(rec, dict):
            continue
        g = gate(rec, set(GENERAL_CHECKS))
        if g['id'] in GENERAL_CHECKS and g['id'] not in seen_g:
            seen_g.add(g['id'])
            g['check'] = GENERAL_CHECKS[g['id']]
            general.append(g)
    general = general[:5]
    return {'statuses': statuses, 'general': general}


def _frames(exp: dict) -> list:
    from rsigame.agent.evolve.read_trace import _pick_frames
    return _pick_frames(exp.get('steps') or [], cap=MAX_FRAMES)


def bootstrap(game: str, exp_path, *, checklist: dict | None = None,
              static_view: str = '', model: str | None = None,
              click_evidence: bool | None = None,
              merge: bool = False, stage: str = '',
              stage_items: list | None = None,
              scenarios: list | None = None) -> dict:
    """Initial checklist state from one session. One call, then the gates.

    `static_view` is passed through from the caller rather than built here --
    it needs the game tree, which this module has no business knowing about.
    The task's own document is deliberately NOT sent: the checklist is its
    structured form, and sending both says the same thing twice while inviting
    the model back into "what is wrong with this game".
    """
    from rsigame.agent.evolve.read_trace import click_marker_note
    from rsigame.agent.evolve.verifier_agent import _chat, _data_uri

    # LOAD .env BEFORE READING `RSIGAME_MODELS_LOOP_MODEL`, not after.
    #
    # Nothing in this package auto-loads it; `stage1._load_env` does, and it
    # runs when stage1 is imported -- which happens INSIDE `_chat`, after the
    # model name has already been resolved to None here. The acceptance test
    # never saw this because it builds a static view, and that imports
    # the development loop, which pulls stage1 in early by accident. The sweep passes
    # no static view, imports nothing else, and every one of its 40 calls died
    # on "no model was given to _client()".
    from rsigame.agent.llm import _load_env
    _load_env()

    exp = json.loads(Path(exp_path).read_text())
    cl = checklist if checklist is not None else TC.extract(game)
    items = cl.get('items') or []
    if not items:
        return {'game': game, 'error': 'no checklist items'}
    if exp.get('stopped_by') in ('never_built', 'session_error', 'crashed') \
            or not (exp.get('steps') or []):
        return {'game': game, 'error': f"no play happened "
                f"({exp.get('stopped_by') or 'no steps'})",
                'statuses': [{'id': it['id'], 'status': 'unverified',
                              'cites': [], 'basis': '', 'quote': '',
                              'note': '', 'gate': 'no session'}
                             for it in items], 'general': []}

    session, _ = trace_text(exp)
    frames = _frames(exp)
    body = _PROMPT.format(
        n_items=len(items), library=_library(),
        checklist='\n'.join(f"  {it['id']}  {it['requirement']}"
                            for it in items),
        session=session + (f'\n\n{static_view}' if static_view else '')
        + f"\n\nThe {len(frames)} frames below follow, each labelled with its "
          f"step. " + click_marker_note(exp, default=click_evidence))

    if merge:
        ids = [i['id'] for i in (stage_items or [])]
        body += _MERGE_TAIL.format(
            stage_line=(
                f"Draw the goal ONLY from these items, which are what the "
                f"'{stage}' stage is responsible for right now: "
                f"{', '.join(ids)}. A gap outside that set keeps its place on "
                f"the board and is not this round's job."
                if ids else
                "There is no stage constraint this round: any confirmed gap is "
                "eligible."),
            scenario_line=(
                f"The game can be launched straight into any of these named "
                f"states: {', '.join(scenarios)}. Put one in `scenario` if the "
                f"probe needs it, or leave it empty for normal play."
                if scenarios else
                "This game declares no launchable states; leave `scenario` "
                "empty."))

    content = [{'type': 'text', 'text': body}]
    for n, fp in frames:
        f = Path(fp)
        if not f.is_file():
            continue
        content.append({'type': 'image_url',
                        'image_url': {'url': _data_uri(f)}})
        content.append({'type': 'text', 'text': f'(the frame after step {n})'})

    _prev = os.environ.get('RSIGAME_MODELS_CHAT_REASONING_CAP')
    os.environ['RSIGAME_MODELS_CHAT_REASONING_CAP'] = '1'
    try:
        parsed, _ = _chat([{'role': 'user', 'content': content}],
                          model=model or os.environ.get('RSIGAME_MODELS_LOOP_MODEL'))
    except Exception as exc:
        return {'game': game,
                'error': f'{type(exc).__name__}: {str(exc)[:160]}'}
    finally:
        if _prev is None:
            os.environ.pop('RSIGAME_MODELS_CHAT_REASONING_CAP', None)
        else:
            os.environ['RSIGAME_MODELS_CHAT_REASONING_CAP'] = _prev

    out = apply_gates(parsed or {}, items, exp)
    if merge:
        out.update(_merged_decisions(parsed or {}, out, stage_items,
                                     scenarios))
    by_id = {it['id']: it['requirement'] for it in items}
    for s in out['statuses']:
        s['requirement'] = by_id.get(s['id'], '')
    out.update({'game': game, 'exp': str(exp_path), 'n_items': len(items),
                'n_frames': len(frames), 'model': model,
                'tally': tally(out)})
    return out



def _merged_decisions(raw: dict, gated: dict, stage_items: list | None,
                      scenarios: list | None) -> dict:
    """Keep the goal and the probe only where the reading earned them.

    Both are checked against ground the same way `probe.pick` and
    `probe.repair_goal` checked theirs, and for the same reason: an id nobody
    offered, or a scenario the game does not declare, sends PLAY to boot a
    silent fallback and report on the wrong screen.

    THE GOAL IS CHECKED AGAINST THE GATED STATUSES, not the raw ones. An item
    the evidence gates just demoted to `unverified` is not a confirmed gap, and
    a goal resting on it would be a repair round spent on something this
    session did not show to be broken -- which is the failure the gates exist
    to prevent, arriving by a different door.
    """
    eligible = {i['id'] for i in (stage_items or [])}
    confirmed = {st['id'] for st in (gated.get('statuses') or [])
                 + (gated.get('general') or [])
                 if st.get('status') == 'confirmed_gap'}
    if eligible:
        confirmed &= eligible

    g = raw.get('repair_goal') or {}
    goal_items = [i for i in (g.get('items') or []) if i in confirmed]
    goal = str(g.get('goal') or '').strip()
    repair_goal = ({'goal': goal[:900], 'items': goal_items,
                    'why': str(g.get('why') or '')[:200]}
                   if goal and goal_items else
                   {'goal': '', 'items': [],
                    'why': ('no confirmed gap in this stage'
                            if not confirmed else
                            'the goal cited nothing this reading confirmed')})

    offered = set(scenarios or [])
    known = {i['id'] for i in (stage_items or [])} or {
        st['id'] for st in (gated.get('statuses') or [])}
    n = raw.get('next_probe') or {}
    ask = str(n.get('ask') or '').strip()
    sc = str(n.get('scenario') or '').strip()
    next_probe = ({'ask': ask[:600],
                   'items': [i for i in (n.get('items') or []) if i in known],
                   'scenarios': [sc] if sc in offered else [],
                   'why': str(n.get('why') or '')[:200]}
                  if ask else {'ask': '', 'items': [], 'scenarios': [],
                               'why': 'the reading proposed no next probe'})
    return {'repair_goal': repair_goal, 'next_probe': next_probe}


def tally(res: dict) -> dict:
    n = {k: 0 for k in STATUSES}
    for s in res.get('statuses') or []:
        n[s['status']] = n.get(s['status'], 0) + 1
    tot = max(1, sum(n.values()))
    n['unverified_pct'] = round(100.0 * n['unverified'] / tot)
    n['demoted'] = sum(1 for s in res.get('statuses') or [] if s.get('gate')
                       and s['gate'] != 'not answered')
    n['unanswered'] = sum(1 for s in res.get('statuses') or []
                          if s.get('gate') == 'not answered')
    return n


def render(res: dict) -> str:
    out = [f"{res.get('game')}   {res.get('n_items')} items, "
           f"{res.get('n_frames')} frames"]
    t = res.get('tally') or {}
    out.append(f"  satisfied {t.get('satisfied', 0)}   "
               f"confirmed_gap {t.get('confirmed_gap', 0)}   "
               f"unverified {t.get('unverified', 0)} "
               f"({t.get('unverified_pct', 0)}%)   "
               f"demoted by a gate {t.get('demoted', 0)}")
    for s in res.get('statuses') or []:
        mark = {'satisfied': 'OK ', 'confirmed_gap': 'GAP', 'unverified': '?  '}
        out.append(f"  {mark[s['status']]} {s['id']}  "
                   f"{(s.get('requirement') or '')[:66]}")
        if s.get('note'):
            out.append(f"          {s['note'][:96]}")
        if s.get('gate') and s['gate'] != 'not answered':
            out.append(f"          demoted: {s['gate']}")
    for g in res.get('general') or []:
        out.append(f"  {g['status']:<14} {g['id']}  {g['check'][:60]}")
    return '\n'.join(out)


# WHICH STAGE EACH ITEM BELONGS TO -- DECIDED ONCE, THEN FROZEN.
#
# Re-classifying every round would put a second moving signal underneath the
# status, and a stage transition could then no longer be attributed to either:
# did the run leave Correct because the gaps closed, or because three items
# quietly became Complete? Measured, the per-item verdict already moves 10% per
# call; a second such signal is not affordable.
#
# Done by a model rather than by keyword rules. Three times in one session a
# lexical threshold has been the wrong instrument for a question about meaning
# -- the quote check that dropped a true requirement over two words, the
# token-Jaccard focal matcher, and the difflib alignment that reported 43% for
# an extraction that reproduces at 92%.
_STAGE_PROMPT = """Sort each requirement of this game into one of three stages.

  correct    without it the game cannot be played at all: it launches, you can
             get from the title into play, the main verb of the core loop
             works, the controls it advertises respond, a failure state lets
             you start again.
  complete   the game plays, but content or quantities the task named are
             missing: six sorties, four endings, three enemy types, a case
             board, a named final level.
Everything goes in one of those two. THERE IS NO POLISH STAGE HERE.

The checklist is answerable with satisfied / confirmed_gap / unverified, and
presentation is not: "the explosion is a plain orange circle" is not a missing
requirement, it is a quality judgement, and forcing it into this vocabulary is
what made the polish items sit at `satisfied` and produce no work at all. Polish
is a separate review mode over the same game -- see the Polish handoff -- and it
does not draw from this list.

So an item about how the game LOOKS still gets a stage here: put it in complete
if the task named the thing (a title screen exists, a sanity meter is visible),
because whether it EXISTS is a checklist question. How good it looks is not.

Reply with JSON and nothing else:
{{"stages": [{{"id": "T01", "stage": "correct"}}]}}   (only those two)

THE REQUIREMENTS

{items}
"""


def classify_stages(items: list, *, model: str | None = None) -> dict:
    """id -> stage, for the whole checklist plus the general checks."""
    from rsigame.agent.evolve.verifier_agent import _chat
    from rsigame.agent.llm import _load_env
    _load_env()
    if not items:
        return {}
    body = '\n'.join(f"  {i['id']}  {i['requirement']}" for i in items)
    _prev = os.environ.get('RSIGAME_MODELS_CHAT_REASONING_CAP')
    os.environ['RSIGAME_MODELS_CHAT_REASONING_CAP'] = '1'
    try:
        parsed, _ = _chat([{'role': 'user',
                            'content': _STAGE_PROMPT.format(items=body)}],
                          model=model or os.environ.get('RSIGAME_MODELS_LOOP_MODEL'))
    except Exception as exc:
        print(f'  classify_stages failed: {type(exc).__name__}: {exc}')
        return {}
    finally:
        if _prev is None:
            os.environ.pop('RSIGAME_MODELS_CHAT_REASONING_CAP', None)
        else:
            os.environ['RSIGAME_MODELS_CHAT_REASONING_CAP'] = _prev
    known = {i['id'] for i in items}
    out = {}
    for r in (parsed or {}).get('stages') or []:
        if not isinstance(r, dict):
            continue
        i, st = str(r.get('id') or '').strip(), str(r.get('stage') or '').strip().lower()
        if i in known and st in ('correct', 'complete'):
            out[i] = st
    # An item the model skipped is `correct`: the earlier stage is the safer
    # default for the same reason the prompt gives -- it gets looked at sooner.
    for i in known - set(out):
        out[i] = 'correct'
    return out


# WHERE EACH ITEM COULD BE SEEN -- ALSO DECIDED ONCE, ALSO FROZEN.
#
# Same reason as the stage: a hint that moved every round would make a probe's
# destination another thing that drifts, and the probe is supposed to be the
# stable part. Handoff 10 calls these optional hints on the checklist, and
# optional is right -- an item with no obvious home is probed in normal play,
# which is where most of them live anyway.
#
# The candidate list comes from `declared_scenarios()`, which reads the game's
# OWN source. Offering a name the game does not dispatch on would send play
# into a silent fallback that looks like the state it asked for.
_HINT_PROMPT = """A game can be launched straight into any of these named
states, which its own source code declares:

{scenarios}

For each requirement below, say which ONE of those states is the best place to
go and check it. Use "" -- the empty string -- when normal play from the title
screen is the right place, which is true of most of them, or when none of these
states obviously helps. Do not invent a state that is not in the list.

Reply with JSON and nothing else:
{{"hints": [{{"id": "T01", "scenario": ""}}]}}

THE REQUIREMENTS

{items}
"""


def scenario_hints(items: list, scenarios: list, *,
                   model: str | None = None) -> dict:
    """id -> scenario name, or '' for normal play."""
    from rsigame.agent.evolve.verifier_agent import _chat
    from rsigame.agent.llm import _load_env
    _load_env()
    if not items or not scenarios:
        return {i['id']: '' for i in items}
    _prev = os.environ.get('RSIGAME_MODELS_CHAT_REASONING_CAP')
    os.environ['RSIGAME_MODELS_CHAT_REASONING_CAP'] = '1'
    try:
        parsed, _ = _chat([{'role': 'user', 'content': _HINT_PROMPT.format(
            scenarios='\n'.join(f'  {x}' for x in scenarios),
            items='\n'.join(f"  {i['id']}  {i['requirement']}"
                             for i in items))}],
            model=model or os.environ.get('RSIGAME_MODELS_LOOP_MODEL'))
    except Exception as exc:
        print(f'  scenario_hints failed: {type(exc).__name__}: {exc}')
        return {i['id']: '' for i in items}
    finally:
        if _prev is None:
            os.environ.pop('RSIGAME_MODELS_CHAT_REASONING_CAP', None)
        else:
            os.environ['RSIGAME_MODELS_CHAT_REASONING_CAP'] = _prev
    ok = set(scenarios)
    out = {i['id']: '' for i in items}
    for r in (parsed or {}).get('hints') or []:
        if not isinstance(r, dict):
            continue
        i, sc = str(r.get('id') or '').strip(), str(r.get('scenario') or '').strip()
        if i in out:
            # An invented name is dropped to '' rather than passed through:
            # play would boot it, get a silent fallback, and report on the
            # wrong screen.
            out[i] = sc if sc in ok else ''
    return out
