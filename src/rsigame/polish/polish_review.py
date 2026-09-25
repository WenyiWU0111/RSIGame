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
"""The Polish review: what is the largest remaining quality bottleneck?

POLISH PART 2. One call, offline, no loop. It does not ask what requirement is
missing -- the checklist owns that and cannot express "the explosion is a plain
orange circle" anyway. It asks the other question, over the same game.

TWO DIMENSIONS ONLY, per the Polish handoff: Visual Quality & Impact, and
Motion / Feedback / Game Feel. Pacing, challenge, narrative and variety are
explicitly out of the first version.

ONE FOCUS, ONE GOAL. Not a ranked list, not a score, not five independent
tasks. The handoff is blunt about this and the reason is measured elsewhere in
this project: a critique asked for five defects returned exactly five in every
one of forty rounds, and the repair budget was then split five ways.

HEADROOM IS ANCHORED. `high | medium | low` is an unanchored scale, and it
drives the only rule that ends Polish -- one wrong `low` would end the phase.
So the two numbers a model cannot argue with go in the pack beside it: how many
shipped images the code never references, and what share of the drawing uses
authored art rather than primitives. Across the five pilot games those run
0-38 unused and 2%-37% authored, which is spread enough to separate them.

The stop rule then needs BOTH the judgement and the facts -- see `saturated`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DIMENSIONS = ('visual', 'feedback')
LEVELS = ('high', 'medium', 'low')

_PROMPT = """This game basically works and is substantially complete. You are
not looking for missing features -- someone else tracks those. You are looking
for the largest remaining QUALITY bottleneck.

Review two dimensions, and only these two.

VISUAL QUALITY & IMPACT -- does the game look intentionally authored, coherent
and impactful?
  A1  authored visual content: are the things on screen drawn art, or shapes
      standing in for art?
  A2  style coherence: do the parts look like they belong to one game?
  A3  state-dependent visual change: does the screen look different when the
      game's state is different?
  A4  composition and hierarchy: does the eye land on what matters?
  A5  legibility: can a player TELL THINGS APART? Text against its panel, the
      player's unit against the ground, an interactive control against the
      chrome around it, one entity type from another. Coherence and legibility
      pull in opposite directions and this list would otherwise only ask for
      one of them: a palette tightened until everything belongs also makes
      everything blend, and a screen nobody can read is not a coherent screen,
      it is an unusable one.

MOTION, FEEDBACK & GAME FEEL -- does acting on this game feel like anything?
  B1  action feedback: does a hit, a pickup, a score look like it happened?
  B2  state-transition feedback: do screens and phases change with any
      punctuation, or do they cut?
  B3  motion / animation: does anything that should move, move?
  B4  responsiveness: does input read as immediate?

{intent}

{facts}

{interactions}

The frames below follow, each labelled with the moment it came from.

HOW TO ANSWER

For each dimension give `headroom`: how much obvious room is left.
  high    an improvement here would visibly change the game for a player
  medium  worth doing, but not transformative
  low     already good, or nothing actionable is left

The counts above are facts about this build; use them. Under half the drawing
using authored art means real room in absolute terms, whatever other games do
-- a game whose world is mostly primitives looks unfinished to a player even if
every game beside it is too. A build shipping images its code never loads has
room by definition.

Visual comes first. A game that looks like a prototype is not helped by better
hit feedback, so unless the visual side genuinely has little room left, that is
where the round should go.

Then pick exactly ONE focus and write ONE coherent development goal for the
round -- the single change worth making now. Weigh three things: how much room
is left, whether a player would feel the difference across an important part of
the game, and whether one round of work could plausibly do it.

  Bad goal:  change font, add particles, redesign boss, improve HUD
  Good goal: replace the prototype-looking combat presentation with the game's
             own authored aviation assets, keeping the plane and the HUD
             clearly separated from the sky behind them

A goal that unifies must say what stays distinguishable. "Bring X into the
same visual language" is half a goal: the half that is missing is the one that
stops the answer being a screen where everything is the same three colours.

Reply with JSON and nothing else:

{{"visual":   {{"headroom": "high|medium|low", "main_issue": "..."}},
  "feedback": {{"headroom": "high|medium|low", "main_issue": "..."}},
  "focus": "visual|feedback|none",
  "next_goal": "...",
  "why_this_goal": "...",
  "evidence_refs": [4, 9]}}

`evidence_refs` are step numbers from the frames or interactions above. `focus`
is "none" only when neither dimension has anything worth a round.
"""


def _history_block(hist: list) -> str:
    """What Polish already tried, so a round is not a fresh review.

    Handoff 14: a round must first ask whether the last goal actually improved
    the game, and 15: the same goal attempted repeatedly without progress is a
    reason to reconsider, not to repeat.
    """
    if not hist:
        return ''
    out = ['WHAT POLISH HAS ALREADY TRIED, OLDEST FIRST', '']
    for h in hist[-4:]:
        out.append(f"  round {h.get('round')}  focus {h.get('focus')}: "
                   f"{str(h.get('goal'))[:150]}")
        out.append(f"      afterwards {h.get('files_changed', '?')} file(s) "
                   f"changed; headroom then read "
                   f"{json.dumps(h.get('headroom_after') or {}, ensure_ascii=False)}")
    out.append('')
    out.append('Say whether the last goal actually moved the game before '
               'choosing this round. Repeating a goal that has not worked '
               'twice is a reason to change focus or approach, not to try '
               'again.')
    return '\n'.join(out)


def review(pack: dict, *, history: list | None = None,
           model: str | None = None) -> dict:
    """One review. `pack` comes from `polish_evidence.build`."""
    from rsigame.agent.evolve.verifier_agent import _chat, _data_uri
    from rsigame.agent.llm import _load_env
    _load_env()
    import rsigame.polish.polish_evidence as PE

    inter = pack.get('interactions') or []
    itxt = ('WHAT ACTIONS DID TO THE SCREEN\n\n' + '\n'.join(
        f"  step {i['step']}: {i['action']} {i['what']} -- "
        f"{i['screen_changed_pct']}% of the screen changed"
        for i in inter)) if inter else ''
    body = _PROMPT.format(
        intent=('WHAT THE TASK SAYS THIS GAME IS\n\n' + pack['intent'])
        if pack.get('intent') else '',
        facts=PE.render_facts(pack['facts']),
        interactions=itxt)
    h = _history_block(history or [])
    if h:
        body += '\n\n' + h

    content = [{'type': 'text', 'text': body}]
    for n, fp, label in (pack.get('frames') or []):
        p = Path(fp)
        if not p.is_file():
            continue
        content.append({'type': 'image_url',
                        'image_url': {'url': _data_uri(p)}})
        content.append({'type': 'text',
                        'text': f'(step {n} -- {label})'})

    _prev = os.environ.get('RSIGAME_MODELS_CHAT_REASONING_CAP')
    os.environ['RSIGAME_MODELS_CHAT_REASONING_CAP'] = '1'
    try:
        parsed, _ = _chat([{'role': 'user', 'content': content}],
                          model=model or os.environ.get('RSIGAME_MODELS_LOOP_MODEL'))
    except Exception as exc:
        return {'error': f'{type(exc).__name__}: {str(exc)[:160]}'}
    finally:
        if _prev is None:
            os.environ.pop('RSIGAME_MODELS_CHAT_REASONING_CAP', None)
        else:
            os.environ['RSIGAME_MODELS_CHAT_REASONING_CAP'] = _prev
    return _gate(parsed or {}, pack)


def _gate(p: dict, pack: dict) -> dict:
    """Keep the shape honest. Same discipline as everywhere else here.

    A malformed headroom becomes `medium` rather than being dropped: `low` is
    the value that ends the phase and must never be reached by accident, and
    `high` is the value that spends a round.
    """
    out = {}
    for d in DIMENSIONS:
        got = p.get(d) if isinstance(p.get(d), dict) else {}
        hr = str(got.get('headroom') or '').strip().lower()
        out[d] = {'headroom': hr if hr in LEVELS else 'medium',
                  'main_issue': str(got.get('main_issue') or '')[:400],
                  'gate': '' if hr in LEVELS else f'unreadable headroom {hr!r}'}
    focus = str(p.get('focus') or '').strip().lower()
    goal = str(p.get('next_goal') or '').strip()
    steps = {n for n, _, _ in (pack.get('frames') or [])} | \
            {i['step'] for i in (pack.get('interactions') or [])}
    refs = [r for r in (p.get('evidence_refs') or []) if r in steps]
    if focus not in DIMENSIONS or not goal:
        out.update({'focus': 'none', 'next_goal': '', 'evidence_refs': [],
                    'why_this_goal': '',
                    'gate': f'no usable focus/goal (focus={focus!r})'})
        return out
    out.update({'focus': focus, 'next_goal': goal[:900],
                'why_this_goal': str(p.get('why_this_goal') or '')[:300],
                'evidence_refs': refs,
                'gate': '' if refs else 'goal cites no evidence from the pack'})
    return out


def saturated(rev: dict, facts: dict) -> tuple:
    """Is Polish done? Needs the judgement AND the facts to agree.

    Handoff 16 stops when neither dimension reads `high`. That alone puts the
    end of the phase on one unanchored three-valued call, so the mechanical
    half has to agree: a build still shipping images it never loads, or drawing
    mostly with primitives, has room whatever the review said.
    """
    judged = all(rev.get(d, {}).get('headroom') != 'high' for d in DIMENSIONS)
    room = (facts.get('images_unused', 0) > 0
            or facts.get('authored_share', 1.0) < 0.5)
    if judged and not room:
        return True, 'neither dimension reads high and the build shows no ' \
                     'unused art or primitive-dominated drawing'
    if judged and room:
        return False, (f"the review says no high headroom, but the build still "
                       f"has {facts.get('images_unused')} unused image(s) and "
                       f"{facts.get('authored_share', 0) * 100:.0f}% authored "
                       f"drawing -- keeping Polish open")
    return False, 'a dimension still reads high'


def render(rev: dict) -> str:
    out = []
    for d in DIMENSIONS:
        v = rev.get(d) or {}
        out.append(f"  {d:<9} headroom {v.get('headroom','?'):<7} "
                   f"{(v.get('main_issue') or '')[:96]}")
    out.append(f"  focus     {rev.get('focus')}")
    if rev.get('next_goal'):
        out.append(f"  goal      {rev['next_goal'][:200]}")
        out.append(f"  why       {(rev.get('why_this_goal') or '')[:150]}")
        out.append(f"  evidence  steps {rev.get('evidence_refs')}")
    if rev.get('gate'):
        out.append(f"  GATE      {rev['gate']}")
    return '\n'.join(out)
