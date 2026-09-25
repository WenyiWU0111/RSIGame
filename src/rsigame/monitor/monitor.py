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
"""M5 -- the global stage. Shadow mode: it records an opinion, nothing acts on it.

Shadow mode is structural here, not a flag. This module returns a record and
exposes no hook that could alter a run, so wiring it into the loop has to be a
deliberate later edit rather than something that happens by forgetting to set
an environment variable.

The split follows the redesign doc and what M2/M3 measured.

  The harness decides `progress` (S25) and owns the stop guard (S27). Both are
  arithmetic over M3's counts, and both are places where a model given room to
  improvise would eventually improvise a `stop`. In particular the control run
  -- one build replayed twice -- must read as `similar`; a monitor that calls
  pure replay noise `improved` would reward a round that changed nothing.

  The model decides headroom, the next direction, and the reason. Those need
  judgment about what a game still needs, which is not in any count.

  The model may propose `stop`; it cannot cause one. S27's guard is applied
  afterwards in code, and a vetoed stop is recorded as a veto rather than
  quietly rewritten, so the disagreement stays visible.

It never sees screenshots (S22) and never sees the held-out rubric.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUN))

from rsigame.monitor.consolidate import DIRECTIONS  # noqa: E402

ACTIONS = ('continue', 'redirect', 'stop')
HEADROOM = ('high', 'medium', 'low')
PROGRESS = ('improved', 'similar', 'regressed')
COVERAGE = ('sufficient', 'partial')

# A starting policy, not a measured constant. It separates the three datasets
# M2 and M3 were judged on -- the recovery reads 0.86, the broken build 0.45,
# the noise control 0.36 -- and the broken build reading `partial` is the
# behaviour we want: a checkpoint where nothing could be observed is exactly
# where a `stop` must not be allowed.
SUFFICIENT_AT = 0.8


def progress_of(summary: dict) -> tuple[str, str]:
    """S25, in code. Returns the label and the sentence that justifies it."""
    reg = summary['regressions']
    if reg['critical']:
        return 'regressed', reg['why']
    n = summary['substantive_improvements']
    if n >= 1:
        return 'improved', f'{n} substantive improvement(s), no critical regression'
    return 'similar', 'no substantive improvement and no critical regression'


def coverage_of(summary: dict) -> tuple[str, str]:
    cov = summary['coverage']
    failed = summary['demos']['failed']
    if failed:
        return 'partial', f'{len(failed)} demo(s) could not be read'
    if cov['rate'] >= SUFFICIENT_AT:
        return 'sufficient', f"{len(cov['observed'])}/{cov['n_items']} requirements observed"
    return 'partial', (f"only {len(cov['observed'])}/{cov['n_items']} requirements "
                       f"observed ({cov['rate']:.0%})")


def aggregate_headroom(rec: dict) -> str:
    """The four directions, reduced. `overall` no longer decides anything.

    It could not: it read `high` at all forty checkpoints, because one label
    over four directions says whatever the worst one says and one of them was
    always unfinished. The guard needs a value it can act on, so it takes the
    highest of the four it can actually see -- and a direction whose evidence
    was missing counts as high, since not having looked is not low.
    """
    by = rec.get('headroom_by_direction') or {}
    order = {'high': 3, 'medium': 2, 'low': 1}
    seen = [order[v] for v in by.values() if v in order]
    if len(seen) < len(DIRECTIONS):
        return 'high'
    return {3: 'high', 2: 'medium', 1: 'low'}[max(seen)]


def stop_allowed(summary: dict, headroom: str, coverage: str,
                 unresolved_ids: list[str]) -> tuple[bool, str]:
    """S27. Every clause must hold; the first that fails is the reason."""
    if headroom != 'low':
        return False, f'headroom is {headroom}, not low'
    if coverage != 'sufficient':
        return False, f'evidence coverage is {coverage}'
    if unresolved_ids:
        return False, (f'{len(unresolved_ids)} task requirement(s) still '
                       f'unresolved: {", ".join(unresolved_ids[:5])}')
    return True, 'headroom low, coverage sufficient, no requirement unresolved'


_PROMPT = """You are the global development monitor for a game that is being
improved one round at a time.

You do not write code, propose patches, diagnose source, or plan repairs.
Another agent does that. You answer two questions: how much actionable
development headroom is left, and where the remaining budget should go.

Everything below comes from replaying fixed input scripts on the previous
build and the current one. The behavioural part is a frame-by-frame comparison
of the two; the presentation part is a separate reader that looked at the
current build's frames and judged how finished it looks and feels.

====================
TASK REQUIREMENTS (frozen), and where each one stands
====================
`satisfied` and `confirmed_gap` were both established from evidence.
`unverified` means no scenario has exercised it -- that is news about the
scenarios, not about the game. Do not treat an unverified requirement as a
confirmed defect; with sparse coverage that alone would keep headroom high
indefinitely.

{items}

====================
HOW FINISHED IT LOOKS AND FEELS, on the current build
====================
A reader looks at the current build's frames every round and judges two
things it can see and the comparison above cannot. This is the only evidence
here about `presentation`.

{polish}

====================
WHAT CHANGED SINCE THE LAST CHECKPOINT
====================
Evidence coverage: {coverage}
  ({coverage_why})

Behaviour, counted across {n_demos} replayed scenarios. `unchanged` here means
no REGRESSION was detected -- it is not a statement that the direction has no
room left, and it never was:
{by_direction}

How the presentation reading moved since the last checkpoint:
{polish_delta}

Requirements never exercised by any scenario: {unobserved}

{estimator}====================
DEVELOPMENT HISTORY
====================
{history}

Budget: round {used} of {total} spent, {budget} remaining
Current direction: {current}

====================
WHAT TO DECIDE
====================
BEFORE ANYTHING ELSE: IF THE LAST BLOCK BROKE SOMETHING, FIX IT.

This is a gate, not a consideration to weigh against the others. If a proxy
measurement is reported above and its overall delta is -0.15 or worse, or any
category's delta is -0.3 or worse -- especially with coverage falling at the
same time, which is what a build that renders nothing looks like from here --
then the last block did not merely fail to pay. It destroyed something that
worked. The next round goes to putting that back, and your `reason` must name
the break.

A concrete, actionable defect somewhere else is NOT a reason to skip this
gate. Spending the next block on the prettiest available defect while the game
draws a blank screen is how a run loses ten rounds. Measured on
platformer-plumber-kingdom's third round: an instruction to simplify the
backdrop deleted the sky layer, the build came up a solid blue screen with no
game content, the proxy read -0.633 with coverage collapsed to 0.375 and
functional coverage at zero -- and the decision was a redirect to the
best-described cosmetic defect.

If no proxy measurement is reported above, this gate does not apply and you
decide exactly as you did before.

Otherwise:

You are choosing the next development direction from evidence about the
current artifact. For each direction, use its own current-state evidence:

  correctness / completeness   the requirement statuses above, and behaviour
  feedback                     observed responsiveness, and what the reader
                               says about whether acting on the game lands
  presentation                 what the reader says about how the game looks

Presentation is a valid development direction even when functional issues
remain. Do not require correctness or completeness to be fully solved before
considering it.

Choose based on where meaningful improvement opportunity is actually visible
in the current artifact -- not on which direction has the most unresolved
items, and not on which one has been left untried.

Headroom means work that is worth doing and that the evidence supports, not
everything a game could conceivably have.

  high    an unresolved task requirement, a broken interaction, a real
          gameplay gap, or several clearly actionable quality problems
  medium  the game mostly works and a meaningful player-visible improvement
          is still available
  low     the observed requirements are satisfied, nothing is blocking, and
          what is left is minor refinement

Not seeing a problem is not proof that there is none. Where coverage is
partial, say so in the headroom rather than reading silence as quality.

On direction: prefer to continue a direction that is producing verified
improvement while headroom remains. Repeated effort on a direction is NOT by
itself a reason to abandon it -- some defects genuinely take several attempts.
But if several attempts have produced no verified progress, continuing needs
either new evidence, a different target, or a clearly stronger unresolved need
than the alternatives offer.

`recent_return` is what recent investment in a direction actually produced,
not how much is left. A direction nobody has spent a round on is `unknown`,
never `low` -- and `unknown` on its own is not a reason to choose it; it needs
visible headroom too.

But where an untried direction DOES have visible headroom and the current one
has stalled, the untried one is the better spend. A stalled direction's next
round is the third or fourth attempt at something that has not worked; an
untried one buys the first piece of evidence about itself. `unknown` is a hole
in the record, not a low score, and one round is the only thing that closes
it. Measured on one game, the reader reported high feedback headroom and named
the defect in so many words -- inputs produce almost no visible response --
and `feedback` still went twenty-one rounds without a single round spent on
it, because every checkpoint compared its `unknown` against a direction that
already had a record.

You may propose `stop`. It is only honoured when headroom is low, coverage is
sufficient, and no task requirement is unresolved; otherwise it is recorded
and overridden.

Reply with strict JSON and nothing else:

{{"remaining_headroom": {{"overall": "high|medium|low",
                        "by_direction": {{"correctness": "high|medium|low",
                                        "completeness": "high|medium|low",
                                        "feedback": "high|medium|low",
                                        "presentation": "high|medium|low"}}}},
 "recent_return": {{"correctness": "high|medium|low|unknown",
                  "completeness": "high|medium|low|unknown",
                  "feedback": "high|medium|low|unknown",
                  "presentation": "high|medium|low|unknown"}},
 "development_control": {{"action": "continue|redirect|stop",
                        "next_direction": "correctness|completeness|feedback|presentation|none",
                        "stalled_direction": "correctness|completeness|feedback|presentation|none",
                        "reason": "<one sentence, grounded in the evidence above>"}}}}
"""


# The block that carries Call 1 into Call 2. Everything about how to read it
# is measured, not assumed -- see `_ESTIMATOR_HEAD`.
_ESTIMATOR_HEAD = """====================
PROXY MEASUREMENT OF THE LAST BLOCK
====================
A separate reader, which was shown only the two builds' frames and a fixed
sixteen-item checklist, scored this checkpoint and the one before it. It was
never told which build was newer, what anyone was trying to fix, how much
budget is left, or what you are deciding. Its numbers:

{table}
`delta` is its score for this checkpoint minus its score for the previous one.
`coverage` is the share of its checklist the frames actually exercised.

HOW MUCH OF THIS TO BELIEVE, MEASURED ON 86 CHECKPOINT PAIRS OF TEN GAMES:

  THE DELTAS ARE EVIDENCE. Against the held-out official score, the proxy's
  overall delta correlates at r = 0.71, and at r = 0.82 on the pairs where the
  official score actually moved by 0.05 or more. Where a category's delta is
  clearly positive, that direction paid in the block just spent; where it is
  flat, that block bought little that the frames can see.

  THE ABSOLUTE VALUES ARE NOT EVIDENCE OF HEADROOM. They correlate at only
  r = 0.46, and they saturate early: the proxy read 0.99 or higher at 28% of
  these checkpoints, and on those the official score ran as low as 0.415. Its
  sixteen generic criteria are satisfied by a game that is roughly two-thirds
  finished. So a high `new` value, or a `headroom` near zero, is NOT a reason
  to call headroom low and NOT a reason to stop. Judge headroom from the
  requirement statuses, the reader, and the behaviour above, exactly as you
  did before this block existed.

  A FLAT DELTA IS NOT A STALL ON ITS OWN. The proxy scores each item on three
  points, so its smallest visible step is coarse; real gains of two or three
  hundredths read as 0.000 to it. Treat a flat delta as weak evidence that
  pairs with the verified-repair record, never as proof that a direction is
  finished.

  A LARGE NEGATIVE DELTA IS NOT A STALL EITHER -- IT IS BREAKAGE, AND IT
  OUTRANKS EVERY OTHER CONSIDERATION ON THIS PAGE. A direction that produced
  no gain and a block that destroyed what already worked are different events
  and must not lead to the same decision. When the overall delta is -0.15 or
  worse, or a category's delta is -0.3 or worse, the last block broke
  something, and the next round goes to putting it back -- not to the
  direction with the most unresolved requirements, and not to an untried one.
  Name the broken thing in your reason.

  COLLAPSED COVERAGE IS PART OF THAT SIGNAL. Coverage falling sharply while
  the delta goes negative means the criteria could no longer be judged at all,
  which is what a build that renders nothing looks like from here. Measured on
  platformer-plumber-kingdom's third round: an instruction to simplify the
  backdrop deleted the sky layer, the game came up a solid blue screen with no
  content, the proxy read -0.633 with coverage down to 0.375 and functional
  coverage at zero -- and the reading was passed over for a redirect to
  `completeness`, which would have spent the next block adding content to a
  game that was drawing none.

"""


def _estimator_text(est: dict | None) -> str:
    """Call 1's output, rendered for Call 2. Empty when there is none."""
    if not est:
        return ''
    order = ('functional', 'content', 'feedback', 'presentation')
    name = {'functional': 'correctness', 'content': 'completeness',
            'feedback': 'feedback', 'presentation': 'presentation'}
    lines = [f'  {"direction":14s} {"previous":>9s} {"now":>9s} '
             f'{"delta":>8s} {"coverage":>9s}']
    for k in order:
        c = est.get(k) or {}
        f = lambda v, sgn=False: ('       --' if v is None
                                  else (f'{v:+9.3f}' if sgn else f'{v:9.3f}'))
        lines.append(f'  {name[k]:14s}{f(c.get("old"))}{f(c.get("new"))}'
                     f'{f(c.get("delta"), True)}{f(c.get("coverage"))}')
    o = est.get('overall') or {}
    f = lambda v, sgn=False: ('       --' if v is None
                              else (f'{v:+9.3f}' if sgn else f'{v:9.3f}'))
    lines.append(f'  {"OVERALL":14s}{f(o.get("old"))}{f(o.get("new"))}'
                 f'{f(o.get("delta"), True)}{f(o.get("coverage"))}')
    return _ESTIMATOR_HEAD.format(table='\n'.join(lines))

def _history_text(rows: list[dict]) -> str:
    head = (f"{'direction':14s}{'rounds':>7s}{'accepted':>10s}{'failed':>8s}"
            f"{'last':>11s}{'stalled':>9s}")
    out = [head]
    for r in rows:
        out.append(f"{r['direction']:14s}{r['rounds']:>7d}{r['accepted']:>10d}"
                   f"{r['failed']:>8d}{r['last_progress']:>11s}"
                   f"{r['consecutive_stall']:>9d}")
    return '\n'.join(out)


def _direction_text(summary: dict) -> str:
    out = []
    for d in DIRECTIONS:
        v = {k: n for k, n in summary['by_direction'][d].items() if n}
        out.append(f'  {d:14s} ' + (', '.join(f'{k} {n}' for k, n in sorted(v.items()))
                                    if v else '(nothing observed)'))
    return '\n'.join(out)


def _items_text(items: list[dict]) -> str:
    """Requirement, and where it stands. The status was missing entirely.

    Forty checkpoints were given `id + requirement` and nothing else, so even
    correctness and completeness -- the two directions the monitor did choose
    between -- were only partly observable to it.
    """
    out = []
    for i in items:
        st = i.get('status')
        tail = ''
        if st:
            tail = f"   [{st}"
            if i.get('stale'):
                tail += ', the code changed afterwards'
            tail += ']'
        out.append(f"  {i['id']}  {i['requirement']}{tail}")
    return '\n'.join(out)


def _polish_text(cur: dict | None) -> str:
    if not cur or cur.get('error'):
        return ('  (no presentation reading at this checkpoint -- treat '
                'presentation headroom as unknown rather than low)')
    hr = cur.get('headroom') or {}
    mi = cur.get('main_issue') or {}
    out = []
    for d, lab in (('visual', 'presentation'), ('feedback', 'feedback')):
        out.append(f"  {lab:14s} headroom {hr.get(d, '?')}")
        if mi.get(d):
            out.append(f"      {str(mi[d])[:400]}")
    return '\n'.join(out)


def _polish_delta_text(cur: dict | None, prev: dict | None) -> str:
    """Did the presentation reading move since the last checkpoint.

    Free: the reader already runs every round, so the two ends of the interval
    are both on disk. This is the visual half of `recent_return`, and it has
    NOT been validated against a same-build control the way the behavioural
    comparator was -- a change reported here may be the reader moving rather
    than the game. Read it with that in mind.
    """
    if not cur or cur.get('error'):
        return '  (no reading now)'
    if not prev or prev.get('error'):
        return '  (no earlier reading to compare with -- unknown, not "same")'
    a, b = prev.get('headroom') or {}, cur.get('headroom') or {}
    rank = {'high': 3, 'medium': 2, 'low': 1}
    out = []
    for d, lab in (('visual', 'presentation'), ('feedback', 'feedback')):
        x, y = rank.get(a.get(d)), rank.get(b.get(d))
        if x is None or y is None:
            mv = 'unknown'
        elif y < x:
            mv = f'headroom fell {a.get(d)} -> {b.get(d)} (it improved)'
        elif y > x:
            mv = f'headroom rose {a.get(d)} -> {b.get(d)} (it got worse)'
        else:
            mv = f'headroom unchanged at {b.get(d)}'
        out.append(f'  {lab:14s} {mv}')
    out.append('  (this channel has no same-build control; a small move may '
               'be the reader, not the game)')
    return '\n'.join(out)


def build_prompt(summary: dict, items: list[dict], history: list[dict], *,
                 budget: int, total_budget: int | None = None,
                 estimator: dict | None = None,
                 current_direction: str = 'none',
                 polish: dict | None = None,
                 polish_prev: dict | None = None) -> str:
    """What the monitor is shown. Separated from asking so it can be rendered
    for the record without spending a call -- the input is the half that has to
    be answerable afterwards, and until now only the reply was kept."""
    used = sum(h['rounds'] for h in history)
    total = total_budget if total_budget is not None else used + budget
    cov, cov_why = coverage_of(summary)
    unobs = (summary['coverage']['unobserved_only']
             + summary['coverage']['never_mentioned'])
    return _PROMPT.format(
        items=_items_text(items),
        polish=_polish_text(polish),
        polish_delta=_polish_delta_text(polish, polish_prev),
        coverage=cov, coverage_why=cov_why,
        n_demos=summary['demos']['read'], by_direction=_direction_text(summary),
        unobserved=', '.join(sorted(unobs)) or '(none)',
        estimator=_estimator_text(estimator),
        history=_history_text(history), budget=budget, used=used,
        total=total, current=current_direction)


def assess(summary: dict, items: list[dict], history: list[dict], *,
           budget: int, total_budget: int | None = None,
           estimator: dict | None = None,
           current_direction: str = 'none',
           polish: dict | None = None, polish_prev: dict | None = None,
           prompt_out: str | None = None,
           model: str | None = None) -> dict:
    """One global reading. Advisory only -- see the module docstring."""
    import rsigame.verify.post_verify as PV

    # Where the run sits in its arc, not just what is left: six rounds
    # remaining out of fifteen is past halfway and time to converge, six out
    # of eight has barely started. The harness computes all three -- S23 is
    # explicit that the model must not be left to count rounds itself, and
    # the direction table is exactly the free-form source it would have to
    # add up otherwise.
    used = sum(h['rounds'] for h in history)
    total = total_budget if total_budget is not None else used + budget

    prog, prog_why = progress_of(summary)
    cov, cov_why = coverage_of(summary)
    unobs = summary['coverage']['unobserved_only'] + summary['coverage']['never_mentioned']

    body = build_prompt(
        summary, items, history, budget=budget, total_budget=total_budget,
        estimator=estimator, current_direction=current_direction,
        polish=polish, polish_prev=polish_prev)
    if prompt_out:
        Path(prompt_out).parent.mkdir(parents=True, exist_ok=True)
        Path(prompt_out).write_text(body)

    rec = {'advisory': True, 'progress': prog, 'progress_why': prog_why,
           'evidence_coverage': cov, 'coverage_why': cov_why,
           'polish_seen': bool(polish and not polish.get('error')),
           'budget': budget, 'budget_used': used, 'budget_total': total,
           'current_direction': current_direction}
    parsed, _ = PV._ask([{'role': 'user', 'content': body}],
                        model=model or os.environ.get('RSIGAME_MODELS_LOOP_MODEL'))
    if not isinstance(parsed, dict):
        rec['error'] = 'the reading returned nothing after three tries'
        return rec

    rec.update(validate(parsed))

    # S27's guard, applied after the model has spoken so a vetoed stop stays
    # visible as a disagreement rather than being silently rewritten.
    unresolved = [i for i in summary['coverage']['never_mentioned']]
    if rec.get('action') == 'stop':
        ok, why = stop_allowed(summary, aggregate_headroom(rec),
                               cov, unresolved)
        rec['stop_requested'] = True
        rec['stop_allowed'] = ok
        rec['stop_guard'] = why
        if not ok:
            rec['action'] = 'continue'
            rec['action_overridden_from'] = 'stop'
    return rec


def validate(d: dict) -> dict:
    bad = []

    def enum(v, allowed, where):
        if v in allowed:
            return v
        bad.append(f'{where}={v!r}')
        return None

    hr = d.get('remaining_headroom') or {}
    ctl = d.get('development_control') or {}
    by = hr.get('by_direction') or {}
    return {
        'overall_headroom': enum(hr.get('overall'), HEADROOM, 'overall'),
        'headroom_by_direction': {
            k: enum(by.get(k), HEADROOM, f'by_direction.{k}') for k in DIRECTIONS},
        'recent_return': {
            k: enum((d.get('recent_return') or {}).get(k),
                    HEADROOM + ('unknown',), f'recent_return.{k}')
            for k in DIRECTIONS},
        'action': enum(ctl.get('action'), ACTIONS, 'action'),
        'next_direction': enum(ctl.get('next_direction'),
                               DIRECTIONS + ('none',), 'next_direction'),
        'stalled_direction': enum(ctl.get('stalled_direction'),
                                  DIRECTIONS + ('none',), 'stalled_direction'),
        'reason': str(ctl.get('reason') or '')[:400],
        'schema_problems': bad,
    }
