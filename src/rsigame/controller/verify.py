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
"""Did the round's issues get fixed? Two steps, and they are kept apart.

STEP ONE asks a model to look again, and it is never told that anyone tried to
repair anything. That blindness is the same one the pairwise comparator was
built with, and for the same measured reason: a reader who knows what was
attempted starts finding it. So an issue reaches this stage as a prior
observation to re-check --

    Earlier this was seen: "the resource label is clipped at the right edge".
    In these frames of the current build, is it still there?

-- which is a question about the frames. `still_there`, `gone`, or
`not_exercised`, and nothing about success.

STEP TWO is arithmetic, in the harness, over that answer and the repair agent's
own report of what it attempted. The agent knows what it tried; only the frames
know what happened; neither knows both, and the verdict needs both:

    attempted + gone          fixed
    attempted + still there   failed
    skipped   + still there   not_attempted        (with the reason it gave)
    skipped   + gone          fixed_incidentally   (worth knowing, not hidden)
    anything  + not exercised unobserved           (never `failed`)

That last line is the rule the whole framework rests on, in its verification
form. A replay that never reached the state has not shown the problem is still
there, and calling it failed would send the next round to repair something it
has no evidence about.

An issue-level verdict is never collapsed into one boolean for the round. A
round whose primary target failed while two small fixes landed did produce
verified improvement, and a single `success` flag would hide both halves.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUN))

OBSERVED = ('still_there', 'gone', 'not_exercised')
VERDICTS = ('fixed', 'failed', 'not_attempted', 'fixed_incidentally',
            'unobserved')


_PROMPT = """You are checking whether things that were seen in a game earlier
are still visible now.

You are not grading anyone's work. Nobody has told you whether these were
worked on, and you should not guess -- some may have been changed, some not,
and assuming either way is how a reader starts seeing what it expects.

====================
WHAT THE TASK REQUIRES
====================
{items}

====================
WHAT WAS SEEN EARLIER
====================
{issues}

====================
WHAT YOU ARE LOOKING AT
====================
The same fixed input script was replayed against the game as it is now. Each
frame below is labelled with the input that produced it.

Two things in the frames are not part of the game. The mouse pointer is drawn
by the recorder as a small cross. And the recording is not frame-exact -- the
same build replayed twice cuts a moment up to a frame or two apart, which alone
can make a score read one different or an animation look half a beat behind.

{note}

====================
RULES
====================
- Answer only from these frames.
- `still_there` means you can see it in these frames.
- `gone` means these frames show the thing it described working, or show the
  place it was and it is not there.
- `not_exercised` means the replay never reached where you would have to look.
  That is not the same as gone. If the script never opens the screen the label
  was clipped on, you have not seen it fixed.
- A behaviour the task requires is not a defect, however strange it looks.

Reply with strict JSON and nothing else:

{{"observations": [{{"issue_id": "<id>",
                   "observed": "still_there|gone|not_exercised",
                   "where": "<which moment above>",
                   "note": "<one sentence about what the frames show>"}}]}}
"""


def observe(issues: list[dict], items: list[dict], frames: list,
            *, model: str | None = None) -> dict:
    """Step one. One call for every issue, on one set of frames."""
    import rsigame.verify.post_verify as PV
    from rsigame.agent.evolve.verifier_agent import _data_uri

    frames = [(l, f) for l, f in (frames or []) if Path(f).is_file()]
    if not issues:
        return {'observations': [], 'schema_problems': []}

    lines = '\n'.join(f"  {i['issue_id']}: {i['issue']}\n"
                      f"        it was seen at: {', '.join(i.get('evidence_refs') or []) or 'no reference recorded'}"
                      for i in issues)
    note = (f'{len(frames)} frames follow.' if frames else
            'NO FRAMES WERE CAPTURED. Answer `not_exercised` for everything: '
            'without frames you have not looked.')
    body = _PROMPT.format(items='\n'.join(f"  {i['id']}  {i['requirement']}"
                                          for i in items) or '  (none)',
                          issues=lines, note=note)
    content = [{'type': 'text', 'text': body}]
    for label, fp in frames:
        content.append({'type': 'image_url', 'image_url': {'url': _data_uri(Path(fp))}})
        content.append({'type': 'text', 'text': f'({label})'})

    parsed, _ = PV._ask([{'role': 'user', 'content': content}],
                        model=model or os.environ.get('RSIGAME_MODELS_LOOP_MODEL'), timeout=180.0)
    if not isinstance(parsed, dict):
        return {'observations': [], 'error': 'the reading returned nothing'}

    known = {i['issue_id'] for i in issues}
    out, bad = [], []
    for o in (parsed.get('observations') or []):
        if not isinstance(o, dict):
            continue
        iid, ob = o.get('issue_id'), o.get('observed')
        if iid not in known:
            bad.append(f'issue_id={iid!r} was not asked about')
            continue
        if ob not in OBSERVED:
            bad.append(f'{iid}.observed={ob!r}')
            continue
        out.append({'issue_id': iid, 'observed': ob,
                    'where': str(o.get('where') or '')[:120],
                    'note': str(o.get('note') or '')[:300]})
    seen = {o['issue_id'] for o in out}
    for i in issues:
        if i['issue_id'] not in seen:
            # Silence is not an answer. An issue the reader skipped has not
            # been looked at, and saying so is the honest record.
            out.append({'issue_id': i['issue_id'], 'observed': 'not_exercised',
                        'where': '', 'note': 'the reading did not mention it'})
            bad.append(f"{i['issue_id']} was not answered")
    return {'observations': out, 'schema_problems': bad}


def verdicts(observations: list[dict], report: dict | None,
             selection: dict) -> dict:
    """Step two. Arithmetic over what was seen and what was attempted."""
    attempted, reasons = {}, {}
    rep = report or {}
    prim = rep.get('primary') or {}
    if prim.get('issue_id'):
        attempted[prim['issue_id']] = bool(prim.get('attempted'))
        if prim.get('reason'):
            reasons[prim['issue_id']] = str(prim['reason'])[:200]
    for c in (rep.get('concurrent') or []):
        if isinstance(c, dict) and c.get('issue_id'):
            attempted[c['issue_id']] = bool(c.get('attempted'))
            if c.get('reason'):
                reasons[c['issue_id']] = str(c['reason'])[:200]

    # THE ART ASK, if the round carried one. `attempted` comes from the
    # report's own art block rather than from the issue list, because the ask
    # never was an issue -- it came from the other reader.
    art_rep = rep.get('art') or {}
    if any(o.get('issue_id') == 'ART' for o in observations):
        attempted['ART'] = bool(art_rep.get('attempted'))
        if art_rep.get('files'):
            reasons['ART'] = 'files: ' + ', '.join(
                str(f)[:60] for f in art_rep['files'][:4])

    obs = {o['issue_id']: o for o in observations}
    primary_id = selection.get('primary_target')
    ids = [primary_id] + list(selection.get('concurrent_minor_fixes') or [])
    if 'ART' in obs:
        ids = ids + ['ART']
    out = []
    for iid in [x for x in ids if x]:
        o = obs.get(iid)
        seen = o['observed'] if o else 'not_exercised'
        # No report at all is not the same as a report saying "skipped": the
        # agent may simply have stopped without writing one, and guessing
        # `attempted` either way would put a fact in the record that nobody
        # supplied.
        tried = attempted.get(iid)
        if seen == 'not_exercised':
            v = 'unobserved'
        elif tried is False:
            v = 'fixed_incidentally' if seen == 'gone' else 'not_attempted'
        else:
            v = 'fixed' if seen == 'gone' else 'failed'
        out.append({'issue_id': iid,
                    'role': ('primary' if iid == primary_id
                             else 'art' if iid == 'ART' else 'concurrent'),
                    'observed': seen, 'attempted': tried,
                    'verdict': v, 'reason': reasons.get(iid, ''),
                    'note': (o or {}).get('note', '')})

    primary = next((x for x in out if x['role'] == 'primary'), None)
    return {'issues': out,
            'primary_verdict': primary['verdict'] if primary else None,
            'report_missing': report is None,
            # Deliberately no round-level success flag. See the module note:
            # a failed primary with two landed side fixes is both of those
            # things, and one boolean can only be one of them.
            'counts': {v: sum(1 for x in out if x['verdict'] == v)
                       for v in VERDICTS if any(x['verdict'] == v for x in out)}}
