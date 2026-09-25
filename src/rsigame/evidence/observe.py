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
"""Two summaries a replay needs, each one model call for the whole round.

WHAT A DEMO DOES is written once per game and frozen. It is a description of
the script, not a claim about coverage: a demo that means to reach the victory
screen may never get there, and doc S6 is explicit that stating what a demo
covers would read as authority it has not earned. So the wording asked for is
what the inputs attempt, and the prompt says so.

WHAT HAPPENED comes from the frames the round already recorded, and it is per
round because it is about one build. One call covers every demo rather than one
call each: the repair agent's loop is already 72% of a round's wall clock, and
eight more calls a round would grow the dominant cost to describe evidence that
mostly says "unchanged".

Neither summary may propose a repair. Doc rule 5: evidence summaries carry what
was seen, and a summary that also says what to do turns evidence into an
instruction that the next reader cannot weigh independently.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUN))

# post_verify batches at 40 images; three per demo leaves room for a dozen
# demos and still shows each one's start, end and last frame.
PER_DEMO = 3


_DESCRIBE = """Below are fixed input scripts for one game. Each replays the
same keys and clicks every time it runs.

Say what each script ATTEMPTS. Not what it proves, not what it covers -- a
script that means to reach the victory screen does not always get there, and
you are looking at the inputs, not at a recording.

One sentence each, plainly worded, naming what a player doing this would be
trying to do.

{scripts}

Reply with strict JSON and nothing else:
{{"<demo id>": "<one sentence>"}}
"""


_OBSERVE = """Below are frames from replaying fixed input scripts against one
build of a game. Each script sends the same inputs every time; these frames are
what this build did with them.

For each demo, say what happened. Ground every sentence in the frames you were
given.

  - If the demo reached what it was going for, say what that looked like.
  - If an input produced no visible change, say which one.
  - If the demo never got far enough to show something, say it was not reached
    -- not that it is missing. Not seeing a behaviour is not seeing it absent.
  - Do not say what should be changed. Someone else decides that, and they need
    your account to weigh independently.

One thing in the frames is not part of the game: the mouse pointer, drawn by
the recorder as a small cross.

{demos}

Reply with strict JSON and nothing else:
{{"<demo id>": "<one or two sentences>"}}
"""


def describe(work: str | Path, cache: str | Path | None = None,
             *, model: str | None = None) -> dict:
    """One call, all demos, frozen afterwards."""
    work = Path(work)
    cache = Path(cache) if cache else None
    if cache and cache.is_file():
        return json.loads(cache.read_text())

    import rsigame.verify.post_verify as PV
    from rsigame.verify.post_repair_replay import _describe as describe_event, INPUT_TYPES

    blocks = []
    for p in sorted((work / 'demo_outputs').glob('*.json')):
        t = json.loads(p.read_text())
        evs = [e for e in (t.get('events') or []) if e.get('type') in INPUT_TYPES]
        lines = '\n'.join(f'    {i}. {describe_event(e)}'
                          for i, e in enumerate(evs, start=1))
        flag = f" (launched with --scenario {t['scenario']})" if t.get('scenario') else ''
        blocks.append(f'  {p.stem}{flag}\n{lines or "    (no input)"}')

    parsed, _ = PV._ask([{'role': 'user',
                          'content': _DESCRIBE.format(scripts='\n\n'.join(blocks))}],
                        model=model or os.environ.get('RSIGAME_MODELS_LOOP_MODEL'),
                        timeout=120.0)
    out = {k: str(v)[:220] for k, v in (parsed or {}).items()} if isinstance(parsed, dict) else {}
    if cache and out:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def observe_round(run_dir: str | Path, r: int, *, model: str | None = None,
                  cache: str | Path | None = None) -> dict:
    """One call, every demo this round replayed, on the frames already on disk."""
    cache = Path(cache) if cache else None
    if cache and cache.is_file():
        return json.loads(cache.read_text())

    import rsigame.verify.post_verify as PV
    from rsigame.agent.evolve.verifier_agent import _data_uri
    from rsigame.evidence.artifact import from_round, probes_at

    run_dir = Path(run_dir)
    lines, imgs = [], []
    for pid in probes_at(run_dir, r):
        a = from_round(run_dir, r, pid)
        if not a or not a['events']:
            continue
        evs = a['events']
        picks = []
        first = evs[0]
        if first.get('before'):
            picks.append((f'{pid} first input, before', first['before']))
        last = evs[-1]
        if last.get('after'):
            picks.append((f'{pid} last input, after', last['after'][0]))
        if a.get('final_frame'):
            picks.append((f'{pid} final frame', a['final_frame']))
        picks = picks[:PER_DEMO]
        if not picks:
            continue
        lines.append(f"  {pid}: {len(evs)} input(s) -- "
                     + ', '.join(lbl.split(' ', 1)[1] for lbl, _ in picks))
        imgs += picks

    if not imgs:
        return {}
    content = [{'type': 'text', 'text': _OBSERVE.format(demos='\n'.join(lines))}]
    for label, fp in imgs:
        p = Path(fp)
        if not p.is_file():
            continue
        content.append({'type': 'image_url', 'image_url': {'url': _data_uri(p)}})
        content.append({'type': 'text', 'text': f'({label})'})

    parsed, _ = PV._ask([{'role': 'user', 'content': content}],
                        model=model or os.environ.get('RSIGAME_MODELS_LOOP_MODEL'), timeout=180.0)
    out = {k: str(v)[:400] for k, v in (parsed or {}).items()} if isinstance(parsed, dict) else {}
    if cache and out:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def observe_one(rec: dict, probe_id: str, description: str | None = None,
                *, cache: str | Path | None = None,
                model: str | None = None) -> str:
    """What one replay did, from the frames it just wrote.

    A fixed demo produces no account of itself -- the harness cuts frames and
    hands them on -- so a stage that reads only text gets nothing, and a stage
    that reads only frames has to guess what the demo was for. This writes the
    sentence that was missing, from the same frames, in one call.
    """
    cache = Path(cache) if cache else None
    if cache and cache.is_file():
        return json.loads(cache.read_text()).get('observation', '')

    import rsigame.verify.post_verify as PV
    from rsigame.agent.evolve.verifier_agent import _data_uri

    evs = rec.get('events') or []
    picks = []
    for e in evs:
        if e.get('before'):
            picks.append((f"input {e['n']} before: {e['what']}", e['before']))
        for f in (e.get('after') or [])[:1]:
            picks.append((f"input {e['n']} after: {e['what']}", f))
    if rec.get('final_frame'):
        picks.append(('final frame', rec['final_frame']))
    if len(picks) > 16:
        step = len(picks) / 15
        keep = sorted({int(i * step) for i in range(15)} | {len(picks) - 1})
        picks = [picks[i] for i in keep]
    picks = [(l, f) for l, f in picks if Path(f).is_file()]
    if not picks:
        return ''

    head = (f'The demo `{probe_id}` was replayed against one build of a game.'
            + (f' What it attempts: {description}' if description else ''))
    body = (f'{head}\n\nSay what happened, grounded in these frames. If an input '
            'produced no visible change, say which. If the demo never got far '
            'enough to show something, say it was not reached -- not that it is '
            'missing. Do not say what should be changed.\n\nThe small cross in '
            'every frame is the recorder drawing the mouse pointer.\n\n'
            'Reply with strict JSON: {"observation": "<two or three sentences>"}')
    content = [{'type': 'text', 'text': body}]
    for label, fp in picks:
        content.append({'type': 'image_url', 'image_url': {'url': _data_uri(Path(fp))}})
        content.append({'type': 'text', 'text': f'({label})'})
    parsed, _ = PV._ask([{'role': 'user', 'content': content}],
                        model=model or os.environ.get('RSIGAME_MODELS_LOOP_MODEL'), timeout=180.0)
    out = str((parsed or {}).get('observation') or '')[:600] if isinstance(parsed, dict) else ''
    if cache and out:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({'observation': out}, ensure_ascii=False))
    return out
