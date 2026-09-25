#!/usr/bin/env python
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
"""Director review: local web backend.

    python review/server.py --cases review/cases/pilot1 --root $RSIGAME_RUNS/outer_guidance \
                            [--port 8765]

Binds 127.0.0.1 only; open it through SSH / VS Code port forwarding. One game
runs at a time: opening a case pauses whichever other session had its game up
(its counts and draft stay on disk and it resumes where it stopped).

Pages:  /                       case list for a reviewer (?reviewer=ID)
        /review/<case_id>       the review page (?reviewer=ID)
Stream: /stream/<session_id>    MJPEG of the live display
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import uvicorn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from .session import ReviewSession  # noqa: E402

app = FastAPI()
CASES: dict[str, dict] = {}
ROOT = Path('.')
SESSIONS: dict[str, ReviewSession] = {}
ID_RE = re.compile(r'^[A-Za-z0-9_.-]{1,40}$')


def _reviewer(rid: str) -> str:
    if not ID_RE.match(rid or ''):
        raise HTTPException(400, 'reviewer id: letters, digits, _ . - only (1-40 chars)')
    return rid


def _case(cid: str) -> dict:
    if cid not in CASES:
        raise HTTPException(404, f'no case {cid}')
    return CASES[cid]


def _sessions_dir(case: dict, rid: str) -> Path:
    return ROOT / case['run_id'] / case['game_id'] / case['checkpoint_id'] / 'human' / rid


def _latest(case: dict, rid: str) -> dict | None:
    d = _sessions_dir(case, rid)
    metas = sorted(d.glob('*/session.json')) if d.is_dir() else []
    return json.loads(metas[-1].read_text()) if metas else None


def _get(sid: str) -> ReviewSession:
    if sid not in SESSIONS:
        raise HTTPException(404, 'session not open; reload the page')
    return SESSIONS[sid]


@app.on_event('shutdown')
def _close_games():
    for s in SESSIONS.values():
        if s.game:
            s.close_game('server shutdown')


# ------------------------------------------------------------------ pages
@app.get('/')
def home():
    return FileResponse(HERE / 'static' / 'home.html')


@app.get('/review/{cid}')
def review_page(cid: str):
    _case(cid)
    return FileResponse(HERE / 'static' / 'review.html')


@app.get('/api/cases')
def cases(reviewer: str):
    rid = _reviewer(reviewer)
    out = []
    for cid, c in sorted(CASES.items()):
        m = _latest(c, rid)
        status = 'not started' if not m else ('submitted' if m['submitted'] else 'in progress')
        out.append({'case_id': cid, 'title': c['title'], 'checkpoint_id': c['checkpoint_id'],
                    'status': status, 'actions_used': m['actions_used'] if m else 0,
                    'actions_max': c['budget']['actions']})
    return out


# ---------------------------------------------------------------- session
@app.post('/api/session/open')
async def open_session(req: Request):
    body = await req.json()
    c, rid = _case(body.get('case_id')), _reviewer(body.get('reviewer_id'))
    m = _latest(c, rid)
    if m and m['submitted']:
        return {'submitted': True, 'session_id': m['session_id'], 'case': _public(c)}
    sid = m['session_id'] if m else f"{rid}-{time.strftime('%Y%m%d-%H%M%S')}"
    s = SESSIONS.get(sid) or ReviewSession(c, ROOT, 'human', rid, sid)
    SESSIONS[sid] = s
    for other in list(SESSIONS.values()):          # one live HUMAN game at a time
        if other is not s and other.game and other.kind == 'human':
            await asyncio.to_thread(other.close_game, 'another case opened')
    await asyncio.to_thread(s.open_game)
    return {'submitted': False, 'session_id': sid, 'case': _public(c),
            'state': s.state(), 'draft': s.load_draft()}


def _public(c: dict) -> dict:
    """What the page may show: never scores, never paths."""
    return {k: c[k] for k in ('case_id', 'title', 'checkpoint_id', 'spec', 'saturation_summary', 'budget')}


@app.post('/api/model/open')
async def open_model_session(req: Request):
    """A model director's session: same case, same budget, same files, under model/<model_id>/."""
    body = await req.json()
    c, mid = _case(body.get('case_id')), _reviewer(body.get('model_id'))
    d = ROOT / c['run_id'] / c['game_id'] / c['checkpoint_id'] / 'model' / mid
    metas = sorted(d.glob('*/session.json')) if d.is_dir() else []
    m = json.loads(metas[-1].read_text()) if metas else None
    if m and m['submitted']:
        return {'submitted': True, 'session_id': m['session_id']}
    sid = m['session_id'] if m else f"{mid}-{time.strftime('%Y%m%d-%H%M%S')}"
    s = SESSIONS.get(sid) or ReviewSession(c, ROOT, 'model', mid, sid)
    SESSIONS[sid] = s
    await asyncio.to_thread(s.open_game)
    await asyncio.sleep(1.0)
    first = s.grab('m000_open')
    s.game.pause()
    return {'submitted': False, 'session_id': sid, 'case': _public(c), 'state': s.state(), 'frames': [first]}


@app.post('/api/model/{sid}/act')
async def model_act(sid: str, req: Request):
    return await asyncio.to_thread(_get(sid).model_act, await req.json())


@app.get('/api/session/{sid}/state')
def state(sid: str):
    return _get(sid).state()


@app.post('/api/session/{sid}/input')
async def send_input(sid: str, req: Request):
    s, e = _get(sid), await req.json()
    typ = e.get('type')
    if typ in ('keydown', 'keyup'):
        return await asyncio.to_thread(s.key, str(e.get('code', '')), typ == 'keydown', e.get('client_t'))
    if typ == 'click':
        x = max(0, min(int(e.get('x', 0)), 1279))
        y = max(0, min(int(e.get('y', 0)), 719))
        return await asyncio.to_thread(s.click, x, y, str(e.get('button', 'left')), e.get('client_t'))
    if typ == 'move':
        await asyncio.to_thread(s.move, int(e.get('x', 0)), int(e.get('y', 0)))
        return {'ok': True}
    raise HTTPException(400, f'unknown input type {typ!r}')


@app.post('/api/session/{sid}/reset')
async def reset(sid: str):
    return await asyncio.to_thread(_get(sid).reset)


@app.post('/api/session/{sid}/relaunch')
async def relaunch(sid: str):
    s = _get(sid)
    if s.game and s.game.alive():
        return s.state()
    return await asyncio.to_thread(s.relaunch_after_crash)


@app.put('/api/session/{sid}/draft')
async def draft(sid: str, req: Request):
    body = await req.json()
    return _get(sid).save_draft(body.get('fields') or {})


@app.post('/api/session/{sid}/submit')
async def submit(sid: str, req: Request):
    body = await req.json()
    s = _get(sid)
    s.save_draft(body.get('fields') or {})
    r = await asyncio.to_thread(s.submit, body.get('fields') or {})
    return JSONResponse(r, status_code=200 if r['ok'] else 422)


@app.post('/api/session/{sid}/pause')
async def pause(sid: str):
    s = _get(sid)
    await asyncio.to_thread(s.close_game, 'reviewer left the page')
    return s.state()


@app.get('/stream/{sid}')
async def stream(sid: str):
    s = _get(sid)

    async def frames():
        last = 0.0
        while s.game is not None:
            g = s.game
            if g and g.latest_t > last:
                last = g.latest_t
                yield (b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                       + str(len(g.latest)).encode() + b'\r\n\r\n' + g.latest + b'\r\n')
            await asyncio.sleep(1 / 20)
    return StreamingResponse(frames(), media_type='multipart/x-mixed-replace; boundary=frame',
                             headers={'Cache-Control': 'no-store'})


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--port', type=int, default=8765)
    a = ap.parse_args()
    ROOT = Path(a.root)
    for p in sorted(Path(a.cases).glob('*.json')):
        c = json.loads(p.read_text())
        CASES[c['case_id']] = c
    print(f'{len(CASES)} cases; writing under {ROOT}; http://127.0.0.1:{a.port}/', flush=True)
    uvicorn.run(app, host='127.0.0.1', port=a.port, log_level='warning')


if __name__ == '__main__':
    main()
