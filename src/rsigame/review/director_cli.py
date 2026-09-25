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
"""The model director's controls: a thin client for the review backend.

    director_cli.py open  CASE_ID --model MODEL_ID      start (or resume) a review; prints the brief material
    director_cli.py press SID KEY  [--note WHY]         tap a key                      1 action
    director_cli.py hold  SID KEY MS [--note WHY]       hold a key for MS (100-5000)   1 action
    director_cli.py click SID X Y [--button right] [--note WHY]   click at game pixels 1 action
    director_cli.py wait  SID MS                        let the game run (100-5000)    free
    director_cli.py look  SID                           frame on screen now            free
    director_cli.py reset SID                           restart the game               1 of the reset budget
    director_cli.py state SID
    director_cli.py submit SID --brief BRIEF.json       the Development Brief; ends the session

KEY is a browser KeyboardEvent.code: Enter, Space, Escape, ArrowLeft, ArrowUp,
KeyA ... KeyZ, Digit0 ... Digit9, ShiftLeft, Tab. A single letter or digit is
accepted and converted. Game pixels: x in [0,1280), y in [0,720).

Every step prints JSON with the paths of the frames it captured; open them with
your image-reading tool. Only this client talks to the game.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

URL = os.environ.get('RSIGAME_REVIEW_REVIEW_URL', 'http://127.0.0.1:8932')
ALIASES = {'enter': 'Enter', 'return': 'Enter', 'space': 'Space', ' ': 'Space', 'esc': 'Escape',
           'escape': 'Escape', 'left': 'ArrowLeft', 'right': 'ArrowRight', 'up': 'ArrowUp',
           'down': 'ArrowDown', 'shift': 'ShiftLeft', 'tab': 'Tab', 'backspace': 'Backspace'}


def code(k: str) -> str:
    if k.lower() in ALIASES:
        return ALIASES[k.lower()]
    if len(k) == 1 and k.isalpha():
        return 'Key' + k.upper()
    if len(k) == 1 and k.isdigit():
        return 'Digit' + k
    return k


def call(path, body=None, method='POST'):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(URL + path, data=data, method=method,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {'http_status': e.code, **json.loads(e.read() or b'{}')}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('open'); o.add_argument('case'); o.add_argument('--model', required=True)
    for name in ('press', 'hold'):
        p = sub.add_parser(name); p.add_argument('sid'); p.add_argument('key')
        if name == 'hold':
            p.add_argument('ms', type=int)
        p.add_argument('--note', default='')
    c = sub.add_parser('click'); c.add_argument('sid'); c.add_argument('x', type=int); c.add_argument('y', type=int)
    c.add_argument('--button', default='left'); c.add_argument('--note', default='')
    w = sub.add_parser('wait'); w.add_argument('sid'); w.add_argument('ms', type=int)
    for name in ('look', 'reset', 'state'):
        sub.add_parser(name).add_argument('sid')
    s = sub.add_parser('submit'); s.add_argument('sid'); s.add_argument('--brief', required=True)
    a = ap.parse_args()

    if a.cmd == 'open':
        r = call('/api/model/open', {'case_id': a.case, 'model_id': a.model})
    elif a.cmd in ('press', 'hold'):
        body = {'action': a.cmd, 'key': code(a.key), 'note': a.note}
        if a.cmd == 'hold':
            body['ms'] = a.ms
        r = call(f'/api/model/{a.sid}/act', body)
    elif a.cmd == 'click':
        r = call(f'/api/model/{a.sid}/act', {'action': 'click', 'x': a.x, 'y': a.y,
                                            'button': a.button, 'note': a.note})
    elif a.cmd == 'wait':
        r = call(f'/api/model/{a.sid}/act', {'action': 'wait', 'ms': a.ms})
    elif a.cmd == 'look':
        r = call(f'/api/model/{a.sid}/act', {'action': 'look'})
    elif a.cmd == 'reset':
        r = call(f'/api/session/{a.sid}/reset', {})
        if 'actions_used' in r:
            r['frames'] = call(f'/api/model/{a.sid}/act', {'action': 'wait', 'ms': 1500}).get('frames')
    elif a.cmd == 'state':
        r = call(f'/api/session/{a.sid}/state', method='GET')
    else:
        brief = json.loads(open(a.brief).read())
        r = call(f'/api/session/{a.sid}/submit', {'fields': brief})
    for k in ('held', 'submitted', 'session_id'):
        if a.cmd not in ('open', 'state', 'submit'):
            r.pop(k, None)
    print(json.dumps(r, ensure_ascii=False, indent=1))
    return 0 if 'error' not in r and not r.get('http_status') else 1


if __name__ == '__main__':
    sys.exit(main())
