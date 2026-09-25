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
"""Count the repair agent's tokens, because nothing else on this path does.

THE GAP THIS FILLS. `agent_run.ts` reads `usage` off the SDK's result message
and writes it to each session's `log.json`, and on this endpoint it comes back
`null` -- measured, on a real repair session: `cost: null`, and no token field
anywhere in `.qwen/trajectory.jsonl` either. So the coding half of the arm --
the LARGER half -- has no token count at all, and the deliverable asks for one.

`cost_report.py`'s answer to the same gap is 4 chars per token. That is a
reasonable stopgap for reading a run afterwards and a bad number to put in a
main table whose entire purpose is to answer "did you win by spending more".

WHY A PROXY AND NOT A PATCH. The repair agent talks OpenAI-compatible chat
completions (`authType: 'openai'` in `agent_run.ts`), and `repair.py` already
carries an env hook -- `RSIGAME_REPAIR_OPENAI_BASE_URL` -- whose whole purpose is
redirecting THIS subprocess and nothing else. So the count can be taken without
editing a line of the gamedoctor package or the SDK, which also means the
EvoRepair arm can be measured later through the same proxy, on the same
convention. Two arms whose token columns were produced different ways are not
comparable, and that column is the one reviewers will read hardest.

ONE PROXY PER GAME PROCESS, on a loopback port the OS picks. That is what
makes attribution exact: there is no request in this proxy's log that belongs
to another game. It also keeps the failure blast radius to one game.

LOOPBACK ONLY, ALWAYS. This host has no firewall and has been mined on; a
proxy holding an upstream API key is the last thing on it that should accept a
connection from anywhere else. The bind address is not configurable.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

# Hop-by-hop headers, plus the two that describe a body we are re-sending
# ourselves. Forwarding `Content-Length` from a request we may have rewritten,
# or `Content-Encoding` for a body we asked upstream not to compress, produces
# a response the client cannot parse -- and it would look like the model
# failing rather than like the proxy lying about the body.
_DROP_REQ = {'host', 'content-length', 'accept-encoding', 'connection'}
# `server` and `date` are dropped because BaseHTTPRequestHandler emits its own
# and relaying upstream's too produces one header with two comma-joined values.
# A tolerant client shrugs; the openai SDK's stream parser returned an empty
# completion, which reads exactly like the model saying nothing.
_DROP_RESP = {'content-length', 'content-encoding', 'transfer-encoding',
              'connection', 'keep-alive', 'server', 'date'}


class TokenProxy:
    """A counting pass-through to the real endpoint.

    `totals` is exact for every call whose response carried usage, and
    `calls_without_usage` counts the rest rather than letting them read as
    free. Both go into the run record.
    """

    def __init__(self, upstream: str, log_path: Path, api_key: str = ''):
        self.upstream = upstream.rstrip('/')
        self.api_key = api_key
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        # `cost` is dollars, which OpenRouter returns per call. Free to
        # capture and it answers the budget question directly, without anyone
        # having to re-derive it from tokens and a price list that will have
        # changed by the time the paper is read.
        #
        # `cached` is tracked apart from `input` and this is not bookkeeping
        # fastidiousness: the first repair call measured here was 21,329 input
        # tokens of which 21,312 were cache reads. Folding those into input
        # would overstate what the arm paid by three orders of magnitude, on
        # exactly the column a reviewer reads to ask whether we won by
        # spending more.
        self.totals = {'calls': 0, 'input': 0, 'output': 0, 'cached': 0,
                       'reasoning': 0, 'cost_usd': 0.0,
                       'calls_without_usage': 0, 'errors': 0}
        self._srv = None
        self._thread = None
        self.port = 0

    # -- accounting ------------------------------------------------------
    def record(self, usage: dict | None, meta: dict) -> None:
        with self.lock:
            self.totals['calls'] += 1
            if not usage:
                self.totals['calls_without_usage'] += 1
            else:
                self.totals['input'] += int(usage.get('prompt_tokens') or 0)
                self.totals['output'] += int(usage.get('completion_tokens') or 0)
                d = usage.get('prompt_tokens_details') or {}
                self.totals['cached'] += int(d.get('cached_tokens') or 0)
                c = usage.get('completion_tokens_details') or {}
                self.totals['reasoning'] += int(c.get('reasoning_tokens') or 0)
                try:
                    self.totals['cost_usd'] += float(usage.get('cost') or 0.0)
                except (TypeError, ValueError):
                    pass
            try:
                with self.log_path.open('a') as fh:
                    fh.write(json.dumps({'t': round(time.time(), 3),
                                         **meta, 'usage': usage}) + '\n')
            except OSError:
                pass          # accounting must never fail a repair session

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.totals)

    # -- lifecycle -------------------------------------------------------
    def start(self) -> str:
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *a):        # not our stdout to spend
                pass

            def log_error(self, fmt, *a):
                # NOT routed to log_message. It was, and that silently ate
                # every unhandled exception in the handler -- a streamed call
                # came back empty with zero calls counted and zero errors,
                # which reads as "the proxy was never asked" rather than as
                # "the proxy crashed". An accounting component that hides its
                # own failures is worse than no accounting.
                import sys as _s
                print('[token_proxy] ' + (fmt % a), file=_s.stderr, flush=True)

            def _relay(self, method: str):
                body = b''
                n = int(self.headers.get('Content-Length') or 0)
                if n:
                    body = self.rfile.read(n)
                path = self.path
                url = proxy.upstream + path

                model = ''
                streaming = False
                if body:
                    try:
                        payload = json.loads(body)
                        model = str(payload.get('model') or '')
                        streaming = bool(payload.get('stream'))
                        if streaming:
                            # A streamed completion carries usage only in a
                            # final chunk, and only if asked. Without this the
                            # proxy would relay perfectly and count nothing --
                            # the same silent zero we are here to remove.
                            so = dict(payload.get('stream_options') or {})
                            so['include_usage'] = True
                            payload['stream_options'] = so
                            body = json.dumps(payload).encode()
                    except Exception:
                        pass          # not JSON: relay it untouched

                headers = {k: v for k, v in self.headers.items()
                           if k.lower() not in _DROP_REQ}
                # Identity, so SSE text can be read as it passes. Asking for a
                # compressed body and then relaying it verbatim while stripping
                # Content-Encoding is how a proxy corrupts a response.
                headers['Accept-Encoding'] = 'identity'
                if proxy.api_key:
                    headers['Authorization'] = f'Bearer {proxy.api_key}'

                try:
                    r = requests.request(method, url, headers=headers,
                                         data=body or None, stream=True,
                                         timeout=600)
                except Exception as exc:
                    with proxy.lock:
                        proxy.totals['errors'] += 1
                    self.send_response(502)
                    self.send_header('Content-Type', 'application/json')
                    msg = json.dumps({'error': {'message':
                                      f'proxy could not reach upstream: '
                                      f'{type(exc).__name__}'}}).encode()
                    self.send_header('Content-Length', str(len(msg)))
                    self.end_headers()
                    self.wfile.write(msg)
                    return

                self.send_response(r.status_code)
                for k, v in r.headers.items():
                    if k.lower() not in _DROP_RESP:
                        self.send_header(k, v)
                self.send_header('Transfer-Encoding', 'chunked')
                self.end_headers()

                usage, tail = None, b''
                try:
                    for chunk in r.iter_content(chunk_size=8192):
                        if not chunk:
                            continue
                        self.wfile.write(b'%x\r\n%s\r\n' % (len(chunk), chunk))
                        self.wfile.flush()
                        # Scanned as it goes, so nothing has to be buffered.
                        # The last `usage` seen wins: a streamed response sends
                        # it once, at the end.
                        if b'usage' in chunk or tail:
                            tail = (tail + chunk)[-65536:]
                    self.wfile.write(b'0\r\n\r\n')
                    self.wfile.flush()
                except Exception:
                    with proxy.lock:
                        proxy.totals['errors'] += 1

                if tail:
                    usage = _find_usage(tail)
                proxy.record(usage, {'path': path, 'model': model,
                                     'stream': streaming,
                                     'status': r.status_code})

            def do_POST(self):
                self._relay('POST')

            def do_GET(self):
                self._relay('GET')

        self._srv = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.port = self._srv.server_address[1]
        self._thread = threading.Thread(target=self._srv.serve_forever,
                                        daemon=True)
        self._thread.start()
        return f'http://127.0.0.1:{self.port}'

    def stop(self) -> None:
        if self._srv is not None:
            try:
                self._srv.shutdown()
                self._srv.server_close()
            except Exception:
                pass


def _find_usage(blob: bytes) -> dict | None:
    """The last `usage` object in a body, streamed or not."""
    text = blob.decode('utf-8', 'replace')
    found = None
    # SSE first: `data: {...}` lines, the usage one normally last.
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith('data:'):
            continue
        payload = line[5:].strip()
        if payload in ('', '[DONE]'):
            continue
        try:
            d = json.loads(payload)
        except Exception:
            continue
        if isinstance(d, dict) and isinstance(d.get('usage'), dict):
            found = d['usage']
    if found:
        return found
    # A non-streamed body is one JSON object; it may be truncated at the front
    # because only the tail was kept, so look for the object by key.
    try:
        d = json.loads(text)
        if isinstance(d, dict) and isinstance(d.get('usage'), dict):
            return d['usage']
    except Exception:
        pass
    i = text.rfind('"usage"')
    if i >= 0:
        depth, start = 0, text.find('{', i)
        if start >= 0:
            for j in range(start, len(text)):
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:j + 1])
                        except Exception:
                            return None
    return None


def free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


if __name__ == '__main__':
    # A live check against the real endpoint: does a call relay correctly and
    # does its usage come back? Run it before trusting a batch's token column.
    import sys
    base = os.environ.get('OPENAI_BASE_URL') or 'https://openrouter.ai/api/v1'
    key = (os.environ.get('OPENROUTER_API_KEY')
           or os.environ.get('OPENAI_API_KEY') or '')
    model = sys.argv[1] if len(sys.argv) > 1 else 'z-ai/glm-5.3-flash'
    stream = '--stream' in sys.argv
    p = TokenProxy(base, Path('/tmp/token_proxy_selftest.jsonl'), key)
    url = p.start()
    print(f'proxy at {url} -> {base}; model={model} stream={stream}')
    try:
        from openai import OpenAI
        c = OpenAI(api_key=key or 'x', base_url=url, timeout=120)
        r = c.chat.completions.create(
            # NOT a small number. At 40 this reasoning model spent the whole
            # budget thinking and returned empty content, which looked like
            # the proxy dropping the body -- it cost two rounds of debugging
            # the wrong component.
            model=model, stream=stream, max_completion_tokens=300,
            messages=[{'role': 'user', 'content': 'Reply with one word: ok'}])
        if stream:
            got = ''.join(ch.choices[0].delta.content or ''
                          for ch in r if ch.choices)
            print('streamed reply:', repr(got[:60]))
        else:
            print('reply:', repr((r.choices[0].message.content or '')[:60]))
        print('counted:', json.dumps(p.snapshot()))
    finally:
        p.stop()
