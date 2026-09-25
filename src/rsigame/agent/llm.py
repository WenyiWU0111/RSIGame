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
"""Talking to a model: the client, the keys, and the shapes the answers come in.

Lifted out of `evolve/stage1.py`, where it had ended up by accident. Six modules
across the loop imported `_client`, `_chat`'s helpers, `_extract_json` and
`_load_env` from a 2000-line module whose actual subject was a corpus-wide
brainstorm that nothing runs any more -- so the plumbing kept 1700 lines of a
retired design alive.
"""
from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

# gpt-5.5 is a reasoning model -- real test confirmed even a 1-candidate reconcile
# call spent 145 of its output tokens on internal reasoning before any visible
# JSON. A 3000-token budget was silently exhausted by reasoning alone on the real
# 17-candidate reconcile call (empty content, no exception -- finish_reason
# wasn't even being logged before this was diagnosed). Budget generously; a
# reasoning-heavy call that legitimately needs more will fail loudly with
# finish_reason=length now, not vanish.
_MAX_TOKENS = 8000


# The other half of that fix. Raising the ceiling is the expensive answer to a
# budget eaten by reasoning; bounding the reasoning is the cheap one, and on the
# real 12-frame reader payload it was also the better one. Measured on
# glm-5.3-flash: uncapped, 6031 reasoning tokens and 121s; capped, 161 tokens
# and 19s -- and it reported MORE defects, not fewer. Turning reasoning off
# outright is not on the table: the provider answers 400, "Reasoning is
# mandatory". The cap is an OpenRouter extension, so any caller that sends it
# must survive an endpoint that refuses it.
_BUDGET_CEILING = 24000


_REASONING_CAP = {"reasoning": {"effort": "low"}}


def _load_env() -> None:
    """Load `.env` explicitly instead of depending on the caller's shell.

    Stage 1 makes its own model calls, and nothing in this package auto-loads
    a .env; every working judge call used to rely on whoever started the run
    having sourced one.

    One file, the repo's own `.env`; `rsigame.config` decides where it is.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    from .. import paths
    p = paths.env_file()
    if p.exists():
        load_dotenv(p, override=False)


def _sanitize_ascii_key(key: str) -> str:
    """An API key must be ASCII; truncate at the first non-ASCII byte rather
    than pass a corrupted value through to httpx (which throws deep inside
    header-building with a confusing UnicodeEncodeError, not at the call site
    where the key was actually read)."""
    for i, ch in enumerate(key):
        if ord(ch) > 127:
            print(f"[stage1] WARNING: API key had non-ASCII content from "
                  f"position {i} onward (len {len(key)}) -- truncated. Check "
                  f"the .env file for a corrupted/garbled key value.")
            return key[:i]
    return key


def _data_uri(path: Path) -> str:
    import base64
    mt = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    b64 = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mt};base64,{b64}"


# A model served locally, and the endpoint it is served on. Set as
#   RSIGAME_MODELS_LOCAL_MODELS="qwen38-27b=http://127.0.0.1:8038/v1,other=..."
# One entry per model. Anything not named here keeps the shared base URL.
#
# WHY PER MODEL AND NOT ONE BASE URL. This loop runs three different models --
# a play agent, a repair agent and a judge -- and only the first is served
# here. Pointing the single `OPENAI_BASE_URL` at the local server would send
# all three to a vLLM that has one model loaded, and the other two would fail
# on an unknown model rather than fall back.
def _local_endpoints() -> dict:
    out = {}
    for pair in (os.environ.get("RSIGAME_MODELS_LOCAL_MODELS") or "").split(","):
        if "=" in pair:
            name, url = pair.split("=", 1)
            if name.strip() and url.strip():
                out[name.strip()] = url.strip()
    return out


# EVERY paid call in the Python half goes through `_client`, which makes it the
# one place a wrong model can be stopped. It needs stopping because the failure
# is silent and expensive: the codebase carries 58 literal `'gpt-5.5'` defaults
# and `agent_repair.ts` falls back to `anthropic/claude-opus-4.6`, so deleting
# one line from `.env` does not raise -- it bills at 12x and finishes normally.
#
# The allow-list is the three models the loop is configured with, read from the
# same env the loop reads. `RSIGAME_AGENT_ALLOWED_MODELS` overrides it (comma
# separated) for a deliberate one-off; `RSIGAME_AGENT_ALLOW_ANY_MODEL=1` disables
# the check entirely, which is what a test or a genuinely new model wants.
#
# Local vLLM endpoints are exempt: those are our own GPUs and cost nothing.
def _allowed_models() -> set:
    raw = os.environ.get("RSIGAME_AGENT_ALLOWED_MODELS")
    if raw:
        return {m.strip() for m in raw.split(",") if m.strip()}
    return {os.environ[k] for k in ("RSIGAME_MODELS_LOOP_MODEL", "RSIGAME_MODELS_PLAY_MODEL", "RSIGAME_REPAIR_MODEL",
                                    "GAME_AGENT_MODEL")
            if os.environ.get(k)}


def _client(model: str):
    from openai import OpenAI
    local = _local_endpoints().get(str(model or "").strip())
    if not local and not os.environ.get("RSIGAME_AGENT_ALLOW_ANY_MODEL"):
        allowed = _allowed_models()
        name = str(model or "").strip()
        if not name:
            raise RuntimeError(
                "no model was given to _client(). Something fell through to a "
                "default instead of passing one; refusing to guess, because "
                "the defaults in this codebase are the expensive models.")
        # An EMPTY allow-list is the dangerous case, not the permissive one: it
        # means none of the four variables is set, which is exactly the state
        # where every default in this codebase resolves to gpt-5.5 or opus. The
        # first version of this guard let it through.
        if not allowed:
            raise RuntimeError(
                f"refusing to call {name!r}: none of RSIGAME_MODELS_LOOP_MODEL / RSIGAME_MODELS_PLAY_MODEL / "
                "RSIGAME_REPAIR_MODEL / GAME_AGENT_MODEL is set, so there is nothing to "
                "check it against and the codebase's own defaults are gpt-5.5 "
                "and anthropic/claude-opus-4.6. Load the .env, or set "
                "RSIGAME_AGENT_ALLOWED_MODELS / RSIGAME_AGENT_ALLOW_ANY_MODEL=1.")
        if name not in allowed:
            raise RuntimeError(
                f"model {name!r} is not in the allow-list {sorted(allowed)}. "
                "This guard exists because a missing RSIGAME_MODELS_LOOP_MODEL / RSIGAME_MODELS_PLAY_MODEL / "
                "RSIGAME_REPAIR_MODEL silently falls back to gpt-5.5 or claude-opus, "
                "on a key shared with the rest of the lab. Set the variable in "
                ".env, or pass RSIGAME_AGENT_ALLOWED_MODELS / "
                "RSIGAME_AGENT_ALLOW_ANY_MODEL=1 if this is deliberate.")
    if local:
        # vLLM's OpenAI server does not check the key, but the SDK refuses to
        # construct without one.
        return OpenAI(api_key=os.environ.get("RSIGAME_MODELS_LOCAL_API_KEY") or "local",
                      base_url=local, timeout=180, max_retries=2)
    # THE JUDGE'S KEY IS NOT THE DEFAULT KEY. It used to be first in this list,
    # and a smoke test walked straight into what that costs: the bench judge runs
    # on a local vLLM whose key is the literal "local", so setting the judge key
    # sent "local" to OpenRouter and every model call in the loop came back 401
    # Missing Authentication header. The judge is configured through the bench's
    # own GAMECRAFT_BENCH_JUDGE_OPENAI_* variables; its key is only a fallback here.
    key = (os.environ.get("RSIGAME_MODELS_API_KEY") or os.environ.get("RSIGAME_GENERATOR_API_KEY")
           or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
           or os.environ.get("RSIGAME_JUDGE_API_KEY"))
    if not key:
        raise RuntimeError("set RSIGAME_MODELS_API_KEY / OPENROUTER_API_KEY / OPENAI_API_KEY")
    key = _sanitize_ascii_key(key)
    base_url = (os.environ.get("RSIGAME_JUDGE_QUALITY_JUDGE_BASE_URL") or os.environ.get("RSIGAME_GENERATOR_BASE_URL")
               or os.environ.get("OPENAI_BASE_URL") or "https://openrouter.ai/api/v1")
    return OpenAI(api_key=key, base_url=base_url, timeout=180, max_retries=2)


def _extract_json(text: str):
    m = re.search(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", text, re.S)
    if m:
        payload = m.group(1)
    else:
        i = min((text.find(c) for c in "[{" if c in text), default=-1)
        if i < 0:
            return None
        payload = text[i:]
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", payload))
        except json.JSONDecodeError:
            return None


