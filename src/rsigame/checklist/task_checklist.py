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
"""What the task actually asked for, as a list you can check off.

PART 1 of the serial checklist framework, and only that: it turns a task
document into an immutable list of requirements. It does not track status, it
does not know about stages, and it is not wired into the loop -- an item here
has no `satisfied`/`confirmed_gap`/`unverified` because deciding those needs
evidence, which is Part 2.

WHY A LIST AND NOT A PROSE DIGEST. Compressing the spec into prose keeps the
same document to 250 words and hands it to the critique every round. That is a
summary, and a summary can only support "say what is wrong with this game". A
list supports a different question -- "is this one thing there?" -- and that is
the question the loop has never been able to ask. Measured over four arms:
depth carries 0.35 of the score, and across five games its rubric items moved
in exactly one of them across eight rounds. Nobody ever asked how many sorties
the task called for. The answer is in the task's second paragraph.

CONSERVATIVE BY CONSTRUCTION. Two rules, both of which cost something to
follow:

  - Numbers are copied, never inferred. The task for shooter-sky-duel names
    three enemy behaviours; the benchmark rubric asks for four. Writing four
    here would be guessing at the rubric, which is held out, so this writes
    three. The checklist will be incomplete against the rubric on purpose.
  - Every item carries the sentence it came from, and that quote is checked
    back against the document mechanically. An item the document does not say
    is dropped, not kept with a warning: a requirement the task never stated is
    the one failure mode that would poison everything downstream, because the
    loop would spend rounds building something nobody asked for. The check is
    on substance, not transcription -- the quote has to be most of some real
    passage, not a character-perfect copy of one.

GRANULARITY is one independently verifiable behaviour per item. Coarser (one
per document section) cannot express "half of this works"; finer (one per verb)
produces forty items for a fourteen-step session to check. Measured on the two
tasks read by hand, this lands near fifteen to twenty items, against thirteen
to eighteen rubric lines -- close enough that the checklist is a plausible
proxy for the graded axis without ever having seen it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

RUN = Path(__file__).resolve().parent
BENCH_TASKS = Path(os.environ.get('GAMECRAFT_BENCH_TASKS')
                   or (Path(os.environ['GAMECRAFT_BENCH']) / 'tasks'
                       if os.environ.get('GAMECRAFT_BENCH')
                       else RUN.parent / 'gamecraft-bench' / 'tasks'))

# Bump when the prompt or the schema changes, so a cached extraction from an
# older shape is not silently reused. The content hash alone cannot catch this:
# the document is unchanged, what changed is how we read it.
SCHEMA_VERSION = 2

# The prompt below tells the model its quote must be verbatim, and
# `verify_quotes` does not actually require that. The asymmetry is deliberate:
# asking for an exact copy is what makes the quotes good enough to be worth
# checking at all, while being strict about it on our side only threw away true
# requirements over transcription slips.
_PROMPT = """Below is the task document a game was built from. Turn it into a
checklist: the things that must be true of the finished game, one per line.

Reply with JSON and nothing else:

{{"items": [{{"requirement": "...", "quote": "...", "section": "..."}}]}}

  requirement  ONE independently verifiable thing, in your own words. Someone
               should be able to look at the running game and answer yes or no.
               Not "the game has good controls" -- "holding A or D moves the
               player horizontally".
  quote        The sentence or clause from the document below that this comes
               from, COPIED EXACTLY. It is checked against the document; an
               item whose quote is not found verbatim is discarded.
  section      The heading it sits under, or "" if none.

GRANULARITY. One verifiable behaviour per item. If a sentence names four
controls, that is four items. If a paragraph describes one system from three
angles, that is one item.

COPY THE NUMBERS, DO NOT INFER THEM. If the document says six levels, write
six. If it names three enemy types, write three -- not four because a fuller
game might have four. If it gives no number, do not invent one.

WHAT TO LEAVE OUT. Anything about how the project is built or delivered rather
than what the game does for a player: file layout, engine flags, build
commands, the demo trace format, where assets are mounted, how the evaluator
runs.

That includes the `--scenario` launch contract. A sentence like "when
`--scenario <id>` is present the game must skip menus and set up the named
state" describes how the evaluator starts the build, not something a player
experiences, and it is checked every time a scenario is booted anyway. Leave it
out. Do keep sentences about what NORMAL play shows -- "normal play starts from
the title screen and demonstrates the core loop" is a real requirement.

WHAT TO KEEP even though it is hard to check: named content that must exist
(levels, locations, enemy types, endings), stated quantities, explicit
mechanics, and the flow between screens.

THE TASK DOCUMENT

{body}
"""


def task_document(game: str, work: Path | None = None) -> tuple[str, str]:
    """The document this game was built from, and where it came from.

    The same two sources, read in the same order, so the checklist
    and the digest can never disagree about what the task said.
    """
    if work is not None:
        p = Path(work) / 'GAME_DESIGN.md'
        if p.is_file():
            return p.read_text(errors='replace'), str(p)
    p = BENCH_TASKS / game / 'instruction.md'
    if p.is_file():
        return p.read_text(errors='replace'), str(p)
    return '', ''


def _norm(s: str) -> str:
    """Whitespace-insensitive form, for the exact-match fast path."""
    return re.sub(r'\s+', ' ', s or '').strip().lower()


def _tokens(s: str) -> list:
    return re.findall(r'[a-z0-9]+', (s or '').lower())


# HOW CLOSE A QUOTE HAS TO BE.
#
# This started as an exact substring test and was wrong in the expensive
# direction. horror-tape-archive lost a real requirement -- "normal play starts
# from the title screen and demonstrates the core gameplay loop" -- because the
# model wrote "the core gameplay loop" where the document says "the task's core
# gameplay loop". Two words of transcription, and a requirement the task really
# does state disappeared without a trace.
#
# What this guard is for is a requirement the document never states. That is a
# question about substance, and tying it to transcription precision made it
# answer a different, much pickier question. So compare the quote against the
# nearest same-length passage of the document instead, and require it to be
# mostly that passage.
#
# The threshold is measured, not chosen. On horror-tape-archive:
#
#     verbatim sentences pulled from the document      1.000  (x6)
#     the real quote missing "the task's"              0.857
#     a real quote with one word swapped               0.909
#     a real quote in the wrong case                   1.000
#     a sentence assembled out of document vocabulary  0.357
#     a plausible-sounding invented requirement        0.200
#     a wholly invented requirement                    0.273
#
# True quotes bottom out at 0.857, invented ones top out at 0.357, and nothing
# lands between. 0.70 sits in the middle of that gap, so the cut does not
# depend on where exactly in it the line is drawn.
_QUOTE_MATCH = 0.70


def quote_match(quote: str, hay_tokens: list) -> float:
    """Just the score; see `quote_locate` for where the match sits."""
    return quote_locate(quote, hay_tokens)[0]


def quote_locate(quote: str, hay_tokens: list) -> tuple:
    """How close the quote comes to the nearest passage of the document, 0-1.

    Returns (score, token index of the best window), because the caller needs
    to know WHERE the quote came from, not only whether it is there: Part 2
    rejects a claimed defect whose evidence turns out to sit inside the play
    agent's own list of things it never reached.

    Sliding window at the quote's own length, scored by `difflib` on tokens, so
    a dropped word, an added word or a substitution each cost about one token
    out of twenty rather than the whole item. The set prefilter is a sound
    bound -- a match has to consume a window token that occurs in the quote, so
    a window with too few of those cannot reach the threshold whatever the
    ordering -- and it skips the expensive call on nearly every position.
    """
    import difflib
    q = _tokens(quote)
    if not q or not hay_tokens:
        return 0.0, -1
    n, qset = len(q), set(q)
    floor = _QUOTE_MATCH * n
    sm = difflib.SequenceMatcher(None, autojunk=False)
    sm.set_seq2(q)                      # cached across windows
    best, at = 0.0, -1
    for i in range(max(1, len(hay_tokens) - n + 1)):
        w = hay_tokens[i:i + n]
        if sum(1 for t in w if t in qset) < floor:
            continue
        sm.set_seq1(w)
        r = sm.ratio()
        if r > best:
            best, at = r, i
            if best > 0.995:
                break
    return best, at


def verify_quotes(items: list, raw: str) -> tuple[list, list]:
    """Split items into those the document really says, and those it does not.

    This is the guard the whole part rests on. A requirement the task never
    stated would send the loop building something nobody asked for, and unlike
    a missing item it would never be noticed -- it looks exactly like a real
    one. So the failure is a drop, not a flag.

    Every kept item carries the score it was kept on, so a weak match is
    auditable rather than silent.
    """
    hay, hay_tokens = _norm(raw), _tokens(raw)
    kept, dropped = [], []
    for it in items:
        q = _norm(it.get('quote'))
        if q and q in hay:
            it['quote_match'] = 1.0          # verbatim
        elif q:
            it['quote_match'] = round(quote_match(q, hay_tokens), 3)
        else:
            it['quote_match'] = 0.0
        (kept if it['quote_match'] >= _QUOTE_MATCH else dropped).append(it)
    return kept, dropped


# THE LAUNCH CONTRACT IS NOT A GAME REQUIREMENT.
#
# `### Scenarios` tells the builder that the game must accept `--scenario <id>`
# and set that state up deterministically. That is true, and the model is not
# wrong to read it as a behaviour -- it asked for exactly that and produced an
# item indistinguishable in shape from "the campaign spans six sorties": same
# `source: task`, same verbatim quote, same everything downstream can see.
#
# It has to go, for three reasons that compound. No rubric line scores it --
# thirty-one requirements read across two games, none mentions scenarios. It
# would consume a probe to verify, and a probe is a play step, which EV1 just
# showed is the scarcest thing in the loop: turning on grounding cost scenario
# coverage 88% -> 39% purely because steps ran out. And if it ever came back
# `confirmed_gap` it would spend a whole repair round on the evaluator's launch
# plumbing.
#
# It is also redundant three times over: `declared_scenarios()` already reads
# the scenario list out of the game's own source, `boot_scenario` answers the
# question every time it is called, and the Scenario Map will own it properly.
#
# Kept in `harness` rather than deleted, so the extraction stays auditable --
# these were found, and this is why they are not in the list.
_LAUNCH_CONTRACT = re.compile(r'--scenario|scenario\s*<\s*id\s*>|'
                              r'cmdline|command[- ]line argument', re.I)


def split_harness(items: list) -> tuple[list, list]:
    """Game requirements, and the evaluator's launch contract."""
    game, harness = [], []
    for it in items:
        text = f"{it.get('requirement', '')} {it.get('quote', '')}"
        (harness if _LAUNCH_CONTRACT.search(text) else game).append(it)
    return game, harness


def _cache_path(raw: str) -> Path:
    d = Path(os.environ.get('RSIGAME_AGENT_BRIEF_CACHE')
             or (RUN / '.brief_cache')) / 'checklist'
    d.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(
        f'v{SCHEMA_VERSION}\n{raw}'.encode()).hexdigest()[:24]
    return d / f'{key}.json'


def extract(game: str, work: Path | None = None, *, model: str | None = None,
            use_cache: bool = True) -> dict:
    """The task's requirements, as a checklist. Immutable; no status.

    Cached by (schema version, document content) outside any game directory --
    a cache file inside one would show up in the tree diff that decides whether
    a round changed anything. The caller copies the result into the run, so a
    run stays readable without the cache.
    """
    raw, source = task_document(game, work)
    if not raw.strip():
        return {'game': game, 'source': '', 'items': [], 'dropped': [],
                'error': 'no task document found'}

    cache = _cache_path(raw)
    if use_cache and cache.is_file():
        try:
            got = json.loads(cache.read_text())
            if got.get('items'):
                # RE-FILTER, DO NOT REPLAY. What is cached is the extraction --
                # what the model said the task asks for. How much of it we keep
                # is this module's judgement, and that judgement has changed
                # since some of these were written. Replaying the stored split
                # would silently answer with the old threshold; re-deriving it
                # from the full pre-filter set answers with the current one, at
                # no API cost. Re-extracting instead would also churn the item
                # counts for reasons unrelated to the change -- the extraction
                # is not deterministic, idle-spell-tower has come back with 25
                # items and with 16.
                return _finish(got.get('raw_items') or _all_items(got),
                               game, source, raw, got.get('model'),
                               cache=None, from_cache=True)
        except Exception:
            pass

    from rsigame.agent.evolve.verifier_agent import _chat
    # `.env` carries RSIGAME_MODELS_LOOP_MODEL and nothing in this package loads it on its own;
    # stage1 does, but only once `_chat` imports it, which is after the line
    # below has already resolved the model to None. See bootstrap_verifier.
    from rsigame.agent.llm import _load_env
    _load_env()
    model = model or os.environ.get('RSIGAME_MODELS_LOOP_MODEL')

    # CAP THE REASONING, AND SCOPE THE CAP TO THIS CALL.
    #
    # Twenty-odd items with a quote each is a long structured answer, and
    # uncapped the model spends the whole 8000-token budget thinking and never
    # begins it: `_chat` then retries at 16000 with the cap on, so every
    # extraction costs two requests and about three minutes. The cap is the
    # measured-better setting anyway -- on the reader payload it took
    # glm-5.3-flash from 6031 reasoning tokens and 121s to 161 and 19s, and it
    # returned MORE findings, not fewer.
    #
    # `_chat` reads it from the environment, so setting it globally would also
    # change the critique -- and every arm so far ran without it, which would
    # make the next one incomparable to them on an axis nobody chose. Set it
    # around this call and put it back, so turning it on for the critique stays
    # a decision someone makes on purpose.
    _prev = os.environ.get('RSIGAME_MODELS_CHAT_REASONING_CAP')
    os.environ['RSIGAME_MODELS_CHAT_REASONING_CAP'] = '1'
    try:
        parsed, _ = _chat(
            [{'role': 'user',
              'content': _PROMPT.format(body=raw[:20000])}], model=model)
    except Exception as exc:
        return {'game': game, 'source': source, 'items': [], 'dropped': [],
                'error': f'{type(exc).__name__}: {str(exc)[:160]}'}
    finally:
        if _prev is None:
            os.environ.pop('RSIGAME_MODELS_CHAT_REASONING_CAP', None)
        else:
            os.environ['RSIGAME_MODELS_CHAT_REASONING_CAP'] = _prev

    raw_items = (parsed or {}).get('items')
    if not isinstance(raw_items, list):
        return {'game': game, 'source': source, 'items': [], 'dropped': [],
                'error': 'the extraction returned no item list'}

    clean = []
    for it in raw_items:
        if not isinstance(it, dict):
            continue
        req = str(it.get('requirement') or '').strip()
        if req:
            clean.append({'requirement': req[:400],
                          'quote': str(it.get('quote') or '').strip()[:400],
                          'section': str(it.get('section') or '').strip()[:80]})

    # `use_cache=False` MEANS BOTH DIRECTIONS.
    #
    # It used to mean only "do not read", and the overnight stability run --
    # three uncached extractions per game, to measure how much this moves --
    # overwrote the cache three times per game as a side effect. It replaced
    # the exact extraction the acceptance run had reported on and the sweep was
    # still using, and took the pilot from 97 items to 105 with nothing in the
    # documents changed. A caller asking to bypass the cache is asking not to
    # be affected BY it; silently rewriting it makes the measurement destroy
    # what it was measuring against.
    return _finish(clean, game, source, raw, model,
                   cache=cache if use_cache else None)


def _all_items(got: dict) -> list:
    """Everything the extraction produced, from a cache written before we
    stored it as such -- the three buckets it was split into, minus the labels
    the split added, so it can be split again."""
    out = []
    for bucket in ('items', 'harness', 'dropped'):
        for it in got.get(bucket) or []:
            out.append({k: v for k, v in it.items()
                        if k not in ('id', 'source', 'quote_match')})
    return out


def _finish(raw_items: list, game: str, source: str, raw: str,
            model, *, cache, from_cache: bool = False) -> dict:
    """Apply this module's judgement to an extraction and number the result."""
    kept, dropped = verify_quotes(list(raw_items), raw)
    kept, harness = split_harness(kept)
    for i, it in enumerate(kept, 1):
        it['id'] = f'T{i:02d}'
        it['source'] = 'task'

    out = {'game': game, 'source': source, 'schema_version': SCHEMA_VERSION,
           'source_chars': len(raw), 'model': model,
           'items': kept, 'dropped': dropped, 'harness': harness,
           'raw_items': raw_items, 'from_cache': from_cache}
    if cache is not None:
        try:
            cache.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        except OSError:
            pass
    return out


def write_to_run(result: dict, run_dir) -> Path:
    """Copy the extraction into the run, so the run is self-contained.

    The cache is keyed by document content and shared across arms; a run
    directory has to be readable on its own months later, without depending on
    a cache that may have been cleared or on a document that may have moved.
    """
    p = Path(run_dir) / 'task_checklist.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    return p


def render(result: dict) -> str:
    """The checklist as text, for a prompt or for reading."""
    if not result.get('items'):
        return ''
    out = ['WHAT THE TASK ASKED FOR', '']
    sec = None
    for it in result['items']:
        if it.get('section') and it['section'] != sec:
            sec = it['section']
            out.append(f'  [{sec}]')
        out.append(f"  {it['id']}  {it['requirement']}")
    return '\n'.join(out)
