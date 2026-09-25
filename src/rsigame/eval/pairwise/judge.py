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
"""Blind pairwise judge: for every (pair, demo), which build is better.

Frames come from replay.py (the scoring system's replay and 0.5 s sampler) and
are selected with the scoring judge's own `_select_frames` (evenly spaced, at
most 40), so the judge sees what an official scoring pass would see. Both
frame sets go into one request, labelled Game 1 and Game 2, after the rubric
(rubric.md) and the task's design document (instruction.md). The official
per-task requirements are NOT shown: this is an independent comparison, not a
replay of the official score.

Three judgments per demo, local judge at temperature 0 (so variety has to come
from the input, not from sampling):
  v1  self = Game 1, all selected frames
  v2  self = Game 2, all selected frames          (position swapped)
  v3  self = Game 1, the other half of the frames (odd-indexed, both sides)
The verdict per dimension is the majority; three different answers is a tie.

    judge.py MANIFEST REPLAY_DIR OUT_DIR [--jobs 6] [--only g1,g2] [--limit-demos N]
Writes OUT_DIR/raw/<game>__<demo>__v<k>.json and OUT_DIR/verdicts.jsonl.
"""
from ... import paths
import argparse
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# the sys.path insert waits until the scorer is actually called, import so import resolves no paths
from openai import OpenAI


def _bench_judge_helpers():
    """GameCraft-Bench GameCraft-Bench's frame-selection and encoding helpers, fetched on use.

    this used to be a module-level `from gamecraft_bench... import`,which only worked if someone had put the bench on
    sys.path before the import —— i.e. the module only imported from one working directory.
    bench the bench is a package with its own pyproject, install it and the import just works;the path injection is only the fallback.
    """
    try:
        from gamecraft_bench.verifier.judges.openai_gpt import _select_frames, _data_uri
    except ModuleNotFoundError:
        b = str(paths.bench())
        if b not in sys.path:
            sys.path.insert(0, b)
        from gamecraft_bench.verifier.judges.openai_gpt import _select_frames, _data_uri
    return _select_frames, _data_uri

HERE = Path(__file__).resolve().parent
DIMS = ['mechanics', 'depth', 'visuals', 'art', 'overall']
MODEL = 'qwen38-27b'
MAX_DOC_CHARS = 9000


def frames_for(replay_dir: Path, game: str, arm: str, demo: str):
    d = replay_dir / game / arm / 'score' / 'p1' / 'demos' / demo / 'frames'
    _select_frames, _ = _bench_judge_helpers()
    return _select_frames(sorted(d.glob('*.png'))) if d.is_dir() else []


def build_messages(rubric, doc, first, second, demo):
    content = [{'type': 'text', 'text':
                f'# Design document\n\n{doc[:MAX_DOC_CHARS]}\n\n'
                f'# Demo\n\nBoth builds were played with the same scripted input: "{demo}".\n\n'
                f'# Game 1 ({len(first)} frames, in time order)'}]
    for i, f in enumerate(first, 1):
        _, _data_uri = _bench_judge_helpers()
        content += [{'type': 'image_url', 'image_url': {'url': _data_uri(f)}},
                    {'type': 'text', 'text': f'(Game 1, frame {i}/{len(first)})'}]
    content.append({'type': 'text', 'text': f'# Game 2 ({len(second)} frames, in time order)'})
    for i, f in enumerate(second, 1):
        content += [{'type': 'image_url', 'image_url': {'url': _data_uri(f)}},
                    {'type': 'text', 'text': f'(Game 2, frame {i}/{len(second)})'}]
    content.append({'type': 'text', 'text': 'Compare Game 1 and Game 2 following the rubric. Return only the JSON object.'})
    return [{'role': 'system', 'content': rubric}, {'role': 'user', 'content': content}]


def parse(text):
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        raise ValueError('no JSON object')
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        # The verdict fields come first and are simple; a stray unescaped quote
        # inside a free-text reason breaks strict JSON without touching them.
        # Temperature 0 repeats the same text on retry, so read them directly.
        obj = {k: v for k, v in re.findall(r'"(mechanics|depth|visuals|art|overall)"\s*:\s*"(1|2|tie)"', text)}
        obj['reasons'] = {'_unparsed': m.group(0)[:2000]}
    out = {}
    for d in DIMS:
        v = str(obj.get(d, '')).strip().lower().strip('"').replace('game ', '')
        if v not in ('1', '2', 'tie'):
            raise ValueError(f'bad value for {d}: {obj.get(d)!r}')
        out[d] = v
    return out, obj.get('reasons', {})


def judge_one(client, rubric, pair, demo, replay_dir, raw_dir, k):
    raw_path = raw_dir / f'{pair["game"]}__{demo}__v{k}.json'
    if raw_path.is_file():
        return json.loads(raw_path.read_text())
    fs = {arm: frames_for(replay_dir, pair['game'], arm, demo) for arm in ('self', 'robin')}
    if not fs['self'] or not fs['robin']:
        return None
    if k == 3:                                      # the other half of the frames
        fs = {arm: v[1::2] or v for arm, v in fs.items()}
    self_first = k in (1, 3)
    first, second = (fs['self'], fs['robin']) if self_first else (fs['robin'], fs['self'])
    doc = Path(pair['instruction']).read_text(errors='ignore')
    msgs = build_messages(rubric, doc, first, second, demo)
    err, text, usage = None, '', {}
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL, messages=msgs, temperature=0.0, max_tokens=1500,
                extra_body={'chat_template_kwargs': {'enable_thinking': False}}, timeout=900)
            text = r.choices[0].message.content or ''
            usage = r.usage.model_dump() if r.usage else {}
            votes, reasons = parse(text)
            err = None
            break
        except Exception as e:                      # retried; a failed judgment is never a vote
            err = f'{type(e).__name__}: {str(e)[:200]}'
            time.sleep(5 * (attempt + 1))
    rec = dict(game=pair['game'], set=pair['set'], demo=demo, k=k, self_first=self_first,
               n_frames={a: len(v) for a, v in fs.items()}, usage=usage, raw=text, error=err)
    if err is not None:
        with (raw_dir.parent / 'errors.jsonl').open('a') as f:
            f.write(json.dumps(dict(game=pair['game'], demo=demo, k=k, error=err, raw=text[:600],
                                    n_frames=rec['n_frames'])) + '\n')
    if err is None:
        # translate "1"/"2" back to arms
        to_arm = {'1': 'self' if self_first else 'robin', '2': 'robin' if self_first else 'self', 'tie': 'tie'}
        rec['verdict'] = {d: to_arm[votes[d]] for d in DIMS}
        rec['reasons'] = reasons
        raw_path.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    return rec


def majority(vs):
    c = Counter(vs)
    top, n = c.most_common(1)[0]
    return top if n >= 2 else 'tie'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('manifest')
    ap.add_argument('replays')
    ap.add_argument('out')
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--only', default='')
    ap.add_argument('--limit-demos', type=int, default=0)
    a = ap.parse_args()
    out, replay_dir = Path(a.out), Path(a.replays)
    raw_dir = out / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    pairs = json.loads(Path(a.manifest).read_text())['pairs']
    if a.only:
        pairs = [p for p in pairs if p['game'] in set(a.only.split(','))]
    rubric = (HERE / 'rubric.md').read_text()
    client = OpenAI(base_url='http://127.0.0.1:8038/v1', api_key='local')

    tasks = []
    for p in pairs:
        demos = [Path(d).stem for d in p['demos']]
        if a.limit_demos:
            demos = demos[:a.limit_demos]
        for demo in demos:
            for k in (1, 2, 3):
                tasks.append((p, demo, k))
    with ThreadPoolExecutor(a.jobs) as ex:
        recs = [r for r in ex.map(lambda t: judge_one(client, rubric, t[0], t[1], replay_dir, raw_dir, t[2]), tasks) if r]

    by = {}
    for r in recs:
        by.setdefault((r['game'], r['set'], r['demo']), []).append(r)
    n_err = sum(1 for r in recs if r.get('error'))
    flips = total = 0
    with (out / 'verdicts.jsonl').open('w') as f:
        for (game, set_, demo), rs in sorted(by.items()):
            ok = [r for r in rs if not r.get('error')]
            if len(ok) < 3:
                continue
            v = {r['k']: r['verdict'] for r in ok}
            for d in DIMS:
                total += 1
                flips += v[1][d] != v[2][d]          # same frames, sides swapped
                f.write(json.dumps(dict(game=game, set=set_, demo=demo,
                                        dim={'mechanics': 'M', 'depth': 'D', 'visuals': 'V',
                                             'art': 'A', 'overall': 'overall'}[d],
                                        verdict=majority([v[1][d], v[2][d], v[3][d]]),
                                        votes=[v[1][d], v[2][d], v[3][d]])) + '\n')
    print(f'{len(recs)} judgments, {n_err} errors; position flip rate {flips}/{total} '
          f'= {flips / max(total, 1):.0%}; verdicts -> {out / "verdicts.jsonl"}')


if __name__ == '__main__':
    main()
