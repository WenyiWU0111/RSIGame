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
"""Two checkpoints of one game, packed as the estimator's input.

WHAT CHANGED AND WHY.

The first version paired frames by INPUT NUMBER: the old build just before its
fourth keypress next to the new build just before its fourth keypress. That is
time-aligned by construction -- the input script is frame-timed, so input four
lands at the same moment in both recordings -- and it is still the wrong thing
to send. Time-aligned is not comparable. Measured on battle-critter-clash
r12 vs r15, 29% of r12's recording is pure black: the build had not started
drawing, so the scripted inputs landed on nothing. Pairing then puts a black
rectangle beside a battle screen and labels the two "the same moment", which
is true and useless. The reader cannot tell a build that lost the feature from
a build that had not woken up yet, and neither can a person.

WHAT IT DOES NOW is the simplest thing that can work, and the same thing the
official evaluator does: take the whole recording and sample it evenly, each
side independently, labelled by where in the recording the frame sits. No
pixel statistic steers the sample (a brightness gate picks startup blacks --
see the note on comparator.screen below), no input numbering implies a
correspondence the pictures do not honour. What the reader gets is two
sequences of the same length covering the same span, which is what "these are
two builds of one game" actually looks like.

WHERE THE FRAMES COME FROM. `post_replay` keeps only the ~31 input-anchored
PNGs; `post_repair_replay.prune()` deletes the mp4 once those are cut. So for
a checkpoint already on disk the recording is recovered from the scorer's
`score/p1/demos/<demo>/<demo>.mp4` -- READ AS A VIDEO FILE AND NOTHING ELSE.
`breakdown.json`, `judge_log.json` and `reward.txt` sit in that same directory
and are never opened here; the rubric stays held out. Using the scorer's own
recording is not a concession, it is the cleaner experiment: estimator and
evaluator then watch identical footage, so a disagreement between them is a
disagreement about judgement rather than about what was filmed.

WHAT IT DOES NOT USE is `comparator.screen`, which picks moments by how much
the two builds' pixels differ. That gate was measured for one repair's before
and after, where both sides are already drawn. Across checkpoints three rounds
apart a black frame against a lit one measures ~200 where a real change in
play measures ~60, so the screen preferentially selects the moments where one
side had not rendered yet. On critter r12/r15 all four selected moments were
the opening blacks and all six turns of the battle were screened out.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
sys.path.insert(0, str(RUN))
# `_data_uri` lives in the agent repo, which is a sibling checkout rather than
# an installed package -- the same resolution `preflight.py` does, so a run
# started outside the loop's shell finds it without a PYTHONPATH.

RUNS = Path(os.environ.get('RSIGAME_MONITOR_RUNS')
            or RUN / 'newloop_dispatch30' / 'runs')

# 22MB of base64, the ceiling the art round measured: one game's frames came
# to 38.7MB and every read of it failed with a 413 the retry path misreported
# as something else entirely.
URI_BUDGET = int(os.environ.get('RSIGAME_MONITOR_URI_BUDGET') or 22_000_000)
MAX_DEMOS = int(os.environ.get('RSIGAME_MONITOR_MAX_DEMOS') or 3)
# Per build, per demo. The official evaluator samples 40 across one recording;
# here two recordings share one request, so 16 a side keeps the pair inside a
# single call while still landing a frame every ~1.9s of a 30s demo.
N_FRAMES = int(os.environ.get('RSIGAME_MONITOR_N_FRAMES') or 16)

CACHE = Path(os.environ.get('RSIGAME_MONITOR_FRAME_CACHE') or HERE / 'frames_cache')


# --------------------------------------------------------------- discovery

def checkpoints(game: str) -> list[int]:
    out = []
    for d in sorted((RUNS / game).glob('round_*')):
        if (d / 'post_replay').is_dir() and any(d.glob('post_replay/frames_*')):
            try:
                out.append(int(d.name.split('_')[1]))
            except ValueError:
                pass
    return out


def demo_ids(game: str, r: int) -> list[str]:
    src = RUNS / game / f'round_{r:02d}' / 'post_replay'
    return sorted(p.name[len('frames_'):] for p in src.glob('frames_*'))


def video_for(game: str, r: int, demo: str) -> Path | None:
    """The recording of one demo at one checkpoint, or None.

    The loop's own mp4 first, for rounds recorded after prune() learned to
    keep it; the scorer's pass-1 recording otherwise. Pass 1 and not the
    median pass: the choice must not depend on any score.
    """
    rd = RUNS / game / f'round_{r:02d}'
    own = rd / 'post_replay' / f'{demo}.mp4'
    if own.is_file():
        return own
    for p in sorted((rd / 'score').glob(f'p*/demos/{demo}/{demo}.mp4')):
        return p
    return None


# --------------------------------------------------------------- sampling

def _n_frames(mp4: Path) -> int:
    """How long the recording is, in frames. Counted, not trusted.

    `nb_frames` is absent from some containers and wrong in others; the frame
    count is what the -vf select below indexes into, so an over-estimate puts
    the last samples past the end and silently returns fewer pictures than
    were asked for.
    """
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
         '-show_entries', 'stream=nb_read_frames', '-of', 'csv=p=0', str(mp4)],
        capture_output=True, text=True, timeout=300).stdout.strip()
    try:
        return int(out.split(',')[0])
    except (ValueError, IndexError):
        return 0


def _evenly(n: int, k: int) -> list[int]:
    """k indices spread from the first frame to the last."""
    if n <= 0:
        return []
    if k >= n:
        return list(range(n))
    if k == 1:
        return [n // 2]
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def recorded(game: str, r: int, demo: str) -> list[dict]:
    """The uniform sample the replay itself cut, if that round cut one.

    Rounds recorded before `post_repair_replay` learned to keep a uniform
    sample have only the input-anchored frames, and there the mp4 below is the
    only way back to the recording. Rounds recorded after it have this, which
    costs no ffmpeg pass and does not depend on the scorer having run.
    """
    f = RUNS / game / f'round_{r:02d}' / 'post_replay' / 'replay.json'
    if not f.is_file():
        return []
    try:
        rec = json.loads(f.read_text())
    except (OSError, ValueError):
        return []
    for d in rec.get('demos') or []:
        if d.get('demo_id') != demo:
            continue
        out = []
        for i, u in enumerate(d.get('uniform') or [], start=1):
            if Path(u['path']).is_file():
                out.append({'path': u['path'], 'i': i, 'frame': u['frame'],
                            'pos': u['pos']})
        return out
    return []


def extract(game: str, r: int, demo: str, k: int = N_FRAMES) -> list[dict]:
    """-> [{'path', 'i', 'frame', 'pos'}], evenly spaced over the recording.

    Cached: the same checkpoint is re-read by every pair it takes part in and
    by every arm of the ablation, and cutting frames costs more than the call
    that reads them.
    """
    have = recorded(game, r, demo)
    if have:
        return [have[i] for i in _evenly(len(have), k)] if k < len(have) \
            else have
    mp4 = video_for(game, r, demo)
    if mp4 is None:
        return []
    dst = CACHE / game / f'r{r:02d}' / demo / f'k{k}'
    done = dst / '.done'
    if not done.is_file():
        tmp = Path(str(dst) + '.tmp')
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        n = _n_frames(mp4)
        if n <= 0:
            return []
        want = _evenly(n, k)
        expr = '+'.join(f'eq(n\\,{f})' for f in want)
        cp = subprocess.run(
            ['ffmpeg', '-v', 'error', '-y', '-i', str(mp4),
             '-vf', f'select={expr}', '-vsync', '0',
             str(tmp / 'u%03d.png')],
            capture_output=True, text=True, timeout=1800)
        got = sorted(tmp.glob('u*.png'))
        if cp.returncode != 0 or not got:
            shutil.rmtree(tmp, ignore_errors=True)
            return []
        # `select` emits in order, so the i-th picture is the i-th wanted
        # frame -- but only if ffmpeg produced as many as were asked for.
        # It does not when a requested index lands past the last decodable
        # frame, and mislabelling the tail is worse than dropping it.
        want = want[:len(got)]
        for p, f in zip(got, want):
            p.rename(tmp / f'f{f:06d}.png')
        (tmp / 'meta.json').write_text(json.dumps(
            {'mp4': str(mp4), 'n_frames': n, 'frames': want}))
        done.parent.mkdir(parents=True, exist_ok=True)
        tmp.replace(dst)
        (dst / '.done').write_text('')
    meta = json.loads((dst / 'meta.json').read_text())
    n = meta['n_frames']
    out = []
    for i, f in enumerate(meta['frames'], start=1):
        p = dst / f'f{f:06d}.png'
        if p.is_file():
            out.append({'path': str(p), 'i': i, 'frame': f,
                        'pos': (f / max(n - 1, 1))})
    return out


def pick_demos(ids: list[str], game: str, r_old: int, r_new: int,
               cap: int) -> list[str]:
    """Which demos this pair sends.

    SEEDED ON THE PAIR, not on the clock. Every arm of the ablation must see
    the same demos, or the comparison is between two sets of evidence rather
    than between two rubrics; and a rerun of one arm must reproduce its own
    sample.
    """
    import random
    if cap <= 0 or len(ids) <= cap:
        return list(ids)
    rng = random.Random(f'{game}:{r_old}:{r_new}')
    return sorted(rng.sample(sorted(ids), cap))


def _scenario(game: str, r: int, demo: str) -> str:
    f = RUNS / game / f'round_{r:02d}' / 'post_replay' / f'{demo}.padded.json'
    if f.is_file():
        try:
            return json.loads(f.read_text()).get('scenario') or ''
        except Exception:
            pass
    return ''


# --------------------------------------------------------------- packing

def build(game: str, r_old: int, r_new: int, *, max_demos: int | None = None,
          n_frames: int = N_FRAMES):
    """-> {'demos': [{demo_id, content, shown, problems}], 'dropped': [...]}"""
    from rsigame.agent.evolve.verifier_agent import _data_uri

    ids = sorted(set(demo_ids(game, r_old)) & set(demo_ids(game, r_new)))
    if not ids:
        return {'error': f'{game} r{r_old}/r{r_new}: no demo name in common'}
    cap = MAX_DEMOS if max_demos is None else max_demos
    keep_ids = pick_demos(ids, game, r_old, r_new, cap)
    dropped = [d for d in ids if d not in keep_ids]

    out = []
    for demo in keep_ids:
        problems = []
        a = extract(game, r_old, demo, n_frames)
        b = extract(game, r_new, demo, n_frames)
        if not a or not b:
            for side, got, r in (('old', a, r_old), ('new', b, r_new)):
                if not got:
                    problems.append(
                        f'{side} (round {r}) no usable recording'
                        if video_for(game, r, demo) is None
                        else f'{side} (round {r}) frame extraction failed')
            out.append({'demo_id': demo, 'content': None, 'shown': [],
                        'problems': problems})
            continue

        # SAME COUNT ON BOTH SIDES, so "the old sequence is shorter" can never
        # be read as "the old build stopped early". A recording that really is
        # shorter still covers its own span; the positions say so.
        k = min(len(a), len(b))
        a = [a[i] for i in _evenly(len(a), k)]
        b = [b[i] for i in _evenly(len(b), k)]

        # WHAT FITS, DECIDED ON THE REAL FILES. The frames are already cut, so
        # there is nothing to estimate: measure them, and if the pair is over
        # budget drop back to a smaller EVEN sample of what was measured.
        # Fewer frames covering the whole recording beats the full count with
        # the tail missing -- truncation is exactly what even spacing is for.
        def cost(xs):
            return sum(Path(x['path']).stat().st_size for x in xs) * 4 // 3 \
                + 400 * len(xs)

        # THE WHOLE BUDGET, PER DEMO. Each demo is its own call (`score_pair`
        # scores one demo at a time), so dividing the ceiling by the number of
        # demos costs every one of them frames it could have sent -- measured
        # on critter, 11/9/8 a side instead of the full 16.
        while k > 2 and cost(a) + cost(b) > URI_BUDGET:
            k -= 1
            a = [a[i] for i in _evenly(len(a), k)]
            b = [b[i] for i in _evenly(len(b), k)]

        content, shown, used = [], [], 0

        def add(x, side):
            nonlocal used
            uri = _data_uri(Path(x['path']))
            if used + len(uri) > URI_BUDGET:
                return False
            used += len(uri)
            label = f"{side}: frame {x['i']}/{k} ({x['pos']*100:.0f}% in)"
            # THE LABEL GOES FIRST. Image-then-caption is ambiguous in a flat
            # content list -- a caption can as easily belong to the picture
            # after it -- and measured on this pair the reader took every
            # caption as naming the NEXT image, shifting the whole sequence by
            # one and swapping `old` with `new` end to end: it reported the
            # black startup as the new build's, when the black one is the old.
            # Nothing in the scores looked wrong; only the rationales gave it
            # away.
            content.append({'type': 'text', 'text': f'[{label}]'})
            content.append({'type': 'image_url', 'image_url': {'url': uri}})
            return True

        # INTERLEAVED BY POSITION, not concatenated. Both recordings run the
        # same frame-timed script, so the i-th sample of each sits at the same
        # moment of the same scenario; putting them next to each other is the
        # comparison, and the label says what it rests on.
        for x, y in zip(a, b):
            ok = add(x, 'old') and add(y, 'new')
            if ok:
                shown.append({'i': x['i'], 'old_frame': x['frame'],
                              'new_frame': y['frame'],
                              'pos': round(x['pos'], 3)})
        if k < n_frames:
            problems.append(f'budget/length cap: per side {k} frames (wanted {n_frames}）')
        out.append({'demo_id': demo,
                    'scenario': _scenario(game, r_new, demo) or demo,
                    'content': content, 'shown': shown,
                    'n_per_side': k, 'bytes': used, 'problems': problems})
    return {'game': game, 'r_old': r_old, 'r_new': r_new,
            'demos': out, 'dropped': dropped}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('game'); ap.add_argument('r_old', type=int)
    ap.add_argument('r_new', type=int)
    ap.add_argument('--max-demos', type=int, default=None)
    ap.add_argument('--n-frames', type=int, default=N_FRAMES)
    a = ap.parse_args()
    d = build(a.game, a.r_old, a.r_new, max_demos=a.max_demos,
              n_frames=a.n_frames)
    if d.get('error'):
        raise SystemExit(d['error'])
    print(f"  dropped demos: {d['dropped']}" if d['dropped'] else "  all demos sent")
    tot = 0
    for x in d['demos']:
        n_img = sum(1 for c in (x['content'] or []) if c['type'] == 'image_url')
        tot += n_img
        print(f"  {x['demo_id']:24s} per side {x.get('n_per_side')} frames · "
              f"images {n_img} · {(x.get('bytes') or 0)/1e6:.1f}MB"
              + (f" · problems {x['problems']}" if x['problems'] else ''))
    print(f"  total {len(d['demos'])} demos · {tot} images")
