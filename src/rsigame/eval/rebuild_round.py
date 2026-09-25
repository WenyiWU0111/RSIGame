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
"""Reconstruct the game tree as it stood after round N.

Each round writes `round_NN/changed/` holding the FULL text of every file that
round touched -- not a diff -- so replaying rounds 1..N over a copy of the
frozen G0 gives the tree that round N shipped.

Verified rather than assumed: rebuilding the last round and comparing against
the run's own `work/` is the check, and it is what `--verify` does.

  python rebuild_round.py --run self_refine_godot_gpt_rp2 --game X --round 6 --out DIR
  python rebuild_round.py --run ... --game X --verify
"""
from __future__ import annotations
import argparse, hashlib, shutil, sys
from pathlib import Path

RUN = Path(__file__).resolve().parent


def rebuild(run: str, game: str, upto: int, g0: Path, out: Path) -> dict:
    src = g0 / game
    if not src.is_dir():
        raise SystemExit(f'no frozen G0 at {src}')
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out, ignore=shutil.ignore_patterns('.godot'))
    gdir = RUN / run / 'runs' / game
    applied = files = 0
    for r in range(1, upto + 1):
        ch = gdir / f'round_{r:02d}' / 'changed'
        if not ch.is_dir():
            continue
        applied += 1
        for p in sorted(ch.rglob('*')):
            if not p.is_file():
                continue
            rel = p.relative_to(ch)
            tgt = out / rel
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, tgt)
            files += 1

    # ASSETS COME FROM `work/`, DATED BY MTIME.
    #
    # The per-round snapshot keeps only what can stop the project loading --
    # .gd/.tscn/.tres/.cfg/.import -- so a sprite the repair generated is not in
    # `changed/` even though its .import file is. Rebuilding without them gives
    # round-N code over G0 art, which understates exactly the dimension the
    # loop moves most: roguelike-wildwood gained 0.367 and its bear.png,
    # wolf.png and stag.png are all missing from the snapshot.
    #
    # Their mtimes place them cleanly (that game's assets land on rounds 10, 12
    # and 14), so anything written no later than round N's own directory is
    # part of round N's tree. Files the agent wrote for itself are left out.
    SCRATCH = {'.qwen', '.godot', '_diag', '_repair_evidence', '_tmpview',
               'QWEN.md', '_repairs_work'}
    MEDIA = ('.png', '.jpg', '.jpeg', '.webp', '.ogg', '.wav', '.mp3', '.ttf')
    rd = gdir / f'round_{upto:02d}'
    cutoff = rd.stat().st_mtime if rd.is_dir() else None
    assets = 0
    work = gdir / 'work'
    if cutoff and work.is_dir():
        for p in sorted(work.rglob('*')):
            if not p.is_file() or not p.name.endswith(MEDIA):
                continue
            rel = p.relative_to(work)
            if SCRATCH & set(rel.parts) or rel.parts[0] in SCRATCH:
                continue
            if p.stat().st_mtime > cutoff:
                continue
            tgt = out / rel
            if tgt.exists() and tgt.stat().st_size == p.stat().st_size:
                continue
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, tgt)
            assets += 1
    return {'rounds_applied': applied, 'files_written': files,
            'assets_from_work': assets}


def _digest(root: Path) -> dict:
    """Content of every file that is not an import cache, by relative path."""
    out = {}
    for p in sorted(root.rglob('*')):
        if not p.is_file() or '.godot' in p.parts:
            continue
        out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--game', required=True)
    ap.add_argument('--round', type=int)
    ap.add_argument('--out')
    ap.add_argument('--g0', default=None, help='frozen corpus dir')
    ap.add_argument('--verify', action='store_true')
    a = ap.parse_args()
    g0 = Path(a.g0) if a.g0 else (
        RUN / ('godot/corpus_qwen' if 'qwen' in a.run else 'godot/corpus_gpt'))
    gdir = RUN / a.run / 'runs' / a.game
    if a.verify:
        import json
        n = len(json.load(open(gdir / 'result.json'))['rounds'])
        tmp = Path(os.environ.get('TMPDIR') or '/tmp') / 'rsigame-rebuild'
        info = rebuild(a.run, a.game, n, g0, tmp)
        got, want = _digest(tmp), _digest(gdir / 'work')
        only_got = sorted(set(got) - set(want))
        only_want = sorted(set(want) - set(got))
        differ = sorted(k for k in set(got) & set(want) if got[k] != want[k])
        print(f'  {a.game}: applied {info["rounds_applied"]}/{n} rounds, wrote {info["files_written"]} files')
        print(f'  rebuilt {len(got)} files vs work tree {len(want)} files')
        print(f'  differing {len(differ)}   rebuild-only {len(only_got)}   work-only {len(only_want)}')
        for k in (differ + only_want + only_got)[:6]:
            print(f'     {k}')
        shutil.rmtree(tmp, ignore_errors=True)
        sys.exit(0 if not (differ or only_want) else 1)
    info = rebuild(a.run, a.game, a.round, g0, Path(a.out))
    print(f'  {a.game} r{a.round}: {info}')
