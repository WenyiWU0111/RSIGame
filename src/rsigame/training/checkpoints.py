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
"""Keep checkpoints somewhere that outlives the job, and resume from them.

THE FAILURE THIS PREVENTS
    Checkpoints are written to node-local scratch, because that is the only
    storage fast enough to not dominate the step time. Node-local scratch dies
    with the job. Multi-card runs have disappeared five and six hours in with
    no marker of any kind, and an unsynced checkpoint is a lost run. So a
    background sync copies each new checkpoint to durable storage, and a run
    RESUMES from the newest one -- which turns a kill from "start over" into
    "lose one sync interval".

THE MARKER IS THE WHOLE DESIGN
    `.complete` is written only after the copy returns, so it certifies that
    the copy finished. Two consequences, and both have been paid for:

      * resume picks the newest checkpoint WITH a marker. A partial copy left
        by a kill mid-sync has none and must never be resumed from -- it looks
        like a checkpoint and loads as garbage.
      * the sync skips a checkpoint only if its marker exists, never merely
        because the destination directory does. Checking the directory means a
        partial copy is skipped forever.

    Checkpoints nest one level deeper than you expect: the trainer writes them
    under `<out>/<version>/checkpoint-N`, so a one-level glob matched nothing
    and a sync loop silently copied nothing for a whole run. Search two levels.
"""
from __future__ import annotations

import argparse
import re
import shutil
import time
from pathlib import Path

MARKER = '.complete'
STEP = re.compile(r'checkpoint-(\d+)$')


def _step(p: Path) -> int:
    m = STEP.search(p.name)
    return int(m.group(1)) if m else -1


def local_checkpoints(out: Path) -> list[Path]:
    """Checkpoint directories under a run's output, at either nesting depth."""
    found = [p for p in out.glob('checkpoint-*') if p.is_dir()]
    found += [p for p in out.glob('*/checkpoint-*') if p.is_dir()]
    return sorted(found, key=_step)


def newest_complete(remote: Path) -> Path | None:
    """The newest checkpoint that finished copying, or None."""
    done = [p for p in remote.glob('checkpoint-*') if p.is_dir() and (p / MARKER).exists()]
    return max(done, key=_step) if done else None


def sync_once(out: Path, remote: Path) -> list[str]:
    """Copy every not-yet-complete checkpoint to durable storage. -> names copied."""
    remote.mkdir(parents=True, exist_ok=True)
    copied = []
    for c in local_checkpoints(out):
        dst = remote / c.name
        if (dst / MARKER).exists():
            continue
        try:
            if dst.exists():
                shutil.rmtree(dst)          # a partial copy is not a base to build on
            shutil.copytree(c, dst)
        except OSError as e:
            print(f'  sync failed for {c.name}: {e}', flush=True)
            continue
        (dst / MARKER).touch()              # only now is it complete
        copied.append(c.name)
        print(f'  SYNCED {c.name}', flush=True)
    return copied


def sync_loop(out: Path, remote: Path, interval: int) -> None:
    while True:
        time.sleep(interval)
        try:
            sync_once(out, remote)
        except Exception as e:                  # noqa: BLE001 - a sync error must not kill the run
            print(f'  sync error: {e}', flush=True)


def resume_arg(remote: Path, out: Path) -> list[str]:
    """-> the trainer's resume flag, after staging the checkpoint locally.

    Returns [] when there is nothing to resume from, which is a fresh run and
    not an error.
    """
    last = newest_complete(remote)
    if last is None:
        print('RESUME none -- fresh run')
        return []
    out.mkdir(parents=True, exist_ok=True)
    staged = out / last.name
    if not staged.exists():
        print(f'RESUME from {last} (staging to local scratch)')
        shutil.copytree(last, staged)
    return ['--resume_from_checkpoint', str(staged)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog='rsigame.training checkpoints', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('sync', help='copy new checkpoints to durable storage')
    s.add_argument('--out', type=Path, required=True, help='the run output on local scratch')
    s.add_argument('--remote', type=Path, required=True, help='durable destination')
    s.add_argument('--interval', type=int, default=0,
                   help='seconds between passes; 0 syncs once and exits')
    r = sub.add_parser('resume', help='print the resume flag for the newest complete checkpoint')
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--remote', type=Path, required=True)
    a = ap.parse_args(argv)

    if a.cmd == 'sync':
        if a.interval:
            sync_loop(a.out, a.remote, a.interval)
            return 0
        n = sync_once(a.out, a.remote)
        print(f'{len(n)} checkpoint(s) synced')
        return 0
    print(' '.join(resume_arg(a.remote, a.out)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
