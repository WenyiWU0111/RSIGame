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
"""A run's event files -> a plain CSV of every scalar it logged.

The training image has no TensorBoard and no tbparse, and installing either on
a machine with no outbound network to reach a package index is not a thing you
want between you and a loss curve. So this parses the TFRecord container and
the Event message directly. The framing is:

    uint64 length | uint32 crc(length) | bytes data | uint32 crc(data)

and inside `data` only Event.step and Event.summary.value[].{tag, simple_value}
are needed. Rather than depend on a protobuf schema, it walks the wire format:
a small, stable subset, and the alternative is a dependency that has to be
installed on every machine a run might land on.

Output is one row per scalar -- step, tag, value -- so loss, token accuracy,
seconds per iteration and grad-norm all land in one file that needs no server
to read. The CSV is what the paper's loss curves are drawn from.
"""
from __future__ import annotations

import argparse
import csv
import struct
from pathlib import Path


def records(path: Path):
    """Yield raw Event payloads from one TFRecord file."""
    with open(path, 'rb') as f:
        while True:
            head = f.read(8)
            if len(head) < 8:
                return
            (ln,) = struct.unpack('<Q', head)
            f.read(4)                      # crc of the length
            data = f.read(ln)
            f.read(4)                      # crc of the data
            if len(data) < ln:
                return                     # a torn tail: the run is still writing
            yield data


def varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = res = 0
    while i < len(buf):
        b = buf[i]
        i += 1
        res |= (b & 0x7F) << shift
        if not b & 0x80:
            return res, i
        shift += 7
    return res, i


def fields(buf: bytes):
    """Yield (field_number, wire_type, value) over one protobuf message."""
    i = 0
    while i < len(buf):
        key, i = varint(buf, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = varint(buf, i)
            yield fn, wt, v
        elif wt == 2:
            ln, i = varint(buf, i)
            yield fn, wt, buf[i:i + ln]
            i += ln
        elif wt == 5:
            yield fn, wt, buf[i:i + 4]
            i += 4
        elif wt == 1:
            yield fn, wt, buf[i:i + 8]
            i += 8
        else:
            return


def parse_event(data: bytes) -> tuple[int | None, list[tuple[str, float]]]:
    """-> (step, [(tag, value)])"""
    step: int | None = None
    out: list[tuple[str, float]] = []
    for fn, wt, v in fields(data):
        if fn == 2 and wt == 0:                     # Event.step
            step = v
        elif fn == 5 and wt == 2:                   # Event.summary
            for sfn, swt, sv in fields(v):
                if sfn == 1 and swt == 2:           # Summary.value
                    tag = val = None
                    for vfn, vwt, vv in fields(sv):
                        if vfn == 1 and vwt == 2:
                            tag = vv.decode('utf-8', 'replace')
                        elif vfn == 2 and vwt == 5:
                            (val,) = struct.unpack('<f', vv)
                    if tag is not None and val is not None:
                        out.append((tag, val))
    return step, out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog='rsigame.training metrics', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True, help='a run directory, searched recursively')
    ap.add_argument('--out', type=Path, required=True, help='the CSV to write')
    a = ap.parse_args(argv)

    evs = sorted(a.run.rglob('events.out.tfevents*'))
    print(f'event files: {len(evs)}')

    rows: list[tuple] = []
    tags: dict[str, int] = {}
    for ev in evs:
        run = ev.parent.parent.name
        n = 0
        for data in records(ev):
            try:
                step, kv = parse_event(data)
            except Exception:                       # noqa: BLE001 - a torn record ends this file
                continue
            if step is None:
                continue
            for tag, val in kv:
                rows.append((run, step, tag, val))
                tags[tag] = tags.get(tag, 0) + 1
                n += 1
        print(f'  {run:-26s} {n:6d} scalars')

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'step', 'tag', 'value'])
        w.writerows(rows)
    print(f'\nwrote {len(rows)} rows -> {a.out}')
    print(f'tags: {sorted(tags.items(), key=lambda kv: -kv[1])[:10]}')
    return 0 if rows else 1


if __name__ == '__main__':
    raise SystemExit(main())
