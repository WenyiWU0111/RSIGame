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
"""Reading and writing the one format every step in this package speaks.

Three of the converters used to open files by hand and each got a different
detail wrong: one died on the torn last line a live recording always has, one
wrote `ensure_ascii=True` and mangled the Chinese in a brief, one forgot to
create the parent directory and lost an hour of conversion to an IOError on
the final write.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator


def read(path: str | Path, *, strict: bool = False) -> Iterator[dict]:
    """Rows of a .jsonl file.

    A recording still being written ends in a torn line. That is a fact about
    live corpora, not a corrupt file, so it is skipped unless `strict`.
    """
    with Path(path).open(errors='replace') as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                if strict:
                    raise ValueError(f'{path}:{n}: not JSON')


def write(path: str | Path, rows: Iterable[dict]) -> int:
    """Rows -> .jsonl, parents created. Returns how many were written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open('w') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
            n += 1
    return n


def count(path: str | Path) -> int:
    with Path(path).open(errors='replace') as fh:
        return sum(1 for line in fh if line.strip())
