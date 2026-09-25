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
"""Which recorded rows are intact enough to train on, and how to read only those.

Rows arrive as parquet shards carrying quality flags alongside the messages.
Two things make this its own module rather than three lines in the caller:

QUALITY. A row whose history holds a UI-summary tool result is missing content
the model actually saw -- the agent read a full file, the recording kept a
one-line summary, and the target turn depends on the part that is gone.
Filtering on that has two settings and they are far apart:

    strict   no summary anywhere in the row      22,348 -> 3,385 web rows
    soft     no summary in the observation block
             immediately before the target       22,348 -> 17,872 web rows

The soft rule still excludes every row whose target depends on a lost
observation, which is the actual defect; the strict rule additionally throws
away rows whose lost observation was ten turns ago and irrelevant. Soft is the
default for that reason, and the flag is explicit so a paper can state which
ran. (The Godot generation shards carry no summaries at all, so the setting
does not apply to them.)

For agentic Godot repair there is a third setting: `tool_results_source ==
"full_chat_log"` keeps only the recording arm that logged results in full.

MEMORY. The `messages` column IS the corpus. Reading the table whole to filter
it has been enough to get the process killed with no traceback on a machine
with a memory cap. So the flags are read first as their own narrow table, the
sample is chosen as INDICES, and only the chosen rows are ever materialised --
streamed batch by batch, keeping just the hits.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

FLAG_COLS = ('in_s1_holdout', 'row_has_summary_only', 'target_follows_summary_only',
             'tool_results_source', 'n_tokens', 'engine', 'task')


def dataset_dir() -> Path:
    """Where the published shards are. Machine-local, so it is configuration."""
    v = os.environ.get('RSIGAME_CORPUS_DATASET_DIR')
    if not v:
        raise RuntimeError(
            'RSIGAME_CORPUS_DATASET_DIR is not set (the directory holding the '
            'downloaded corpus shards). Set it in configs/default.toml under '
            '[corpus], or export it.')
    return Path(v)


def _dataset(pattern: str, root: Path | None = None):
    import pyarrow.parquet as pq
    root = root or dataset_dir()
    files = sorted(glob.glob(str(root / pattern)))
    if not files:
        raise SystemExit(f'no parquet matching {pattern} under {root}')
    return pq.ParquetDataset(files)


def clean_indices(pattern: str, *, soft_summary: bool = True,
                  require_full_log: bool = False, root: Path | None = None):
    """Indices of the rows that survive the quality filters.

    Reads the flag columns ONLY -- see the memory note above.
    """
    import numpy as np
    ds = _dataset(pattern, root)
    have = set(ds.schema.names)
    cols = [c for c in FLAG_COLS if c in have]
    tbl = ds.read(columns=cols) if cols else ds.read(columns=[ds.schema.names[0]])
    keep = np.ones(tbl.num_rows, bool)
    if 'in_s1_holdout' in cols:
        keep &= ~np.array(tbl.column('in_s1_holdout').to_pylist(), bool)
    if require_full_log and 'tool_results_source' in cols:
        keep &= np.array(tbl.column('tool_results_source').to_pylist()) == 'full_chat_log'
    elif soft_summary and 'target_follows_summary_only' in cols:
        keep &= ~np.array(tbl.column('target_follows_summary_only').to_pylist(), bool)
    elif 'row_has_summary_only' in cols:
        keep &= ~np.array(tbl.column('row_has_summary_only').to_pylist(), bool)
    return np.nonzero(keep)[0]


def materialise(pattern: str, wanted_idx, root: Path | None = None) -> list[dict]:
    """Load only the chosen rows, streaming batches so peak memory stays small."""
    ds = _dataset(pattern, root)
    have = set(ds.schema.names)
    cols = ['messages'] + [c for c in ('tools', 'n_tokens') if c in have]
    wanted = {int(i) for i in wanted_idx}
    out: list[dict] = []
    base = 0
    for frag in ds.fragments:
        for batch in frag.to_batches(columns=cols):
            n = batch.num_rows
            hits = [i for i in range(n) if base + i in wanted]
            if hits:
                d = batch.to_pydict()
                for i in hits:
                    r: dict = {'messages': d['messages'][i]}
                    if 'tools' in d and d['tools'][i]:
                        r['tools'] = d['tools'][i]
                    r['_n_tokens'] = d['n_tokens'][i] if 'n_tokens' in d else None
                    out.append(r)
            base += n
    return out
