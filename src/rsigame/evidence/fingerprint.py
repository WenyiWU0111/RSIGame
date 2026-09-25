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
"""What makes two replays comparable.

A replay is about one build of one game driven by one script. Two replays are
interchangeable only if both halves match, so the fingerprint hashes both.

The project hash deliberately covers MORE than the code snapshot a repair
attempt records.
That function exists to roll back files that can stop the project loading, so
it skips assets on purpose -- a round that adds a sprite and breaks a script
should keep the sprite. A cache key cannot skip them: a round that changes only
a texture produces a visibly different replay and an identical code hash, and
the cache would hand back the old frames as if they were current.

Excluded instead: `.godot/` (an import cache, rebuilt per machine and not part
of the game), `demo_outputs/` (the scripts, hashed separately as the probe half
of the fingerprint), and the usual VCS/tool directories.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SKIP_DIRS = {'.godot', '.git', '.import', 'node_modules', '_repair_evidence',
             '__pycache__', 'demo_outputs'}


def _walk(root: Path):
    for p in sorted(root.rglob('*')):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(root).parts):
            continue
        yield p


def project_hash(work: str | Path) -> str:
    """A content hash of everything that can change what a replay looks like."""
    work = Path(work)
    h = hashlib.sha256()
    for p in _walk(work):
        rel = str(p.relative_to(work)).replace('\\', '/')
        h.update(rel.encode())
        h.update(b'\0')
        try:
            h.update(hashlib.sha256(p.read_bytes()).digest())
        except OSError:
            h.update(b'<unreadable>')
    return h.hexdigest()[:16]


def script_hash(trace_path: str | Path) -> str:
    """The demo script's inputs, not its formatting.

    Hashing the file bytes would make a reindented script a different probe.
    What identifies a probe is the events it sends and how long it runs.
    """
    t = json.loads(Path(trace_path).read_text())
    body = {'duration_frames': t.get('duration_frames'),
            'scenario': t.get('scenario'),
            'events': t.get('events') or []}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()[:16]


def fingerprint(work: str | Path, probe_id: str, trace_path: str | Path,
                *, engine: str = 'godot', mode: str = 'fixed_demo') -> dict:
    return {'project_hash': project_hash(work), 'probe_id': probe_id,
            'script_hash': script_hash(trace_path), 'engine': engine,
            'mode': mode,
            # No seed exists: `replay_trace` takes none and the games do not
            # seed their own RNG. Recorded as null rather than omitted so a
            # later reader is not left wondering whether it was forgotten.
            'seed': None}


def matches(a: dict, b: dict) -> bool:
    """Interchangeable, in the only sense the harness can guarantee."""
    return all(a.get(k) == b.get(k)
               for k in ('project_hash', 'probe_id', 'script_hash', 'engine'))
