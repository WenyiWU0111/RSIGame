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
"""Normalise what a round already wrote into one ReplayArtifact.

Nothing is copied and nothing is regenerated. The frames a replay produced are
already on disk under the round that produced them, and the artifact points at
them; a store that copied them would double the disk and give two answers to
"where is frame 44".

Two honesty rules shape the fields.

VERSION. A replay is evidence about one build, and the build is identified by
hashing the tree that produced it. Live, that tree is `work/` at the moment of
the replay and the hash is exact. Reading a finished corpus, `work/` is the
FINAL tree -- round 3's build is gone -- so the hash cannot be recovered, and
this records `project_hash: None` with `version_key` falling back to the run
and round. It does not invent a hash: a wrong cache key is worse than none,
because it would hand back one build's frames as another's.

OBSERVATION. Free exploration writes a closing note; a fixed-demo replay writes
none -- what was seen ends up scattered through the verifier's prose. So a
fixed-demo artifact carries `observation: None`, which is true, rather than a
sentence assembled from somewhere else.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUN))

from rsigame.evidence.fingerprint import project_hash, script_hash  # noqa: E402

MODES = ('fixed_demo', 'extended_demo', 'custom_probe', 'free_exploration',
         'post_repair_verification')


def _rec(source: Path, probe_id: str, frozen_trace: Path | None):
    from rsigame.monitor.replay_pair import _rec_from_recorded
    return _rec_from_recorded(source, probe_id, frozen_trace)


def from_round(run_dir: str | Path, r: int, probe_id: str, *,
               mode: str = 'post_repair_verification',
               work: str | Path | None = None) -> dict | None:
    """Build an artifact from frames a finished round left behind."""
    run_dir = Path(run_dir)
    source = run_dir / f'round_{r:02d}' / 'post_replay'
    trace = (Path(work) if work else run_dir / 'work') / 'demo_outputs' / f'{probe_id}.json'
    rec = _rec(source, probe_id, trace if trace.is_file() else None)
    if rec is None:
        return None

    live_tree = Path(work) if work else None
    ph = project_hash(live_tree) if live_tree else None
    return {
        'replay_id': f'RP_{ph or run_dir.name}_r{r:02d}_{probe_id}',
        'version_key': ph or f'{run_dir.name}@r{r:02d}',
        'project': {'project_hash': ph, 'round': r, 'engine': 'godot',
                    'hash_available': ph is not None,
                    'why': None if ph else
                    'read from a finished run; the tree that produced this '
                    'build no longer exists'},
        'probe': {'probe_id': probe_id, 'mode': mode,
                  'script_hash': script_hash(trace) if trace.is_file() else None,
                  'semantic_goal': None},
        'execution': {'seed': None, 'num_actions': rec.get('n_events') or 0},
        'events': rec.get('events') or [],
        'final_frame': rec.get('final_frame'),
        'observation': None,
        'observation_why': 'a fixed-demo replay writes no per-demo summary',
        'source_dir': str(source),
    }




def probes_at(run_dir: str | Path, r: int) -> list[str]:
    d = Path(run_dir) / f'round_{r:02d}' / 'post_replay'
    return sorted(p.name[len('frames_'):] for p in d.glob('frames_*')) if d.is_dir() else []
