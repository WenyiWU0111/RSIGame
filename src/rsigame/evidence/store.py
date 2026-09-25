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
"""Where artifacts live, and how a consumer asks for a slice of one.

Indexed by version and probe, because those are the two things that decide
whether a stored replay answers a question about the build in front of you.

Freshness is not a score and not a decay curve. An artifact is fresh when its
version key equals the current one and historical otherwise, and the two are
never mixed: doc rule 1 is that old-version evidence is not current-version
evidence, and the failure it prevents is a summary that reads as a statement
about a build it was never about. Historical artifacts stay queryable -- they
are what attempt history and before/after comparison are made of -- but every
result says which it is.

Compaction happens here, at read time, and never by editing the artifact.
The controller wants a sentence and a few frames; the pairwise comparator wants
matched event pairs; the verifier wants something else again. Those are three
views of one object, not three objects, and the raw one is never trimmed to
fit whichever asked last.
"""
from __future__ import annotations

import json
from pathlib import Path

FRESH, HISTORICAL = 'fresh', 'historical'


def _safe(key: str) -> str:
    return ''.join(c if (c.isalnum() or c in '-_.@') else '_' for c in key)




def compact(artifact: dict, *, purpose: str = 'controller',
            max_events: int = 4) -> dict:
    """One consumer's slice. The artifact itself is not touched.

    `controller` gets the written note plus the first and last moments -- it is
    deciding whether to look again, not reading the replay frame by frame.
    `comparator` gets every event, because pairing is what it does.
    """
    evs = artifact.get('events') or []
    if purpose == 'comparator':
        shown = evs
    elif len(evs) <= max_events:
        shown = evs
    else:
        # ends plus a middle sample: the ends say where the run started and
        # finished, and a run whose ends look fine can still break in between.
        step = max(1, len(evs) // (max_events - 1))
        shown = [evs[0]] + evs[step:-1:step][:max_events - 2] + [evs[-1]]
    return {
        'probe_id': artifact['probe']['probe_id'],
        'mode': artifact['probe']['mode'],
        'version_key': artifact['version_key'],
        'round': artifact['project'].get('round'),
        'observation': artifact.get('observation'),
        'observation_why': artifact.get('observation_why'),
        'reached': artifact.get('reached') or [],
        'not_reached': artifact.get('not_reached') or [],
        'events': shown,
        'events_shown': len(shown), 'events_total': len(evs),
        'final_frame': artifact.get('final_frame'),
    }
