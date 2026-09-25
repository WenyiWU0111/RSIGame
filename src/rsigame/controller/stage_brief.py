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
"""The Development Brief for the current stage (outer guidance), if one is set.

RSIGAME_CONTROLLER_BRIEF=<path to development_brief.json>. Unset: every function returns ''
and no prompt changes. Set: the same block is shown to the planner, the repair
agent and the art round -- as the stage's objective, not as a task list; what
to investigate, what to bundle and how to implement stay the loop's own calls.
"""
import json
import os
from pathlib import Path

_HEAD = """====================
DEVELOPMENT BRIEF FOR THIS STAGE
====================
A game director played this build and set the objective for the rounds that
follow. It is high-level guidance, not a checklist: you still decide what to
investigate, what to bundle and how to implement it. When a choice is open,
prefer the one that serves this objective, and do not break what it asks to
preserve. (It may be written in another language; read it as written.)

WHAT THE DIRECTOR SAYS IT SAW IS NOT EVIDENCE. It is where to look. Nothing in
this brief counts as observed on the current build: never cite it as an
observation, never treat it as settling whether a problem exists, and never
skip observing because of it. A problem it describes is confirmed only by
evidence recorded on this build in this run -- a replay or a probe.
"""


def text() -> str:
    p = os.environ.get('RSIGAME_CONTROLLER_BRIEF', '')
    if not p:
        return ''
    b = json.loads(Path(p).read_text())
    lines = [_HEAD, f"Stage objective: {b['stage_objective']}", '', f"Why now: {b['why_now']}", '', 'Priorities:']
    lines += [f'  {i}. {x}' for i, x in enumerate(b.get('priorities') or [], 1)]
    if b.get('preserve'):
        lines += ['', 'Preserve:'] + [f'  - {x}' for x in b['preserve']]
    return '\n'.join(lines) + '\n'
