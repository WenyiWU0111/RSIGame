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
"""Replay artifacts, their fingerprints, and the store that indexes them.

Separate from `monitor/` and `controller/` because both consume it. Like them,
nothing here may import `score_game`: the rubric is held out.

Read the cost note before treating any of this as a performance feature. Across
1471 rounds, game execution is 28% of wall clock and the repair agent's own
loop is 72%; the reusable slice is the free exploration, 10.5%, and only on the
rounds a controller judges reusable. This exists because the controller cannot
answer "is the existing evidence enough" without it, and because re-running a
replay you already have is measurably WORSE than reusing it -- two replays of
one build differ by more than some real changes do.
"""
