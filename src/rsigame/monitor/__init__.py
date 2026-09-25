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
"""Global Monitor -- the outer, budget-aware control loop.

Kept apart from the inner loop's modules on purpose: this package imports from
them, never the other way round, so the development loop keeps running while
this is built.

One import is deliberately absent. `score_game` is the benchmark judge, and the
monitor must never see its output -- the design allows reusing the same replay
protocol but forbids reusing the judge's verdict. Any module here that needs a
score is doing something wrong; there is a test that asserts the import is not
present.
"""
