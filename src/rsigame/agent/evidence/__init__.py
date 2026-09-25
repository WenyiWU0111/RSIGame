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
"""Three readers the exploration arm asks a game tree for.

  scan_gdscript_bindings   which input a .gd script binds, and to what
  scan_bindings            the same for a web build's sources
  hud_numbers              the numbers a HUD is showing right now

This was a whole evidence-acquisition package once -- an orchestrator that
opened a game and ran a dozen probes into one Evidence object, over a fact
framework with entities, provenance and invariant audits. Nothing called the
orchestrator; these three are what explore() actually reads.
"""
