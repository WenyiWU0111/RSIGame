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
"""The agent side of the loop: what plays a game, what it records, what edits it.

  evidence/   plays the build and writes down what happened -- probes, traces,
              layout and binding checks, the frames a round is judged on
  evolve/     the exploration arm: sessions against a running game, the skills
              it may use, what it already knows about this game
  replay/     the browser driver that replays a web build
  generator/  the repair agent the loop shells out to, and the instruction
              shape the evidence layer reads
  evaluator/  the vision judge and the visual rubric the art round reads
  core/       the evidence dataclasses those layers pass around

This package descends from a design with three separately-packaged roles (a
generator, an evolvable diagnostic verifier, a pinned protected evaluator) and
a promotion tree over their verdicts. That design is not what this repo runs --
the inner/outer loop in `rsigame.loop` is -- and its unreachable parts have
been removed rather than left as scaffolding.
"""
