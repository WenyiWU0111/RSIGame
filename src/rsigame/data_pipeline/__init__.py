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
"""Recorded agent sessions -> a supervised fine-tuning corpus.

The development loop produces recordings: an agent is given a brief, works for
dozens of turns, and leaves behind a trajectory and a built game. This package
turns those recordings into training rows, which is what closes the recursive
loop -- the model is trained on the sessions the loop itself produced.

THE ORDER THE STEPS RUN IN

    web_trajectories    .jsonl event stream  -> one row per web session
    godot_trajectories  trajectory.json      -> one row per Godot session
        Two recorders, two formats, one output shape: ms-swift `messages`.

    decisions           whole sessions       -> one row per agent DECISION
        A session is up to 81k tokens and dozens of decisions. Training on it
        whole spends an 81k window to supervise turns a 32k window supervises
        individually. This is what makes the corpus trainable on four cards.

    plans_extract       web sessions         -> stage-2 plan rows (extracted)
    plans_distil        Godot briefs         -> stage-2 plan rows (distilled)
    plans_merge         both halves          -> one plan corpus, one prompt
    repair_records      self-refine records  -> stage-3 repair rows

    split               corpus               -> train / holdout, without leaks
    lengths             rows                 -> rows that fit the context
    filters             rows                 -> rows whose history is intact
    mix                 all of the above     -> one arm's training file
    validate            the training file    -> ship it, or say why not

THE TWO DECISIONS WORTH KNOWING BEFORE READING ANY OF IT

  * Stage shares are balanced by OPTIMIZER STEPS, not tokens. Training runs at
    batch size 1 (sequence parallelism, not data parallelism), so every row is
    exactly one step however long it is: row share IS step share. Ignoring this
    once gave plan+repair 59.4% of the updates while they carried 4.6% of the
    tokens -- a majority of training spent on single-turn, zero-tool data when
    the evaluation is a ~110-turn agentic loop. `mix` balances on steps.

  * Generation stays dominant on purpose. Stage 2 and stage 3 enter as a capped
    share of the stage-1 budget rather than as everything available, so the
    arms differ only in what was ADDED and a drop cannot be explained by having
    seen less generation data.
"""
