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
"""Supervised fine-tuning: the corpus goes in, an adapter comes out.

WHAT ONE RUN IS
    LoRA (rank 16, alpha 32, rsLoRA) on every linear module of a 27B base,
    with a 4-bit NF4 resident base and bf16 compute, one epoch, batch size 1,
    sequence parallelism across four cards. The corpus is agent sessions, so a
    row is a conversation and the loss falls on assistant turns.

THE THREE ARMS
    `s1` generation only, `s1s2` + plan, `s1s2s3` + repair. They share the same
    generation rows by construction (see `data_pipeline.arms`), so the
    comparison is about what was ADDED.

WHY FOUR CARDS AND NOT EIGHT
    The model has 4 KV heads, and ms-swift sets sp = gcd(kv_heads, world_size)
    with rp = world_size / sp. Four cards give sp=4, rp=1: pure Ulysses, no ring
    attention. Eight give sp=4, rp=2, which needs ring attention -- unimplemented
    for this model's linear-attention blocks, so it refuses. More cards cannot
    shorten the per-rank sequence here; they just fail differently.

WHAT THE MODULES ARE
    sft          build and run one training job from config
    launch.sh    the portable single-node entry point that sft.py invokes
    checkpoints  keep checkpoints somewhere that outlives the job, and resume
    adapters     remap, merge and repair the saved adapter for serving
    metrics      the run's event files -> a loss curve you can plot

WHAT IS DELIBERATELY NOT HERE
    Anything that talks to one particular cluster: queue names, image
    registries, node babysitters, an out-of-band job watcher. A run is started
    by `launch.sh` on whatever machine has the cards, and everything it needs
    is an argument or a config key.
"""
