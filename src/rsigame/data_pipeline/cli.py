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
"""One entry point for the corpus steps, in the order they run.

    python -m rsigame.data_pipeline web        --roots ... --out gen_web.jsonl
    python -m rsigame.data_pipeline godot      --roots ... --out gen_godot.jsonl --normalize
    python -m rsigame.data_pipeline decisions  --inp gen.jsonl --out decisions.jsonl
    python -m rsigame.data_pipeline plans-extract --roots ... --out plans_web.jsonl
    python -m rsigame.data_pipeline plans-distil  --gen ... --plans ... --out plans_godot.jsonl
    python -m rsigame.data_pipeline plans-merge   --web ... --godot ... --out plans.jsonl
    python -m rsigame.data_pipeline repair     --roots ... --out repair.jsonl
    python -m rsigame.data_pipeline split      --inp ... --train ... --test ...
    python -m rsigame.data_pipeline lengths    --inp ... --out ... --manifest ...
    python -m rsigame.data_pipeline mix        --gen ... --plan ... --repair ... --out arm.jsonl
    python -m rsigame.data_pipeline arms       --arm s1s2s3 --out arm.jsonl
    python -m rsigame.data_pipeline validate   --inp arm.jsonl

Each step is also runnable on its own (`python -m rsigame.data_pipeline.mix`),
and every one of them takes `--help`.
"""
from __future__ import annotations

import sys

STEPS = {
    'web': 'web_trajectories',
    'godot': 'godot_trajectories',
    'decisions': 'decisions',
    'plans-extract': 'plans_extract',
    'plans-distil': 'plans_distil',
    'plans-merge': 'plans_merge',
    'repair': 'repair_records',
    'split': 'split',
    'lengths': 'lengths',
    'mix': 'mix',
    'arms': 'arms',
    'validate': 'validate',
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ('-h', '--help'):
        print(__doc__)
        return 0 if argv else 2
    step = argv[0]
    if step not in STEPS:
        print(f'unknown step {step!r}. One of: {", ".join(STEPS)}', file=sys.stderr)
        return 2
    from importlib import import_module
    from .. import config
    config.ensure_training()        # configs/default_training.toml -> the environment
    mod = import_module(f'.{STEPS[step]}', __package__)
    return int(mod.main(argv[1:]) or 0)


if __name__ == '__main__':
    raise SystemExit(main())
