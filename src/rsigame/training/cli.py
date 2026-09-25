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
"""One entry point for the training steps.

    python -m rsigame.training sft         --data arm.jsonl --out /scratch/run
    python -m rsigame.training checkpoints sync --out /scratch/run --remote /durable/run
    python -m rsigame.training adapters remap --src ckpt --dst remapped
    python -m rsigame.training adapters merge --adapter remapped --out merged
    python -m rsigame.training adapters repair --merged merged
    python -m rsigame.training metrics     --run /durable/run --out loss.csv

`launch.sh` wraps the first of these for a scheduler that can only run a shell
command. Every step takes `--help`.
"""
from __future__ import annotations

import sys

STEPS = {'sft': 'sft', 'checkpoints': 'checkpoints',
         'adapters': 'adapters', 'metrics': 'metrics'}


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
