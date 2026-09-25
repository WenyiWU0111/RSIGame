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
"""Where things are, asked once instead of written down 27 times.

Every one of these used to be an absolute path literal inside whichever
module needed it. Two consequences, and the second is the bad one:

  * the code only ran on one machine, and
  * some call sites wrapped the lookup in `try/except` and carried on when the
    path was missing -- `evolve/skills.py` fell back to "every file is a game
    file" and changed what the loop counted as an empty repair, silently.

So these raise. A path that is not configured is a configuration error, not a
default.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # the repo


def _need(env: str, what: str) -> Path:
    v = os.environ.get(env)
    if not v:
        raise RuntimeError(
            f'{env} is not set ({what}). Set it in configs/default.toml under '
            f'[paths], or export it. `python -m rsigame.config` shows what is resolved.')
    return Path(v)


def bench() -> Path:
    """The GameCraft-Bench checkout (tasks, rubrics, replay harness)."""
    return _need('GAMECRAFT_BENCH', 'the GameCraft-Bench checkout')


def bench_tasks() -> Path:
    v = os.environ.get('GAMECRAFT_BENCH_TASKS')
    return Path(v) if v else bench() / 'tasks'


def godot_bin() -> Path:
    return _need('GAMECRAFT_BENCH_GODOT_BIN', 'the Godot binary used to build and replay')


def run_root() -> Path:
    """Where runs, snapshots and scores are written. Not inside the repo."""
    v = os.environ.get('RSIGAME_PATHS_RUN')
    return Path(v) if v else ROOT / 'runs'


def agent_repo() -> Path:
    """Used to be a sibling checkout on PYTHONPATH; it is this package now."""
    return Path(__file__).resolve().parent


def env_file() -> Path:
    return Path(os.environ.get('RSIGAME_PATHS_ENV') or ROOT / '.env')


class LazyPath(os.PathLike):
    """A path resolved on first use, not at import.

    Module-level `BENCH = paths.bench()` would make importing the package fail
    on a machine that has not configured the bench yet -- and importing is what
    `--help`, the tests and the config dump all do. Resolution is deferred to
    the first real use, where a missing path is genuinely fatal.
    """

    __slots__ = ('_fn',)

    def __init__(self, fn):
        self._fn = fn

    def _p(self) -> Path:
        return self._fn()

    def __fspath__(self) -> str:
        return str(self._p())

    def __str__(self) -> str:
        return str(self._p())

    def __repr__(self) -> str:
        return f'LazyPath({self._fn.__name__})'

    def __truediv__(self, other):
        return self._p() / other

    def __getattr__(self, name):
        return getattr(self._p(), name)


def lazy_bench() -> LazyPath:
    return LazyPath(bench)


def lazy_bench_tasks() -> LazyPath:
    return LazyPath(bench_tasks)
