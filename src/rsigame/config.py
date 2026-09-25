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
"""The one place configuration is decided.

Before this, a value could come from a shell script that every launcher had to
source, from a bare `export`, or from an `os.environ.get(..., default)` buried
in any of 140 modules -- and which one won took reading the code. 106
variables, no file listing them, and paths guessed with `cd ..`.

Three layers now, each overriding the one above:

    configs/default.toml            every knob and its default, with comments
    -c configs/experiment/X.toml    what this experiment changes
    environment variables           a temporary override while a run is up

Secrets are never in any toml: they come from `.env` (never committed) or from
the environment.

NAMING. Every variable this project owns is `RSIGAME_<SECTION>_<KEY>`, matching
`[section] key` in the toml exactly, so a name says where to change it. The
exceptions are names other programs read and we only pass through:
`GAMECRAFT_BENCH*` (the benchmark), `OPENGAME_*` (the OpenGame agent),
`OPENAI_*` / `OPENROUTER_API_KEY` (provider conventions).

HOW IT REACHES THE CODE. `load()` resolves the three layers and exports the
result into `os.environ`, so the existing `os.environ.get(...)` call sites keep
working unchanged and an operator can still override one knob mid-run. Call
sites move to `cfg` gradually; nothing has to move at once for the config to
become readable.
"""
from __future__ import annotations

import json
import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / 'configs'
_ENVMAP: dict[str, str] = json.loads((Path(__file__).parent / '_envmap.json').read_text())

# Two of these are other people's names -- OpenRouter's and OpenAI's -- and stay
# spelled the way every other tool spells them. The rest are ours: RSIGAME_<what>.
SECRETS = ('OPENROUTER_API_KEY', 'OPENAI_API_KEY', 'RSIGAME_JUDGE_API_KEY',
           'RSIGAME_MONITOR_API_KEY', 'RSIGAME_REPAIR_API_KEY',
           'RSIGAME_GENERATOR_API_KEY', 'RSIGAME_MODELS_LOCAL_API_KEY')


class ConfigError(RuntimeError):
    """Something is missing or wrong, said before a 30-round run starts rather
    than at round 12 -- which is when these used to surface."""


@dataclass
class Config:
    data: dict = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)

    def get(self, dotted: str, default=None):
        sec, _, key = dotted.partition('.')
        return self.data.get(sec, {}).get(key, default)

    def export(self) -> int:
        """Resolved values -> os.environ. An existing env var always wins."""
        n = 0
        for dotted, env in _ENVMAP.items():
            v = self.get(dotted)
            if v is None or env in os.environ:
                continue
            os.environ[env] = str(int(v)) if isinstance(v, bool) else str(v)
            n += 1
        return n


def _read(p: Path) -> dict:
    if not p.is_file():
        raise ConfigError(f'config file not found: {p}')
    return tomllib.loads(p.read_text())


def _merge(base: dict, over: dict) -> dict:
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in base.items()}
    for k, v in over.items():
        out[k] = {**out.get(k, {}), **v} if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_env_file(path: Path | None = None) -> int:
    """`.env` -> os.environ, without overwriting anything already set.

    Read at load time, not at import time: a module that read the file while
    being imported used to take the whole package down on a machine that had
    no .env yet.
    """
    path = path or (ROOT / '.env')
    if not path.is_file():
        return 0
    n = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v; n += 1
    return n


def load(overlay: str | Path | None = None, *, export: bool = True,
         validate_paths: bool = True) -> Config:
    cfg = Config(_read(CONFIGS / 'default.toml'), [str(CONFIGS / 'default.toml')])
    if overlay:
        p = Path(overlay)
        if not p.is_file():
            p = CONFIGS / 'experiment' / f'{overlay}.toml'
        cfg.data = _merge(cfg.data, _read(p)); cfg.sources.append(str(p))
    load_env_file()
    if export:
        cfg.export()
    if validate_paths:
        check(cfg)
    return cfg


def check(cfg: Config) -> list[str]:
    """Fail loud and early. Returns the problems; raises if any are fatal."""
    problems = []
    for dotted, label in (('paths.bench', 'GameCraft-Bench checkout'),
                          ('paths.bench_godot_bin', 'Godot binary')):
        v = cfg.get(dotted) or os.environ.get(_ENVMAP.get(dotted, ''), '')
        if not v:
            problems.append(f'{dotted} is unset ({label})')
        elif not Path(v).exists():
            problems.append(f'{dotted} points at nothing: {v}')
    if not any(os.environ.get(k) for k in SECRETS):
        problems.append('no model API key in the environment or .env '
                        f'(one of: {", ".join(SECRETS)})')
    if problems:
        raise ConfigError('configuration is not usable:\n  - ' + '\n  - '.join(problems))
    return problems


_ENSURED = False


def ensure(overlay: str | Path | None = None) -> None:
    """Load the configuration if no one has yet. Safe to call from any entry.

    Every module that can be started with `python -m rsigame.<x>` calls this,
    because the alternative is what a smoke test found: running the loop
    directly resolved none of the 110 settings, the model name for one call
    came back empty, and `_client()` refused to guess -- at the first round,
    with a stack trace three frames deep in an unrelated module.
    """
    global _ENSURED
    if _ENSURED or os.environ.get('RSIGAME_CONFIG_LOADED'):
        return
    try:
        load(overlay, validate_paths=False)
    except ConfigError:
        return                    # an entry point that needs a path will say so itself
    _ENSURED = True
    os.environ['RSIGAME_CONFIG_LOADED'] = '1'


def main(argv: list[str] | None = None) -> int:
    """`python -m rsigame.config` prints what a run would actually use."""
    import argparse
    ap = argparse.ArgumentParser(description='show the resolved configuration')
    ap.add_argument('-c', '--config', help='experiment overlay (name or path)')
    ap.add_argument('--no-check', action='store_true')
    a = ap.parse_args(argv)
    try:
        cfg = load(a.config, export=False, validate_paths=not a.no_check)
    except ConfigError as e:
        print(e, file=sys.stderr); return 2
    print('sources:', ' + '.join(cfg.sources))
    for sec in sorted(cfg.data):
        print(f'[{sec}]')
        for k, v in sorted(cfg.data[sec].items()):
            print(f'  {k:<26} {v!r:<28} ({_ENVMAP.get(f"{sec}.{k}", "-")})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
