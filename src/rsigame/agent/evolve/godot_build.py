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
"""Does this Godot project still load, after something edited it.

`structural_veto` needs "the child builds" to mean something on Godot, and
the obvious answer is wrong. Measured on hajimi 2026-08-25: append

    func _broken_syntax(:
        this is not valid gdscript ===

to a script, run `godot --headless --path <dir> --quit`, and the process
exits **0**. The parse failure is on stdout --

    SCRIPT ERROR: Parse Error: Expected parameter name.
    ERROR: Failed to load script "res://scripts/Main.gd" with error "Parse error"

-- and nowhere else. A build gate reading the exit code would have called
that child sound, so a repair that broke the game outright could have been
promoted with `child_builds: True` behind it.

So the check reads the OUTPUT. Exit code is recorded but never decides,
except that a non-zero one is still a failure -- it just cannot be trusted
in the other direction.

Deliberately narrow. This answers "does the project load and do its scripts
parse", which is what the structural veto asks. It is not a test run and not
a quality signal; whether the game is any good is the verifier's question,
asked by playing it.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

GODOT_BIN = 'godot'
# Lines that mean the project did not come up cleanly. `SCRIPT ERROR` covers
# parse and runtime script failures; the resource-load line catches a scene
# referencing something that no longer exists, which an edit can easily cause
# and which no parse check would see.
FATAL = re.compile(
    r'SCRIPT ERROR|Parse Error|Failed to load script|'
    r'Failed loading resource|Cannot open file|'
    r'Script inherits from native type .* so it can\'t be instantiated',
    re.I)


def check_project(src, *, timeout_s: int = 120,
                  godot_bin: str = GODOT_BIN) -> dict:
    """Load the project headless and read what it says about itself.

    Returns `{'builds': bool, 'why': str, 'returncode': int, 'errors': [...]}`.
    `builds` is False when any fatal line appears OR the process fails to
    run -- never merely because the exit code was zero.
    """
    src = Path(src)
    if not (src / 'project.godot').is_file():
        return {'builds': False, 'returncode': None, 'errors': [],
                'why': f'no project.godot in {src}, so this is not a Godot '
                       f'project and nothing was checked'}
    try:
        p = subprocess.run(
            [godot_bin, '--headless', '--path', str(src), '--quit'],
            capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {'builds': False, 'returncode': None, 'errors': [],
                'why': f'the project did not finish loading within '
                       f'{timeout_s}s; a load that hangs is not a load that '
                       f'succeeded'}
    except FileNotFoundError:
        return {'builds': False, 'returncode': None, 'errors': [],
                'why': f'{godot_bin!r} is not on PATH; this is an '
                       f'infrastructure failure, not a fact about the game'}

    out = f'{p.stdout}\n{p.stderr}'
    bad = [ln.strip() for ln in out.splitlines() if FATAL.search(ln)]
    if bad:
        return {'builds': False, 'returncode': p.returncode,
                'errors': bad[:8],
                'why': f'the project loaded with {len(bad)} script/resource '
                       f'error(s); the exit code was {p.returncode} and does '
                       f'NOT reflect them'}
    if p.returncode != 0:
        return {'builds': False, 'returncode': p.returncode, 'errors': [],
                'why': f'godot exited {p.returncode} with no error line to '
                       f'explain it'}
    return {'builds': True, 'returncode': 0, 'errors': [],
            'why': 'the project loaded and every script parsed'}
