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
"""Every import of our own code must point at something that exists.

Deleting a package is easy to get almost right: a module-level import fails
loudly the moment anything touches it, but an import INSIDE A FUNCTION does
not. `evolve/stage1.py` kept importing the deleted verifier tree that way --
the package imported cleanly, the tests passed, and the only thing that would
have failed was a real run, at the moment the function was first called.

So this resolves every intra-package import statically, at any nesting depth,
against the files on disk.
"""
from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / 'src'
PKG = SRC / 'rsigame'


def _exists(dotted: str) -> bool:
    rel = Path(*dotted.split('.'))
    return (SRC / rel).with_suffix('.py').is_file() or (SRC / rel / '__init__.py').is_file()


def _resolve(node: ast.ImportFrom, path: Path) -> str | None:
    """A relative import's absolute module name, or None when it is not ours."""
    if not node.level:
        return node.module if (node.module or '').startswith('rsigame') else None
    pkg = list(path.parent.relative_to(SRC).parts)
    up = node.level - 1
    if up:
        pkg = pkg[:-up] if up <= len(pkg) else []
    return '.'.join(pkg + ([node.module] if node.module else []))


def test_every_internal_import_resolves():
    missing = []
    for p in sorted(PKG.rglob('*.py')):
        tree = ast.parse(p.read_text(errors='replace'), str(p))
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                mod = _resolve(n, p)
                if not mod or _exists(mod):
                    continue
                # `from pkg import name` where name is a module in that package
                if any(_exists(f'{mod}.{a.name}') for a in n.names):
                    continue
                missing.append(f'{p.relative_to(PKG)}:{n.lineno}: from {"." * n.level}{n.module or ""}')
            elif isinstance(n, ast.Import):
                for a in n.names:
                    if a.name.startswith('rsigame') and not _exists(a.name):
                        missing.append(f'{p.relative_to(PKG)}:{n.lineno}: import {a.name}')
    assert not missing, 'imports point at modules that do not exist:\n  ' + '\n  '.join(missing)
