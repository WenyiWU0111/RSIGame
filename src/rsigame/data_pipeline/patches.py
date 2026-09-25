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
"""Shrink a full-tree repair diff to the hunks that are actually the fix.

A repair is recorded by diffing the whole project tree before and after, which
makes the raw patch ~140 KB -- about 40k tokens for one training target. It is
not 40k tokens of fix. Measured over the collected patches, by share of bytes:

    agent scaffolding (the recorder's own directory)   95.5%
    SOURCE code                                         1.9%
    public/assets                                       1.5%
    config json                                         0.3%

The scaffolding is the agent's own trajectory recording plus a prompt file
whose diff is nothing but path rewrites -- an artefact of diffing a tree that
moved between machines, not part of any repair. Dropping it and the binary and
lockfile noise leaves the real change: p50 140,574 -> 2,535 bytes, a 55x cut.

The output stays a valid `diff -ruN`: whole per-file sections are kept or
dropped, never split, so the result still applies with `patch -p0`.

ONE CHANGE FROM THE ORIGINAL. The keep-list was web-only (.ts/.tsx/.js/.jsx
plus config .json), which silently minimised every Godot patch to the empty
string. Godot's source extensions are in the list now; `minimize` returning
empty is reported by the caller rather than written as a training target.
"""
from __future__ import annotations

import re

SEC = re.compile(r'^--- (\S+)', re.M)

KEEP = re.compile(r'\.(ts|tsx|js|jsx|json|gd|tscn|godot|tres|cfg)$', re.I)
DROP = re.compile(r'(/\.qwen/|/_trace/|/_repair_evidence/|node_modules/|'
                  r'package-lock\.json|yarn\.lock|pnpm-lock|'
                  r'\.(png|jpe?g|gif|webp|mp3|wav|ogg|ttf|woff2?|ico|import)$)', re.I)


def split_sections(txt: str) -> list[tuple[str, str]]:
    """-> [(path, section_text)], split on the `--- <path>` file headers."""
    idx = [(m.start(), m.group(1)) for m in SEC.finditer(txt)]
    out = []
    for i, (pos, path) in enumerate(idx):
        end = idx[i + 1][0] if i + 1 < len(idx) else len(txt)
        out.append((path, txt[pos:end]))
    return out


def minimize(txt: str) -> str:
    """Full-tree patch -> only the sections that carry the fix."""
    kept = []
    for path, body in split_sections(txt):
        if DROP.search(path):
            continue
        if not KEEP.search(path):
            continue
        kept.append(body)
    return ''.join(kept)
