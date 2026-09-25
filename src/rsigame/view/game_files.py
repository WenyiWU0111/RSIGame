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
"""What counts as a change to the game, as opposed to a change to the workspace.

This test existed in eight copies across this directory and the repo, each
written as some variant of

    p.startswith('src/') and p.endswith(('.ts', '.tsx', '.js'))

and every copy was wrong in the same two ways. It missed anything outside
`src/` -- and `public/` holds the level maps, which is where 297 of the
corpus's 933 recorded changes land. And it missed any extension outside the
three, so a repair that edited a map, a config or the page itself read as
having changed nothing at all.

Measured consequence: `fixed` verdicts with no diff behind them were reported
at 24%. Taken from the tree with this test, they are 6%.

A WHITELIST OF WHAT A GAME IS, not a blacklist of what scratch is. Agents
invent scratch names as they go -- `_evidence/`, `probe.py`, `shot3.png`,
`.tmp_views/` -- so a blacklist is permanently one probe behind, while the set
of things a generated game ships with is fixed and small.
"""
import os

# Every top-level entry a clean corpus game contains, minus the two that are
# regenerated: `dist` is build output and `node_modules` is dependencies.
GAME_TOP = frozenset({
    'src', 'public', 'index.html', 'package.json', 'package-lock.json',
    'tsconfig.json', 'vite.config.js', 'tailwind.config.js',
    'postcss.config.js', 'docs', 'GAME_DESIGN.md', 'README.md', 'result.json',
})

# The same whitelist, for a Godot project. Derived the same way the web one
# was: every top-level entry that appears across the 62 Godot games in the
# gamecraft-bench corpora, minus what is regenerated and minus what belongs to
# the evaluator rather than the game.
#
# `.godot` is the import cache. Godot rebuilds it on open, it is enormous, and
# a repair that "changed" it changed nothing -- it is this engine's
# `node_modules`, and it is in `_DERIVED` below rather than here.
#
# `demo_outputs` IS NOT A GAME FILE, in either engine. The task instruction
# calls it the agent's shipped input traces and the evaluator replays them:
# "The evaluator launches a fresh game per trace, replays your trace as
# synthetic mouse and keyboard input". A repair that edits one is not
# repairing the game, it is editing the exam. Keeping it out of this set is
# what makes such an edit invisible to `game_files` and catchable by the
# explicit byte-comparison the arms run against G0.
#
# The long tail seen once each -- `_dbg.gd`, `tmp_tools`, `__pycache__`,
# `telemetry.log`, `saves`, `tools_replay.py` -- is agent scratch, and is left
# out for the same reason the web set leaves out `probe.py` and `shot3.png`.
GODOT_TOP = frozenset({
    'project.godot', 'scenes', 'scripts', 'assets', 'levels',
    'Main.tscn', 'Main.gd', 'icon.svg', 'icon.svg.import',
    'default_bus_layout.tres', 'export_presets.cfg', 'README.md',
})

# Regenerated or vendored: never a repair, even under a game path.
_DERIVED = frozenset({'dist', 'node_modules', '.godot'})


# ONE SET, NOT ONE PER ENGINE PASSED IN BY THE CALLER. The two vocabularies do
# not overlap -- nothing in a web game is called `project.godot` and nothing in
# a Godot game is called `package.json` -- so the union answers both without an
# `engine` argument that every one of the eight historical copies of this test
# would have had to be taught to pass, and would have got wrong somewhere.
_ALL_TOP = GAME_TOP | GODOT_TOP


def is_game_file(p) -> bool:
    """Is this path part of the game, rather than the agent's workspace?"""
    if not isinstance(p, str) or not p:
        return False
    p = p.replace('\\', '/').lstrip('./')
    if p.startswith('/'):            # an absolute path is not a repo path
        return False
    head = p.split('/')[0]
    return head in _ALL_TOP and head not in _DERIVED


def game_files(paths) -> list:
    """The subset of `paths` that is the game."""
    return [p for p in (paths or []) if is_game_file(p)]


def scratch_files(paths) -> list:
    """The rest: what the agent wrote to look at the game with."""
    return [p for p in (paths or []) if not is_game_file(p)]


def changed_game(paths) -> bool:
    """Did this repair change the game at all?"""
    return bool(game_files(paths))
