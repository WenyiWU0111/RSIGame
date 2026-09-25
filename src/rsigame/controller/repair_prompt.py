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
"""What the repair agent is told, in the redesigned loop.

Four things differ from the prompt this replaces, and each is a fix for
something measured rather than a preference.

WHO SAW WHAT. The old opening said "You played it a moment ago and this is what
you concluded was wrong with it." In this loop that is simply false: a play
agent drove the game, a grounding stage decided what the frames support, and a
ranking stage chose what gets the budget. Telling the repair agent those are
its own conclusions invites it to top them up from a memory it does not have.

ART. The old asset block opened by forbidding most of what it described --
"use it ONLY when the defect is that an asset itself is wrong or missing", "art
that is already right does not need regenerating" -- and that paragraph was
24% of a round-1 prompt. The packet now carries a standing art section, so the
prompt would have held two opposite instructions, and an agent reading both
follows the prohibition because it is the more specific. The Godot mechanics
below it are kept verbatim: they are hard-won and still true.

HISTORY. The old block listed every file every round touched. By round 13 of a
real run that was 8303 characters -- 60% of the whole prompt -- and it said, in
substance, "main.gd was edited thirteen times and it always compiled". The task
itself was down to 4.9%. What belongs there is what was attempted and what came
of it, which is shorter and is the thing a repair agent actually needs.

REPORTING. The packet asks the agent to say what it skipped and why. Nothing
used to catch that answer, so a skip was indistinguishable from an oversight
and the next round could not tell them apart. The prompt now ends by asking for
one JSON block, which the harness reads.
"""
from __future__ import annotations

import json
import os

_ART = """
ART AND EFFECTS. You have `generate_game_assets`. This game is judged partly on
whether it looks and feels made rather than assembled, and that work is never
finished -- the reader that looks at every build has not once called it done.
So this is available every round, not only when something is broken, and the
packet above says what is worth doing now.

Each image costs real money and takes minutes, so generate the few that carry
the game rather than a set. Beyond that, how far to take it is yours to judge
with the code in front of you.

THIS IS A GODOT PROJECT, and four things differ from the web games the tool was
written for:

  - Write into the tree's own `assets/<kind>/` -- pass it as `output_dir_name`.
  - GODOT CANNOT LOAD A RESOURCE IT HAS NOT IMPORTED. After adding files, run

        godot --headless --path . --import --quit

    or `load()` returns null at runtime. Nothing warns you: the build still
    passes, and the sprite simply never appears.
  - The code has to use it -- `load("res://assets/...")` and `draw_texture` or
    `draw_texture_rect` -- and you must DELETE the `draw_rect` / `draw_circle`
    that was drawing the same thing. They coexist happily and the procedural
    one usually wins, which leaves a correct new file sitting unused while the
    game looks exactly as it did.
  - Check your own work by playing the build AFTER the import, not before.

PUT THEM WHERE THE GAME ALREADY LOOKS. `output_dir_name` must be a directory
this project ALREADY loads from -- grep a `res://` path out of the source and
use that. Do not invent `assets/v2/`, `assets/sprites/`, `assets/blocks/` or
any other new folder: the tool will happily write there, the code keeps loading
the old path, and the round ends with two copies of one sprite and no visible
change. Measured on the first three art rounds -- 49 files generated, 26 loaded,
and every one of the 23 misses was a second copy in a folder nobody reads.

CHECK EVERY KEY BEFORE YOU STOP. For each key you generated, grep the source
for its filename. If it is not there, the file changes nothing on screen. That
check costs one call and is the difference between a round that did something
and a round that spent money.

ASK FOR SEVERAL AT ONCE. `assets` is a list. One call generates the whole set,
and generating them one at a time spends a tool call each on work that costs
the same together. Split into two calls only if the set is large: animations
and backgrounds in one, the rest in the other.

A REQUEST THAT NAMES PARTS IS ONE REQUEST. A character is a run cycle, an
idle, a jump, a facing per direction -- nine images of one figure, asked for
together under one `style_anchor`, in one call, for one wait. Asked for
separately they come back as nine different characters. So a nine-frame
character is not nine times the work of a background and is not a reason to
leave the character as it is: measured across six art rounds on two games, the
player character was never once replaced while the UI around it was redrawn
four times.

ALPHA IS FOR THE SHAPE OF A THING, NOT FOR A LAYER OVER THE GAME.

A sprite's transparency is its outline -- a pixel is either part of the
critter or it is not -- and that is what makes it a sprite rather than a
rectangle. Keep it. What does not work is using a mostly-empty image as a
LAYER: tiling a sparse texture across the play field to add grain, or laying a
translucent sheet over the board to tint it. One game did exactly that with a
grass tile over its 9x5 lawn and the whole field rendered as a solid white
block -- forty per cent of the screen -- through a build that compiled, an
import that reported no failures, and a reader that called it playable.

So:

  - A thing that sits ON the game (a critter, a block, an icon, a plant) is a
    cutout. Alpha says where its edge is. That is right.
  - The ground, the board, the sky, a panel -- anything that fills an area --
    is OPAQUE. If you want mowed stripes in the lawn or grain in the floor,
    paint them INTO that opaque image, do not lay a second image over it.
  - Do not tile a texture across the play field. Do not draw a translucent
    rectangle or texture over a region to tint it.

THE TOOL DOES NOT DECIDE THE SIZE, YOU DO, IN THE CODE.
Its output is a fixed size -- `image` and `animation` come back 386x560,
`background` is the only kind that takes a `resolution`. Do not pass `size` to
an `image`; it is ignored and the call errors on some shapes.

The right size is already written down in the line you are replacing. A
`draw_rect(Rect2(x, y, w, h), colour)` states exactly where the thing goes and
how big it is, and that rectangle is correct -- it has been running in the
game. Keep it and draw into it:

    draw_texture_rect(tex, Rect2(x, y, w, h), false)

or, for a node, set `Sprite2D.scale` (or `region_rect`) from the same numbers.
An image dropped in at its native 386x560 lands enormous and off-centre, which
looks like a broken sprite and is really an unset rectangle.

Pixel art needs `texture_filter = TEXTURE_FILTER_NEAREST` on the node or in
`_draw`, or scaling turns it to mush.
"""

# THE SAME BLOCK FOR A PHASER GAME. Everything above that is about the tool and
# about art direction is engine-neutral and kept word for word; only the
# mechanics differ -- where files go, how the code picks them up, what drawing
# they replace, and how to size them. Selected by `engine`, so a Godot prompt
# is byte-identical to what it was.
_ART_WEB = _ART.split('THIS IS A GODOT PROJECT')[0] + """THIS IS A PHASER (web) PROJECT, built with `npm run build`:

  - Write into `public/assets/` -- the tool's default. It adds every key it
    generates to `public/assets/asset-pack.json`, which the Preloader scene
    loads (`this.load.pack(...)`), so a new file is available BY ITS KEY with no
    loader edit.
  - The code has to use the KEY -- `this.add.image(x, y, 'key')`,
    `this.add.sprite(...)`, `setTexture('key')`, or an animation built from the
    frame keys -- and you must DELETE the `this.add.rectangle(...)` /
    `graphics.fillRect` / `fillCircle` that was drawing the same thing. They
    coexist happily and the procedural one is usually drawn on top, which
    leaves a correct new file sitting unused while the game looks exactly as it
    did.
  - `public/` is copied into `dist/` by the build, and the game that is played
    and judged is `dist/`. Run `npm run build` after adding files and after
    wiring them, and check your own work by playing the build AFTER that.

PUT THEM WHERE THE GAME ALREADY LOOKS. Keep `output_dir_name` at the directory
this project ALREADY loads from -- the one holding `asset-pack.json`. Do not
invent `public/assets/v2/`, `sprites/` or any other new folder: the tool will
happily write there, the pack the game loads never hears of it, and the round
ends with two copies of one sprite and no visible change. Measured on the first
three art rounds -- 49 files generated, 26 loaded, and every one of the 23
misses was a second copy in a folder nobody reads.

CHECK EVERY KEY BEFORE YOU STOP. For each key you generated, grep `src/` for
it. If it is not there, the file changes nothing on screen. That check costs one
call and is the difference between a round that did something and a round that
spent money.

""" + _ART.split('ASK FOR SEVERAL AT ONCE.')[1].split('THE TOOL DOES NOT DECIDE THE SIZE')[0].join(['ASK FOR SEVERAL AT ONCE.', '']) + """THE TOOL DOES NOT DECIDE THE SIZE, YOU DO, IN THE CODE.
Its output is a fixed size -- `image` and `animation` come back 386x560,
`background` is the only kind that takes a `resolution`. Do not pass `size` to
an `image`; it is ignored and the call errors on some shapes.

The right size is already written down in the line you are replacing. A
`this.add.rectangle(x, y, w, h, colour)` states exactly where the thing goes and
how big it is, and that rectangle is correct -- it has been running in the
game. Keep it and draw into it:

    this.add.image(x, y, 'key').setDisplaySize(w, h)

Mind the origin: a rectangle and an image both default to a centred origin,
while `graphics.fillRect(x, y, w, h)` is measured from its top-left corner --
use `.setOrigin(0, 0)` when replacing one of those. An image dropped in at its
native 386x560 lands enormous and off-centre, which looks like a broken sprite
and is really an unset size.

Pixel art needs `pixelArt: true` in the game config, or scaling turns it to
mush.
"""


_REPORT = """
WRITE THIS AS SOON AS YOU KNOW IT, not at the end. Sessions run out of tool
calls and out of time -- all four of the first real rounds did, and none of
them reached this instruction. One JSON block, and you can revise it by
writing another later. Use the issue ids
from the packet above; if the packet named no issues, `primary` is null and the
art entry carries what you did -- do not invent an id for it.

```json
{"primary": {"issue_id": "<id from the packet>", "attempted": true},
 "concurrent": [{"issue_id": "<id>", "attempted": true},
                {"issue_id": "<id>", "attempted": false,
                 "reason": "<why not>"}],
 "art": {"attempted": true, "files": ["assets/..."]},
 "note": "<what you worked out about this codebase that the next round should
           not have to find again -- where a thing lives, what the fix
           touched, what turned out not to be the cause>",
 "todo": "<what you did not get to, and where you had got to>"}
```

If you cannot find where the behaviour lives -- say by your tenth call or so
-- stop and say that, instead of spending the round looking. Write:

```json
{"repair_status": "blocked",
 "reason": "<what you could not localise>",
 "files_inspected": ["main.gd", "..."],
 "needs": "<what would let the next round find it>"}
```

Admitting at call ten that it cannot be located is worth more than reaching
call twenty-six having changed nothing: the round ends early, and what you
looked at goes to whoever picks the next target.

`note` and `todo` are the round's memory. Every round so far has started by
reading the same files again -- one spent all twenty-six of its calls doing
that and changed nothing. What you write there is what the next round is told,
so name files and functions, not impressions.

Say `attempted` for what you tried, not for what you believe worked -- whether
it worked is decided by replaying the game, not by you. A skip with a reason is
a decision the next round can act on; a silent one cannot be told from having
forgotten.
"""

_HEAD = """You are improving a small game. This is round {r} of at most {R}.

You did not play this build. Someone else did: a play agent drove it, the
frames were read, and what follows is what that evidence supports and what was
chosen for this round.

So the problem below has already been established. You do not have to find out
whether it is real, and you should not replay the game to confirm it -- that
work is done, and repeating it spends the calls you need for the repair. Across
the rounds measured before this instruction existed, the median one did not
touch code until its fourteenth call out of twenty-six, and more than half
replayed the game first. Your job starts at "where in the code does this live".

{packet}
{history}
WHAT YOU HAVE
  - the source, in the directory you are working in
  - `{build_cmd}` -- it must pass when you stop
  - you can run the build and watch it:

      {py} -m rsigame.agent.evolve.play_cli . \\
          --steps '[{{"action":"boot"}},{{"action":"press_key","key":"Enter"}},{{"action":"read_state"}},{{"action":"screenshot"}}]'

    with PYTHONPATH={repo}. It boots the build, acts, and prints JSON: the
    runtime's own state object, plus frames you can open with your file tool.
    Many of these games open on a title screen -- the state object stays null
    until you dismiss it, which is not a defect.
{project_map}{report}
RULES
  - Change the game, not the harness. Do not edit `node_modules`, `dist`,
    `demo_outputs/`, or anything under `_repair_evidence/`.
  - Never kill processes by pattern -- no `pkill -f`, no `killall`, no
    `kill -- -PGID`. Every process in this project shares a path, so a pattern
    stops the other games running beside yours and the batch driving them all;
    one `pkill -f gamedoctor` here took six services in 24 milliseconds. Note
    the pid when you start something and kill that pid.
  - If you start a server of your own, bind it to 127.0.0.1 and kill it before
    you stop. Prefer `play_cli`, which cleans up after itself.
  - Do not edit `demo_outputs/`. Those are the input traces the evaluator
    replays; changing them changes the exam, not the game.
{art}"""


# LEAN VARIANT (RSIGAME_LOOP_LEAN_REPAIR=1). Same instructions, same order, same
# report format; what is cut is the evidence behind each instruction (the
# measurements and incidents quoted to justify it) and the rule stated twice.
# Those were written for whoever maintains the prompt, and the agent re-read
# them with every one of its ~20 calls per round.
_ART_LEAN = """
ART AND EFFECTS. You have `generate_game_assets`. This game is judged partly on
whether it looks and feels made rather than assembled, so this is available
every round, not only when something is broken; the packet above says what is
worth doing now. Each image costs money and minutes: generate the few that
carry the game, not a set. How far to take it is yours to judge.

THIS IS A GODOT PROJECT:
  - Write into the tree's own `assets/<kind>/` -- pass it as `output_dir_name`.
  - GODOT CANNOT LOAD A RESOURCE IT HAS NOT IMPORTED. After adding files, run
        godot --headless --path . --import --quit
    or `load()` returns null at runtime, silently.
  - The code has to use it -- `load("res://assets/...")` and `draw_texture` or
    `draw_texture_rect` -- and you must DELETE the `draw_rect` / `draw_circle`
    that drew the same thing, or the procedural one keeps winning.
  - Check your work by playing the build AFTER the import.

PUT THEM WHERE THE GAME ALREADY LOOKS. `output_dir_name` must be a directory
this project ALREADY loads from (grep a `res://` path out of the source). Never
invent a new folder: the code keeps loading the old path.

CHECK EVERY KEY BEFORE YOU STOP: grep the source for each generated filename.
One that is not referenced changes nothing on screen.

ASK FOR SEVERAL AT ONCE. `assets` is a list; one call generates the whole set.
Split into two calls only for a large set (animations and backgrounds, then the
rest). A character's parts (run cycle, idle, jump, facings) are ONE request
under one `style_anchor`, or they come back as different characters.

ALPHA IS FOR THE SHAPE OF A THING, NOT FOR A LAYER OVER THE GAME.
  - A thing that sits ON the game (a critter, a block, an icon) is a cutout.
  - The ground, the board, the sky, a panel -- anything that fills an area --
    is OPAQUE. Paint stripes or grain INTO that image.
  - Do not tile a texture across the play field. Do not draw a translucent
    rectangle or texture over a region to tint it.

THE TOOL DOES NOT DECIDE THE SIZE, YOU DO, IN THE CODE. `image` and
`animation` come back 386x560; only `background` takes a `resolution`; do not
pass `size` to an `image`. Keep the rectangle of the line you replace:
    draw_texture_rect(tex, Rect2(x, y, w, h), false)
or set `Sprite2D.scale` / `region_rect` from the same numbers. Pixel art needs
`texture_filter = TEXTURE_FILTER_NEAREST`.
"""

_REPORT_LEAN = """
WRITE THIS AS SOON AS YOU KNOW IT, not at the end -- sessions run out of calls
and time. One JSON block; you can revise it by writing another later. Use the
issue ids from the packet above; if the packet named no issues, `primary` is
null and the art entry carries what you did -- do not invent an id.

```json
{"primary": {"issue_id": "<id from the packet>", "attempted": true},
 "concurrent": [{"issue_id": "<id>", "attempted": true},
                {"issue_id": "<id>", "attempted": false,
                 "reason": "<why not>"}],
 "art": {"attempted": true, "files": ["assets/..."]},
 "note": "<what you worked out about this codebase that the next round should
           not have to find again -- where a thing lives, what the fix
           touched, what turned out not to be the cause>",
 "todo": "<what you did not get to, and where you had got to>"}
```

If you cannot find where the behaviour lives by about your tenth call, stop and
write:

```json
{"repair_status": "blocked",
 "reason": "<what you could not localise>",
 "files_inspected": ["main.gd", "..."],
 "needs": "<what would let the next round find it>"}
```

`note` and `todo` are what the next round is told: name files and functions,
not impressions. Say `attempted` for what you tried, not for what you believe
worked -- replaying the game decides that.
"""

_HEAD_LEAN = """You are improving a small game. This is round {r} of at most {R}.

You did not play this build. A play agent drove it, the frames were read, and
what follows is what that evidence supports and what was chosen for this round.
The problem is already established: do not replay the game to confirm it. Your
job starts at "where in the code does this live".

{packet}
{history}
WHAT YOU HAVE
  - the source, in the directory you are working in
  - `{build_cmd}` -- it must pass when you stop
  - you can run the build and watch it:

      {py} -m rsigame.agent.evolve.play_cli . \\
          --steps '[{{"action":"boot"}},{{"action":"press_key","key":"Enter"}},{{"action":"read_state"}},{{"action":"screenshot"}}]'

    with PYTHONPATH={repo}. It prints JSON: the runtime's state object, plus
    frames you can open with your file tool. A title screen leaves the state
    object null until dismissed; that is not a defect.
{project_map}{report}
RULES
  - Change the game, not the harness. Do not edit `node_modules`, `dist`,
    `demo_outputs/` (the evaluator's input traces), or `_repair_evidence/`.
  - Never kill processes by pattern (no `pkill -f`, `killall`, `kill -- -PGID`):
    other games run beside yours. Kill the pid you started.
  - A server of your own binds 127.0.0.1 and is killed before you stop. Prefer
    `play_cli`, which cleans up after itself.
{art}"""


_ART_WEB_LEAN = _ART_LEAN.split('THIS IS A GODOT PROJECT')[0] + """THIS IS A PHASER (web) PROJECT, built with `npm run build`:
  - Write into `public/assets/` (the tool's default). Every generated key is
    added to `public/assets/asset-pack.json`, which the Preloader loads, so a
    new file is available BY ITS KEY with no loader edit.
  - The code has to use the KEY -- `this.add.image(x, y, 'key')`,
    `this.add.sprite(...)`, `setTexture('key')`, or an animation from the frame
    keys -- and you must DELETE the `this.add.rectangle(...)` /
    `graphics.fillRect` / `fillCircle` that drew the same thing, or the
    procedural one stays on top.
  - The game that is played and judged is `dist/`. Run `npm run build` after
    adding and after wiring files, and check your work by playing that build.

PUT THEM WHERE THE GAME ALREADY LOOKS. Keep `output_dir_name` at the directory
holding `asset-pack.json`. Never invent a new folder: the pack the game loads
never hears of it.

CHECK EVERY KEY BEFORE YOU STOP: grep `src/` for each generated key. One that
is not referenced changes nothing on screen.

""" + _ART_LEAN.split('CHECK EVERY KEY BEFORE YOU STOP')[1].split('\n\n', 1)[1].split('THE TOOL DOES NOT DECIDE THE SIZE')[0] + """THE TOOL DOES NOT DECIDE THE SIZE, YOU DO, IN THE CODE. `image` and
`animation` come back 386x560; only `background` takes a `resolution`; do not
pass `size` to an `image`. Keep the size of the shape you replace:
    this.add.image(x, y, 'key').setDisplaySize(w, h)
A rectangle and an image both default to a centred origin, while
`graphics.fillRect(x, y, w, h)` is measured from its top-left corner -- use
`.setOrigin(0, 0)` when replacing one of those. Pixel art needs `pixelArt: true`
in the game config.
"""


import os as _os


def history_text(rounds: list[dict], limit: int = 0) -> str:

    """What was attempted and what came of it. Not what files were touched.

    Kept to the last few rounds on purpose. The block this replaces grew
    without bound because it appended a line per round forever, and by the end
    of a run the oldest entries were crowding out the packet while saying
    nothing that was still true.
    """
    limit = limit or int(_os.environ.get('RSIGAME_LOOP_HISTORY_ROUNDS', '5'))
    if not rounds:
        return ''
    out = ['', 'WHAT EARLIER ROUNDS TRIED']
    for h in rounds[-limit:]:
        v = h.get('verdict')
        mark = {'fixed': 'worked', 'failed': 'did not work',
                'not_attempted': 'was not attempted'}.get(v, 'outcome unknown')
        out.append(f"  round {h.get('round')}: {str(h.get('goal') or '?')[:140]}"
                   f"  --  {mark}")
        # What it changed, for the most recent rounds only. A round that
        # breaks the game leaves the next one looking at a symptom with no
        # idea what caused it: three rounds running spent their whole budget
        # on a grey screen without once being told which files the round
        # before had added or edited.
        if h.get('files'):
            out.append('      it changed: ' + ', '.join(h['files'][:8]))
        if h.get('art_ask'):
            out.append(f"      it was also asked, about how the game looks: "
                       f"{h['art_ask'][:150]}"
                       + (f"  -- {h['art_verdict']}" if h.get('art_verdict')
                          else ''))
        if h.get('note'):
            out.append(f"      it worked out: {h['note'][:220]}")
        if h.get('todo'):
            out.append(f"      it did not get to: {h['todo'][:200]}")
        for s in (h.get('skipped') or [])[:2]:
            out.append(f"      skipped {s.get('issue_id')}: {s.get('reason', '')[:110]}")
        if h.get('build_ok') is False:
            out.append('      the build did not compile')
    if len(rounds) > limit:
        out.insert(2, f"  ({len(rounds) - limit} earlier round(s) not listed)")
    return '\n'.join(out) + '\n'


def render(packet: str, *, r: int, total: int, build_cmd: str, py: str,
           repo: str, rounds: list[dict] | None = None,
           project_map: str = '', engine: str = 'godot',
           want_art: bool = True) -> str:
    pm = ('\n' + project_map + '\n') if project_map else ''
    from rsigame.controller.stage_brief import text as _stage_brief
    if _stage_brief():
        packet = _stage_brief() + '\n' + packet
    lean = os.environ.get('RSIGAME_LOOP_LEAN_REPAIR') == '1'
    # the art prompt is chosen per engine:Godot Godot speaks of scenes and .tres,Phaser Phaser of asset-pack.json and npm run build.
    # the Godot line used to require `engine == 'godot'` before giving any, so a Phaser art round said nothing at all.
    if engine == 'godot':
        art = _ART_LEAN if lean else _ART
    else:
        art = _ART_WEB_LEAN if lean else _ART_WEB
    return (_HEAD_LEAN if lean else _HEAD).format(
        r=r, R=total, packet=packet, history=history_text(rounds or []),
        build_cmd=build_cmd, py=py, repo=repo, project_map=pm,
        art=(art if want_art else ''),
        report=(_REPORT_LEAN if lean else _REPORT))


def parse_report(text: str) -> dict:
    """Read the agent's closing JSON. A missing one is recorded, not guessed."""
    import re
    m = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text or '', re.S)
    for blob in reversed(m):
        try:
            d = json.loads(blob)
        except Exception:
            continue
        if isinstance(d, dict) and ('primary' in d or 'concurrent' in d):
            return {'report': d, 'ok': True}
    return {'report': None, 'ok': False,
            'why': 'the agent stopped without the closing JSON block'}


_ART_HEAD = """You are finishing a small game. This is round {r} of {R}, and
{scope}

You did not play this build. A reader did -- the demo library was replayed
against the current tree and every frame was looked at -- and what follows is
what it wrote down. The problems are established. Do not replay the game to
confirm them; that spends the calls you need to do the work.

{brief}
{history}
HOW TO SPEND THE ROUND

Generate the assets in ONE call, then wire every one of them in, then import,
then look. A round that ends with new files nobody loads has changed nothing on
screen and cost real money -- that is the failure mode this round exists to
avoid, and it has happened: one game went from 30 images to 150 while the code
went from loading 3 to loading 5.

If the list has more than you can finish, do fewer things completely. Half a
list wired up beats a whole list generated.

The asset list arrives ordered by how much the reader expects each to change
the look of the game. That order reflects expected visual impact, not a
mandatory execution order. Spend most effort on the highest-impact
improvements, while taking advantage of cheap layout and effect changes that
can be completed alongside them. DO NOT LET MANY SMALL INEXPENSIVE EDITS CROWD
OUT A VISUALLY DOMINANT ASSET OR SCREEN-LEVEL IMPROVEMENT.

The two kinds of work cost differently, which is why they can be combined
rather than traded off. Generating is one batched call you wait on, and it is
the only part of this round with a real wait in it -- one wait whether the call
makes one image or nine. Layout and effect work is
plain code: no call, no wait, and it does not come out of the generation
budget. The round that moved a game furthest so far did both -- a background, a
character sprite and the platforms they stand on, together with four pieces of
hit and transition feedback, in twenty-eight tool calls.

WHAT YOU HAVE
  - the source, in the directory you are working in
  - `{build_cmd}` -- it must pass when you stop. Nothing rolls back if it
    does not, and there is no round after this one to repair it.
  - you can run the build and watch it:

      {py} -m rsigame.agent.evolve.play_cli . \\
          --steps '[{{"action":"boot"}},{{"action":"press_key","key":"Enter"}},{{"action":"read_state"}},{{"action":"screenshot"}}]'

    with PYTHONPATH={repo}. Use it AFTER the import to see your own work.
{project_map}{report}
RULES
  - Change the game, not the harness. Do not edit `node_modules`, `dist`,
    `demo_outputs/`, or anything under `_repair_evidence/`.
  - Never kill processes by pattern -- no `pkill -f`, no `killall`, no
    `kill -- -PGID`. Every process in this project shares a path, so a pattern
    stops the other games running beside yours and the batch driving them all.
    Note the pid when you start something and kill that pid.
  - If you start a server of your own, bind it to 127.0.0.1 and kill it before
    you stop.
  - Do not edit `demo_outputs/`. Those are the input traces the evaluator
    replays; changing them changes the exam, not the game.
{art}"""


_ART_REPORT = """
WRITE THIS AS SOON AS YOU KNOW IT, not at the end -- sessions run out of calls
before they reach the last instruction. One JSON block, revisable by writing
another later.

```json
{"style_anchor": "<the style_anchor you passed, verbatim>",
 "generated": ["key you asked the tool for", "..."],
 "wired": ["key you actually loaded and drew with", "..."],
 "effects_done": ["which effect items you did"],
 "layout_done": ["which layout items you did"],
 "imported": true,
 "note": "<what the next round should not have to find again -- where the
           drawing happens, what the wiring took>",
 "todo": "<what you did not get to, and where you had got to>"}
```

`generated` and `wired` are different lists on purpose. A key in the first and
not the second is a file on disk that changes nothing on screen, and saying so
is more useful than a round that reports success.
"""


_INTENT = """
====================
WHAT THE TASK ASKS THIS GAME TO LOOK LIKE
====================
The task document's own words, not a reader's judgement. Where it names a
style, a character, a layout or a piece of feedback, that is a requirement and
not a suggestion.

{intent}
"""


_STYLE = """
THE STYLE THIS GAME IS ALREADY IN

    {style}

Every image generated for this game so far was asked for with that
`style_anchor`. Pass it again, word for word, unless you judge the style
itself to be what is wrong -- and say so in your report if you change it. Art
asked for under a freshly invented phrase does not match what is already on
screen: across six rounds of one game six different anchors were used, swinging
between watercolour storybook and 16-bit pixel art, and no two batches of that
game's art belong to the same picture.
"""


def render_art(brief_text: str, *, r: int, total: int, n_art: int,
               build_cmd: str, py: str, repo: str,
               rounds: list[dict] | None = None,
               project_map: str = '', intent: str = '',
               style: str = '', engine: str = 'godot') -> str:
    """The art round's prompt. Same skeleton, different job."""
    pm = ('\n' + project_map + '\n') if project_map else ''
    from rsigame.controller.stage_brief import text as _stage_brief
    if _stage_brief():
        brief_text = _stage_brief() + '\n' + brief_text
    extra = (_INTENT.format(intent=intent) if intent else '')
    extra += (_STYLE.format(style=style) if style else '')
    if engine != 'godot':
        # The skeleton's two mentions of Godot's import step, said the way a
        # web build does it. Applied to the web prompt only.
        return _ART_HEAD.format(
            r=r, R=total, n=n_art, scope=(
                f'the last {n_art} rounds are for how the game LOOKS and FEELS. '
                'What it does has had its rounds and is not your job now.'
                if n_art else
                'it is for how the game LOOKS and FEELS, not for what it does.'),
            brief=brief_text + extra,
            history=history_text(rounds or []), build_cmd=build_cmd, py=py,
            repo=repo, project_map=pm, art=_ART_WEB, report=_ART_REPORT
        ).replace('then wire every one of them in, then import,\nthen look.',
                  'then wire every one of them in, then\n`npm run build`, then look.'
        ).replace('Use it AFTER the import to see your own work.',
                  'Use it AFTER `npm run build` to see your own work.'
        ).replace('"imported": true,', '"built": true,')
    return _ART_HEAD.format(
        r=r, R=total, n=n_art, scope=(
            f'the last {n_art} rounds are for how the game LOOKS and FEELS. '
            'What it does has had its rounds and is not your job now.'
            if n_art else
            'it is for how the game LOOKS and FEELS, not for what it does.'),
        brief=brief_text + extra,
        history=history_text(rounds or []), build_cmd=build_cmd, py=py,
        repo=repo, project_map=pm, art=_ART, report=_ART_REPORT)


def parse_art_report(text: str) -> dict:
    """The art round's closing JSON. A missing one is recorded, not guessed."""
    import re
    m = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text or '', re.S)
    for blob in reversed(m):
        try:
            d = json.loads(blob)
        except Exception:
            continue
        if isinstance(d, dict) and ('generated' in d or 'wired' in d):
            return {'report': d, 'ok': True}
    return {'report': None, 'ok': False,
            'why': 'the agent stopped without the closing JSON block'}
