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
"""What a Polish review is shown. POLISH PART 2, the evidence half.

Polish asks a different question from the checklist -- not "is this
requirement met" but "what is the largest remaining quality bottleneck" -- so
it needs a different pack, and a compact one. Polish handoff 10 is explicit:
do not send all raw traces or source.

FOUR SECTIONS, and three of them already exist somewhere in this repo:

  design intent          what the task said it should LOOK like
  representative sheet   title / core play / special state / ending
  static visual facts    assets present, referenced, unused; how the screen
                         is drawn -- all read mechanically off the tree
  interaction evidence   a few actions with their before and after

NO RUBRIC, NO SCORES, NO WEIGHTS. Handoff 10.3 says so in as many words, and
the rubric is held out anyway. Everything here is either the task's own words
or a count taken off the source.

THE COUNTS ARE THE ANCHOR. `headroom: high | medium | low` is an unanchored
three-valued scale, and it drives the only rule that ends Polish -- a model
that guesses `low` once would end the phase. So the two numbers a model cannot
argue with go in the pack beside it:

    unused assets     shooter-sky-duel ships 57 images and loads 19
    authored share    horror-tape-archive draws 121 primitives against 2
                      textures, which is 2%

Measured across the five pilot games the share runs 2%, 7%, 8%, 13%, 37% and
unused assets 0, 0, 2, 3, 38 -- enough spread to tell games apart, which is
what an anchor has to do.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

SRC = ('.gd', '.tscn', '.godot', '.ts')
IMG = ('.png', '.jpg', '.jpeg', '.webp')
SKIP = ('.godot', '.git', 'node_modules', 'dist', '_repair_evidence',
        'demo_outputs', '.qwen')
_PRIMITIVE = re.compile(
    r'draw_rect|draw_circle|draw_line|draw_polygon|draw_colored_polygon|'
    r'ColorRect|StyleBoxFlat|draw_arc\b')
_TEXTURE = re.compile(r'draw_texture|Sprite2D|TextureRect|AnimatedSprite')

# What the task itself says about how the game should look. Its own words, not
# a graded requirement -- a Polish review with no intent has nothing to hold a
# screenshot against, and would drift into generic art advice.
#
# CHOSEN BY EXCLUSION, not by keyword. Matching headings against words like
# "visual" or "style" picked `## Assets` out of shooter-sky-duel -- a list of
# which sprite packs to download, not a statement of intent -- because the word
# "art" appears in its body. The sections that describe the GAME are whatever
# is left once the build-and-evaluator half is removed, which is the same
# division Part 1 draws when it decides what belongs on a checklist.
_INTENT_DROP = re.compile(
    r'^\s*#*\s*(assets?|project layout|demos?|scenarios?|trace file|'
    r'file layout|build|deliverable|submission)\b', re.I)


def _sources(work: Path) -> str:
    out = []
    for p in Path(work).rglob('*'):
        if p.suffix.lower() not in SRC or not p.is_file():
            continue
        if any(part in SKIP for part in p.parts):
            continue
        try:
            out.append(p.read_text(errors='replace'))
        except OSError:
            pass
    return '\n'.join(out)


BYPRODUCTS = ('_contact_sheet.png', 'style_anchor.png')


def _loaders(src: str) -> tuple:
    """Every `res://...png` the source can produce: literal, and assembled.

    THE SECOND KIND IS MOST OF THE ART. These projects load families of
    textures by building the path at runtime --

        load("res://assets/sprites/critters/%s_%s.png" % [id, pose])
        load("res://assets/blocks/%s.png" % _tex_name())

    -- and a check that looks for literal paths sees none of them. On
    battle-critter-clash that counted 5 of 33 images as used where 25 are
    loaded, and `images_used` is exactly the number the polish reader and the
    art list are shown. Both were told for months that a game wiring its art up
    properly was a game drawing almost nothing, and both duly asked for the
    wiring again.

    Bare filenames are deliberately NOT accepted. A second copy of a sprite in
    a folder nobody reads matches on its name and changes nothing on screen,
    which is the other half of the same misreading.
    """
    lits = set(re.findall(r'res://[^"\')\s]+\.(?:png|jpg|jpeg|webp)', src))
    pats = []
    for f in set(re.findall(r'res://[^"\')\s]*%[sd][^"\')\s]*\.(?:png|jpg|jpeg|webp)',
                            src)):
        # `re.escape` leaves `%` alone, so the placeholders survive it intact.
        pats.append(re.compile('^' + re.escape(f).replace('%s', '[A-Za-z0-9_\\-]+')
                               .replace('%d', r'\d+') + '$'))
    return lits, pats


def _is_loaded(rel: str, src: str, _cache={}) -> bool:
    # A Phaser tree's art lives under `public/`, which no Godot project has, so
    # this branch is never taken for a Godot file (see `_web_is_loaded`).
    if rel.startswith('public/'):
        return _web_is_loaded(rel, src, {})
    key = id(src)
    if key not in _cache:
        _cache.clear()
        _cache[key] = _loaders(src)
    lits, pats = _cache[key]
    full = f'res://{rel}'
    return full in lits or any(x.match(full) for x in pats)


# ---------------------------------------------------------------- Phaser (web)
#
# THE SAME COUNTS, READ THE WAY A PHASER GAME IS BUILT. Every regex above is a
# Godot one: `res://` paths and `draw_rect`. On a web tree they find nothing, so
# `horror-tape-archive` read "12 images present, 0 used, 0 draws" -- a game
# drawing entirely with textures reported as one drawing nothing at all, and
# that number goes to the polish reader and the art list as fact.
#
# A Phaser game ships its art under `public/` and loads it through a pack file
# (`public/assets/asset-pack.json`: key -> url); the code then names the KEY
# (`this.add.image(x, y, 'title_bg')`). So an image counts as used when its pack
# key, its url, or its file stem appears in the source -- and an animation frame
# `player_die_02`, whose key is usually assembled at runtime from `player_die`,
# counts when that prefix does. Generous on purpose, for the reason `_loaders`
# gives: calling a used asset unused is the error that sends the art list back
# for wiring that is already there.
#
# Chosen by the tree, not by a flag: a project with `project.godot` never takes
# this path, so every Godot count stays exactly what it was.
WEB_SRC = ('.ts', '.tsx', '.js', '.mjs')
_WEB_PRIMITIVE = re.compile(
    r'\.add\.(?:rectangle|circle|ellipse|triangle|polygon|star|arc|line|grid)\(|'
    r'\.(?:fill|stroke)(?:Rect|RoundedRect|Circle|Ellipse|Triangle|Points|Path)\(')
_WEB_TEXTURE = re.compile(
    r'\.add\.(?:image|sprite|tileSprite|nineslice|video)\(|'
    r'\.physics\.add\.(?:image|sprite|staticImage|staticSprite)\(|'
    r'\.make\.(?:image|sprite)\(|\.setTexture\(|\.createFromObjects\(|'
    r'\.addTilesetImage\(')


def is_web(work: Path) -> bool:
    return not (Path(work) / 'project.godot').is_file()


def _web_sources(work: Path) -> str:
    out = []
    for p in Path(work).rglob('*'):
        if p.suffix.lower() not in WEB_SRC or not p.is_file():
            continue
        if any(part in SKIP for part in p.parts):
            continue
        try:
            out.append(p.read_text(errors='replace'))
        except OSError:
            pass
    return '\n'.join(out)


def _web_pack_keys(work: Path) -> dict:
    """url (relative to `public/`) -> pack key, from every pack under public/."""
    out = {}
    for pack in (Path(work) / 'public').rglob('*.json'):
        if any(part in SKIP for part in pack.parts):
            continue
        try:
            d = json.loads(pack.read_text(errors='replace'))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        for sec in d.values():
            for f in ((sec or {}).get('files') or []) if isinstance(sec, dict) else []:
                if isinstance(f, dict) and f.get('url') and f.get('key'):
                    urls = f['url'] if isinstance(f['url'], list) else [f['url']]
                    for u in urls:
                        out[str(u).lstrip('./')] = str(f['key'])
    return out


def _web_is_loaded(rel: str, src: str, keys: dict) -> bool:
    """`rel` is relative to the tree, e.g. `public/assets/title_bg.png`."""
    url = rel[len('public/'):] if rel.startswith('public/') else rel
    stem = Path(rel).stem
    names = {url, Path(rel).name, stem}
    if keys.get(url):
        names.add(keys[url])
    base = re.sub(r'[_\-]?\d+$', '', stem)
    if base and base != stem and len(base) >= 3:
        names.add(base)
    return any(n and n in src for n in names)


def _web_visual_facts(work: Path) -> dict:
    src = _web_sources(work)
    keys = _web_pack_keys(work)
    imgs = [p for p in work.rglob('*')
            if p.suffix.lower() in IMG and p.is_file()
            and p.name not in BYPRODUCTS
            and not any(part in SKIP for part in p.parts)]
    used = [p for p in imgs
            if _web_is_loaded(p.relative_to(work).as_posix(), src, keys)]
    prim = len(_WEB_PRIMITIVE.findall(src))
    tex = len(_WEB_TEXTURE.findall(src))
    return {'images_present': len(imgs), 'images_used': len(used),
            'images_unused': len(imgs) - len(used),
            'unused_names': sorted(p.name for p in imgs if p not in used)[:12],
            'primitive_draws': prim, 'texture_draws': tex,
            'authored_share': round(tex / max(1, tex + prim), 3),
            'audio_present': len([p for p in work.rglob('*')
                                  if p.suffix.lower() in ('.ogg', '.wav', '.mp3')
                                  and not any(x in SKIP for x in p.parts)])}


def visual_facts(work: Path) -> dict:
    """Counts only. Nothing here is a verdict about whether they are enough."""
    work = Path(work)
    if is_web(work):
        return _web_visual_facts(work)
    src = _sources(work)
    # THE TOOL'S OWN BY-PRODUCTS ARE NOT THE GAME'S ART. Every
    # `generate_game_assets` call drops a `_contact_sheet.png` (a labelled
    # grid of what it made) and a `style_anchor.png` (the style reference it
    # was given) beside the real files. No game ever loads either, so counting
    # them inflates `images_present` and nothing else -- across three art
    # rounds they were 10 of the 49 files reported as generated, which makes
    # the wiring rate unreadable exactly when it matters.
    imgs = [p for p in work.rglob('*')
            if p.suffix.lower() in IMG and p.is_file()
            and p.name not in BYPRODUCTS
            and not any(part in SKIP for part in p.parts)]
    used = [p for p in imgs if _is_loaded(p.relative_to(work).as_posix(), src)]
    prim = len(_PRIMITIVE.findall(src))
    tex = len(_TEXTURE.findall(src))
    return {'images_present': len(imgs), 'images_used': len(used),
            'images_unused': len(imgs) - len(used),
            'unused_names': sorted(p.name for p in imgs if p not in used)[:12],
            'primitive_draws': prim, 'texture_draws': tex,
            'authored_share': round(tex / max(1, tex + prim), 3),
            'audio_present': len([p for p in work.rglob('*')
                                  if p.suffix.lower() in ('.ogg', '.wav', '.mp3')
                                  and not any(x in SKIP for x in p.parts)])}


def design_intent(game: str, work: Path | None = None) -> str:
    """The task's own words about how it should look, headings included."""
    import rsigame.checklist.task_checklist as TC
    raw = TC.task_document(game, work)[0]
    if not raw.strip():
        return ''
    keep = []
    for sec in re.split(r'(?m)^(?=#+ )', raw):
        head = sec.split('\n', 1)[0]
        if _INTENT_DROP.search(head):
            continue
        keep.append(sec.strip())
    if not keep:
        keep = [raw.strip()[:1200]]
    return '\n\n'.join(keep)[:2500]


# WHICH FRAMES A REVIEW SHOULD SEE, and why not just the last N.
#
# The checklist reader takes frames beside an input, because it is judging
# whether an action did something. A Polish review is judging how the game
# LOOKS, so it wants breadth: the title, ordinary play, whatever unusual state
# the session reached, and the end. One or two of each, per handoff 10.2.
def representative_frames(exp: dict, per_bucket: int = 2) -> list:
    steps = exp.get('steps') or []
    buckets = {'title': [], 'core': [], 'special': [], 'ending': []}
    n = len(steps)
    for i, s in enumerate(steps):
        f = s.get('frame') or s.get('marked_frame')
        if not f:
            continue
        scen = str((s.get('args') or {}).get('id') or '')
        if s.get('action') == 'boot' or i == 0:
            buckets['title'].append((s.get('n'), f, 'title / first screen'))
        elif s.get('action') == 'boot_scenario' and scen:
            buckets['special'].append((s.get('n'), f, f'scenario {scen}'))
        elif i >= n - 2:
            buckets['ending'].append((s.get('n'), f, 'last state reached'))
        else:
            buckets['core'].append((s.get('n'), f, 'core play'))
    out = []
    for k in ('title', 'core', 'special', 'ending'):
        out += buckets[k][:per_bucket]
    return out


def interactions(exp: dict, cap: int = 5) -> list:
    """A few actions with what the screen did. Facts, no interpretation.

    Picked by how much the screen moved, both ends: the biggest change and the
    smallest. A Polish review cares about both -- an action with no visible
    result is missing feedback, and the loudest moment is where feedback
    already exists and can be judged.
    """
    acted = [s for s in (exp.get('steps') or [])
             if s.get('action') in ('click', 'press_key', 'hold_key')
             and s.get('pixels_changed_pct') is not None]
    acted.sort(key=lambda s: s['pixels_changed_pct'])
    picked = acted[:cap // 2] + acted[-(cap - cap // 2):] if len(acted) > cap \
        else acted
    out = []
    for s in picked:
        a = s.get('args') or {}
        what = a.get('target') or a.get('key') or f"({a.get('x')},{a.get('y')})"
        out.append({'step': s.get('n'), 'action': s.get('action'),
                    'what': str(what)[:60],
                    'screen_changed_pct': s.get('pixels_changed_pct'),
                    'thought': (s.get('thought') or '')[:140]})
    return out


def build(game: str, work: Path, exp: dict) -> dict:
    return {'game': game,
            'intent': design_intent(game, work),
            'facts': visual_facts(work),
            'frames': representative_frames(exp),
            'interactions': interactions(exp)}


def render_facts(f: dict) -> str:
    lines = [
        'WHAT THE BUILD ACTUALLY CONTAINS  (counted off the source; not a verdict)',
        '',
        f"  image files present   {f['images_present']}",
        f"  referenced by code    {f['images_used']}",
        f"  never referenced      {f['images_unused']}"
        + (f"   e.g. {', '.join(f['unused_names'][:6])}"
           if f['unused_names'] else ''),
        f"  audio files present   {f['audio_present']}",
        '',
        f"  calls that draw a texture or place a sprite   {f['texture_draws']}",
        f"  calls that draw a primitive or a flat widget  {f['primitive_draws']}",
        f"  share of drawing that uses authored art       "
        f"{f['authored_share'] * 100:.0f}%"
        + ('   (under half: the world is mostly primitives)'
           if f['authored_share'] < 0.5 else ''),
    ]
    return '\n'.join(lines)


# WHERE THIS GAME SITS AMONG ITS PEERS. KEPT FOR ANALYSIS, NOT USED IN THE
# REVIEW -- see the note at the end of this block for why it was taken back out.
#
# NOT USED because ranking within a weak corpus answers the wrong question.
# Every game here draws mostly with primitives (median 9%, best 44%), so the
# percentile only says which game is least bad, and shooter-sky-duel reading
# `medium` at the 95th percentile is wrong in the way that matters: it is still
# 36% authored, still looks like a prototype to a player, and is still capped
# by the graded art clause. An absolute half is the honest line, and if that
# makes every game read `high` on the visual axis, that is the true answer for
# this corpus -- fix the art first, then worry about feel.
#
# The first review put `visual: high` on all five pilot games, which is a
# three-valued scale with one value in it. The cause was the anchor: the prompt
# said a game already drawing mostly with authored art has less room, and NO
# GAME HERE IS IN THAT STATE. Measured over all forty Godot games the authored
# share runs 0% to 44% with a median of 9% -- primitive-dominated drawing is
# the norm, not the exception, so an absolute reading of 2% and of 36% both
# land on "mostly primitives" even though they differ eighteenfold.
#
# The percentile separates them: horror-tape-archive sits at the 5th and
# shooter-sky-duel at the 95th. That is a fact about our own corpus, computed
# from the same counts, and it carries no rubric.
_DIST_CACHE = None


def corpus_distribution(corpus=None) -> dict:
    """Authored-share and unused-asset spread across the corpus, cached."""
    global _DIST_CACHE
    if _DIST_CACHE is not None:
        return _DIST_CACHE
    root = Path(corpus or (Path(__file__).resolve().parent
                           / 'godot' / 'corpus_gpt'))
    sh, un = [], []
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        if not (d / 'project.godot').is_file():
            continue
        f = visual_facts(d)
        sh.append(f['authored_share'])
        un.append(f['images_unused'])
    _DIST_CACHE = {'n': len(sh), 'authored': sorted(sh),
                   'unused': sorted(un)}
    return _DIST_CACHE


def _pct(sorted_vals: list, v) -> int:
    if not sorted_vals:
        return 50
    return round(100 * sum(1 for x in sorted_vals if x < v) / len(sorted_vals))


def render_facts_with_peers(f: dict, dist: dict) -> str:
    a, u = dist.get('authored') or [], dist.get('unused') or []
    q = lambda vals, p: vals[int(p * (len(vals) - 1))] if vals else 0
    return render_facts(f) + '\n\n' + '\n'.join([
        f"HOW THAT COMPARES WITH THE OTHER {dist.get('n', 0)} GAMES BUILT THE "
        f"SAME WAY",
        '',
        f"  authored share    this game {f['authored_share'] * 100:.0f}%"
        f"   ->  {_pct(a, f['authored_share'])}th percentile",
        f"                    corpus: lowest {q(a, 0) * 100:.0f}%, "
        f"median {q(a, .5) * 100:.0f}%, highest {q(a, 1) * 100:.0f}%",
        f"  unused images     this game {f['images_unused']}"
        f"   ->  {_pct(u, f['images_unused'])}th percentile",
        f"                    corpus: median {q(u, .5)}, highest {q(u, 1)}",
        '',
        "  Primitive-dominated drawing is the NORM in this corpus, not the",
        "  exception, so a low absolute share is not by itself a finding. What",
        "  matters is where this game sits against the others and what the",
        "  frames actually show.",
    ])
