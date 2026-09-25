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
"""One look at the finished game, and three lists of what is still unmade.

THE LAST ROUNDS ARE ART ROUNDS. By then the requirement work has had its
budget and the thing holding the game back is not what it does but what it
looks like: measured over ten games, the loop wired 27 more images onto the
screen across twelve rounds while the polish reader called neither dimension
finished on any build it ever saw.

This replaces the whole per-round chain for those rounds -- plan, probe,
ground, rank, polish read, polish ask, five calls -- with one. It can, because
an art round does not have to decide what to investigate: the evidence is the
demo library replayed, which the round before already produced, and the
question is the same every time.

ONE CALL, THREE ANSWERS, and the first two are not about art at all:

  playable    did the last round's work leave a game anyone can still play.
              An art round deletes `draw_rect` calls and adds textures, which
              is exactly how a screen goes black, and round 15 has no round
              after it to notice. This is the post-verify, folded in -- a
              separate call would look at the same frames and ask a question
              this one already has to answer.

  landed      the previous art round asked for specific things. Are they on
              screen. Nobody was checking: the art ask was in the packet as
              co-equal work and outside the only reading that says what
              happened.

  the lists   assets that are still shapes standing in for drawings, actions
              that land without any feedback, and panels that are laid out
              badly -- HUD, title, game over.

Every entry must name the frame it was seen in. An item that cannot point at
one is a guess about a game nobody looked at, and the repair agent would spend
a generation call on it.

Nothing here says how to fix anything, and nothing here mentions how the game
is graded.
"""
from __future__ import annotations

import json
import re
import os
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUN))

KINDS = ('image', 'background', 'tileset', 'animation')
PANELS = ('hud', 'title', 'game_over', 'other')
PLAYABLE = ('playable', 'broken', 'cannot_tell')

# Enough for a round's work and no more. Each generated image costs real money
# and minutes, and a list of twenty is a list nobody finishes -- the round ends
# with twenty files on disk and none of them wired up, which is the failure
# this whole line of work exists to stop.
# THE ROUND'S OWN LIST GETS THE LARGER CAP.
#
# These were 6/4/4 with the caps fixed, which is right when all three lists are
# equal and wrong the moment one of them leads: the first feedback round was
# capped at four effects while assets -- explicitly secondary that round -- was
# allowed six, and all three games came back with exactly four. A cap that
# matches the count exactly is a cap that bit.
# The endpoint refuses a request whose images exceed 30MB; base64 costs a
# third on top of the file, and the prompt and labels have to fit too.
_URI_BUDGET = int(os.environ.get('RSIGAME_ART_URI_BUDGET') or 22_000_000)

LEAD_CAP = 6
SIDE_CAP = 4


def _caps(focus: str | None) -> dict:
    if focus == 'feedback':
        return {'assets': SIDE_CAP, 'effects': LEAD_CAP, 'layout': SIDE_CAP}
    return {'assets': LEAD_CAP, 'effects': SIDE_CAP, 'layout': SIDE_CAP}


_PROMPT = """You are looking at a small game near the end of the work on it.
What it does has had its rounds. What is left is how it looks and feels, and
that is what you are here to write down.

You are not grading anyone and you are not guessing at causes. You say what is
on the screen.

====================
WHAT THE TASK SAYS THIS GAME IS
====================
{intent}

====================
WHAT THE BUILD CONTAINS, counted off the source
====================
{facts}

====================
WHAT WAS ASKED FOR BEFORE, AND WHAT IS TRUE ABOUT IT NOW
====================
{asked}

====================
THE FRAMES
====================
{note}

====================
WHAT TO ANSWER
====================

1. IS IT STILL PLAYABLE. The round before this one may have deleted the code
   that was drawing something and added a texture in its place, which is how a
   screen goes blank. Decide from the frames only:

     playable      the frames show a game -- content on screen, and the screen
                   is not the same flat colour throughout
     broken        somewhere that had content is now empty, or a frame is one
                   flat colour, or nothing on screen ever changes
     cannot_tell   the replay never got far enough to say

   `cannot_tell` is not `broken`. Not having looked is not having seen it fail.

2. DID THE EARLIER REQUESTS LAND. For each thing listed above, say whether you
   can see it in these frames: `yes`, `no`, or `cannot_tell`. If nothing was
   asked for, return an empty list.

   The rows above are read mechanically off the tree; the last line of each is
   yours to answer. Use this evidence when deciding whether generating another
   asset is likely to help. Repeated generation is unlikely to be useful if an
   appropriate asset already exists and is referenced; consider whether the
   remaining problem is integration, state selection, layering, scaling, or
   visibility.

{focus_block}
3. THREE LISTS. Only what these frames show, and every entry names its frame.

   ASSETS -- something on screen whose art is not carrying the game. That is
   usually one of two things, and the second is easy to miss:

     a placeholder -- a coloured rectangle, a circle, or a flat default widget
       standing in for a drawing;

     a drawing that is not good enough -- generic, off-style from the rest of
       the screen, or plainly cruder than what sits beside it. THAT AN IMAGE
       ALREADY EXISTS IS NOT A REASON TO LEAVE IT ALONE. A player character
       drawn as a generic mascot, in a game whose backgrounds are painted, is
       a gap, and by this point in a run it is often the largest one left.

   Say what it is now, and what it should be instead, concretely enough that
   someone can draw it without seeing the game. Say which kind of asset it is:
     image       one thing on a transparent background -- a character, a prop,
                 an icon, a projectile
     background  a full scene behind everything
     tileset     a repeating surface -- ground, walls, water
     animation   a thing that needs more than one frame to read

   EFFECTS -- an action that happens with no punctuation. A hit that just
   removes the object, a pickup with no flash, a screen that cuts to the next
   one. Say where in play it happens and what should mark it.

   LAYOUT -- a panel that is laid out badly. The HUD, the title screen, the
   game over screen. Text against an edge, a number no one can read against
   what is behind it, elements that touch, a panel with nothing considered
   about it. Say which panel and what should change.

   A SINGLE ENTRY MAY BE A SET. A character is rarely one image: it is a run
   cycle, an idle, a jump, one per facing, one per creature. Where that is
   what the thing needs, put the pieces in `parts` and list them. The tool
   takes a list and makes the whole set in one call under one style, which is
   both the cheapest way to do it and the only way they come out matching --
   so a nine-frame character is one request here, not nine, and not a reason
   to leave the character alone.

{rank_block}

Say in `leverage` which of those considerations put each entry where you put
it. A round may not finish the list, and what is at the bottom is what will not
get done, so the order is a decision and not a formality.

Any of the three may be empty; an empty list is a real answer. If all three
are empty, say why in `note` -- that is a strong claim about a game the reader
of these frames has not once called finished.

Do not ask for more than {max_assets} assets, {max_effects} effects or
{max_layout} layout changes. One round has to be able to finish the list.

Return ONLY this JSON:

{{"playable": "playable|broken|cannot_tell",
 "playable_why": "<one sentence, naming a frame>",
 "landed": [{{"asked": "<the request, shortened>", "seen": "yes|no|cannot_tell",
             "note": "<one sentence>"}}],
 "assets": [{{"key": "<short_snake_case_name>",
             "now": "<what is drawn there today>",
             "want": "<what should be drawn, described for someone who cannot see the game>",
             "kind": "image|background|tileset|animation",
             "parts": ["<names of the pieces, if this is a set; else omit>"],
             "leverage": "<why this sits where it does in the order>",
             "seen_at": "<which frame>"}}],
 "effects": [{{"when": "<what the player does or what happens>",
              "now": "<what the frames show happening>",
              "want": "<what should mark it>",
              "leverage": "<why this sits where it does in the order>",
              "seen_at": "<which frame>"}}],
 "layout": [{{"panel": "hud|title|game_over|other",
             "problem": "<what is wrong with how it is laid out>",
             "want": "<what should change>",
             "seen_at": "<which frame>"}}],
 "note": "<one sentence, or why all three lists are empty>"}}
"""


_SKIP_TOKENS = {'sprite', 'sprites', 'image', 'img', 'png', 'art', 'tex',
                'texture', 'asset', 'the', 'a', 'new', 'tile', 'anim'}


def _tokens(key: str) -> set:
    # Trailing plurals are stripped: the same thing came back as
    # `battle_platforms` and then `battle_platform_art`, and an unstemmed
    # comparison calls those two different things.
    out = set()
    for t in re.findall(r'[a-z0-9]+', str(key).lower()):
        if len(t) > 3 and t.endswith('s') and not t.endswith('ss'):
            t = t[:-1]
        if t not in _SKIP_TOKENS and len(t) > 2:
            out.add(t)
    return out


def _same_thing(a: str, b: str) -> bool:
    """Two requests for one thing, under keys that drifted.

    They drift constantly: `player_critter_back_sprite` came back as
    `player_back_sprite` the next round, `battle_platforms` as
    `battle_platform_art`, `stone_platform_blocks` as `stone_block_tile`. An
    exact-key counter reported 0 earlier requests for a sprite that had been
    asked for three rounds running.
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    return ta <= tb or tb <= ta or len(ta & tb) / len(ta | tb) >= 0.5


def asset_index(work) -> dict:
    """Every key the asset tool has produced in this tree, and its file.

    From the tool's OWN report rather than by guessing at filenames. Each call
    leaves an `_asset_report.json` beside what it made, listing key, type and
    file -- which is the only exact key-to-file mapping that exists.

    The `file` field is not usable as a path: it reads `assets/platform_pad.png`
    for a file that actually sits at `assets/sprites/platform_pad.png`, because
    it is recorded relative to the output directory the call was given. The
    basename is reliable, so that is what is used, resolved against the report's
    own directory.
    """
    out = {}
    for rep in Path(work).rglob('_asset_report.json'):
        try:
            d = json.loads(rep.read_text())
        except Exception:
            continue
        for a in (d.get('assets') or []):
            key = str(a.get('key') or '').strip()
            if not key:
                continue
            f = rep.parent / Path(str(a.get('file') or '')).name
            if f.is_file():
                out[key] = (f, str(a.get('type') or ''))
    return out


def _by_name(work: Path, key: str, cap: int = 4) -> list:
    """Files whose names carry the request's distinctive words."""
    want = _tokens(key)
    if not want:
        return []
    out = []
    for f in sorted(Path(work).rglob('*.png')):
        if any(x in f.parts for x in ('.godot', 'demo_outputs', '_repair_evidence')):
            continue
        if f.name in ('_contact_sheet.png', 'style_anchor.png'):
            continue
        if _tokens(f.stem) & want:
            out.append(f)
    return out[:cap]


def history(work, rounds: list) -> str:
    """What was asked for before, and what is mechanically true about it now.

    Four things can be established without judgement: whether the asset tool
    ever produced that key, whether a file resembling it is on disk, whether
    the code loads that file, and how many earlier rounds asked for the same
    thing. Whether it is actually on screen is the one line this cannot answer,
    and the reader answers it from the frames.

    No conclusion is drawn here on purpose. `file exists + code loads it + not
    visible` does not mean the wiring is wrong: the state may never be entered,
    something may be drawn over it, the scale or position may be off, the wrong
    variant may be selected, or it may be transparent. Naming one of those as
    the answer would be a diagnosis this module cannot support.
    """
    work = Path(work)
    asked = []                       # (key, want, round)
    for r in rounds:
        for a in ((r.get('art_list') or {}).get('assets') or []):
            asked.append((a.get('key', ''), a.get('want', ''), r.get('round')))
    if not asked:
        return '  (nothing has been asked for yet)'

    idx = asset_index(work)
    import rsigame.polish.polish_evidence as PE
    src = PE._sources(work)

    latest, seen = [], set()
    for key, want, rnd in reversed(asked):          # newest first, one row each
        if any(_same_thing(key, k) for k in seen):
            continue
        seen.add(key)
        latest.append((key, want, rnd))
    latest = latest[:8]

    lines = ['These rows are read off the tree, not judged. The last line of '
             'each is the one you answer.', '']
    for key, want, rnd in latest:
        n_before = sum(1 for k, _, _ in asked if _same_thing(key, k))
        hit = idx.get(key) or next(
            ((f, t) for k, (f, t) in idx.items() if _same_thing(key, k)), None)
        lines.append(f'  {key}   (asked in round {rnd}'
                     + (f', {n_before} requests in all)' if n_before > 1 else ')'))
        lines.append(f'      wanted: {str(want)[:150]}')
        if hit is None:
            # NO KEY MATCH IS NOT NO ASSET. The five back-view sprites a game
            # needed were generated one per creature -- `sproutling_back`,
            # `emberkit_back` -- while the request had been written as
            # `player_back_sprite`. On keys alone that reads as "nothing like
            # this exists", which is the reading that had the same sprite
            # requested three rounds running. So the filenames are searched
            # too, and what is found is reported without a conclusion.
            near = _by_name(work, key)
            if near:
                lines.append('      no asset carries this exact key, but these '
                             'files are named like it:')
                for f in near:
                    rel = f.relative_to(work).as_posix()
                    lines.append(f'         {rel}   '
                                 f'{"the code loads it" if PE._is_loaded(rel, src) else "the code does not load it"}')
            else:
                lines.append('      no asset key and no filename resembles this  no')
        else:
            f, kind = hit
            rel = f.relative_to(work).as_posix()
            lines.append(f'      a file for it exists                          '
                         f'yes  {rel}')
            loaded = PE._is_loaded(rel, src)
            lines.append(f'      the code appears to load that file            '
                         f'{"yes" if loaded else "no "}')
        lines.append('      visible in these frames                       <you say>')
        lines.append('')
    return '\n'.join(lines)


_FOCUS = {
    'presentation': """
THIS ROUND IS FOR HOW THE GAME LOOKS. Write all three lists -- what is seen is
seen -- but the assets and the layout are what this round will be spent on,
and they are the ones to put in order. Effects you notice still belong in the
list; they are simply not what the round is for.
""",
    'feedback': """
THIS ROUND IS FOR HOW THE GAME ANSWERS. The frames below come in sets: the
frame before an input, and the two frames after it. Read them as sets. Two
after-frames are given because a response that renders a frame late looks
exactly like no response at all in a single frame -- only something absent
across both is absent.

Write all three lists, but EFFECTS is what this round will be spent on and the
one to put in order. Assets and layout you notice still belong in the list;
they are simply not what the round is for.

Things worth checking before you conclude the list is short. This is not a
form to fill in and an empty answer to any of them is fine:

  the core actions      a hit, being hit, a pickup, a score -- does the moment
                        read as having happened, or only as a number changing
  numbers changing      when health, score or a resource moves, is there
                        anything on screen that moves with it
  visible cause         a result happens whose cause is never on screen -- the
                        train that kills you and is never drawn, the ranged
                        attack where nothing travels
  punctuation           do starts, endings, level changes and hand-overs cut
                        straight, or are they marked
  weight of outcomes    does winning look different from losing
""",
}

_RANK = {
    'presentation': """4. PUT THE ASSET LIST IN ORDER, most worth doing first.

Rank requests by how much improving that element would change the player's
overall visual impression of the game. Consider how visually prominent it is,
how often players see it, whether it is a focal gameplay subject, whether it
defines an important game state, and how visibly underdeveloped it currently
is. Large or persistent backgrounds, player-controlled characters, major
gameplay subjects, and full-screen title/transition/result states often have
high leverage; small peripheral widgets or decorative details usually have
lower leverage unless they are unusually salient or visibly broken.""",
    'feedback': """4. PUT THE EFFECTS LIST IN ORDER, most worth doing first.

Rank by how much adding this response would change how the game feels to play.
Consider how often the player performs that action, whether it is the core
verb of the game, whether the outcome is currently legible at all or only
inferable from a number, and whether its absence makes the moment read as
nothing having happened. The game's main verb, and any result whose cause is
never shown on screen, often have high leverage; a transition between menus
usually has less unless it is jarring.""",
}
_RANK[None] = _RANK['presentation']
_FOCUS[None] = ''


def read(game: str, work, frames: list, *, facts: dict | None = None,
         focus: str | None = None,
         asked: str = '', model: str | None = None) -> dict:
    """`frames` are (label, path) pairs, labelled with the demo and moment."""
    import rsigame.polish.polish_evidence as PE
    import rsigame.verify.post_verify as PV
    from rsigame.agent.evolve.verifier_agent import _data_uri

    frames = [(l, f) for l, f in (frames or []) if Path(f).is_file()]
    cap = _caps(focus)
    facts = facts if facts is not None else PE.visual_facts(Path(work))
    note = (f'{len(frames)} frames follow, each labelled with the demo it came '
            f'from and where in that demo it sits. The mouse pointer is drawn '
            f'by the recorder as a small cross and is not part of the game.'
            if frames else
            'NO FRAMES WERE CAPTURED. Answer `cannot_tell` for playability and '
            'return three empty lists: without frames you have not looked.')
    body = _PROMPT.format(
        intent=PE.design_intent(game, Path(work)) or '  (the task says nothing '
                                                     'about how it should look)',
        facts=PE.render_facts(facts), note=note,
        asked=asked or '  (nothing has been asked for yet)',
        focus_block=_FOCUS.get(focus, ''), rank_block=_RANK.get(focus, _RANK[None]),
        max_assets=cap['assets'], max_effects=cap['effects'],
        max_layout=cap['layout'])

    # A BUDGET IN BYTES, NOT ONLY IN FRAMES. The frame caps were chosen for
    # coverage -- three moments per demo, or a before and two afters per input
    # -- and say nothing about how big a frame is. One game's recordings are
    # large enough that sixty of them came to more than the endpoint's 30MB
    # limit, and every read of it failed with a 413 that the retry path
    # misreported as a refused reasoning ceiling. Five rounds died 35 seconds
    # in, recorded as art rounds with no repair in them at all.
    #
    # Dropped from the end, so what goes first is the tail of the last demo
    # rather than the opening of the first.
    content = [{'type': 'text', 'text': body}]
    used, dropped = 0, 0
    for label, fp in frames:
        uri = _data_uri(Path(fp))
        if used + len(uri) > _URI_BUDGET and content:
            dropped += 1
            continue
        used += len(uri)
        content.append({'type': 'image_url', 'image_url': {'url': uri}})
        content.append({'type': 'text', 'text': f'({label})'})
    if dropped:
        print(f'[art_list] {len(frames) - dropped}/{len(frames)} frames '
              f'({used / 1e6:.1f}MB)，dropped to fit the request cap {dropped} frames', flush=True)

    parsed, _ = PV._ask([{'role': 'user', 'content': content}],
                        model=model or os.environ.get('RSIGAME_MODELS_LOOP_MODEL'),
                        timeout=240.0)
    if not isinstance(parsed, dict):
        return {'error': 'the reading returned nothing after three tries'}
    out = _gate(parsed, facts, cap)
    out['focus'] = focus
    return out


def _s(x, n=300) -> str:
    return str(x or '').strip()[:n]


def _gate(p: dict, facts: dict, cap: dict | None = None) -> dict:
    """Keep the shape honest and drop what cannot point at a frame."""
    play = _s(p.get('playable'), 20).lower()
    out = {'playable': play if play in PLAYABLE else 'cannot_tell',
           'playable_why': _s(p.get('playable_why')),
           'landed': [], 'assets': [], 'effects': [], 'layout': [],
           'note': _s(p.get('note'), 400), 'dropped': [], 'facts': facts,
           # WHICH LISTS THE CAP CUT. Silently truncating hides the difference
           # between a reader that found three things and a reader that found
           # nine and was allowed to say four.
           'capped': []}
    cap = cap or {'assets': LEAD_CAP, 'effects': SIDE_CAP, 'layout': SIDE_CAP}
    raw = {k: len([x for x in (p.get(k) or []) if isinstance(x, dict)])
           for k in ('assets', 'effects', 'layout')}

    for x in (p.get('landed') or [])[:8]:
        if not isinstance(x, dict):
            continue
        seen = _s(x.get('seen'), 12).lower()
        out['landed'].append({'asked': _s(x.get('asked'), 200),
                              'seen': seen if seen in ('yes', 'no', 'cannot_tell')
                              else 'cannot_tell',
                              'note': _s(x.get('note'), 200)})

    for x in (p.get('assets') or []):
        if not isinstance(x, dict) or len(out['assets']) >= cap['assets']:
            continue
        key = _s(x.get('key'), 40).replace(' ', '_').lower()
        kind = _s(x.get('kind'), 20).lower()
        if not key or not _s(x.get('want')):
            out['dropped'].append(f'asset {key!r}: no key or nothing asked for')
            continue
        if not _s(x.get('seen_at')):
            # The rule the whole framework runs on, in its art form. An asset
            # nobody can point at is an asset nobody looked at, and it would
            # cost a generation call and a wiring edit to find that out.
            out['dropped'].append(f'asset {key!r}: names no frame')
            continue
        # ORDER IS PRESERVED, NOT RE-SORTED. The list arrives ranked by how
        # much the reader thinks each would change the look of the game, and
        # the cap below bites at the bottom -- which is only meaningful if
        # nothing here reorders it.
        parts = [_s(t, 40) for t in (x.get('parts') or [])
                 if _s(t, 40)][:12] if isinstance(x.get('parts'), list) else []
        out['assets'].append({'key': key, 'now': _s(x.get('now')),
                              'want': _s(x.get('want'), 400),
                              'kind': kind if kind in KINDS else 'image',
                              'parts': parts,
                              'leverage': _s(x.get('leverage'), 300),
                              'rank': len(out['assets']) + 1,
                              'seen_at': _s(x.get('seen_at'), 120)})

    for x in (p.get('effects') or []):
        if not isinstance(x, dict) or len(out['effects']) >= cap['effects']:
            continue
        if not _s(x.get('want')) or not _s(x.get('seen_at')):
            out['dropped'].append('effect: nothing asked for, or names no frame')
            continue
        out['effects'].append({'when': _s(x.get('when')), 'now': _s(x.get('now')),
                               'want': _s(x.get('want'), 400),
                               'leverage': _s(x.get('leverage'), 300),
                               'rank': len(out['effects']) + 1,
                               'seen_at': _s(x.get('seen_at'), 120)})

    for x in (p.get('layout') or []):
        if not isinstance(x, dict) or len(out['layout']) >= cap['layout']:
            continue
        if not _s(x.get('want')) or not _s(x.get('seen_at')):
            out['dropped'].append('layout: nothing asked for, or names no frame')
            continue
        panel = _s(x.get('panel'), 20).lower()
        out['layout'].append({'panel': panel if panel in PANELS else 'other',
                              'problem': _s(x.get('problem')),
                              'want': _s(x.get('want'), 400),
                              'seen_at': _s(x.get('seen_at'), 120)})
    for k in ('assets', 'effects', 'layout'):
        if raw[k] > cap[k]:
            out['capped'].append(f'{k}: the reader gave {raw[k]} items, cap {cap[k]}')
    out['n'] = len(out['assets']) + len(out['effects']) + len(out['layout'])
    return out


def brief(a: dict, focus: str | None = None) -> str:
    """The three lists as the repair agent reads them, the round's own first."""
    head = ('WHAT THIS GAME DOES NOT ANSWER' if focus == 'feedback'
            else 'WHAT IS STILL UNMADE ABOUT THIS GAME')
    lead = ('This round is for how the game ANSWERS: an action happens and the '
            'screen says so. The effects below are what it is for; the art and '
            'layout items are real but secondary this round.'
            if focus == 'feedback' else
            'A reader looked at the current build -- the demo library replayed, '
            'every frame below named -- and wrote these down. They are what '
            'this round is for. Nothing else is.')
    L = ['====================', head, '====================', '', lead, '']

    if focus == 'feedback' and a.get('effects'):
        L += ['ACTIONS THAT HAPPEN WITH NO ANSWER, in the order the reader put them',
              '']
        for x in a['effects']:
            L += [f"  when {x['when']}",
                  f"    now:  {x['now']}",
                  f"    want: {x['want']}"]
            if x.get('leverage'):
                L.append(f"    why it is here: {x['leverage']}")
            L += [f"    seen: {x['seen_at']}", '']

    if a.get('assets'):
        L += ['WHAT THE ART IS NOT CARRYING, in the order the reader put it',
              '']
        for x in a['assets']:
            L += [f"  {x['key']}   ({x['kind']})"
                  + (f"   -- a set: {', '.join(x['parts'])}"
                     if x.get('parts') else ''),
                  f"    now:  {x['now']}",
                  f"    want: {x['want']}"]
            if x.get('leverage'):
                L.append(f"    why it is here: {x['leverage']}")
            L += [f"    seen: {x['seen_at']}", '']
    if a.get('effects') and focus != 'feedback':
        L += ['ACTIONS THAT HAPPEN WITH NO PUNCTUATION', '']
        for x in a['effects']:
            L += [f"  when {x['when']}",
                  f"    now:  {x['now']}",
                  f"    want: {x['want']}",
                  f"    seen: {x['seen_at']}", '']
    if a.get('layout'):
        L += ['PANELS THAT ARE LAID OUT BADLY', '']
        for x in a['layout']:
            L += [f"  {x['panel']}: {x['problem']}",
                  f"    want: {x['want']}",
                  f"    seen: {x['seen_at']}", '']
    if not a.get('n'):
        L += ['The reader found nothing it could point at a frame for. Its '
              f"note: {a.get('note') or '(none)'}", '',
              'Use the round on the weakest thing you can see in the code '
              'yourself, and say in your report what you chose and why.', '']

    if a.get('landed'):
        L += ['WHAT LAST ROUND ASKED FOR, AND WHETHER IT IS ON SCREEN NOW', '']
        for x in a['landed']:
            L.append(f"  [{x['seen']}] {x['asked']}"
                     + (f" -- {x['note']}" if x['note'] else ''))
        L += ['', 'A `no` here is not a reason to ask again by itself. Look at '
                  'what is in the tree: an image that exists and is not loaded '
                  'needs wiring, not another image.', '']
    return '\n'.join(L)
