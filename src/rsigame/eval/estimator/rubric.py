#!/usr/bin/env python
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
"""The frozen proxy rubric: sixteen criteria, reworded once per task.

INSTANTIATED, NOT APPENDED. A generic criterion and its task-specific form are
the same criterion -- "the core interaction produces its intended consequence"
and "pressing attack visibly affects the enemy" are one question, not two --
so the generic set goes INTO the call and sixteen come back out, keeping their
ids. Concatenating the two would score the same thing twice, and would make
the generic-vs-instantiated ablation a comparison of sixteen items against
thirty rather than of one wording against another.

WHAT THE MODEL MAY CHANGE: the sentence and the three anchors. Nothing else.
Ids, categories and aggregation are structure -- they are what lets two arms
and ten games be added up by the same code -- and a rubric that quietly grew a
seventeenth item or renamed `max` to `mean` would break the comparison without
failing. `_verify_v2` refuses those and the call is retried.

FROZEN BY THE TASK DOCUMENT'S HASH. Generated once, cached, and never
regenerated after repair outcomes have been seen.

LANGUAGE. `DIMENSIONS` and `_PROMPT_V2` below are Chinese, and stay that way.
They are sent to the judge, and every number in the paper was produced with this
exact wording -- translating them would change the method, not its presentation.
Everything else in this file is English.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
sys.path.insert(0, str(RUN))
CACHE = HERE / 'rubrics'

# The judge, addressed directly rather than through the loop's model plumbing:
# the estimator must stay separable from the thing it measures.
#
# NOT A GPT-5.x MODEL, deliberately. The official evaluator is GPT-5.5, and an
# estimator validated against a label its own judge produced measures agreement
# with itself. Qwen keeps that separation and keeps continuity with the trend
# data already collected on the local qwen38-27b.
#
# ON THE API RATHER THAN :8038, because the local endpoint is the same GPU the
# repair loop and the 320-tree scoring run are on: a 32-image call queues
# behind ten of theirs and takes minutes that have nothing to do with the work.
BASE_URL = os.environ.get('RSIGAME_MONITOR_BASE_URL') or 'https://openrouter.ai/api/v1'
MODEL = os.environ.get('RSIGAME_MONITOR_MODEL') or 'qwen/qwen3.8-flash'


PROVIDER = os.environ.get('RSIGAME_MONITOR_PROVIDER') or 'alibaba'


def _local() -> bool:
    return '127.0.0.1' in BASE_URL or 'localhost' in BASE_URL

CATEGORIES = {'F': 'functional', 'C': 'content', 'I': 'feedback',
              'P': 'presentation'}

# `max` where the question is whether the game can do a thing at all -- one
# demo reaching a later stage proves later stages are reachable. `mean` where
# the question is a property the whole artifact either has or lacks, and one
# good screen does not redeem five bad ones.
GENERIC: list[dict] = [
    {'id': 'F1', 'category': 'F', 'aggregation': 'max',
     'criterion': 'In the scenarios observed, the core player interaction '
                  'behaves as the task describes.',
     'anchors': {
         '0.0': 'The interaction is attempted in the frames and does not work, '
                'or behaves contrary to what the task describes.',
         '0.5': 'It works partly, intermittently, or only in some of the cases '
                'the frames show.',
         '1.0': 'It works as described everywhere the frames exercise it.'}},
    {'id': 'F2', 'category': 'F', 'aggregation': 'max',
     'criterion': 'Rules produce consequences that are visible on screen.',
     'anchors': {
         '0.0': 'A rule fires and nothing observable follows.',
         '0.5': 'A consequence follows but is partial, delayed, or only '
                'inferable from a number.',
         '1.0': 'The consequence is directly visible when the rule fires.'}},
    {'id': 'F3', 'category': 'F', 'aggregation': 'mean',
     'criterion': 'The game stays playable and responsive throughout the '
                  'observed frames.',
     'anchors': {
         '0.0': 'It stalls, freezes, or stops responding to input.',
         '0.5': 'It keeps running but with visible hitches, lost inputs, or '
                'recovery that takes noticeable time.',
         '1.0': 'It runs and answers input steadily throughout.'}},
    {'id': 'F4', 'category': 'F', 'aggregation': 'mean',
     'criterion': 'No blocking malfunction appears in the observed scenarios.',
     'anchors': {
         '0.0': 'Something stops the scenario from continuing -- a crash, a '
                'blank screen, an unusable control.',
         '0.5': 'A malfunction is visible but the scenario still continues.',
         '1.0': 'Nothing in the frames blocks or degrades the scenario.'}},
    {'id': 'C1', 'category': 'C', 'aggregation': 'max',
     'criterion': 'The observed play advances materially beyond the opening '
                  'state.',
     'anchors': {
         '0.0': 'The frames end where they began; nothing has progressed.',
         '0.5': 'Something advanced, but only slightly or in one dimension.',
         '1.0': 'The run clearly moves into a different state of play.'}},
    {'id': 'C2', 'category': 'C', 'aggregation': 'max',
     'criterion': 'Observable variety exists in the content or the play.',
     'anchors': {
         '0.0': 'Everything observed is the same element or the same beat '
                'repeated.',
         '0.5': 'Some variation exists but it is thin or cosmetic.',
         '1.0': 'Distinct content or distinct situations are visible.'}},
    {'id': 'C3', 'category': 'C', 'aggregation': 'max',
     'criterion': 'The later-stage or goal states the task requires are '
                  'reachable in the frames.',
     'anchors': {
         '0.0': 'A required later state is attempted and never reached.',
         '0.5': 'It is reached but incompletely, or only its edge is visible.',
         '1.0': 'It is reached and shown.'}},
    {'id': 'C4', 'category': 'C', 'aggregation': 'max',
     'criterion': 'The experience extends past a single trivial interaction '
                  'loop.',
     'anchors': {
         '0.0': 'One action repeated is the whole of what the frames show.',
         '0.5': 'There is a second layer, but it is shallow or barely used.',
         '1.0': 'Several interacting elements make up the observed play.'}},
    {'id': 'I1', 'category': 'I', 'aggregation': 'max',
     'criterion': 'Player actions produce clear visible feedback.',
     'anchors': {
         '0.0': 'The action happens and the screen says nothing.',
         '0.5': 'Some feedback exists but is weak, late, ambiguous, or only a '
                'number changing.',
         '1.0': 'The action produces an immediate, unambiguous visible '
                'response.'}},
    {'id': 'I2', 'category': 'I', 'aggregation': 'mean',
     'criterion': 'Important state changes are readable as they happen.',
     'anchors': {
         '0.0': 'State changes without anything on screen marking it.',
         '0.5': 'It is marked, but easy to miss or hard to read.',
         '1.0': 'Each important change is plainly readable.'}},
    {'id': 'I3', 'category': 'I', 'aggregation': 'max',
     'criterion': 'Success, failure and transitions are clearly communicated.',
     'anchors': {
         '0.0': 'An outcome or transition occurs with no announcement.',
         '0.5': 'It is announced but thinly -- a line of text, a silent cut.',
         '1.0': 'The outcome is presented so the player cannot miss it.'}},
    {'id': 'I4', 'category': 'I', 'aggregation': 'mean',
     'criterion': 'The UI stays legible and stable during interaction.',
     'anchors': {
         '0.0': 'Text is clipped, overlapping, or unreadable; elements jump.',
         '0.5': 'Mostly readable with some crowding, clipping or drift.',
         '1.0': 'Legible and stable wherever the frames show it.'}},
    {'id': 'P1', 'category': 'P', 'aggregation': 'mean',
     'criterion': 'The main visual subjects look deliberately authored rather '
                  'than placeholder.',
     'anchors': {
         '0.0': 'Main subjects are plain geometry or obvious stand-ins.',
         '0.5': 'Authored but generic, off-style, or rougher than their '
                'surroundings.',
         '1.0': 'Authored, specific, and of a piece with the rest.'}},
    {'id': 'P2', 'category': 'P', 'aggregation': 'mean',
     'criterion': 'Scene and background composition has clear hierarchy and '
                  'coherence.',
     'anchors': {
         '0.0': 'The screen reads as unrelated pieces; no foreground/ '
                'background separation.',
         '0.5': 'A hierarchy exists but competes with itself or is uneven.',
         '1.0': 'The eye is led; foreground, playfield and background hold '
                'together.'}},
    {'id': 'P3', 'category': 'P', 'aggregation': 'max',
     'criterion': 'Full-screen, state-defining presentation is complete where '
                  'the frames reach it.',
     'anchors': {
         '0.0': 'A title, transition or end screen is reached and is blank, '
                'broken, or a bare rectangle.',
         '0.5': 'Present but unfinished -- placeholder text, missing art, '
                'partial layout.',
         '1.0': 'Complete and finished-looking where shown.'}},
    {'id': 'P4', 'category': 'P', 'aggregation': 'mean',
     'criterion': 'Effects, animation and motion contribute to a finished '
                  'feel.',
     'anchors': {
         '0.0': 'Everything moves as hard cuts or teleports; no motion '
                'treatment.',
         '0.5': 'Some motion or effects, applied unevenly.',
         '1.0': 'Motion and effects are present and consistent with the '
                'rest.'}},
]



def _key() -> str:
    """The key, from the environment or from the file the host keeps it in.

    `~/.env` is where every other arm on this machine gets it; reading it here
    means a run started from a bare shell does not silently fall back to an
    unauthenticated call and retry three times.
    """
    for v in ('RSIGAME_MONITOR_API_KEY', 'OPENROUTER_API_KEY', 'OPENAI_API_KEY'):
        if os.environ.get(v):
            return os.environ[v]
    f = Path(os.environ.get('RSIGAME_PATHS_ENV') or Path.cwd() / '.env')
    if f.is_file():
        for line in f.read_text().splitlines():
            k, _, v = line.partition('=')
            if k.strip() == 'OPENROUTER_API_KEY' and v.strip():
                return v.strip()
    return 'local'


def _client():
    from openai import OpenAI
    return OpenAI(base_url=BASE_URL, api_key=_key())


def ask(messages: list, *, max_tokens: int = 8000, timeout: float = 300.0):
    """One strict-JSON call to the local judge, with the two settings that
    matter for this model measured elsewhere in this repo.

    `enable_thinking: False` -- a reasoning model served by vLLM otherwise
    spends the whole budget thinking and returns an empty body (85s and zero
    characters against 21s and a complete object). `temperature: 0` -- vLLM
    otherwise applies the model's own generation_config (1.0 / top_p 0.95), and
    a rubric graded at 1.0 by a 27B moved 57 of 65 item scores between passes.
    """
    import re
    cl = _client()
    last = ''
    for attempt in range(3):
        if attempt:
            time.sleep(3 * attempt)
        try:
            # THE SAME SETTING, SPELLED TWO WAYS.
            #
            # `chat_template_kwargs` is a vLLM serving argument, not part of
            # the API -- a gateway either rejects it or passes it to a provider
            # that ignores it -- so the local server gets that and the gateway
            # gets `reasoning: {enabled: false}`, which is how OpenRouter says
            # the same thing. IT IS NOT OPTIONAL ON EITHER: measured on this
            # exact 32-image call, the default spent all 6000 completion tokens
            # on reasoning and returned `finish_reason=length` with a body of
            # zero characters; with reasoning off the same call answered in 25s
            # with 5.5KB of valid JSON.
            #
            # PROVIDER PINNED, because OpenRouter routed the probe calls to
            # Alibaba and Makora on consecutive requests, and two checkpoints
            # of one game scored on two providers are not comparable.
            kw = ({'extra_body': {'chat_template_kwargs':
                                  {'enable_thinking': False}}} if _local() else
                  {'extra_body': {
                      'reasoning': {'enabled': False},
                      'provider': {'order': [PROVIDER],
                                   'allow_fallbacks': False},
                      # CONSTRAINED DECODING, because the failures that got
                      # through were not truncation and not reasoning -- they
                      # were single malformed characters, once in ~50 calls:
                      # `"new: frame 16/16 (100% in)"\n      ",` where the
                      # closing bracket belonged. The retry covers it at the
                      # cost of another 100-second call; this removes the
                      # class. Verified accepted by the pinned provider.
                      'response_format': {'type': 'json_object'}}})
            r = cl.chat.completions.create(
                model=MODEL, messages=messages, max_tokens=max_tokens,
                temperature=0.0, timeout=timeout, **kw)
        except Exception as exc:
            last = f'{type(exc).__name__}: {str(exc)[:200]}'
            print(f'[est] try {attempt + 1}: {last}', flush=True)
            continue
        txt = (r.choices[0].message.content or '') if r.choices else ''
        last = txt[:200]
        m = re.search(r'\{.*\}', txt, re.S)
        if m:
            try:
                return json.loads(m.group(0)), txt
            except Exception as exc:
                last = f'JSON: {exc}'
        # THE TEXT ITSELF, NOT ITS LENGTH. A character count invites a guess
        # about the cause -- the first guess here was "truncated at the token
        # ceiling", which the usage numbers contradict -- so the body is kept
        # and the guess is unnecessary.
        d = Path(os.environ.get('RSIGAME_MONITOR_FAIL_DIR') or HERE / 'failures')
        try:
            d.mkdir(parents=True, exist_ok=True)
            f = d / f'{time.strftime("%H%M%S")}_{attempt + 1}_{os.getpid()}.txt'
            f.write_text(txt or '(empty body)')
        except OSError:
            f = '(could not be written)'
        print(f'[est] try {attempt + 1}: answer is not JSON ({len(txt)} chars) -> {f}',
              flush=True)
    return None, last




def key_for(game: str) -> tuple[str, str]:
    import rsigame.checklist.task_checklist as TC
    raw, src = TC.task_document(game)
    return hashlib.sha256(raw.encode()).hexdigest()[:16], raw


# ---------------------------------------------------------------- v2
#
# The skeleton stops being "sixteen generic slots" and becomes "what THIS game
# was asked for".
#
# WHY. v1 handed the model sixteen fixed ids and let it reword them, so the
# ruler's resolution had nothing to do with the game. On critter, one item --
# "rules produce visible consequences" -- covered seven separate task
# requirements at once (type advantage, status effects, damage, the advantage
# hint, burn, sleep, paralysis), and satisfying the easiest of them (the health
# bar moves) scored full marks. Measured: thirteen of the sixteen items sat at
# exactly 1.000 on both sides of both checkpoints, all four categories had 0.0
# headroom, and an unrepaired G0 scored 0.93-1.00 where the official scorer gave
# it 0.422. A ruler that cannot separate two builds is of no use to the monitor.
#
# PROVENANCE IS REQUIRED FOR F AND C ONLY. Those two dimensions ask whether what
# the task document asked for was done, so inventing an item there means marking
# a game down for something nobody ever asked of it. I and P ask something else:
# whether it was done WELL -- and "well" is by definition beyond the document.
# Whether the art was made for this game, whether a hit has an effect, whether
# the text is legible, whether losing health is visible: a task document lists
# none of these, and they are exactly what separates a running prototype from a
# finished game. The measurement agrees: under v2 the document half still sits
# near 1.0 and almost all of the separation comes from I/P -- G0's "hit effect"
# is 0.0 and only reaches 0.333 by r3.
#
# EVERY ITEM RECORDS ITS `origin`, because there is a real tension here: the
# estimator is validated against the official score, and the official scorer does
# not necessarily reward "good beyond the document". Whether to keep that kind of
# item cannot be settled by argument -- it needs each half correlated against the
# official score separately, and that is only computable if the origin was
# written down.

SCHEMA = 'v2'

DIMENSIONS = """
F  功能 —— 这个游戏到底能不能玩
   F 要问的是"按下去有没有发生该发生的事"，不是"好不好看"。
   · 说明书点名的那个核心操作，做出来之后屏幕上真的出现了它该产生的结果
   · 规则触发的那一刻，画面上有对应的变化（而不是只有一个数字悄悄变了）
   · 没有卡死、黑屏、点不动的控件、按了没反应的按钮
   · 失败或结束之后能重来，不需要重启

C  内容 —— 有多少东西真的做出来了
   C 要问的是"玩下去还有没有东西"，不是"有没有 bug"。
   · 观察到的过程越过了开场，而不是停在第一屏或第一回合
   · 说明书点名的那几类元素真的都出现了（不同的敌人、道具、关卡、招式、
     阶段……具体哪几类由说明书决定）
   · 胜利/失败/结算这类终局画面真的到达了，并且内容完整

I  反馈 —— 游戏有没有把发生的事告诉玩家
   I 要问的是"玩家能不能当场知道发生了什么"。
   · 每一次操作都有立刻可见的回应：高亮、动作、闪光、文字
   · 关键数值变化——掉血、得分、命数、弹药、状态、进度——在它发生的那一刻
     屏幕上看得见，而且读得出来是变了多少
   · HUD 上玩家正在消耗或积累的那些数字一直在画面上，需要时读得清
   · 结局有专门的画面明确告知，不是悄悄回到标题

P  呈现 —— 好不好看
   P 要问的是"这东西像不像一个做完的游戏"。
   · 美术素材是专门做的、好看：角色/场景/道具不是纯色块、几何体、占位图
   · 交互特效做足了：命中闪光、受击震动、粒子、拖尾、过场，而不是硬切
   · 文字清晰：不裁切、不重叠、不出屏、和背景对比度够
   · 画面构图有层次：前景、玩法区、背景分得开，HUD 不压住玩法
""".strip()

_PROMPT_V2 = """你要为一个具体的游戏写一份质检清单，给一个从没玩过它的人用——
那个人只会看到这个游戏运行时的若干张静止画面，没有声音，没有视频。

下面给你三样东西：四个维度各自"要看什么"、这个游戏的说明书、以及一份从说明书
里拆出来的需求清单。请据此写出 {lo} 到 {hi} 条针对这个游戏的具体检查项。

硬规则：

- 每条只检查一件事。不要把"选招、换人、用道具"三件事写成一条——那样做到
  最容易的一件就能拿满分，这正是上一版失败的原因。
- 只用说明书里出现过的叫法。说明书叫它 Critter 就写 Critter，不要换成"宝可梦"
  或任何别的游戏的词汇——那会让看图的人去找一个根本不存在的东西。
- 每条必须点名这个游戏里真实存在的东西：真实的按键、真实的画面、真实的对象、
  真实的数值。"按 SPACE 后小鸡向前跳一格"是可用的；"核心操作正常"不是。
- 每条必须能只看静止画面回答，**三档锚点也一样**。凡是要听声音、要看帧率、
  要测时间精度、要连续观察几秒、要亲手点一下才知道的，一律不要写进来。
  "点了之后崩不崩溃"、"有没有悬停高亮"这种，看图的人答不了。
- 待在自己的维度里。F 问"做没做到"，I 问"有没有告诉你"，P 问"好不好看"。
  "点击后角色高亮"是 I 不是 F；"换人换成功了"是 F 不是 I。同一件事写进两个
  维度会被算两次，四个维度就不再独立。
- P 维度必须把四件事都覆盖到，每件至少一条：①美术素材是不是专门做的、好看
  ②交互特效够不够（命中闪光、受击震动、粒子、拖尾、过场）③文字清不清楚
  ④构图/HUD 有没有层次。少一件就不合格。
- I 维度必须包含一条专门问"关键数值变化在发生的那一刻看不看得见"
  （掉血、得分、命数、弹药、进度——取这个游戏真正有的那个）。
- 每条必须有区分度：一个做得平庸的版本应该会在这条上失分。如果一条"只要游戏
  能跑起来就满分"，它就没有价值，删掉换一条。
- 四个维度都要有。F 和 C 合起来**至少 6 条**（那是这个游戏被要求做到的东西，
  是清单的主体）；I 和 P 合起来**不超过 8 条**（质量标准重要，但不该压过要求本身）。
- 出处规则，F/C 和 I/P 不一样：
    F 功能 / C 内容 —— 问的是"任务书要求的东西做没做到"。每条必须在 `quote` 里
      给出说明书的**原文片段**：从说明书里**逐字复制**，保持原来的语言和大小写，
      不要翻译、不要改写、不要合并两句。转述会被机械核验挡掉，那条就白写了。
      抄不出原文的，说明是你自己加的，删掉。
    I 反馈 / P 呈现 —— 问的是"做得好不好"，而"好"本来就超出说明书：美术好不好看、
      特效够不够、文字清不清楚、掉血看不看得见，没有一件是任务书会逐条写的，
      而正是这类要求把一个能跑的原型和一个做完的游戏分开。这两个维度 `quote`
      一律写 "GENERIC"，不需要原文。但仍然要点名这个游戏里的具体对象——具体那个
      数值、那块 HUD、那个角色、那种特效——并且必须是一个平庸的版本会失分的问题。
      "掉血看不看得见"要落到这个游戏里掉血到底该长什么样。
  说明书提了但静止画面读不出来的（伤害公式、随机种子、跨次启动的存档），不要写。
- aggregation 不是随便选的，按这条规则定：
    问"这个游戏做没做到某件事、某个东西出没出现过" -> max
      （一个 demo 演示到就算数，别的 demo 没走到那里不该扣分）
    问"整体是不是都这样、是不是处处如此" -> mean
      （一个画面干净抵消不了另外四个乱的）
  凡是写成"是否至少出现过一次 / 是否出现了 / 能不能"的，一律 max。
- 三档锚点必须描述同一件事的三种程度，而且 0.0 要是"确实做了检查、确实不行"，
  不是"没看到"。没看到由打分时的 null 处理，不归 0。

每条给出：
  id           F1 C1 I1 P1 这样的编号，同维度内依次编号
  dimension    F / C / I / P
  aggregation  max 或 mean。问"这个游戏能不能做到某件事"用 max（一个 demo
               演示到就算数）；问"整体是不是都这样"用 mean（一个画面干净不能
               抵消另外四个乱的）
  criterion    一句话，就是上面说的那种具体问题
  anchors      三档 "0.0" "0.5" "1.0"
  quote        说明书里的原文片段（P 维度的通用四条可写 "GENERIC"）
  why          一句话说明这条在检查什么

只回严格 JSON：
{{"items": [{{"id": "F1", "dimension": "F", "aggregation": "max",
            "criterion": "...", "anchors": {{"0.0": "...", "0.5": "...", "1.0": "..."}},
            "quote": "...", "why": "..."}}, ...]}}

====================
四个维度：要看什么
====================
{dims}

====================
这个游戏的说明书
====================
{task}

====================
从说明书拆出来的需求清单（供定位，不必逐条对应）
====================
{reqs}
"""


def _requirements(game: str) -> str:
    import rsigame.checklist.task_checklist as TC
    got = TC.extract(game)
    return '\n'.join(
        f"{it.get('id')}  [{it.get('stage')}]  {it.get('requirement')}"
        for it in (got.get('items') or []))


def _verify_v2(items, raw):
    """Check where F and C items came from, and fix aggregation from the wording.

    The check reuses task_checklist's `quote_match`: it exists to answer exactly
    "is this sentence in the task document or not", so this asks it rather than
    inventing a second ruler.
    """
    import rsigame.checklist.task_checklist as TC
    toks = TC._tokens(raw)
    keep, drop = [], []
    for it in items:
        if not isinstance(it, dict):
            continue
        dim = it.get('dimension')
        if dim not in CATEGORIES:
            drop.append((it.get('id'), f'unknown dimension {dim!r}'))
            continue
        q = (it.get('quote') or '').strip()
        if dim in ('I', 'P'):
            it['origin'] = 'generic'
        elif not q or q.upper() == 'GENERIC':
            drop.append((it.get('id'), f'{dim} requires a quote from the document'))
            continue
        else:
            m = TC.quote_match(q, toks)
            it['quote_match'] = round(m, 3)
            it['origin'] = 'task'
            if m < 0.6:
                drop.append((it.get('id'), f'quote does not match ({m:.2f}): {q[:60]}'))
                continue
        if it.get('aggregation') not in ('max', 'mean'):
            it['aggregation'] = 'max'
        c = it.get('criterion') or ''
        if not c or not isinstance(it.get('anchors'), dict) \
                or any(k not in it['anchors'] for k in ('0.0', '0.5', '1.0')):
            drop.append((it.get('id'), 'missing criterion or the three anchors'))
            continue
        # `mean` is wrong for an existence question: a demo that never got
        # there should not drag the score down, which is what `max` is for.
        # One run produced five items phrased "does it appear at least once"
        # and tagged them `mean`.
        # These stay in Chinese on purpose: the prompt above is Chinese, so the
        # criterion the model writes is too, and this matches its wording.
        if any(k in c for k in ('至少', '是否出现', '出现过', '能不能', '有没有出现')):
            it['aggregation'] = 'max'
        keep.append(it)
    return keep, drop


def instantiate(game: str, *, force: bool = False) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    h, raw = key_for(game)
    out = CACHE / f'{game}.{SCHEMA}.{h}.json'
    if out.is_file() and not force:
        return json.loads(out.read_text())
    if not raw.strip():
        raise SystemExit(f'{game}: no task document found')
    body = _PROMPT_V2.format(lo=12, hi=20, dims=DIMENSIONS, task=raw[:14000],
                             reqs=_requirements(game))
    for attempt in range(3):
        parsed, txt = ask([{'role': 'user', 'content': body}], max_tokens=12000)
        items = (parsed or {}).get('items') if isinstance(parsed, dict) else None
        if not items:
            print(f'[rubric] {game}: attempt {attempt + 1} returned no items', flush=True)
            continue
        keep, drop = _verify_v2(items, raw)
        # A dropped item is usually a quote that was paraphrased rather than a
        # requirement that was wrong. Asking once more for those costs less than
        # losing two real functional requirements.
        if drop:
            why = '\n'.join(f'{a}: {b}' for a, b in drop)
            ids = ', '.join(str(a) for a, _ in drop)
            parsed2, _ = ask([{'role': 'user', 'content': body
                               + f'\n\nThese items were refused by the mechanical check:\n{why}\n\n'
                                 f'The problem is almost always the quote: an F or C quote must be '
                                 f'copied verbatim from the task document, not paraphrased or '
                                 f'translated. Return only those items ({ids}), same format.'}],
                             max_tokens=6000)
            if isinstance(parsed2, dict) and parsed2.get('items'):
                more, drop = _verify_v2(parsed2['items'], raw)
                keep += more
        n_fc = sum(1 for x in keep if x['dimension'] in ('F', 'C'))
        if len(keep) < 8 or n_fc < 3:
            print(f'[rubric] {game}: attempt {attempt + 1} kept only {len(keep)} items '
                  f'(F/C {n_fc}), retrying', flush=True)
            continue
        rb = {'game': game, 'schema': SCHEMA, 'task_hash': h,
              'items': [{'id': x['id'], 'category': x['dimension'],
                         'aggregation': x['aggregation'],
                         'criterion': x['criterion'], 'anchors': x['anchors'],
                         'origin': x.get('origin'),
                         'quote': x.get('quote'),
                         'quote_match': x.get('quote_match')}
                        for x in keep],
              'dropped': drop}
        out.write_text(json.dumps(rb, indent=1, ensure_ascii=False))
        print(f"[rubric] {game}: {len(rb['items'])} items "
              f"(from the document {sum(1 for x in rb['items'] if x['origin'] == 'task')} / "
              f"generic {sum(1 for x in rb['items'] if x['origin'] == 'generic')})"
              f" -> {out.name}", flush=True)
        return rb
    raise SystemExit(f'{game}: three attempts produced no usable checklist')


def load(game: str) -> dict:
    return instantiate(game)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('games', nargs='+')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    for g in a.games:
        instantiate(g, force=a.force)
