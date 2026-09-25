"""The corpus invariants that are easy to break and expensive to notice.

Each of these has a matching failure in the history of the pipeline: a stage
mix that spent most of its steps on single-turn data, a subsample that dropped
whole games instead of trimming depth, a repair target that taught the model to
emit one machine's absolute paths, and a sliding window that handed the model a
tool result whose call it had never seen.
"""
from __future__ import annotations

import random

from rsigame.data_pipeline import arms, decisions, mix, split


# --------------------------------------------------------------- step balance
def test_non_agentic_share_matches_the_target():
    """The mixer balances on ROWS, because one row is one optimizer step."""
    gen = 1000
    for target in (0.2, 0.387, 0.5):
        budget = target / (1.0 - target) * gen        # the rule mix.main applies
        total = gen + budget
        assert abs(budget / total - target) < 1e-9


def test_plan_is_cut_harder_than_repair():
    """Plan is the most redundant source, so it is the one subsampled hardest."""
    assert mix.PLAN_KEEP < 1.0
    assert mix.TARGET_NON_AGENTIC < 0.5, 'generation must stay the majority of the steps'


def test_stratify_keeps_every_game_and_trims_depth():
    rows = [{'task': f'game{g}', 'round': f'r{r:02d}'} for g in range(20) for r in range(10)]
    picked = mix.stratify(rows, 40, random.Random(0))
    assert len(picked) == 40
    assert len({r['task'] for r in picked}) == 20, 'breadth is kept, depth is trimmed'


def test_stratify_returns_everything_when_the_budget_is_larger():
    rows = [{'task': 'a', 'round': 'r01'}, {'task': 'b', 'round': 'r01'}]
    assert len(mix.stratify(rows, 99, random.Random(0))) == 2


# ------------------------------------------------------------------ arm shares
def test_repair_pools_are_a_partition_of_the_repair_budget():
    assert abs(sum(share for _, share, _, _ in arms.REPAIR_POOLS) - 1.0) < 1e-9


def test_agentic_godot_is_the_largest_repair_pool():
    """It is the closest shape to what the development loop actually runs."""
    biggest = max(arms.REPAIR_POOLS, key=lambda p: p[1])
    assert biggest[2] == 's3_agent_godot'


# ---------------------------------------------------------------- patch targets
def test_absolute_diff_headers_are_rewritten():
    patch = ('--- /somewhere/on/a/machine/mygame/src/main.gd\n'
             '+++ /somewhere/on/a/machine/mygame/src/main.gd\n'
             '@@ -1 +1 @@\n-a\n+b\n')
    out = mix.fix_patch('mygame', patch)
    assert '/somewhere/on/a/machine' not in out
    assert out.startswith('--- src/main.gd')


def test_relative_headers_are_left_alone():
    patch = '--- src/main.gd\n+++ src/main.gd\n'
    assert mix.fix_patch('mygame', patch) == patch


# ------------------------------------------------------------------ the split
def test_brief_stem_collapses_the_difficulty_suffix():
    assert split.brief_stem('platformer-moving-D3-044') == 'platformer-moving'
    assert split.brief_stem('cardgame-autobattler') == 'cardgame-autobattler'


def test_holdout_never_puts_one_stem_on_both_sides():
    by_stem = {f'fam{f}-{s}': [{'engine': 'web'}] * (s + 1)
               for f in range(4) for s in range(5)}
    chosen = split.choose_holdout(by_stem, 0.2, random.Random(1))
    assert chosen, 'a 20% holdout of 20 stems must not be empty'
    assert chosen <= set(by_stem)


# -------------------------------------------------------------- the decisions
def _session():
    return {'task': 't', 'engine': 'web', 'messages': [
        {'role': 'user', 'content': 'build a game'},
        {'role': 'assistant', 'tool_calls': [
            {'id': 'c1', 'type': 'function',
             'function': {'name': 'todo_write', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'ok'},
        {'role': 'assistant', 'tool_calls': [
            {'id': 'c2', 'type': 'function',
             'function': {'name': 'write_file', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'c2', 'content': 'written'},
        {'role': 'assistant', 'content': 'done'},
    ]}


def test_one_sample_per_assistant_turn():
    out = decisions.split_row(_session(), budget=4096, keep_plan=True, hist_as_ctx=False)
    assert len(out) == 3
    assert all(s['messages'][-1]['role'] == 'assistant' for s in out)


def test_no_sample_strands_a_tool_result():
    """A `tool` turn whose originating call fell outside the window teaches a
    result arriving from nowhere."""
    for s in decisions.split_row(_session(), budget=4096, keep_plan=True, hist_as_ctx=False):
        ids = {c['id'] for m in s['messages'] for c in (m.get('tool_calls') or [])}
        stranded = [m for m in s['messages']
                    if m['role'] == 'tool' and m.get('tool_call_id') not in ids]
        assert not stranded


def test_the_plan_turn_is_not_its_own_context():
    """Pinning the first assistant turn into the prefix of ITS OWN sample shows
    the model the answer and then asks for it."""
    out = decisions.split_row(_session(), budget=4096, keep_plan=True, hist_as_ctx=False)
    first = out[0]
    assert len(first['messages']) == 2, 'the first decision sees the brief only'


def test_history_as_context_leaves_one_supervised_turn():
    out = decisions.split_row(_session(), budget=4096, keep_plan=False, hist_as_ctx=True)
    last = out[-1]
    assert sum(1 for m in last['messages'] if m['role'] == 'assistant') == 1
