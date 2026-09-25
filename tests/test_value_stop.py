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
"""Value Stop: the rule that decides when a run has stopped paying."""
import pytest

from rsigame.monitor.value_stop import value_stop


def trace(champions):
    """A champion trace as vm_replay writes it: one record per checkpoint."""
    return [{'round': i * 3, 'champion': c} for i, c in enumerate(champions)]


def test_stops_once_the_champion_has_survived_k_checkpoints():
    # champion settles on r09 and is never beaten again
    s = value_stop(trace([0, 3, 3, 9, 9, 9, 9, 21, 21, 21, 21]), k=3)
    assert s.saturated and s.round == 18 and s.champion == 9


def test_a_run_that_keeps_improving_never_settles():
    s = value_stop(trace(list(range(11))), k=3)
    assert not s.saturated and s.round == 30


def test_k_is_consecutive_not_cumulative():
    # three unchanged checkpoints, then a new champion, then three more:
    # the stop belongs to the second stretch, not the first.
    s = value_stop(trace([0, 0, 0, 6, 6, 6, 6]), k=3)
    assert s.saturated and s.champion == 6 and s.round == 18


def test_the_base_game_can_be_the_delivered_build():
    s = value_stop(trace([0, 0, 0, 0]), k=3)
    assert s.saturated and s.champion == 0 and s.round == 9


@pytest.mark.parametrize('k', [1, 2, 5])
def test_k_is_honoured(k):
    s = value_stop(trace([0] * 8), k=k)
    assert s.saturated and s.checkpoint == k
