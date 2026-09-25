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
"""Offline champion replay. The code lives in the package; this is the CLI.

    RSIGAME_MONITOR_RUN=... RSIGAME_MONITOR_R0_DIR=... RSIGAME_MONITOR_OUT=... python scripts/vm_replay.py --games list.txt
"""
import sys

from rsigame.monitor import vm_replay

if __name__ == '__main__':
    vm_replay.setup()
    raise SystemExit(vm_replay.main())
