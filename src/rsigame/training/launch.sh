#!/usr/bin/env bash
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
# One SFT run on one node, from a shell that assumes nothing about the cluster.
#
#   ARM=s1s2s3 DATA=/data/s1s2s3.jsonl OUT=/scratch/run bash launch.sh
#
# Everything else has a default or comes from the environment:
#
#   RSIGAME_SFT_BASE_MODEL       the base model (required)
#   RSIGAME_SFT_CHECKPOINT_DIR   durable checkpoint destination (optional)
#   RSIGAME_SFT_NPROC            cards to use (default 4 -- see sft.py on why)
#   STAGE_MODEL=1                copy the model to local scratch first
#
# The Python entry point does the real work; this exists so a scheduler that
# can only run a shell command has something to run, and so the two checks
# worth making before a multi-hour job are made.
set -uo pipefail

: "${ARM:?set ARM to the arm being trained (s1 | s1s2 | s1s2s3)}"
: "${DATA:?set DATA to the training file for this arm}"
: "${OUT:=/tmp/rsigame_sft_${ARM}}"
: "${RSIGAME_SFT_NPROC:=4}"
MODEL=${RSIGAME_SFT_BASE_MODEL:?set RSIGAME_SFT_BASE_MODEL to the base model}

echo "### arm=$ARM data=$DATA out=$OUT host=$(hostname)"
[ -s "$DATA" ] || { echo "### ABORT: no training data at $DATA"; exit 4; }
echo "### rows: $(wc -l < "$DATA")"

# ---- check the cards before spending anything ------------------------------
# Capture once and slice with awk: `nvidia-smi | head -1` under `pipefail` is a
# SIGPIPE that exits 141 before the check it guards can even run.
SMI=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null || true)
NGPU=$(printf '%s\n' "$SMI" | grep -c . || true)
NAME=$(printf '%s\n' "$SMI" | awk -F', *' 'NR==1{print $1}')
MEM=$(printf '%s\n' "$SMI" | awk -F', *' 'NR==1{print $2}')
echo "### cards: ${NGPU}x ${NAME:-none} ${MEM:-0}MiB"
[ "${NGPU:-0}" -lt "$RSIGAME_SFT_NPROC" ] && {
  echo "### ABORT: $NGPU cards, need $RSIGAME_SFT_NPROC"; exit 10; }
[ "${MEM:-0}" -lt 79000 ] && {
  echo "### ABORT: ${MEM}MiB per card is not enough for a 27B base at 32k context"; exit 10; }

# ---- optionally stage the model onto node-local disk -----------------------
# One sequential reader beats N parallel mmap faulters on a network mount by an
# order of magnitude (measured: 4.5 MB/s parallel against ~76 MB/s sequential),
# and the model is read by every rank.
if [ "${STAGE_MODEL:-0}" = "1" ]; then
  LOCAL=/tmp/model_local/$(basename "$MODEL")
  if [ ! -s "$LOCAL/config.json" ]; then
    echo "### staging model -> $LOCAL"
    mkdir -p "$LOCAL"
    t0=$(date +%s)
    cp -r "$MODEL"/. "$LOCAL"/ || { echo "### ABORT: staging failed"; exit 5; }
    echo "### staged in $(( $(date +%s) - t0 ))s"
  fi
  export RSIGAME_SFT_BASE_MODEL="$LOCAL"
fi

python3 -m rsigame.training sft --data "$DATA" --out "$OUT" "$@"
rc=$?
echo "### done rc=$rc"
exit $rc
