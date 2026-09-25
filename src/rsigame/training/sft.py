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
"""One supervised fine-tuning run: build the command, run it, keep the checkpoints.

WHAT IT RUNS
    ms-swift's SFT entry under `torch.distributed.run`, single node, one rank
    per card. LoRA on every linear module, a 4-bit NF4 resident base with bf16
    compute, batch size 1, one epoch, cosine schedule.

THE SETTINGS THAT ARE NOT DEFAULTS, AND WHY

  sequence_parallel_size = 4
      Forced by the model, not chosen: ms-swift sets sp = gcd(kv_heads,
      world_size) and rp = world_size / sp. With 4 KV heads, four cards give
      sp=4, rp=1 -- pure Ulysses. Eight give rp=2, which needs ring attention,
      which is unimplemented for this model's linear-attention blocks and
      refuses. Asking for more cards does not shorten the per-rank sequence.

  quantisation (nf4, bf16 compute)
      Drops the resident weights from ~49 to ~14 GiB per rank, which is what
      makes long agent sessions fit at all.

  deepspeed off
      With a 4-bit resident base there is nothing left to shard; ZeRO-3 only
      adds communication.

  dataset_shuffle = false
      The corpus is ordered on purpose -- `data_pipeline.mix` explains the
      canary-then-shuffle ordering and `data_pipeline.arms` the longest-first
      one. Re-shuffling here would throw both away.

  loss_scale
      This is the one that decides what a row teaches, and it is worth stating
      plainly because we got it wrong once. With per-decision rows,
      `last_round` supervises only the final assistant turn: a median of 114
      tokens per step. A run at that setting produced a loss with no trend at
      all -- corr(loss, step) ~= 0 over 350 steps, where whole-session runs
      reached -0.35. Supervising every assistant turn (`default`) with the
      higher learning rate restored the trend. `default` is the default here
      for that reason; `last_round` remains available because it is the right
      answer if the rows are ever rebuilt so history is context rather than
      target (see `data_pipeline.decisions --history-as-context`).

  attn_impl = sdpa, with the cuDNN backend disabled in-process
      Every run on this stack died in cuDNN's fused attention
      (`mha_graph->execute`). The documented environment switch is IGNORED by
      this torch build, so the backend is turned off in Python, in every
      worker, through a sitecustomize this module writes.

CHECKPOINTS go to local scratch and are synced by a background thread; see
`checkpoints`, which also explains the resume rule.
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path

from . import checkpoints

def free_port() -> int:
    """Ask the kernel for a port instead of guessing one.

    The obvious default (29500) was already held on these nodes and killed
    three attempts about two seconds in, before the model had even loaded. A
    second hard-coded guess is the same bet.
    """
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def write_sitecustomize(dirpath: Path) -> Path:
    """Disable the cuDNN SDPA backend in every worker process.

    `TORCH_CUDNN_SDPA_ENABLED=0` is ignored by this torch build -- measured,
    not assumed: every run still died in `mha_graph->execute` with it set. The
    only switch that takes effect is the Python one, and it has to run in each
    spawned worker, which is what sitecustomize is for.
    """
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / 'sitecustomize.py'
    p.write_text('import torch\ntorch.backends.cuda.enable_cudnn_sdp(False)\n')
    return p


def build_command(a: argparse.Namespace, resume: list[str]) -> list[str]:
    """-> the full argv of the training process."""
    port = a.port or free_port()
    cmd = [
        sys.executable, '-m', 'torch.distributed.run',
        '--nproc_per_node', str(a.nproc),
        '--rdzv-backend=c10d', f'--rdzv-endpoint=127.0.0.1:{port}',
        '--master-addr=127.0.0.1', f'--master-port={port}',
        '--max-restarts', '0',
        '-m', 'swift.cli.sft',
        '--model', str(a.model),
        '--dataset', str(a.data),
        '--tuner_type', 'lora',
        '--lora_rank', str(a.lora_rank),
        '--lora_alpha', str(a.lora_alpha),
        '--lora_dropout', str(a.lora_dropout),
        '--target_modules', 'all-linear',
        '--torch_dtype', 'bfloat16',
        '--max_length', str(a.max_length),
        '--truncation_strategy', a.truncation,
        '--sequence_parallel_size', str(a.sequence_parallel_size),
        '--dataset_shuffle', 'false',
        '--train_dataloader_shuffle', 'false',
        '--attn_impl', a.attn,
        '--use_liger_kernel', 'true',
        '--gradient_checkpointing', 'true',
        '--per_device_train_batch_size', '1',
        '--gradient_accumulation_steps', '1',
        '--num_train_epochs', '1',
        '--learning_rate', str(a.learning_rate),
        '--weight_decay', str(a.weight_decay),
        '--warmup_ratio', str(a.warmup_ratio),
        '--max_grad_norm', str(a.max_grad_norm),
        '--lr_scheduler_type', a.scheduler,
        '--logging_steps', '1',
        '--save_steps', str(a.save_steps),
        '--save_total_limit', str(a.save_limit),
        '--report_to', 'tensorboard',
        '--output_dir', str(a.out),
    ]
    if a.rslora:
        cmd += ['--use_rslora', 'true']
    if a.quantize:
        cmd += ['--quant_method', 'bnb', '--quant_bits', '4',
                '--bnb_4bit_compute_dtype', 'bfloat16',
                '--bnb_4bit_quant_type', 'nf4',
                '--bnb_4bit_use_double_quant', 'true']
    if a.loss_scale:
        cmd += ['--loss_scale', a.loss_scale]
    if a.agent_template:
        cmd += ['--agent_template', a.agent_template]
    if a.padding_free:
        cmd += ['--padding_free', 'true']
    if a.deepspeed:
        cmd += ['--deepspeed', a.deepspeed]
    return cmd + resume


def main(argv: list[str] | None = None) -> int:
    env = os.environ.get
    ap = argparse.ArgumentParser(prog='rsigame.training sft', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help="the arm's training file (.jsonl)")
    ap.add_argument('--out', type=Path, required=True, help='run output, on local scratch')
    ap.add_argument('--model', default=env('RSIGAME_SFT_BASE_MODEL'),
                    help='base model (RSIGAME_SFT_BASE_MODEL)')
    ap.add_argument('--remote', type=Path, default=env('RSIGAME_SFT_CHECKPOINT_DIR'),
                    help='durable checkpoint destination (RSIGAME_SFT_CHECKPOINT_DIR)')
    ap.add_argument('--sync-interval', type=int, default=180)
    ap.add_argument('--nproc', type=int, default=int(env('RSIGAME_SFT_NPROC', '4')))
    ap.add_argument('--port', type=int, default=0, help='0 asks the kernel for a free one')
    ap.add_argument('--attn', default='sdpa')
    ap.add_argument('--truncation', default='delete',
                    help="'delete' drops a row that does not fit rather than cutting it")
    ap.add_argument('--no-quantize', dest='quantize', action='store_false')
    ap.add_argument('--no-rslora', dest='rslora', action='store_false')
    ap.add_argument('--padding-free', action='store_true')
    ap.add_argument('--deepspeed', default=None, help='e.g. zero3_offload; off by default')
    ap.add_argument('--dry-run', action='store_true', help='print the command and exit')
    ap.add_argument('--lora-rank', type=int, default=int(env('RSIGAME_SFT_LORA_RANK', 16)))
    ap.add_argument('--lora-alpha', type=int, default=int(env('RSIGAME_SFT_LORA_ALPHA', 32)))
    ap.add_argument('--lora-dropout', type=float,
                    default=float(env('RSIGAME_SFT_LORA_DROPOUT', 0.05)))
    ap.add_argument('--learning-rate', default=env('RSIGAME_SFT_LEARNING_RATE', '1e-4'))
    ap.add_argument('--weight-decay', type=float, default=float(env('RSIGAME_SFT_WEIGHT_DECAY', 0.01)))
    ap.add_argument('--warmup-ratio', type=float, default=float(env('RSIGAME_SFT_WARMUP_RATIO', 0.03)))
    ap.add_argument('--max-grad-norm', type=float, default=float(env('RSIGAME_SFT_MAX_GRAD_NORM', 1.0)))
    ap.add_argument('--scheduler', default=env('RSIGAME_SFT_SCHEDULER', 'cosine'))
    ap.add_argument('--max-length', type=int, default=int(env('RSIGAME_SFT_MAX_LENGTH', 32768)))
    ap.add_argument('--save-steps', type=int, default=int(env('RSIGAME_SFT_SAVE_STEPS', 25)))
    ap.add_argument('--save-limit', type=int, default=int(env('RSIGAME_SFT_SAVE_LIMIT', 6)))
    ap.add_argument('--sequence-parallel-size', type=int,
                    default=int(env('RSIGAME_SFT_SEQUENCE_PARALLEL_SIZE', 4)),
                    help='forced by the KV-head count; see the module docstring')
    ap.add_argument('--loss-scale', default=env('RSIGAME_SFT_LOSS_SCALE', 'default'),
                    help="'default' supervises every assistant turn; 'last_round' only the target")
    ap.add_argument('--agent-template', default=env('RSIGAME_SFT_AGENT_TEMPLATE', 'hermes'),
                    help='how the tools column is rendered; must match how the corpus was built')
    a = ap.parse_args(argv)
    if not a.model:
        ap.error('no base model: pass --model or set RSIGAME_SFT_BASE_MODEL')
    if not Path(a.data).is_file():
        ap.error(f'no training data at {a.data}')

    a.out = Path(a.out)
    a.out.mkdir(parents=True, exist_ok=True)

    resume = checkpoints.resume_arg(Path(a.remote), a.out) if a.remote else []
    cmd = build_command(a, resume)

    env2 = dict(os.environ)
    site = write_sitecustomize(a.out / '_site')
    env2['PYTHONPATH'] = os.pathsep.join(
        [str(site.parent)] + ([env2['PYTHONPATH']] if env2.get('PYTHONPATH') else []))
    env2.setdefault('TOKENIZERS_PARALLELISM', 'false')
    env2.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    # A rank that dies must take the job down rather than leave the other cards
    # spinning at 0%: a crashed rank holding three idle cards is worse than a
    # failed job, because nothing reports it.
    env2.setdefault('TORCHELASTIC_ERROR_FILE', str(a.out / 'torchelastic_err.json'))

    print('training command:\n  ' + ' '.join(cmd) + '\n')
    if a.dry_run:
        return 0

    stop = threading.Event()
    if a.remote:
        def syncer() -> None:
            while not stop.wait(a.sync_interval):
                try:
                    checkpoints.sync_once(a.out, Path(a.remote))
                except Exception as e:              # noqa: BLE001 - never kill the run
                    print(f'  sync error: {e}', flush=True)

        threading.Thread(target=syncer, daemon=True).start()

    rc = subprocess.call(cmd, env=env2)
    stop.set()
    print(f'\ntraining exited rc={rc}')
    if a.remote:
        print('final sync:')
        checkpoints.sync_once(a.out, Path(a.remote))
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
