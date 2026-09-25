# training

The corpus goes in, a LoRA adapter comes out, and the adapter is merged into
something that can be served.

```bash
# one run
python -m rsigame.training sft --data s1s2s3.jsonl --out /scratch/s1s2s3

# or, from a scheduler that can only run a shell command
ARM=s1s2s3 DATA=s1s2s3.jsonl OUT=/scratch/s1s2s3 bash src/rsigame/training/launch.sh

# afterwards
python -m rsigame.training adapters remap  --src /scratch/s1s2s3/checkpoint-3000 --dst remapped
python -m rsigame.training adapters merge  --adapter remapped --out merged
python -m rsigame.training adapters repair --merged merged
python -m rsigame.training metrics --run /durable/s1s2s3 --out loss.csv
```

## What one run is

LoRA (rank 16, alpha 32, rsLoRA) on every linear module of a 27B base, 4-bit
NF4 resident weights with bf16 compute, batch size 1, one epoch, cosine
schedule, 32,768-token context. Three arms — `s1` generation only, `s1s2`
+ plan, `s1s2s3` + repair — sharing the same generation rows by construction.

## The settings that are not defaults

| setting | why |
|---|---|
| `sequence_parallel_size = 4` | forced, not chosen: sp = gcd(kv_heads, world_size). With 4 KV heads, four cards give pure Ulysses; eight need ring attention, which is unimplemented for this model's linear-attention blocks and refuses. |
| 4-bit NF4 base | drops resident weights ~49 → ~14 GiB per rank, which is what makes long agent sessions fit at all |
| DeepSpeed off | with a 4-bit resident base there is nothing left to shard |
| `dataset_shuffle = false` | the corpus is ordered deliberately — see `data_pipeline.mix` |
| `truncation_strategy = delete` | a row that does not fit is dropped, never cut: half a tool-call batch is not a state the agent ever occupies |
| `attn_impl = sdpa`, cuDNN backend off in-process | every run died in cuDNN's fused attention, and the documented environment switch is ignored by this torch build |

## The loss_scale finding

With per-decision rows, `--loss_scale last_round` supervises only the final
assistant turn: a median of **114 tokens per step**. A run at that setting
produced a loss with **no trend at all** — corr(loss, step) ≈ 0 over 350 steps,
where whole-session runs reached −0.35. Supervising every assistant turn
(`default`) together with a learning rate of 1e-4 restored it. `default` is the
default here for that reason.

`last_round` stays available because it is the right answer if the rows are
rebuilt so that history is context rather than target — which is what
`data_pipeline.decisions --history-as-context` produces.

## Checkpoints

Checkpoints are written to node-local scratch, because that is the only storage
fast enough not to dominate the step time — and node-local scratch dies with
the job. A background thread copies each new checkpoint to `sft.checkpoint_dir`
and writes a `.complete` marker *after* the copy returns; a resume picks the
newest checkpoint that has one. A partial copy left by a kill mid-sync has no
marker and is never resumed from: it looks like a checkpoint and loads as
garbage.

They also nest one level deeper than you expect — `<out>/<version>/checkpoint-N`
— which is why a one-level glob once synced nothing for an entire run.

## Merging, and why it is verified rather than trusted

The adapter targets this model's linear-attention projections as well as the
usual q/k/v/o and MLP, and a serving stack that supports only the standard
projections may skip the rest *silently*. So the adapter is merged, not served
live, and the merge is proven: 496 (lora_A, lora_B) pairs are 496 modules that
must move, every one is fingerprinted before and after, and a merge where any
module comes out bit-identical to the base is not saved.

Two further traps the three steps exist to close: training nests the decoder
under `language_model` while serving flattens it, so a naive merge matches
**0 of 496** modules (`remap`); and saving through the causal-LM class returns
only the text tower, dropping **348 of 1,199** tensors and, once, shipping an
empty chat template (`repair`).

## What is deliberately not here

Anything that talks to one particular cluster: queue names, image registries,
node babysitters, out-of-band job watchers. A run is started by `launch.sh` on
whatever machine has the cards, and everything it needs is a flag or a config
key under `[sft]`.
