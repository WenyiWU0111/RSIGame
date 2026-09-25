# data_pipeline

Recorded agent sessions in, a supervised fine-tuning corpus out. This is the
half of recursive self-improvement that closes the loop: the model is trained
on sessions the development loop itself produced.

## The order the steps run in

```
  web_trajectories ─┐
                    ├─→ decisions ─→ split ─→ lengths ─┐
  godot_trajectories┘                                   │
                                                        ├─→ mix ─→ validate
  plans_extract ─┐                                      │
                 ├─→ plans_merge ──────────────────────┤
  plans_distil ──┘                                      │
                                                        │
  repair_records ───────────────────────────────────────┘
```

Or, from the published corpus shards rather than from raw recordings,
`arms` does the whole right-hand side in one step.

| step | what it does |
|---|---|
| `web_trajectories` | a web recording's event stream → one row per session |
| `godot_trajectories` | a Godot `trajectory.json` → one row per task |
| `decisions` | whole sessions → one row per agent decision |
| `plans_extract` | the plan a web agent wrote before coding (147/150 sessions have one) |
| `plans_distil` | Godot plans, which must be distilled because 0/150 sessions contain one |
| `plans_merge` | both halves → one plan corpus, under one byte-identical system prompt |
| `repair_records` | self-refine records → repair rows, gated on an independent verdict |
| `patches` | shrink a full-tree repair diff to the hunks that are the fix (55×) |
| `split` | train / holdout, on brief stems, with the scored briefs excluded outright |
| `lengths` | count tokens with the real tokenizer, drop what will not fit |
| `filters` | the quality flags, and how to read a big corpus without running out of memory |
| `mix` | one arm's file, balanced by optimizer steps |
| `arms` | the three ablation arms, straight from the published shards |
| `validate` | refuse to ship a corpus that would train badly |

Every step is a module with a `--help`, and the CLI runs them by name:

```bash
python -m rsigame.data_pipeline web    --roots /recordings --out gen_web.jsonl
python -m rsigame.data_pipeline arms   --arm s1s2s3 --out s1s2s3.jsonl
python -m rsigame.data_pipeline validate --inp s1s2s3.jsonl
```

## The two decisions worth knowing before reading any of it

**Stage shares are balanced by optimizer steps, not tokens.** Training runs at
batch size 1, so every row is exactly one step however long it is: row share
*is* step share. Ignoring that once gave plan+repair 59.4% of stage 3's updates
while they carried 4.6% of its tokens — a majority of training spent on
single-turn, zero-tool data when the evaluation is a ~110-turn agentic loop.

**Generation stays dominant on purpose.** Stage 2 and stage 3 enter as a capped
share of the stage-1 budget (15% and 20% by default), not as everything
available. The arms then differ only in what was *added*, so a drop cannot be
explained by one arm having seen less generation data. There is far more plan
and repair data than that cap; pouring it in pushes one-shot generation down,
which is the metric the system is judged on.

## Configuration

Three keys, all under `[corpus]` in `configs/default.toml`, and all only needed
when rebuilding a corpus:

| key | what it is |
|---|---|
| `web_roots` | comma-separated directories holding web recordings |
| `dataset_dir` | the downloaded corpus shards, for `arms` and `filters` |
| `plan_teacher` | the model `plans_distil` asks for Godot plans |

`lengths` and `validate` also read `sft.base_model`, because the tokenizer that
decides whether a row fits is the model's own.
