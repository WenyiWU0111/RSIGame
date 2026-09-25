gamecraft-bench_replay_wait_for_draw.patch
  Original single fix, against commit a31555b93fefd306a956c40d518e5c4e1eb49cfd (branch
  judge-local-vllm-and-repeats of our internal fork of gamecraft-bench).

For a fresh clone of upstream FreedomIntelligence/gamecraft-bench (main, a43347534374df9a0c1a6c001aa9380862783f6d), apply
these two INSTEAD, in this order:
  gamecraft-bench_evogame_scoring.patch          judge run locally or hosted, averaged passes, temperature 0,
                                                 replay waits for the first drawn frame, unreapable-godot and
                                                 ffmpeg-start fixes (includes replay_wait_for_draw)
  gamecraft-bench_openrouter_provider_pin.patch  GAMECRAFT_BENCH_JUDGE_PROVIDER pins an OpenRouter provider with
                                                 no fallback and turns thinking off; no effect when unset
Both were checked with git apply on a clean export of that main commit.
