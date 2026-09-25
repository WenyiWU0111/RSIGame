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
"""Count each row's tokens with the REAL tokenizer, and filter to a context.

A chars/3.6 rule of thumb is the usual estimate. Measured on this model's
tokenizer, code runs about 2.70 chars per token, so that rule is ~33%
optimistic -- which is the difference between a corpus that fits the context
and one that silently loses its longest rows to truncation. This counts for
real, applies the model's own chat template (so template and tool-call
scaffolding are included), stamps `n_tokens` on every surviving row, and writes
a manifest of exactly what trained.

It streams one row at a time: the corpus is ~78 MB of JSON and materialising it
whole has been enough to hit a memory cap on a shared machine.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def default_model() -> str:
    return os.environ.get('RSIGAME_SFT_BASE_MODEL', '')


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog='rsigame.data_pipeline lengths', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--inp', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--model', default=default_model(),
                    help='tokenizer to count with (RSIGAME_SFT_BASE_MODEL)')
    ap.add_argument('--max-tokens', type=int, default=32768)
    a = ap.parse_args(argv)
    if not a.model:
        ap.error('no tokenizer: pass --model or set RSIGAME_SFT_BASE_MODEL')

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)

    def count(msgs: list[dict]) -> tuple[int, str]:
        """Prefer the model's chat template; fall back to raw concatenation.

        The fallback is not equivalent -- it misses the template's own tokens --
        so which one ran is reported rather than hidden.
        """
        try:
            ids = tk.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
            return len(ids), 'chat_template'
        except Exception:                      # noqa: BLE001 - any template error falls back
            buf = []
            for m in msgs:
                buf.append(str(m.get('content') or ''))
                for c in (m.get('tool_calls') or []):
                    fn = c.get('function') or {}
                    buf.append(str(fn.get('name', '')))
                    buf.append(str(fn.get('arguments', '')))
            return len(tk('\n'.join(buf)).input_ids), 'concat_fallback'

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.manifest.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    kept = dropped = 0
    modes: set[str] = set()
    with a.inp.open() as fin, a.out.open('w') as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            n, mode = count(s['messages'])
            modes.add(mode)
            ok = n <= a.max_tokens
            rows.append({'task': s.get('task'), 'engine': s.get('engine'),
                         'model': s.get('model'), 'tokens': n, 'chars': len(line),
                         'n_tool_calls': s.get('n_tool_calls'), 'kept': ok})
            if ok:
                s['n_tokens'] = n
                fout.write(json.dumps(s, ensure_ascii=False) + '\n')
                kept += 1
            else:
                dropped += 1
            if (i + 1) % 50 == 0:
                print(f'  ... {i + 1} scanned ({kept} kept)', flush=True)

    a.manifest.write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    toks = sorted(r['tokens'] for r in rows)

    def q(p: float) -> int:
        return toks[min(len(toks) - 1, int(p * len(toks)))] if toks else 0

    cpt = sum(r['chars'] for r in rows) / max(1, sum(r['tokens'] for r in rows))
    print()
    print(f"tokenizer mode       {', '.join(sorted(modes))}")
    print(f'rows in              {len(rows)}')
    print(f'kept <= {a.max_tokens:<12} {kept}')
    print(f'dropped              {dropped}')
    print(f'measured chars/token {cpt:.2f}   (the chars/3.6 rule of thumb is optimistic)')
    print(f'tokens  p50 {q(.5)}  p90 {q(.9)}  p99 {q(.99)}  max {q(1.0)}')
    print()
    print('fit by context:')
    for ctx in (32768, 65536, 131072, 262144):
        n = sum(1 for t in toks if t <= ctx)
        print(f'  {ctx:7d}  {n:5d} / {len(toks)}  ({100 * n / max(1, len(toks)):.0f}%)')
    print()
    print(f"engine spread (kept): {dict(Counter(r['engine'] for r in rows if r['kept']))}")
    print(f'manifest -> {a.manifest}')
    return 0 if kept else 1


if __name__ == '__main__':
    raise SystemExit(main())
