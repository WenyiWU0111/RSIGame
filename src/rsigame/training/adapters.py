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
"""Turn a trained adapter into something that can be served, and PROVE it applied.

Three steps, in this order, each of which exists because skipping it produced a
model that looked fine and was not:

  remap   Training and serving disagree about where the decoder lives. The
          training-time class nests it under `language_model`; the serving
          class flattens it:

              adapter : base_model.model.model.language_model.layers.0.linear_attn.in_proj_a
              base    :                        model.layers.0.linear_attn.in_proj_a

          Every target exists -- one path segment differs. peft matched 0 of
          496 modules for that reason alone. Rewrites the tensor keys AND the
          `target_modules` pattern, then re-derives the module census so the
          remap is verified rather than assumed.

  merge   Merge the LoRA into the base rather than serving it live: the
          adapter targets this model's linear-attention projections
          (in_proj_a / in_proj_b / in_proj_qkv / in_proj_z) as well as the
          usual q/k/v/o and MLP, and a serving stack that supports only the
          standard projections may SKIP the rest SILENTLY -- which serves a
          partially applied adapter that looks fine. Merging removes the
          question. "peft did not raise" is not evidence, so every targeted
          weight is fingerprinted before and after: 496 (lora_A, lora_B) pairs
          are 496 modules that MUST move, and a module bit-identical to the
          base afterwards did not get its delta. A merge that fails this is
          not saved.

  repair  Saving through the causal-LM class returns only the TEXT tower: 348
          of 1,199 tensors (333 visual, 15 MTP) vanish. They are not LoRA
          territory, so they are lifted from the base verbatim -- while
          anything missing under the decoder WOULD be LoRA territory and means
          the merge itself failed, so that refuses. This step also force-syncs
          the tokenizer, because one merge shipped an EMPTY chat template (0
          characters against the base's 8,952), which changes how the arm is
          prompted with no error anywhere.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys
import time
from pathlib import Path


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def module_census(keys) -> dict[str, int]:
    """LoRA tensor keys -> how many modules of each leaf kind they cover."""
    mods = set()
    for k in keys:
        if '.lora_A.' in k:
            mods.add(k.split('.lora_A.')[0])
        elif '.lora_B.' in k:
            mods.add(k.split('.lora_B.')[0])
    kinds: collections.Counter = collections.Counter(m.split('.')[-1] for m in mods)
    return {'modules': len(mods), **dict(sorted(kinds.items()))}


# --------------------------------------------------------------------------- remap
def remap(src: Path, dst: Path, expect: int = 0) -> int:
    from safetensors import safe_open
    from safetensors.torch import save_file

    dst.mkdir(parents=True, exist_ok=True)
    tensors: dict = {}
    with safe_open(src / 'adapter_model.safetensors', framework='pt') as f:
        meta = f.metadata() or {}
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    before = module_census(tensors)
    renamed: dict = {}
    n_changed = 0
    for k, v in tensors.items():
        nk = k.replace('.model.language_model.', '.model.')
        n_changed += nk != k
        renamed[nk] = v
    log(f'tensors {len(tensors)}, keys rewritten {n_changed}')
    save_file(renamed, str(dst / 'adapter_model.safetensors'), metadata=meta)

    cfg = json.loads((src / 'adapter_config.json').read_text())
    old_tm = cfg.get('target_modules')
    if isinstance(old_tm, str):
        cfg['target_modules'] = old_tm.replace('model\\.language_model', 'model')
    log(f"target_modules:\n  before {old_tm}\n  after  {cfg.get('target_modules')}")
    (dst / 'adapter_config.json').write_text(json.dumps(cfg, indent=1))

    after = module_census(renamed)
    left = [k for k in renamed if 'language_model' in k]
    log(f'module census before {before}')
    log(f'module census after  {after}')
    log(f'still containing language_model: {len(left)} (expect 0)')
    ok = after == before and not left and (not expect or after['modules'] == expect)
    log('REMAP ' + ('OK' if ok else 'FAILED'))
    return 0 if ok else 2


# --------------------------------------------------------------------------- merge
def merge(base: Path, adapter: Path, out: Path, report: Path | None = None) -> int:
    import torch
    from peft import PeftModel
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with safe_open(adapter / 'adapter_model.safetensors', framework='pt') as f:
        keys = list(f.keys())
    census = module_census(keys)
    pairs = set()
    for k in keys:
        if '.lora_A.' in k:
            pairs.add(k.split('.lora_A.')[0])
        elif '.lora_B.' in k:
            pairs.add(k.split('.lora_B.')[0])
    log(f"adapter tensors {len(keys)} -> {census['modules']} target modules")

    # base_model.model.model.layers... -> model.layers...
    targets = {p.replace('base_model.model.', '') for p in pairs}

    log(f'loading base {base}')
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map='cpu', trust_remote_code=True)

    # ---- fingerprint every target BEFORE the merge ------------------------
    sd = model.state_dict()
    fingerprint: dict[str, float] = {}
    missing: list[str] = []
    for t in sorted(targets):
        w = sd.get(t + '.weight')
        if w is None:
            missing.append(t)
            continue
        fingerprint[t] = w.detach().to(torch.float32).sum().item()
    log(f'fingerprinted {len(fingerprint)}/{len(targets)} target weights')
    if missing:
        log(f'WARNING {len(missing)} targets not in the base state_dict, e.g. {missing[:3]}')

    log(f'applying adapter {adapter}')
    model = PeftModel.from_pretrained(model, adapter, torch_dtype=torch.bfloat16)
    log('merging')
    model = model.merge_and_unload()

    # ---- verify every targeted weight actually moved ----------------------
    sd2 = model.state_dict()
    moved, still = 0, []
    for t, v in fingerprint.items():
        w = sd2.get(t + '.weight')
        if w is None:
            still.append(t + ' (gone)')
            continue
        if abs(w.detach().to(torch.float32).sum().item() - v) > 1e-6:
            moved += 1
        else:
            still.append(t)

    log(f'MODULES MOVED: {moved} / {len(fingerprint)}')
    if still:
        log(f'UNCHANGED ({len(still)}) e.g. {still[:5]}')
    ok = moved == len(fingerprint) and len(fingerprint) == len(targets)
    log('SANITY: ' + ('PASS -- every targeted module changed'
                      if ok else 'FAIL -- this merge is NOT trustworthy'))
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(
            {'adapter_tensors': len(keys), 'target_modules': len(targets),
             'fingerprinted': len(fingerprint), 'moved': moved,
             'unchanged': len(still), 'pass': bool(ok)}, indent=1))
    if not ok:
        log('refusing to save a merge that failed verification')
        return 2

    log(f'saving to {out}')
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base, trust_remote_code=True).save_pretrained(out)
    log('saved.')
    return 0


# --------------------------------------------------------------------------- repair
def _template_chars(d: Path) -> int:
    t = json.loads((d / 'tokenizer_config.json').read_text()).get('chat_template') or ''
    jinja = d / 'chat_template.jinja'
    if not t and jinja.exists():
        t = jinja.read_text()
    return len(t)


def repair(base: Path, merged: Path) -> int:
    from safetensors import safe_open
    from safetensors.torch import save_file

    bmap = json.loads((base / 'model.safetensors.index.json').read_text())['weight_map']
    midx = json.loads((merged / 'model.safetensors.index.json').read_text())
    mmap = midx['weight_map']

    missing = sorted(k for k in bmap if k not in mmap)
    log(f'missing from merged: {len(missing)}')
    log('prefixes: ' + str(dict(collections.Counter(
        '.'.join(k.split('.')[:2]) for k in missing))))

    unsafe = [k for k in missing if k.startswith('model.language_model.')]
    if unsafe:
        log(f'REFUSING: {len(unsafe)} missing keys are under the decoder -- '
            'that is LoRA territory, so the merge itself failed')
        return 2

    by_shard: dict[str, list[str]] = collections.defaultdict(list)
    for k in missing:
        by_shard[bmap[k]].append(k)
    tensors: dict = {}
    nbytes = 0
    for shard, keys in sorted(by_shard.items()):
        with safe_open(base / shard, framework='pt') as h:
            for k in keys:
                t = h.get_tensor(k)
                tensors[k] = t
                nbytes += t.numel() * t.element_size()
    log(f'loaded {len(tensors)} tensors, {nbytes / 1024 ** 3:.2f} GiB')

    if tensors:
        save_file(tensors, str(merged / 'model-restored.safetensors'),
                  metadata={'format': 'pt'})
        for k in missing:
            mmap[k] = 'model-restored.safetensors'
        md = midx.setdefault('metadata', {})
        md['total_size'] = md.get('total_size', 0) + nbytes
        (merged / 'model.safetensors.index.json').write_text(json.dumps(midx, indent=2))

    shutil.copy(base / 'config.json', merged / 'config.json')
    for f in os.listdir(base):
        if f.startswith(('tokenizer', 'vocab', 'merges', 'chat_template', 'preprocessor',
                         'generation_config', 'special_tokens')) or f.endswith('.jinja'):
            shutil.copy(base / f, merged / f)

    bkeys = set(json.loads((base / 'model.safetensors.index.json').read_text())['weight_map'])
    mkeys = set(json.loads((merged / 'model.safetensors.index.json').read_text())['weight_map'])
    lb, lm = _template_chars(base), _template_chars(merged)
    bad = [f for f in set(mmap.values()) if not (merged / f).exists()]
    log(f'keys base={len(bkeys)} merged={len(mkeys)} identical={bkeys == mkeys}')
    log(f'chat_template chars: base={lb} merged={lm} match={lb == lm}')
    log(f"missing shard files: {bad or 'none'}")
    ok = bkeys == mkeys and lb == lm and lb > 0 and not bad
    log('VERIFY: ' + ('OK' if ok else 'FAIL'))
    return 0 if ok else 3


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog='rsigame.training adapters', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='step', required=True)

    r = sub.add_parser('remap', help="rewrite the adapter to the serving stack's naming")
    r.add_argument('--src', type=Path, required=True)
    r.add_argument('--dst', type=Path, required=True)
    r.add_argument('--expect', type=int, default=0, help='assert this module count (0 = just compare)')

    m = sub.add_parser('merge', help='merge the adapter into the base, verifying every module')
    m.add_argument('--base', type=Path, default=os.environ.get('RSIGAME_SFT_BASE_MODEL'))
    m.add_argument('--adapter', type=Path, required=True)
    m.add_argument('--out', type=Path, required=True)
    m.add_argument('--report', type=Path, default=None)

    p = sub.add_parser('repair', help='restore the towers the save dropped, and the tokenizer')
    p.add_argument('--base', type=Path, default=os.environ.get('RSIGAME_SFT_BASE_MODEL'))
    p.add_argument('--merged', type=Path, required=True)

    a = ap.parse_args(argv)
    if a.step == 'remap':
        return remap(a.src, a.dst, a.expect)
    if not a.base:
        ap.error('no base model: pass --base or set RSIGAME_SFT_BASE_MODEL')
    if a.step == 'merge':
        return merge(Path(a.base), a.adapter, a.out, a.report)
    return repair(Path(a.base), a.merged)


if __name__ == '__main__':
    sys.exit(main())
