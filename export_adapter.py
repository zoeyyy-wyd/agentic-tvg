#!/usr/bin/env python
"""Merge the SFT LoRA into the base weights, producing a plain HF model.

verl saves the whole model in one fp32 state_dict -- base + adapter, 19G for
Qwen3-VL-4B -- with the trained layers wrapped PEFT-style:

    ...q_proj.base_layer.weight        (4096, 2560)   frozen
    ...q_proj.lora_A.default.weight    (16, 2560)     trained
    ...q_proj.lora_B.default.weight    (4096, 16)     trained

There is no adapter directory in the checkpoint (`huggingface/` holds configs
and the tokenizer, no weights), so this rebuilds the PEFT wrapper in memory,
loads those tensors into it, and lets PEFT do the merge. GRPO then starts from
the result with nothing but an env var:

    MODEL_PATH=results/sft-mix/merged bash run_grpo.sh

Why PEFT rather than folding `W += (alpha/r) * B @ A` by hand: that formula is
only correct for plain LoRA. rsLoRA scales by alpha/sqrt(r), DoRA carries an
extra magnitude vector -- and a hand-rolled merge would apply the wrong one
*silently*, producing a model that loads, runs, and is subtly wrong. Delegating
to `merge_and_unload` makes the variant PEFT's problem, not ours.

Merging rather than passing `lora_adapter_path` to verl keeps the two stages
separable: SFT knowledge becomes part of the frozen base, so the LoRA GRPO
trains is purely what RL added. It also avoids verl's LoRA-RL + vLLM
weight-sync path, which this repo has never exercised.

Usage:
    python export_adapter.py                         # -> results/sft-mix/merged
    python export_adapter.py --ckpt DIR --out DIR
    python export_adapter.py --no-verify             # skip the logits checks
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import AutoModelForImageTextToText

# PEFT stores the adapter *name* in checkpoint keys and strips it on save, so
# the state dict it wants back is the one without `.default`.
ADAPTER_NAME_RE = re.compile(r"\.default(?=\.weight$)")
PEFT_PREFIX = "base_model.model."
BASE_LAYER = ".base_layer"

# Everything the hand-rolled merge could not have handled correctly. If a run
# ever trains one of these, stop rather than merge with the wrong formula --
# PEFT handles them, but only if the LoraConfig we rebuild says so.
LORA_VARIANTS = ("use_rslora", "use_dora", "lora_bias", "rank_pattern", "alpha_pattern")


def newest_step(root: Path) -> Path:
    """The global_step_N with the largest N -- not the newest mtime, which a
    resumed run rewrites on directories it did not produce."""
    steps = [(int(p.name.split("_")[-1]), p) for p in root.glob("global_step_*") if p.is_dir()]
    if not steps:
        raise SystemExit(f"no global_step_* under {root}")
    return max(steps)[1]


def check_base_untouched(sd: dict, model, sample: int = 8) -> None:
    """The frozen half of the checkpoint must equal the shipped base weights.

    Merging into the *installed* base assumes it is the one training ran
    against; this is what turns that assumption into a checked fact. It also
    confirms exclude_modules really did keep the ViT frozen.
    """
    live = dict(model.named_parameters())
    checked = 0
    for key, value in sd.items():
        if "lora_" in key or BASE_LAYER in key or "visual" not in key:
            continue
        name = key.removeprefix(PEFT_PREFIX)
        if name not in live:
            raise SystemExit(f"{name} is not a parameter of the base model -- key layout changed")
        if not torch.equal(value.to(live[name].dtype), live[name].detach()):
            raise SystemExit(f"{name} differs from the shipped base -- wrong base model?")
        checked += 1
        if checked == sample:
            break
    if not checked:
        raise SystemExit("no frozen visual tensors found -- key layout changed")
    print(f"  [check] {checked} frozen ViT tensors identical to the shipped base")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=Path("results/sft-mix/ckpt"),
                    help="a global_step_N dir, or the parent holding several")
    ap.add_argument("--out", type=Path, default=None, help="default: <ckpt parent>/merged")
    ap.add_argument("--base", type=Path, default=Path("models/Qwen3-VL-4B-Instruct"))
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    step = args.ckpt if args.ckpt.name.startswith("global_step_") else newest_step(args.ckpt)
    out = args.out or step.parent.parent / "merged"
    meta = json.loads((step / "lora_train_meta.json").read_text())
    unsupported = [k for k in LORA_VARIANTS if meta.get(k)]
    if unsupported:
        raise SystemExit(f"checkpoint trained with {unsupported}; rebuild LoraConfig to match "
                         "before merging -- the defaults below would be wrong")
    print(f"{step} (r={meta['r']}, alpha={meta['lora_alpha']}) -> {out}")

    sd = torch.load(step / "model_world_size_1_rank_0.pt",
                    map_location="cpu", mmap=True, weights_only=True)
    adapter = {ADAPTER_NAME_RE.sub("", k): v.clone().float()
               for k, v in sd.items() if "lora_" in k}
    # The resolved module list, read off the keys -- not the `all-linear` spec
    # training used, which re-resolves differently against a different base.
    targets = sorted({k.split(".lora_")[0].split(".")[-1] for k in adapter})
    print(f"  {len(adapter)} adapter tensors | targets={targets}")

    # fp32 for the merge: B@A is a rank-16 product and bf16's 8-bit mantissa
    # loses real precision folding it in. Cast once, at save time.
    print(f"  loading {args.base} (fp32) ...")
    model = AutoModelForImageTextToText.from_pretrained(args.base, dtype=torch.float32)
    model.eval()
    check_base_untouched(sd, model)

    ids = torch.arange(1, 17).unsqueeze(0)
    with torch.no_grad():
        base_logits = None if args.no_verify else model(input_ids=ids).logits.clone()

    peft_model = get_peft_model(model, LoraConfig(
        r=meta["r"], lora_alpha=meta["lora_alpha"], target_modules=targets,
        task_type=meta["task_type"], lora_dropout=0.0, bias="none"))
    result = set_peft_model_state_dict(peft_model, adapter)
    if getattr(result, "unexpected_keys", None):
        raise SystemExit(f"{len(result.unexpected_keys)} keys matched nothing, "
                         f"e.g. {result.unexpected_keys[0]}")
    print(f"  [check] set_peft_model_state_dict accepted all {len(adapter)} tensors")

    peft_model.eval()
    with torch.no_grad():
        # With the adapter attached but not yet folded in.
        wrapped_logits = None if args.no_verify else peft_model(input_ids=ids).logits.clone()

    merged = peft_model.merge_and_unload()
    merged.eval()

    if not args.no_verify:
        with torch.no_grad():
            merged_logits = merged(input_ids=ids).logits
        moved = (wrapped_logits - base_logits).abs().max().item()
        print(f"  [check] adapter changes the base by {moved:.4g}")
        if moved == 0.0:
            raise SystemExit("adapter has no effect -- it was not applied")
        # The merge must be arithmetically transparent: folding W += BA into the
        # weights has to reproduce what the wrapped model computed.
        drift = (merged_logits - wrapped_logits).abs().max().item()
        print(f"  [check] merged vs wrapped drift = {drift:.4g}")
        if drift > 1e-2:
            raise SystemExit(f"merge changed the output by {drift:.4g} -- not a faithful fold")

    out.mkdir(parents=True, exist_ok=True)
    # save_pretrained writes weights + config only; vLLM also needs the
    # tokenizer, preprocessor and video-preprocessor configs. Copy first so the
    # freshly written config.json wins.
    for f in sorted(args.base.iterdir()):
        if f.is_file() and f.suffix != ".safetensors" and f.name != "model.safetensors.index.json":
            shutil.copy2(f, out / f.name)
    merged.to(torch.bfloat16).save_pretrained(out)
    total = sum(f.stat().st_size for f in out.iterdir() if f.is_file())
    print(f"  wrote {out} ({total / 2**30:.1f} GiB)")


if __name__ == "__main__":
    main()
