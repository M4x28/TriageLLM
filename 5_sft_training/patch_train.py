"""Targeted LoRA patch on the merged 8B SFT checkpoint.

Fixes the HOME CARE branch framework-leak: adds a LoRA adapter on top of the
already-merged checkpoint, trains for 1-2 epochs on a small patch dataset, then
merges to a new checkpoint. Uses a lower LR than the original run to avoid
catastrophic forgetting.

Usage:
  CUDA_VISIBLE_DEVICES=0 python 5_sft_training/patch_train.py \
    --base data/sft/checkpoints/qwen3-8b/merged \
    --data data/augment/patch_homecare.jsonl \
    --out  data/sft/checkpoints/qwen3-8b/merged_patch \
    --epochs 2 \
    --learning-rate 5e-5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load_jsonl_messages(path: Path):
    from datasets import Dataset
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append({"messages": json.loads(line)["messages"]})
    return Dataset.from_list(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(REPO / "data/sft/checkpoints/qwen3-8b/merged"))
    ap.add_argument("--data", default=str(REPO / "data/augment/patch_homecare.jsonl"))
    ap.add_argument("--out",  default=str(REPO / "data/sft/checkpoints/qwen3-8b/merged_patch"))
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--learning-rate", type=float, default=5e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTConfig, SFTTrainer

    print(f"loading base from {args.base}")
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.base, torch_dtype=torch.bfloat16,
            device_map="cuda", attn_implementation="flash_attention_2",
        )
        print("loaded with flash_attention_2")
    except (ImportError, ValueError) as e:
        print(f"flash_attention_2 unavailable ({e}); using sdpa")
        model = AutoModelForCausalLM.from_pretrained(
            args.base, torch_dtype=torch.bfloat16,
            device_map="cuda", attn_implementation="sdpa",
        )

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    ds = load_jsonl_messages(Path(args.data))
    print(f"patch dataset: {len(ds)} examples")

    adapter_dir = Path(args.out + "_adapter")
    adapter_dir.mkdir(parents=True, exist_ok=True)

    sft_cfg = SFTConfig(
        output_dir=str(adapter_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_length=2048,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=5,
        save_steps=200,
        save_total_limit=1,
        eval_strategy="no",
        report_to="none",
        seed=42,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds,
        peft_config=lora_cfg,
        processing_class=tok,
    )

    print("starting patch training")
    trainer.train()
    print("training done; merging adapter into base weights")

    merged = trainer.model.merge_and_unload()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True)
    tok.save_pretrained(str(out))
    print(f"merged checkpoint saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
