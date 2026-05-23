"""LoRA SFT training for one of the 3 candidate models.

Reads `data/augment/train.jsonl` (filter on metadata.split=='train') and
`data/augment/validation.jsonl`. Trains with TRL SFTTrainer + PEFT LoRA in
bf16 on a single L40S GPU.

Usage:
  CUDA_VISIBLE_DEVICES=0 python 5_sft_training/train.py --model qwen3-1.7b
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    AUGMENT_DIR,
    CHECKPOINT_DIR,
    MODELS,
    load_hf_messages_dataset,
    setup_logging,
)


def main() -> int:
    log = setup_logging("triagellm.sft.train")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--logging-steps", type=int, default=20)
    args = ap.parse_args()

    spec = MODELS[args.model]
    out_dir = CHECKPOINT_DIR / spec.slug
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("model=%s out=%s", spec.slug, out_dir)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_id, trust_remote_code=spec.needs_trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="flash_attention_2",
            trust_remote_code=spec.needs_trust_remote_code,
        )
        log.info("loaded with flash_attention_2")
    except (ImportError, ValueError) as e:
        log.warning("flash_attention_2 unavailable (%s); falling back to sdpa", e)
        model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="sdpa",
            trust_remote_code=spec.needs_trust_remote_code,
        )

    log.info("loading datasets")
    train_ds = load_hf_messages_dataset(AUGMENT_DIR / "train.jsonl")
    val_ds = load_hf_messages_dataset(AUGMENT_DIR / "validation.jsonl")
    log.info("train=%d val=%d", len(train_ds), len(val_ds))

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=spec.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=spec.per_device_batch,
        per_device_eval_batch_size=spec.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_length=args.max_seq_length,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=42,
        dataloader_num_workers=2,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_cfg,
        processing_class=tokenizer,
    )

    log.info("starting training")
    trainer.train()
    log.info("training complete; saving adapter")
    trainer.save_model(str(out_dir / "adapter"))
    tokenizer.save_pretrained(str(out_dir / "adapter"))
    log.info("done. checkpoint at %s", out_dir / "adapter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
