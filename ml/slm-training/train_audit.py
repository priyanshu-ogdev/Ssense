#!/usr/bin/env python3
"""
train_audit.py – Industrial-Grade SFT + SimPO Pipeline for the DPDP Forensic Auditor SLM (`r=128`, `beta=2.0`)

Architecture & MLOps Specifications:
- Memory & Optimizer: Hardcoded 32-bit FP32 `adamw_torch` (`weight_decay=0.05`) to leverage 128 GB VRAM headroom.
- LoRA Configuration: Rank-Stabilized LoRA (`rsLoRA`, `r=128`) with fused Triton kernels (`lora_dropout=0`).
- Tokenizer Alignment: Right-side truncation (`truncation_side="right"`) and right-side padding to preserve system preambles.
- Context & Truncation: Hardcoded `max_prompt_length=23500` to preserve extensive corporate privacy policies at index ~23,000.
- Process Isolation: OS-level `spawn` multi-processing to isolate CUDA contexts and eliminate memory leaks between SFT and DPO phases.
- Unified Adapter Continuity: Exports unmerged LoRA adapter after Phase 1 SFT, reloaded with `is_trainable=True` for reference-free Phase 2 SimPO.
"""

import os
# Prevent Rust multi-threaded tokenizer deadlocks across Python worker processes
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import torch
import multiprocessing
from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import SFTTrainer, DPOTrainer, SFTConfig, DPOConfig
from datasets import load_dataset
from unsloth.chat_templates import train_on_responses_only
from transformers import EarlyStoppingCallback

# Monkey-patch TRL's DPOTrainer with Unsloth memory optimizations
PatchDPOTrainer()

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
DEFAULT_BASE_MODEL = "../models/Qwen3.5-9B"
BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH", DEFAULT_BASE_MODEL if os.path.exists(DEFAULT_BASE_MODEL) else "Qwen/Qwen3.5-9B")
SFT_DATA_PATH = "./data/audit_sft_data.jsonl"
DPO_DATA_PATH = "./data/audit_dpo_data.jsonl"
OUTPUT_DIR_SFT = "../models/audit-model-sft-intermediate"
OUTPUT_DIR_SFT_ADAPTER = "../models/audit-model-sft-intermediate-adapter"
OUTPUT_DIR_FINAL = "../models/audit-model-final"

MAX_SEQ_LENGTH = 24576
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8
EPOCHS_SFT = 2
EPOCHS_DPO = 1

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: SUPERVISED FINE-TUNING (SFT)
# ═══════════════════════════════════════════════════════════════════════════
def run_sft():
    print("🚀 PHASE 1: Starting Forensic Auditor SFT (32-bit FP32 AdamW + rsLoRA r=128 + FlashAttn2)...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        use_flash_attention_2=True,
    )

    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=128,
        lora_alpha=128,
        use_rslora=True,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    dataset = load_dataset("json", data_files=SFT_DATA_PATH, split="train")
    
    def apply_chat_template(examples):
        texts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False) 
            for msgs in examples["messages"]
        ]
        return {"text": texts}

    dataset = dataset.map(apply_chat_template, batched=True, num_proc=8)
    split = dataset.train_test_split(test_size=0.05, seed=42)

    sft_args = SFTConfig(
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        warmup_ratio=0.05,
        num_train_epochs=EPOCHS_SFT,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        bf16=True,
        optim="adamw_torch",
        weight_decay=0.05,
        max_grad_norm=1.0,
        output_dir=OUTPUT_DIR_SFT,
        logging_steps=10,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
        remove_unused_columns=False,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        args=sft_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Mask user prompts so loss is strictly computed over assistant completions
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print("🏋️ Training SFT...")
    trainer.train()

    print("💾 Saving Phase 1 SFT Adapter for Phase 2 Unified Adapter Continuity...")
    model.save_pretrained_merged(OUTPUT_DIR_SFT_ADAPTER, tokenizer, save_method="lora")
    print("💾 Saving standalone 16-bit intermediate checkpoint to:", OUTPUT_DIR_SFT)
    model.save_pretrained_merged(OUTPUT_DIR_SFT, tokenizer, save_method="merged_16bit")
    
    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()
    print("✅ Phase 1 SFT Complete. Sub-process terminating to release CUDA resources.")

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: SIMPLE PREFERENCE OPTIMIZATION (SimPO)
# ═══════════════════════════════════════════════════════════════════════════
def run_dpo():
    print("🚀 PHASE 2: Starting Forensic Auditor SimPO (Length-Normalized Margin Gamma=0.5)...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=OUTPUT_DIR_SFT_ADAPTER,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        use_flash_attention_2=True,
        is_trainable=True,
    )

    FastLanguageModel.for_training(model, use_gradient_checkpointing="unsloth")

    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=DPO_DATA_PATH, split="train")
    
    def extract_turn_content(turn):
        """Extract exact text from chosen/rejected turn array or dict using index [-1]."""
        if isinstance(turn, list) and len(turn) > 0:
            if isinstance(turn[-1], dict) and "content" in turn[-1]:
                return turn[-1]["content"].strip()
        elif isinstance(turn, dict) and "content" in turn:
            return turn["content"].strip()
        elif isinstance(turn, str):
            return turn.strip()
        return str(turn).strip()

    def format_preference_dataset(examples):
        prompts, chosens, rejecteds = [], [], []
        for i in range(len(examples["prompt"])):
            prompt_str = tokenizer.apply_chat_template(
                examples["prompt"][i], 
                tokenize=False, 
                add_generation_prompt=True
            )
            chosen_str = extract_turn_content(examples["chosen"][i]) + "<|im_end|>\n"
            rejected_str = extract_turn_content(examples["rejected"][i]) + "<|im_end|>\n"
            
            prompts.append(prompt_str)
            chosens.append(chosen_str)
            rejecteds.append(rejected_str)
        return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}

    dataset = dataset.map(format_preference_dataset, batched=True, num_proc=8)
    split = dataset.train_test_split(test_size=0.05, seed=42)

    dpo_args = DPOConfig(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_ratio=0.05,
        num_train_epochs=EPOCHS_DPO,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        bf16=True,
        optim="adamw_torch",
        weight_decay=0.05,
        loss_type="simpo",
        beta=2.0,
        simpo_gamma=0.5,
        max_length=MAX_SEQ_LENGTH,
        max_prompt_length=23500,
        output_dir=OUTPUT_DIR_FINAL,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=42,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        args=dpo_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("🏋️ Training SimPO...")
    trainer.train()

    print("💾 Executing Dual-Deployment Export Hooks...")
    print("   -> Saving standalone 16-bit bfloat16 safetensors to:", OUTPUT_DIR_FINAL)
    model.save_pretrained_merged(OUTPUT_DIR_FINAL, tokenizer, save_method="merged_16bit")
    
    adapter_out = OUTPUT_DIR_FINAL + "-adapter"
    print("   -> Saving Unified LoRA adapter for Multi-LoRA serving to:", adapter_out)
    model.save_pretrained_merged(adapter_out, tokenizer, save_method="lora")
    
    print("\n✅ FORENSIC AUDITOR TRAINING COMPLETE.")
    print("🚀 vLLM Multi-LoRA Production Startup Command:")
    print(f"   vllm serve {BASE_MODEL_PATH} --enable-lora --max-loras 2 --max-lora-rank 128 --lora-modules audit={adapter_out}")

# ═══════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    if not os.path.exists(SFT_DATA_PATH): raise FileNotFoundError(f"Missing {SFT_DATA_PATH}")
    if not os.path.exists(DPO_DATA_PATH): raise FileNotFoundError(f"Missing {DPO_DATA_PATH}")

    print("[INIT] Launching Phase 1 SFT inside isolated OS sub-process...")
    p_sft = multiprocessing.Process(target=run_sft)
    p_sft.start()
    p_sft.join()
    if p_sft.exitcode != 0:
        raise RuntimeError(f"Phase 1 SFT terminated with non-zero exit code: {p_sft.exitcode}")

    print("[INIT] Launching Phase 2 SimPO inside isolated OS sub-process...")
    p_dpo = multiprocessing.Process(target=run_dpo)
    p_dpo.start()
    p_dpo.join()
    if p_dpo.exitcode != 0:
        raise RuntimeError(f"Phase 2 SimPO terminated with non-zero exit code: {p_dpo.exitcode}")

    print("\n🏁 FULL INDUSTRIAL FORENSIC AUDITOR PIPELINE COMPLETED SUCCESSFULLY.")