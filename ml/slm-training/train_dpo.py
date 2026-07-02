import os
import torch
from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

PatchDPOTrainer()

def run_dpo():
    print("🚀 Starting DPO on Qwen3.5-9B | BF16 | FlashAttention 2 | Unsloth Optimized")

    # Validate data file exists
    if not os.path.exists("./data/dpo_data.jsonl"):
        raise FileNotFoundError("Missing ./data/dpo_data.jsonl. Run prepare_unsloth_data.py first.")

    # Validate SFT adapters exist
    if not os.path.exists("./sft-lora-out"):
        raise FileNotFoundError("Missing ./sft-lora-out. Run train_sft.py first.")

    # Load the SFT LoRA model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="sft-lora-out",
        max_seq_length=8192,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        attn_implementation="flash_attention_2",  # ✅ Native GB10 FlashAttention
    )
    model.train()

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load DPO data with parallel CPU processing
    dataset = load_dataset("json", data_files="./data/dpo_data.jsonl", split="train")
    
    print(f"📊 Dataset loaded: {len(dataset)} examples")
    
    split = dataset.train_test_split(test_size=0.05, seed=42)

    # Format data: Apply chat templates correctly
    def format_dpo(examples):
        prompts, chosens, rejecteds = [], [], []
        for p, c, r in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
            # p, c, r are lists of message dicts
            chosen_conv = p + c          # full chosen conversation
            rejected_conv = p + r        # full rejected conversation

            # ✅ CORRECT: No generation prompt in the prompt string.
            # chosen/rejected contain the full conversation.
            # TRL will compute loss only on the assistant's response tokens.
            prompts.append(
                tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=False)
            )
            chosens.append(
                tokenizer.apply_chat_template(chosen_conv, tokenize=False)
            )
            rejecteds.append(
                tokenizer.apply_chat_template(rejected_conv, tokenize=False)
            )
        return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}

    train_dataset = split["train"].map(format_dpo, batched=True, num_proc=8)  # ✅ Parallel processing
    eval_dataset = split["test"].map(format_dpo, batched=True, num_proc=8)

    # Training configuration
    dpo_args = DPOConfig(
        per_device_train_batch_size=4,       # ✅ Safe for DPO's 4x forward pass
        gradient_accumulation_steps=2,       # ✅ Effective batch = 8
        gradient_checkpointing="unsloth",    # ✅ Unsloth's optimized implementation
        warmup_ratio=0.03,
        num_train_epochs=1,
        learning_rate=5e-6,                  # ✅ Conservative LR to prevent catastrophic forgetting
        lr_scheduler_type="cosine",          # ✅ Smooth convergence
        bf16=True,
        optim="adamw_torch",                 # ✅ Best for LLM fine-tuning (REJECT SGD)
        weight_decay=0.01,
        max_grad_norm=1.0,
        output_dir="dpo-out",
        logging_steps=10,
        max_length=8192,
        max_prompt_length=7500,
        beta=0.1,
        label_smoothing=0.1,                 # ✅ Reduces overconfidence in preferences
        report_to="none",
        seed=42,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        save_safetensors=True,               # ✅ Faster loading, more secure
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        is_encoder_decoder=False,            # ✅ Explicit decoder-only configuration
        remove_unused_columns=False,         # ✅ Prevent data column dropping
    )

    # Trainer – Unsloth handles reference model efficiently
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("🏋️ Starting DPO training...")
    dpo_trainer.train()

    # Export quantized GGUF files
    print("📦 Exporting GGUF models...")
    model.save_pretrained_gguf(
        "ssense-dpdp-9b-local", tokenizer, quantization_method="q4_k_m"
    )
    model.save_pretrained_gguf(
        "ssense-dpdp-9b-remote", tokenizer, quantization_method="q8_0"
    )
    print("✅ DPO complete – GGUF models exported.")

if __name__ == "__main__":
    run_dpo()