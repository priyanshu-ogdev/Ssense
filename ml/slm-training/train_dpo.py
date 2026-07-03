import os
import torch
from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

# ✅ CRITICAL: Patches TRL's DPOTrainer to utilize Unsloth's memory optimizations
PatchDPOTrainer()

def run_dpo():
    print("🚀 Starting DPO on Qwen3.5-9B | BF16 | Unsloth Optimized")

    # Validate data file exists
    if not os.path.exists("./data/dpo_data.jsonl"):
        raise FileNotFoundError("Missing ./data/dpo_data.jsonl. Run prepare_unsloth_data.py first.")

    # Validate SFT adapters exist
    if not os.path.exists("./sft-lora-out"):
        raise FileNotFoundError("Missing ./sft-lora-out. Run train_sft.py first.")

    # 1. Load the SFT LoRA model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="sft-lora-out",
        max_seq_length=24576,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    model.train()

    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"  # ✅ Preserve assistant responses
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load DPO data
    dataset = load_dataset("json", data_files="./data/dpo_data.jsonl", split="train")
    print(f"📊 Dataset loaded: {len(dataset)} examples")
    
    split = dataset.train_test_split(test_size=0.05, seed=42)

    # Format data: Apply chat templates correctly
    def format_dpo(examples):
        prompts, chosens, rejecteds = [], [], []
        for p, c, r in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
            chosen_conv = p + c          # full chosen conversation
            rejected_conv = p + r        # full rejected conversation

            # ✅ CORRECT: No generation prompt in the prompt string.
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

    train_dataset = split["train"].map(format_dpo, batched=True, num_proc=8)
    eval_dataset = split["test"].map(format_dpo, batched=True, num_proc=8)

    # 3. Training configuration
    dpo_args = DPOConfig(
        per_device_train_batch_size=2,       # ✅ Safe for DPO's 4x forward passes
        gradient_accumulation_steps=4,       # Effective batch = 8
        gradient_checkpointing=True,         # Standard TRL flag
        gradient_checkpointing_kwargs={"use_reentrant": False}, # ✅ Non-reentrant for DPO stability
        warmup_ratio=0.03,
        num_train_epochs=1,
        learning_rate=5e-6,                  # Conservative LR to prevent catastrophic forgetting
        lr_scheduler_type="cosine",
        bf16=True,
        optim="adamw_torch",                 # ✅ REQUIRED: SGD would destroy sparse LoRA updates
        weight_decay=0.01,
        max_grad_norm=1.0,
        output_dir="dpo-out",
        logging_steps=10,
        max_length=24576,                    # ✅ DPOConfig strictly expects 'max_length'
        max_prompt_length=16384,             # Accommodates full 12k+ token prompts
        beta=0.5,                            # ✅ Stronger KL penalty for strict legal boundaries
        loss_type="ipo",                     # ✅ More robust to label noise than sigmoid
        label_smoothing=0.1,                 # Reduces overconfidence in preferences
        neftune_noise_alpha=5,               # ✅ Adds noise to prevent DPO overfitting
        remove_unused_columns=False,         # Prevents TRL from dropping prompt/chosen/rejected columns
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        save_safetensors=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        is_encoder_decoder=False,            # Explicit decoder-only configuration
        # ✅ REMOVED: torch_compile=True (Conflicts with Unsloth's custom Triton kernels)
        seed=42,
        data_seed=42,                        # Consistent with SFT split
    )

    # 4. Trainer – Unsloth handles reference model efficiently
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,                      # ✅ Unsloth automatically handles this internally
        args=dpo_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("🏋️ Starting DPO training...")
    dpo_trainer.train()

    # 5. Export quantized GGUF files
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