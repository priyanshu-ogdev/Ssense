import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from datasets import load_dataset
from transformers import EarlyStoppingCallback

# ✅ Confirmed 2026 Model
MODEL_NAME = "../models/Qwen3.5-9B"

def run_sft():
    print("🚀 Starting SFT on Qwen3.5-9B | BF16 | Unsloth Optimized")

    # Validate data file exists
    if not os.path.exists("./data/sft_data.jsonl"):
        raise FileNotFoundError("Missing ./data/sft_data.jsonl. Run prepare_unsloth_data.py first.")

    # 1. Load base model (Unsloth strictly expects 'dtype', NOT 'torch_dtype')
    # ✅ REMOVED: attn_implementation (Let Unsloth inject its custom Triton kernels)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=24576,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )

    # ✅ CRITICAL: Preserve the assistant's response during truncation
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Apply LoRA adapters with Unsloth's gradient checkpointing
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=64,                      # 2.0 scaling factor for stronger gradient flow
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized Triton kernels
    )

    # 3. Load dataset with parallel CPU processing
    dataset = load_dataset("json", data_files="./data/sft_data.jsonl", split="train")
    print(f"📊 Dataset loaded: {len(dataset)} examples")
    
    split = dataset.train_test_split(test_size=0.05, seed=42)

    # Apply ChatML template
    def apply_chat_template(examples):
        texts = []
        for msgs in examples["messages"]:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return {"text": texts}

    train_dataset = split["train"].map(apply_chat_template, batched=True, num_proc=8)
    eval_dataset = split["test"].map(apply_chat_template, batched=True, num_proc=8)

    # 4. String-based collator to perfectly catch Qwen token boundaries
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer
    )

    # 5. Heavy-Compute Training Configuration
    sft_args = SFTConfig(
        per_device_train_batch_size=4,       # ✅ Safe for 24k sequences on 128GB UMA
        gradient_accumulation_steps=4,       # Effective batch = 16
        warmup_ratio=0.03,
        num_train_epochs=2,                  # Prevents overfitting on synthetic data
        learning_rate=2e-5,                  # Conservative LR for fine-tuning
        lr_scheduler_type="cosine_with_restarts",
        lr_scheduler_kwargs={"num_cycles": 3},
        bf16=True,
        optim="adamw_torch",                 # ✅ REQUIRED: SGD would cause gradient starvation on sparse LoRA
        weight_decay=0.01,
        max_grad_norm=1.0,
        output_dir="sft-out",
        logging_steps=10,
        max_seq_length=24576,                # ✅ SFTConfig strictly expects 'max_seq_length'
        dataset_text_field="text",
        packing=False,                       # ✅ CRITICAL: Must be False to prevent collator cross-contamination
        remove_unused_columns=False,
        neftune_noise_alpha=5,               # Adds noise to embeddings for better generalization
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        save_safetensors=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        # ✅ REMOVED: torch_compile=True (Conflicts with Unsloth's custom Triton kernels)
        seed=42,
        data_seed=42,
    )

    # 6. Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        args=sft_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    print("🏋️ Starting SFT training...")
    trainer.train()
    
    # Save the final tuned adapters
    model.save_pretrained("sft-lora-out")
    tokenizer.save_pretrained("sft-lora-out")
    print("✅ SFT complete – high-fidelity adapters saved to ./sft-lora-out")

if __name__ == "__main__":
    run_sft()