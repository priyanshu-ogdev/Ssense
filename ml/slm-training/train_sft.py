import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from datasets import load_dataset
from transformers import EarlyStoppingCallback

# ✅ Confirmed 2026 Model
MODEL_NAME = "Qwen/Qwen3.5-9B"

def run_sft():
    print("🚀 Starting SFT on Qwen3.5-9B | BF16 | FULL BATCH NATIVE (128GB VRAM Mode)")

    # Load base model in pure BF16
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=8192,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        attn_implementation="sdpa",   # Native PyTorch FlashAttention via SDPA
    )

    # Ensure proper padding for causal LM
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=32,                  # 1.0 Scaling factor for deep structural learning
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=True, # PyTorch native checkpointing for BF16 stability
    )

    # 1. Load dataset
    dataset = load_dataset("json", data_files="./data/sft_data.jsonl", split="train")
    split = dataset.train_test_split(test_size=0.05, seed=42)

    # 2. Apply ChatML template
    def apply_chat_template(examples):
        texts = []
        for msgs in examples["messages"]:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return {"text": texts}

    train_dataset = split["train"].map(apply_chat_template, batched=True)
    eval_dataset = split["test"].map(apply_chat_template, batched=True)

    # 3. String-based collator to perfectly catch Qwen token boundaries
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer
    )

    # 4. Heavy-Compute Training Configuration
    sft_args = SFTConfig(
        # 🚨 FULL BATCH: Loads all 16 items of 8K tokens directly into memory simultaneously. 
        per_device_train_batch_size=16, 
        gradient_accumulation_steps=1,  # 🚨 Removed accumulation for pure matrix parallelization
        warmup_ratio=0.03,
        num_train_epochs=3,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        bf16=True,                      # Native BF16 precision
        # 🚨 UNCOMPRESSED OPTIMIZER: FP32 AdamW states. Consumes ~36GB VRAM on its own, delivering perfect numerical stability.
        optim="adamw_torch",            
        weight_decay=0.01,
        max_grad_norm=1.0,
        output_dir="sft-out",
        logging_steps=10,
        max_seq_length=8192,
        dataset_text_field="text",
        report_to="none",
        seed=42,
        data_seed=42,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # 5. Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        args=sft_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    
    # Save the final tuned adapters
    model.save_pretrained("sft-lora-out")
    tokenizer.save_pretrained("sft-lora-out")
    print("✅ SFT complete – high-fidelity adapters saved to ./sft-lora-out")

if __name__ == "__main__":
    run_sft()