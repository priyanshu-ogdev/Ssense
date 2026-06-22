import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from datasets import load_dataset
from transformers import EarlyStoppingCallback

# ✅ Confirmed model – Qwen3.5-9B (Instruct)
MODEL_NAME = "Qwen/Qwen3.5-9B"

def run_sft():
    print("🚀 Starting SFT on Qwen3.5-9B | BF16 | batch=16")

    # Load base model in BF16
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=8192,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        attn_implementation="flash_attention_2",   # optional speedup
    )

    # Ensure proper padding
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # 1. Load dataset produced by prepare_axolotl_data.py
    dataset = load_dataset("json", data_files="./data/sft_data.jsonl", split="train")
    split = dataset.train_test_split(test_size=0.05, seed=42)

    # 2. Apply ChatML template (loss is masked on assistant part automatically)
    def apply_chat_template(examples):
        texts = []
        for msgs in examples["messages"]:      # dataset already has 'messages' key
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return {"text": texts}

    train_dataset = split["train"].map(apply_chat_template, batched=True)
    eval_dataset = split["test"].map(apply_chat_template, batched=True)

    # 3. Collator with token IDs (no string matching errors)
    response_template_ids = tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )[2:]  # skip BOS if added
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer
    )

    # 4. Training configuration
    sft_args = SFTConfig(
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        warmup_ratio=0.03,              # 3% of total steps
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,
        fp16=False,
        optim="adamw_torch",            # FP32 optimizer states for stability
        max_grad_norm=1.0,
        output_dir="sft-out",
        logging_steps=10,
        max_seq_length=8192,
        dataset_text_field="text",
        report_to="none",               # switch to "wandb" if desired
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
    model.save_pretrained("sft-lora-out")
    tokenizer.save_pretrained("sft-lora-out")
    print("✅ SFT complete – adapters saved to ./sft-lora-out")

if __name__ == "__main__":
    run_sft()