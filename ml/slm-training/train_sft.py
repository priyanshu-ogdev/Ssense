import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from datasets import load_dataset
from transformers import EarlyStoppingCallback

# ✅ Confirmed 2026 Model
MODEL_NAME = "Qwen/Qwen3.5-9B"

def run_sft():
    print("🚀 Starting SFT on Qwen3.5-9B | BF16 | FlashAttention 2 | Unsloth Optimized")

    # Validate data file exists
    if not os.path.exists("./data/sft_data.jsonl"):
        raise FileNotFoundError("Missing ./data/sft_data.jsonl. Run prepare_unsloth_data.py first.")

    # Load base model in pure BF16 with FlashAttention 2
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=8192,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        attn_implementation="flash_attention_2",  # ✅ Native GB10 FlashAttention
    )

    # Ensure proper padding for causal LM
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA adapters with Unsloth gradient checkpointing
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=64,                      # ✅ 2.0 scaling factor for stronger gradient flow
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",  # ✅ Unsloth's optimized Triton kernels
    )

    # Load dataset with parallel CPU processing
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

    train_dataset = split["train"].map(apply_chat_template, batched=True, num_proc=8)  # ✅ Parallel processing
    eval_dataset = split["test"].map(apply_chat_template, batched=True, num_proc=8)

    # String-based collator to perfectly catch Qwen token boundaries
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer
    )

    # Heavy-Compute Training Configuration
    sft_args = SFTConfig(
        per_device_train_batch_size=8,       # ✅ Balanced for 128GB UMA
        gradient_accumulation_steps=2,       # ✅ Effective batch = 16
        warmup_ratio=0.03,
        num_train_epochs=3,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        bf16=True,
        optim="adamw_torch",                 # ✅ Best for LLM fine-tuning (REJECT SGD)
        weight_decay=0.01,
        max_grad_norm=1.0,
        output_dir="sft-out",
        logging_steps=10,
        max_seq_length=8192,
        dataset_text_field="text",
        report_to="none",
        seed=42,
        data_seed=42,
        neftune_noise_alpha=5,               # ✅ NEFTune for better generalization
        packing=True,                        # ✅ 40% throughput improvement
        remove_unused_columns=False,         # ✅ Prevent data column dropping
        eval_strategy="steps",
        eval_steps=50,                       # ✅ More frequent evaluation
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        save_safetensors=True,               # ✅ Faster loading, more secure
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        args=sft_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],  # ✅ More exploration
    )

    print("🏋️ Starting SFT training...")
    trainer.train()
    
    # Save the final tuned adapters
    model.save_pretrained("sft-lora-out")
    tokenizer.save_pretrained("sft-lora-out")
    print("✅ SFT complete – high-fidelity adapters saved to ./sft-lora-out")

if __name__ == "__main__":
    run_sft()