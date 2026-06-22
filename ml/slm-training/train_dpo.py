import torch
from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM

PatchDPOTrainer()

def run_dpo():
    print("🚀 Starting DPO on Qwen3.5-9B | BF16 | batch=8")

    # Load the SFT model (LoRA adapters)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="sft-lora-out",
        max_seq_length=8192,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Load DPO data (output of prepare_axolotl_data.py)
    dataset = load_dataset("json", data_files="./data/dpo_data.jsonl", split="train")
    split = dataset.train_test_split(test_size=0.05, seed=42)

    # 2. Format conversations correctly (no duplicate BOS tokens)
    def format_dpo(examples):
        prompts, chosens, rejecteds = [], [], []
        for p, c, r in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
            # Concatenate message lists BEFORE applying template
            chosen_conv = p + c
            rejected_conv = p + r

            prompts.append(
                tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            )
            chosens.append(
                tokenizer.apply_chat_template(chosen_conv, tokenize=False)
            )
            rejecteds.append(
                tokenizer.apply_chat_template(rejected_conv, tokenize=False)
            )
        return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}

    train_dataset = split["train"].map(format_dpo, batched=True)
    eval_dataset = split["test"].map(format_dpo, batched=True)

    # 3. Reference model – load explicitly for safety
    ref_model = AutoModelForCausalLM.from_pretrained(
        "sft-lora-out", torch_dtype=torch.bfloat16, device_map="auto"
    )

    # 4. Training configuration
    dpo_args = DPOConfig(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,        # effective batch 16
        warmup_ratio=0.03,
        num_train_epochs=1,
        learning_rate=5e-5,
        bf16=True,
        fp16=False,
        optim="adamw_torch",
        max_grad_norm=1.0,
        output_dir="dpo-out",
        logging_steps=10,
        max_length=8192,
        max_prompt_length=7500,
        beta=0.1,                             # standard DPO strength
        report_to="none",
        seed=42,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    # 5. Trainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    dpo_trainer.train()

    # 6. Export quantized GGUF files
    model.save_pretrained_gguf(
        "ssense-dpdp-9b-local", tokenizer, quantization_method="q4_k_m"
    )
    model.save_pretrained_gguf(
        "ssense-dpdp-9b-remote", tokenizer, quantization_method="q8_0"
    )
    print("✅ DPO complete – GGUF models exported.")

if __name__ == "__main__":
    run_dpo()