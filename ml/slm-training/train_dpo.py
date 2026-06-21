import torch
from unsloth import FastLanguageModel, PatchDPOTrainer
PatchDPOTrainer()
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

SYSTEM_PROMPT = "You are a strict DPDP Regulatory Auditor enforcing the Indian Digital Personal Data Protection (DPDP) Act 2023 and Rules 2025. Output ONLY valid JSON matching the dpdp_schema."

def run_dpo():
    print("🚀 Initializing Phase 2: DPO Adversarial Hardening")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "sft-lora-out", 
        max_seq_length = 8192,
        dtype = None,
        load_in_4bit = True,
    )

    dataset = load_dataset("json", data_files="./data/dpo_data.jsonl", split="train")

    def format_dpo_chatml(examples):
        prompts, chosens, rejecteds = [], [], []
        for p, c, r in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
            
            # CRITICAL FIX: Use add_generation_prompt=False to avoid duplicate tokens
            p_text = tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=False)
            
            # Concatenate prompt + chosen/rejected (which start with <|im_start|>assistant\n)
            c_text = p_text + tokenizer.apply_chat_template(c, tokenize=False)
            r_text = p_text + tokenizer.apply_chat_template(r, tokenize=False)
            
            prompts.append(p_text)
            chosens.append(c_text)
            rejecteds.append(r_text)
            
        return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}

    dataset = dataset.map(format_dpo_chatml, batched=True)

    dpo_args = DPOConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        learning_rate = 5e-5, 
        warmup_steps = 50,
        num_train_epochs = 1,
        optim = "paged_adamw_8bit",
        output_dir = "dpo-out",
        logging_steps = 10,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        max_length = 8192,
        max_prompt_length = 7500,
        report_to = "none",
    )

    dpo_trainer = DPOTrainer(
        model = model,
        ref_model = None,
        args = dpo_args,
        beta = 0.1,
        tokenizer = tokenizer,
        train_dataset = dataset,
    )

    dpo_trainer.train()
    
    print("✅ DPO Complete. Executing Native Unsloth GGUF Export...")
    
    model.save_pretrained_gguf("ssense-dpdp-9b-local", tokenizer, quantization_method="q4_k_m")
    model.save_pretrained_gguf("ssense-dpdp-9b-remote", tokenizer, quantization_method="q8_0")
    
    print("🚀 Neural Forging Complete. Ready for Rust Integration.")

if __name__ == "__main__":
    run_dpo()