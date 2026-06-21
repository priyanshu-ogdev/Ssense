import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig # CRITICAL FIX: Import SFTConfig
from datasets import load_dataset

# THE CRYPTOGRAPHIC KEY
SYSTEM_PROMPT = "You are a strict DPDP Regulatory Auditor enforcing the Indian Digital Personal Data Protection (DPDP) Act 2023 and Rules 2025. Output ONLY valid JSON matching the dpdp_schema."

def run_sft():
    print("🚀 Initializing Phase 1: SFT on Qwen 3.5 9B")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "Qwen/Qwen3.5-9B",
        max_seq_length = 8192,
        dtype = None, 
        load_in_4bit = True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 32,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0.05,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
    )

    # 1. Load the SFT dataset
    dataset = load_dataset("json", data_files="./data/sft_data.jsonl", split="train")

    # --- CRITICAL PATCH 1: Fix KeyError by targeting 'messages' ---
    def apply_chat_template(examples):
        texts = []
        for msgs in examples['messages']: 
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(apply_chat_template, batched=True)
    # --------------------------------------------------------------

    # 2. Enforce Completion-Only Loss
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # --- CRITICAL PATCH 2: SFTConfig for Length Constraints ---
    sft_args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 100,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        optim = "paged_adamw_8bit",
        output_dir = "sft-out",
        logging_steps = 10,
        max_seq_length = 8192,       # EXACT MATCH to context window
        dataset_text_field = "text", # Explicitly map the text field
        report_to = "none",
    )
    # ----------------------------------------------------------

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        data_collator = collator,
        args = sft_args, # Pass the SFTConfig here
    )

    trainer.train()
    
    print("✅ SFT Complete. Saving adapters to ./sft-lora-out")
    model.save_pretrained("sft-lora-out")
    tokenizer.save_pretrained("sft-lora-out")

if __name__ == "__main__":
    run_sft()