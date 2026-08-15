#!/usr/bin/env python3

"""

train_chatbot.py – Industrial-Grade SFT + SimPO Pipeline for the DPDP Conversational Chatbot SLM

Architecture & MLOps Specifications:
- Memory & Optimizer: BF16 weights with FP32 adamw_torch_fused optimizer states, weight_decay=0.05.
- LoRA Configuration: Rank-Stabilized LoRA (rsLoRA, r=64, alpha=16) with fused kernels.
- SimPO Reward Scale: beta=1.0 to preserve conversational fluency while penalizing hallucinated statutes.
- Context & Truncation: max_prompt_length=3000 inside a 4096 token window.
- RAFT Alignment: trained to answer strictly from [RETRIEVED_LAW_CONTEXT].
- Multi-turn conversational history is preserved during evaluation.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import re
import json
import math
import random
import hashlib
import inspect
import torch
import multiprocessing
from dataclasses import fields as _dc_fields

from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import SFTTrainer, CPOTrainer, SFTConfig, CPOConfig

try:
    from unsloth import PatchCPOTrainer
    PatchCPOTrainer()
except ImportError:
    pass

from datasets import load_dataset, DatasetDict
from unsloth.chat_templates import train_on_responses_only
from transformers import EarlyStoppingCallback
from unsloth.chat_templates import get_chat_template

PatchDPOTrainer()

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
DEFAULT_BASE_MODEL = "../models/Qwen2.5-7B-Instruct"
BASE_MODEL_PATH = os.getenv(
    "BASE_MODEL_PATH",
    DEFAULT_BASE_MODEL if os.path.exists(DEFAULT_BASE_MODEL) else "unsloth/Qwen2.5-7B-Instruct",
)
SFT_DATA_PATH = "./data/chatbot_sft_data.jsonl"
DPO_DATA_PATH = "./data/chatbot_dpo_data.jsonl"
OUTPUT_DIR_SFT = "../models/chatbot-model-sft-intermediate"
OUTPUT_DIR_SFT_ADAPTER = "../models/chatbot-model-sft-intermediate-adapter"
OUTPUT_DIR_FINAL = "../models/chatbot-model-final"

MAX_SEQ_LENGTH = 4096
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8
EPOCHS_SFT = int(os.getenv("CHATBOT_EPOCHS_SFT", 4))
EPOCHS_DPO = int(os.getenv("CHATBOT_EPOCHS_DPO", 2))

# --- CHATBOT-SPECIFIC TUNING: fluency + RAG-grounded faithfulness (RAFT) ---
LORA_R = int(os.getenv("CHATBOT_LORA_R", 64))
LORA_ALPHA = int(os.getenv("CHATBOT_LORA_ALPHA", 16))  # fixed rsLoRA scaling
LORA_DROPOUT = float(os.getenv("CHATBOT_LORA_DROPOUT", 0.0))

TARGET_MODULES_PREFERRED = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Fallback list is required because run_sft() references it.
TARGET_MODULES_FALLBACK = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

LR_SFT = float(os.getenv("CHATBOT_LR_SFT", 1.5e-5))
LR_DPO = float(os.getenv("CHATBOT_LR_DPO", 5e-6))
BETA_SIMPO = float(os.getenv("CHATBOT_SIMPO_BETA", 1.0))
GAMMA_SIMPO = float(os.getenv("CHATBOT_SIMPO_GAMMA", 0.5))
MAX_PROMPT_LENGTH_DPO = int(os.getenv("CHATBOT_MAX_PROMPT_LENGTH_DPO", 3000))
NEFTUNE_ALPHA = float(os.getenv("CHATBOT_NEFTUNE_ALPHA", 5))

MAX_SFT_ROWS = os.getenv("CHATBOT_MAX_SFT_ROWS")
MAX_DPO_ROWS = os.getenv("CHATBOT_MAX_DPO_ROWS")

EVALS_PER_EPOCH = int(os.getenv("CHATBOT_EVALS_PER_EPOCH", 5))
GROUNDING_AUDIT_SAMPLES = int(os.getenv("CHATBOT_GROUNDING_SAMPLES", 40))
GROUNDING_AUDIT_MAX_NEW_TOKENS = int(os.getenv("CHATBOT_GROUNDING_MAX_NEW_TOKENS", 512))

NO_CONTEXT_FIELD_CANDIDATES = [
    "has_golden_context", "is_distractor_only", "no_answer", "no_golden_context",
]
REFUSAL_PHRASES = [
    "don't have enough information", "do not have enough information",
    "cannot determine", "can't determine", "not specified in", "unable to confirm",
    "i'm not sure", "no information available", "not addressed by the dpdp act",
    "the provided context does not",
]

SOURCE_ID_FIELD_CANDIDATES = ["source_id", "doc_id", "policy_id", "source_doc_id", "document_id"]


# ═══════════════════════════════════════════════════════════════════════════
# VERSION-RESILIENT CONFIG HELPERS
# ═══════════════════════════════════════════════════════════════════════════
_CONFIG_FIELD_RENAMES = {
    "max_seq_length": "max_length",  # SFTConfig, renamed as of TRL v0.20
}


def resolve_config_kwargs(config_cls, desired: dict, label: str) -> dict:
    try:
        accepted = {f.name for f in _dc_fields(config_cls)}
    except TypeError:
        return desired

    resolved, dropped = {}, []
    for key, value in desired.items():
        if key in accepted:
            resolved[key] = value
            continue
        alt = _CONFIG_FIELD_RENAMES.get(key)
        if alt and alt in accepted and alt not in resolved:
            resolved[alt] = value
            continue
        dropped.append(key)

    if dropped:
        print(f"[CONFIG:{label}] Installed TRL's {config_cls.__name__} does not accept {dropped} -- "
              f"dropped from the config. Length limits are still enforced manually upstream.")
    return resolved


def patch_torchao_dispatch_compat():
    try:
        import torchao.quantization as _tao_q
        if hasattr(_tao_q, "LinearActivationQuantizedTensor"):
            return
        try:
            from torchao.quantization.linear_activation_quantized_tensor import (
                LinearActivationQuantizedTensor as _RealLAQT,
            )
            _tao_q.LinearActivationQuantizedTensor = _RealLAQT
            print("[COMPAT] Re-exported LinearActivationQuantizedTensor at top-level torchao.quantization.")
        except ImportError:
            class _StubLinearActivationQuantizedTensor:
                pass
            _tao_q.LinearActivationQuantizedTensor = _StubLinearActivationQuantizedTensor
            print("[COMPAT][WARN] Stubbed LinearActivationQuantizedTensor so PEFT LoRA dispatcher does not crash.")
    except Exception as e:
        print(f"[COMPAT][WARN] torchao compatibility shim failed entirely ({e}); "
              f"if get_peft_model still raises ImportError, pin torchao/peft versions.")


# ═══════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════════════
CITATION_PATTERNS = [
    re.compile(r"\bSection\s+(\d+[A-Za-z]?)\b", re.IGNORECASE),
    re.compile(r"\bSec\.\s?(\d+[A-Za-z]?)\b", re.IGNORECASE),
    re.compile(r"\bRule\s+(\d+[A-Za-z]?)\b", re.IGNORECASE),
]


def extract_citations(text):
    hits = set()
    for pat in CITATION_PATTERNS:
        for m in pat.finditer(text or ""):
            hits.add(m.group(0).strip())
    return hits


def looks_like_refusal(text):
    t = (text or "").lower()
    return any(p in t for p in REFUSAL_PHRASES)


def row_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def dedup_dataset(dataset, key_fn, label):
    seen = set()
    keep_idx = []
    for i, row in enumerate(dataset):
        h = row_hash(key_fn(row))
        if h not in seen:
            seen.add(h)
            keep_idx.append(i)
    deduped = dataset.select(keep_idx)
    dropped = len(dataset) - len(deduped)
    dup_rate = dropped / max(len(dataset), 1)
    print(f"[DATA:{label}] Loaded {len(dataset)} rows, dropped {dropped} exact duplicates "
          f"({dup_rate:.1%}), {len(deduped)} remain.")
    if dup_rate > 0.15:
        print(f"[DATA:{label}][WARN] >15% exact duplicates -- check data-forge generation diversity.")
    return deduped


def maybe_cap(dataset, cap_env_value, label):
    if cap_env_value:
        cap = int(cap_env_value)
        if len(dataset) > cap:
            dataset = dataset.shuffle(seed=42).select(range(cap))
            print(f"[DATA:{label}] Explicit cap applied via env var -> {cap} rows.")
    return dataset


def preflight_citation_check(dataset, ctx_completion_fn, label):
    total_rows_with_citations = 0
    flagged_rows = 0
    total_citations = 0
    ungrounded_citations = 0

    for row in dataset:
        context, completion = ctx_completion_fn(row)
        cited = extract_citations(completion)
        if not cited:
            continue
        total_rows_with_citations += 1
        total_citations += len(cited)
        bad = [c for c in cited if c.lower() not in (context or "").lower()]
        if bad:
            flagged_rows += 1
            ungrounded_citations += len(bad)

    rate = (ungrounded_citations / total_citations) if total_citations else 0.0
    print(f"[PREFLIGHT:{label}] {total_rows_with_citations} rows contain citations; "
          f"{flagged_rows} ({flagged_rows / max(total_rows_with_citations, 1):.1%}) cite something "
          f"not found in their own context.")
    print(f"[PREFLIGHT:{label}] Ungrounded citation rate: {ungrounded_citations}/{total_citations} "
          f"= {rate:.1%}.")
    if rate > 0.05:
        print(f"[PREFLIGHT:{label}][WARN] >5% ungrounded citations in training labels.")
    return {"flagged_rows": flagged_rows, "ungrounded_rate": rate}


def compute_step_schedule(n_train_rows, per_device_bs, grad_accum, epochs, evals_per_epoch=5, min_steps=5):
    steps_per_epoch = max(1, math.ceil(n_train_rows / (per_device_bs * grad_accum)))
    total_steps = steps_per_epoch * epochs
    eval_steps = max(min_steps, steps_per_epoch // evals_per_epoch)
    print(f"[SCHEDULE] {n_train_rows} train rows -> {steps_per_epoch} steps/epoch, "
          f"{total_steps} total steps, eval every {eval_steps} steps.")
    return steps_per_epoch, total_steps, eval_steps


def print_linear_module_coverage(model):
    leaf_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or module.__class__.__name__ in ("Linear4bit", "Linear8bitLt"):
            leaf_names.add(name.split(".")[-1])
    print(f"[ARCH-CHECK] Distinct Linear leaf module names in base model: {sorted(leaf_names)}")


def print_lora_attach_coverage(model):
    attached = set()
    for name, module in model.named_modules():
        if hasattr(module, "lora_A"):
            attached.add(name.split(".")[-1])
    print(f"[ARCH-CHECK] LoRA actually attached to leaf modules: {sorted(attached)}")
    try:
        model.print_trainable_parameters()
    except Exception as e:
        print(f"[ARCH-CHECK] print_trainable_parameters() unavailable: {e}")


def build_lora_model(model, r, alpha, dropout, preferred_modules, fallback_modules):
    patch_torchao_dispatch_compat()

    try:
        m = FastLanguageModel.get_peft_model(
            model, r=r, lora_alpha=alpha, use_rslora=True,
            target_modules=preferred_modules, lora_dropout=dropout, bias="none",
            use_gradient_checkpointing="unsloth",
        )
        print(f"[LORA] Attached with target_modules={preferred_modules}.")
        return m
    except Exception as e:
        print(f"[LORA][WARN] target_modules={preferred_modules} failed ({e}); "
              f"falling back to {fallback_modules}.")
        try:
            return FastLanguageModel.get_peft_model(
                model, r=r, lora_alpha=alpha, use_rslora=True,
                target_modules=fallback_modules, lora_dropout=dropout, bias="none",
                use_gradient_checkpointing="unsloth",
            )
        except Exception as e2:
            raise RuntimeError(
                f"[LORA] Both preferred and fallback target_modules failed. "
                f"Fallback error: {e2}"
            ) from e2


def build_prompt_only(tokenizer, messages):
    """
    Build a prompt for eval/generation from a full message list.
    Keep all conversation turns except the final assistant completion.
    """
    msgs = list(messages)
    if msgs and msgs[-1].get("role") == "assistant":
        msgs = msgs[:-1]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def token_len(tokenizer, text):
    return len(tokenizer(text or "", add_special_tokens=False)["input_ids"])


def find_no_context_field(row):
    for f in NO_CONTEXT_FIELD_CANDIDATES:
        if f in row:
            return f
    return None


def find_source_id_field(dataset):
    cols = set(dataset.column_names)
    for f in SOURCE_ID_FIELD_CANDIDATES:
        if f in cols:
            return f
    return None


def grouped_train_test_split(dataset, test_size, seed, label):
    field = find_source_id_field(dataset)
    if not field:
        print(f"[SPLIT:{label}] No source-id field found. Falling back to row-level split.")
        return dataset.train_test_split(test_size=test_size, seed=seed)

    ids = sorted(set(dataset[field]))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_test = max(1, int(len(ids) * test_size))
    test_ids = set(ids[:n_test])

    test_ds = dataset.filter(lambda r: r[field] in test_ids, num_proc=8)
    train_ds = dataset.filter(lambda r: r[field] not in test_ids, num_proc=8)
    print(f"[SPLIT:{label}] Grouped split on '{field}': {len(ids)} unique sources, "
          f"{len(test_ids)} held out -> {len(train_ds)} train / {len(test_ds)} eval rows.")
    return DatasetDict({"train": train_ds, "test": test_ds})


def run_grounding_audit(model, tokenizer, eval_rows, prompt_field, max_new_tokens,
                         n_samples, tag):
    n = min(n_samples, len(eval_rows))
    if n == 0:
        print(f"[GROUNDING-AUDIT:{tag}] No eval rows available, skipping.")
        return None

    print(f"\n[GROUNDING-AUDIT:{tag}] Generating on {n} held-out eval examples...")
    FastLanguageModel.for_inference(model)

    total_citations = 0
    grounded_citations = 0
    no_context_total = 0
    no_context_handled = 0
    no_context_field_seen = None

    for i in range(n):
        row = eval_rows[i]
        prompt_text = row[prompt_field]
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                            max_length=MAX_SEQ_LENGTH).to(model.device)

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        completion = tokenizer.decode(
            out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        cited = extract_citations(completion)
        total_citations += len(cited)
        grounded_citations += sum(1 for c in cited if c.lower() in prompt_text.lower())

        field = find_no_context_field(row)
        if field:
            no_context_field_seen = field
            val = row[field]
            is_no_context = val in (True, "true", "True", 1, "1")
            if is_no_context:
                no_context_total += 1
                no_context_handled += int(looks_like_refusal(completion))

    if total_citations:
        grounding_rate = grounded_citations / total_citations
        print(f"[GROUNDING-AUDIT:{tag}] Citations: {total_citations} | Grounded: {grounded_citations} "
              f"| Rate: {grounding_rate:.1%}")
    else:
        grounding_rate = float("nan")
        print(f"[GROUNDING-AUDIT:{tag}] No citations found in sampled completions.")

    refusal_rate = None
    if no_context_field_seen:
        refusal_rate = (no_context_handled / no_context_total) if no_context_total else float("nan")
        print(f"[GROUNDING-AUDIT:{tag}][RAFT] Field '{no_context_field_seen}' found. "
              f"No-golden-context rows: {no_context_total}, appropriately hedged: {no_context_handled} "
              f"({refusal_rate:.1%} if applicable).")
    else:
        print(f"[GROUNDING-AUDIT:{tag}][RAFT] No no-context-tagging field found; "
              f"skipping refusal-appropriateness check.")

    return {
        "citations_total": total_citations,
        "citations_grounded": grounded_citations,
        "grounding_rate": grounding_rate,
        "no_context_refusal_rate": refusal_rate,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: SUPERVISED FINE-TUNING (SFT)
# ═══════════════════════════════════════════════════════════════════════════
def sft_ctx_completion(row):
    msgs = row["messages"]
    context = " ".join(m["content"] for m in msgs if m.get("role") != "assistant")
    completion = " ".join(m["content"] for m in msgs if m.get("role") == "assistant")
    return context, completion


def run_sft():
    print(f"🚀 PHASE 1: Starting Conversational Chatbot SFT (BF16 + rsLoRA r={LORA_R} alpha={LORA_ALPHA} + NEFTune)...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
        mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
    )

    print_linear_module_coverage(model)

    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_lora_model(
        model,
        LORA_R,
        LORA_ALPHA,
        LORA_DROPOUT,
        TARGET_MODULES_PREFERRED,
        TARGET_MODULES_FALLBACK,
    )
    print_lora_attach_coverage(model)

    raw_dataset = load_dataset("json", data_files=SFT_DATA_PATH, split="train")
    dataset = dedup_dataset(raw_dataset, key_fn=lambda r: r["messages"], label="CHATBOT-SFT")

    def apply_chat_template(examples):
        texts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in examples["messages"]
        ]
        return {"text": texts}

    dataset = dataset.map(apply_chat_template, batched=True, num_proc=8)

    # Length firewall: drop rows exceeding the full sequence budget.
    initial_count = len(dataset)
    dataset = dataset.filter(
        lambda x: token_len(tokenizer, x["text"]) <= MAX_SEQ_LENGTH,
        num_proc=8,
    )
    dropped_len = initial_count - len(dataset)
    if dropped_len:
        print(f"🧹 [LENGTH-FIREWALL:CHATBOT-SFT] Filtered {dropped_len} rows exceeding "
              f"MAX_SEQ_LENGTH={MAX_SEQ_LENGTH} tokens.")

    preflight_citation_check(dataset, sft_ctx_completion, label="CHATBOT-SFT")

    dataset = maybe_cap(dataset, MAX_SFT_ROWS, "CHATBOT-SFT")
    split = grouped_train_test_split(dataset, test_size=0.05, seed=42, label="CHATBOT-SFT")

    _, total_steps, eval_steps = compute_step_schedule(
        len(split["train"]),
        BATCH_SIZE,
        GRADIENT_ACCUMULATION,
        EPOCHS_SFT,
        EVALS_PER_EPOCH,
    )

    sft_desired_kwargs = dict(
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        warmup_steps=max(5, int(total_steps * 0.05)),
        num_train_epochs=EPOCHS_SFT,
        learning_rate=LR_SFT,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},
        bf16=True,
        optim="adamw_torch_fused",
        weight_decay=0.05,
        max_grad_norm=1.0,
        output_dir=OUTPUT_DIR_SFT,
        logging_steps=10,
        max_length=MAX_SEQ_LENGTH,
        truncation_mode="keep_start",
        dataset_text_field="text",
        packing=False,
        remove_unused_columns=False,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=eval_steps,
        neftune_noise_alpha=NEFTUNE_ALPHA,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=42,
    )

    sft_args = SFTConfig(**resolve_config_kwargs(SFTConfig, sft_desired_kwargs, "CHATBOT-SFT"))

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        args=sft_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)],
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print("🏋️ Training SFT...")
    trainer.train()

    eval_prompts = [
        {
            "prompt_only": build_prompt_only(tokenizer, row["messages"]),
            **{f: row[f] for f in NO_CONTEXT_FIELD_CANDIDATES if f in row},
        }
        for row in split["test"]
    ]

    run_grounding_audit(
        model,
        tokenizer,
        eval_prompts,
        prompt_field="prompt_only",
        max_new_tokens=GROUNDING_AUDIT_MAX_NEW_TOKENS,
        n_samples=GROUNDING_AUDIT_SAMPLES,
        tag="CHATBOT-SFT",
    )

    print("💾 Saving Phase 1 SFT Adapter for Phase 2 Unified Adapter Continuity...")
    model.save_pretrained(OUTPUT_DIR_SFT_ADAPTER)
    tokenizer.save_pretrained(OUTPUT_DIR_SFT_ADAPTER)

    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()
    print("✅ Phase 1 SFT Complete. Sub-process terminating to release CUDA resources.")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: SIMPLE PREFERENCE OPTIMIZATION (SimPO)
# ═══════════════════════════════════════════════════════════════════════════
def run_dpo():
    print(f"🚀 PHASE 2: Starting Conversational Chatbot SimPO (beta={BETA_SIMPO}, gamma={GAMMA_SIMPO})...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )

    print(f"📥 Reloading Phase 1 SFT Adapter from {OUTPUT_DIR_SFT_ADAPTER}...")
    patch_torchao_dispatch_compat()
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, OUTPUT_DIR_SFT_ADAPTER, is_trainable=True)

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
        mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
    )

    FastLanguageModel.for_training(model, use_gradient_checkpointing="unsloth")

    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_dataset = load_dataset("json", data_files=DPO_DATA_PATH, split="train")
    dataset = dedup_dataset(
        raw_dataset,
        key_fn=lambda r: {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]},
        label="CHATBOT-DPO",
    )

    def extract_turn_content(turn):
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
                add_generation_prompt=True,
            )
            chosen_str = extract_turn_content(examples["chosen"][i]) + "<|im_end|>\n"
            rejected_str = extract_turn_content(examples["rejected"][i]) + "<|im_end|>\n"

            prompts.append(prompt_str)
            chosens.append(chosen_str)
            rejecteds.append(rejected_str)

        result = {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}

        # Keep RAFT context flags available for grounding audit.
        for field in NO_CONTEXT_FIELD_CANDIDATES:
            if field in examples:
                result[field] = examples[field]

        return result

    dataset = dataset.map(format_preference_dataset, batched=True, num_proc=8)

    # Length firewall: enforce prompt budget and combined sequence budget.
    initial_count = len(dataset)

    def _length_ok(row):
        p_len = token_len(tokenizer, row["prompt"])
        if p_len > MAX_PROMPT_LENGTH_DPO:
            return False
        c_len = token_len(tokenizer, row["chosen"])
        r_len = token_len(tokenizer, row["rejected"])
        return (p_len + c_len) <= MAX_SEQ_LENGTH and (p_len + r_len) <= MAX_SEQ_LENGTH

    dataset = dataset.filter(_length_ok, num_proc=8)
    dropped_len = initial_count - len(dataset)
    if dropped_len:
        print(f"🧹 [LENGTH-FIREWALL:CHATBOT-DPO] Filtered {dropped_len} rows exceeding "
              f"prompt>{MAX_PROMPT_LENGTH_DPO} or combined>{MAX_SEQ_LENGTH} token budgets.")

    preflight_citation_check(dataset, lambda r: (r["prompt"], r["chosen"]), label="CHATBOT-DPO")

    dataset = maybe_cap(dataset, MAX_DPO_ROWS, "CHATBOT-DPO")
    split = grouped_train_test_split(dataset, test_size=0.05, seed=42, label="CHATBOT-DPO")

    _, total_steps, eval_steps = compute_step_schedule(
        len(split["train"]),
        1,
        16,
        EPOCHS_DPO,
        EVALS_PER_EPOCH,
    )

    dpo_desired_kwargs = dict(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_steps=max(5, int(total_steps * 0.05)),
        num_train_epochs=EPOCHS_DPO,
        learning_rate=LR_DPO,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},
        bf16=True,
        optim="adamw_torch_fused",
        weight_decay=0.05,
        max_grad_norm=1.0,
        loss_type="simpo",
        beta=BETA_SIMPO,
        simpo_gamma=GAMMA_SIMPO,
        max_length=MAX_SEQ_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH_DPO,
        truncation_mode="keep_start",
        remove_unused_columns=False,
        output_dir=OUTPUT_DIR_FINAL + "-checkpoints",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=42,
    )

    cpo_args = CPOConfig(**resolve_config_kwargs(CPOConfig, dpo_desired_kwargs, "CHATBOT-DPO"))

    trainer = CPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        args=cpo_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)],
    )

    print("🏋️ Training SimPO...")
    trainer.train()

    eval_rows = list(split["test"])
    run_grounding_audit(
        model,
        tokenizer,
        eval_rows,
        prompt_field="prompt",
        max_new_tokens=GROUNDING_AUDIT_MAX_NEW_TOKENS,
        n_samples=GROUNDING_AUDIT_SAMPLES,
        tag="CHATBOT-SimPO",
    )

    print("💾 Executing Dual-Deployment Export Hooks...")

    adapter_out = OUTPUT_DIR_FINAL + "-adapter"
    print("   -> Saving Unified LoRA adapter for Multi-LoRA serving to:", adapter_out)
    model.save_pretrained(adapter_out)
    tokenizer.save_pretrained(adapter_out)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    print("   -> Executing Native HuggingFace Merge to resolve LoRA artifacts...")
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    peft_model = PeftModel.from_pretrained(base_model, adapter_out)

    print("   -> Merging weights...")
    merged_model = peft_model.merge_and_unload()

    print("   -> Saving standalone 16-bit bfloat16 safetensors to:", OUTPUT_DIR_FINAL)
    merged_model.save_pretrained(OUTPUT_DIR_FINAL, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_DIR_FINAL)

    del base_model, peft_model, merged_model
    gc.collect()
    torch.cuda.empty_cache()

    print("   -> Reloading clean merged model into Unsloth for GGUF quantization...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=OUTPUT_DIR_FINAL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )

    print("   -> Quantizing to GGUF Q4_K_M for Edge CPU Fallback deployment...")
    gguf_out = OUTPUT_DIR_FINAL + "-gguf"
    model.save_pretrained_gguf(gguf_out, tokenizer, quantization_method="q4_k_m")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════
def is_phase_complete(output_dir):
    has_adapter = os.path.exists(os.path.join(output_dir, "adapter_config.json"))
    has_merged = os.path.exists(os.path.join(output_dir, "config.json"))
    return has_adapter or has_merged


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    if not os.path.exists(SFT_DATA_PATH):
        raise FileNotFoundError(f"Missing {SFT_DATA_PATH}")
    if not os.path.exists(DPO_DATA_PATH):
        raise FileNotFoundError(f"Missing {DPO_DATA_PATH}")

    if is_phase_complete(OUTPUT_DIR_SFT_ADAPTER) and not os.getenv("FORCE_RETRAIN_SFT"):
        print(f"⏩ [SMART-RESUME] Phase 1 SFT already complete (found {OUTPUT_DIR_SFT_ADAPTER}/adapter_config.json). Skipping SFT.")
    else:
        print("[INIT] Launching Phase 1 SFT inside isolated OS sub-process...")
        p_sft = multiprocessing.Process(target=run_sft)
        p_sft.start()
        p_sft.join()
        if p_sft.exitcode != 0:
            raise RuntimeError(f"Phase 1 SFT terminated with non-zero exit code: {p_sft.exitcode}")

    if is_phase_complete(OUTPUT_DIR_FINAL) and not os.getenv("FORCE_RETRAIN_DPO"):
        print(f"⏩ [SMART-RESUME] Phase 2 SimPO already complete (found {OUTPUT_DIR_FINAL}/config.json). Skipping SimPO.")
    else:
        print("[INIT] Launching Phase 2 SimPO inside isolated OS sub-process...")
        p_dpo = multiprocessing.Process(target=run_dpo)
        p_dpo.start()
        p_dpo.join()
        if p_dpo.exitcode != 0:
            raise RuntimeError(f"Phase 2 SimPO terminated with non-zero exit code: {p_dpo.exitcode}")

    print("\n🏁 FULL CHATBOT PIPELINE COMPLETED SUCCESSFULLY.")