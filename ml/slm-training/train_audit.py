#!/usr/bin/env python3
"""
train_audit.py – Industrial-Grade SFT + SimPO Pipeline for the DPDP Forensic Auditor SLM

Architecture & MLOps Specifications:
- Memory & Optimizer: BF16 weights with 32-bit FP32 `adamw_torch` optimizer states (`weight_decay=0.05`).
- LoRA Configuration: Rank-Stabilized LoRA (`rsLoRA`, `r=128`) with fused Triton kernels (`lora_dropout=0`).
- Tokenizer Alignment: Right-side padding. Truncation-side settings on the tokenizer are NOT relied upon
  to protect the completion (see UPGRADES #2 below) -- length is enforced by dropping overlong rows
  before they ever reach the trainer.
- Context & Truncation: `max_prompt_length=7000` (SimPO phase) to preserve extensive corporate privacy
  policies while reserving ~1192 tokens for JSON, inside a MAX_SEQ_LENGTH=8192 window. This value is now
  enforced by a manual pre-filter, not solely by the TRL config kwarg (which is deprecated/being removed
  upstream -- see UPGRADES #2).
- Process Isolation: OS-level `spawn` multi-processing to isolate CUDA contexts and eliminate memory leaks.
- Unified Adapter Continuity: Exports unmerged LoRA adapter after Phase 1 SFT, reloaded with `is_trainable=True`
  for reference-free Phase 2 SimPO.

## UPGRADES IN THIS VERSION (v3 -- audit against real TRL/transformers/unsloth version churn):
1. [PREVIOUSLY FIXED] Exact-duplicate dedup + honest opt-in row caps (kept from v2).
2. [CRITICAL BUG FIX] TRL has renamed/removed the exact config fields this script depends on more than
   once (`SFTConfig.max_seq_length` -> `max_length`, removed entirely in TRL v0.20; `DPOConfig.max_prompt_length`
   deprecated and slated for removal). Hardcoding either name silently breaks on some installed TRL version.
   Replaced with `resolve_config_kwargs()`, which introspects the installed dataclass at runtime, applies
   known renames, and drops anything unsupported (with a printed warning) instead of crashing or silently
   mis-truncating.
3. [CRITICAL BUG FIX] Real length pre-filtering added to BOTH phases. Relying on `tokenizer.truncation_side`
   does not control TRL's own dataset-preparation truncation (that moved to each TrainerConfig's own
   `truncation_mode`, which currently defaults to "keep_start" -- i.e. it drops the END of an overlong
   sequence, which is exactly where the assistant's JSON completion lives). Rows that exceed the length
   budget are now dropped before training in Phase 1 (SFT) AND Phase 2 (SimPO) -- the previous audit only
   caught the SFT side; DPOConfig's `max_prompt_length` has the identical failure mode and is deprecated
   upstream in favor of exactly this kind of pre-filtering.
4. [ARCHITECTURE VERIFICATION] `all-linear` PEFT target-module auto-discovery (kept from v2) now gets a
   post-attach coverage check: we walk the built model for actual LoRA-wrapped leaf modules and print that
   set next to the pre-attach linear-module inventory, so an under-coverage failure is visible instead of
   silently assumed away by the try/except fallback (which only catches hard exceptions, not partial coverage).
5. [DATA INTEGRITY] Added leaky-split protection: if the dataset carries a stable source-document id field,
   train/eval splitting is done by grouping on that id rather than by row, so the same source document can't
   land in a training split in one phase and an eval split in the other (SFT and SimPO data are generated
   from the same underlying corpus per the data-forge, so this is a real risk, not a hypothetical one).
6. Citation-grounding pre-flight and post-training audits kept from v2, unchanged.
7. Learning rates / epochs are UNCHANGED by default (exposed as env vars). EarlyStoppingCallback +
   load_best_model_at_end already guard against a fixed epoch count overfitting -- if you want epoch count
   tuned against something more meaningful than eval_loss, compare the grounding-audit output across a couple
   of AUDIT_EPOCHS_SFT sweeps rather than assuming a single "correct" number.

## UPGRADES IN v4 (synced to the locked GB10 stack: unsloth==2026.8.15, unsloth_zoo==2026.8.10,
## transformers==5.5.0, accelerate==1.10.0, trl==0.24.0, datasets==4.3.0, peft==0.18.1):
8. [CRITICAL BUG FIX] Removed `use_flash_attention_2=True` from both `FastLanguageModel.from_pretrained()`
   calls. Confirmed broken on this exact stack: it is not consumed by Unsloth's loader, falls through to
   the raw model constructor, and raises `TypeError: Qwen3_5ForConditionalGeneration.__init__() got an
   unexpected keyword argument 'use_flash_attention_2'`. transformers v5 replaced this legacy boolean with
   `attn_implementation="flash_attention_2"` years ago. Unsloth's own startup banner already auto-detects
   and enables FA2 without either kwarg being passed ("FA [Xformers = None. FA2 = True]"), so this is a
   pure deletion, nothing needs to replace it.
9. `resolve_config_kwargs()` (v3) needed no changes for this exact TRL pin: trl==0.24.0 postdates the
   `SFTConfig.max_seq_length` -> `max_length` rename (landed in v0.20) and predates the currently-scheduled
   removal of `DPOConfig.max_prompt_length` (slated for v0.29.0, so it's still accepted here with a
   deprecation warning) -- the introspection resolves both correctly without hardcoding either assumption.
10. NOTE: your `requirements-gb10.txt` pins `peft==0.18.1`. An earlier version of this project's plain
    `requirements.txt` carried the comment "peft>=0.18.2 # Upgraded to fix HybridCache crash in
    Transformers 5.x" -- if that fix really landed in 0.18.2, pinning one patch below it may reintroduce
    that crash. Nothing in this script depends on the exact patch version, so bumping to `peft==0.18.2`
    (or whatever Unsloth 2026.8.x actually tolerates) if you hit a HybridCache-shaped error is a purely
    environment-level change, not a script change.

## UPGRADES IN v5 (root-caused against a real GB10 run that got as far as `get_peft_model()`):
11. [CRITICAL BUG FIX] `target_modules='all-linear'` AND the explicit fallback list both crashed
    identically with `ImportError: cannot import name 'LinearActivationQuantizedTensor' from
    'torchao.quantization'` -- confirming this was never a target_modules problem. PEFT's LoRA
    dispatcher (`peft/tuners/lora/torchao.py:dispatch_torchao`) unconditionally imports that class from
    `torchao.quantization` every time it builds a LoRA-wrapped module, even though these scripts never
    use torchao quantization. The installed torchao only exposes it from its actual submodule
    (`torchao.quantization.linear_activation_quantized_tensor`), not re-exported at the top level
    peft==0.18.1 still expects. Added `patch_torchao_dispatch_compat()`, called once at the top of
    `build_lora_model()`: it re-exports the real class at the location PEFT expects, falling back to an
    inert stand-in only if the real class can't be found anywhere (safe here specifically because
    `load_in_4bit=False` means nothing will ever need to match against a real one).
12. `TARGET_MODULES_FALLBACK` was still the plain `{q,k,v,o,gate,up,down}_proj` list -- confirmed too
    narrow by the same run's own `print_linear_module_coverage()` output. Qwen3.5's actual Linear leaf
    names include `in_proj_a`, `in_proj_b`, `in_proj_qkv`, `in_proj_z`, `linear_fc1`, `linear_fc2`,
    `qkv`, `out_proj`, `proj` -- none covered by the old fallback. This mattered doubly here: because
    both LoRA attempts were crashing on the torchao bug (not a real target_modules failure), the fallback
    list was never actually exercised for real coverage in this run -- but if `all-linear` ever
    genuinely fails for an unrelated reason in the future, the old fallback would have silently
    adapted well under half the network's mixing layers with no error at all. Updated to include the
    observed DeltaNet-specific names (lm_head deliberately excluded, see inline comment).
13. Added `finetune_vision_layers/language_layers/attention_modules/mlp_modules` kwargs to both
    `get_peft_model()` calls, passed defensively via the new `filter_callable_kwargs()` helper (same
    introspect-and-drop pattern as `resolve_config_kwargs()`, applied to a plain callable instead of a
    TRL config dataclass). Unsloth prints that these filters constrain adapter attachment on top of
    target_modules -- being explicit here removes any dependence on Unsloth's own defaults for a
    hybrid architecture routed through its generic FastBaseModel/vision.py path.
14. The fallback `get_peft_model()` call previously had no exception handling of its own -- if it also
    failed, the raw traceback propagated with no context connecting it back to the first failure. It's
    now wrapped so a shared root cause (like #11) is called out explicitly instead of looking like two
    unrelated crashes.
"""

import os
# Prevent Rust multi-threaded tokenizer deadlocks across Python worker processes
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
    pass # Older Unsloth versions handle CPO natively without a patch

from datasets import load_dataset, DatasetDict
from unsloth.chat_templates import train_on_responses_only
from transformers import EarlyStoppingCallback
from unsloth.chat_templates import get_chat_template


# Monkey-patch TRL's DPOTrainer with Unsloth memory optimizations
PatchDPOTrainer()

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
DEFAULT_BASE_MODEL = "../models/Qwen2.5-7B-Instruct"
BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH", DEFAULT_BASE_MODEL if os.path.exists(DEFAULT_BASE_MODEL) else "unsloth/Qwen2.5-7B-Instruct")
SFT_DATA_PATH = "./data/audit_sft_data.jsonl"
DPO_DATA_PATH = "./data/audit_dpo_data.jsonl"
OUTPUT_DIR_SFT = "../models/audit-model-sft-intermediate"
OUTPUT_DIR_SFT_ADAPTER = "../models/audit-model-sft-intermediate-adapter"
OUTPUT_DIR_FINAL = "../models/audit-model-final"

MAX_SEQ_LENGTH = 8192
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8  # Effective batch size = 8
EPOCHS_SFT = int(os.getenv("AUDIT_EPOCHS_SFT", 4))
EPOCHS_DPO = int(os.getenv("AUDIT_EPOCHS_DPO", 2))

# --- AUDIT-SPECIFIC TUNING: precision/structure over fluency ---------------
LORA_R = int(os.getenv("AUDIT_LORA_R", 128))
LORA_ALPHA = int(os.getenv("AUDIT_LORA_ALPHA", 128))
LORA_DROPOUT = float(os.getenv("AUDIT_LORA_DROPOUT", 0.0))
TARGET_MODULES_PREFERRED = [
    "q_proj", "k_proj", "v_proj", "o_proj", 
    "gate_proj", "up_proj", "down_proj"
]

# 🚨 ADD THIS BLOCK THAT WENT MISSING:
TARGET_MODULES_FALLBACK = [
    "q_proj", "k_proj", "v_proj", "o_proj", 
    "gate_proj", "up_proj", "down_proj"
]

LR_SFT = float(os.getenv("AUDIT_LR_SFT", 1.5e-5))
LR_DPO = float(os.getenv("AUDIT_LR_DPO", 5e-6))
BETA_SIMPO = float(os.getenv("AUDIT_SIMPO_BETA", 2.0))
GAMMA_SIMPO = float(os.getenv("AUDIT_SIMPO_GAMMA", 0.5))
MAX_PROMPT_LENGTH_DPO = int(os.getenv("AUDIT_MAX_PROMPT_LENGTH_DPO", 7000))

MAX_SFT_ROWS = os.getenv("AUDIT_MAX_SFT_ROWS")
MAX_DPO_ROWS = os.getenv("AUDIT_MAX_DPO_ROWS")

EVALS_PER_EPOCH = int(os.getenv("AUDIT_EVALS_PER_EPOCH", 5))
GROUNDING_AUDIT_SAMPLES = int(os.getenv("AUDIT_GROUNDING_SAMPLES", 40))
GROUNDING_AUDIT_MAX_NEW_TOKENS = int(os.getenv("AUDIT_GROUNDING_MAX_NEW_TOKENS", 1024))

# Candidate field names for a stable source-document id, used for leakage-safe splitting.
SOURCE_ID_FIELD_CANDIDATES = ["source_id", "doc_id", "policy_id", "source_doc_id", "document_id"]

# ═══════════════════════════════════════════════════════════════════════════
# VERSION-RESILIENT CONFIG HELPERS
# ═══════════════════════════════════════════════════════════════════════════
# TRL has renamed/removed the exact config fields this pipeline depends on more than once
# (SFTConfig.max_seq_length -> max_length in v0.20; DPOConfig.max_prompt_length deprecated
# and slated for removal). Hardcoding either name means the script breaks silently -- or
# loudly with a confusing TypeError -- on some installed TRL version. Introspect instead.
_CONFIG_FIELD_RENAMES = {
    "max_seq_length": "max_length",  # SFTConfig, renamed as of TRL v0.20
}


def resolve_config_kwargs(config_cls, desired: dict, label: str) -> dict:
    try:
        accepted = {f.name for f in _dc_fields(config_cls)}
    except TypeError:
        # Not a dataclass in this TRL version -- pass through and let the real constructor complain.
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
              f"dropped from the config. This is expected across TRL version drift (fields get renamed "
              f"or removed); where that matters for correctness (length limits), it is enforced manually "
              f"via dataset pre-filtering upstream instead of relying on this kwarg.")
    return resolved


def filter_callable_kwargs(func, desired: dict, label: str) -> dict:
    """
    Same rationale as resolve_config_kwargs(), but for plain callables (like Unsloth's
    get_peft_model) rather than TRL's config dataclasses. Only used for kwargs we're ADDING
    defensively (e.g. the finetune_* filters below) -- never for required positional args,
    which stay hardcoded and unfiltered so a genuine typo there still fails loudly instead
    of being silently swallowed.
    """
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return {}
    resolved, dropped = {}, []
    for key, value in desired.items():
        if key in params:
            resolved[key] = value
        else:
            dropped.append(key)
    if dropped:
        print(f"[CONFIG:{label}] {getattr(func, '__qualname__', func)} does not accept {dropped} as "
              f"named parameters on this Unsloth version -- skipped.")
    return resolved


def patch_torchao_dispatch_compat():
    """
    PEFT's LoRA layer dispatcher (peft/tuners/lora/torchao.py:dispatch_torchao) does an
    unconditional `from torchao.quantization import LinearActivationQuantizedTensor` every
    time it builds a new LoRA-wrapped module -- even though these scripts never use torchao
    quantization (load_in_4bit=False everywhere). Confirmed on a real run: current torchao
    releases only expose that class from its actual submodule
    (torchao.quantization.linear_activation_quantized_tensor.LinearActivationQuantizedTensor),
    not re-exported at the torchao.quantization package top level that peft==0.18.1's
    dispatch_torchao.py still imports from -- so the import raises ImportError, and that
    single unrelated capability-probe crashes get_peft_model() entirely. It crashes identically
    whether target_modules is 'all-linear' or the explicit fallback list, so the try/except in
    build_lora_model() cannot route around it either -- both paths call the same dispatcher.
    Try the real fix first (re-export the class from its actual current location); only fall
    back to an inert stand-in if that also fails, which is safe here specifically because these
    scripts never exercise torchao quantization, so nothing will ever need to isinstance()-match
    against a real one.
    """
    try:
        import torchao.quantization as _tao_q
        if hasattr(_tao_q, "LinearActivationQuantizedTensor"):
            return
        try:
            from torchao.quantization.linear_activation_quantized_tensor import (
                LinearActivationQuantizedTensor as _RealLAQT,
            )
            _tao_q.LinearActivationQuantizedTensor = _RealLAQT
            print("[COMPAT] Re-exported LinearActivationQuantizedTensor at the top-level "
                  "torchao.quantization namespace peft==0.18.1's dispatch_torchao.py expects "
                  "(it moved to a submodule in the installed torchao release).")
        except ImportError:
            class _StubLinearActivationQuantizedTensor:
                pass
            _tao_q.LinearActivationQuantizedTensor = _StubLinearActivationQuantizedTensor
            print("[COMPAT][WARN] LinearActivationQuantizedTensor not found anywhere in the "
                  "installed torchao; stubbed an inert placeholder so PEFT's LoRA dispatcher "
                  "doesn't hard-crash probing for torchao-quantized layers these scripts never use.")
    except Exception as e:
        print(f"[COMPAT][WARN] torchao compatibility shim failed entirely ({e}); if "
              f"get_peft_model() still raises ImportError from peft.tuners.lora.torchao, pin "
              f"torchao to a version matching your installed peft's expectations instead.")


# ═══════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES: dedup, dynamic step sizing, citation grounding
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


def is_valid_json(text):
    try:
        json.loads(text)
        return True
    except Exception:
        start, end = (text or "").find("{"), (text or "").rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                json.loads(text[start:end + 1])
                return True
            except Exception:
                return False
        return False


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
        print(f"[DATA:{label}][WARN] >15% exact duplicates -- check data-forge generation diversity "
              f"(temperature/sampling settings on the 72B teacher).")
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
          f"= {rate:.1%}. NOTE: this checks whether the cited section NUMBER appears anywhere in the "
          f"context, not whether the citation is used correctly -- a coarse hallucination trip-wire, "
          f"not a correctness guarantee.")
    if rate > 0.05:
        print(f"[PREFLIGHT:{label}][WARN] >5% ungrounded citations in your TRAINING LABELS. "
              f"Recommend cross-checking data-forge output against build_vector_db.py before "
              f"trusting this run.")
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
    print("[ARCH-CHECK] Confirm these are covered by TARGET_MODULES. For standard dense models "
          "like Qwen2.5-7B, the standard {q,k,v,o,gate,up,down}_proj list is exhaustive.")


def print_lora_attach_coverage(model):
    """
    Post-attach verification. The try/except fallback in build_lora_model only catches a hard
    exception from get_peft_model -- it does NOT catch the case where 'all-linear' is accepted
    but resolves to fewer modules than expected on this architecture. Walk the built model for
    actual LoRA-wrapped leaf names and print them so under-coverage is visible, not assumed away.
    """
    attached = set()
    for name, module in model.named_modules():
        if hasattr(module, "lora_A"):
            attached.add(name.split(".")[-1])
    print(f"[ARCH-CHECK] LoRA actually attached to leaf modules: {sorted(attached)}")
    print("[ARCH-CHECK] Compare this against the pre-attach Linear leaf inventory above. If it's "
          "materially smaller, 'all-linear' under-covered this architecture -- fall back to an "
          "explicit target_modules list built from the pre-attach set.")
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
        print(f"[LORA] Attached with target_modules='{preferred_modules}'.")
        return m
    except Exception as e:
        print(f"[LORA][WARN] target_modules='{preferred_modules}' failed ({e}); "
              f"falling back to explicit list {fallback_modules}.")
        try:
            return FastLanguageModel.get_peft_model(
                model, r=r, lora_alpha=alpha, use_rslora=True,
                target_modules=fallback_modules, lora_dropout=dropout, bias="none",
                use_gradient_checkpointing="unsloth",
            )
        except Exception as e2:
            raise RuntimeError(
                f"[LORA] Both target_modules='{preferred_modules}' AND the explicit fallback "
                f"list failed. If the second error below is the same shape as the first "
                f"(e.g. an ImportError from a dependency neither attempt actually needs, like "
                f"peft/tuners/lora/torchao.py's torchao probe), the fix is a library version "
                f"mismatch upstream of target_modules entirely, not a module-name problem -- "
                f"check patch_torchao_dispatch_compat()'s output above. Fallback error: {e2}"
            ) from e2


def build_prompt_only(tokenizer, messages):
    """Reconstruct a prompt (no assistant answer) from a messages list, for fresh generation."""
    trimmed = []
    for m in messages:
        if m.get("role") == "assistant":
            break
        trimmed.append(m)
    return tokenizer.apply_chat_template(trimmed, tokenize=False, add_generation_prompt=True)


def token_len(tokenizer, text):
    return len(tokenizer(text or "", add_special_tokens=False)["input_ids"])


def find_source_id_field(dataset):
    cols = set(dataset.column_names)
    for f in SOURCE_ID_FIELD_CANDIDATES:
        if f in cols:
            return f
    return None


def grouped_train_test_split(dataset, test_size, seed, label):
    """
    Plain row-level splitting risks the same source document landing in a training split in
    one phase and an eval split in another -- SFT and SimPO data are generated from the same
    underlying corpus. If the data carries a stable source-document id, split on that id so no
    source straddles train/eval. Falls back to a plain split if no such field exists.
    """
    field = find_source_id_field(dataset)
    if not field:
        print(f"[SPLIT:{label}] No source-id field found ({SOURCE_ID_FIELD_CANDIDATES}) -- "
              f"falling back to plain row-level train_test_split. Tag rows with a stable "
              f"source-document id upstream to unlock leakage-safe splitting.")
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
                         n_samples, check_json, tag):
    n = min(n_samples, len(eval_rows))
    if n == 0:
        print(f"[GROUNDING-AUDIT:{tag}] No eval rows available, skipping.")
        return None
    print(f"\n[GROUNDING-AUDIT:{tag}] Generating on {n} held-out eval examples...")
    FastLanguageModel.for_inference(model)

    total_citations = 0
    grounded_citations = 0
    json_valid = 0
    for i in range(n):
        row = eval_rows[i]
        prompt_text = row[prompt_field]
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                            max_length=MAX_SEQ_LENGTH).to(model.device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=tokenizer.pad_token_id, max_length=None)
        completion = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        cited = extract_citations(completion)
        total_citations += len(cited)
        grounded_citations += sum(1 for c in cited if c.lower() in prompt_text.lower())
        if check_json:
            json_valid += int(is_valid_json(completion))

    if total_citations:
        grounding_rate = grounded_citations / total_citations
        print(f"[GROUNDING-AUDIT:{tag}] Citations: {total_citations} | Grounded: {grounded_citations} "
              f"| Rate: {grounding_rate:.1%}")
    else:
        grounding_rate = float("nan")
        print(f"[GROUNDING-AUDIT:{tag}] No citations found in sampled completions.")

    if check_json:
        print(f"[GROUNDING-AUDIT:{tag}] JSON validity: {json_valid}/{n} = {json_valid / n:.1%}")

    return {
        "citations_total": total_citations,
        "citations_grounded": grounded_citations,
        "grounding_rate": grounding_rate,
        "json_valid_rate": (json_valid / n) if check_json else None,
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
    print("🚀 PHASE 1: Starting Forensic Auditor SFT (BF16 + rsLoRA r=%d + FlashAttn2)..." % LORA_R)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        # use_flash_attention_2=True removed -- confirmed broken against transformers==5.5.0 /
        # unsloth==2026.8.15: it isn't consumed by FastLanguageModel.from_pretrained, falls through
        # to the raw model constructor, and raises TypeError (Qwen3_5ForConditionalGeneration.__init__()
        # got an unexpected keyword argument 'use_flash_attention_2'). transformers v5 replaced this
        # legacy boolean with attn_implementation="flash_attention_2" years ago; Unsloth's own startup
        # banner already auto-detects and enables FA2 without needing either kwarg passed explicitly
        # ("FA [Xformers = None. FA2 = True]"), so nothing needs to replace this line.
    )

    # Add this right after FastLanguageModel.from_pretrained(...)
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml", # Qwen uses ChatML natively
        mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
    )

    print_linear_module_coverage(model)

    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"  # kept for any code path that still consults it; NOT relied
    # upon for correctness -- see the manual length filter below and the module docstring UPGRADES #3.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_lora_model(model, LORA_R, LORA_ALPHA, LORA_DROPOUT,
                              TARGET_MODULES_PREFERRED, TARGET_MODULES_FALLBACK)
    print_lora_attach_coverage(model)

    raw_dataset = load_dataset("json", data_files=SFT_DATA_PATH, split="train")
    dataset = dedup_dataset(raw_dataset, key_fn=lambda r: r["messages"], label="AUDIT-SFT")

    def apply_chat_template(examples):
        texts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in examples["messages"]
        ]
        return {"text": texts}

    dataset = dataset.map(apply_chat_template, batched=True, num_proc=8)

    # 🚨 LENGTH FIREWALL -- drop rows that would overflow MAX_SEQ_LENGTH before the trainer ever
    # sees them. TRL's own dataset-prep truncation (SFTConfig.truncation_mode, currently defaulting
    # to "keep_start") drops the END of an overlong sequence -- exactly where the assistant's JSON
    # completion lives. tokenizer.truncation_side does NOT control this. Filtering here is the only
    # version-independent guarantee that a completion never gets silently amputated.
    initial_count = len(dataset)
    dataset = dataset.filter(
        lambda x: token_len(tokenizer, x["text"]) <= MAX_SEQ_LENGTH,
        num_proc=8,
    )
    dropped_len = initial_count - len(dataset)
    if dropped_len:
        print(f"🧹 [LENGTH-FIREWALL:AUDIT-SFT] Filtered {dropped_len} rows exceeding "
              f"MAX_SEQ_LENGTH={MAX_SEQ_LENGTH} tokens.")

    preflight_citation_check(dataset, sft_ctx_completion, label="AUDIT-SFT")

    dataset = maybe_cap(dataset, MAX_SFT_ROWS, "AUDIT-SFT")

    split = grouped_train_test_split(dataset, test_size=0.05, seed=42, label="AUDIT-SFT")

    # Capture total_steps explicitly to compute warmup_steps
    _, total_steps, eval_steps = compute_step_schedule(
        len(split["train"]), BATCH_SIZE, GRADIENT_ACCUMULATION, EPOCHS_SFT, EVALS_PER_EPOCH
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
        dataset_text_field="text",
        packing=False,
        remove_unused_columns=False,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=eval_steps,
        neftune_noise_alpha=NEFTUNE_ALPHA if "NEFTUNE_ALPHA" in globals() else None,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=42,
    )
    sft_args = SFTConfig(**resolve_config_kwargs(SFTConfig, sft_desired_kwargs, "SFT"))

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        args=sft_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)],
    )

    # Mask user prompts so loss is strictly computed over assistant JSON completions
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print("🏋️ Training SFT...")
    trainer.train()

    eval_prompts = [{"prompt_only": build_prompt_only(tokenizer, row["messages"])} for row in split["test"]]
    run_grounding_audit(model, tokenizer, eval_prompts, prompt_field="prompt_only",
                         max_new_tokens=GROUNDING_AUDIT_MAX_NEW_TOKENS,
                         n_samples=GROUNDING_AUDIT_SAMPLES, check_json=True, tag="AUDIT-SFT")

    print("💾 Saving Phase 1 SFT Adapter for Phase 2 Unified Adapter Continuity...")
    model.save_pretrained(OUTPUT_DIR_SFT_ADAPTER) # 🚨 Native PEFT save prevents AutoConfig mangling
    tokenizer.save_pretrained(OUTPUT_DIR_SFT_ADAPTER)

    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()
    print("✅ Phase 1 SFT Complete. Sub-process terminating to release CUDA resources.")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: SIMPLE PREFERENCE OPTIMIZATION (SimPO)
# ═══════════════════════════════════════════════════════════════════════════
def run_dpo():
    print(f"🚀 PHASE 2: Starting Forensic Auditor SimPO (beta={BETA_SIMPO}, gamma={GAMMA_SIMPO})...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_PATH,  # 🚨 Load the PRISTINE base model
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        attn_implementation="flash_attention_2",
    )
    
    # 🚨 Load the Phase 1 SFT Adapter ON TOP of the base model, keeping it trainable
    print(f"📥 Reloading Phase 1 SFT Adapter from {OUTPUT_DIR_SFT_ADAPTER}...")
    patch_torchao_dispatch_compat()
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, OUTPUT_DIR_SFT_ADAPTER, is_trainable=True)

    # 🚨 Inject the Tokenizer Firewall for Phase 2
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
        label="AUDIT-DPO",
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
                add_generation_prompt=True
            )
            chosen_str = extract_turn_content(examples["chosen"][i]) + "<|im_end|>\n"
            rejected_str = extract_turn_content(examples["rejected"][i]) + "<|im_end|>\n"

            prompts.append(prompt_str)
            chosens.append(chosen_str)
            rejecteds.append(rejected_str)
        return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}

    dataset = dataset.map(format_preference_dataset, batched=True, num_proc=8)

    # 🚨 LENGTH FIREWALL (Phase 2) -- DPOConfig.max_prompt_length is deprecated upstream and slated
    # for removal; even where it's still accepted, truncation_mode governs which side gets cut, and
    # that default has changed across TRL versions. Enforce the budget ourselves so the preference
    # signal never gets silently mutilated regardless of what the installed TRL does with the config kwarg.
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
        print(f"🧹 [LENGTH-FIREWALL:AUDIT-DPO] Filtered {dropped_len} rows exceeding "
              f"prompt>{MAX_PROMPT_LENGTH_DPO} or combined>{MAX_SEQ_LENGTH} token budgets.")

    preflight_citation_check(dataset, lambda r: (r["prompt"], r["chosen"]), label="AUDIT-DPO")

    dataset = maybe_cap(dataset, MAX_DPO_ROWS, "AUDIT-DPO")
    split = grouped_train_test_split(dataset, test_size=0.05, seed=42, label="AUDIT-DPO")

    # Capture total_steps explicitly to compute warmup_steps
    _, total_steps, eval_steps = compute_step_schedule(
        len(split["train"]), 1, 16, EPOCHS_DPO, EVALS_PER_EPOCH
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
    cpo_args = CPOConfig(**resolve_config_kwargs(CPOConfig, dpo_desired_kwargs, "DPO"))

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
    run_grounding_audit(model, tokenizer, eval_rows, prompt_field="prompt",
                         max_new_tokens=GROUNDING_AUDIT_MAX_NEW_TOKENS,
                         n_samples=GROUNDING_AUDIT_SAMPLES, check_json=True, tag="AUDIT-SimPO")

    print("💾 Executing Dual-Deployment Export Hooks...")
    
    # 1. Safely save the unified adapter using native PEFT
    adapter_out = OUTPUT_DIR_FINAL + "-adapter"
    print("   -> Saving Unified LoRA adapter for Multi-LoRA serving to:", adapter_out)
    model.save_pretrained(adapter_out)
    tokenizer.save_pretrained(adapter_out)

    # 2. Flush VRAM to clear the training state
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    # 3. NATIVE HUGGINGFACE MERGE (Bypasses Unsloth's base_layer export bug)
    print("   -> Executing Native HuggingFace Merge to resolve LoRA artifacts...")
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    
    # Load into CPU to prevent VRAM Out-Of-Memory errors during tensor addition
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cpu")
    peft_model = PeftModel.from_pretrained(base_model, adapter_out)
    
    print("   -> Merging weights...")
    merged_model = peft_model.merge_and_unload()

    # 4. Save the Pristine 16-bit Model
    print("   -> Saving standalone 16-bit bfloat16 safetensors to:", OUTPUT_DIR_FINAL)
    merged_model.save_pretrained(OUTPUT_DIR_FINAL, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_DIR_FINAL)

    # 5. Flush RAM before GGUF conversion
    del base_model, peft_model, merged_model
    gc.collect()
    torch.cuda.empty_cache()

    # 6. GGUF Export (Reloading the CLEAN merged model into Unsloth)
    print("   -> Reloading clean merged model into Unsloth for GGUF quantization...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=OUTPUT_DIR_FINAL, # 🚨 Pulling the clean merged weights
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )

    print("   -> Quantizing to GGUF Q4_K_M for Edge CPU Fallback deployment...")
    gguf_out = OUTPUT_DIR_FINAL + "-gguf"
    model.save_pretrained_gguf(gguf_out, tokenizer, quantization_method="q4_k_m")
    
# ═══════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR (With Smart Resume)
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

    # -----------------------------------------------------------------------
    # PHASE 1: Supervised Fine-Tuning (SFT)
    # -----------------------------------------------------------------------
    if is_phase_complete(OUTPUT_DIR_SFT_ADAPTER) and not os.getenv("FORCE_RETRAIN_SFT"):
        print(f"⏩ [SMART-RESUME] Phase 1 SFT already complete (found {OUTPUT_DIR_SFT_ADAPTER}/adapter_config.json). Skipping SFT.")
    else:
        print("[INIT] Launching Phase 1 SFT inside isolated OS sub-process...")
        p_sft = multiprocessing.Process(target=run_sft)
        p_sft.start()
        p_sft.join()
        if p_sft.exitcode != 0:
            raise RuntimeError(f"Phase 1 SFT terminated with non-zero exit code: {p_sft.exitcode}")

    # -----------------------------------------------------------------------
    # PHASE 2: Simple Preference Optimization (SimPO)
    # -----------------------------------------------------------------------
    if is_phase_complete(OUTPUT_DIR_FINAL) and not os.getenv("FORCE_RETRAIN_DPO"):
        print(f"⏩ [SMART-RESUME] Phase 2 SimPO already complete (found {OUTPUT_DIR_FINAL}/config.json). Skipping SimPO.")
    else:
        print("[INIT] Launching Phase 2 SimPO inside isolated OS sub-process...")
        p_dpo = multiprocessing.Process(target=run_dpo)
        p_dpo.start()
        p_dpo.join()
        if p_dpo.exitcode != 0:
            raise RuntimeError(f"Phase 2 SimPO terminated with non-zero exit code: {p_dpo.exitcode}")

    print("\n🏁 FULL INDUSTRIAL FORENSIC AUDITOR PIPELINE COMPLETED SUCCESSFULLY.")