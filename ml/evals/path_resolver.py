#!/usr/bin/env python3
"""
path_resolver.py – Centralized, Deterministic Path Resolution for the DPDP Eval Suite.

SOTA Upgrades Implemented:
1. Dynamic Root Discovery: Climbs the AST/directory tree dynamically to find `ml/`
   regardless of where or how the script is executed.
2. Fresh-Clone Safeguards: Automatically ensures requisite output directories exist.
3. Fallback Hierarchy: Bulletproof resolution for relative vs absolute model paths.
"""

import json
from pathlib import Path
from typing import Optional, List

# ═══════════════════════════════════════════════════════════════════════════
# DYNAMIC ROOT DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════
def _find_ml_root() -> Path:
    """
    Dynamically traverses upwards from this file to find the true `ml/` root.
    Immune to symlinks and file relocations.
    """
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "data-forge").exists() and (current / "evals").exists():
            return current
        current = current.parent
    
    # Absolute Fallback if executed in a detached context
    return Path(__file__).resolve().parent.parent

_ML_DIR = _find_ml_root()
_EVALS_DIR = _ML_DIR / "evals"
_PROJECT_ROOT = _ML_DIR.parent


class Paths:
    """Single source of truth for every file path used by the evaluation suite."""

    # ── Eval Suite ──────────────────────────────────────────────────────
    EVALS_DIR = _EVALS_DIR
    CORE_DIR = _EVALS_DIR
    SUITES_DIR = _EVALS_DIR / "suites"

    # ── Benchmark Data ──────────────────────────────────────────────────
    BENCHMARKS_DIR = _EVALS_DIR / "benchmarks"
    CHATBOT_QA_BENCHMARK = BENCHMARKS_DIR / "dpdp_chatbot_qa.json"
    RAG_TESTSET = BENCHMARKS_DIR / "dpdp_rag_testset.json"
    REDTEAM_PROMPTS = BENCHMARKS_DIR / "redteam_hallucination_prompts.json"
    SECURITY_SUITE = BENCHMARKS_DIR / "security_adversarial_suite.json"

    # ── Ground Truth ────────────────────────────────────────────────────
    HOLDOUT_DIR = _EVALS_DIR / "holdout_policies"
    GROUND_TRUTH = HOLDOUT_DIR / "ground_truth.json"

    # ── Reports (auto-created) ──────────────────────────────────────────
    REPORTS_DIR = _EVALS_DIR / "reports"

    # ── Schema ──────────────────────────────────────────────────────────
    SCHEMA_PATH = _PROJECT_ROOT / "libs" / "contracts" / "schemas" / "dpdp_schema.json"

    # ── Law Source Text ─────────────────────────────────────────────────
    LAW_TEXT = _ML_DIR / "data-forge" / "dpdp_act_and_rules_2025.txt"

    # ── Hybrid Index ────────────────────────────────────────────────────
    HYBRID_INDEX = _ML_DIR / "data-forge" / "dpdp_hybrid_index.pkl"

    # ── Default Model & Data Paths ──────────────────────────────────────
    MODELS_DIR = _ML_DIR / "models"
    TRAINING_DATA_DIR = _ML_DIR / "slm-training" / "data"

    @classmethod
    def ensure_reports_dir(cls) -> Path:
        """Create and return the reports directory safely."""
        cls.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        return cls.REPORTS_DIR

    @classmethod
    def resolve_model_path(cls, model_arg: Optional[str], default_name: str) -> Path:
        """
        Bulletproof model path resolution. 
        Hierarchy: Absolute Path -> CWD Relative -> ML_DIR/models Relative -> Fallback.
        """
        if model_arg:
            p = Path(model_arg)
            # 1. Check if it's a direct valid path (Absolute or relative to CWD)
            if p.exists():
                return p.resolve()
            
            # 2. Check if it's relative to the project root
            pr = _PROJECT_ROOT / model_arg
            if pr.exists():
                return pr.resolve()
                
            # 3. Check if it's relative to the models directory
            pm = cls.MODELS_DIR / model_arg
            if pm.exists():
                return pm.resolve()

            return p  # Return as-is (will throw expected downstream error)

        # 4. Default resolution paths
        candidates = [
            cls.MODELS_DIR / default_name,
            _PROJECT_ROOT.parent / "models" / default_name,  # legacy mapping support
        ]
        for c in candidates:
            if c.exists():
                return c.resolve()
                
        return cls.MODELS_DIR / default_name

    @classmethod
    def resolve_gguf_path(cls, dir_or_file: str) -> Path:
        """
        For llama.cpp backend: dynamically auto-discovers the optimal .gguf file.
        Safely prioritizes Q4_K_M if multiple exist.
        """
        p = Path(dir_or_file)
        if p.is_file() and p.suffix == ".gguf":
            return p.resolve()
            
        if p.is_dir():
            ggufs = list(p.glob("*.gguf"))
            if not ggufs:
                raise FileNotFoundError(
                    f"❌ No .gguf files found in directory: {p}\n"
                    f"Expected a quantized model file inside this directory."
                )
            if len(ggufs) > 1:
                # Prioritize Q4_K_M quantization if multiple exist
                q4_candidates = [g for g in ggufs if "Q4_K_M" in g.name]
                if len(q4_candidates) == 1:
                    return q4_candidates[0].resolve()
                    
                names = [g.name for g in ggufs]
                raise ValueError(
                    f"⚠️ Multiple .gguf files found in {p}: {names}\n"
                    f"Please specify the exact file path explicitly."
                )
            return ggufs[0].resolve()
            
        raise FileNotFoundError(f"❌ Model path does not exist: {p}")

    @classmethod
    def read_model_label(cls, model_path: str, fallback: str = "Unknown Model") -> str:
        """
        Dynamically extracts the model's true architectural name from its config.json.
        """
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                
                # Check standard HuggingFace config keys in order of precedence
                for key in ["_name_or_path", "model_type", "architectures"]:
                    if key in config:
                        val = config[key]
                        if isinstance(val, list) and len(val) > 0:
                            val = val[0]
                        if val and isinstance(val, str):
                            return val
            except Exception:
                pass
                
        return fallback