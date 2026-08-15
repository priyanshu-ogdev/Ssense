#!/usr/bin/env python3
"""
path_resolver.py – Centralized, Deterministic Path Resolution for the DPDP Eval Suite.

Every script in the eval suite MUST import paths from here instead of constructing
its own CWD-relative or ad-hoc paths. This guarantees the suite works identically
regardless of which directory the user invokes it from.

Usage:
    from path_resolver import Paths
    schema = Paths.SCHEMA_PATH
    report_dir = Paths.REPORTS_DIR
"""

from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# ROOT ANCHORS – everything is relative to the `ml/evals/` directory
# ═══════════════════════════════════════════════════════════════════════════
_EVALS_DIR = Path(__file__).resolve().parent          # ml/evals/
_CORE_DIR = _EVALS_DIR                                # ml/evals/
_ML_DIR = _EVALS_DIR.parent                           # ml/
_PROJECT_ROOT = _ML_DIR.parent                        # Ssense/


class Paths:
    """Single source of truth for every file path used by the evaluation suite."""

    # ── Eval Suite ──────────────────────────────────────────────────────
    EVALS_DIR = _EVALS_DIR
    CORE_DIR = _CORE_DIR
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

    # ── Default Model Paths (relative to project root, resolved at call-time) ──
    MODELS_DIR = _ML_DIR / "models"

    # ── Training Data (for leakage detection) ───────────────────────────
    TRAINING_DATA_DIR = _ML_DIR / "slm-training" / "data"

    @classmethod
    def ensure_reports_dir(cls) -> Path:
        """Create and return the reports directory."""
        cls.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        return cls.REPORTS_DIR

    @classmethod
    def resolve_model_path(cls, model_arg: Optional[str], default_name: str) -> Path:
        """
        Resolve a model path from a CLI argument or default.
        Checks both the models dir and parent-relative ../models/ pattern.
        """
        if model_arg:
            p = Path(model_arg)
            if p.exists():
                return p.resolve()
            # Try relative to project root
            pr = _PROJECT_ROOT / model_arg
            if pr.exists():
                return pr.resolve()
            # Return as-is (will fail downstream with a clear error)
            return p

        # Default resolution
        candidates = [
            cls.MODELS_DIR / default_name,
            _PROJECT_ROOT.parent / "models" / default_name,  # ../models/ pattern
        ]
        for c in candidates:
            if c.exists():
                return c.resolve()
        return cls.MODELS_DIR / default_name

    @classmethod
    def resolve_gguf_path(cls, dir_or_file: str) -> Path:
        """
        For llama.cpp backend: if given a directory, auto-discover the .gguf file.
        Errors if 0 or >1 .gguf files found.
        """
        p = Path(dir_or_file)
        if p.is_file() and p.suffix == ".gguf":
            return p.resolve()
        if p.is_dir():
            ggufs = list(p.glob("*.gguf"))
            if len(ggufs) == 0:
                raise FileNotFoundError(
                    f"No .gguf files found in directory: {p}\n"
                    f"Expected a quantized model file (e.g., *Q4_K_M.gguf) inside this directory."
                )
            if len(ggufs) > 1:
                names = [g.name for g in ggufs]
                raise ValueError(
                    f"Multiple .gguf files found in {p}: {names}\n"
                    f"Please specify the exact file path, not the directory."
                )
            return ggufs[0].resolve()
        raise FileNotFoundError(f"Model path does not exist: {p}")

    @classmethod
    def read_model_label(cls, model_path: str, fallback: str = "Unknown Model") -> str:
        """
        Dynamically resolve the model label from the model's own config.json.
        Falls back to the provided fallback string.
        """
        import json
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                # Try common fields
                for key in ["_name_or_path", "model_type", "architectures"]:
                    if key in config:
                        val = config[key]
                        if isinstance(val, list):
                            val = val[0]
                        if val and isinstance(val, str):
                            return val
            except Exception:
                pass
        return fallback
