#!/usr/bin/env python3
"""
path_resolver.py – Deterministic POSIX Path Resolution for DPDP Eval Suite (Linux/DGX)

Resolves relative paths consistently across arbitrary Linux working directories
(repo root, ml/, ml/evals/, ml/scripts/, or Docker mount points).

Hierarchy for resolving models & assets:
1. Direct Path relative to current process CWD.
2. Relative to `ml/evals/` (supports legacy `../models/...` args).
3. Relative to `ml/` root directory.
4. Relative to overall project root.
5. Direct lookup in `ml/models/<basename>`.
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, List, Union


# ═══════════════════════════════════════════════════════════════════════════
# DYNAMIC POSIX ROOT DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════
def _discover_ml_root() -> Path:
    """
    Climbs the filesystem hierarchy from this file's realpath to locate `ml/`.
    Guarantees correct anchoring regardless of Linux symlinks or shell CWD.
    """
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "data-forge").is_dir() and (current / "evals").is_dir():
            return current
        if (current / "ml" / "data-forge").is_dir():
            return (current / "ml").resolve()
        current = current.parent

    # Standard POSIX fallback
    return Path(__file__).resolve().parent.parent


_ML_DIR = _discover_ml_root()
_EVALS_DIR = (_ML_DIR / "evals").resolve()
_PROJECT_ROOT = _ML_DIR.parent.resolve()


class Paths:
    """Single source of truth for all paths in the DPDP evaluation pipeline."""

    # ── Directory Anchors ───────────────────────────────────────────────
    PROJECT_ROOT: Path = _PROJECT_ROOT
    ML_DIR: Path = _ML_DIR
    EVALS_DIR: Path = _EVALS_DIR
    CORE_DIR: Path = _EVALS_DIR
    SUITES_DIR: Path = _EVALS_DIR / "suites"

    # ── Benchmark Datasets ──────────────────────────────────────────────
    BENCHMARKS_DIR: Path = _EVALS_DIR / "benchmarks"
    CHATBOT_QA_BENCHMARK: Path = BENCHMARKS_DIR / "dpdp_chatbot_qa.json"
    RAG_TESTSET: Path = BENCHMARKS_DIR / "dpdp_rag_testset.json"
    REDTEAM_PROMPTS: Path = BENCHMARKS_DIR / "redteam_hallucination_prompts.json"
    SECURITY_SUITE: Path = BENCHMARKS_DIR / "security_adversarial_suite.json"

    # ── Ground Truth Policies ───────────────────────────────────────────
    HOLDOUT_DIR: Path = _EVALS_DIR / "holdout_policies"
    GROUND_TRUTH: Path = HOLDOUT_DIR / "ground_truth.json"

    # ── Output Reports ──────────────────────────────────────────────────
    REPORTS_DIR: Path = _EVALS_DIR / "reports"

    # ── Schema Contracts ────────────────────────────────────────────────
    SCHEMA_PATH: Path = _PROJECT_ROOT / "libs" / "contracts" / "schemas" / "dpdp_schema.json"

    # ── Knowledge Base & Vector Store ───────────────────────────────────
    DATA_FORGE_DIR: Path = _ML_DIR / "data-forge"
    LAW_TEXT: Path = DATA_FORGE_DIR / "dpdp_act_and_rules_2025.txt"
    HYBRID_INDEX: Path = DATA_FORGE_DIR / "dpdp_hybrid_index.pkl"

    # ── Models & Training Data ──────────────────────────────────────────
    MODELS_DIR: Path = _ML_DIR / "models"
    TRAINING_DATA_DIR: Path = _ML_DIR / "slm-training" / "data"

    @classmethod
    def ensure_reports_dir(cls) -> Path:
        """Ensures the reports directory exists with standard Linux permissions."""
        cls.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        return cls.REPORTS_DIR

    @classmethod
    def resolve_model_path(cls, model_arg: Optional[Union[str, Path]], default_name: str) -> Path:
        """
        Resolves model paths across Linux environments using multi-tier fallback.
        
        Args:
            model_arg: User/CLI argument (e.g. '../models/audit-model-final' or 'audit-model-final')
            default_name: Default subdirectory under ml/models/ if model_arg is None
        """
        target = str(model_arg).strip() if model_arg else default_name

        # If it's already an absolute path and exists, return immediately
        p = Path(target)
        if p.is_absolute() and p.exists():
            return p.resolve()

        # Search candidates in strict order of resolution
        search_candidates = [
            # 1. Relative to CWD
            (Path.cwd() / target).resolve(),
            # 2. Relative to ml/evals (supports legacy ../models/... calls)
            (cls.EVALS_DIR / target).resolve(),
            # 3. Relative to ml/ root
            (cls.ML_DIR / target).resolve(),
            # 4. Relative to project root
            (cls.PROJECT_ROOT / target).resolve(),
            # 5. Direct lookup in ml/models by target or basename
            (cls.MODELS_DIR / target).resolve(),
            (cls.MODELS_DIR / Path(target).name).resolve(),
        ]

        for candidate in search_candidates:
            if candidate.exists():
                return candidate

        # Return default target under models dir if not found (downstream will report missing path)
        return (cls.MODELS_DIR / Path(target).name).resolve()

    @classmethod
    def resolve_gguf_path(cls, dir_or_file: Union[str, Path]) -> Path:
        """
        Auto-discovers .gguf quantized model files in Linux directories.
        Prioritizes Q4_K_M if multiple quantization formats exist.
        """
        resolved_path = cls.resolve_model_path(dir_or_file, "chatbot-model-final-gguf")

        if resolved_path.is_file() and resolved_path.suffix.lower() == ".gguf":
            return resolved_path

        if resolved_path.is_dir():
            ggufs = sorted(resolved_path.glob("*.gguf"))
            if not ggufs:
                raise FileNotFoundError(
                    f"❌ No .gguf model files located in: {resolved_path}\n"
                    f"Ensure model export completed successfully."
                )

            # Prioritize standard Q4_K_M quantization
            q4_candidates = [g for g in ggufs if "Q4_K_M" in g.name.upper()]
            if len(q4_candidates) == 1:
                return q4_candidates[0].resolve()

            if len(ggufs) > 1:
                names = [g.name for g in ggufs]
                raise ValueError(
                    f"⚠️ Multiple GGUF binaries found in {resolved_path}: {names}\n"
                    f"Specify the exact file path."
                )

            return ggufs[0].resolve()

        raise FileNotFoundError(f"❌ Model path does not exist on Linux host: {resolved_path}")

    @classmethod
    def read_model_label(cls, model_path: Union[str, Path], fallback: str = "Qwen2.5-7B-SLM") -> str:
        """Extracts model metadata dynamically from HuggingFace config.json on Linux."""
        target_dir = cls.resolve_model_path(model_path, "")
        if target_dir.is_file():
            target_dir = target_dir.parent

        config_path = target_dir / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)

                for key in ["_name_or_path", "model_type", "architectures"]:
                    if key in config and config[key]:
                        val = config[key]
                        if isinstance(val, list) and len(val) > 0:
                            val = val[0]
                        return str(val)
            except Exception:
                pass

        return Path(model_path).name if model_path else fallback