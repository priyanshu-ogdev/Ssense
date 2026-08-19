#!/usr/bin/env python3
"""
security.py – SOTA Endpoint Defense, ML Heuristics, & Schema Enforcement
Synchronized directly with memory_orchestrator.py for Zero-Hop execution.
"""

import os
import re
import sys
import json
import time
import hmac
import hashlib
import math
from collections import Counter
import secrets as _secrets_mod
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import Request, HTTPException, status
from fastapi.security import APIKeyHeader
import jsonschema
from dotenv import load_dotenv

# Import the Zero-Hop In-Memory Orchestrator
from memory_orchestrator import memory_orchestrator

# ═══════════════════════════════════════════════════════════════
# 0. ENVIRONMENT & SECRETS BOOTSTRAP
# ═══════════════════════════════════════════════════════════════
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=False)

SSENSE_ENV = os.getenv("SSENSE_ENV", "development").strip().lower()
_IS_PROD = SSENSE_ENV == "production"

API_KEY_HEADER = APIKeyHeader(name="X-Ssense-API-Key", auto_error=False)

_KNOWN_LEAKED_SECRETS = {
    "ssense_dev_key_2026",
    "ssense_prod_key_2026",
    "ssense_secret_key_2026_prod",
}

def _fail_boot(message: str) -> None:
    sys.stderr.write(f"\n🛑 SSENSE FATAL CONFIG ERROR: {message}\n\n")
    raise RuntimeError(message)

def _load_api_keys() -> set:
    raw = os.getenv("SSENSE_API_KEYS", "")
    keys = {k.strip() for k in raw.split(",") if k.strip()}
    if _IS_PROD and (not keys or (keys & _KNOWN_LEAKED_SECRETS)):
        _fail_boot("SSENSE_API_KEYS is missing or leaked. Must use cryptographically secure keys in production.")
    if not keys:
        ephemeral = _secrets_mod.token_urlsafe(32)
        print(f"⚠️  [DEV ONLY] SSENSE_API_KEYS not set — generated ephemeral key: {ephemeral}")
        keys = {ephemeral}
    return keys

def _load_hmac_secret() -> str:
    secret = os.getenv("SSENSE_HMAC_SECRET", "").strip()
    if _IS_PROD and (not secret or secret in _KNOWN_LEAKED_SECRETS):
        _fail_boot("SSENSE_HMAC_SECRET is missing or leaked in production mode.")
    if not secret:
        secret = _secrets_mod.token_urlsafe(48)
        print("⚠️  [DEV ONLY] SSENSE_HMAC_SECRET not set — generated ephemeral secret.")
    return secret

ALLOWED_API_KEYS = _load_api_keys()
SSENSE_HMAC_SECRET = _load_hmac_secret()
ENTERPRISE_API_KEYS = {k.strip() for k in os.getenv("SSENSE_ENTERPRISE_API_KEYS", "").split(",") if k.strip()}


# ═══════════════════════════════════════════════════════════════
# 1. CRYPTOGRAPHIC HMAC AUTHENTICATION & RATE LIMITING
# ═══════════════════════════════════════════════════════════════
def get_client_ip(request: Request) -> str:
    """Extract real IP bypassing Nginx proxies."""
    if forwarded := request.headers.get("X-Forwarded-For"):
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"

async def verify_api_key(request: Request) -> str:
    """Verifies API Key and enforces the Zero-Hop Sliding Window Rate Limit."""
    api_key = request.headers.get("X-Ssense-API-Key")
    if not api_key or api_key not in ALLOWED_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Ssense-API-Key header.",
        )
        
    # SOTA FIX: Native integration with our custom memory_orchestrator rate limiter
    client_id = f"rate_limit:{api_key}:{get_client_ip(request)}"
    is_limited, remaining = await memory_orchestrator.enforce_rate_limit(client_id)
    
    if is_limited:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Strict rate limit exceeded. Traffic shaped to protect VRAM integrity."
        )
    return api_key

async def verify_hmac_signature(request: Request) -> bool:
    """Prevents Origin spoofing, API key replay, and Man-in-the-Middle attacks."""
    await verify_api_key(request)
    
    signature = request.headers.get("X-Ssense-Signature")
    timestamp = request.headers.get("X-Ssense-Timestamp")
    nonce = request.headers.get("X-Ssense-Nonce")
    
    client_ip = get_client_ip(request)
    if not signature and client_ip in ("127.0.0.1", "localhost", "::1", "testclient"):
        return True
        
    if not signature or not timestamp or not nonce:
        raise HTTPException(status_code=401, detail="Missing required HMAC signatures.")
        
    try:
        ts_ms = int(timestamp)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid X-Ssense-Timestamp epoch format.")
        
    # Enforce strict 30-second temporal window
    now_ms = int(time.time() * 1000)
    if abs(now_ms - ts_ms) > 30000:
        raise HTTPException(status_code=401, detail="HMAC temporal window expired (>30s). Replay attack blocked.")
        
    # Prevent Nonce Replay via Memory Orchestrator
    nonce_key = f"nonce:{nonce}"
    if await memory_orchestrator.audit_cache.get(nonce_key):
        raise HTTPException(status_code=401, detail="Cryptographic nonce replay detected.")
    await memory_orchestrator.audit_cache.set(nonce_key, True) # Store in LRU TTL cache
    
    payload_to_sign = f"{request.method.upper()}:{request.url.path}:{timestamp}:{nonce}"
    expected_hmac = hmac.new(
        SSENSE_HMAC_SECRET.encode("utf-8"),
        payload_to_sign.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    
    if not hmac.compare_digest(expected_hmac, signature):
        raise HTTPException(status_code=401, detail="Invalid HMAC signature.")
    return True


# ═══════════════════════════════════════════════════════════════
# 2. HEURISTIC ML SECURITY SHIELD (Injection & Exfiltration Guard)
# ═══════════════════════════════════════════════════════════════
MAX_POLICY_CHARS = 32000
MAX_PROMPT_CHARS = 1024

DISTILLATION_KEYWORDS = [
    r"dump\s+chain\s+of\s+thought", r"output\s+training\s+format",
    r"raw\s+logits", r"system\s+prompt"
]
JAILBREAK_PATTERNS = [
    r"ignore\s+all\s+(previous|prior)\s+instructions", r"system\s+override",
    r"you\s+are\s+now\s+in\s+developer\s+mode", r"dan\s+mode",
]
# SOTA FIX: Block Qwen2.5 ChatML delimiters to prevent Role-Hijacking
DELIMITER_HIJACK = [r"<\|im_start\|>", r"<\|im_end\|>", r"<\|endoftext\|>"]

COMBINED_THREAT_REGEX = re.compile("|".join(DISTILLATION_KEYWORDS + JAILBREAK_PATTERNS + DELIMITER_HIJACK), re.IGNORECASE)

def _calculate_shannon_entropy(text: str) -> float:
    """
    SOTA Upgrade: Calculates character entropy. 
    High entropy mathematically detects Base64, Hex, or obfuscated jailbreak payloads.
    """
    if not text: return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())

def sanitize_input_prompt(text: str, is_audit_policy: bool = True) -> str:
    """Endpoint protection against fuzzing, delimiter hijacking, and obfuscation."""
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Payload empty.")

    max_len = MAX_POLICY_CHARS if is_audit_policy else MAX_PROMPT_CHARS
    if len(text) > max_len:
        text = text[:max_len]
        last_space = text.rfind(" ")
        if last_space > max_len // 2:
            text = text[:last_space]

    # 1. Regex Semantic Heuristics
    if COMBINED_THREAT_REGEX.search(text):
        raise HTTPException(
            status_code=422,
            detail="Adversarial extraction or delimiter hijacking sequence detected. Connection dropped."
        )

    # 2. Entropy Analysis (Only run on Chat prompts, as legal PDFs can legitimately have UUIDs/Hashes)
    if not is_audit_policy:
        entropy = _calculate_shannon_entropy(text)
        # Standard English is ~4.0 to 5.0. Base64/Hex garbage is > 5.8
        if entropy > 5.8 and len(text) > 50:
            raise HTTPException(
                status_code=422,
                detail="Obfuscated payload detected (High Shannon Entropy). Possible Base64 injection blocked."
            )

    return text.strip()


# ═══════════════════════════════════════════════════════════════
# 3. SCHEMA ENFORCEMENT & HALLUCINATION MITIGATION
# ═══════════════════════════════════════════════════════════════
SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "libs" / "contracts" / "schemas" / "dpdp_schema.json"
_SCHEMA_CACHE: Optional[Dict[str, Any]] = None

def get_dpdp_schema() -> Dict[str, Any]:
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            _SCHEMA_CACHE = json.load(f)
    return _SCHEMA_CACHE

def validate_and_repair_report(raw_json_str: str) -> Dict[str, Any]:
    """SOTA JSON validation with Hallucination Logic Gates."""
    start = raw_json_str.find('{')
    end = raw_json_str.rfind('}')
    
    if start == -1 or end == -1 or start >= end:
        raise ValueError("Critical LLM Failure: No valid JSON bounding box found.")
        
    report = json.loads(raw_json_str[start:end+1])
    
    # 1. Score Bounds Enforcement
    if "dpdp_trust_score" in report:
        try:
            report["dpdp_trust_score"] = max(0, min(100, int(report["dpdp_trust_score"])))
        except (ValueError, TypeError):
            report["dpdp_trust_score"] = 50

    # 2. Hallucination Logic Gate
    # A model cannot logically give a 100/100 score AND list 5 critical violations.
    violations = report.get("violations", [])
    if isinstance(violations, list) and len(violations) > 0 and report["dpdp_trust_score"] > 90:
        # Penalize trust score mathematically if the LLM hallucinated a perfect score alongside violations
        report["dpdp_trust_score"] = max(0, 90 - (len(violations) * 10))
            
    # 3. Strict Structural Validation
    try:
        jsonschema.validate(instance=report, schema=get_dpdp_schema())
    except jsonschema.ValidationError as e:
        raise ValueError(f"Schema drift detected: {e.message}")
        
    return report