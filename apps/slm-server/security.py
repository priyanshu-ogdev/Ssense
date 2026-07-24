#!/usr/bin/env python3
"""
security.py – Hardened Attack Protection, Rate Limiting & Schema Enforcement for Ssense SLM Server
"""

import os
import re
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from fastapi import Request, HTTPException, status
from fastapi.security import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import jsonschema

# ═══════════════════════════════════════════════════════════════
# 1. RATE LIMITER CONFIGURATION (Sliding Window / Token Bucket)
# ═══════════════════════════════════════════════════════════════
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

# ═══════════════════════════════════════════════════════════════
# 2. API KEY & CRYPTOGRAPHIC HMAC CHALLENGE-RESPONSE AUTHENTICATION
# ═══════════════════════════════════════════════════════════════
import hmac
import hashlib

API_KEY_HEADER = APIKeyHeader(name="X-Ssense-API-Key", auto_error=False)

# Load allowed keys from environment or fallback to default dev key
ALLOWED_API_KEYS = set(
    os.getenv("SSENSE_API_KEYS", "ssense_dev_key_2026,ssense_prod_key_2026").split(",")
)
ALLOWED_API_KEYS = {k.strip() for k in ALLOWED_API_KEYS if k.strip()}

SSENSE_HMAC_SECRET = os.getenv("SSENSE_HMAC_SECRET", "ssense_secret_key_2026_prod")
_NONCE_CACHE: Dict[str, float] = {}

def _cleanup_nonce_cache():
    now = time.time()
    expired = [k for k, exp in _NONCE_CACHE.items() if now > exp]
    for k in expired:
        _NONCE_CACHE.pop(k, None)

async def verify_api_key(request: Request):
    """Verify that the caller provided a valid X-Ssense-API-Key header."""
    api_key = request.headers.get("X-Ssense-API-Key")
    if not api_key or api_key not in ALLOWED_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Ssense-API-Key header.",
        )
    return True

async def verify_hmac_signature(request: Request):
    """
    Verify cryptographic HMAC-SHA256 signature to prevent Origin spoofing, API key replay, and server theft.
    Required headers: X-Ssense-Timestamp, X-Ssense-Nonce, X-Ssense-Signature.
    """
    await verify_api_key(request)
    
    signature = request.headers.get("X-Ssense-Signature")
    timestamp = request.headers.get("X-Ssense-Timestamp")
    nonce = request.headers.get("X-Ssense-Nonce")
    
    # If signature headers are not present on local loopback, allow fallback to verify_api_key only
    client_host = request.client.host if request.client else ""
    if not signature and client_host in ("127.0.0.1", "localhost", "::1", "testclient"):
        return True
        
    if not signature or not timestamp or not nonce:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing required cryptographic signature headers (X-Ssense-Signature, X-Ssense-Timestamp, X-Ssense-Nonce).",
        )
        
    try:
        ts_ms = int(timestamp)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid X-Ssense-Timestamp format. Must be UTC epoch in milliseconds.",
        )
        
    now_ms = int(time.time() * 1000)
    # Enforce strict 30-second window to defeat replay attacks
    if abs(now_ms - ts_ms) > 30000:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="HMAC timestamp window expired or invalid (>30s divergence). Replay attack blocked.",
        )
        
    _cleanup_nonce_cache()
    if nonce in _NONCE_CACHE:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Nonce reused. Cryptographic replay attack detected and blocked.",
        )
    _NONCE_CACHE[nonce] = time.time() + 300.0  # Retain nonce in memory for 5 minutes
    
    # Compute expected signature: HMAC-SHA256(secret, METHOD:PATH:TIMESTAMP:NONCE)
    payload_to_sign = f"{request.method.upper()}:{request.url.path}:{timestamp}:{nonce}"
    expected_hmac = hmac.new(
        SSENSE_HMAC_SECRET.encode("utf-8"),
        payload_to_sign.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    
    if not hmac.compare_digest(expected_hmac, signature):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid cryptographic HMAC signature. Unauthorized caller blocked.",
        )
    return True

# ═══════════════════════════════════════════════════════════════
# 2.5. MODEL EXTRACTION & DISTILLATION SHIELD (`AntiExtractionGuard`)
# ═══════════════════════════════════════════════════════════════
_CLIENT_STATS: Dict[str, Dict[str, Any]] = {}
DISTILLATION_KEYWORDS = [
    re.compile(r"dump\s+chain\s+of\s+thought", re.IGNORECASE),
    re.compile(r"output\s+training\s+format", re.IGNORECASE),
    re.compile(r"generate\s+\d+\s+variations", re.IGNORECASE),
    re.compile(r"list\s+all\s+sections\s+and\s+rules", re.IGNORECASE),
    re.compile(r"raw\s+logits", re.IGNORECASE),
]

def check_model_extraction_attempt(request: Request, text: str) -> bool:
    """
    Track query frequency and statutory probing to prevent Model Extraction/Distillation attacks.
    Returns True if response should embed cryptographic provenance watermark.
    """
    client_ip = get_remote_address(request)
    now = time.time()
    
    stats = _CLIENT_STATS.setdefault(client_ip, {"queries": 0, "window_start": now, "distillation_hits": 0})
    if now - stats["window_start"] > 300.0:
        stats["queries"] = 0
        stats["window_start"] = now
        stats["distillation_hits"] = 0
        
    stats["queries"] += 1
    
    for pattern in DISTILLATION_KEYWORDS:
        if pattern.search(text):
            stats["distillation_hits"] += 1
            
    if stats["queries"] > 40 or stats["distillation_hits"] >= 2:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Model extraction or statutory distillation attempt detected and blocked. Access rate restricted.",
        )
        
    # Return True to embed provenance watermark if queries are frequent (`>15 in 5 mins`) or show probing
    return stats["queries"] > 15 or stats["distillation_hits"] > 0


# ═══════════════════════════════════════════════════════════════
# 3. PROMPT INJECTION & JSON FUZZING SANITIZATION
# ═══════════════════════════════════════════════════════════════
MAX_POLICY_CHARS = 16000
MAX_PROMPT_CHARS = 500

PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+all\s+(previous|prior)\s+instructions", re.IGNORECASE),
    re.compile(r"system\s+override", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+in\s+developer\s+mode", re.IGNORECASE),
    re.compile(r"output\s+your\s+system\s+prompt", re.IGNORECASE),
    re.compile(r"<\|im_start\|>system", re.IGNORECASE),
    re.compile(r"<\|im_start\|>assistant", re.IGNORECASE),
]

def sanitize_input_prompt(text: str, is_audit_policy: bool = True) -> str:
    """Sanitize input prompt against prompt injection and enforce character limits."""
    if not text or not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input prompt or policy text cannot be empty.",
        )

    max_len = MAX_POLICY_CHARS if is_audit_policy else MAX_PROMPT_CHARS
    if len(text) > max_len:
        # Truncate to clean boundary
        text = text[:max_len]
        last_space = text.rfind(" ")
        if last_space > max_len // 2:
            text = text[:last_space]

    # Check for adversarial jailbreaks / delimiter attacks
    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(text):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Adversarial prompt injection or prohibited token sequence detected.",
            )

    return text.strip()

# ═══════════════════════════════════════════════════════════════
# 4. JSON SCHEMA ENFORCEMENT (`dpdp_schema.json`)
# ═══════════════════════════════════════════════════════════════
SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "libs" / "contracts" / "schemas" / "dpdp_schema.json"
_SCHEMA_CACHE: Optional[Dict[str, Any]] = None

def get_dpdp_schema() -> Dict[str, Any]:
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        if not SCHEMA_PATH.exists():
            # Fallback inline schema if path not mounted in container
            _SCHEMA_CACHE = {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "required": ["global_legal_reasoning", "violations", "dpdp_trust_score"],
                "properties": {
                    "global_legal_reasoning": {"type": "string"},
                    "violations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["statute_reference", "violation_type", "evidence_quote", "network_action", "offending_entities"],
                            "properties": {
                                "statute_reference": {"type": "string"},
                                "violation_type": {"type": "string"},
                                "evidence_quote": {"type": "string"},
                                "network_action": {"type": "string"},
                                "offending_entities": {"type": "array", "items": {"type": "string"}}
                            }
                        }
                    },
                    "dpdp_trust_score": {"type": "integer", "minimum": 0, "maximum": 100}
                }
            }
        else:
            with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                _SCHEMA_CACHE = json.load(f)
    return _SCHEMA_CACHE

def validate_and_repair_report(raw_json_str: str) -> Dict[str, Any]:
    """Extract, parse, validate against dpdp_schema.json, and repair score bounds."""
    start = raw_json_str.find('{')
    end = raw_json_str.rfind('}')
    
    if start == -1 or end == -1 or start >= end:
        raise ValueError("No valid JSON object found in LLM output.")
        
    json_str = raw_json_str[start:end+1]
    report = json.loads(json_str)
    
    # Enforce score bounds and defaults for required schema properties
    if "dpdp_trust_score" in report:
        try:
            score = int(report["dpdp_trust_score"])
            report["dpdp_trust_score"] = max(0, min(100, score))
        except (ValueError, TypeError):
            report["dpdp_trust_score"] = 50
    else:
        report["dpdp_trust_score"] = 50

    if "subtlety_score" in report:
        try:
            sub = int(report["subtlety_score"])
            report["subtlety_score"] = max(0, min(100, sub))
        except (ValueError, TypeError):
            report["subtlety_score"] = 50
    else:
        report["subtlety_score"] = 50
            
    # Validate schema
    schema = get_dpdp_schema()
    try:
        jsonschema.validate(instance=report, schema=schema)
    except jsonschema.ValidationError as e:
        raise ValueError(f"Schema validation failed: {e.message}")
        
    return report
