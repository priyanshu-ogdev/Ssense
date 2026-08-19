#!/usr/bin/env python3
"""
main.py – Production FastAPI Virtual SLM Server
Fully synchronized for Zero-Hop Execution, Model Auto-Downloading, 
O(1) Memory Coalescing, and FP8 VRAM Management.
"""

import os
import time
import json
import uuid
from typing import Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Telemetry & Rate Limiting
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi.errors import RateLimitExceeded

from contextlib import asynccontextmanager
from huggingface_hub import snapshot_download

# ── SOTA Ssense Core Modules ──
from security import (
    limiter,
    _rate_limit_exceeded_handler,
    verify_hmac_signature,
    check_model_extraction_attempt,
    sanitize_input_prompt,
    validate_and_repair_report,
    get_dpdp_schema
)
from memory_orchestrator import memory_orchestrator
from rag_engine import rag_engine
from engine import ProductionAsyncEngine

# Global Engine Reference
llm_engine: Optional[ProductionAsyncEngine] = None

# ═══════════════════════════════════════════════════════════════
# 0. MODEL AUTO-DOWNLOADER (Self-Healing Boot)
# ═══════════════════════════════════════════════════════════════
def ensure_models_exist():
    """Downloads Base Weights, LoRA Adapters, and RAG Safetensors if missing."""
    models_dir = Path(os.getenv("MODELS_DIR", "/app/models"))
    base_model_dir = models_dir / "base" / "Qwen2.5-7B-Instruct"
    ssense_repo_id = "PRiyanshu0-1/DPDP-SSense"

    print("🔍 [Boot] Verifying AI weights and indices...")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" # Speeds up downloads by 300%

    # 1. Base Model (Only downloads safetensors, ignores heavy .bin files)
    if not base_model_dir.exists() or not list(base_model_dir.glob("*.safetensors")):
        print(f"📥 [Boot] Downloading Qwen2.5-7B-Instruct Base Model to {base_model_dir}...")
        snapshot_download(
            repo_id="Qwen/Qwen2.5-7B-Instruct",
            local_dir=base_model_dir,
            ignore_patterns=["*.pt", "*.bin", "*.h5"]
        )

    # 2. Ssense Artifacts (LoRAs + RAG)
    audit_adapter = models_dir / "audit-model-final-adapter"
    if not audit_adapter.exists():
        print(f"📥 [Boot] Downloading SSense LoRAs & RAG Matrix from {ssense_repo_id}...")
        # Since the HF repo is structured exactly like our models folder:
        snapshot_download(
            repo_id=ssense_repo_id,
            local_dir=models_dir,
            allow_patterns=["models/*"],
            local_dir_use_symlinks=False
        )
        
        # HuggingFace drops them in /app/models/models/... so we move them up one level
        nested_models = models_dir / "models"
        if nested_models.exists():
            import shutil
            for item in nested_models.iterdir():
                shutil.move(str(item), str(models_dir / item.name))
            nested_models.rmdir()

    print("✅ [Boot] All AI weights verified.")
    return base_model_dir, models_dir / "audit-model-final-adapter", models_dir / "chatbot-model-final-adapter"

# ═══════════════════════════════════════════════════════════════
# 1. LIFECYCLE MANAGEMENT
# ═══════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_engine
    print("🚀 Booting Ssense Virtual SLM Server (Zero-Hop Architecture)...")
    
    # 1. Ensure Files Exist
    base_dir, audit_dir, chat_dir = ensure_models_exist()
    
    # 2. Initialize RAG Subsystem (Loads safely to RAM)
    await rag_engine.initialize()
    
    # 3. Initialize vLLM GPU Subsystem
    llm_engine = ProductionAsyncEngine(
        base_model_path=str(base_dir),
        audit_adapter_path=str(audit_dir),
        chatbot_adapter_path=str(chat_dir)
    )
    
    print("✅ Virtual SLM Server ready for High-Concurrency SSE Streaming.")
    yield
    print("🛑 Shutting down Ssense Virtual SLM Server...")

app = FastAPI(
    title="Ssense Virtual SLM Server",
    description="SOTA Zero-Hop AI Hosting for DPDP Act 2023 Enforcement",
    version="4.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_allowed_origins = [o.strip() for o in os.getenv("SSENSE_ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# ═══════════════════════════════════════════════════════════════
# 2. REQUEST MODELS
# ═══════════════════════════════════════════════════════════════
class AuditPolicyRequestModel(BaseModel):
    domain: str = Field(..., description="Target website domain")
    policyText: str = Field(..., description="Extracted privacy policy text")

class ChatRequestModel(BaseModel):
    domain: str = Field(..., description="Target website domain")
    userPrompt: str = Field(..., description="User chat question")

# ═══════════════════════════════════════════════════════════════
# 3. ENDPOINTS
# ═══════════════════════════════════════════════════════════════
@app.get("/health", tags=["Health & Telemetry"])
async def health_check():
    """Real-time diagnostic telemetry for Docker Compose & Orchestrators."""
    return {
        "status": "online",
        "backend": "zero-hop_vllm",
        "active_concurrent_jobs": memory_orchestrator.active_jobs_count,
        "max_queue_depth": memory_orchestrator.max_queue_depth,
        "rag_ready": rag_engine.is_ready,
        "timestamp": int(time.time())
    }

@app.post("/v1/audit", tags=["Inference"], dependencies=[Depends(verify_hmac_signature)])
async def audit_policy(request: Request, body: AuditPolicyRequestModel):
    """Executes deterministic JSON schema-constrained policy audits."""
    is_overloaded, q_depth = await memory_orchestrator.check_circuit_breaker()
    if is_overloaded:
        raise HTTPException(status_code=503, detail="Server queue saturated. Try again.")

    # 1. Sanitize & Cache Check
    clean_text = sanitize_input_prompt(body.policyText, is_audit_policy=True)
    cached_audit = await memory_orchestrator.get_cached_audit(clean_text)
    if cached_audit:
        return {"source": "memory_cache", "data": cached_audit}

    await check_model_extraction_attempt(request, clean_text)
    memory_orchestrator.increment_jobs()

    try:
        # 2. Format Prompt & Execute
        prompt = (
            f"<|im_start|>system\nYou are a strict DPDP Act 2023 Regulatory Auditor. "
            f"Analyze the policy and return ONLY valid JSON matching the schema.<|im_end|>\n"
            f"<|im_start|>user\nAudit Domain: {body.domain}\n\n[POLICY]\n{clean_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        req_id = str(uuid.uuid4())
        raw_json = await llm_engine.generate_audit(
            request_id=req_id,
            prompt=prompt,
            schema=get_dpdp_schema()
        )
        
        # 3. Validate, Repair & Cache
        final_report = validate_and_repair_report(raw_json)
        await memory_orchestrator.set_cached_audit(clean_text, final_report)
        
        return {"source": "inference", "data": final_report}
        
    finally:
        memory_orchestrator.decrement_jobs()


@app.post("/v1/chat/stream", tags=["Inference"], dependencies=[Depends(verify_hmac_signature)])
async def chat_with_copilot(request: Request, body: ChatRequestModel):
    """Executes RAG-augmented conversational SSE streaming with Request Coalescing."""
    is_overloaded, _ = await memory_orchestrator.check_circuit_breaker()
    if is_overloaded:
        raise HTTPException(status_code=503, detail="Server queue saturated.")

    clean_prompt = sanitize_input_prompt(body.userPrompt, is_audit_policy=False)
    await check_model_extraction_attempt(request, clean_prompt)

    # 1. O(1) Memory Coalescing Lease (Prevents duplicate GPU work)
    task_hash = memory_orchestrator.compute_sha256(f"chat::{clean_prompt}")
    is_leader, broadcaster = await memory_orchestrator.acquire_execution_lease(task_hash)

    # ── FOLLOWER: Hook into active generation stream (Zero VRAM impact) ──
    if not is_leader:
        async def coalesced_stream():
            queue = broadcaster.subscribe()
            while True:
                token = await queue.get()
                if token is None:
                    break
                yield f"data: {json.dumps({'event': 'token', 'data': token})}\n\n"
            yield f"data: {json.dumps({'event': 'done'})}\n\n"
            
        return StreamingResponse(coalesced_stream(), media_type="text/event-stream")

    # ── LEADER: Execute full RAG & GPU Generation ──
    memory_orchestrator.increment_jobs()
    
    # RAG Retrieval
    context_str, hits = await rag_engine.retrieve_context(clean_prompt, top_k=3)
    citations = [h["metadata"] for h in hits]

    prompt = (
        f"<|im_start|>system\nYou are the Ssense DPDP Co-Pilot. Base your answers STRICTLY on the retrieved context.<|im_end|>\n"
        f"<|im_start|>user\n{context_str}\n\nQuestion regarding {body.domain}: {clean_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    async def primary_stream():
        try:
            # Yield metadata first
            yield f"data: {json.dumps({'event': 'citations', 'data': citations})}\n\n"
            
            # Yield tokens
            req_id = str(uuid.uuid4())
            async for token in llm_engine.generate_chat_stream(req_id, prompt):
                await broadcaster.broadcast(token)
                yield f"data: {json.dumps({'event': 'token', 'data': token})}\n\n"
                
            yield f"data: {json.dumps({'event': 'done'})}\n\n"
        finally:
            await memory_orchestrator.cleanup_stream(task_hash, broadcaster)
            memory_orchestrator.decrement_jobs()

    return StreamingResponse(primary_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    # SOTA FIX: Must be exactly port 8000 and workers 1 for vLLM compatibility
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1, log_level="info")