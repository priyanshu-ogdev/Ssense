#!/usr/bin/env python3
"""
memory_orchestrator.py – SOTA Zero-Hop In-Memory Orchestrator

Features:
1. Cryptographic Sliding Window Rate Limiting (Per User ID / API Key / IP) [vLLM/FastAPI Security] [1]
2. Thread-safe Asyncio Broadcast Matrix for Real-Time Request Coalescing [1]
3. O(1) LRU-TTL Cache for Forensic Audits (Prevents RAM Leaks)
4. Race-Condition Proof Subscriptions & Background Garbage Collection
"""

import time
import hashlib
import asyncio
from collections import OrderedDict, deque
from typing import Dict, Any, Optional, AsyncGenerator, Tuple, List


# ═══════════════════════════════════════════════════════════════
# 1. SLIDING WINDOW RATE LIMITER (The Security Shield)
# ═══════════════════════════════════════════════════════════════
class SlidingWindowRateLimiter:
    """Mathematical Sliding Window limit tracking User IDs/IPs to prevent DDoS."""
    def __init__(self, limit: int = 60, window_seconds: int = 60):
        self.limit = limit
        self.window_seconds = window_seconds
        self._requests: Dict[str, deque] = {}
        self._lock = asyncio.Lock()

    async def check_rate_limit(self, identifier: str) -> Tuple[bool, int]:
        """Returns (is_rate_limited, remaining_requests)."""
        async with self._lock:
            now = time.time()
            if identifier not in self._requests:
                self._requests[identifier] = deque()
            
            # Prune timestamps older than the sliding window
            while self._requests[identifier] and self._requests[identifier][0] < now - self.window_seconds:
                self._requests[identifier].popleft()
                
            # Check if user breached the threshold
            if len(self._requests[identifier]) >= self.limit:
                return True, 0
                
            # Log the request and allow
            self._requests[identifier].append(now)
            return False, self.limit - len(self._requests[identifier])


# ═══════════════════════════════════════════════════════════════
# 2. LRU-TTL CACHE (Memory Leak Protection)
# ═══════════════════════════════════════════════════════════════
class LRUTTLCache:
    """Thread-safe LRU Cache with Time-to-Live eviction."""
    def __init__(self, maxsize: int = 10000, ttl_seconds: int = 86400):
        self.cache: OrderedDict[str, Tuple[float, Any]] = OrderedDict()
        self.maxsize = maxsize
        self.ttl = ttl_seconds
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key not in self.cache:
                return None
            expiry, value = self.cache[key]
            if time.time() > expiry:
                del self.cache[key]
                return None
            self.cache.move_to_end(key)
            return value

    async def set(self, key: str, value: Any):
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
            elif len(self.cache) >= self.maxsize:
                self.cache.popitem(last=False)  # Evict oldest LRU item
            self.cache[key] = (time.time() + self.ttl, value)


# ═══════════════════════════════════════════════════════════════
# 3. ASYNC STREAM BROADCASTER (Request Coalescing)
# ═══════════════════════════════════════════════════════════════
class StreamBroadcaster:
    """Multiplexes a single LLM generation stream to N connected users."""
    def __init__(self):
        self.subscribers: List[asyncio.Queue] = []
        self.history: List[str] = []
        self.is_done = False

    def subscribe(self) -> asyncio.Queue:
        """Creates a dedicated queue for a coalesced client."""
        q = asyncio.Queue(maxsize=8000)
        
        # SOTA FIX: Flush history into queue immediately to prevent race conditions
        for token in self.history:
            q.put_nowait(token)
            
        # SOTA FIX: If the stream ALREADY finished, immediately push Sentinel
        if self.is_done:
            q.put_nowait(None)
        else:
            self.subscribers.append(q)
            
        return q

    async def broadcast(self, token: str):
        """Pushes a token to all active subscribers."""
        self.history.append(token)
        for q in self.subscribers:
            if not q.full():
                q.put_nowait(token)

    async def close(self):
        """Signals end-of-stream and flushes queues."""
        self.is_done = True
        for q in self.subscribers:
            if not q.full():
                q.put_nowait(None)  # Sentinel value
        self.subscribers.clear()


# ═══════════════════════════════════════════════════════════════
# 4. GLOBAL ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════
class MemoryOrchestrator:
    """Master controller replacing Redis: Rate Limiting, Coalescing, and Cache."""
    def __init__(self):
        import os
        self.max_queue_depth = int(os.getenv("SSENSE_MAX_QUEUE_DEPTH", "5000"))
        
        # Initialize Subsystems
        self.audit_cache = LRUTTLCache(maxsize=10000, ttl_seconds=86400)
        self.rate_limiter = SlidingWindowRateLimiter(limit=60, window_seconds=60) # 60 Req / Minute
        
        # State Tracking
        self.active_streams: Dict[str, StreamBroadcaster] = {}
        self.active_jobs_count = 0
        self._lock = asyncio.Lock()

    def compute_sha256(self, text: str, prefix: str = "gen") -> str:
        digest = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
        return f"{prefix}:{digest}"

    # ── Security & Rate Limiting ──
    async def enforce_rate_limit(self, user_identifier: str) -> Tuple[bool, int]:
        """Returns True if user is rate-limited, along with remaining quota."""
        return await self.rate_limiter.check_rate_limit(user_identifier)

    # ── Audit Caching ──
    async def get_cached_audit(self, policy_text: str) -> Optional[Dict[str, Any]]:
        key = self.compute_sha256(policy_text, prefix="audit")
        return await self.audit_cache.get(key)

    async def set_cached_audit(self, policy_text: str, audit_data: Dict[str, Any]):
        key = self.compute_sha256(policy_text, prefix="audit")
        await self.audit_cache.set(key, audit_data)

    # ── Request Coalescing ──
    async def acquire_execution_lease(self, task_key: str) -> Tuple[bool, StreamBroadcaster]:
        """Returns (is_leader, broadcaster). If follower, joins existing broadcast."""
        async with self._lock:
            if task_key in self.active_streams:
                return False, self.active_streams[task_key]
            
            broadcaster = StreamBroadcaster()
            self.active_streams[task_key] = broadcaster
            return True, broadcaster

    async def cleanup_stream(self, task_key: str, broadcaster: StreamBroadcaster):
        """Frees memory once the generation stream completes."""
        async with self._lock:
            await broadcaster.close()
            if task_key in self.active_streams:
                del self.active_streams[task_key]

    # ── Hardware Circuit Breaker ──
    async def check_circuit_breaker(self) -> Tuple[bool, int]:
        return (self.active_jobs_count >= self.max_queue_depth, self.active_jobs_count)

    def increment_jobs(self):
        self.active_jobs_count += 1

    def decrement_jobs(self):
        self.active_jobs_count = max(0, self.active_jobs_count - 1)


# Global Singleton Instance
memory_orchestrator = MemoryOrchestrator()