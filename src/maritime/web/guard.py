"""Protections for running the map in public (`MARITIME_PUBLIC=1`), behind a Cloudflare Tunnel.

The app only reads, so the risks are load and scraping, not changes to data. In public mode:
- per-visitor rate limit on /api (token bucket, keyed by the visitor's IP as Cloudflare reports it);
- short server-side caching of the API responses, so many visitors cost about as much as one;
- a cap on live map connections (WebSocket), in total and per visitor;
- security headers on every response; the interactive API docs are switched off (see app.create_app).

The app listens on 127.0.0.1 only, so the CF-Connecting-IP header can only come from cloudflared (or an SSH tunnel).
"""

from __future__ import annotations

import functools
import os
import threading
import time
from collections import OrderedDict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

API_RATE_PER_MIN = 120      # sustained requests per minute per visitor; the map makes ~6 per minute when idle
API_BURST = 40              # requests allowed at once (page load makes ~6, a vessel click 1)
WS_MAX_TOTAL = 60
WS_MAX_PER_IP = 3
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), camera=(), microphone=()",
}


def is_public() -> bool:
    return os.environ.get("MARITIME_PUBLIC", "").strip().lower() in ("1", "true", "yes")


def client_ip(scope_headers, fallback: str | None) -> str:
    for k, v in scope_headers:
        if k.lower() in (b"cf-connecting-ip", "cf-connecting-ip"):
            return v.decode() if isinstance(v, bytes) else v
    return fallback or "unknown"


class TokenBucket:
    def __init__(self, rate_per_min: float = API_RATE_PER_MIN, burst: int = API_BURST, max_keys: int = 10_000):
        self.rate, self.burst, self.max_keys = rate_per_min / 60.0, burst, max_keys
        self._b: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            tokens, last = self._b.pop(key, (float(self.burst), now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            ok = tokens >= 1
            self._b[key] = (tokens - 1 if ok else tokens, now)
            while len(self._b) > self.max_keys:
                self._b.popitem(last=False)
            return ok


class Guard(BaseHTTPMiddleware):
    def __init__(self, app, public: bool, bucket: TokenBucket | None = None):
        super().__init__(app)
        self.public = public
        self.bucket = bucket or TokenBucket()

    async def dispatch(self, request, call_next):
        if self.public and request.url.path.startswith("/api/"):
            ip = client_ip(request.headers.items(), request.client.host if request.client else None)
            if not self.bucket.allow(ip):
                return JSONResponse({"detail": "too many requests, slow down"}, status_code=429,
                                    headers={"Retry-After": "10", **SECURITY_HEADERS})
        response = await call_next(request)
        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        if self.public and request.url.path.startswith("/static/"):
            response.headers.setdefault("Cache-Control", "public, max-age=3600")
        return response


def ttl_cache(seconds: float, enabled: bool, max_entries: int = 256):
    """Cache an endpoint's return value per argument set for `seconds` (only when `enabled`)."""
    def deco(fn):
        if not enabled:
            return fn
        store: OrderedDict = OrderedDict()
        lock = threading.Lock()

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.monotonic()
            with lock:
                hit = store.get(key)
                if hit and now - hit[0] < seconds:
                    return hit[1]
            value = fn(*args, **kwargs)
            with lock:
                store[key] = (now, value)
                store.move_to_end(key)
                while len(store) > max_entries:
                    store.popitem(last=False)
            return value
        return wrapper
    return deco


class WsLimiter:
    def __init__(self, total: int = WS_MAX_TOTAL, per_ip: int = WS_MAX_PER_IP):
        self.total, self.per_ip = total, per_ip
        self.by_ip: dict[str, int] = {}
        self._lock = threading.Lock()

    def acquire(self, ip: str) -> bool:
        with self._lock:
            if sum(self.by_ip.values()) >= self.total or self.by_ip.get(ip, 0) >= self.per_ip:
                return False
            self.by_ip[ip] = self.by_ip.get(ip, 0) + 1
            return True

    def release(self, ip: str) -> None:
        with self._lock:
            n = self.by_ip.get(ip, 0) - 1
            if n > 0:
                self.by_ip[ip] = n
            else:
                self.by_ip.pop(ip, None)
