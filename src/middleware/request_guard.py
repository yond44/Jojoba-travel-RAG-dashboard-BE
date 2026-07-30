from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Deque, Dict, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.utils.log import logger

PRUNE_EVERY_REQUESTS = 500
CLIENT_IP_HEADERS = ("cf-connecting-ip", "x-real-ip", "x-forwarded-for")


class RequestGuardMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limits: Dict[str, Tuple[int, int]],
                 max_body_bytes: int):
        super().__init__(app)
        self.limits = dict(
            sorted(limits.items(), key=lambda item: len(item[0]), reverse=True))
        self.max_body_bytes = max_body_bytes
        self.hits: Dict[str, Deque[float]] = defaultdict(deque)
        self.request_counter = 0

    # ---------- Identifikasi klien ----------
    def _client_ip(self, request: Request) -> str:
        for header in CLIENT_IP_HEADERS:
            value = request.headers.get(header)
            if value:
                return value.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    # ---------- Pemilihan batas per jalur ----------
    def _limit_for_path(self, path: str) -> Tuple[str, int, int] | None:
        for prefix, (max_requests, window_seconds) in self.limits.items():
            if path.startswith(prefix):
                return prefix, max_requests, window_seconds
        return None

    # ---------- Jendela geser ----------
    def _register_hit(self, bucket_key: str, max_requests: int,
                      window_seconds: int) -> bool:
        now = time.monotonic()
        window_start = now - window_seconds
        timestamps = self.hits[bucket_key]
        while timestamps and timestamps[0] < window_start:
            timestamps.popleft()
        if len(timestamps) >= max_requests:
            return False
        timestamps.append(now)
        return True

    def _prune_idle_buckets(self) -> None:
        longest_window = max(window for _, window in self.limits.values())
        cutoff = time.monotonic() - longest_window
        idle_keys = [key for key, stamps in self.hits.items()
                     if not stamps or stamps[-1] < cutoff]
        for key in idle_keys:
            del self.hits[key]

    # ---------- Pintu masuk ----------
    async def dispatch(self, request: Request, call_next) -> Response:
        matched = self._limit_for_path(request.url.path)
        if matched is None:
            return await call_next(request)

        prefix, max_requests, window_seconds = matched

        declared_length = request.headers.get("content-length")
        if declared_length:
            try:
                body_bytes = int(declared_length)
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid Content-Length header"})
            if body_bytes > self.max_body_bytes:
                logger.warning("Body ditolak: %d byte pada %s",
                               body_bytes, request.url.path)
                return JSONResponse(
                    status_code=413,
                    content={"error": "Request body too large",
                             "max_bytes": self.max_body_bytes})

        self.request_counter += 1
        if self.request_counter % PRUNE_EVERY_REQUESTS == 0:
            self._prune_idle_buckets()

        bucket_key = f"{prefix}|{self._client_ip(request)}"
        if not self._register_hit(bucket_key, max_requests, window_seconds):
            logger.warning("Rate limit terlampaui: %s", bucket_key)
            return JSONResponse(
                status_code=429,
                content={"error": "Too many requests",
                         "limit": max_requests,
                         "window_seconds": window_seconds},
                headers={"Retry-After": str(window_seconds)})

        return await call_next(request)
