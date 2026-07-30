"""
Reusable security-headers middleware.
=====================================

Sets a conservative baseline of hardening headers on every response. Kept
separate from CORS so you can reason about (and test) each concern alone.

These are cheap, standard, and exactly the kind of thing a reviewer checks for
on a portfolio backend. Tune the CSP for your frontend if you serve HTML from
here (this API mostly serves JSON, so a strict default is fine).
"""
from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

_DEFAULT_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    "Cross-Origin-Opener-Policy": "same-origin",
}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, headers: dict | None = None, hsts: bool = False):
        super().__init__(app)
        self._headers = {**_DEFAULT_HEADERS, **(headers or {})}
        if hsts:
            self._headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        for k, v in self._headers.items():
            response.headers.setdefault(k, v)
        return response
