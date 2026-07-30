"""
Logging Middleware
===================

Access logging for every request/response, correlated with the request ID
from RequestContextMiddleware. Must be added AFTER RequestContextMiddleware
so request_id is already set in context by the time this runs.
"""
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import time

from src.middleware.request_context import current_request_id

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = current_request_id()
        start_time = time.perf_counter()

        logger.info(f"📥 {request.method} {request.url.path} | request_id={rid}")

        response = await call_next(request)

        process_time = time.perf_counter() - start_time
        logger.info(
            f"📤 {response.status_code} - {process_time:.3f}s | request_id={rid}"
        )

        return response