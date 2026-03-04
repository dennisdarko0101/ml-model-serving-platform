"""FastAPI middleware for request logging and correlation IDs."""

from __future__ import annotations

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger()

CORRELATION_HEADER = "X-Correlation-ID"


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Add correlation ID to every request and log request/response details."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        correlation_id = request.headers.get(CORRELATION_HEADER, str(uuid.uuid4()))
        request.state.correlation_id = correlation_id

        start = time.monotonic()

        logger.info(
            "request_started",
            method=request.method,
            path=request.url.path,
            correlation_id=correlation_id,
        )

        try:
            response = await call_next(request)
        except Exception:
            logger.error(
                "request_failed",
                method=request.method,
                path=request.url.path,
                correlation_id=correlation_id,
                exc_info=True,
            )
            raise

        duration_ms = (time.monotonic() - start) * 1000

        response.headers[CORRELATION_HEADER] = correlation_id

        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(duration_ms, 2),
            correlation_id=correlation_id,
        )

        return response
