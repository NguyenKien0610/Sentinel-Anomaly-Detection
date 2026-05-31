from time import perf_counter
from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.logging import get_logger, request_id_context


logger = get_logger("sentinel.request")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        token = request_id_context.set(request_id)
        request.state.request_id = request_id
        started_at = perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((perf_counter() - started_at) * 1000, 3)
            logger.exception(
                "request_failed",
                extra={
                    "extra_fields": {
                        "method": request.method,
                        "path": request.url.path,
                        "duration_ms": duration_ms,
                        "client_host": request.client.host if request.client else None,
                    }
                },
            )
            request_id_context.reset(token)
            raise

        duration_ms = round((perf_counter() - started_at) * 1000, 3)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_completed",
            extra={
                "extra_fields": {
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                    "client_host": request.client.host if request.client else None,
                }
            },
        )
        request_id_context.reset(token)
        return response
