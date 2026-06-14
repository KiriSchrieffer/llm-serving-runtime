import json
import logging
from typing import Any


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for machine-readable request lifecycle tracing."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        for key in (
            "request_id",
            "event",
            "elapsed_ms",
            "tokens",
            "batch_size",
            "reason",
            "status_code",
            "error",
        ):
            value = getattr(record, key, None)
            if value is not None:
                entry[key] = value
        return json.dumps(entry, default=str)


class RequestLogger:
    """Structured logger that emits JSON events for request lifecycle stages.

    Each log event carries a `request_id` and `event` field so downstream consumers
    can correlate stages across a single request and compute derived metrics from logs.
    """

    def __init__(self, name: str = "llm_runtime.request") -> None:
        self._logger = logging.getLogger(name)

    def request_received(self, request_id: str, model: str, stream: bool = False) -> None:
        self._logger.info(
            "request received",
            extra={"request_id": request_id, "event": "request_received", "model": model,
                   "stream": stream},
        )

    def request_enqueued(self, request_id: str) -> None:
        self._logger.info(
            "request enqueued",
            extra={"request_id": request_id, "event": "request_enqueued"},
        )

    def request_rejected(
        self,
        request_id: str,
        reason: str,
        status_code: int,
    ) -> None:
        self._logger.info(
            "request rejected",
            extra={
                "request_id": request_id,
                "event": "request_rejected",
                "reason": reason,
                "status_code": status_code,
            },
        )

    def batch_formed(self, batch_size: int, request_ids: list[str]) -> None:
        self._logger.info(
            "batch formed for execution",
            extra={"event": "batch_formed", "batch_size": batch_size,
                   "request_ids": request_ids},
        )

    def token_generated(self, request_id: str, token_index: int) -> None:
        self._logger.debug(
            "token generated",
            extra={"request_id": request_id, "event": "token_generated",
                   "token_index": token_index},
        )

    def request_completed(
        self, request_id: str, tokens: int, elapsed_ms: float
    ) -> None:
        self._logger.info(
            "request completed",
            extra={"request_id": request_id, "event": "request_completed",
                   "tokens": tokens, "elapsed_ms": elapsed_ms},
        )

    def request_failed(self, request_id: str, error: str) -> None:
        self._logger.info(
            "request failed",
            extra={"request_id": request_id, "event": "request_failed", "error": error},
        )


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)
