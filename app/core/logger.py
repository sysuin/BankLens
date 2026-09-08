"""
Structured logging setup for BankLens.

Every module in the application imports get_logger() from here.
This ensures a consistent log format, timestamp, and level across
the entire codebase — making logs easy to read in local development
and easy to parse in a cloud environment (CloudWatch, etc.).

Every record carries the request-scoped context from app.core.context —
request_id, tenant and user — so a single grep on a request id returns the
whole story of one call across pipeline stages. Two output formats:

    text (default)  2024-03-01 10:00:00 | INFO | app.pipeline.rag | req=ab12 tenant=meridian | Loading knowledge base
    json            {"ts": "...", "level": "INFO", "logger": "app.pipeline.rag", "request_id": "ab12", ...}

Set LOG_FORMAT=json for log shippers.
"""

import json
import logging
import sys
from datetime import datetime, timezone

# The stream every new logger writes to. Defaults to stdout, which is right
# for the app (Docker and CloudWatch read stdout). The MCP server overrides
# this to stderr via route_logs_to_stderr(), because under stdio transport
# stdout IS the JSON-RPC channel — a log line printed there is a protocol
# corruption, not a log.
_LOG_STREAM = sys.stdout


class ContextFilter(logging.Filter):
    """Attach request_id / tenant / user from contextvars to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Imported lazily: context imports settings, and settings must not
        # import the logger at module load.
        from app.core import context

        record.request_id = context.current_request_id() or "-"
        record.tenant = context._tenant.get() or "-"
        record.user = context.current_user() or "-"
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line; safe for CloudWatch, Loki, Datadog."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "tenant": getattr(record, "tenant", "-"),
            "user": getattr(record, "user", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _formatter() -> logging.Formatter:
    from app.core.config import settings

    if settings.log_format.strip().lower() == "json":
        return JsonFormatter()
    return logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | "
        "req=%(request_id)s tenant=%(tenant)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def route_logs_to_stderr() -> None:
    """
    Send all BankLens logging to stderr — existing handlers and future ones.

    Call before serving MCP over stdio. Discovered the hard way: the first
    end-to-end client test worked only because the client skipped the
    unparseable "log line pretending to be JSON-RPC" frames.
    """
    global _LOG_STREAM
    _LOG_STREAM = sys.stderr
    root = logging.getLogger()
    for existing in [root] + [
        logging.getLogger(name) for name in logging.root.manager.loggerDict
    ]:
        for handler in existing.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setStream(sys.stderr)


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a configured logger for the given module name.

    The logger writes to stdout so that Docker and AWS CloudWatch can
    capture logs without any extra file-handler configuration.

    Args:
        name: The module name, typically passed as __name__ from the
              calling module (e.g. 'app.pipeline.rag').

    Returns:
        A Logger instance with a StreamHandler writing to stdout.
    """
    logger = logging.getLogger(name)

    # Guard: only add a handler if one has not been added already.
    # This prevents duplicate log lines when the module is imported
    # more than once (common in Streamlit's re-run model).
    if not logger.handlers:
        handler = logging.StreamHandler(_LOG_STREAM)
        handler.setFormatter(_formatter())
        handler.addFilter(ContextFilter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
