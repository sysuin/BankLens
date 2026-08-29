"""
Structured logging setup for BankLens.

Every module in the application imports get_logger() from here.
This ensures a consistent log format, timestamp, and level across
the entire codebase — making logs easy to read in local development
and easy to parse in a cloud environment (CloudWatch, etc.).

Log format:
    2024-03-01 10:00:00 | INFO     | app.pipeline.rag | Loading knowledge base
"""

import logging
import sys

# The stream every new logger writes to. Defaults to stdout, which is right
# for the app (Docker and CloudWatch read stdout). The MCP server overrides
# this to stderr via route_logs_to_stderr(), because under stdio transport
# stdout IS the JSON-RPC channel — a log line printed there is a protocol
# corruption, not a log.
_LOG_STREAM = sys.stdout


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
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
