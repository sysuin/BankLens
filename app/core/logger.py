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
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
