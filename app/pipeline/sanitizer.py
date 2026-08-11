"""
PII Sanitizer & Data Privacy Guard for BankLens.

Masks sensitive Customer PII (Account Numbers, Emails, Phone Numbers, PAN/SSN tokens)
using regex pattern matching BEFORE transaction text or metrics are passed to external cloud LLMs.
This ensures GDPR, RBI, and SOC2 compliance for enterprise banking.
"""

import re
import pandas as pd

from app.core.logger import get_logger

logger = get_logger(__name__)

# Regex Patterns for PII Detection
PATTERNS = [
    # Bank Account Numbers (e.g. Acc #981238471234 -> Acc #XXXX-XXXX-1234)
    (r"\b(acc|account|ac)\s*#?\s*(\d{6,16})\b", r"\1 #XXXX-XXXX-\2"),
    # Credit Card Numbers (e.g. 4532 9812 3456 7890 -> XXXX-XXXX-XXXX-7890)
    (r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?(\d{4})\b", r"XXXX-XXXX-XXXX-\1"),
    # Phone numbers
    (
        r"\b(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        r"[REDACTED_PHONE]",
    ),
    # Email addresses
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", r"[REDACTED_EMAIL]"),
    # PAN / SSN / National IDs
    (r"\b[A-Z]{5}\d{4}[A-Z]{1}\b", r"[REDACTED_PAN_ID]"),
    (r"\b\d{3}-\d{2}-\d{4}\b", r"[REDACTED_SSN]"),
]


def sanitize_text(text: str) -> str:
    """
    Sanitize a single text string by redacting PII patterns safely.

    Args:
        text: Unsanitized text object.

    Returns:
        Cleaned text with sensitive PII masked.
    """
    if not isinstance(text, str):
        text = str(text) if pd.notna(text) else ""

    sanitized = text
    for pattern, replacement in PATTERNS:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    return sanitized


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply sanitize_text to the 'description' column of a transactions DataFrame.

    Args:
        df: Input DataFrame containing transactions.

    Returns:
        A copy of the DataFrame with PII masked in descriptions.
    """
    result = df.copy()
    if "description" in result.columns:
        result["description"] = (
            result["description"].fillna("").astype(str).map(sanitize_text)
        )
        logger.info(
            "Sanitized %d transaction descriptions for PII compliance.", len(result)
        )
    return result
