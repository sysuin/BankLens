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

# Regex Patterns for PII Detection, applied in order. A replacement may be a
# function of the match, so an account number keeps exactly its last four
# digits whatever its length.


def _mask_account(match: re.Match) -> str:
    return f"{match.group(1)} #XXXX-XXXX-{match.group(2)[-4:]}"


PATTERNS = [
    # Bank account numbers, any length from 6 digits (e.g. Acc #981238471234,
    # A/c No. 12345678 -> Acc #XXXX-XXXX-1234). The replacement used to be the
    # whole captured number, which re-inserted it: 6-9, 14-15 and 17+ digit
    # accounts came out in full, and 10-13 digits were masked only because the
    # phone pattern happened to match the leftover digits.
    (
        r"\b(acc|account|acct|a/c|ac)\b\.?\s*(?:no\.?|number)?\s*[#:.-]?\s*(\d{6,})\b",
        _mask_account,
    ),
    # Credit Card Numbers (e.g. 4532 9812 3456 7890 -> XXXX-XXXX-XXXX-7890)
    (r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?(\d{4})\b", r"XXXX-XXXX-XXXX-\1"),
    # Indian mobile numbers as they are usually written: 98765 43210,
    # +91-9876543210, 09876543210. Not inside a longer digit run, so UPI and
    # NEFT reference numbers are left alone.
    (r"(?<!\d)(?:\+91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)", r"[REDACTED_PHONE]"),
    # Phone numbers (North American shapes)
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
