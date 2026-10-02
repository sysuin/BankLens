"""
Unit tests for app.pipeline.sanitizer.

Tests verify that:
    - Account numbers are properly masked (e.g. Acc #9812384712 -> Acc #XXXX-XXXX-12)
    - Phone numbers and email addresses are redacted
    - PAN / SSN IDs are masked
    - sanitize_dataframe() processes the 'description' column without throwing errors
"""

import pandas as pd

from app.pipeline.sanitizer import sanitize_text, sanitize_dataframe


class TestSanitizer:
    """Tests for PII sanitization functions."""

    def test_account_number_masking(self):
        """Bank account numbers should be masked with XXXX-XXXX."""
        input_str = "Transfer to Acc #981238471234 for rent"
        output_str = sanitize_text(input_str)
        assert "981238471234" not in output_str
        assert "XXXX-XXXX" in output_str

    def test_email_redaction(self):
        """Email addresses should be replaced with [REDACTED_EMAIL]."""
        input_str = "Payment to user john.doe@example.com"
        output_str = sanitize_text(input_str)
        assert "john.doe@example.com" not in output_str
        assert "[REDACTED_EMAIL]" in output_str

    def test_pan_id_redaction(self):
        """PAN card format strings should be masked."""
        input_str = "Tax payment ABCDE1234F verified"
        output_str = sanitize_text(input_str)
        assert "ABCDE1234F" not in output_str
        assert "[REDACTED_PAN_ID]" in output_str

    def test_sanitize_dataframe(self):
        """sanitize_dataframe should clean descriptions in a pandas DataFrame."""
        df = pd.DataFrame(
            {
                "date": ["2024-03-01"],
                "description": ["Acc #9812384712 Transfer"],
                "amount": [5000.0],
                "type": ["Credit"],
            }
        )
        cleaned_df = sanitize_dataframe(df)
        assert "9812384712" not in cleaned_df["description"].iloc[0]


# ── Properties, not examples ──────────────────────────────────────────────────
#
# The account pattern once re-inserted the full number it had captured, and
# the example-based tests above still passed: their 10- and 12-digit numbers
# were masked by accident by the phone pattern. These check the property —
# the original number never survives — over every length and common spelling.

import pytest  # noqa: E402

ACCOUNT_PREFIXES = [
    "Acc #",
    "acc ",
    "Account ",
    "ACCOUNT: ",
    "A/c No. ",
    "a/c ",
    "Acct ",
    "AC ",
]


def _digits(n: int) -> str:
    return "".join(str((i * 7 + 3) % 10) for i in range(n))


@pytest.mark.parametrize("length", range(6, 21))
@pytest.mark.parametrize("prefix", ACCOUNT_PREFIXES)
def test_no_part_of_an_account_number_survives(prefix, length):
    number = _digits(length)
    out = sanitize_text(f"NEFT to {prefix}{number} rent")
    assert number not in out
    # No six consecutive digits of the original remain anywhere.
    assert not any(number[i : i + 6] in out for i in range(length - 5))
    assert out.endswith(f"#XXXX-XXXX-{number[-4:]} rent")


@pytest.mark.parametrize(
    "text, number",
    [
        ("call 98765 43210 now", "98765 43210"),
        ("UPI +91-9876543210", "9876543210"),
        ("UPI +91 98765 43210", "98765 43210"),
        ("mobile 09876543210", "9876543210"),
        ("ph 6123456789 ok", "6123456789"),
    ],
)
def test_indian_mobile_numbers_are_masked(text, number):
    out = sanitize_text(text)
    assert number not in out and number.replace(" ", "") not in out
    assert "[REDACTED_PHONE]" in out


@pytest.mark.parametrize(
    "text",
    [
        "Salary credit ACME Pvt Ltd",
        "2024-03-01 grocery 4,250.00",
        "ATM WDL 2000",
        "Rent March 2024",
    ],
)
def test_ordinary_descriptions_are_unchanged(text):
    assert sanitize_text(text) == text
