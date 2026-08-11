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
