"""
Unit tests for app.pipeline.pdf_parser.

Tests verify that:
    - parse_pdf_statement() handles invalid streams gracefully
    - parse_pdf_statement() parses sample PDF bank statements cleanly
    - Engine 2 classifies Credit vs Debit correctly (explicit token, then
      running-balance movement, then keywords)
"""

import os
import pytest
import pandas as pd
from app.pipeline.pdf_parser import (
    parse_pdf_statement,
    _build_text_row,
    _resolve_row_types,
)

SAMPLE_PDF_PATH = "data/sample_3_cashflow_stressed.pdf"


def _engine2_rows(lines: list[str]) -> list[dict]:
    """Run a list of single-line transaction blocks through Engine 2's row builder."""
    rows: list[dict] = []
    for line in lines:
        _build_text_row("2024-01-01", [line], rows)
    _resolve_row_types(rows)
    return rows


class TestPDFParser:
    """Tests for PDF statement parsing."""

    def test_invalid_pdf_raises_value_error(self):
        """Invalid PDF bytes should raise a ValueError."""
        invalid_bytes = b"Not a valid PDF file stream"
        with pytest.raises(ValueError) as excinfo:
            parse_pdf_statement(invalid_bytes)
        assert "Could not read PDF" in str(
            excinfo.value
        ) or "No transaction records" in str(excinfo.value)

    def test_sample_pdf_statement_parsing(self):
        """Valid PDF bank statement should parse into date, description, amount, and type columns."""
        if os.path.exists(SAMPLE_PDF_PATH):
            df = parse_pdf_statement(SAMPLE_PDF_PATH)
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert set(df.columns) == {"date", "description", "amount", "type"}
            assert (df["amount"] > 0).all()


class TestEngine2TypeInference:
    """Tests for Engine 2's Credit/Debit classification."""

    def test_explicit_type_token_wins_over_keywords(self):
        """A trailing 'Debit' token must beat the word 'Credit' in the merchant name."""
        rows = _engine2_rows(["Credit Card Minimum Payment 12000.00 Debit"])
        assert rows[0]["type"] == "Debit"

    def test_explicit_type_token_is_stripped_from_description(self):
        """The trailing type column must not be left in the description text."""
        rows = _engine2_rows(["Personal Loan EMI Debit 14500.00 Debit"])
        assert rows[0]["description"] == "Personal Loan EMI Debit"

    def test_explicit_credit_token_is_honoured(self):
        rows = _engine2_rows(["Freelance Consulting Inflow Credit 50000.00 Credit"])
        assert rows[0]["type"] == "Credit"
        assert rows[0]["amount"] == 50000.00

    def test_rising_balance_is_classified_as_credit(self):
        """With no type token, a balance that rose by the amount means a credit."""
        rows = _engine2_rows(
            [
                "OPENING ENTRY 1000.00 10000.00",
                "NEFT FROM ACME LTD 5000.00 15000.00",
            ]
        )
        assert rows[1]["type"] == "Credit"

    def test_falling_balance_is_classified_as_debit(self):
        """A balance that fell by the amount means a debit, whatever the wording."""
        rows = _engine2_rows(
            [
                "OPENING ENTRY 1000.00 10000.00",
                "UPI DEPOSIT BOX RENTAL 2000.00 8000.00",
            ]
        )
        assert rows[1]["type"] == "Debit"

    def test_credit_card_payment_falls_back_to_debit(self):
        """With no token and no balance, 'Credit Card' must not read as a credit."""
        rows = _engine2_rows(["CREDIT CARD BILL PAYMENT 9000.00"])
        assert rows[0]["type"] == "Debit"

    def test_salary_keyword_falls_back_to_credit(self):
        rows = _engine2_rows(["ACME CORP SALARY FOR JAN 85000.00"])
        assert rows[0]["type"] == "Credit"

    def test_long_decimal_is_not_truncated(self):
        """'12.345' must not be silently read as an amount of 12.34."""
        rows = _engine2_rows(["REF 12.345 TRANSFER FEE 250.00"])
        assert rows[0]["amount"] == 250.00

    def test_sample_pdf_engine2_matches_csv_ground_truth(self):
        """Engine 2 alone must reproduce the same Credit/Debit split as the CSV twin."""
        if not os.path.exists(SAMPLE_PDF_PATH):
            pytest.skip("sample PDF not available")

        import pdfplumber
        from app.pipeline.pdf_parser import _parse_via_text_blocks

        with pdfplumber.open(SAMPLE_PDF_PATH) as pdf:
            rows = _parse_via_text_blocks(pdf)

        engine2 = pd.DataFrame(rows)["type"].value_counts().to_dict()
        truth = (
            pd.read_csv("data/sample_3_cashflow_stressed.csv")["type"]
            .value_counts()
            .to_dict()
        )
        assert engine2 == truth
