"""
Unit tests for the vision OCR fallback (Engine 3).

Offline by design: the LLM call itself is exercised manually and by the demo
fixture; these tests cover the gating, rendering, and row-repair logic that
must hold whether or not a vision model is reachable.
"""

import pytest

from app.core.config import settings
from app.pipeline.vision_ocr import (
    ExtractedTransaction,
    StatementExtraction,
    extract_via_vision,
    render_pdf_pages,
    rows_from_extraction,
)

SCANNED_FIXTURE = "data/sample_4_scanned_statement.pdf"


class TestRowRepair:
    """Model output must be validated, never trusted."""

    def _extraction(self, **overrides):
        row = {
            "date": "2024-01-02",
            "description": "Apartment Rent",
            "amount": 28000.0,
            "type": "Debit",
        }
        row.update(overrides)
        return StatementExtraction(rows=[ExtractedTransaction(**row)])

    def test_valid_row_passes(self):
        rows = rows_from_extraction(self._extraction())
        assert rows == [
            {
                "date": "2024-01-02",
                "description": "Apartment Rent",
                "amount": 28000.0,
                "type": "Debit",
            }
        ]

    def test_direction_is_normalized(self):
        rows = rows_from_extraction(self._extraction(type="  credit "))
        assert rows[0]["type"] == "Credit"

    @pytest.mark.parametrize(
        "overrides",
        [
            {"type": "Transfer"},
            {"date": "   "},
            {"description": ""},
            {"amount": 0.0},
            {"amount": -50.0},
        ],
    )
    def test_unusable_rows_are_dropped_not_repaired(self, overrides):
        assert rows_from_extraction(self._extraction(**overrides)) == []


class TestRendering:
    def test_scanned_fixture_renders_to_png(self):
        from pathlib import Path

        images = render_pdf_pages(Path(SCANNED_FIXTURE).read_bytes(), max_pages=4)
        assert len(images) == 1
        assert images[0][:8] == b"\x89PNG\r\n\x1a\n"

    def test_max_pages_is_respected(self):
        from pathlib import Path

        images = render_pdf_pages(Path(SCANNED_FIXTURE).read_bytes(), max_pages=0)
        assert images == []


class TestGating:
    def test_disabled_by_default_returns_empty(self, monkeypatch):
        monkeypatch.setattr(settings, "vision_ocr_enabled", False)
        from pathlib import Path

        assert extract_via_vision(Path(SCANNED_FIXTURE).read_bytes()) == []

    def test_enabled_without_key_returns_empty(self, monkeypatch):
        monkeypatch.setattr(settings, "vision_ocr_enabled", True)
        monkeypatch.setattr(settings, "openai_api_key", "")
        from pathlib import Path

        assert extract_via_vision(Path(SCANNED_FIXTURE).read_bytes()) == []


class TestParserIntegration:
    def test_scanned_pdf_with_vision_disabled_raises_actionable_error(
        self, monkeypatch
    ):
        monkeypatch.setattr(settings, "vision_ocr_enabled", False)
        from app.pipeline.pdf_parser import parse_pdf_statement

        with pytest.raises(ValueError, match="VISION_OCR_ENABLED"):
            parse_pdf_statement(SCANNED_FIXTURE)
