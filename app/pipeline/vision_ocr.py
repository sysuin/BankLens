"""
Engine 3: vision OCR fallback for scanned PDF statements.

Engines 1 and 2 read the PDF's text layer. A scanned statement has none —
pdfplumber sees pages of pixels and extracts nothing. This module renders
those pages to images and asks a vision model to read the transactions off
them, emitting the same canonical schema as the other engines.

Disabled by default, and that is a considered decision, not caution theatre:
the PII sanitizer masks account numbers, phone numbers and card numbers in
*text* before anything reaches an external model, but a page image cannot be
masked — enabling this sends the raw statement image, PII and all, to the
vision API. The text extracted *back* is still sanitized before profiling,
so exposure is limited to the OCR call itself. Set VISION_OCR_ENABLED=true
to accept that trade-off.
"""

import base64
import io

from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class ExtractedTransaction(BaseModel):
    """One transaction row read off a statement image."""

    date: str = Field(
        description="Transaction date as printed, e.g. '2024-03-01' or '01 Mar 2024'."
    )
    description: str = Field(description="Merchant name or memo text.")
    amount: float = Field(description="Transaction amount, always positive.")
    type: str = Field(description="'Credit' for money in, 'Debit' for money out.")


class StatementExtraction(BaseModel):
    """Everything readable from the statement images."""

    rows: list[ExtractedTransaction] = Field(
        default_factory=list,
        description="Every transaction visible, in the order printed.",
    )


VISION_SYSTEM_PROMPT = """\
You read scanned bank statement pages and transcribe the transaction table.

Rules:
  - Transcribe every transaction row you can read. Do not invent rows, and do
    not fill in values you cannot read — skip the row instead.
  - Amounts are always positive numbers; the direction goes in `type`.
  - Determine Credit vs Debit from an explicit Cr/Dr column if present,
    otherwise from separate deposit/withdrawal columns, otherwise from the
    running balance direction.
  - Ignore headers, footers, page numbers, marketing text and summary boxes.
"""


def render_pdf_pages(pdf_bytes: bytes, max_pages: int) -> list[bytes]:
    """Render the first max_pages of a PDF to PNG bytes via pypdfium2."""
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(pdf_bytes)
    try:
        images: list[bytes] = []
        for index in range(min(len(document), max_pages)):
            page = document[index]
            bitmap = page.render(scale=2.0)
            pil_image = bitmap.to_pil()
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            images.append(buffer.getvalue())
        return images
    finally:
        document.close()


def rows_from_extraction(extraction: StatementExtraction) -> list[dict]:
    """
    Validate model output into engine-canonical rows.

    Vision models transcribe imperfectly; a row with a nonsensical amount or
    direction is dropped rather than repaired, per the do-not-invent rule.
    """
    rows: list[dict] = []
    for item in extraction.rows:
        direction = item.type.strip().title()
        if direction not in ("Credit", "Debit"):
            continue
        if not item.date.strip() or not item.description.strip():
            continue
        if item.amount <= 0:
            continue
        rows.append(
            {
                "date": item.date.strip(),
                "description": item.description.strip(),
                "amount": float(item.amount),
                "type": direction,
            }
        )
    return rows


def extract_via_vision(pdf_bytes: bytes) -> list[dict]:
    """
    Read transactions off a scanned statement with the vision model.

    Returns [] when the feature is disabled, no API key is set, or the model
    finds nothing readable — the caller treats an empty list exactly as it
    treats the other engines finding nothing, so this can never turn a clear
    error into a confusing one.
    """
    if not settings.vision_ocr_enabled:
        logger.info(
            "Vision OCR disabled (VISION_OCR_ENABLED=false); skipping Engine 3."
        )
        return []
    if not settings.openai_api_key:
        logger.warning("Vision OCR enabled but no API key configured; skipping.")
        return []

    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_openai import ChatOpenAI

    images = render_pdf_pages(pdf_bytes, settings.vision_ocr_max_pages)
    if not images:
        return []

    logger.info(
        "Engine 3: sending %d rendered page(s) to the vision model.", len(images)
    )

    parser = PydanticOutputParser(pydantic_object=StatementExtraction)
    content: list[dict] = [
        {
            "type": "text",
            "text": "Transcribe every transaction from these statement pages.\n\n"
            + parser.get_format_instructions(),
        }
    ]
    for png in images:
        encoded = base64.b64encode(png).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}"},
            }
        )

    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.0,
        openai_api_key=settings.openai_api_key,
    )

    try:
        response = llm.invoke(
            [SystemMessage(content=VISION_SYSTEM_PROMPT), HumanMessage(content=content)]
        )
        extraction = parser.parse(response.content)
    except Exception as exc:  # noqa: BLE001 - fall through to the normal error path
        logger.warning(
            "Vision OCR failed (%s); falling back to normal error path.", exc
        )
        return []

    rows = rows_from_extraction(extraction)
    logger.info("Engine 3 extracted %d transaction(s) via vision OCR.", len(rows))
    return rows
