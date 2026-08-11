"""
Universal Multi-Engine PDF Bank Statement Parser for BankLens.

Ingests bank statements from any commercial bank (ICICI, HDFC, SBI, Axis, Kotak, Chase, Citi) using:
    - Engine 1: Table Grid Parser (handles 4-col, 5-col, and 6-col DEPOSITS vs WITHDRAWALS layouts)
    - Engine 2: Multi-line Date-Block Text Parser (handles borderless PDFs and complex multi-line UPI/NEFT memos)
"""

import io
import re
import pandas as pd
import pdfplumber

from app.core.logger import get_logger

logger = get_logger(__name__)

DATE_RE = re.compile(r"\b(\d{2}[-/\.]\d{2}[-/\.]\d{4}|\d{4}[-/\.]\d{2}[-/\.]\d{2})\b")

# Monetary amounts: 1,234.56 / 50000.00 — anchored so that a longer decimal
# such as "12.345" is not silently truncated to "12.34".
AMOUNT_RE = re.compile(
    r"(?<![\d.,])\d{1,3}(?:,\d{2,3})*\.\d{2}(?![\d])|(?<![\d.,])\d+\.\d{2}(?![\d])"
)

# Many statements print an explicit Credit/Debit (or Cr/Dr) token as the final
# column of the row. When present this is authoritative and must win over any
# keyword guess made from the merchant name.
TYPE_TOKEN_RE = re.compile(r"\s(credit|debit|cr|dr)\.?\s*$", re.IGNORECASE)

# Phrases that name a *product* rather than describe an inflow. These are
# checked first so that "Credit Card Minimum Payment" is not read as a credit
# merely because the word "credit" appears in it.
DEBIT_OVERRIDE_KEYWORDS = (
    "credit card",
    "card payment",
    "card bill",
    "loan emi",
    "emi debit",
)

# Conservative inflow markers. Deliberately excludes the bare tokens "credit"
# and "neft-", both of which appear just as often on outgoing rows.
CREDIT_KEYWORDS = (
    "salary",
    "inflow",
    "credited",
    "by clearing",
    "deposit",
    "interest earned",
    "refund",
    "cashback",
    "dividend",
    "reversal",
    "neft-cr",
    "imps-cr",
    "upi-cr",
    "bonus",
)


def _infer_type_from_keywords(description: str) -> str:
    """Fallback Credit/Debit inference from the description text alone."""
    text = description.lower()

    if any(kw in text for kw in DEBIT_OVERRIDE_KEYWORDS):
        return "Debit"
    if any(kw in text for kw in CREDIT_KEYWORDS):
        return "Credit"
    return "Debit"


def _parse_via_tables(pdf) -> list[dict]:
    """Engine 1: Extract rows using pdfplumber table structure."""
    rows = []
    active_headers = None

    for page in pdf.pages:
        tables = page.extract_tables()
        for table in tables:
            if not table or len(table) < 1:
                continue

            first_row_cells = [
                str(cell).strip().lower() for cell in table[0] if cell is not None
            ]
            first_row_str = " ".join(first_row_cells)

            is_trans_header = ("date" in first_row_str) and any(
                kw in first_row_str
                for kw in [
                    "particular",
                    "desc",
                    "amount",
                    "deposit",
                    "withdrawal",
                    "narrative",
                    "memo",
                    "chq",
                    "cheque",
                    "balance",
                ]
            )

            if is_trans_header:
                active_headers = [
                    str(cell).strip().lower() if cell is not None else ""
                    for cell in table[0]
                ]
                data_rows = table[1:]
            elif active_headers is not None:
                data_rows = table
            else:
                continue

            for row in data_rows:
                if not row or len(row) < 3:
                    continue

                date_cell = str(row[0]).strip() if row[0] is not None else ""
                if not DATE_RE.search(date_cell):
                    continue

                row_dict = {}
                for idx, cell in enumerate(row):
                    if idx < len(active_headers):
                        row_dict[active_headers[idx]] = (
                            str(cell).strip() if cell is not None else ""
                        )
                rows.append(row_dict)

    return rows


def _parse_via_text_blocks(pdf) -> list[dict]:
    """Engine 2: Extract multi-line date-block transaction text (ideal for ICICI / HDFC multi-line UPI PDF layouts)."""
    rows = []

    for page in pdf.pages:
        text = page.extract_text() or ""
        lines = text.split("\n")

        curr_date = None
        curr_lines = []

        for line in lines:
            line_s = line.strip()
            if not line_s:
                continue

            match = DATE_RE.search(line_s)
            if match and line_s.startswith(match.group(1)):
                if curr_date and curr_lines:
                    _build_text_row(curr_date, curr_lines, rows)
                curr_date = match.group(1)
                curr_lines = [line_s[len(curr_date) :].strip()]
            elif curr_date:
                curr_lines.append(line_s)

        if curr_date and curr_lines:
            _build_text_row(curr_date, curr_lines, rows)

    _resolve_row_types(rows)
    return rows


def _build_text_row(
    date_str: str, lines_list: list[str], output_rows: list[dict]
) -> None:
    """
    Process a single multi-line transaction text block.

    Sets 'type' when the row carries an explicit Credit/Debit token, and
    records the trailing running balance (when one is present) as '_balance'
    so that _parse_via_text_blocks can resolve the remaining rows by balance
    movement. Rows left with type=None are resolved in that later pass.
    """
    full_text = " ".join(lines_list)

    if "b/f" in full_text.lower() or (
        "total" in full_text.lower() and len(lines_list) < 2
    ):
        return

    # An explicit trailing Credit/Debit token is authoritative — capture it and
    # strip it so it does not pollute the description.
    explicit_type = None
    token_match = TYPE_TOKEN_RE.search(full_text)
    if token_match:
        token = token_match.group(1).lower()
        explicit_type = "Credit" if token in ("credit", "cr") else "Debit"
        full_text = full_text[: token_match.start()]

    nums = AMOUNT_RE.findall(full_text)
    if not nums:
        return

    # In bank text layouts the trailing number is the running balance and the
    # one before it is the transaction amount. With a single number there is no
    # balance column, so that number is the amount.
    if len(nums) >= 2:
        amount = float(nums[-2].replace(",", ""))
        balance = float(nums[-1].replace(",", ""))
    else:
        amount = float(nums[-1].replace(",", ""))
        balance = None

    # Clean description string
    desc = full_text
    for num in nums:
        desc = desc.replace(num, "")
    desc = re.sub(r"\s+", " ", desc).strip()

    if amount > 0 and len(desc) >= 2:
        output_rows.append(
            {
                "date": date_str,
                "description": desc[:120],
                "amount": amount,
                "type": explicit_type,
                "_balance": balance,
            }
        )


def _resolve_row_types(rows: list[dict]) -> None:
    """
    Fill in 'type' for rows that carried no explicit Credit/Debit token.

    Preference order:
        1. Running-balance movement — if the balance rose by the transaction
           amount, the row is a credit; if it fell, a debit. This is exact.
        2. Keyword inference from the description, as a last resort.
    """
    prev_balance = None

    for row in rows:
        balance = row.get("_balance")

        if row["type"] is None:
            resolved = None

            if balance is not None and prev_balance is not None:
                delta = balance - prev_balance
                # Guard against unrelated numbers being misread as a balance:
                # only trust the delta when it actually matches the amount.
                if abs(abs(delta) - row["amount"]) <= 0.01:
                    resolved = "Credit" if delta > 0 else "Debit"

            row["type"] = resolved or _infer_type_from_keywords(row["description"])

        if balance is not None:
            prev_balance = balance


def parse_pdf_statement(file_or_bytes) -> pd.DataFrame:
    """
    Extract and normalize transactions from any PDF bank statement.

    Args:
        file_or_bytes: File path (str), bytes, BytesIO, or Streamlit UploadedFile.

    Returns:
        A pandas DataFrame with normalized columns: date, description, amount, type.
    """
    # Handle Streamlit UploadedFile, bytes, BytesIO, or file path
    if hasattr(file_or_bytes, "getvalue"):
        pdf_file = io.BytesIO(file_or_bytes.getvalue())
    elif isinstance(file_or_bytes, bytes):
        pdf_file = io.BytesIO(file_or_bytes)
    elif hasattr(file_or_bytes, "read"):
        pdf_file = io.BytesIO(file_or_bytes.read())
    else:
        pdf_file = file_or_bytes

    try:
        with pdfplumber.open(pdf_file) as pdf:
            # Run Engine 1 (Table Extractor)
            table_rows = _parse_via_tables(pdf)

            # Run Engine 2 (Text Block Extractor)
            text_rows = _parse_via_text_blocks(pdf)

            # Choose the engine that extracted more complete records
            if len(text_rows) > len(table_rows) and len(text_rows) >= 5:
                logger.info(
                    "Engine 2 (Text Block Parser) extracted %d transactions (vs %d in Engine 1). Using Engine 2.",
                    len(text_rows),
                    len(table_rows),
                )
                rows = text_rows
                is_text_engine = True
            else:
                logger.info(
                    "Engine 1 (Table Parser) extracted %d transactions. Using Engine 1.",
                    len(table_rows),
                )
                rows = table_rows
                is_text_engine = False

    except Exception as e:
        logger.error("Failed to parse PDF statement: %s", e)
        raise ValueError(f"Could not read PDF bank statement. Error: {e}")

    if not rows:
        raise ValueError("No transaction records found in the uploaded PDF statement.")

    df = pd.DataFrame(rows)

    if is_text_engine:
        # Engine 2 already produces canonical date, description, amount, type.
        # '_balance' is internal bookkeeping and is not part of the output.
        df = df.drop(columns=["_balance"], errors="ignore")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        df["type"] = df["type"].astype(str)
    else:
        df.columns = [str(col).strip().lower() for col in df.columns]

        deposit_col = next(
            (
                c
                for c in df.columns
                if ("deposit" in c or "credit" in c or c == "cr")
                and "dr/cr" not in c
                and "cr/dr" not in c
            ),
            None,
        )
        withdrawal_col = next(
            (
                c
                for c in df.columns
                if ("withdrawal" in c or "debit" in c or c == "dr")
                and "dr/cr" not in c
                and "cr/dr" not in c
            ),
            None,
        )
        part_col = next(
            (
                c
                for c in df.columns
                if "particular" in c or "desc" in c or "narrative" in c or "memo" in c
            ),
            None,
        )
        date_col = next((c for c in df.columns if "date" in c or "time" in c), None)

        if deposit_col and withdrawal_col and "amount" not in df.columns:
            logger.info(
                "Detected dual column bank format (Deposits: '%s', Withdrawals: '%s'). Normalizing...",
                deposit_col,
                withdrawal_col,
            )
            normalized_rows = []
            for _, row in df.iterrows():
                d_val = (
                    str(row.get(deposit_col, ""))
                    .replace(",", "")
                    .replace("₹", "")
                    .strip()
                )
                w_val = (
                    str(row.get(withdrawal_col, ""))
                    .replace(",", "")
                    .replace("₹", "")
                    .strip()
                )

                d_num = pd.to_numeric(d_val, errors="coerce")
                w_num = pd.to_numeric(w_val, errors="coerce")

                mode_val = (
                    str(row.get("mode", "")).strip() if "mode" in df.columns else ""
                )
                part_val = str(row.get(part_col, "")).strip() if part_col else ""
                desc_val = f"{mode_val} {part_val}".strip()
                date_val = str(row.get(date_col, "")).strip() if date_col else ""

                if "total" in desc_val.lower() or "b/f" in desc_val.lower():
                    continue

                if pd.notna(d_num) and d_num > 0:
                    normalized_rows.append(
                        {
                            "date": date_val,
                            "description": desc_val,
                            "amount": float(d_num),
                            "type": "Credit",
                        }
                    )
                elif pd.notna(w_num) and w_num > 0:
                    normalized_rows.append(
                        {
                            "date": date_val,
                            "description": desc_val,
                            "amount": float(w_num),
                            "type": "Debit",
                        }
                    )

            df = pd.DataFrame(normalized_rows)

        col_mapping = {}
        for col in df.columns:
            if "date" in col or "time" in col:
                col_mapping[col] = "date"
            elif (
                "desc" in col
                or "particular" in col
                or "narrative" in col
                or "memo" in col
            ):
                col_mapping[col] = "description"
            elif "amount" in col or "val" in col or "sum" in col:
                col_mapping[col] = "amount"
            elif "type" in col or "dr/cr" in col or "cr/dr" in col:
                col_mapping[col] = "type"

        df = df.rename(columns=col_mapping)

        df["amount"] = (
            df["amount"]
            .astype(str)
            .str.replace(",", "")
            .str.replace("₹", "")
            .str.replace("$", "")
        )
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

        df["type"] = (
            df["type"]
            .astype(str)
            .apply(
                lambda t: (
                    "Credit"
                    if "cr" in str(t).lower()
                    or "credit" in str(t).lower()
                    or "deposit" in str(t).lower()
                    else "Debit"
                )
            )
        )

    # Clean data types
    df["date"] = df["date"].astype(str)
    df["description"] = df["description"].astype(str)

    # Filter out empty/invalid header rows or summary lines
    df = df[
        (df["description"].str.strip() != "")
        & (df["amount"] > 0)
        & (~df["description"].str.lower().str.contains("total"))
        & (~df["description"].str.lower().str.contains("description"))
        & (~df["date"].str.lower().str.contains("date"))
    ].copy()

    logger.info("Successfully parsed %d transactions from PDF.", len(df))
    return df[["date", "description", "amount", "type"]]
