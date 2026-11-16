"""Parse operator-reviewed, normalized cash-transfer statements from Angel One."""

from __future__ import annotations

import csv
import hashlib
import io
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

MAX_STATEMENT_BYTES = 1_000_000
MAX_STATEMENT_ROWS = 10_000
EXPECTED_COLUMNS = {"transaction_id", "timestamp", "direction", "amount", "status"}


@dataclass(frozen=True)
class CashFlowStatementEntry:
    reference: str
    amount: float
    occurred_at: str


def parse_angelone_cash_flow_statement(
    content: str,
) -> tuple[str, list[CashFlowStatementEntry]]:
    """Validate normalized completed transfers and return a content digest plus entries.

    The CSV must contain only transaction_id,timestamp,direction,amount,status columns.
    Timestamps must include an offset; direction is DEPOSIT or WITHDRAWAL, amount is a
    positive rupee value, and status is SUCCESS or COMPLETED.
    """
    if not isinstance(content, str) or not content:
        raise ValueError("Cash-flow statement is empty")
    try:
        encoded = content.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("Cash-flow statement is not valid UTF-8 text") from exc
    if len(encoded) > MAX_STATEMENT_BYTES:
        raise ValueError("Cash-flow statement exceeds the 1 MB limit")

    reader = csv.DictReader(io.StringIO(content.lstrip("\ufeff"), newline=""))
    try:
        if reader.fieldnames is None:
            raise ValueError("Cash-flow statement has no header row")
    except csv.Error as exc:
        raise ValueError("Cash-flow statement CSV is malformed") from exc
    headers = [header.strip().lower() for header in reader.fieldnames]
    if len(headers) != len(set(headers)) or set(headers) != EXPECTED_COLUMNS:
        raise ValueError(
            "Cash-flow statement columns must be transaction_id,timestamp,direction,amount,status"
        )
    reader.fieldnames = headers

    entries: list[CashFlowStatementEntry] = []
    seen_references: set[str] = set()
    try:
        for line_number, row in enumerate(reader, start=2):
            if line_number - 1 > MAX_STATEMENT_ROWS:
                raise ValueError("Cash-flow statement exceeds the 10,000-row limit")
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Cash-flow statement row {line_number} has the wrong number of columns")
            if not any((value or "").strip() for value in row.values()):
                continue

            reference = (row["transaction_id"] or "").strip()
            direction = (row["direction"] or "").strip().upper()
            amount_text = (row["amount"] or "").strip()
            status = (row["status"] or "").strip().upper()
            timestamp_text = (row["timestamp"] or "").strip()
            if not reference or len(reference) > 200 or not reference.isprintable():
                raise ValueError(f"Cash-flow statement row {line_number} has an invalid transaction ID")
            if reference in seen_references:
                raise ValueError(f"Cash-flow statement repeats transaction ID on row {line_number}")
            if direction not in {"DEPOSIT", "WITHDRAWAL"}:
                raise ValueError(f"Cash-flow statement row {line_number} has an unsupported direction")
            if status not in {"SUCCESS", "COMPLETED"}:
                raise ValueError(f"Cash-flow statement row {line_number} is not a completed transfer")
            try:
                amount = Decimal(amount_text)
            except InvalidOperation as exc:
                raise ValueError(f"Cash-flow statement row {line_number} has an invalid amount") from exc
            try:
                has_paise_precision = amount == amount.quantize(Decimal("0.01"))
            except InvalidOperation:
                has_paise_precision = False
            if not amount.is_finite() or amount <= 0 or not has_paise_precision:
                raise ValueError(f"Cash-flow statement row {line_number} amount must be positive rupees")
            try:
                timestamp = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"Cash-flow statement row {line_number} has an invalid timestamp") from exc
            if timestamp.tzinfo is None:
                raise ValueError(f"Cash-flow statement row {line_number} timestamp must include a timezone")
            signed_amount = float(amount if direction == "DEPOSIT" else -amount)
            if not math.isfinite(signed_amount):
                raise ValueError(f"Cash-flow statement row {line_number} amount is out of range")
            entries.append(CashFlowStatementEntry(
                reference=reference,
                amount=signed_amount,
                occurred_at=timestamp.astimezone(timezone.utc).isoformat(),
            ))
            seen_references.add(reference)
    except csv.Error as exc:
        raise ValueError("Cash-flow statement CSV is malformed") from exc

    if not entries:
        raise ValueError("Cash-flow statement contains no completed transfers")
    digest = hashlib.sha256(encoded).hexdigest()
    return digest, entries
