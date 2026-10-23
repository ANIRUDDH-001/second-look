"""Bill fields as transcribed from a bill, and the verdict the checker returns.

Money is Decimal pounds, rates are Decimal pence, quantities are Decimal kWh or days.
Every field is optional: a transcription can miss anything, and a missing field must
lead to "insufficient evidence", never to a guess.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChargeLine(BaseModel):
    kind: Literal["energy", "standing"]
    start: Optional[date] = None
    end: Optional[date] = None  # inclusive
    quantity: Optional[Decimal] = None  # kWh for energy, days for standing
    rate_p: Optional[Decimal] = None  # pence per kWh or per day, before VAT
    amount_gbp: Optional[Decimal] = None


class VatLine(BaseModel):
    pct: Optional[Decimal] = None
    on_gbp: Optional[Decimal] = None
    amount_gbp: Optional[Decimal] = None


class Bill(BaseModel):
    supplier: Optional[str] = None
    account_ref: Optional[str] = None
    tariff_name: Optional[str] = None
    bill_date: Optional[date] = None
    period_start: Optional[date] = None
    period_end: Optional[date] = None  # inclusive
    prev_reading: Optional[int] = None
    prev_reading_type: Optional[Literal["A", "E"]] = None  # Actual / Estimated
    curr_reading: Optional[int] = None
    curr_reading_type: Optional[Literal["A", "E"]] = None
    units_kwh: Optional[Decimal] = None
    lines: list[ChargeLine] = Field(default_factory=list)
    subtotal_gbp: Optional[Decimal] = None
    vat: list[VatLine] = Field(default_factory=list)
    total_gbp: Optional[Decimal] = None


class VerdictKind(str, Enum):
    RECONCILED = "reconciled"
    DISCREPANCY = "discrepancy"
    INSUFFICIENT = "insufficient_evidence"
    UNSUPPORTED = "unsupported_tariff"


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"  # inputs missing
    WARN = "warn"  # cannot be confirmed from the bill alone


class Finding(BaseModel):
    check: str
    component: str
    status: CheckStatus
    rule: str
    fields: list[str] = Field(default_factory=list)
    expected: Optional[str] = None
    billed: Optional[str] = None
    detail: Optional[str] = None


class Verdict(BaseModel):
    kind: VerdictKind
    headline: str
    findings: list[Finding]
    expected_total_gbp: Optional[Decimal] = None
    billed_total_gbp: Optional[Decimal] = None
    difference_gbp: Optional[Decimal] = None
    tariff_version: Optional[str] = None
    tariff_sources: list[str] = Field(default_factory=list)
