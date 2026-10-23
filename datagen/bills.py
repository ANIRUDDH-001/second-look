"""Build ground-truth bills from real household usage, and plant labelled faults.

Usage comes from Low Carbon London households (2011-2014), shifted forward by 678 weeks
so that weekdays line up and the periods fall in 2025-2026, where Ofgem caps are encoded.
Charges come from the same tariff engine the checker uses, so a clean bill is correct
by construction and a planted fault is the only thing wrong with it.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal

import pandas as pd

from secondlook.models import Bill, ChargeLine, VatLine, VerdictKind
from secondlook.tariff.engine import Tariff, expected_bill, money

SHIFT = timedelta(weeks=678)
SUPPLIERS = {"classic": "Northwind Energy", "modern": "Brightside Power", "statement": "Meridian Utilities"}

FAULTS = {
    "wrong_unit_rate": VerdictKind.DISCREPANCY,
    "stale_rate": VerdictKind.DISCREPANCY,
    "standing_day_count": VerdictKind.DISCREPANCY,
    "vat_after_oct_2026": VerdictKind.DISCREPANCY,
    "reading_mismatch": VerdictKind.DISCREPANCY,
    "estimated_reading": VerdictKind.INSUFFICIENT,
    "total_arithmetic": VerdictKind.DISCREPANCY,
    "unsupported_tariff": VerdictKind.UNSUPPORTED,
}


@dataclass
class Sample:
    bill_id: str
    household: str
    layout: str
    bill: Bill
    fault: str | None
    expected_verdict: VerdictKind
    extras: dict = field(default_factory=dict)  # distractor values printed on the bill


def shifted_daily(daily: pd.DataFrame, hh: str) -> pd.Series:
    s = daily[daily["hh"] == hh].set_index("date")["kwh"].sort_index()
    s.index = s.index + SHIFT
    return s


def periods(start: date, end: date, rng: random.Random) -> list[tuple[date, date]]:
    out, d = [], start
    while True:
        n = rng.randint(28, 35)
        e = d + timedelta(days=n - 1)
        if e > end:
            return out
        out.append((d, e))
        d = e + timedelta(days=1)


def clean_bill(tariff: Tariff, layout: str, hh: str, usage: pd.Series, start: date, end: date,
               rng: random.Random, opening_reading: int) -> tuple[Bill, dict]:
    days = pd.date_range(start, end)
    if usage.reindex(days).isna().mean() > 0.1:
        raise ValueError("too many missing days")
    kwh = float(usage.reindex(days).interpolate(limit_direction="both").sum())
    prev = opening_reading
    curr = prev + int(round(kwh))
    units = Decimal(curr - prev)
    exp = expected_bill(tariff, start, end, units)
    lines = []
    for s in exp.segments:
        lines.append(ChargeLine(kind="energy", start=s.start, end=s.end, quantity=s.kwh,
                                rate_p=s.quarter.unit_rate_p, amount_gbp=s.energy_gbp))
        lines.append(ChargeLine(kind="standing", start=s.start, end=s.end, quantity=Decimal(s.days),
                                rate_p=s.quarter.standing_p, amount_gbp=s.standing_gbp))
    vat = [VatLine(pct=p, on_gbp=b, amount_gbp=a) for p, (b, a) in sorted(exp.vat_by_pct.items(), reverse=True)]
    bill = Bill(
        supplier=SUPPLIERS[layout], account_ref=f"{rng.randint(10, 99)}-{rng.randint(100000, 999999)}",
        tariff_name="Standard Variable (price cap)", bill_date=end + timedelta(days=rng.randint(2, 6)),
        period_start=start, period_end=end, prev_reading=prev, prev_reading_type="A",
        curr_reading=curr, curr_reading_type="A", units_kwh=units, lines=lines,
        subtotal_gbp=exp.subtotal_gbp, vat=vat, total_gbp=exp.total_gbp,
    )
    extras = {
        "annual_cost_gbp": money(exp.total_gbp * Decimal(365) / ((end - start).days + 1) * Decimal(rng.uniform(0.9, 1.1))),
        "last_year_kwh": int(round(kwh * rng.uniform(0.8, 1.2))),
        "due_date": bill.bill_date + timedelta(days=14),
    }
    return bill, extras


def _rebuild_totals(bill: Bill, tariff: Tariff, vat_override: Decimal | None = None) -> None:
    """Recompute subtotal, VAT and total from the (possibly faulty) lines, as a supplier's system would."""
    bill.subtotal_gbp = sum((ln.amount_gbp for ln in bill.lines), Decimal(0))
    by_pct: dict[Decimal, Decimal] = {}
    for ln in bill.lines:
        pct = vat_override if vat_override is not None else tariff.quarter_for(ln.start).vat_pct
        by_pct[pct] = by_pct.get(pct, Decimal(0)) + ln.amount_gbp
    bill.vat = [VatLine(pct=p, on_gbp=b, amount_gbp=money(b * p / 100)) for p, b in sorted(by_pct.items(), reverse=True)]
    bill.total_gbp = bill.subtotal_gbp + sum((v.amount_gbp for v in bill.vat), Decimal(0))


def can_apply(fault: str, bill: Bill, tariff: Tariff) -> bool:
    crosses = tariff.quarter_for(bill.period_start) != tariff.quarter_for(bill.period_end)
    if fault == "stale_rate":
        return crosses
    if fault == "vat_after_oct_2026":
        return bill.period_start >= date(2026, 10, 1)
    return True


def apply_fault(fault: str, bill: Bill, tariff: Tariff, rng: random.Random) -> Bill:
    b = bill.model_copy(deep=True)
    energy = [ln for ln in b.lines if ln.kind == "energy"]
    standing = [ln for ln in b.lines if ln.kind == "standing"]
    if fault == "wrong_unit_rate":
        ln = rng.choice(energy)
        ln.rate_p = ln.rate_p + Decimal(rng.choice(["1.250", "2.000", "2.480", "3.150", "4.000"]))
        ln.amount_gbp = money(ln.quantity * ln.rate_p / 100)
        _rebuild_totals(b, tariff)
    elif fault == "stale_rate":
        q_old = tariff.quarter_for(b.period_start)
        units, days = b.units_kwh, Decimal((b.period_end - b.period_start).days + 1)
        b.lines = [
            ChargeLine(kind="energy", start=b.period_start, end=b.period_end, quantity=units,
                       rate_p=q_old.unit_rate_p, amount_gbp=money(units * q_old.unit_rate_p / 100)),
            ChargeLine(kind="standing", start=b.period_start, end=b.period_end, quantity=days,
                       rate_p=q_old.standing_p, amount_gbp=money(days * q_old.standing_p / 100)),
        ]
        _rebuild_totals(b, tariff, vat_override=q_old.vat_pct)
    elif fault == "standing_day_count":
        ln = rng.choice(standing)
        ln.quantity += rng.choice([1, 1, 2])
        ln.amount_gbp = money(ln.quantity * ln.rate_p / 100)
        _rebuild_totals(b, tariff)
    elif fault == "vat_after_oct_2026":
        _rebuild_totals(b, tariff, vat_override=Decimal(5))
    elif fault == "reading_mismatch":
        extra = Decimal(rng.randint(8, 60))
        b.units_kwh += extra
        energy[-1].quantity += extra
        energy[-1].amount_gbp = money(energy[-1].quantity * energy[-1].rate_p / 100)
        _rebuild_totals(b, tariff)
    elif fault == "estimated_reading":
        b.curr_reading_type = "E"
    elif fault == "total_arithmetic":
        b.total_gbp += Decimal(rng.choice(["10.00", "1.00", "9.00", "0.90", "18.00"]))
    elif fault == "unsupported_tariff":
        b.tariff_name = rng.choice(["Fixed Saver 12M v4", "Economy 7 Flex", "Fixed Online Nov26"])
        for ln in b.lines:
            ln.rate_p = (ln.rate_p * Decimal("0.92")).quantize(Decimal("0.001"))
            ln.amount_gbp = money(ln.quantity * ln.rate_p / 100)
        _rebuild_totals(b, tariff)
    else:
        raise ValueError(fault)
    return b
