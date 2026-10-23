"""Effective-dated tariff engine for the Ofgem price cap.

All arithmetic is Decimal. Rounding rule (the same one the bill generator uses):
net rates to 3 dp of a penny, every charge line rounded half-up to the penny,
VAT computed per rate group and rounded half-up to the penny.
Usage that spans a rate change is split across the segments in proportion to days,
in whole kWh, with the last segment taking the remainder.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from pathlib import Path

import yaml

CAPS_FILE = Path(__file__).with_name("ofgem_caps.yaml")
PENNY = Decimal("0.01")
RATE_DP = Decimal("0.001")


def money(x: Decimal) -> Decimal:
    return x.quantize(PENNY, rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class Quarter:
    start: date
    end: date  # inclusive
    unit_rate_p: Decimal  # net of VAT
    standing_p: Decimal  # net of VAT
    vat_pct: Decimal
    source: str

    def covers(self, d: date) -> bool:
        return self.start <= d <= self.end


@dataclass(frozen=True)
class Tariff:
    tariff_id: str
    version: str  # content hash of the YAML, recorded in every verdict
    quarters: tuple[Quarter, ...]

    def quarter_for(self, d: date) -> Quarter | None:
        return next((q for q in self.quarters if q.covers(d)), None)

    @property
    def first_day(self) -> date:
        return self.quarters[0].start

    @property
    def last_day(self) -> date:
        return self.quarters[-1].end


def net_rate(published_incl_vat: Decimal, vat_pct: Decimal) -> Decimal:
    return (published_incl_vat / (1 + vat_pct / 100)).quantize(RATE_DP, rounding=ROUND_HALF_UP)


@lru_cache(maxsize=None)
def load_tariff(path: Path = CAPS_FILE) -> Tariff:
    raw = path.read_bytes()
    doc = yaml.safe_load(raw)
    quarters = []
    for q in doc["quarters"]:
        vat = Decimal(q["vat_pct"])
        quarters.append(Quarter(
            start=date.fromisoformat(str(q["start"])),
            end=date.fromisoformat(str(q["end"])),
            unit_rate_p=net_rate(Decimal(q["unit_rate_p_incl_vat"]), vat),
            standing_p=net_rate(Decimal(q["standing_p_per_day_incl_vat"]), vat),
            vat_pct=vat,
            source=q["source"],
        ))
    quarters.sort(key=lambda q: q.start)
    for a, b in zip(quarters, quarters[1:]):
        if a.end + timedelta(days=1) != b.start:
            raise ValueError(f"tariff gap or overlap between {a.end} and {b.start}")
    return Tariff(doc["tariff_id"], hashlib.sha256(raw).hexdigest()[:12], tuple(quarters))


@dataclass(frozen=True)
class Segment:
    quarter: Quarter
    start: date
    end: date
    days: int
    kwh: Decimal
    energy_gbp: Decimal
    standing_gbp: Decimal


@dataclass(frozen=True)
class Expected:
    segments: tuple[Segment, ...]
    subtotal_gbp: Decimal
    vat_by_pct: dict[Decimal, tuple[Decimal, Decimal]]  # pct -> (base, amount)
    total_gbp: Decimal


class OutsideTariff(ValueError):
    pass


def split_period(tariff: Tariff, start: date, end: date) -> list[tuple[Quarter, date, date]]:
    if end < start:
        raise ValueError("period ends before it starts")
    out, d = [], start
    while d <= end:
        q = tariff.quarter_for(d)
        if q is None:
            raise OutsideTariff(f"no published cap covers {d}")
        seg_end = min(q.end, end)
        out.append((q, d, seg_end))
        d = seg_end + timedelta(days=1)
    return out


def split_units(units: Decimal, day_counts: list[int]) -> list[Decimal]:
    total = sum(day_counts)
    shares = [(units * n / total).quantize(Decimal("1"), rounding=ROUND_HALF_UP) for n in day_counts[:-1]]
    return shares + [units - sum(shares, Decimal(0))]


def expected_bill(tariff: Tariff, start: date, end: date, units: Decimal) -> Expected:
    parts = split_period(tariff, start, end)
    days = [(e - s).days + 1 for _, s, e in parts]
    kwhs = split_units(units, days)
    segs = []
    for (q, s, e), n, k in zip(parts, days, kwhs):
        segs.append(Segment(q, s, e, n, k,
                            money(k * q.unit_rate_p / 100),
                            money(n * q.standing_p / 100)))
    subtotal = sum((s.energy_gbp + s.standing_gbp for s in segs), Decimal(0))
    vat: dict[Decimal, tuple[Decimal, Decimal]] = {}
    for s in segs:
        base = vat.get(s.quarter.vat_pct, (Decimal(0), Decimal(0)))[0] + s.energy_gbp + s.standing_gbp
        vat[s.quarter.vat_pct] = (base, Decimal(0))
    vat = {pct: (base, money(base * pct / 100)) for pct, (base, _) in vat.items()}
    total = subtotal + sum((a for _, a in vat.values()), Decimal(0))
    return Expected(tuple(segs), subtotal, vat, total)
