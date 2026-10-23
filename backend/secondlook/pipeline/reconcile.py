"""Stages 2 and 3: cross-field consistency checks, then reconciliation against the tariff.

Pure functions of (Bill, Tariff). No model is involved, so the same bill always yields
the same verdict. Precedence: Unsupported tariff > Discrepancy > Insufficient evidence > Reconciled.
"""
from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from secondlook.models import Bill, CheckStatus, Finding, Verdict, VerdictKind
from secondlook.tariff.engine import OutsideTariff, Tariff, expected_bill, money

SUPPORTED_TARIFF_NAMES = ("standard variable", "default tariff", "price cap")
LINE_TOLERANCE = Decimal("0.01")  # one penny per line for rounding
RATE_TOLERANCE = Decimal("0.0005")  # pence


def _gbp(x: Decimal | None) -> str | None:
    return None if x is None else f"£{x:,.2f}"


def _days(start: date, end: date) -> int:
    return (end - start).days + 1


class _Checks:
    def __init__(self) -> None:
        self.findings: list[Finding] = []

    def add(self, check, component, ok, rule, fields, expected=None, billed=None, detail=None):
        status = CheckStatus.PASS if ok else CheckStatus.FAIL
        if ok is None:
            status = CheckStatus.SKIP
        self.findings.append(Finding(check=check, component=component, status=status, rule=rule,
                                     fields=fields, expected=expected, billed=billed, detail=detail))

    def warn(self, check, component, rule, fields, detail):
        self.findings.append(Finding(check=check, component=component, status=CheckStatus.WARN,
                                     rule=rule, fields=fields, detail=detail))


def _supported(bill: Bill) -> str | None:
    name = (bill.tariff_name or "").lower()
    if not name:
        return None  # unknown is an evidence problem, not an unsupported tariff
    if not any(s in name for s in SUPPORTED_TARIFF_NAMES):
        return f"Tariff '{bill.tariff_name}' is not the Ofgem price-cap default tariff."
    if any(w in name for w in ("economy", "e7", "fixed", "tracker", "agile", "time of use")):
        return f"Tariff '{bill.tariff_name}' is not single-rate price-cap pricing."
    return None


def consistency(bill: Bill, c: _Checks) -> None:
    """Checks that use only numbers printed on the bill."""
    if bill.prev_reading is not None and bill.curr_reading is not None and bill.units_kwh is not None:
        advance = Decimal(bill.curr_reading - bill.prev_reading)
        c.add("meter_advance", "units", advance == bill.units_kwh,
              "current reading − previous reading = units billed",
              ["prev_reading", "curr_reading", "units_kwh"], f"{advance} kWh", f"{bill.units_kwh} kWh")
    else:
        c.add("meter_advance", "units", None, "current reading − previous reading = units billed",
              ["prev_reading", "curr_reading", "units_kwh"], detail="readings or units not found")

    energy = [ln for ln in bill.lines if ln.kind == "energy"]
    standing = [ln for ln in bill.lines if ln.kind == "standing"]
    if energy and bill.units_kwh is not None and all(ln.quantity is not None for ln in energy):
        q = sum((ln.quantity for ln in energy), Decimal(0))
        c.add("energy_quantity", "units", q == bill.units_kwh, "energy line quantities sum to units billed",
              ["lines", "units_kwh"], f"{bill.units_kwh} kWh", f"{q} kWh")

    if bill.period_start and bill.period_end and standing and all(ln.quantity is not None for ln in standing):
        n = _days(bill.period_start, bill.period_end)
        q = sum((ln.quantity for ln in standing), Decimal(0))
        c.add("standing_days", "standing charge", q == n,
              "standing-charge days = days in the billing period",
              ["period_start", "period_end", "lines"], f"{n} days", f"{q} days")

    for i, ln in enumerate(bill.lines):
        if ln.quantity is None or ln.rate_p is None or ln.amount_gbp is None:
            c.add(f"line_arith_{i}", ln.kind, None, "quantity × rate = amount", [f"lines[{i}]"],
                  detail="line incomplete")
            continue
        exp = money(ln.quantity * ln.rate_p / 100)
        c.add(f"line_arith_{i}", "standing charge" if ln.kind == "standing" else "energy",
              abs(exp - ln.amount_gbp) <= LINE_TOLERANCE, "quantity × rate = amount",
              [f"lines[{i}]"], _gbp(exp), _gbp(ln.amount_gbp))

    amounts = [ln.amount_gbp for ln in bill.lines]
    if bill.subtotal_gbp is not None and amounts and None not in amounts:
        s = sum(amounts, Decimal(0))
        c.add("subtotal", "total", abs(s - bill.subtotal_gbp) <= LINE_TOLERANCE,
              "charge lines sum to the subtotal", ["lines", "subtotal_gbp"], _gbp(s), _gbp(bill.subtotal_gbp))

    vat_amounts = [v.amount_gbp for v in bill.vat]
    if bill.subtotal_gbp is not None and bill.total_gbp is not None and None not in vat_amounts:
        t = bill.subtotal_gbp + sum(vat_amounts, Decimal(0))
        c.add("total", "total", abs(t - bill.total_gbp) <= LINE_TOLERANCE,
              "subtotal + VAT = total", ["subtotal_gbp", "vat", "total_gbp"], _gbp(t), _gbp(bill.total_gbp))


def tariff_checks(bill: Bill, tariff: Tariff, c: _Checks) -> None:
    """Checks that compare the bill with the published cap."""
    for i, ln in enumerate(bill.lines):
        if ln.start is None or ln.end is None or ln.rate_p is None:
            continue
        q0, q1 = tariff.quarter_for(ln.start), tariff.quarter_for(ln.end)
        comp = "standing charge" if ln.kind == "standing" else "unit rate"
        if q0 is None or q1 is None:
            continue  # handled by the period coverage check
        if q0 != q1:
            c.add(f"line_rate_{i}", comp, False, "each charge line uses one quarter's rate",
                  [f"lines[{i}]"], f"split at {q1.start.isoformat()}",
                  f"one rate {ln.rate_p}p for {ln.start}..{ln.end}",
                  "The cap changed inside this line, but one rate was applied to all of it.")
            continue
        cap = q0.unit_rate_p if ln.kind == "energy" else q0.standing_p
        c.add(f"line_rate_{i}", comp, abs(ln.rate_p - cap) <= RATE_TOLERANCE,
              f"rate = Ofgem cap for {q0.start:%b %Y} quarter (net of VAT)",
              [f"lines[{i}]"], f"{cap}p", f"{ln.rate_p}p")

    # Every day of the period appears exactly once among standing lines.
    standing = [ln for ln in bill.lines if ln.kind == "standing" and ln.start and ln.end]
    if bill.period_start and bill.period_end and standing:
        covered: dict[date, int] = {}
        for ln in standing:
            d = ln.start
            while d <= ln.end:
                covered[d] = covered.get(d, 0) + 1
                d += timedelta(days=1)
        all_days = [bill.period_start + timedelta(days=k) for k in range(_days(bill.period_start, bill.period_end))]
        missing = [d for d in all_days if d not in covered]
        doubled = [d for d, n in covered.items() if n > 1]
        c.add("standing_coverage", "standing charge", not missing and not doubled,
              "standing charge covers each day of the period once", ["lines", "period_start", "period_end"],
              f"{len(all_days)} days", f"{len(covered)} days",
              (f"missing {', '.join(map(str, missing[:3]))}" if missing else "")
              + (f" doubled {', '.join(map(str, doubled[:3]))}" if doubled else "") or None)

    # VAT rate must match the supply dates, and the VAT base must match the lines it covers.
    if bill.vat and all(v.pct is not None for v in bill.vat):
        by_pct: dict[Decimal, Decimal] = {}
        for ln in bill.lines:
            if ln.start is None or ln.amount_gbp is None:
                continue
            q = tariff.quarter_for(ln.start)
            if q is not None:
                by_pct[q.vat_pct] = by_pct.get(q.vat_pct, Decimal(0)) + ln.amount_gbp
        billed_pcts = {v.pct for v in bill.vat if v.amount_gbp}
        expected_pcts = {p for p, base in by_pct.items() if base != 0}
        wrong = billed_pcts - expected_pcts
        c.add("vat_rate", "VAT", not wrong, "VAT rate matches the supply dates (0% on electricity from 1 Oct 2026)",
              ["vat", "lines"], ", ".join(f"{p}%" for p in sorted(expected_pcts)),
              ", ".join(f"{v.pct}%" for v in bill.vat))
        for k, v in enumerate(bill.vat):
            if v.pct is None or v.on_gbp is None or v.amount_gbp is None:
                continue
            if v.pct in by_pct:
                c.add(f"vat_base_{k}", "VAT", abs(by_pct[v.pct] - v.on_gbp) <= LINE_TOLERANCE,
                      f"VAT {v.pct}% is charged on the lines supplied at that rate", [f"vat[{k}]", "lines"],
                      _gbp(by_pct[v.pct]), _gbp(v.on_gbp))
            exp = money(v.on_gbp * v.pct / 100)
            c.add(f"vat_arith_{k}", "VAT", abs(exp - v.amount_gbp) <= LINE_TOLERANCE,
                  "VAT base × rate = VAT amount", [f"vat[{k}]"], _gbp(exp), _gbp(v.amount_gbp))


def reconcile(bill: Bill, tariff: Tariff) -> Verdict:
    c = _Checks()
    why = _supported(bill)
    if why:
        return Verdict(kind=VerdictKind.UNSUPPORTED, headline=why, findings=[],
                       billed_total_gbp=bill.total_gbp, tariff_version=tariff.version)
    if bill.period_start and bill.period_end and (
            bill.period_start < tariff.first_day or bill.period_end > tariff.last_day):
        return Verdict(kind=VerdictKind.UNSUPPORTED,
                       headline=f"No published cap is encoded for {bill.period_start}..{bill.period_end}.",
                       findings=[], billed_total_gbp=bill.total_gbp, tariff_version=tariff.version)

    consistency(bill, c)
    tariff_checks(bill, tariff, c)

    for side, kind in (("prev", bill.prev_reading_type), ("curr", bill.curr_reading_type)):
        if kind == "E":
            c.warn(f"{side}_reading_estimated", "units", "readings must be actual to confirm usage",
                   [f"{side}_reading_type"], f"The {'previous' if side == 'prev' else 'current'} reading is "
                   "estimated, so the kWh billed cannot be confirmed. Submit a meter reading.")

    exp_total = None
    sources: list[str] = []
    if bill.period_start and bill.period_end and bill.units_kwh is not None:
        try:
            exp = expected_bill(tariff, bill.period_start, bill.period_end, bill.units_kwh)
            exp_total = exp.total_gbp
            sources = sorted({s.quarter.source for s in exp.segments})
            if bill.total_gbp is not None:
                c.add("expected_total", "total", abs(exp_total - bill.total_gbp) <= Decimal("0.02"),
                      "billed total = independently recomputed total for these units and dates",
                      ["total_gbp", "units_kwh", "period_start", "period_end"], _gbp(exp_total), _gbp(bill.total_gbp))
        except OutsideTariff as e:
            return Verdict(kind=VerdictKind.UNSUPPORTED, headline=str(e), findings=c.findings,
                           tariff_version=tariff.version)

    fails = [f for f in c.findings if f.status == CheckStatus.FAIL]
    warns = [f for f in c.findings if f.status == CheckStatus.WARN]
    required = [bill.period_start, bill.period_end, bill.units_kwh, bill.total_gbp, bill.tariff_name]
    diff = None if exp_total is None or bill.total_gbp is None else bill.total_gbp - exp_total

    if fails:
        comps = sorted({f.component for f in fails if f.check != "expected_total"} or {"total"})
        head = f"Check the {', '.join(comps)}: {len(fails)} check{'s' if len(fails) > 1 else ''} failed."
        if diff:
            head += f" Billed {_gbp(bill.total_gbp)}, recomputed {_gbp(exp_total)} ({'+' if diff > 0 else '−'}{_gbp(abs(diff))})."
        kind = VerdictKind.DISCREPANCY
    elif any(x is None for x in required) or not bill.lines:
        kind, head = VerdictKind.INSUFFICIENT, "Some fields needed to check this bill were not found."
    elif warns:
        kind, head = VerdictKind.INSUFFICIENT, "The charges add up, but the usage is based on an estimated reading."
    else:
        kind, head = VerdictKind.RECONCILED, f"Every line matches the Ofgem cap. Recomputed total {_gbp(exp_total)}."
    return Verdict(kind=kind, headline=head, findings=c.findings, expected_total_gbp=exp_total,
                   billed_total_gbp=bill.total_gbp, difference_gbp=diff,
                   tariff_version=tariff.version, tariff_sources=sources)


def gate(v: Verdict, unverified: set[str] | list[str]) -> Verdict:
    """A discrepancy must rest on values found printed on the bill.

    If a failed check uses a value the second reader could not find, the failure may be a misread,
    not a billing error: report "not enough evidence" and keep the findings so the user sees why.
    """
    unverified = set(unverified)
    if v.kind != VerdictKind.DISCREPANCY or not unverified:
        return v
    # a check names a field ("lines[0].end") or a group ("lines[0]", "lines"); a group covers its leaves
    used = {f for x in v.findings if x.status == CheckStatus.FAIL for f in x.fields}
    shaky = sorted(u for u in unverified
                   if any(u == f or u.startswith(f + ".") or u.startswith(f + "[") for f in used))
    if not shaky:
        return v
    n = len(shaky)
    return v.model_copy(update={
        "kind": VerdictKind.INSUFFICIENT,
        "headline": f"A check failed, but it uses {n} value{'s' if n > 1 else ''} not found on the bill. "
                    "Check {} first.".format("them" if n > 1 else "it")})
