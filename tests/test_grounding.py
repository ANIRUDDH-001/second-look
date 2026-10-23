from datetime import date
from decimal import Decimal

from secondlook.models import Bill, ChargeLine, VatLine
from secondlook.pipeline.ocr import Line, ground, repair


def _line(text, y):
    return Line([(w, (10 * i, y, 10 * i + 8, y + 10)) for i, w in enumerate(text.split())])


PAGE = [
    _line("Bill period 13 Jun 2025 to 16 Jul 2025", 0),
    _line("Electricity used 73 kWh 25.743p/kWh £18.79", 20),
    _line("13 Jun 2025 — 30 Jun 2025", 40),
    _line("Before VAT £45.95", 60),
    _line("VAT 5% on £45.95 £2.30", 80),
    _line("Total £48.25", 100),
]


def _bill(**kw):
    line = ChargeLine(kind="energy", start=date(2025, 6, 13), end=kw.pop("end", date(2025, 6, 30)),
                      quantity=Decimal(73), rate_p=kw.pop("rate", Decimal("25.743")), amount_gbp=Decimal("18.79"))
    return Bill(lines=[line], vat=[VatLine(pct=Decimal(5), on_gbp=Decimal("45.95"), amount_gbp=Decimal("2.30"))],
                subtotal_gbp=kw.pop("subtotal", Decimal("45.95")), total_gbp=Decimal("48.25"))


def test_a_line_date_must_be_on_that_lines_row():
    # 16 Jul 2025 is printed on the page, but not on this charge's row
    boxes = ground(_bill(end=date(2025, 7, 16)), PAGE)
    assert boxes["lines[0].end"] is None
    assert ground(_bill(), PAGE)["lines[0].end"] is not None


def test_a_total_copied_into_the_subtotal_is_not_found():
    boxes = ground(_bill(subtotal=Decimal("48.25")), PAGE)
    assert boxes["subtotal_gbp"] is None


def test_repair_takes_printed_digits_and_reports_it():
    for short in ("25.74", "25.7"):  # truncated, and rounded to fewer places
        b = _bill(rate=Decimal(short))
        fixed, fixes = repair(b, PAGE, ground(b, PAGE))
        assert fixed.lines[0].rate_p == Decimal("25.743")
        assert fixes[0]["field"] == "lines[0].rate_p" and fixes[0]["model"] == short
    b = _bill(subtotal=Decimal("48.25"))
    fixed, fixes = repair(b, PAGE, ground(b, PAGE))
    assert fixed.subtotal_gbp == Decimal("45.95") and fixes[0]["field"] == "subtotal_gbp"


def test_repair_leaves_a_genuine_value_alone():
    b = _bill(rate=Decimal("26.1"))  # not a shortening of anything printed
    fixed, fixes = repair(b, PAGE, ground(b, PAGE))
    assert fixes == [] and fixed.lines[0].rate_p == Decimal("26.1")


def test_a_failure_on_an_unfound_value_is_not_called_a_discrepancy():
    from secondlook.models import VerdictKind
    from secondlook.pipeline.reconcile import gate, reconcile
    from secondlook.tariff.engine import load_tariff

    b = _bill()
    b.period_start, b.period_end = date(2025, 6, 13), date(2025, 6, 30)
    b.lines[0].rate_p = Decimal("30.000")  # wrong rate -> the line checks fail
    v = reconcile(b, load_tariff())
    assert v.kind == VerdictKind.DISCREPANCY
    failed = {f for x in v.findings if x.status.value == "fail" for f in x.fields}
    assert gate(v, failed).kind == VerdictKind.INSUFFICIENT
    assert gate(v, ["lines[0].rate_p"]).kind == VerdictKind.INSUFFICIENT  # a leaf inside a failed group
    assert gate(v, ["account_ref"]).kind == VerdictKind.DISCREPANCY  # unrelated unfound value
