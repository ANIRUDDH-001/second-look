import random
from datetime import date, timedelta

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from datagen.bills import FAULTS, apply_fault, can_apply, clean_bill
from secondlook.models import VerdictKind
from secondlook.pipeline.reconcile import reconcile
from secondlook.tariff.engine import load_tariff

T = load_tariff()
USAGE = pd.Series(9.5, index=pd.date_range("2025-01-01", "2026-12-31"))


def bill_for(start, length=30, layout="classic", seed=1):
    return clean_bill(T, layout, "hh", USAGE, start, start + timedelta(days=length - 1),
                      random.Random(seed), 12000)[0]


@settings(max_examples=150, deadline=None)
@given(offset=st.integers(0, 690), length=st.integers(28, 35))
def test_every_clean_bill_reconciles(offset, length):
    b = bill_for(date(2025, 1, 1) + timedelta(days=offset), length)
    v = reconcile(b, T)
    assert v.kind == VerdictKind.RECONCILED, [f for f in v.findings if f.status != "pass"]
    assert v.difference_gbp == 0


CROSSING_VAT = date(2026, 10, 5)  # fully after 1 Oct 2026 so the VAT fault applies
CROSSING_Q = date(2026, 3, 20)


@pytest.mark.parametrize("fault", list(FAULTS))
def test_each_planted_fault_gets_its_expected_verdict(fault):
    start = CROSSING_VAT if fault == "vat_after_oct_2026" else CROSSING_Q
    b = bill_for(start)
    assert can_apply(fault, b, T)
    for seed in range(5):
        bad = apply_fault(fault, b, T, random.Random(seed))
        assert reconcile(bad, T).kind == FAULTS[fault], fault


def test_missing_total_is_insufficient_not_reconciled():
    b = bill_for(date(2025, 5, 3))
    b.total_gbp = None
    assert reconcile(b, T).kind == VerdictKind.INSUFFICIENT


def test_period_outside_caps_is_unsupported():
    b = bill_for(date(2026, 11, 20))
    b.period_end = date(2027, 1, 5)
    assert reconcile(b, T).kind == VerdictKind.UNSUPPORTED


def test_verdict_is_deterministic():
    b = bill_for(date(2026, 6, 14))
    assert reconcile(b, T).model_dump() == reconcile(b, T).model_dump()
