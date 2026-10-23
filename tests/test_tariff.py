from datetime import date, timedelta
from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from secondlook.tariff.engine import OutsideTariff, expected_bill, load_tariff, net_rate, split_units

T = load_tariff()


def test_net_rate_strips_vat():
    assert net_rate(Decimal("26.11"), Decimal(5)) == Decimal("24.867")
    assert net_rate(Decimal("26.32"), Decimal(0)) == Decimal("26.320")


def test_quarters_are_contiguous_and_cover_2025_2026():
    assert T.first_day == date(2025, 1, 1) and T.last_day == date(2026, 12, 31)


def test_vat_drops_to_zero_from_1_october_2026():
    assert T.quarter_for(date(2026, 9, 30)).vat_pct == 5
    assert T.quarter_for(date(2026, 10, 1)).vat_pct == 0


def test_period_across_rate_change_is_split_by_days():
    e = expected_bill(T, date(2026, 3, 17), date(2026, 4, 15), Decimal(300))
    assert [s.days for s in e.segments] == [15, 15]
    assert [s.kwh for s in e.segments] == [150, 150]
    assert e.segments[0].quarter.unit_rate_p != e.segments[1].quarter.unit_rate_p


def test_period_across_vat_switch_has_two_vat_groups():
    e = expected_bill(T, date(2026, 9, 16), date(2026, 10, 15), Decimal(250))
    assert set(e.vat_by_pct) == {Decimal(5), Decimal(0)}
    assert e.vat_by_pct[Decimal(0)][1] == 0


def test_outside_published_caps_raises():
    with pytest.raises(OutsideTariff):
        expected_bill(T, date(2026, 12, 20), date(2027, 1, 18), Decimal(100))


@given(units=st.integers(0, 5000), days=st.lists(st.integers(1, 92), min_size=1, max_size=3))
def test_split_units_conserves_kwh(units, days):
    parts = split_units(Decimal(units), days)
    assert sum(parts) == units and len(parts) == len(days)


@settings(max_examples=300)
@given(offset=st.integers(0, 690), length=st.integers(1, 40), units=st.integers(0, 3000))
def test_expected_total_is_consistent(offset, length, units):
    start = date(2025, 1, 1) + timedelta(days=offset)
    end = start + timedelta(days=length - 1)
    e = expected_bill(T, start, end, Decimal(units))
    assert sum(s.days for s in e.segments) == length
    assert sum(s.kwh for s in e.segments) == units
    assert e.total_gbp == e.subtotal_gbp + sum(a for _, a in e.vat_by_pct.values())
    assert e.total_gbp >= 0
    # more usage never costs less
    assert expected_bill(T, start, end, Decimal(units + 1)).total_gbp >= e.total_gbp
