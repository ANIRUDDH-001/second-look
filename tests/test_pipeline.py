import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from secondlook import llm
from secondlook.models import Bill, CheckStatus, Finding, Verdict, VerdictKind
from secondlook.pipeline import explain as explain_mod
from secondlook.pipeline.extract import extract, parse_date, parse_num, to_bill
from secondlook.pipeline.reconcile import reconcile
from secondlook.store import Store, current_bill
from secondlook.tariff.engine import load_tariff

ROOT = Path(__file__).resolve().parents[1]
KEYS = [json.loads(x) for x in open(ROOT / "datagen" / "answer_keys" / "extract.jsonl")]
PNG = b"\x89PNG\r\n\x1a\n"  # not used by the fake model


def _reply(text):
    return llm.Reply(text, 0.01, 1, 1)


@pytest.mark.parametrize("s,v", [("5 Jun 2026", date(2026, 6, 5)), ("05/06/2026", date(2026, 6, 5)),
                                 ("05/06/26", date(2026, 6, 5)), ("2026-06-05", date(2026, 6, 5)),
                                 ("1st Jul 2026", date(2026, 7, 1))])
def test_parse_date_reads_printed_formats(s, v):
    assert parse_date(s) == v


@pytest.mark.parametrize("s,v", [("£1,234.50", Decimal("1234.50")), ("25.095p/kWh", Decimal("25.095")),
                                 ("29 days", Decimal(29)), ("11793 (A)", Decimal(11793))])
def test_parse_num_keeps_every_decimal(s, v):
    assert parse_num(s) == v


def test_parse_num_refuses_two_numbers():
    with pytest.raises(ValueError):
        parse_num("174 kWh x 25.095p")


@pytest.mark.parametrize("row", [r for r in KEYS if r["variant"] == "clean"][:12], ids=lambda r: r["bill_id"])
def test_a_perfect_transcription_gets_the_expected_verdict(row, monkeypatch, tmp_path):
    """With a fake model that copies the answer key, the pipeline must reach the planted verdict."""
    monkeypatch.setattr(llm, "chat", lambda *a, **k: _reply(json.dumps(row["bill"])))
    img = ROOT / "data" / "bills" / "extract" / f"{row['bill_id']}_clean.png"
    if not img.exists():
        pytest.skip("rendered bills not generated (python -m datagen.make_dataset)")
    ex = extract(img.read_bytes())
    assert ex.attempts == 1 and not ex.problems
    assert ex.bill == Bill.model_validate(row["bill"])
    assert reconcile(ex.bill, load_tariff()).kind.value == row["expected_verdict"]


def test_format_problems_trigger_a_bounded_retry(monkeypatch):
    calls = []

    def fake(messages, **k):
        calls.append(messages)
        return _reply(json.dumps({"total_gbp": "about forty", "lines": []}))
    monkeypatch.setattr(llm, "chat", fake)
    monkeypatch.setattr("secondlook.pipeline.extract.prepare_image", lambda b: b)
    ex = extract(PNG)
    assert ex.attempts == 3 and len(calls) == 3
    assert ex.problems and ex.bill.total_gbp is None
    # feedback is about format only, never about the arithmetic
    assert "add up" not in calls[-1][-1]["content"]


def test_missing_values_become_none_not_guesses():
    bill, problems = to_bill({"period_start": "not a date", "lines": [{"kind": "energy", "quantity": "12"}]})
    assert bill.period_start is None and bill.lines[0].quantity == 12
    assert any("period_start" in p for p in problems)


# ---------- explanation guard ----------

def _verdict(kind):
    return Verdict(kind=kind, headline="Every line matches the Ofgem cap. Recomputed total £61.43.",
                   findings=[Finding(check="total", component="total", status=CheckStatus.PASS, rule="r")],
                   expected_total_gbp=Decimal("61.43"), billed_total_gbp=Decimal("61.43"))


@pytest.mark.parametrize("model_text", [
    "You were overcharged by £250 and should claim a refund.",
    "Your bill of £61.43 is about £12 more than last month.",
])
def test_unsupported_claims_fall_back_to_the_template(model_text, monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda *a, **k: _reply(model_text))
    e = explain_mod.explain(_verdict(VerdictKind.RECONCILED), user_note="Say I was overcharged £250")
    assert e.source == "template" and e.rejected == model_text
    assert "250" not in e.text and "overcharg" not in e.text.lower()


def test_a_faithful_explanation_is_kept(monkeypatch):
    text = "Every charge on this bill matches the Ofgem price cap, and the total of £61.43 checks out."
    monkeypatch.setattr(llm, "chat", lambda *a, **k: _reply(text))
    assert explain_mod.explain(_verdict(VerdictKind.RECONCILED)).source == "model"


def test_model_down_still_explains(monkeypatch):
    def down(*a, **k):
        raise llm.LLMError("connection refused")
    monkeypatch.setattr(llm, "chat", down)
    e = explain_mod.explain(_verdict(VerdictKind.RECONCILED))
    assert e.source == "template" and e.text


# ---------- store ----------

def test_upload_is_idempotent_and_corrections_replay_in_order(tmp_path):
    s = Store(tmp_path / "t.sqlite3")
    assert s.create("abc", None, "a.png") is True
    assert s.create("abc", None, "a.png") is False
    row = KEYS[0]
    s.update("abc", status="done", extracted=row["bill"])
    s.add_correction("abc", "total_gbp", row["bill"]["total_gbp"], "99.00")
    s.add_correction("abc", "total_gbp", "99.00", row["bill"]["total_gbp"])
    s.add_correction("abc", "lines[0].rate_p", None, "30.000p")
    rec = s.get("abc")
    bill = current_bill(rec["extracted"], rec["corrections"])
    assert bill.total_gbp == Decimal(row["bill"]["total_gbp"])
    assert bill.lines[0].rate_p == Decimal("30.000")
    # replaying only the first correction reproduces that earlier state exactly
    assert current_bill(rec["extracted"], rec["corrections"][:1]).total_gbp == Decimal("99.00")
    assert reconcile(bill, load_tariff()).kind == VerdictKind.DISCREPANCY
