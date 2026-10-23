"""Experiment B: does reconciliation catch planted errors without crying wolf?

    python -m eval.exp_b_faults

Runs the deterministic checker on the ground-truth bills of the "faults" set (60 clean, 10 per fault
class), so this measures the rules alone, separate from transcription (Experiment A).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from secondlook.models import Bill, CheckStatus
from secondlook.pipeline.reconcile import reconcile
from secondlook.tariff.engine import load_tariff

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "eval" / "results"


def main() -> pd.DataFrame:
    tariff = load_tariff()
    out = []
    for line in open(ROOT / "datagen" / "answer_keys" / "faults.jsonl"):
        r = json.loads(line)
        v = reconcile(Bill.model_validate(r["bill"]), tariff)
        failed = sorted({f.check.rstrip("_0123456789") for f in v.findings if f.status == CheckStatus.FAIL})
        out.append({"bill_id": r["bill_id"], "fault": r["fault"] or "none", "expected": r["expected_verdict"],
                    "verdict": v.kind.value, "correct": v.kind.value == r["expected_verdict"],
                    "failed_checks": " ".join(failed),
                    "difference_gbp": "" if v.difference_gbp is None else str(v.difference_gbp)})
    df = pd.DataFrame(out)
    RES.mkdir(parents=True, exist_ok=True)
    df.to_csv(RES / "exp_b_bills.csv", index=False)
    summary = df.groupby("fault").agg(bills=("bill_id", "count"), expected=("expected", "first"),
                                      correct=("correct", "mean")).round(3)
    summary["verdicts"] = df.groupby("fault")["verdict"].agg(lambda s: dict(s.value_counts()))
    summary["checks_that_fired"] = df.groupby("fault")["failed_checks"].agg(
        lambda s: dict(pd.Series(" ".join(s).split()).value_counts()) if " ".join(s).strip() else {})
    summary.to_csv(RES / "exp_b_summary.csv")
    print(summary.to_string())
    clean = df[df.fault == "none"]
    print(f"\nfalse alarms on clean bills: {(clean.verdict != 'reconciled').sum()}/{len(clean)}")
    return df


if __name__ == "__main__":
    main()
