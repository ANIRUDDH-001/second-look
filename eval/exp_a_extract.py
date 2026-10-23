"""Experiment A: can local Gemma 4 E2B transcribe bills well enough, and does the review flag catch its misses?

    python -m eval.exp_a_extract [--limit N] [--variants clean,scan,phone]

For each rendered bill: Gemma extraction (cached per image in eval/results/cache/), Tesseract OCR,
grounding of Gemma's values in the OCR words, and the Tesseract+regex baseline.
Writes eval/results/exp_a_fields.csv (one row per image x method x field) and exp_a_bills.csv.
The three variants share a base bill, so they are robustness views, not independent samples.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from secondlook.models import Bill
from secondlook.pipeline.extract import extract, to_bill
from secondlook.pipeline.ocr import baseline_extract, ground, ocr_lines, repair
from secondlook.pipeline.reconcile import gate, reconcile
from secondlook.tariff.engine import load_tariff

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "eval" / "results"
CACHE = RES / "cache"
NON_CRITICAL = {"supplier", "account_ref", "bill_date"}  # not used by any check


def flatten(b: Bill) -> dict[str, str]:
    d = json.loads(b.model_dump_json())
    out = {k: v for k, v in d.items() if k not in ("lines", "vat")}
    for i, ln in enumerate(d["lines"]):
        out.update({f"lines[{i}].{k}": v for k, v in ln.items()})
    for i, v in enumerate(d["vat"]):
        out.update({f"vat[{i}].{k}": x for k, x in v.items()})
    return {k: (None if v is None else str(v)) for k, v in out.items()}


def compare(truth: Bill, got: Bill) -> tuple[dict[str, bool], int]:
    t, g = flatten(truth), flatten(got)
    t = {k: v for k, v in t.items() if v is not None}
    ok = {k: g.get(k) == v for k, v in t.items()}
    spurious = sum(1 for k, v in g.items() if v is not None and k not in t)
    return ok, spurious


def gemma_cached(path: Path) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{path.stem}.json"
    if f.exists():
        return json.loads(f.read_text())
    ex = extract(path.read_bytes())
    rec = {"raw": ex.raw, "attempts": ex.attempts, "seconds": round(ex.seconds, 2), "problems": ex.problems}
    f.write_text(json.dumps(rec))
    return rec


def main(limit: int | None, variants: set[str]) -> None:
    tariff = load_tariff()
    rows = [json.loads(x) for x in open(ROOT / "datagen" / "answer_keys" / "extract.jsonl")]
    rows = [r for r in rows if r["variant"] in variants][:limit]
    RES.mkdir(parents=True, exist_ok=True)
    fields_out, bills_out = [], []
    for n, r in enumerate(rows, 1):
        path = ROOT / "data" / "bills" / "extract" / f"{r['bill_id']}_{r['variant']}.png"
        truth = Bill.model_validate(r["bill"])
        t0 = time.perf_counter()
        lines = ocr_lines(path.read_bytes())
        ocr_s = time.perf_counter() - t0
        rec = gemma_cached(path)
        g_bill, _ = to_bill(rec["raw"]) if rec["raw"] else (Bill(), None)
        grounding = ground(g_bill, lines)
        r_bill, fixes = repair(g_bill, lines, grounding)
        r_grounding = {**grounding, **{f["field"]: f["box"] for f in fixes}}
        methods = {
            "gemma": (g_bill, rec["seconds"], bool(rec["problems"]) or any(
                v is None for k, v in grounding.items() if k not in NON_CRITICAL)),
            # the shipped pipeline: Gemma, then the OCR second reader fills shortened rates / subtotal
            "gemma_checked": (r_bill, rec["seconds"] + ocr_s, bool(rec["problems"]) or any(
                v is None for k, v in r_grounding.items() if k not in NON_CRITICAL)),
            "tesseract": (baseline_extract(lines), ocr_s, None),
        }
        for method, (bill, secs, flagged) in methods.items():
            ok, spurious = compare(truth, bill)
            crit_ok = all(v for k, v in ok.items() if k not in NON_CRITICAL) and spurious == 0
            v = reconcile(bill, tariff)
            if method == "gemma_checked":  # the shipped pipeline gates failures on unfound values
                v = gate(v, [k for k, b in r_grounding.items() if b is None and k not in NON_CRITICAL])
            verdict = v.kind.value
            if flagged is None:  # the baseline has no second reader; flag only on missing required fields
                flagged = any(getattr(bill, k) is None for k in ("period_start", "period_end", "units_kwh", "total_gbp"))
            for k, v in ok.items():
                fields_out.append({"image": path.stem, "variant": r["variant"], "layout": r["layout"], "method": method,
                                   "field": k, "critical": k not in NON_CRITICAL, "correct": v,
                                   "grounded": ((r_grounding if method == "gemma_checked" else grounding).get(k) is not None)
                                   if method != "tesseract" else ""})
            bills_out.append({
                "image": path.stem, "bill_id": r["bill_id"], "variant": r["variant"], "layout": r["layout"],
                "fault": r["fault"] or "", "method": method, "seconds": round(secs, 2),
                "attempts": rec["attempts"] if method == "gemma" else "",
                "field_acc": round(sum(ok.values()) / len(ok), 4), "spurious": spurious,
                "critical_all_correct": crit_ok, "flagged": flagged,
                "verdict": verdict, "expected_verdict": r["expected_verdict"],
                "verdict_correct": verdict == r["expected_verdict"]})
        g, c = bills_out[-3], bills_out[-2]
        print(f"[{n}/{len(rows)}] {path.stem} gemma {g['seconds']}s acc={g['field_acc']} flag={g['flagged']} "
              f"verdict={g['verdict']} (exp {g['expected_verdict']}) | checked acc={c['field_acc']} "
              f"flag={c['flagged']} verdict={c['verdict']} | tess acc={bills_out[-1]['field_acc']}", flush=True)
    _write(RES / "exp_a_fields.csv", fields_out)
    _write(RES / "exp_a_bills.csv", bills_out)
    summarize(bills_out)


def summarize(bills: list[dict]) -> None:
    import pandas as pd
    df = pd.DataFrame(bills)
    df["outcome"] = df.apply(lambda x: ("correct" if x.critical_all_correct else "wrong") + "_"
                             + ("flagged" if x.flagged else "accepted"), axis=1)
    summary = df.groupby(["method", "variant"]).agg(
        bills=("image", "count"), field_acc=("field_acc", "mean"),
        critical_all_correct=("critical_all_correct", "mean"), flagged=("flagged", "mean"),
        verdict_correct=("verdict_correct", "mean"), seconds=("seconds", "median")).round(3)
    outcomes = df.pivot_table(index=["method", "variant"], columns="outcome", values="image",
                              aggfunc="count", fill_value=0)
    out = summary.join(outcomes)
    out.to_csv(RES / "exp_a_summary.csv")
    print(out.to_string())


def _write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--variants", default="clean,scan,phone")
    a = ap.parse_args()
    main(a.limit, set(a.variants.split(",")))
