"""The whole pipeline for one bill, and the deterministic re-run after a correction.

    extract (Gemma) -> ground (Tesseract) -> consistency + reconcile -> usage context (TabPFN) -> explain
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd

from secondlook.models import Bill
from secondlook.pipeline.context import ROOT, usage_context
from secondlook.pipeline.explain import explain
from secondlook.pipeline.extract import extract
from secondlook.pipeline.ocr import ground, ocr_lines
from secondlook.pipeline.reconcile import reconcile
from secondlook.tariff.engine import load_tariff

USAGE_SHIFT = timedelta(weeks=678)  # LCL 2011-2014 moved onto 2025-2026, weekdays preserved
NON_CRITICAL = {"supplier", "account_ref", "bill_date"}
LAG_ROWS = ROOT / "data" / "processed" / "lag_rows.parquet"
CONTEXT_METHOD = "tabpfn_v2_lags"  # chosen by Experiment C (eval/results/exp_c_summary.csv)


def content_hash(image: bytes) -> str:
    return hashlib.sha256(image).hexdigest()[:16]


@lru_cache(maxsize=1)
def _daily() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data" / "processed" / "lcl_daily.parquet")


def household_usage(hh: str) -> pd.Series:
    d = _daily()
    s = d[d["hh"] == hh].set_index("date")["kwh"].sort_index()
    s.index = s.index + USAGE_SHIFT
    return s


def households() -> list[str]:
    return sorted(_daily()["hh"].unique())


@lru_cache(maxsize=1)
def _lag_rows() -> pd.DataFrame | None:
    return pd.read_parquet(LAG_ROWS) if LAG_ROWS.exists() else None


def _context(bill: Bill, hh: str | None):
    if not hh or not bill.period_start or not bill.period_end or bill.units_kwh is None:
        return None
    train = None
    rows = _lag_rows()
    if CONTEXT_METHOD == "tabpfn_v2_lags" and rows is not None:
        start = pd.Timestamp(bill.period_start)
        train = rows[(rows.hh != hh) & (rows.end < start)]
        train = train.sample(min(len(train), 3000), random_state=0)
    return usage_context(household_usage(hh), bill.period_start, bill.period_end, float(bill.units_kwh), train)


def analyse_bill(bill: Bill, hh: str | None, use_model: bool = True) -> dict:
    """Stages 2-5. Deterministic given the bill, except the model's wording (which never sets numbers)."""
    t = {}
    t0 = time.perf_counter()
    verdict = reconcile(bill, load_tariff())
    t["reconcile"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    ctx = _context(bill, hh)
    t["context"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    ex = explain(verdict, ctx, use_model=use_model)
    t["explain"] = time.perf_counter() - t0
    return {"verdict": json.loads(verdict.model_dump_json()),
            "context": None if ctx is None else ctx.__dict__,
            "explanation": ex.__dict__, "timings": {k: round(v, 3) for k, v in t.items()}}


def run_image(image: bytes, hh: str | None = None, use_model_explain: bool = True) -> dict:
    t0 = time.perf_counter()
    ex = extract(image)
    t_extract = time.perf_counter() - t0
    t0 = time.perf_counter()
    lines = ocr_lines(image)
    boxes = ground(ex.bill, lines)
    t_ground = time.perf_counter() - t0
    review = sorted(k for k, b in boxes.items() if b is None and k not in NON_CRITICAL)
    out = analyse_bill(ex.bill, hh, use_model_explain)
    out["timings"] = {"extract": round(t_extract, 3), "ground": round(t_ground, 3), **out["timings"]}
    out.update({
        "id": content_hash(image), "household": hh,
        "bill": json.loads(ex.bill.model_dump_json()),
        "extraction": {"attempts": ex.attempts, "problems": ex.problems, "model_seconds": round(ex.seconds, 2)},
        "boxes": boxes, "review_fields": review,
    })
    return out
