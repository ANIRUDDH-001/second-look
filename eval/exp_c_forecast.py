"""Experiment C: does TabPFN add anything over naive forecasts of a billing period's usage?

    python -m eval.exp_c_forecast [--households 80] [--cutoffs 6]

Walk-forward on London smart-meter households (time-shifted onto 2025-2026). At each cutoff we
forecast the mean daily kWh of the next 30 days using only data before the cutoff:
  naive_last      the previous 30 days
  naive_seasonal  the same 30 days a year earlier (364 days, so weekdays line up)
  naive_rolling3  the previous 90 days
  tabpfn_v2_lags  TabPFN v2 fitted in context on lag rows from *other* households, ended before the cutoff
  tabpfn_ts       TabPFN-TS (v2 weights) on the household's own daily series
Evaluation households never appear in the lag-model's training rows.
Writes eval/results/exp_c_forecasts.csv and exp_c_summary.csv, and data/processed/lag_rows.parquet.
"""
from __future__ import annotations

import argparse
import random
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from secondlook.pipeline.context import (FEATURES, actual_mean, lags_forecast, period_features,
                                         ts_forecast)
from secondlook.pipeline.run import LAG_ROWS, household_usage, households

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "eval" / "results"
N = 30


def build_lag_rows(step_days: int = 7) -> pd.DataFrame:
    """Lag features and target for 30-day windows every `step_days`, for every household."""
    rows = []
    for hh in households():
        s = household_usage(hh)
        first, last = s.index.min().date(), s.index.max().date()
        d = first + timedelta(days=60)
        while d + timedelta(days=N - 1) <= last:
            e = d + timedelta(days=N - 1)
            target = actual_mean(s, d, e)
            if not np.isnan(target):
                rows.append({"hh": hh, "start": pd.Timestamp(d), "end": pd.Timestamp(e), "target": target,
                             **period_features(s, d, e)})
            d += timedelta(days=step_days)
    df = pd.DataFrame(rows)
    LAG_ROWS.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LAG_ROWS)
    return df


def mase_scale(s: pd.Series, before: date) -> float:
    """In-sample MAE of the previous-period forecast on non-overlapping 30-day blocks before `before`."""
    h = s.loc[: pd.Timestamp(before - timedelta(days=1))]
    blocks = h.iloc[len(h) % N:].to_numpy().reshape(-1, N).mean(axis=1)
    return float(np.mean(np.abs(np.diff(blocks)))) if len(blocks) > 2 else np.nan


def main(n_households: int, n_cutoffs: int, seed: int = 0) -> None:
    lag = pd.read_parquet(LAG_ROWS) if LAG_ROWS.exists() else build_lag_rows()
    hhs = households()
    rng = random.Random(seed)
    # Only households with history across the whole evaluation year are eligible.
    eligible = [h for h in hhs if household_usage(h).index.min() <= pd.Timestamp("2025-01-01")
                and household_usage(h).index.max() >= pd.Timestamp("2027-01-31")]
    eval_hh = sorted(rng.sample(eligible, min(n_households, len(eligible))))
    pool = lag[~lag.hh.isin(eval_hh)]
    months = pd.date_range("2026-01-01", "2026-12-01", freq="MS")
    cutoffs = [m.date() for m in months[:: max(1, len(months) // n_cutoffs)]][:n_cutoffs]
    print(f"{len(eval_hh)} evaluation households (of {len(eligible)} eligible), cutoffs {cutoffs}")

    out = []
    for c in cutoffs:
        e = c + timedelta(days=N - 1)
        series = {h: household_usage(h) for h in eval_hh}
        q = pd.DataFrame([{"hh": h, **period_features(s, c, e)} for h, s in series.items()]).set_index("hh")
        train = pool[pool.end < pd.Timestamp(c)].sample(n=min(3000, (pool.end < pd.Timestamp(c)).sum()), random_state=0)
        t0 = time.perf_counter()
        lags = lags_forecast(train, q[FEATURES])
        t_lags = (time.perf_counter() - t0) / len(q)
        t0 = time.perf_counter()
        ts = ts_forecast({h: s.loc[: pd.Timestamp(c - timedelta(days=1))] for h, s in series.items()}, c, e)
        t_ts = (time.perf_counter() - t0) / len(q)
        for h, s in series.items():
            y = actual_mean(s, c, e)
            if np.isnan(y):
                continue
            f = q.loc[h]
            base = {"hh": h, "cutoff": c, "actual": y, "scale": mase_scale(s, cutoffs[0])}
            for m, pred in (("naive_last", f.prev_mean), ("naive_seasonal", f.last_year_mean),
                            ("naive_rolling3", f.rolling3_mean)):
                out.append({**base, "method": m, "pred": pred, "seconds": 0.0})
            for m, qq, secs in (("tabpfn_v2_lags", lags.loc[h], t_lags), ("tabpfn_ts", ts.loc[h], t_ts)):
                out.append({**base, "method": m, "pred": qq["0.5"], "seconds": secs,
                            "q05": qq["0.05"], "q10": qq["0.1"], "q90": qq["0.9"], "q95": qq["0.95"]})
        print(f"cutoff {c}: lags {t_lags:.2f}s/forecast, ts {t_ts:.2f}s/forecast", flush=True)

    df = pd.DataFrame(out)
    RES.mkdir(parents=True, exist_ok=True)
    df.to_csv(RES / "exp_c_forecasts.csv", index=False)
    summarize(df)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["pred", "actual"]).copy()
    # Compare every method on the same (household, cutoff) pairs: drop pairs any method could not forecast.
    full = df.groupby(["hh", "cutoff"]).method.transform("count") == df.method.nunique()
    df = df[full]
    df["ae"] = (df.pred - df.actual).abs()
    df["se"] = df.ae / df.scale
    df["in80"] = (df.actual >= df.q10) & (df.actual <= df.q90)
    df["in90"] = (df.actual >= df.q05) & (df.actual <= df.q95)
    has_q = df.q05.notna()
    s = df.groupby("method").agg(forecasts=("ae", "size"), mae_kwh_per_day=("ae", "mean"),
                                 mase=("se", "mean"), seconds=("seconds", "mean"))
    s["coverage80"] = df[has_q].groupby("method").in80.mean()
    s["coverage90"] = df[has_q].groupby("method").in90.mean()
    s["width90_kwh_per_day"] = df[has_q].assign(w=df.q95 - df.q05).groupby("method").w.mean()
    s = s.round(4).sort_values("mae_kwh_per_day")
    s.to_csv(RES / "exp_c_summary.csv")
    print(s.to_string())
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--households", type=int, default=80)
    ap.add_argument("--cutoffs", type=int, default=6)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    if a.rebuild:
        build_lag_rows()
    main(a.households, a.cutoffs)
