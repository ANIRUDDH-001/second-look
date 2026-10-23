"""Stage 4: is this period's usage typical for this household? Context, never a billing verdict.

Two TabPFN forecasters for the mean daily kWh over a billing period, both local, CPU, telemetry off:

- lags:  TabPFN v2 regressor on period-level lag features. It is fitted in context on rows from
         *other* households whose periods ended before the cutoff, then asked for quantiles.
- ts:    TabPFN-TS forecasting this household's daily series for the period's days, run with
         the same v2 weights. The period interval sums daily quantiles, which assumes errors move
         together across days, so it is wide on purpose; Experiment C measures its coverage.

The weights are TabPFN v2 (Prior Labs licence, non-commercial). TabPFN-TS 1.3 defaults to the
gated v3.5 checkpoint; we pass the v2 checkpoint explicitly instead.
"""
from __future__ import annotations

import os

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CKPT = Path(os.environ.get("SECONDLOOK_TABPFN_CKPT", ROOT / "models" / "tabpfn-v2-regressor.ckpt"))
QUANTILES = [0.05, 0.1, 0.5, 0.9, 0.95]
FEATURES = ["prev_mean", "prev7_mean", "rolling3_mean", "rolling90_mean", "last_year_mean",
            "doy_sin", "doy_cos", "days"]


# ---------- features shared by every method ----------

def _mean(s: pd.Series, start: date, end: date) -> float:
    v = s.loc[pd.Timestamp(start):pd.Timestamp(end)]
    return float(v.mean()) if len(v) >= 0.8 * ((end - start).days + 1) else np.nan


def period_features(daily: pd.Series, start: date, end: date) -> dict:
    """Features for the period start..end using only days before start. daily: kWh indexed by date."""
    n = (end - start).days + 1
    d1 = start - timedelta(days=1)
    mid = start + timedelta(days=n // 2)
    doy = 2 * np.pi * mid.timetuple().tm_yday / 365.25
    return {
        "prev_mean": _mean(daily, start - timedelta(days=n), d1),
        "prev7_mean": _mean(daily, start - timedelta(days=7), d1),
        "rolling3_mean": _mean(daily, start - timedelta(days=3 * n), d1),
        "rolling90_mean": _mean(daily, start - timedelta(days=90), d1),
        "last_year_mean": _mean(daily, start - timedelta(days=364), end - timedelta(days=364)),
        "doy_sin": np.sin(doy), "doy_cos": np.cos(doy), "days": n,
    }


def actual_mean(daily: pd.Series, start: date, end: date) -> float:
    return _mean(daily, start, end)


# ---------- TabPFN v2 on lag features ----------

@lru_cache(maxsize=1)
def _regressor_factory():
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    def make():
        return TabPFNRegressor.create_default_for_version(
            ModelVersion.V2, model_path=str(CKPT), device="cpu", random_state=0,
            memory_saving_mode=True, ignore_pretraining_limits=True)
    return make


def lags_forecast(train: pd.DataFrame, query: pd.DataFrame) -> pd.DataFrame:
    """train: FEATURES + 'target' rows; query: FEATURES rows. Returns quantile columns per query row."""
    train = train.dropna(subset=["target", "prev_mean"])
    reg = _regressor_factory()()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg.fit(train[FEATURES].to_numpy(np.float32), train["target"].to_numpy(np.float32))
        q = reg.predict(query[FEATURES].to_numpy(np.float32), output_type="quantiles", quantiles=QUANTILES)
    return pd.DataFrame({str(k): v for k, v in zip(QUANTILES, q)}, index=query.index)


# ---------- TabPFN-TS on the daily series ----------

@lru_cache(maxsize=1)
def _ts_pipeline():
    from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline

    return TabPFNTSPipeline(
        max_context_length=730, tabpfn_mode=TabPFNMode.LOCAL,
        tabpfn_model_config={"model_path": str(CKPT), "device": "cpu", "random_state": 0,
                             "ignore_pretraining_limits": True})


def ts_forecast(history: dict[str, pd.Series], start: date, end: date) -> pd.DataFrame:
    """history: item -> daily kWh before start. Returns per item the period-mean quantiles."""
    n = (end - start).days + 1
    rows = []
    for item, s in history.items():
        s = s.loc[: pd.Timestamp(start - timedelta(days=1))].iloc[-730:]
        rows.append(pd.DataFrame({"item_id": item, "timestamp": s.index, "target": s.to_numpy(np.float32)}))
    ctx = pd.concat(rows, ignore_index=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = _ts_pipeline().predict_df(ctx, prediction_length=n, quantiles=QUANTILES)
    pred = pred.reset_index()
    out = {}
    for item, g in pred.groupby("item_id"):
        # Daily quantiles summed: comonotone, so a wide (conservative) interval for the period.
        out[item] = {str(q): float(g[str(q)].clip(lower=0).mean()) for q in QUANTILES}
    return pd.DataFrame.from_dict(out, orient="index")


# ---------- the product stage ----------

@dataclass
class UsageContext:
    method: str
    days: int
    billed_kwh: float
    expected_kwh: float
    low90_kwh: float
    high90_kwh: float
    label: str  # "typical" | "unusually_high" | "unusually_low" | "no_history"
    history_days: int


def usage_context(daily: pd.Series, start: date, end: date, billed_kwh: float,
                  train: pd.DataFrame | None = None) -> UsageContext:
    """Compare billed kWh with a 90% interval from the household's own history.

    `train` holds lag-feature rows from other households (see eval.exp_c_forecast.training_rows);
    when given, the lag-feature regressor is used, otherwise TabPFN-TS on the daily series.
    """
    n = (end - start).days + 1
    hist = daily.loc[: pd.Timestamp(start - timedelta(days=1))]
    if len(hist) < 60:
        return UsageContext("none", n, billed_kwh, float("nan"), float("nan"), float("nan"), "no_history", len(hist))
    if train is not None:
        q = lags_forecast(train, pd.DataFrame([period_features(daily, start, end)])).iloc[0]
        method = "tabpfn_v2_lags"
    else:
        q = ts_forecast({"hh": hist}, start, end).iloc[0]
        method = "tabpfn_ts"
    lo, mid, hi = (float(q[k]) * n for k in ("0.05", "0.5", "0.95"))
    label = "typical"
    if billed_kwh > hi:
        label = "unusually_high"
    elif billed_kwh < lo:
        label = "unusually_low"
    return UsageContext(method, n, billed_kwh, round(mid, 1), round(lo, 1), round(hi, 1), label, len(hist))
