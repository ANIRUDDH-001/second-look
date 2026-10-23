"""Aggregate Low Carbon London half-hourly smart-meter readings to daily kWh per household.

Source: UK Power Networks, "SmartMeter Energy Consumption Data in London Households",
London Datastore, CC BY 4.0. Only standard-tariff ("Std") households are kept.
A day counts only if at least 46 of its 48 half-hour readings are present.

    python -m datagen.lcl_aggregate data/raw/lcl data/processed/lcl_daily.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

MIN_READINGS = 46


def aggregate_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, names=["hh", "tariff", "ts", "kwh"], header=0, dtype=str)
    df = df[df["tariff"].str.strip() == "Std"]
    df["kwh"] = pd.to_numeric(df["kwh"].str.strip(), errors="coerce")
    df = df.dropna(subset=["kwh"])
    # A reading stamped 00:00 closes the previous day's last half hour.
    ts = pd.to_datetime(df["ts"].str[:19]) - pd.Timedelta(minutes=30)
    df["date"] = ts.dt.date
    daily = df.groupby(["hh", "date"]).agg(kwh=("kwh", "sum"), n=("kwh", "size")).reset_index()
    return daily[daily["n"] >= MIN_READINGS].drop(columns="n")


def main(src: str, out: str) -> None:
    files = sorted(Path(src).rglob("*.csv"))
    parts = [aggregate_file(f) for f in files]
    daily = pd.concat(parts).groupby(["hh", "date"], as_index=False)["kwh"].sum()
    daily["date"] = pd.to_datetime(daily["date"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out, index=False)
    print(f"{daily['hh'].nunique()} households, {len(daily)} household-days, "
          f"{daily['date'].min().date()}..{daily['date'].max().date()} -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
