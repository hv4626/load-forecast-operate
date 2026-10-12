"""Parse the lifetime workbook, resample to hourly kW, join Open-Meteo weather."""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    COVARIATES,
    DATA_DIR,
    HOLIDAYS,
    OUTPUT_DIR,
    SITES,
    SOURCE_XLSX,
    TZ,
    WEATHER_HOURLY,
)

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _parse_ts(value) -> pd.Timestamp | pd.NaT:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    if isinstance(value, datetime):
        return pd.Timestamp(value).tz_localize(TZ) if value.tzinfo is None else pd.Timestamp(value)
    text = str(value).strip()
    for fmt in ("%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M", "%d-%m-%Y"):
        try:
            ts = datetime.strptime(text, fmt)
            return pd.Timestamp(ts).tz_localize(TZ)
        except ValueError:
            continue
    return pd.NaT


def _load_5min(path: Path, site_id: str, meta: dict) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=meta["sheet_5min"])
    raw.columns = [str(c).strip() for c in raw.columns]
    ts = raw.iloc[:, 0].map(_parse_ts)
    frame = raw.copy()
    frame["_ts"] = ts
    frame = frame.dropna(subset=["_ts"]).sort_values("_ts")

    schema = meta["schema"]
    if schema == "voltage_first":
        # Timestamp, V1, V2, V3, freq, P1, P2, P3, total_kWh, daily_kWh, power_W, apparent
        power_w = pd.to_numeric(frame.iloc[:, 10], errors="coerce")
        daily_kwh = pd.to_numeric(frame.iloc[:, 9], errors="coerce")
        total_kwh = pd.to_numeric(frame.iloc[:, 8], errors="coerce")
        freq = pd.to_numeric(frame.iloc[:, 4], errors="coerce")
    else:
        # Export skipped voltages: Timestamp, freq, total_kWh, daily_kWh, power_W, ...
        freq = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
        total_kwh = pd.to_numeric(frame.iloc[:, 2], errors="coerce")
        daily_kwh = pd.to_numeric(frame.iloc[:, 3], errors="coerce")
        power_w = pd.to_numeric(frame.iloc[:, 4], errors="coerce")

    out = pd.DataFrame(
        {
            "ts": frame["_ts"].to_numpy(),
            "power_w": power_w.to_numpy(),
            "daily_kwh": daily_kwh.to_numpy(),
            "total_kwh": total_kwh.to_numpy(),
            "freq": freq.to_numpy(),
        }
    )
    out["site_id"] = site_id
    # Drop commissioning garbage (zero frequency, impossible power).
    out = out[(out["freq"].isna()) | ((out["freq"] > 45) & (out["freq"] < 55))]
    out = out[(out["power_w"] >= 0) & (out["power_w"] < 50_000)]
    return out.dropna(subset=["power_w"])


def _to_hourly(fine: pd.DataFrame) -> pd.DataFrame:
    fine = fine.set_index("ts").sort_index()
    hourly = fine.resample("h").agg(
        power_w=("power_w", "mean"),
        n=("power_w", "count"),
        daily_kwh=("daily_kwh", "max"),
        total_kwh=("total_kwh", "max"),
    )
    hourly = hourly[hourly["n"] >= 3]
    hourly["load_kw"] = hourly["power_w"] / 1000.0
    # Fill isolated 1–3 hour gaps.
    hourly = hourly.asfreq("h")
    hourly["load_kw"] = hourly["load_kw"].interpolate(limit=3)
    hourly["gap"] = hourly["load_kw"].isna()
    hourly["load_kw"] = hourly["load_kw"].ffill(limit=1)
    return hourly.reset_index().rename(columns={"ts": "ts"})


def _open_meteo(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(WEATHER_HOURLY),
        "timezone": TZ,
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = json.loads(resp.read().decode())
    hourly = payload["hourly"]
    wx = pd.DataFrame(hourly)
    wx["ts"] = pd.to_datetime(wx["time"]).dt.tz_localize(TZ)
    return wx.drop(columns=["time"])


def _open_meteo_forecast(lat: float, lon: float, days: int = 7) -> pd.DataFrame:
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "hourly": ",".join(WEATHER_HOURLY),
        "forecast_days": str(days),
        "timezone": TZ,
    }
    url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as resp:
        payload = json.loads(resp.read().decode())
    wx = pd.DataFrame(payload["hourly"])
    wx["ts"] = pd.to_datetime(wx["time"]).dt.tz_localize(TZ)
    return wx.drop(columns=["time"])


def add_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    ts = frame["ts"]
    hour = ts.dt.hour
    dow = ts.dt.dayofweek
    doy = ts.dt.dayofyear
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    frame["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    frame["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    frame["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    frame["is_weekend"] = (dow >= 5).astype(float)
    frame["is_holiday"] = ts.dt.strftime("%Y-%m-%d").isin(HOLIDAYS).astype(float)
    return frame


def prepare(xlsx: Path = SOURCE_XLSX) -> pd.DataFrame:
    if not xlsx.exists():
        raise FileNotFoundError(xlsx)

    parts = []
    for site_id, meta in SITES.items():
        print(f"parsing {site_id} …", flush=True)
        fine = _load_5min(xlsx, site_id, meta)
        hourly = _to_hourly(fine)
        hourly["site_id"] = site_id
        start = hourly["ts"].min().date() - timedelta(days=1)
        end = (hourly["ts"].max() + timedelta(days=3)).date()
        print(f"  weather {meta['city']} {start} → {end}", flush=True)
        hist = _open_meteo(meta["lat"], meta["lon"], start, min(end, date.today()))
        try:
            fc = _open_meteo_forecast(meta["lat"], meta["lon"], days=8)
        except Exception as exc:  # noqa: BLE001
            print(f"  forecast weather skipped: {exc}", flush=True)
            fc = hist.iloc[0:0].copy()
        wx = pd.concat([hist, fc], ignore_index=True).drop_duplicates("ts", keep="last")
        merged = hourly.merge(wx, on="ts", how="left")
        # Keep future weather rows so we can issue a live 48 h forecast.
        future = wx[wx["ts"] > hourly["ts"].max()].copy()
        future["site_id"] = site_id
        future["load_kw"] = np.nan
        future["power_w"] = np.nan
        future["n"] = 0
        future["gap"] = True
        merged = pd.concat([merged, future], ignore_index=True, sort=False)
        merged = add_calendar(merged)
        merged = merged.sort_values("ts")
        parts.append(merged)
        print(
            f"  hours with load={merged['load_kw'].notna().sum()} "
            f"span={merged['ts'].min()} → {merged['ts'].max()}",
            flush=True,
        )

    panel = pd.concat(parts, ignore_index=True)
    panel_path = DATA_DIR / "hourly_panel.parquet"
    try:
        panel.to_parquet(panel_path, index=False)
    except Exception:
        panel_path = DATA_DIR / "hourly_panel.csv"
        panel.to_csv(panel_path, index=False)
    csv_path = DATA_DIR / "hourly_panel.csv"
    panel.to_csv(csv_path, index=False)
    print(f"wrote {panel_path} rows={len(panel)}", flush=True)
    return panel


if __name__ == "__main__":
    prepare(Path(sys.argv[1]) if len(sys.argv) > 1 else SOURCE_XLSX)
