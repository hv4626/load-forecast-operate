"""Indian house-load shape from Battery_Data.xlsx.

The series is load_p_load_total (watts), not battery_p_batt. Each site keeps
its own hour-of-day curve, split by weekday and weekend when that hour has
enough train samples. The curve is fit only on hours before the caller's cutoff.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import BATTERY_XLSX, SITES, TZ

_HOURLY: pd.DataFrame | None = None
MIN_BUCKET = 8


def hourly_from_battery() -> pd.DataFrame:
    """Hourly mean kW of load_p_load_total, clock in Asia/Kolkata."""
    global _HOURLY
    if _HOURLY is not None:
        return _HOURLY
    if not BATTERY_XLSX.exists():
        raise FileNotFoundError(BATTERY_XLSX)
    parts = []
    for site_id in SITES:
        raw = pd.read_excel(
            BATTERY_XLSX,
            sheet_name=site_id,
            usecols=["timestamp_ist", "load_p_load_total"],
        )
        raw["ts"] = pd.to_datetime(raw["timestamp_ist"], format="ISO8601")
        raw = raw.dropna(subset=["ts"]).set_index("ts").sort_index()
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize(TZ)
        else:
            raw.index = raw.index.tz_convert(TZ)
        hourly = raw["load_p_load_total"].resample("h").mean() / 1000.0
        frame = hourly.rename("load_kw").to_frame()
        frame["site_id"] = site_id
        parts.append(frame.reset_index())
    _HOURLY = pd.concat(parts, ignore_index=True)
    return _HOURLY


def _curve(hours: pd.Series, values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(24, np.float64)
    count = np.zeros(24, np.int64)
    for hour, value in zip(hours.to_numpy(), values.to_numpy(), strict=True):
        if not np.isfinite(value):
            continue
        total[int(hour)] += float(value)
        count[int(hour)] += 1
    curve = np.full(24, np.nan, np.float64)
    filled = count > 0
    curve[filled] = total[filled] / count[filled]
    return curve, count


def fit_profiles(cutoffs: dict[str, pd.Timestamp]) -> dict[str, dict[str, np.ndarray]]:
    """Per site: hour, weekday, weekend curves in kW, using rows before cutoff."""
    hourly = hourly_from_battery()
    profiles: dict[str, dict[str, np.ndarray]] = {}
    for site_id, cutoff in cutoffs.items():
        sub = hourly[hourly["site_id"] == site_id].dropna(subset=["load_kw"])
        sub = sub[sub["ts"] < cutoff]
        hour_curve, hour_n = _curve(sub["ts"].dt.hour, sub["load_kw"])
        if not np.isfinite(hour_curve).all():
            fill = float(np.nanmean(hour_curve)) if np.isfinite(hour_curve).any() else 0.0
            hour_curve = np.where(np.isfinite(hour_curve), hour_curve, fill)
        weekday = sub[sub["ts"].dt.dayofweek < 5]
        weekend = sub[sub["ts"].dt.dayofweek >= 5]
        wd_curve, wd_n = _curve(weekday["ts"].dt.hour, weekday["load_kw"])
        we_curve, we_n = _curve(weekend["ts"].dt.hour, weekend["load_kw"])
        profiles[site_id] = {
            "hour": hour_curve.astype(np.float32),
            "n_hour": hour_n,
            "weekday": wd_curve.astype(np.float32),
            "n_weekday": wd_n,
            "weekend": we_curve.astype(np.float32),
            "n_weekend": we_n,
        }
        peak = int(np.nanargmax(hour_curve))
        print(
            f"  india profile {site_id} peak {peak:02d}:00 "
            f"{hour_curve[peak]:.2f} kW  n={int(hour_n.sum())} before {cutoff}",
            flush=True,
        )
    return profiles


def lookup_kw(profile: dict[str, np.ndarray], ts: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """kW at each timestamp. Weekend or weekday curve when that hour has enough samples."""
    hour = np.asarray(ts.hour if hasattr(ts, "hour") else pd.DatetimeIndex(ts).hour, dtype=int)
    weekend = np.asarray(
        (ts.dayofweek if hasattr(ts, "dayofweek") else pd.DatetimeIndex(ts).dayofweek) >= 5
    )
    out = profile["hour"][hour].astype(np.float32, copy=True)
    wd = profile["weekday"][hour]
    we = profile["weekend"][hour]
    use_wd = (~weekend) & (profile["n_weekday"][hour] >= MIN_BUCKET) & np.isfinite(wd)
    use_we = weekend & (profile["n_weekend"][hour] >= MIN_BUCKET) & np.isfinite(we)
    out[use_wd] = wd[use_wd]
    out[use_we] = we[use_we]
    return out.astype(np.float32)


def attach_clock(df: pd.DataFrame, profile: dict[str, np.ndarray], holidays: set[str]) -> pd.DataFrame:
    """Add the Indian clock and the battery-file consumption level."""
    out = df.copy()
    ts = out["ts"]
    hour = ts.dt.hour.to_numpy()
    dow = ts.dt.dayofweek.to_numpy()
    doy = ts.dt.dayofyear.to_numpy()
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["is_weekend"] = (dow >= 5).astype(np.float32)
    out["is_holiday"] = ts.dt.strftime("%Y-%m-%d").isin(holidays).astype(np.float32)
    out["india_kw"] = lookup_kw(profile, ts)
    return out
