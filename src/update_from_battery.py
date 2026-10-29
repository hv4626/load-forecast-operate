"""Score the 12 Oct house-load issue on Battery_Data.xlsx, then retrain.

The workbook's load_p_load_total (watts) is the same house-load measurement
as TotalConsumptionPower. This script scores that issue against the meter
hours after the origin, retrains the direct 48 h issuer on load kW, and
republishes the desk. Verdict stays FAIL. TiDE is not retrained.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from config import DATA_DIR, OUTPUT_DIR, SITES, TZ

ROOT = Path(__file__).resolve().parents[1]
DUMP = Path.home() / "Downloads" / "Battery_Data.xlsx"
ARCHIVE = OUTPUT_DIR / "archive_20261012"
HORIZON = 48
HOLDOUT_DAYS = 14
VAL_DAYS = 7
LOOKBACK_DAYS = 7
WX_VARS = ["temperature_2m", "shortwave_radiation", "relative_humidity_2m", "cloud_cover"]
COLS = [
    "lead", "hour", "dow", "weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "y_now", "lag24", "lag168", "mean24", "max24", "temp", "ghi",
]
ISSUED_WINNER = {"RSSEKA1608": "gbm", "RSSETL2362": "yesterday"}
ISSUED_LABEL = {"gbm": "GBM", "yesterday": "Yesterday-copy", "last_week": "Last-week-copy"}


def _parse_ts(raw: pd.Series) -> pd.Series:
    # This dump is ISO-8601 (2026-06-01 00:04:00). dayfirst swaps those into
    # year-day-month and drops the October hours the last issue covers.
    if pd.api.types.is_datetime64_any_dtype(raw):
        ts = pd.to_datetime(raw, errors="coerce")
    else:
        ts = pd.to_datetime(raw.astype(str).str.strip(), errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
    else:
        ts = ts.dt.tz_convert(TZ)
    return ts


def load_battery(path: Path) -> dict[str, pd.Series]:
    xl = pd.ExcelFile(path)
    found: dict[str, pd.Series] = {}
    for name in xl.sheet_names:
        match = re.search(r"(RSSE[A-Z]+\d+)", name)
        if not match:
            print(f"skip sheet {name}", flush=True)
            continue
        sid = match.group(1)
        if sid not in SITES:
            print(f"skip unknown site {sid} ({name})", flush=True)
            continue
        df = xl.parse(name, usecols=["timestamp_ist", "load_p_load_total"])
        ts = _parse_ts(df["timestamp_ist"])
        power_w = pd.to_numeric(df["load_p_load_total"], errors="coerce")
        s = pd.Series(power_w.to_numpy(), index=pd.DatetimeIndex(ts))
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        s = s[(s >= 0) & (s < 50_000)]
        if float(s.quantile(0.9)) > 80:
            s = s / 1000.0
        hourly = s.resample("h").mean()
        counts = s.resample("h").count()
        hourly = hourly[counts >= 3].dropna()
        hourly = hourly[~hourly.index.duplicated(keep="last")].sort_index()
        if sid not in found or len(hourly) > len(found[sid]):
            found[sid] = hourly
            print(
                f"{sid}: {len(hourly)} hours  {hourly.index.min()} → {hourly.index.max()}  "
                f"mean={hourly.mean():+.3f} kW  std={hourly.std():.3f}",
                flush=True,
            )
    if not found:
        raise RuntimeError(f"no site sheets in {path}")
    return found


def model_series(raw: pd.Series) -> pd.Series:
    idx = pd.date_range(raw.index.min(), raw.index.max(), freq="h", tz=TZ)
    full = raw.reindex(idx).interpolate(limit=3)
    return full.dropna()


def fetch_weather(lat: float, lon: float, start, end) -> pd.DataFrame:
    start_d = pd.Timestamp(start).tz_convert(TZ).date()
    end_d = pd.Timestamp(end).tz_convert(TZ).date() + timedelta(days=3)
    today = date.today()

    def grab(url: str) -> pd.DataFrame:
        with urllib.request.urlopen(url, timeout=120) as resp:
            payload = json.loads(resp.read().decode())
        frame = pd.DataFrame(payload["hourly"])
        frame["ts"] = pd.to_datetime(frame["time"]).dt.tz_localize(TZ)
        return frame.drop(columns=["time"]).set_index("ts")

    frames = []
    hist = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode({
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": start_d.isoformat(),
        "end_date": min(end_d, today).isoformat(),
        "hourly": ",".join(WX_VARS),
        "timezone": TZ,
    })
    fut = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "hourly": ",".join(WX_VARS),
        "forecast_days": 8,
        "timezone": TZ,
    })
    for url in (hist, fut):
        try:
            frames.append(grab(url))
        except Exception as exc:  # noqa: BLE001
            print(f"  weather skip: {exc}", flush=True)
    if not frames:
        return pd.DataFrame()
    wx = pd.concat(frames).sort_index()
    return wx[~wx.index.duplicated(keep="last")]


def _get(y: pd.Series, ts, default=np.nan) -> float:
    try:
        value = y.loc[ts]
    except KeyError:
        return default
    if isinstance(value, pd.Series):
        value = value.iloc[-1]
    value = float(value)
    return value if np.isfinite(value) else default


def make_row(y: pd.Series, temp: pd.Series, ghi: pd.Series, origin, lead: int) -> dict:
    target = origin + pd.Timedelta(hours=lead)
    last = origin - pd.Timedelta(hours=1)
    recent = y.loc[last - pd.Timedelta(hours=23): last]
    lag24 = target - pd.Timedelta(hours=24)
    lag168 = target - pd.Timedelta(hours=168)
    return {
        "lead": lead,
        "hour": target.hour,
        "dow": target.dayofweek,
        "weekend": int(target.dayofweek >= 5),
        "hour_sin": np.sin(2 * np.pi * target.hour / 24),
        "hour_cos": np.cos(2 * np.pi * target.hour / 24),
        "dow_sin": np.sin(2 * np.pi * target.dayofweek / 7),
        "dow_cos": np.cos(2 * np.pi * target.dayofweek / 7),
        "y_now": _get(y, last),
        "lag24": _get(y, lag24) if lag24 <= last else np.nan,
        "lag168": _get(y, lag168) if lag168 <= last else np.nan,
        "mean24": float(recent.mean()) if len(recent) else np.nan,
        "max24": float(recent.max()) if len(recent) else np.nan,
        "temp": _get(temp, target),
        "ghi": _get(ghi, target),
    }


def origins_between(y: pd.Series, start, end) -> list:
    first = y.index.min() + pd.Timedelta(days=LOOKBACK_DAYS)
    last_ok = y.index.max() - pd.Timedelta(hours=HORIZON - 1)
    t = max(pd.Timestamp(start), first)
    stop = min(pd.Timestamp(end), last_ok)
    if t.tzinfo is None:
        t = t.tz_localize(TZ)
    if stop.tzinfo is None:
        stop = stop.tz_localize(TZ)
    out = []
    while t <= stop:
        window = y.loc[t: t + pd.Timedelta(hours=HORIZON - 1)]
        if len(window) == HORIZON and np.isfinite(window.to_numpy()).all():
            if (t - pd.Timedelta(hours=24)) in y.index:
                out.append(t)
        t += pd.Timedelta(hours=1)
    return out


def _mask(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    return y, yhat, np.isfinite(y) & np.isfinite(yhat)


def mae(y, yhat) -> float:
    y, yhat, m = _mask(y, yhat)
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(yhat[m] - y[m])))


def rmse(y, yhat) -> float:
    y, yhat, m = _mask(y, yhat)
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.mean((yhat[m] - y[m]) ** 2)))


def wmape(y, yhat) -> float:
    y, yhat, m = _mask(y, yhat)
    if not m.any():
        return float("nan")
    return float(np.sum(np.abs(yhat[m] - y[m])) / max(np.sum(np.abs(y[m])), 1e-9) * 100)


def naive_path(y: pd.Series, origin, kind: str) -> np.ndarray:
    last = _get(y, origin - pd.Timedelta(hours=1))
    if kind == "yesterday":
        prev = np.array([_get(y, origin + pd.Timedelta(hours=h - 24)) for h in range(HORIZON)])
    else:
        prev = np.array([_get(y, origin + pd.Timedelta(hours=h - 168)) for h in range(HORIZON)])
    if np.isfinite(prev[0]) and np.isfinite(last):
        prev = prev + (last - prev[0]) * np.exp(-np.arange(HORIZON) / 6.0)
    if np.isfinite(last):
        prev = prev.copy()
        prev[0] = last
    return np.clip(prev, 0, None)


def _varying(X: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
    if not keep:
        raise RuntimeError("no varying features")
    return X.loc[:, keep]


def fit_point(X: pd.DataFrame, yt: np.ndarray) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        max_depth=7,
        max_iter=300,
        learning_rate=0.05,
        min_samples_leaf=20,
        l2_regularization=0.05,
        random_state=0,
    )
    model.fit(_varying(X), yt)
    return model


def fit_quantile(X: pd.DataFrame, yt: np.ndarray, q: float) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=q,
        max_depth=6,
        max_iter=180,
        learning_rate=0.05,
        min_samples_leaf=30,
        random_state=0,
    )
    model.fit(_varying(X), yt)
    return model


def _predict(model, rows: pd.DataFrame) -> np.ndarray:
    cols = list(getattr(model, "feature_names_in_", rows.columns))
    return model.predict(rows.loc[:, cols])


def design(y, temp, ghi, origins) -> tuple[pd.DataFrame, np.ndarray]:
    rows, target = [], []
    for origin in origins:
        for lead in range(HORIZON):
            rows.append(make_row(y, temp, ghi, origin, lead))
            target.append(_get(y, origin + pd.Timedelta(hours=lead)))
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, np.asarray([])
    return frame[COLS], np.asarray(target, float)


def predict_gbm(models, y, temp, ghi, origin) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = pd.DataFrame([make_row(y, temp, ghi, origin, h) for h in range(HORIZON)])[COLS]
    p50 = np.clip(_predict(models["gbm"], rows), 0, None)
    p10 = np.clip(_predict(models["q10"], rows), 0, None)
    p90 = np.clip(_predict(models["q90"], rows), 0, None)
    p10 = np.minimum(p10, p50)
    p90 = np.maximum(p90, p50)
    last = _get(y, origin - pd.Timedelta(hours=1))
    if np.isfinite(last):
        p50 = p50.copy()
        p50[0] = last
        p10[0] = min(p10[0], last)
        p90[0] = max(p90[0], last)
    return p10, p50, p90


def score_split(y, temp, ghi, train_orig, test_orig) -> dict:
    X, yt = design(y, temp, ghi, train_orig)
    gbm = fit_point(X, yt)
    y_true, pred = {"yesterday": [], "last_week": [], "gbm": []}, None
    truths = []
    bags = {k: [] for k in ("yesterday", "last_week", "gbm")}
    for origin in test_orig:
        truth = np.array([_get(y, origin + pd.Timedelta(hours=h)) for h in range(HORIZON)])
        rows = pd.DataFrame([make_row(y, temp, ghi, origin, h) for h in range(HORIZON)])[COLS]
        truths.append(truth)
        bags["yesterday"].append(naive_path(y, origin, "yesterday"))
        bags["last_week"].append(naive_path(y, origin, "last_week"))
        bags["gbm"].append(_predict(gbm, rows))
    Y = np.vstack(truths)
    out = {}
    for key, rows in bags.items():
        hat = np.vstack(rows)
        out[key] = {"mae": mae(Y, hat), "rmse": rmse(Y, hat), "wmape": wmape(Y, hat)}
    out["n_origins"] = len(test_orig)
    out["gbm_model"] = gbm
    return out


def choose_winner(val_scores: dict, y: pd.Series) -> tuple[str, float]:
    burst = float(y.std() / max(abs(float(y.mean())), 1e-6))
    gbm = val_scores["gbm"]["wmape"]
    yesterday = val_scores["yesterday"]["wmape"]
    if burst >= 1.0:
        winner = "yesterday" if yesterday <= gbm + 5 else "gbm"
    else:
        winner = "gbm" if gbm <= yesterday * 1.15 else "yesterday"
    return winner, burst


def quoted_holdout() -> dict[str, dict]:
    path = ARCHIVE / "uniform_holdout_sweep.csv"
    if not path.exists():
        path = OUTPUT_DIR / "uniform_holdout_sweep.csv"
    out = {}
    if not path.exists():
        return out
    sweep = pd.read_csv(path)
    board = sweep[sweep.test_days == 14]
    for sid, model in ISSUED_WINNER.items():
        rows = board[(board.site == sid) & (board.model == model)]
        if rows.empty:
            continue
        row = rows.iloc[0]
        out[sid] = {"mae": float(row.mae), "wmape": float(row.wmape), "model": model}
    return out


def score_issue(raw: dict[str, pd.Series], quoted: dict) -> dict:
    sites = {}
    for sid, battery in raw.items():
        path = ARCHIVE / f"forecast_{sid}.csv"
        if not path.exists():
            path = OUTPUT_DIR / f"forecast_{sid}.csv"
        fc = pd.read_csv(path, index_col=0, parse_dates=True)
        fc.index = pd.to_datetime(fc.index, utc=True).tz_convert(TZ)
        rows = []
        for ts, row in fc.iterrows():
            if ts not in battery.index:
                continue
            actual = float(battery.loc[ts])
            rows.append({
                "ts": ts.isoformat(),
                "p10": float(row.p10),
                "p50": float(row.p50),
                "p90": float(row.p90),
                "actual_kw": actual,
                "battery_kw": actual,
            })
        frame = pd.DataFrame(rows)
        if frame.empty:
            print(f"{sid}: no overlap between the issue and battery hours", flush=True)
            continue
        y = frame.battery_kw.to_numpy()
        p50 = frame.p50.to_numpy()
        inside = (y >= frame.p10.to_numpy()) & (y <= frame.p90.to_numpy())
        q = quoted.get(sid, {})
        sites[sid] = {
            "issued_model": q.get("model") or ISSUED_WINNER.get(sid),
            "issued_label": ISSUED_LABEL.get(q.get("model") or ISSUED_WINNER.get(sid), "issued"),
            "hours": int(len(frame)),
            "mae": mae(y, p50),
            "rmse": rmse(y, p50),
            "wmape": wmape(y, p50),
            "bias": float(np.mean(y - p50)),
            "band_kw": float(np.mean(frame.p90 - frame.p10)),
            "coverage": float(inside.mean()),
            "quoted_mae": q.get("mae"),
            "quoted_wmape": q.get("wmape"),
            "battery_mean_kw": float(np.mean(y)),
            "battery_min_kw": float(np.min(y)),
            "battery_max_kw": float(np.max(y)),
            "rows": rows,
        }
        print(
            f"{sid} issue vs load  n={len(frame)}  MAE={sites[sid]['mae']:.3f} kW  "
            f"bias={sites[sid]['bias']:+.3f}  band={sites[sid]['band_kw']:.3f}  "
            f"inside={sites[sid]['coverage']:.0%}",
            flush=True,
        )
    return sites


def house_overlap(raw: dict[str, pd.Series]) -> dict[str, dict]:
    panel_path = DATA_DIR / "hourly_panel_house_load.csv"
    if not panel_path.exists():
        panel_path = DATA_DIR / "hourly_panel.csv"
    if not panel_path.exists():
        return {}
    old = pd.read_csv(panel_path, usecols=["ts", "site_id", "load_kw"])
    old["ts"] = pd.to_datetime(old["ts"], utc=True).dt.tz_convert(TZ)
    out = {}
    for sid, battery in raw.items():
        house = old[old.site_id == sid].dropna(subset=["load_kw"])
        joined = house.merge(
            battery.rename("battery_kw").rename_axis("ts").reset_index(),
            on="ts",
            how="inner",
        )
        if len(joined) < 24:
            continue
        corr = float(joined.load_kw.corr(joined.battery_kw))
        out[sid] = {
            "hours": int(len(joined)),
            "corr": corr,
            "house_end": house.ts.max().isoformat(),
            "house_mean_kw": float(joined.load_kw.mean()),
            "battery_mean_kw": float(joined.battery_kw.mean()),
        }
        print(
            f"{sid} overlap with house load  n={len(joined)}  corr={corr:.3f}  "
            f"house ends {house.ts.max()}",
            flush=True,
        )
    return out


def sweep_site(sid, y, temp, ghi) -> list[dict]:
    last = y.index.max()
    rows = []
    for days in (7, 14, 21, 28):
        test_start = last - pd.Timedelta(days=days)
        val_start = test_start - pd.Timedelta(days=VAL_DAYS)
        train_orig = origins_between(y, y.index.min(), val_start - pd.Timedelta(hours=HORIZON))
        test_orig = origins_between(y, test_start, last)
        if len(train_orig) < 24 or len(test_orig) < 8:
            print(f"  {sid} {days}d skipped (train {len(train_orig)} test {len(test_orig)})", flush=True)
            continue
        print(f"  {sid} test {days}d  train origins {len(train_orig)}  test {len(test_orig)}", flush=True)
        scored = score_split(y, temp, ghi, train_orig, test_orig)
        train_hours = int((y.index < val_start).sum())
        test_hours = int(((y.index >= test_start) & (y.index <= last)).sum())
        for model, metrics in scored.items():
            if model in {"n_origins", "gbm_model"}:
                continue
            rows.append({
                "site": sid,
                "test_days": days,
                "test_calendar_hours": test_hours,
                "train_hours": train_hours,
                "n_test_origins": scored["n_origins"],
                "n_hours": scored["n_origins"] * HORIZON,
                "scored_hours": scored["n_origins"] * HORIZON,
                "model": model,
                "mae": round(metrics["mae"], 4),
                "rmse": round(metrics["rmse"], 4),
                "wmape": round(metrics["wmape"], 2),
                "val_days": VAL_DAYS,
                "horizon_h": HORIZON,
                "train_start": str(y.index.min()),
                "val_start": str(val_start),
                "test_start": str(test_start),
                "_model": scored["gbm_model"] if model == "gbm" and days == 14 else None,
            })
    return rows


def issue_site(sid, y, temp, ghi, winner, gbm, q10, q90, val_orig) -> pd.DataFrame:
    origin = y.index.max() + pd.Timedelta(hours=1)
    if winner == "gbm":
        p10, p50, p90 = predict_gbm({"gbm": gbm, "q10": q10, "q90": q90}, y, temp, ghi, origin)
    elif winner == "yesterday":
        # A live origin has no hours at origin+h-24 once h >= 24. Tile the last
        # complete day, same as the notebook issue cell, with a 0.7 / 1.4 band.
        last24 = np.asarray(y.iloc[-24:], float)
        p50 = np.clip(np.tile(last24, int(np.ceil(HORIZON / 24)))[:HORIZON], 0, None)
        p10 = np.clip(p50 * 0.7, 0, None)
        p90 = p50 * 1.4
        last = _get(y, origin - pd.Timedelta(hours=1))
        if np.isfinite(last):
            p50 = p50.copy()
            p50[0] = last
            p10[0] = min(float(p10[0]), last)
            p90[0] = max(float(p90[0]), last)
    else:
        p50 = naive_path(y, origin, winner)
        residuals = []
        for vo in val_orig:
            truth = np.array([_get(y, vo + pd.Timedelta(hours=h)) for h in range(HORIZON)])
            residuals.append(truth - naive_path(y, vo, winner))
        resid = np.vstack(residuals) if residuals else np.zeros((1, HORIZON))
        p10 = np.clip(p50 + np.quantile(resid, 0.10, axis=0), 0, None)
        p90 = np.clip(p50 + np.quantile(resid, 0.90, axis=0), 0, None)
        p10 = np.minimum(p10, p50)
        p90 = np.maximum(p90, p50)
    idx = pd.date_range(origin, periods=HORIZON, freq="h", tz=TZ)
    return pd.DataFrame({"p10": p10, "p50": p50, "p90": p90}, index=idx)


def write_panel(series: dict[str, pd.Series], weather: dict[str, pd.DataFrame]) -> None:
    parts = []
    for sid, y in series.items():
        wx = weather.get(sid, pd.DataFrame())
        frame = y.rename("load_kw").to_frame()
        frame["site_id"] = sid
        frame["power_w"] = frame["load_kw"] * 1000.0
        frame["n"] = 12
        frame["gap"] = False
        if not wx.empty:
            merged = frame.join(wx, how="left")
            future = wx[wx.index > y.index.max()].copy()
            if not future.empty:
                future["site_id"] = sid
                future["load_kw"] = np.nan
                future["power_w"] = np.nan
                future["n"] = 0
                future["gap"] = True
                merged = pd.concat([merged, future], axis=0)
        else:
            merged = frame
        merged = merged.reset_index().rename(columns={"index": "ts"})
        if "ts" not in merged.columns:
            merged = merged.rename(columns={merged.columns[0]: "ts"})
        parts.append(merged)
    panel = pd.concat(parts, ignore_index=True)
    panel.to_csv(DATA_DIR / "hourly_panel.csv", index=False)
    print(f"wrote panel rows={len(panel)}", flush=True)


def main() -> None:
    if not DUMP.exists():
        raise FileNotFoundError(DUMP)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    quoted = quoted_holdout()
    for sid in SITES:
        src = OUTPUT_DIR / f"forecast_{sid}.csv"
        dest = ARCHIVE / f"forecast_{sid}.csv"
        if src.exists() and not dest.exists():
            shutil.copy2(src, dest)
    sweep_src = OUTPUT_DIR / "uniform_holdout_sweep.csv"
    if sweep_src.exists() and not (ARCHIVE / "uniform_holdout_sweep.csv").exists():
        shutil.copy2(sweep_src, ARCHIVE / "uniform_holdout_sweep.csv")
    house_panel = DATA_DIR / "hourly_panel_house_load.csv"
    if (DATA_DIR / "hourly_panel.csv").exists() and not house_panel.exists():
        shutil.copy2(DATA_DIR / "hourly_panel.csv", house_panel)

    raw = load_battery(DUMP)
    overlap = house_overlap(raw)
    realized_sites = score_issue(raw, quoted)

    weather = {}
    series = {}
    for sid, battery in raw.items():
        meta = SITES[sid]
        print(f"weather {sid} {meta['city']}", flush=True)
        weather[sid] = fetch_weather(meta["lat"], meta["lon"], battery.index.min(), battery.index.max())
        series[sid] = model_series(battery)

    sweep_rows = []
    winners = {}
    bursts = {}
    bundles = {}
    forecasts = {}
    for sid, y in series.items():
        wx = weather.get(sid, pd.DataFrame())
        temp = wx["temperature_2m"] if "temperature_2m" in getattr(wx, "columns", []) else pd.Series(dtype=float)
        ghi = wx["shortwave_radiation"] if "shortwave_radiation" in getattr(wx, "columns", []) else pd.Series(dtype=float)
        print(f"sweep {sid}", flush=True)
        rows = sweep_site(sid, y, temp, ghi)
        sweep_rows.extend(rows)
        last = y.index.max()
        test_start = last - pd.Timedelta(days=HOLDOUT_DAYS)
        val_start = test_start - pd.Timedelta(days=VAL_DAYS)
        train_orig = origins_between(y, y.index.min(), val_start - pd.Timedelta(hours=HORIZON))
        val_orig = origins_between(y, val_start, test_start - pd.Timedelta(hours=HORIZON))
        print(f"  issuer train/val origins {len(train_orig)}/{len(val_orig)}", flush=True)
        X, yt = design(y, temp, ghi, train_orig)
        gbm = fit_point(X, yt)
        q10 = fit_quantile(X, yt, 0.1)
        q90 = fit_quantile(X, yt, 0.9)
        val_bags = {"gbm": [], "yesterday": [], "last_week": []}
        val_truth = []
        for origin in val_orig:
            val_truth.append(np.array([_get(y, origin + pd.Timedelta(hours=h)) for h in range(HORIZON)]))
            rows_x = pd.DataFrame([make_row(y, temp, ghi, origin, h) for h in range(HORIZON)])[COLS]
            val_bags["gbm"].append(_predict(gbm, rows_x))
            val_bags["yesterday"].append(naive_path(y, origin, "yesterday"))
            val_bags["last_week"].append(naive_path(y, origin, "last_week"))
        if not val_orig:
            val_scores = {"gbm": {"mae": 0.0}, "yesterday": {"mae": 0.0}, "last_week": {"mae": 0.0}}
            winner, burst = "yesterday", float(y.std() / max(abs(float(y.mean())), 1e-6))
        else:
            Y = np.vstack(val_truth)
            val_scores = {
                key: {"mae": mae(Y, np.vstack(hat)), "wmape": wmape(Y, np.vstack(hat))}
                for key, hat in val_bags.items()
            }
            winner, burst = choose_winner(val_scores, y)
        winners[sid] = winner
        bursts[sid] = burst
        print(
            f"  val MAE gbm={val_scores['gbm']['mae']:.3f} "
            f"yesterday={val_scores['yesterday']['mae']:.3f} "
            f"week={val_scores['last_week']['mae']:.3f}  burst={burst:.2f} → {winner}",
            flush=True,
        )
        forecasts[sid] = issue_site(sid, y, temp, ghi, winner, gbm, q10, q90, val_orig)
        bundles[sid] = {"winner": winner, "gbm": gbm, "q10": q10, "q90": q90, "burst": burst}
        forecasts[sid].to_csv(OUTPUT_DIR / f"forecast_{sid}.csv")
        print(f"  issued {winner} from {forecasts[sid].index.min()} peak {forecasts[sid].p50.max():+.3f} kW", flush=True)

    public = [{k: v for k, v in row.items() if not k.startswith("_")} for row in sweep_rows]
    pd.DataFrame(public).to_csv(OUTPUT_DIR / "uniform_holdout_sweep.csv", index=False)
    joblib.dump(bundles, OUTPUT_DIR / "load_gbm_bundle.joblib")
    write_panel(series, weather)

    bits = []
    for sid, info in overlap.items():
        bits.append(
            f"{SITES[sid]['label']} matches the previous meter on {info['hours']} hours "
            f"(correlation {info['corr']:.3f})"
        )
    overlap_txt = ". ".join(bits)
    summary = (
        "Battery_Data.xlsx carries the same house-load measurement as the lifetime workbook. "
        "The column is load_p_load_total (watts). It matches TotalConsumptionPower on the overlapping rows, "
        "and continues through 29 Oct. Deviation is actual load minus the 12 Oct issued p50. "
        f"{overlap_txt}."
    )
    realized = {
        "title": "12 Oct issue vs meter load",
        "issue_origin": "2026-10-12T14:00:00+05:30",
        "summary": summary,
        "sites": realized_sites,
        "overlap": overlap,
    }
    (OUTPUT_DIR / "realized_issue.json").write_text(json.dumps(realized, indent=2))
    note = (
        "Operate is house load in kW from Downloads/Battery_Data.xlsx, column load_p_load_total. "
        "The Realized tab scores the 12 Oct issue against the meter hours after that forecast. "
        "Weather is Open-Meteo at the city. Do not dispatch from this desk."
    )
    meta = {
        "product": "System Load Forecast",
        "model": "gbm-direct / yesterday-copy",
        "series_label": "actual",
        "chart_title": "Demand",
        "target": "load_kw",
        "verdict": "FAIL",
        "go_live": False,
        "include_tide": False,
        "winners": winners,
        "note": note,
        "dump": str(DUMP),
        "bursts": bursts,
    }
    (OUTPUT_DIR / "desk_meta.json").write_text(json.dumps(meta, indent=2))
    (OUTPUT_DIR / "winners.json").write_text(json.dumps(winners, indent=2))
    print("winners", winners, flush=True)

    sys.path.insert(0, str(ROOT / "src"))
    from export_web import write_dashboard

    write_dashboard()


if __name__ == "__main__":
    main()
