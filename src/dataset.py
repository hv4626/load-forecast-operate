"""Sliding windows for TiDE plus a tabular matrix for the GBM baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import COVARIATES, HOLIDAYS, HORIZON, LOOKBACK, SITES, TEST_DAYS, VAL_DAYS
from consumption import attach_clock, fit_profiles, lookup_kw

COVS = list(COVARIATES)


@dataclass
class SiteNorm:
    mean: float
    std: float


def site_index_map() -> dict[str, int]:
    return {sid: i for i, sid in enumerate(SITES)}


def load_panel(path) -> pd.DataFrame:
    path = str(path)
    if path.endswith(".parquet"):
        try:
            return pd.read_parquet(path)
        except Exception:
            path = path.replace(".parquet", ".csv")
    return pd.read_csv(path, parse_dates=["ts"])


def _site_frame(panel: pd.DataFrame, site_id: str) -> pd.DataFrame:
    df = panel[panel["site_id"] == site_id].sort_values("ts").copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("Asia/Kolkata")
    return df.reset_index(drop=True)


def split_times(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    observed = df.dropna(subset=["load_kw"])
    last = observed["ts"].max()
    test_start = last - pd.Timedelta(days=TEST_DAYS)
    val_start = test_start - pd.Timedelta(days=VAL_DAYS)
    return val_start, test_start


def fit_norm(df: pd.DataFrame, val_start: pd.Timestamp) -> SiteNorm:
    train_y = np.log1p(df.loc[df["ts"] < val_start, "load_kw"].dropna().clip(lower=0))
    std = float(train_y.std()) or 1.0
    return SiteNorm(mean=float(train_y.mean()), std=std)


def to_model(y: np.ndarray, nrm: SiteNorm) -> np.ndarray:
    return (np.log1p(np.clip(y, 0, None)) - nrm.mean) / nrm.std


def from_model(z: np.ndarray, nrm: SiteNorm) -> np.ndarray:
    return np.clip(np.expm1(z * nrm.std + nrm.mean), 0.0, None)


def make_windows(
    panel: pd.DataFrame,
    split: str,
) -> tuple[dict[str, np.ndarray], dict[str, SiteNorm]]:
    """split in {train, val, test}. Test windows have fully observed horizons."""
    idx = site_index_map()
    frames = {site_id: _site_frame(panel, site_id) for site_id in SITES}
    cutoffs = {site_id: split_times(frames[site_id])[0] for site_id in SITES}
    profiles = fit_profiles(cutoffs)
    y_hist, y_fut, x_hist, x_fut, site, stamp = [], [], [], [], [], []
    profile_fut, profile_now = [], []
    norms: dict[str, SiteNorm] = {}
    covs: list[str] = []

    for site_id in SITES:
        df = attach_clock(frames[site_id], profiles[site_id], HOLIDAYS)
        val_start, test_start = split_times(df)
        norms[site_id] = fit_norm(df, val_start)
        nrm = norms[site_id]
        covs = [c for c in COVS if c in df.columns]
        y = df["load_kw"].to_numpy(dtype=np.float32)
        x = df[covs].to_numpy(dtype=np.float32)
        india = lookup_kw(profiles[site_id], df["ts"])
        # Fill remaining covariate holes with column medians from train.
        train_mask = df["ts"] < val_start
        med = np.nanmedian(x[train_mask.to_numpy()], axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        inds = np.where(np.isnan(x))
        x[inds] = np.take(med, inds[1])

        times = df["ts"]
        n = len(df)
        for t in range(LOOKBACK, n - HORIZON + 1):
            horizon_end = times.iloc[t + HORIZON - 1]
            origin = times.iloc[t - 1]
            y_slice = y[t : t + HORIZON]
            hist_slice = y[t - LOOKBACK : t]
            if not np.isfinite(hist_slice).all() or not np.isfinite(y_slice).all():
                continue
            if split == "train" and horizon_end >= val_start:
                continue
            if split == "val" and (origin < val_start or horizon_end >= test_start):
                continue
            if split == "test" and origin < test_start:
                continue
            if split == "test" and not np.isfinite(y_slice).all():
                continue
            y_hist.append(to_model(hist_slice, nrm))
            y_fut.append(to_model(y_slice, nrm))
            x_hist.append(x[t - LOOKBACK : t])
            x_fut.append(x[t : t + HORIZON])
            site.append(idx[site_id])
            stamp.append(str(times.iloc[t]))
            profile_fut.append(to_model(india[t : t + HORIZON], nrm))
            profile_now.append(to_model(np.array([india[t - 1]], np.float32), nrm)[0])

    n_cov = len(covs) or 1
    arrays = {
        "y_hist": np.stack(y_hist).astype(np.float32) if y_hist else np.zeros((0, LOOKBACK), np.float32),
        "y_fut": np.stack(y_fut).astype(np.float32) if y_fut else np.zeros((0, HORIZON), np.float32),
        "x_hist": np.stack(x_hist) if x_hist else np.zeros((0, LOOKBACK, n_cov), np.float32),
        "x_fut": np.stack(x_fut) if x_fut else np.zeros((0, HORIZON, n_cov), np.float32),
        "site": np.asarray(site, np.int64),
        "origin": np.asarray(stamp),
        "covs": np.asarray(covs),
        "profile_fut": np.stack(profile_fut).astype(np.float32) if profile_fut else np.zeros((0, HORIZON), np.float32),
        "profile_now": np.asarray(profile_now, np.float32),
    }
    return arrays, norms


def last_origin_window(panel: pd.DataFrame, norms: dict[str, SiteNorm]) -> dict[str, dict]:
    """One live Operate issue per site: last fully observed lookback."""
    idx = site_index_map()
    frames = {site_id: _site_frame(panel, site_id) for site_id in SITES}
    cutoffs = {site_id: split_times(frames[site_id])[0] for site_id in SITES}
    profiles = fit_profiles(cutoffs)
    out: dict[str, dict] = {}
    for site_id in SITES:
        df = attach_clock(frames[site_id], profiles[site_id], HOLIDAYS)
        covs = [c for c in COVS if c in df.columns]
        observed = np.flatnonzero(df["load_kw"].notna().to_numpy())
        if len(observed) == 0:
            continue
        last_i = int(observed[-1])
        if last_i + 1 < LOOKBACK:
            continue
        start = last_i - LOOKBACK + 1
        hist = df.iloc[start : last_i + 1]
        fut = df.iloc[last_i + 1 : last_i + 1 + HORIZON]
        if len(fut) < HORIZON:
            continue
        nrm = norms[site_id]
        x_hist = hist[covs].to_numpy(np.float32)
        x_fut = fut[covs].to_numpy(np.float32)
        med = np.nanmedian(x_hist, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        for arr in (x_hist, x_fut):
            inds = np.where(np.isnan(arr))
            arr[inds] = np.take(med, inds[1])
        y_hist = to_model(hist["load_kw"].to_numpy(np.float32), nrm).astype(np.float32)
        india = lookup_kw(profiles[site_id], df["ts"])
        out[site_id] = {
            "y_hist": y_hist,
            "x_hist": x_hist,
            "x_fut": x_fut,
            "site": idx[site_id],
            "profile_fut": to_model(india[last_i + 1 : last_i + 1 + HORIZON], nrm).astype(np.float32),
            "profile_now": np.float32(to_model(np.array([india[last_i]], np.float32), nrm)[0]),
            "origin": hist["ts"].iloc[-1],
            "future_ts": fut["ts"].tolist(),
            "hist_ts": hist["ts"].tolist(),
            "hist_load": hist["load_kw"].to_numpy(np.float32),
            "future_temp": fut["temperature_2m"].to_numpy(np.float32)
            if "temperature_2m" in fut
            else np.zeros(HORIZON, np.float32),
            "hist_temp": hist["temperature_2m"].to_numpy(np.float32)
            if "temperature_2m" in hist
            else np.zeros(LOOKBACK, np.float32),
        }
    return out


def gbm_matrix(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """One row per (window, lead): recent load + that hour's covariates + lead index."""
    y_hist = arrays["y_hist"]
    x_fut = arrays["x_fut"]
    y_fut = arrays["y_fut"]
    site = arrays["site"]
    n = len(y_hist)
    if n == 0:
        return np.zeros((0, 1), np.float32), np.zeros((0,), np.float32)
    onehot = np.eye(len(SITES), dtype=np.float32)[site]
    y_recent = y_hist[:, -48:]
    rows = []
    targets = []
    for i in range(n):
        for h in range(HORIZON):
            lead = np.array([h / HORIZON], dtype=np.float32)
            rows.append(np.concatenate([y_recent[i], x_fut[i, h], onehot[i], lead]))
            targets.append(y_fut[i, h])
    return np.stack(rows).astype(np.float32), np.asarray(targets, np.float32)
