"""Build Operate JSON and a self-contained dashboard HTML."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from config import DATA_DIR, HORIZON, OUTPUT_DIR, SITES, WEB_DIR

TZ = ZoneInfo("Asia/Kolkata")
HIST_HOURS = 14 * 24  # past data on the desk
WINNERS = {"RSSEKA1608": "gbm", "RSSETL2362": "yesterday"}
MODEL_LABEL = {
    "gbm": "GBM",
    "yesterday": "Yesterday-copy",
    "last_week": "Last-week-copy",
    "tide": "TiDE",
}


def _iso(ts) -> str:
    if isinstance(ts, str):
        t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    else:
        t = pd.Timestamp(ts).to_pydatetime()
    if t.tzinfo is None:
        t = t.replace(tzinfo=TZ)
    return t.astimezone(TZ).isoformat()


def _num(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return float(v)


def _peak(rows: list[dict], key: str = "p50") -> dict | None:
    vals = [r for r in rows if r.get(key) is not None]
    if not vals:
        return None
    best = max(vals, key=lambda r: r[key])
    return {"ts": best["ts"], "kw": best[key]}


def _load_panel() -> pd.DataFrame:
    path = DATA_DIR / "hourly_panel.csv"
    df = pd.read_csv(path, parse_dates=["ts"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(TZ)
    return df


def _issued_forecast(sid: str) -> pd.DataFrame | None:
    p = OUTPUT_DIR / f"forecast_{sid}.csv"
    if not p.exists():
        return None
    fc = pd.read_csv(p, index_col=0, parse_dates=True)
    fc.index = pd.to_datetime(fc.index, utc=True).tz_convert(TZ)
    return fc


def _uniform_metrics() -> pd.DataFrame:
    p = OUTPUT_DIR / "uniform_holdout_sweep.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _read_json(name: str) -> dict:
    p = OUTPUT_DIR / name
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _tide_metrics() -> dict:
    p = OUTPUT_DIR / "production_eval.json"
    if not p.exists():
        return {}
    blob = json.loads(p.read_text())
    return blob.get("by_site") or {}


def _site_metrics(sid: str, sweep: pd.DataFrame, tide: dict) -> dict:
    out = {}
    if not sweep.empty:
        rows = sweep[(sweep.site == sid) & (sweep.test_days == 14)]
        for _, r in rows.iterrows():
            out[r.model] = {
                "mae": _num(r.mae),
                "rmse": _num(r.rmse),
                "wmape": _num(r.wmape),
                "n_hours": int(r.n_hours),
                "train_hours": int(r.train_hours),
                "test_calendar_hours": int(r.test_calendar_hours),
                "n_test_origins": int(r.n_test_origins),
                "scored_hours": int(r.scored_hours),
                "test_days": int(r.test_days),
                "val_days": int(r.val_days),
                "horizon_h": int(r.horizon_h),
                "train_start": str(r.train_start),
                "val_start": str(r.val_start),
                "test_start": str(r.test_start),
            }
    t = tide.get(sid) or {}
    tm = t.get("tide") or t
    if tm.get("mae") is not None:
        out["tide"] = {
            "mae": _num(tm.get("mae")),
            "rmse": _num(tm.get("rmse")),
            "wmape": _num(tm.get("wmape")),
            "mape": _num(tm.get("mape")),
            "n_hours": int(tm.get("n") or 0),
            "n_test_origins": int(t.get("n_windows") or 0),
            "coverage_p10_p90": _num(t.get("coverage_p10_p90")),
        }
    return out


def build_payload(report: dict | None = None) -> dict:
    report = report or {}
    meta = _read_json("desk_meta.json")
    realized = _read_json("realized_issue.json")
    winners = dict(WINNERS)
    winners.update(meta.get("winners") or {})
    panel = _load_panel()
    sweep = _uniform_metrics()
    tide = {} if meta.get("include_tide") is False else _tide_metrics()
    live = (report.get("live") or {})
    sites_out = {}
    realized_sites = realized.get("sites") or {}

    for sid, site_meta in SITES.items():
        sdf = panel[panel["site_id"] == sid].sort_values("ts")
        observed = sdf[sdf["load_kw"].notna()]
        if observed.empty:
            continue
        last = observed.iloc[-1]
        origin = last["ts"]
        hist = observed[observed["ts"] >= origin - pd.Timedelta(hours=HIST_HOURS - 1)]
        fc = _issued_forecast(sid)
        fut_wx = sdf[sdf["ts"] > origin].set_index("ts")

        forecast_rows = []
        if fc is not None:
            for ts, row in fc.iterrows():
                temp = None
                if ts in fut_wx.index:
                    temp = _num(fut_wx.loc[ts, "temperature_2m"]) if "temperature_2m" in fut_wx else None
                forecast_rows.append({
                    "ts": _iso(ts),
                    "p10": _num(row.get("p10")),
                    "p50": _num(row.get("p50")),
                    "p90": _num(row.get("p90")),
                    "temp_c": temp,
                })
        elif sid in live:
            L = live[sid]
            for t, a, b, c, temp in zip(
                L["future_ts"], L["p10"], L["p50"], L["p90"], L.get("future_temp_c") or [None] * len(L["p50"])
            ):
                forecast_rows.append({
                    "ts": _iso(t), "p10": _num(a), "p50": _num(b), "p90": _num(c), "temp_c": _num(temp),
                })

        winner = winners.get(sid, "gbm")
        metrics = _site_metrics(sid, sweep, tide)
        win_m = metrics.get(winner) or {}
        sites_out[sid] = {
            "id": sid,
            "label": site_meta["label"],
            "subtitle": site_meta["subtitle"],
            "city": site_meta["city"],
            "origin": _iso(origin),
            "winner": winner,
            "winner_label": MODEL_LABEL.get(winner, winner),
            "actuals": [
                {
                    "ts": _iso(r.ts),
                    "load_kw": _num(r.load_kw),
                    "temp_c": _num(r.temperature_2m) if hasattr(r, "temperature_2m") else None,
                }
                for r in hist.itertuples()
            ],
            "forecast": forecast_rows,
            "peak": _peak(forecast_rows),
            "metrics": metrics,
            "holdout": {
                "mae": win_m.get("mae"),
                "rmse": win_m.get("rmse"),
                "wmape": win_m.get("wmape"),
                "train_hours": win_m.get("train_hours"),
                "test_calendar_hours": win_m.get("test_calendar_hours"),
                "n_test_origins": win_m.get("n_test_origins"),
                "scored_hours": win_m.get("scored_hours"),
                "test_days": win_m.get("test_days", 14),
                "val_days": win_m.get("val_days", 7),
            },
            "realized": realized_sites.get(sid) or {},
        }

    sweep_rows = []
    if not sweep.empty:
        for _, r in sweep.iterrows():
            sweep_rows.append({
                "site": r.site,
                "label": SITES.get(r.site, {}).get("label", r.site),
                "test_days": int(r.test_days),
                "test_hours": int(r.test_calendar_hours),
                "train_hours": int(r.train_hours),
                "n_origins": int(r.n_test_origins),
                "model": r.model,
                "mae": _num(r.mae),
                "rmse": _num(r.rmse),
                "wmape": _num(r.wmape),
            })

    last_actual = None
    for site in sites_out.values():
        if site.get("origin") and (last_actual is None or site["origin"] > last_actual):
            last_actual = site["origin"]
    return {
        "product": meta.get("product") or "System Load Forecast",
        "model": meta.get("model") or "gbm-direct / yesterday-copy",
        "series_label": meta.get("series_label") or "actual",
        "chart_title": meta.get("chart_title") or "Demand",
        "note": meta.get("note") or "",
        "verdict": meta.get("verdict") or "FAIL",
        "go_live": bool(meta.get("go_live", False)),
        "horizon_h": HORIZON,
        "default_horizon_h": 24,
        "default_lookback_h": 168,
        "issue_token": datetime.now(TZ).strftime("%Y%m%dT%H0000%z"),
        "timezone": "Asia/Kolkata",
        "split": {
            "protocol": "trailing holdout, no shuffle",
            "test_days": 14,
            "val_days": 7,
            "horizon_h": 48,
            "last_actual": last_actual,
        },
        "sites": sites_out,
        "sweep": sweep_rows,
        "realized": {
            "title": realized.get("title"),
            "summary": realized.get("summary"),
            "issue_origin": realized.get("issue_origin"),
        },
        "verify": report.get("verify") or {},
    }


def write_dashboard(report: dict | None = None) -> Path:
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload(report)
    (OUTPUT_DIR / "operate.json").write_text(json.dumps(payload, indent=2))
    template = (WEB_DIR / "index.template.html").read_text()
    html = template.replace("/*__OPERATE_JSON__*/", json.dumps(payload))
    out = WEB_DIR / "index.html"
    out.write_text(html)
    print(f"wrote {out}", flush=True)
    print(f"wrote {OUTPUT_DIR / 'operate.json'}", flush=True)
    return out


if __name__ == "__main__":
    write_dashboard()
