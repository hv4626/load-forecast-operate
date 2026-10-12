"""Build Operate JSON and a self-contained dashboard HTML."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from config import HORIZON, OUTPUT_DIR, SITES, WEB_DIR

TZ = ZoneInfo("Asia/Kolkata")


def _iso(ts: str) -> str:
    t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if t.tzinfo is None:
        t = t.replace(tzinfo=TZ)
    return t.astimezone(TZ).isoformat()


def _peak(times: list[str], values: list[float]) -> dict | None:
    if not values:
        return None
    i = max(range(len(values)), key=lambda k: values[k])
    return {"ts": times[i], "kw": values[i]}


def build_payload(report: dict) -> dict:
    live = report.get("live") or {}
    sites_out = {}
    for sid, meta in SITES.items():
        L = live.get(sid)
        if not L:
            continue
        hist_ts = [_iso(t) for t in L["hist_ts"]]
        fut_ts = [_iso(t) for t in L["future_ts"]]
        sites_out[sid] = {
            "id": sid,
            "label": meta["label"],
            "subtitle": meta["subtitle"],
            "city": meta["city"],
            "origin": _iso(L["origin"]),
            "actuals": [
                {"ts": t, "load_kw": float(v), "temp_c": float(temp)}
                for t, v, temp in zip(hist_ts, L["hist_load_kw"], L["hist_temp_c"])
            ],
            "forecast": [
                {
                    "ts": t,
                    "p10": float(a),
                    "p50": float(b),
                    "p90": float(c),
                    "temp_c": float(temp),
                }
                for t, a, b, c, temp in zip(
                    fut_ts, L["p10"], L["p50"], L["p90"], L["future_temp_c"]
                )
            ],
            "peak": _peak(fut_ts, L["p50"]),
            "metrics": report.get("by_site", {}).get(sid, {}),
        }
    return {
        "product": "System Load Forecast",
        "model": "tide-hpo1.0.rooftop",
        "verdict": "FAIL",
        "go_live": False,
        "horizon_h": HORIZON,
        "issue_token": datetime.now(TZ).strftime("%Y%m%dT%H0000%z"),
        "timezone": "Asia/Kolkata",
        "sites": sites_out,
        "verify": {
            "tide": report.get("tide"),
            "gbm": report.get("gbm"),
            "seasonal_naive": report.get("seasonal_naive"),
        },
    }


def write_dashboard(report: dict) -> Path:
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload(report)
    (OUTPUT_DIR / "operate.json").write_text(json.dumps(payload, indent=2))
    template = (WEB_DIR / "index.template.html").read_text()
    html = template.replace("/*__OPERATE_JSON__*/", json.dumps(payload))
    out = WEB_DIR / "index.html"
    out.write_text(html)
    print(f"wrote {out}", flush=True)
    return out
