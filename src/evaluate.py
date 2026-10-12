"""Production evaluation: skill vs naive, origin jump, coverage, hour-of-day."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from config import DATA_DIR, HORIZON, LOOKBACK, OUTPUT_DIR, QUANTILES, SITES
from dataset import from_model, load_panel, make_windows, site_index_map
from model import TideQuantile
from train import (
    _filter_site,
    _metrics,
    apply_conformal,
    calibrated_naive,
    denorm,
    predict_tide,
    seasonal_naive,
)

SITE_IDS = list(SITES)


def _arch_model(n_cov: int, arch: dict | None = None) -> TideQuantile:
    arch = arch or {}
    return TideQuantile(
        lookback=LOOKBACK,
        horizon=HORIZON,
        n_covariates=n_cov,
        n_sites=1,
        n_quantiles=len(QUANTILES),
        hidden=arch.get("hidden", 64),
        decoder_dim=arch.get("decoder_dim", 16),
        temporal_width=arch.get("temporal_width", 4),
        encoder_layers=arch.get("encoder_layers", 2),
        decoder_layers=arch.get("decoder_layers", 2),
        dropout=arch.get("dropout", 0.2),
    )


def _load_models(ckpt_path: Path, n_cov: int) -> tuple[dict[str, TideQuantile], dict]:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    n_cov = blob.get("n_cov", n_cov)
    arch = blob.get("arch") or {}
    models = {}
    if blob.get("per_site"):
        for sid, state in blob["states"].items():
            m = _arch_model(n_cov, arch)
            m.load_state_dict(state)
            m.eval()
            models[sid] = m
        return models, blob
    m = _arch_model(n_cov, arch)
    m.load_state_dict(blob["state"])
    m.eval()
    for sid in SITES:
        models[sid] = m
    return models, blob


def _hour_of_day(origins: np.ndarray, lead: int) -> np.ndarray:
    # origin is first forecast hour.
    hours = []
    for o in origins:
        ts = np.datetime64(o)
        # pandas-style: origin string like 2026-09-28 15:00:00+05:30
        hours.append((int(str(o)[11:13]) + lead) % 24)
    return np.asarray(hours)


def report() -> dict:
    panel = load_panel(DATA_DIR / "hourly_panel.csv")
    test, norms = make_windows(panel, "test")
    train, _ = make_windows(panel, "train")
    n_cov = test["x_hist"].shape[-1]
    models, blob = _load_models(OUTPUT_DIR / "tide.pt", n_cov)
    pred = np.zeros((len(test["y_hist"]), HORIZON, 3), np.float32)
    for i, sid in enumerate(SITES):
        mask = test["site"] == i
        if not mask.any() or sid not in models:
            continue
        pred[mask] = predict_tide(models[sid], _filter_site(test, i))
    y_true = denorm(test["y_fut"], test["site"], norms)
    y_p10 = denorm(pred[:, :, 0], test["site"], norms)
    y_p50 = denorm(pred[:, :, 1], test["site"], norms)
    y_p90 = denorm(pred[:, :, 2], test["site"], norms)
    for i, sid in enumerate(SITES):
        cfg = (blob.get("conformal") or {}).get(sid)
        if not cfg:
            continue
        mask = test["site"] == i
        y_p10[mask], y_p90[mask] = apply_conformal(
            y_p50[mask], np.asarray(cfg["lo"]), np.asarray(cfg["hi"])
        )
    y_naive = seasonal_naive(test, norms)
    y_cal = calibrated_naive(test, norms)
    last_obs = denorm(test["y_hist"][:, -1:], test["site"], norms)[:, 0]

    out: dict = {"n_windows": int(len(y_true)), "horizon": HORIZON}
    out["overall"] = {
        "tide": _metrics(y_true, y_p50),
        "naive": _metrics(y_true, y_naive),
        "calibrated_naive": _metrics(y_true, y_cal),
        "coverage_p10_p90": float(np.mean((y_true >= y_p10) & (y_true <= y_p90))),
        "quantile_crossings": float(np.mean((y_p10 > y_p50) | (y_p50 > y_p90))),
    }
    skill = 1.0 - out["overall"]["tide"]["wmape"] / max(out["overall"]["naive"]["wmape"], 1e-6)
    out["overall"]["skill_vs_naive"] = float(skill)

    by_site = {}
    idx = site_index_map()
    for sid, i in idx.items():
        m = test["site"] == i
        if not m.any():
            continue
        yt, yp, yn = y_true[m], y_p50[m], y_naive[m]
        p10, p90 = y_p10[m], y_p90[m]
        last = last_obs[m]
        jump = yp[:, 0] - last
        rel_jump = jump / np.clip(last, 0.05, None)
        hod_rows = []
        origins = test["origin"][m]
        for hour in range(24):
            # lead 0 hour-of-day
            hod = np.array([int(str(o)[11:13]) for o in origins])
            sel = hod == hour
            if sel.sum() < 5:
                continue
            hod_rows.append(
                {
                    "hour": hour,
                    "tide_wmape": _metrics(yt[sel], yp[sel])["wmape"],
                    "naive_wmape": _metrics(yt[sel], yn[sel])["wmape"],
                    "n": int(sel.sum() * HORIZON),
                }
            )
        lead_skill = []
        for h in range(HORIZON):
            tm = _metrics(yt[:, h], yp[:, h])
            nm = _metrics(yt[:, h], yn[:, h])
            lead_skill.append(
                {
                    "lead_h": h + 1,
                    "tide_wmape": tm["wmape"],
                    "naive_wmape": nm["wmape"],
                    "tide_mae": tm["mae"],
                    "skill": 1.0 - tm["wmape"] / max(nm["wmape"], 1e-6),
                }
            )
        by_site[sid] = {
            "n_windows": int(m.sum()),
            "tide": _metrics(yt, yp),
            "naive": _metrics(yt, yn),
            "calibrated_naive": _metrics(yt, y_cal[m]),
            "skill_vs_naive": 1.0 - _metrics(yt, yp)["wmape"] / max(_metrics(yt, yn)["wmape"], 1e-6),
            "skill_vs_calibrated": 1.0 - _metrics(yt, yp)["wmape"] / max(_metrics(yt, y_cal[m])["wmape"], 1e-6),
            "coverage_p10_p90": float(np.mean((yt >= p10) & (yt <= p90))),
            "origin_jump_kw_mean": float(np.mean(jump)),
            "origin_jump_kw_mae": float(np.mean(np.abs(jump))),
            "origin_rel_jump_median": float(np.median(rel_jump)),
            "origin_rel_jump_p90": float(np.quantile(np.abs(rel_jump), 0.9)),
            "by_hour": hod_rows,
            "by_lead": lead_skill,
            "peak_mae": float(np.mean(np.abs(yt.max(axis=1) - yp.max(axis=1)))),
            "naive_peak_mae": float(np.mean(np.abs(yt.max(axis=1) - yn.max(axis=1)))),
        }
    out["by_site"] = by_site

    # Gates
    gates = []
    def gate(name, ok, detail):
        gates.append({"name": name, "pass": bool(ok), "detail": detail})

    gate(
        "beats_naive_overall",
        skill > 0.05,
        f"skill={skill:.3f} (need >0.05 vs weekly naive)",
    )
    for sid, s in by_site.items():
        gate(
            f"{sid}_wmape_vs_naive",
            s["skill_vs_naive"] > 0.0,
            f"skill={s['skill_vs_naive']:.3f} tide_wmape={s['tide']['wmape']:.1f} naive={s['naive']['wmape']:.1f}",
        )
        gate(
            f"{sid}_origin_jump",
            s["origin_rel_jump_p90"] < 0.50,
            f"p90 |Δ|/last={s['origin_rel_jump_p90']:.2f} (need <0.50)",
        )
        gate(
            f"{sid}_coverage",
            0.70 <= s["coverage_p10_p90"] <= 0.92,
            f"coverage={s['coverage_p10_p90']:.2f} (80% interval target 0.70–0.92)",
        )
        gate(
            f"{sid}_peak_vs_naive",
            s["peak_mae"] <= s["naive_peak_mae"] * 1.05,
            f"peak MAE {s['peak_mae']:.3f} vs naive {s['naive_peak_mae']:.3f}",
        )
    # Residential production bar: KA should be <20% wMAPE, TL <40% as advisory.
    if "RSSEKA1608" in by_site:
        gate(
            "KA_advisory_wmape",
            by_site["RSSEKA1608"]["tide"]["wmape"] < 20.0,
            f"KA wMAPE={by_site['RSSEKA1608']['tide']['wmape']:.1f}% (advisory bar <20%)",
        )
    if "RSSETL2362" in by_site:
        gate(
            "TL_advisory_wmape",
            by_site["RSSETL2362"]["tide"]["wmape"] < 40.0,
            f"TL wMAPE={by_site['RSSETL2362']['tide']['wmape']:.1f}% (advisory bar <40%)",
        )
    gate(
        "no_quantile_crossings",
        out["overall"]["quantile_crossings"] < 0.01,
        f"crossing rate={out['overall']['quantile_crossings']:.3f}",
    )
    out["gates"] = gates
    out["verdict"] = "PASS" if all(g["pass"] for g in gates) else "FAIL"
    failed = [g["name"] for g in gates if not g["pass"]]
    out["failed_gates"] = failed
    return out


def main() -> None:
    out = report()
    path = OUTPUT_DIR / "production_eval.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"verdict {out['verdict']}  windows={out['n_windows']}")
    print(
        f"overall tide wMAPE={out['overall']['tide']['wmape']:.1f}%  "
        f"naive={out['overall']['naive']['wmape']:.1f}%  "
        f"skill={out['overall']['skill_vs_naive']:.3f}  "
        f"coverage={out['overall']['coverage_p10_p90']:.2f}  "
        f"crossings={out['overall']['quantile_crossings']:.3f}"
    )
    for sid, s in out["by_site"].items():
        print(
            f"{sid}  tide={s['tide']['wmape']:.1f}%  naive={s['naive']['wmape']:.1f}%  "
            f"skill={s['skill_vs_naive']:.3f}  cov={s['coverage_p10_p90']:.2f}  "
            f"origin |Δ| p90={s['origin_rel_jump_p90']:.2f}  peakMAE={s['peak_mae']:.3f}"
        )
        # worst hours
        worst = sorted(s["by_hour"], key=lambda r: -r["tide_wmape"])[:3]
        print("  worst hours", [(w["hour"], round(w["tide_wmape"], 1), round(w["naive_wmape"], 1)) for w in worst])
    print("gates:")
    for g in out["gates"]:
        print(f"  {'PASS' if g['pass'] else 'FAIL'}  {g['name']}: {g['detail']}")
    print("wrote", path)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
