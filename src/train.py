"""Train TiDE + a quantile GBM baseline. Write checkpoints and Verify metrics."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from torch.utils.data import DataLoader, TensorDataset

from config import HORIZON, LOOKBACK, OUTPUT_DIR, QUANTILES, SITES
from dataset import from_model, gbm_matrix, last_origin_window, load_panel, make_windows
from model import TideQuantile, pinball_loss

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cpu")


def _tensors(arr: dict) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(arr["y_hist"]),
        torch.from_numpy(arr["x_hist"]),
        torch.from_numpy(arr["x_fut"]),
        torch.from_numpy(arr["y_fut"]),
        torch.from_numpy(arr["site"]),
    )


def _metrics(y_true: np.ndarray, y_hat: np.ndarray) -> dict:
    err = y_hat - y_true
    denom = np.clip(np.abs(y_true), 0.05, None)
    wmape = float(np.sum(np.abs(err)) / np.clip(np.sum(np.abs(y_true)), 1e-6, None) * 100)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(np.abs(err) / denom) * 100),
        "wmape": wmape,
        "n": int(y_true.size),
    }


def _lead_table(y_true: np.ndarray, y_hat: np.ndarray) -> list[dict]:
    rows = []
    for h in range(y_true.shape[1]):
        m = _metrics(y_true[:, h], y_hat[:, h])
        m["lead_h"] = h + 1
        rows.append(m)
    return rows


def _filter_site(arr: dict, site_i: int) -> dict:
    mask = arr["site"] == site_i
    out = {}
    n = len(arr["site"])
    for k, v in arr.items():
        if isinstance(v, np.ndarray) and len(v) == n:
            sl = v[mask]
            if k == "site":
                sl = np.zeros(len(sl), dtype=np.int64)
            out[k] = sl
        else:
            out[k] = v
    return out


def train_tide(train, val, n_cov: int, epochs: int = 100) -> TideQuantile:
    model = TideQuantile(
        lookback=LOOKBACK,
        horizon=HORIZON,
        n_covariates=n_cov,
        n_sites=1,
        n_quantiles=len(QUANTILES),
        hidden=64,
        decoder_dim=16,
        temporal_width=4,
        encoder_layers=2,
        decoder_layers=2,
        dropout=0.2,
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    loader = DataLoader(_tensors(train), batch_size=64, shuffle=True)
    val_set = _tensors(val)
    best_state, best_val, patience = None, math.inf, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for y_h, x_h, x_f, y_f, site in loader:
            pred = model(y_h, x_h, x_f, site)
            loss = pinball_loss(pred, y_f, QUANTILES)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            y_h, x_h, x_f, y_f, site = val_set.tensors
            vloss = float(pinball_loss(model(y_h, x_h, x_f, site), y_f, QUANTILES).item())
        if vloss < best_val - 1e-4:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if epoch == 1 or epoch % 10 == 0 or patience == 0:
            print(
                f"  epoch {epoch:3d}  train {np.mean(losses):.4f}  val {vloss:.4f}  "
                f"best {best_val:.4f}  {time.time() - t0:.0f}s",
                flush=True,
            )
        if patience >= 16:
            print("  early stop", flush=True)
            break
    if best_state:
        model.load_state_dict(best_state)
    return model


def predict_tide(model: TideQuantile, arr: dict) -> np.ndarray:
    model.eval()
    ds = _tensors(arr)
    with torch.no_grad():
        y_h, x_h, x_f, _, site = ds.tensors
        pred = model(y_h, x_h, x_f, site).cpu().numpy()
    return pred


def train_gbm(train, val) -> list[HistGradientBoostingRegressor]:
    X_tr, y_tr = gbm_matrix(train)
    X_va, y_va = gbm_matrix(val)
    models = []
    for q in QUANTILES:
        print(f"  GBM q={q} …", flush=True)
        est = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=q,
            max_depth=6,
            max_iter=200,
            learning_rate=0.06,
            min_samples_leaf=20,
            l2_regularization=0.1,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=7,
        )
        # Fit on flattened horizons as extra samples (share one model across leads).
        est.fit(X_tr, y_tr)
        models.append(est)
        if len(X_va):
            n_val = len(val["y_hist"])
            hat = est.predict(X_va).reshape(n_val, HORIZON)
            y_va_h = y_va.reshape(n_val, HORIZON)
            print(f"    val MAE {np.mean(np.abs(hat - y_va_h)):.4f}", flush=True)
    return models


def predict_gbm(models, arr) -> np.ndarray:
    X, _ = gbm_matrix(arr)
    n = len(arr["y_hist"])
    if n == 0:
        return np.zeros((0, HORIZON, len(models)), np.float32)
    stacked = [est.predict(X).reshape(n, HORIZON) for est in models]
    return np.stack(stacked, axis=-1).astype(np.float32)


def denorm(y: np.ndarray, sites: np.ndarray, norms) -> np.ndarray:
    keys = list(SITES)
    out = np.empty_like(y, dtype=np.float64)
    for i, sid in enumerate(keys):
        mask = sites == i
        if np.any(mask):
            out[mask] = from_model(y[mask], norms[sid])
    return out.astype(np.float32)


def seasonal_naive(arr: dict, norms) -> np.ndarray:
    y = arr["y_hist"][:, :HORIZON]
    return denorm(y, arr["site"], norms)


def fit_conformal(y_true: np.ndarray, p50: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-lead additive offsets so val residuals hit p10/p90."""
    err = y_true - p50
    lo = np.quantile(err, 0.10, axis=0)
    hi = np.quantile(err, 0.90, axis=0)
    return lo.astype(np.float32), hi.astype(np.float32)


def apply_conformal(p50: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p10 = np.clip(p50 + lo, 0, None)
    p90 = np.maximum(p50 + hi, p10)
    return p10, p90


def calibrated_naive(arr: dict, norms, tau: float = 6.0) -> np.ndarray:
    weekly = arr["y_hist"][:, :HORIZON]
    last = arr["y_hist"][:, -1:]
    gap = last - weekly[:, :1]
    decay = np.exp(-np.arange(HORIZON)[None, :] / tau)
    y = weekly + gap * decay
    return denorm(y, arr["site"], norms)


def run(panel_path: Path) -> dict:
    panel = load_panel(panel_path)
    print("building windows …", flush=True)
    train, norms = make_windows(panel, "train")
    val, _ = make_windows(panel, "val")
    test, _ = make_windows(panel, "test")
    n_cov = train["x_hist"].shape[-1]
    if len(val["y_hist"]) < 32 and len(train["y_hist"]) > 64:
        n_val = min(96, len(train["y_hist"]) // 6)
        val = {k: (v[-n_val:] if isinstance(v, np.ndarray) and len(v) == len(train["y_hist"]) else v) for k, v in train.items()}
        train = {k: (v[:-n_val] if isinstance(v, np.ndarray) and len(v) == len(train["y_hist"]) else v) for k, v in train.items()}
        print(f"val set thin; held out last {n_val} train windows", flush=True)
    print(
        f"windows train={len(train['y_hist'])} val={len(val['y_hist'])} "
        f"test={len(test['y_hist'])} cov={n_cov}",
        flush=True,
    )
    for sid, nrm in norms.items():
        print(f"  {sid} mean={nrm.mean:.3f} kW  std={nrm.std:.3f}", flush=True)

    print("TiDE per site …", flush=True)
    models: dict[str, TideQuantile] = {}
    states = {}
    for i, sid in enumerate(SITES):
        tr = _filter_site(train, i)
        va = _filter_site(val, i)
        if len(tr["y_hist"]) < 64:
            print(f"  skip {sid}: only {len(tr['y_hist'])} train windows", flush=True)
            continue
        print(f"  {sid} train={len(tr['y_hist'])} val={len(va['y_hist'])}", flush=True)
        models[sid] = train_tide(tr, va if len(va["y_hist"]) >= 16 else tr, n_cov=n_cov)
        states[sid] = {k: v.detach().cpu() for k, v in models[sid].state_dict().items()}
    ckpt = OUTPUT_DIR / "tide.pt"
    torch.save(
        {
            "per_site": True,
            "states": states,
            "n_cov": n_cov,
            "arch": {
                "hidden": 64,
                "decoder_dim": 16,
                "temporal_width": 4,
                "encoder_layers": 2,
                "decoder_layers": 2,
                "dropout": 0.2,
            },
            "norms": {k: {"mean": v.mean, "std": v.std} for k, v in norms.items()},
            "conformal": {},
        },
        ckpt,
    )

    print("GBM baseline …", flush=True)
    gbm = train_gbm(train, val)

    def pack(name, pred_norm, arr):
        y_true = denorm(arr["y_fut"], arr["site"], norms)
        y_hat = denorm(pred_norm[:, :, 1], arr["site"], norms)  # median
        p10 = denorm(pred_norm[:, :, 0], arr["site"], norms)
        p90 = denorm(pred_norm[:, :, 2], arr["site"], norms)
        m = _metrics(y_true, y_hat)
        m["lead"] = _lead_table(y_true, y_hat)
        m["coverage_p10_p90"] = float(np.mean((y_true >= p10) & (y_true <= p90)))
        print(
            f"  {name} wMAPE={m['wmape']:.1f}% MAPE={m['mape']:.1f}% "
            f"MAE={m['mae']:.3f} kW cov={m['coverage_p10_p90']:.2f}",
            flush=True,
        )
        return m, y_true, y_hat, p10, p90

    tide_test = np.zeros((len(test["y_hist"]), HORIZON, 3), np.float32)
    conformal: dict[str, dict] = {}
    for i, sid in enumerate(SITES):
        if sid not in models:
            continue
        va = _filter_site(val, i)
        te = _filter_site(test, i)
        mask = test["site"] == i
        if not mask.any():
            continue
        tide_test[mask] = predict_tide(models[sid], te)
        nrm = norms[sid]
        if len(va["y_hist"]) >= 16:
            vp = predict_tide(models[sid], va)
            yv = from_model(va["y_fut"], nrm)
            pv = from_model(vp[:, :, 1], nrm)
            lo, hi = fit_conformal(yv, pv)
        else:
            lo = np.zeros(HORIZON, np.float32)
            hi = np.zeros(HORIZON, np.float32)
        conformal[sid] = {"lo": lo.tolist(), "hi": hi.tolist()}
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    blob["conformal"] = conformal
    torch.save(blob, ckpt)
    gbm_test = predict_gbm(gbm, test)
    naive = seasonal_naive(test, norms)
    cal_naive = calibrated_naive(test, norms)

    report = {"quantiles": list(QUANTILES), "lookback": LOOKBACK, "horizon": HORIZON}
    report["tide"], y_true, y_tide, p10, p90 = pack("tide", tide_test, test)
    for i, sid in enumerate(SITES):
        if sid not in conformal:
            continue
        mask = test["site"] == i
        lo = np.asarray(conformal[sid]["lo"])
        hi = np.asarray(conformal[sid]["hi"])
        p10[mask], p90[mask] = apply_conformal(y_tide[mask], lo, hi)
    report["tide"]["coverage_p10_p90"] = float(np.mean((y_true >= p10) & (y_true <= p90)))
    print(f"  tide conformal coverage={report['tide']['coverage_p10_p90']:.2f}", flush=True)
    report["gbm"], *_ = pack("gbm", gbm_test, test)
    report["seasonal_naive"] = _metrics(y_true, naive)
    report["calibrated_naive"] = _metrics(y_true, cal_naive)
    print(
        f"  seasonal_naive wMAPE={report['seasonal_naive']['wmape']:.1f}% "
        f"MAE={report['seasonal_naive']['mae']:.3f} kW",
        flush=True,
    )
    print(
        f"  calibrated_naive wMAPE={report['calibrated_naive']['wmape']:.1f}% "
        f"MAE={report['calibrated_naive']['mae']:.3f} kW",
        flush=True,
    )

    # Per-site test MAPE for the dashboard.
    report["by_site"] = {}
    for i, sid in enumerate(SITES):
        mask = test["site"] == i
        if not mask.any():
            continue
        report["by_site"][sid] = _metrics(y_true[mask], y_tide[mask])

    live = last_origin_window(panel, norms)
    live_out = {}
    for sid, w in live.items():
        model = models.get(sid)
        if model is None:
            continue
        model.eval()
        with torch.no_grad():
            pred = model(
                torch.from_numpy(w["y_hist"][None]),
                torch.from_numpy(w["x_hist"][None]),
                torch.from_numpy(w["x_fut"][None]),
                torch.tensor([0]),
            )[0].numpy()
        nrm = norms[sid]
        q = from_model(pred, nrm)
        if sid in conformal:
            lo = np.asarray(conformal[sid]["lo"])
            hi = np.asarray(conformal[sid]["hi"])
            q[:, 0], q[:, 2] = apply_conformal(q[:, 1], lo, hi)
        live_out[sid] = {
            "origin": str(w["origin"]),
            "future_ts": [str(t) for t in w["future_ts"]],
            "hist_ts": [str(t) for t in w["hist_ts"][-48:]],
            "hist_load_kw": w["hist_load"][-48:].tolist(),
            "hist_temp_c": w["hist_temp"][-48:].tolist(),
            "future_temp_c": w["future_temp"].tolist(),
            "p10": q[:, 0].tolist(),
            "p50": q[:, 1].tolist(),
            "p90": q[:, 2].tolist(),
        }
    report["live"] = live_out
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {OUTPUT_DIR / 'metrics.json'}", flush=True)
    return report


if __name__ == "__main__":
    from config import DATA_DIR

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR / "hourly_panel.csv"
    run(path)
