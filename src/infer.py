"""Issue a 48 h forecast from the production checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from config import DATA_DIR, HORIZON, OUTPUT_DIR, SITES
from dataset import from_model, last_origin_window, load_panel, make_windows
from evaluate import _load_models
from train import predict_tide


def issue(site_id: str, panel_path: Path | None = None) -> dict:
    if site_id not in SITES:
        raise KeyError(site_id)
    panel = load_panel(panel_path or DATA_DIR / "hourly_panel.csv")
    _, norms = make_windows(panel, "train")
    models, blob = _load_models(OUTPUT_DIR / "tide.pt", n_cov=15)
    live = last_origin_window(panel, norms)
    w = live[site_id]
    pred = predict_tide(
        models[site_id],
        {
            "y_hist": w["y_hist"][None],
            "x_hist": w["x_hist"][None],
            "x_fut": w["x_fut"][None],
            "site": np.array([0]),
            "y_fut": np.zeros((1, HORIZON), np.float32),
        },
    )[0]
    q = from_model(pred, norms[site_id])
    return {
        "site_id": site_id,
        "origin": str(w["origin"]),
        "unit": "kW",
        "p10": q[:, 0].tolist(),
        "p50": q[:, 1].tolist(),
        "p90": q[:, 2].tolist(),
        "timestamps": [str(t) for t in w["future_ts"]],
        "go_live": False,
        "verdict": "FAIL",
    }


if __name__ == "__main__":
    import sys

    sid = sys.argv[1] if len(sys.argv) > 1 else next(iter(SITES))
    print(json.dumps(issue(sid), indent=2)[:1500])
