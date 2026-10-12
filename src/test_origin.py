"""Guards that must hold before a forecast is issued."""

from __future__ import annotations

import numpy as np
import torch

from config import HORIZON, LOOKBACK
from model import TideQuantile


def test_similar_day_matches_last_actual_at_h0():
    m = TideQuantile(lookback=LOOKBACK, horizon=HORIZON, n_covariates=4, n_sites=1)
    y = torch.randn(8, LOOKBACK)
    naive = m.similar_day(y)
    assert naive.shape == (8, HORIZON)
    err = (naive[:, 0] - y[:, -1]).abs().max().item()
    assert err < 1e-5, err


def test_quantiles_monotone():
    m = TideQuantile(lookback=LOOKBACK, horizon=HORIZON, n_covariates=4, n_sites=1)
    m.eval()
    y = torch.randn(4, LOOKBACK)
    xh = torch.randn(4, LOOKBACK, 4)
    xf = torch.randn(4, HORIZON, 4)
    site = torch.zeros(4, dtype=torch.long)
    with torch.no_grad():
        pred = m(y, xh, xf, site)
    assert torch.all(pred[..., 0] <= pred[..., 1] + 1e-5)
    assert torch.all(pred[..., 1] <= pred[..., 2] + 1e-5)


def test_forward_locks_hour0_to_last_actual():
    m = TideQuantile(lookback=LOOKBACK, horizon=HORIZON, n_covariates=4, n_sites=1)
    m.eval()
    y = torch.randn(4, LOOKBACK)
    xh = torch.randn(4, LOOKBACK, 4)
    xf = torch.randn(4, HORIZON, 4)
    site = torch.zeros(4, dtype=torch.long)
    with torch.no_grad():
        pred = m(y, xh, xf, site)
    err = (pred[:, 0, 1] - y[:, -1]).abs().max().item()
    assert err < 1e-4, err


if __name__ == "__main__":
    test_similar_day_matches_last_actual_at_h0()
    test_quantiles_monotone()
    test_forward_locks_hour0_to_last_actual()
    print("ok")
