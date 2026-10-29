"""Guards that must hold before a forecast is issued."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

from config import HORIZON, LOOKBACK, SITES, TZ
from consumption import fit_profiles
from model import TideQuantile


def _batch(n: int = 4, n_cov: int = 4):
    y = torch.randn(n, LOOKBACK)
    xh = torch.randn(n, LOOKBACK, n_cov)
    xf = torch.randn(n, HORIZON, n_cov)
    site = torch.zeros(n, dtype=torch.long)
    profile_now = y[:, -1]
    profile_fut = profile_now.unsqueeze(1).expand(n, HORIZON).contiguous()
    return y, xh, xf, site, profile_fut, profile_now


def test_similar_day_matches_last_actual_at_h0():
    m = TideQuantile(lookback=LOOKBACK, horizon=HORIZON, n_covariates=4, n_sites=1)
    y = torch.randn(8, LOOKBACK)
    naive = m.similar_day(y)
    assert naive.shape == (8, HORIZON)
    err = (naive[:, 0] - y[:, -1]).abs().max().item()
    assert err < 1e-5, err


def test_indian_level_locks_hour0_and_walks_to_profile():
    """KA-like quiet night and a TL-like noon spike both return to their curve."""
    m = TideQuantile(lookback=LOOKBACK, horizon=HORIZON, n_covariates=4, n_sites=1)
    y = torch.zeros(2, LOOKBACK)
    y[:, -1] = torch.tensor([0.20, 1.50])
    profile_now = torch.tensor([0.20, 0.40])
    profile_fut = profile_now.unsqueeze(1).expand(2, HORIZON).contiguous().clone()
    profile_fut[:, 19] = torch.tensor([0.90, 4.00])
    base = m.indian_level(y, profile_fut, profile_now)
    err = (base[:, 0] - y[:, -1]).abs().max().item()
    assert err < 1e-5, err
    decay = math.exp(-19 / m.origin_tau)
    assert abs(base[0, 19].item() - 0.90) < 1e-4
    expect_tl = 4.00 + (1.50 - 0.40) * decay
    assert abs(base[1, 19].item() - expect_tl) < 1e-4
    # A spike that is not on the clock fades; it is not copied out to 48 h.
    assert abs(base[1, 47].item() - profile_fut[1, 47].item()) < 0.02


def test_quantiles_monotone():
    m = TideQuantile(lookback=LOOKBACK, horizon=HORIZON, n_covariates=4, n_sites=1)
    m.eval()
    y, xh, xf, site, profile_fut, profile_now = _batch()
    with torch.no_grad():
        pred = m(y, xh, xf, site, profile_fut, profile_now)
    assert pred.shape[-1] == 7
    for i in range(pred.shape[-1] - 1):
        assert torch.all(pred[..., i] <= pred[..., i + 1] + 1e-5), i


def test_forward_locks_hour0_to_last_actual():
    m = TideQuantile(lookback=LOOKBACK, horizon=HORIZON, n_covariates=4, n_sites=1)
    m.eval()
    y, xh, xf, site, profile_fut, profile_now = _batch()
    with torch.no_grad():
        pred = m(y, xh, xf, site, profile_fut, profile_now)
    err = (pred[:, 0, 3] - y[:, -1]).abs().max().item()
    assert err < 1e-4, err


def test_battery_profiles_match_indian_houses():
    """Peaks must be the house-load shape, not battery_p_batt."""
    cut = pd.Timestamp("2027-01-01", tz=TZ)
    profiles = fit_profiles({sid: cut for sid in SITES})
    ka = profiles["RSSEKA1608"]["hour"]
    tl = profiles["RSSETL2362"]["hour"]
    assert int(np.nanargmax(ka)) == 19, ka
    assert int(np.nanargmax(tl)) == 12, tl
    assert ka[19] > ka[12]
    assert tl[12] > tl[19] * 2
    assert 0.4 < float(ka[19]) < 2.0
    assert 3.0 < float(tl[12]) < 8.0


if __name__ == "__main__":
    test_similar_day_matches_last_actual_at_h0()
    test_indian_level_locks_hour0_and_walks_to_profile()
    test_quantiles_monotone()
    test_forward_locks_hour0_to_last_actual()
    test_battery_profiles_match_indian_houses()
    print("ok")
