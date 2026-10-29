"""TiDE with an Indian house-load skip and the Operate percentile head.

The residual network predicts a correction on top of each meter's own
hour-of-day curve from Battery_Data.xlsx load_p_load_total. A level error
(last actual minus that hour's climatology) fades with lead, so a noon spike
on TL-2362 decays back onto the Hyderabad shape instead of being copied for
48 h. KA-1608's evening peak and TL-2362's midday peak stay separate curves.

Operate serves a point `value` plus p01, p05, p25, p75, p95, p99.
The head builds those as nested softplus spreads around value, so
p01 ≤ p05 ≤ p25 ≤ value ≤ p75 ≤ p95 ≤ p99.
Lead 0 of value is the last actual: the house issue hour is that hour.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from config import QUANTILES, VALUE_INDEX


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop1(torch.relu(self.fc1(x)))
        h = self.drop2(self.fc2(h))
        return self.norm(h + self.skip(x))


class TideQuantile(nn.Module):
    def __init__(
        self,
        lookback: int,
        horizon: int,
        n_covariates: int,
        n_sites: int = 1,
        n_quantiles: int = 6,
        hidden: int = 100,
        decoder_dim: int = 24,
        temporal_width: int = 8,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        dropout: float = 0.1,
        site_emb: int = 4,
        origin_tau: float = 6.0,
    ) -> None:
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.n_covariates = n_covariates
        if n_quantiles != len(QUANTILES):
            raise ValueError(f"n_quantiles must be {len(QUANTILES)} to match the Operate head")
        self.n_quantiles = n_quantiles
        self.temporal_width = temporal_width
        self.origin_tau = origin_tau

        self.site_embedding = nn.Embedding(max(n_sites, 1), site_emb)
        self.feature_proj = ResidualBlock(n_covariates, hidden, temporal_width, dropout)

        enc_in = lookback + (lookback + horizon) * temporal_width + site_emb
        layers: list[nn.Module] = [ResidualBlock(enc_in, hidden, hidden, dropout)]
        for _ in range(encoder_layers - 1):
            layers.append(ResidualBlock(hidden, hidden, hidden, dropout))
        self.encoder = nn.Sequential(*layers)

        dec_layers: list[nn.Module] = []
        in_dim = hidden
        for i in range(decoder_layers):
            out_dim = hidden if i < decoder_layers - 1 else horizon * decoder_dim
            dec_layers.append(ResidualBlock(in_dim, hidden, out_dim, dropout))
            in_dim = out_dim
        self.decoder = nn.Sequential(*dec_layers)
        self.decoder_dim = decoder_dim

        # p01, p05, p25 spreads, value residual, p75, p95, p99 spreads.
        self.n_outputs = len(QUANTILES) + 1
        self.temporal = ResidualBlock(
            decoder_dim + temporal_width, hidden, self.n_outputs, dropout
        )
        self.spread_floor = nn.Parameter(torch.tensor(0.05))

    def similar_day(self, y_hist: torch.Tensor) -> torch.Tensor:
        """Weekly profile, recentered so hour 0 matches last actual, fading with lead."""
        weekly = y_hist[:, : self.horizon]
        last = y_hist[:, -1]
        gap = last - weekly[:, 0]
        h = torch.arange(self.horizon, device=y_hist.device, dtype=y_hist.dtype)
        decay = torch.exp(-h / self.origin_tau).unsqueeze(0)
        return weekly + gap.unsqueeze(1) * decay

    def indian_level(
        self,
        y_hist: torch.Tensor,
        profile_fut: torch.Tensor,
        profile_now: torch.Tensor,
    ) -> torch.Tensor:
        """Site hour-of-day curve in model space, pinned to the last actual.

        At long lead the path sits on the Indian climatology. The gap between
        the last actual and the climatology at that same hour fades with
        origin_tau. Index 0 is the last actual, not the next hour's curve.
        """
        last = y_hist[:, -1]
        h = torch.arange(self.horizon, device=y_hist.device, dtype=y_hist.dtype)
        decay = torch.exp(-h / self.origin_tau).unsqueeze(0)
        base = profile_fut + (last - profile_now).unsqueeze(1) * decay
        return torch.cat([last.unsqueeze(1), base[:, 1:]], dim=1)

    def forward(
        self,
        y_hist: torch.Tensor,
        x_hist: torch.Tensor,
        x_fut: torch.Tensor,
        site: torch.Tensor,
        profile_fut: torch.Tensor,
        profile_now: torch.Tensor,
    ) -> torch.Tensor:
        bsz = y_hist.size(0)
        x_all = torch.cat([x_hist, x_fut], dim=1)
        x_proj = self.feature_proj(x_all)
        site_idx = site.clamp(min=0, max=self.site_embedding.num_embeddings - 1)
        site_vec = self.site_embedding(site_idx)
        enc_in = torch.cat([y_hist, x_proj.reshape(bsz, -1), site_vec], dim=-1)
        encoded = self.encoder(enc_in)
        decoded = self.decoder(encoded).reshape(bsz, self.horizon, self.decoder_dim)
        x_fut_proj = x_proj[:, self.lookback :, :]
        raw = self.temporal(torch.cat([decoded, x_fut_proj], dim=-1))
        naive = self.indian_level(y_hist, profile_fut, profile_now)
        h = torch.arange(self.horizon, device=y_hist.device, dtype=y_hist.dtype)
        # Residual is zero at the issue hour so value[0] == last actual.
        gate = (1.0 - torch.exp(-h / 4.0)).view(1, -1)
        value = naive + raw[..., VALUE_INDEX] * gate
        floor = F.softplus(self.spread_floor) + 0.02
        s25 = F.softplus(raw[..., 2]) + floor
        s05 = F.softplus(raw[..., 1]) + floor
        s01 = F.softplus(raw[..., 0]) + floor
        s75 = F.softplus(raw[..., 4]) + floor
        s95 = F.softplus(raw[..., 5]) + floor
        s99 = F.softplus(raw[..., 6]) + floor
        p25 = value - s25
        p05 = p25 - s05
        p01 = p05 - s01
        p75 = value + s75
        p95 = p75 + s95
        p99 = p95 + s99
        return torch.stack([p01, p05, p25, value, p75, p95, p99], dim=-1)


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: tuple[float, ...]) -> torch.Tensor:
    q = pred.new_tensor(quantiles).view(1, 1, -1)
    err = target.unsqueeze(-1) - pred
    return torch.mean(torch.maximum(q * err, (q - 1) * err))


def operate_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pinball on Operate's six percentiles, plus a median pinball on value.

    The captured desk never labeled value as p50. It did keep value inside
    p25–p75, which the head enforces. The median pinball is the training
    signal for that point.
    """
    q_idx = [i for i in range(pred.size(-1)) if i != VALUE_INDEX]
    q_loss = pinball_loss(pred[..., q_idx], target, QUANTILES)
    err = target - pred[..., VALUE_INDEX]
    value_loss = torch.mean(torch.maximum(0.5 * err, -0.5 * err))
    return q_loss + value_loss
