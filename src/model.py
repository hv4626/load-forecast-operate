"""TiDE with calibrated similar-day skip and monotone quantile spreads.

The residual network predicts a correction on top of:
  same clock-hour last week + origin-level decay toward the last actual.

Quantile heads are median ± softplus spreads so p10 ≤ p50 ≤ p90 always.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


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
        n_quantiles: int = 3,
        hidden: int = 96,
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

        # 3 channels: lo_spread, median residual, hi_spread
        self.temporal = ResidualBlock(decoder_dim + temporal_width, hidden, 3, dropout)
        self.spread_floor = nn.Parameter(torch.tensor(0.12))

    def similar_day(self, y_hist: torch.Tensor) -> torch.Tensor:
        """Weekly profile, recentered so hour 0 matches last actual, fading with lead."""
        weekly = y_hist[:, : self.horizon]
        last = y_hist[:, -1]
        gap = last - weekly[:, 0]
        h = torch.arange(self.horizon, device=y_hist.device, dtype=y_hist.dtype)
        decay = torch.exp(-h / self.origin_tau).unsqueeze(0)
        return weekly + gap.unsqueeze(1) * decay

    def forward(
        self,
        y_hist: torch.Tensor,
        x_hist: torch.Tensor,
        x_fut: torch.Tensor,
        site: torch.Tensor,
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
        naive = self.similar_day(y_hist)
        h = torch.arange(self.horizon, device=y_hist.device, dtype=y_hist.dtype)
        # Residual is zero at the issue hour so p50[0] == last actual.
        gate = (1.0 - torch.exp(-h / 4.0)).view(1, -1)
        p50 = naive + raw[..., 1] * gate
        floor = F.softplus(self.spread_floor) + 0.05
        lo = F.softplus(raw[..., 0]) + floor
        hi = F.softplus(raw[..., 2]) + floor
        p10 = p50 - lo
        p90 = p50 + hi
        return torch.stack([p10, p50, p90], dim=-1)


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: tuple[float, ...]) -> torch.Tensor:
    q = pred.new_tensor(quantiles).view(1, 1, -1)
    err = target.unsqueeze(-1) - pred
    return torch.mean(torch.maximum(q * err, (q - 1) * err))
