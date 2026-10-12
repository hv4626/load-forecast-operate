# Model card — rooftop load TiDE

**Go-live verdict: FAIL.** Do not use this checkpoint for dispatch, bidding, or
battery control. It can run as a research / Operate-style viewer.

Evaluated 12 Oct 2026 on a 14-day holdout (456 hourly origins × 48 h).

## What the model is

Per-site TiDE (Das et al. 2023): 168 h lookback, 48 h horizon, pinball p10/p50/p90.
p50 at the issue hour is locked to the last actual (similar-day profile + origin
decay). Weather covariates from Open-Meteo (city proxy, not the rooftop).

Checkpoint: `outputs/tide.pt`

## Holdout (14 days)

| Site | TiDE wMAPE | Weekly naive wMAPE | Skill vs naive | p10–p90 coverage | Origin jump p90 |
|---|---|---|---|---|---|
| KA-1608 | 31.5% | 31.6% | +0.4% | **0.79** | **0.00** |
| TL-2362 | 59.0% | 59.9% | +1.5% | 0.57 | **0.00** |
| Combined | 56.9% | 57.8% | +1.5% | 0.65 | 0.00 |

MAPE is not used for go-live. Overnight residential load sits near 0.2 kW and
blows MAPE up. wMAPE / MAE / peak MAE are the production scores.

## Gates

| Gate | Result |
|---|---|
| Beat weekly naive by >5% relative | FAIL (1.5%) |
| KA advisory wMAPE < 20% | FAIL (31.5%) |
| TL advisory wMAPE < 40% | FAIL (59.0%) |
| TL 80% interval coverage 70–92% | FAIL (57%) |
| TL peak MAE ≤ naive | FAIL |
| Origin continuity | PASS |
| Quantile monotonicity | PASS |
| KA coverage | PASS |

## Why it fails production

1. **Almost no skill over last-week similar-day.** An MLP that copies last week
   and locks hour 0 is what this checkpoint is. Weather residual does not pay
   for itself on 4–6 months of two houses.
2. **TL-2362 is occupancy / appliance dominated.** Load jumps 0.2 kW → 7 kW
   inside an afternoon. Temperature at Hyderabad cannot see a geyser or AC
   switching. Interval coverage 57% means the band is not decision-grade.
3. **City weather, not site weather.** Bengaluru / Hyderabad Open-Meteo, not
   the meter coordinates.
4. **Single holdout window** (last 14 days). No rolling-origin across seasons.
   No monsoon vs winter split.

An earlier shared-model run reached KA wMAPE 26% vs naive 32% but jumped the
TL origin by ~8× the last actual. That is unsafe to issue. This checkpoint
prefers origin-safe over that extra KA skill.

## Allowed use

- Dashboard / operator glance for **KA-1608 only**, with the naive forecast
  shown beside it.
- Research on the pipeline (prepare → TiDE → Operate UI → gates).

## Not allowed

- Auto-scheduling batteries or EVSE from p50.
- Serving TL-2362 as a “forecast” to a customer.
- Claiming ISO-grade accuracy. Pravāh Operate is trained on years of ISO MW;
  this is months of two rooftops.

## What would change the verdict

- ≥ 18 months of 5-minute load, site-level weather, occupancy or submeter
  channels for TL.
- Rolling-origin wMAPE < 20% (KA) and < 35% (TL) **and** >10% skill vs
  similar-day **and** 80% band coverage in 75–85%.
- Peak MAE better than naive on both sites.
- Shadow-mode for 30 days against live meter data before any control loop.

Re-run gates: `PYTHONPATH=src python src/evaluate.py`
The process must print `verdict PASS` before go-live is discussed.
