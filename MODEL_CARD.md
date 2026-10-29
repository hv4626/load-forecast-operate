# Model card — rooftop load TiDE

**Go-live verdict: FAIL.** Do not use this checkpoint for dispatch, bidding, or
battery control. It can run as a research / Operate-style viewer. The issued
desk is still the GBM / yesterday-copy path.

Evaluated 29 Oct 2026 on the trailing 14-day holdout (575 origins × 48 h)
of `data/hourly_panel.csv`. Last actual in that panel is 29 Oct 2026 14:00 IST.

## What the model is

Per-site TiDE (Das et al. 2023), hidden width 100. 168 h lookback, 48 h horizon.
The level is each meter's own hour-of-day curve from `Battery_Data.xlsx`
column `load_p_load_total` (watts, stored as kW), fit only on hours before
the validation cut (8 Oct 2026 14:00 IST). Weekday and weekend curves are used
when that hour has at least 8 train samples. A level error versus the origin
hour fades with lead (τ = 6 h). Lead 0 of value is the last actual.

On that train cut, KA-1608 peaks at 20:00 (0.68 kW) and TL-2362 at 12:00
(4.09 kW). The full battery file, through 29 Oct 14:00, peaks at 19:00
(0.71 kW) and 12:00 (4.38 kW). Those are house-load shapes, not `battery_p_batt`.

The head matches the Operate contract observed on NYISO ZONF: a point `value`
plus p01, p05, p25, p75, p95, p99, with value forced inside p25–p75. Covariates
are city Open-Meteo temperature, GHI, wind speed, and relative humidity, plus
hour/weekday/day-of-year Fourier, weekend, holiday, and the raw profile `india_kw`.
This is not Pravāh's trained `tide-hpo1.1.n100` weights.

Checkpoint: `outputs/tide.pt`. The weekly-copy Operate head from earlier the
same day is `outputs/tide_operate_weekly.pt`. The p10/p50/p90 head is
`outputs/tide_p10p50p90.pt`.

## Holdout (14 days)

| Site | TiDE wMAPE | Weekly naive wMAPE | Skill vs naive | p05–p95 coverage | Origin jump p90 |
|---|---|---|---|---|---|
| KA-1608 | 23.4% | 41.1% | +43.0% | 0.989 | **0.00** |
| TL-2362 | 54.8% | 73.7% | +25.6% | **0.87** | **0.00** |
| Combined | 50.6% | 69.4% | +27.0% | **0.93** | 0.00 |

Combined p25–p75 coverage is 0.68 (nominal 0.50). Combined p01–p99 coverage is
1.00. Quantile crossings are 0. KA's p05–p95 band covers 0.989, wider than the
0.98 cap. TL peak MAE is 1.13 kW against a weekly-naive peak MAE of 0.26 kW.
KA peak MAE is 0.19 kW against 0.33 kW.

MAPE is not used for go-live. Overnight residential load sits near 0.2 kW and
blows MAPE up. wMAPE / MAE / peak MAE are the production scores.

## Gates

| Gate | Result |
|---|---|
| Beat weekly naive by >5% relative | PASS (+27.0%) |
| KA advisory wMAPE < 20% | FAIL (23.4%) |
| TL advisory wMAPE < 40% | FAIL (54.8%) |
| p05–p95 coverage 80–98% | FAIL on KA (0.989); TL 0.87 passes |
| TL peak MAE ≤ naive | FAIL (1.13 vs 0.26 kW) |
| Origin continuity | PASS |
| Quantile monotonicity | PASS |
| KA peak MAE ≤ naive | PASS (0.19 vs 0.33 kW) |

## Why it fails production

1. **KA's 90% band is too wide.** p05–p95 covers 98.9% of holdout hours. The
   gate cap is 98%. The curve improved the point forecast; it did not tighten
   the band.
2. **TL-2362 is occupancy / appliance dominated.** The midday climatology is
   real (train peak 4.09 kW at 12:00), and average wMAPE moved from 58.2% to
   54.8%. Spikes still move by hours. Peak MAE is 1.13 kW against a weekly
   naive of 0.26 kW. Hyderabad temperature cannot see a geyser or AC switching.
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
