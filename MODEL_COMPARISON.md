# Model comparison (uniform trailing holdout)

Last actual: **2026-10-12 14:00 IST**. Horizon **48 h**. No shuffle.

**Split (same for every model)**

| Piece | Rule |
|---|---|
| Test | last *N* days of each meter |
| Val | 7 days immediately before test (selection only, not scored) |
| Train | all earlier hours |
| Origins | every hour with a complete 48 h of actuals |
| Scored hours | `n_origins × 48` |

Canonical table uses **N = 14 days**.

| Site | Train hours | Test calendar hours | Test origins | Scored hours |
|---|---|---|---|---|
| KA-1608 | 2,295 | 337 | 290 | 13,920 |
| TL-2362 | 2,990 | 337 | 290 | 13,920 |

---

## GBM

- **Method:** Direct 48 h `HistGradientBoostingRegressor` (sklearn). One tree ensemble predicts every lead 0…47 from last kW, lag-24, lag-168, 24 h mean/max, hour/weekday Fourier, Open-Meteo temperature and GHI at the target hour.
- **Train / test:** trailing split above. Val used only to pick GBM vs naive.
- **Test duration:** 337 hours (14 days).
- **KA-1608:** MAE **0.096 kW**, RMSE **0.140 kW**
- **TL-2362:** MAE **2.065 kW**, RMSE **2.679 kW**

Winner on KA. Loses to yesterday-copy on TL.

## Yesterday-copy

- **Method:** Repeat the last 24 h of actuals for each lead, lock hour 0 to the last measured kW (exponential decay τ = 6 h).
- **Train / test:** no fit. Scored on the same origins as GBM.
- **Test duration:** 337 hours (14 days).
- **KA-1608:** MAE **0.124 kW**, RMSE **0.176 kW**
- **TL-2362:** MAE **1.777 kW**, RMSE **2.581 kW**

Winner on TL.

## Last-week-copy

- **Method:** Copy the value from 168 h earlier, same hour-0 lock as yesterday-copy.
- **Train / test:** no fit. Same origins.
- **Test duration:** 337 hours (14 days).
- **KA-1608:** MAE **0.134 kW**, RMSE **0.198 kW**
- **TL-2362:** MAE **1.901 kW**, RMSE **2.736 kW**

## TiDE

- **Method:** Per-site TiDE (encoder–decoder residual MLP), 168 h lookback, pinball p10/p50/p90, similar-day profile with hour-0 lock. City Open-Meteo covariates.
- **Train / test:** same calendar cut (train → 7-day val → last 14 days). Origins also need 168 consecutive finite history hours, so fewer windows than the tabular models (KA 167, TL 289).
- **Test duration:** 337 hours (14 days). Scored hours: KA 8,016; TL 13,872.
- **KA-1608:** MAE **0.136 kW**, RMSE **0.202 kW**
- **TL-2362:** MAE **1.829 kW**, RMSE **2.569 kW**

Tied with last-week-copy on KA. Not used for live issue; the notebook issues GBM (KA) / yesterday (TL).

---

## 14-day board

| Model | KA MAE | KA RMSE | TL MAE | TL RMSE |
|---|---|---|---|---|
| **GBM** | **0.096** | **0.140** | 2.065 | 2.679 |
| Yesterday-copy | 0.124 | 0.176 | **1.777** | **2.581** |
| Last-week-copy | 0.134 | 0.198 | 1.901 | 2.736 |
| TiDE | 0.136 | 0.202 | 1.829 | 2.569 |

Units are kW. KA mean load ≈ 0.40 kW; TL mean load ≈ 1.43 kW.

---

## Same method, longer test

Retrain GBM on whatever remains after the val block; naives are re-scored only. Test window grows backward from 12 Oct 14:00.

### KA-1608

| Test days | Test hours | Train hours | Origins | GBM MAE | GBM RMSE | Yesterday MAE | Yesterday RMSE |
|---|---|---|---|---|---|---|---|
| 7 | 169 | 2,461 | 122 | 0.113 | 0.172 | 0.141 | 0.205 |
| **14** | **337** | **2,295** | **290** | **0.096** | **0.140** | 0.124 | 0.176 |
| 21 | 503 | 2,127 | 408 | 0.096 | 0.139 | 0.117 | 0.167 |
| 28 | 671 | 1,959 | 576 | 0.111 | 0.157 | 0.127 | 0.185 |

GBM stays ahead of both naives at every length. 7-day is noisier (includes the 11 Oct evening spike). 28-day is a bit worse: train shrinks and the extra week is harder. 14–21 days are stable.

### TL-2362

| Test days | Test hours | Train hours | Origins | GBM MAE | GBM RMSE | Yesterday MAE | Yesterday RMSE |
|---|---|---|---|---|---|---|---|
| 7 | 169 | 3,158 | 122 | 2.070 | 2.683 | 1.829 | 2.820 |
| **14** | **337** | **2,990** | **290** | 2.065 | 2.679 | **1.777** | 2.581 |
| 21 | 505 | 2,822 | 458 | 1.593 | 2.270 | 1.410 | 2.222 |
| 28 | 673 | 2,654 | 626 | 1.340 | 1.998 | 1.234 | 2.012 |

Yesterday-copy wins at every length. Absolute MAE falls as the test grows because recent weeks are the spikiest; **wMAPE stays ~57%**. A longer test does not make TL predictable. It averages a hard week with calmer ones.

CSV: `outputs/uniform_holdout_sweep.csv`
