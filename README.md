# Rooftop load forecast (Pravāh Operate replica)

**Production verdict: FAIL.** See [MODEL_CARD.md](MODEL_CARD.md). Origin-safe 48 h
TiDE for two rooftop meters. Do not dispatch batteries or bid from this
checkpoint. The dashboard is a research Operate desk.

## What this copies

[Pravāh Operate](https://load.pravah.com/operate?iso=nyiso&region=nyiso-system&horizon=48)
issues hourly demand, a TiDE forecast (`*-tide-hpo1.1.n100`), a percentile band,
and weather on one time axis, horizon 48 h.

This repo does the same for two SolarSquare rooftop meters, not an ISO.

## Dashboard

The issued desk is [web/index.html](web/index.html): GBM on KA-1608,
yesterday-copy on TL-2362, verdict FAIL. GitHub Pages serves that same file
from `docs/index.html`. Do not dispatch from it.

## Research model (India hour profile)

Per-site TiDE, 14-day holdout, 575 origins. The level is each meter's own
hour-of-day curve from `load_p_load_total`, fit before the validation cut.
Details and gates: [MODEL_CARD.md](MODEL_CARD.md). What the Operate capture
showed, and what was copied: [docs/pravah-rea.md](docs/pravah-rea.md).

| Site | TiDE wMAPE | Weekly naive | p05–p95 | Peak MAE vs naive |
|---|---|---|---|---|
| KA-1608 | 23.4% | 41.1% | 0.989 | 0.19 vs 0.33 kW |
| TL-2362 | 54.8% | 73.7% | 0.87 | 1.13 vs 0.26 kW |
| Combined | 50.6% | 69.4% | 0.93 | — |

Skill vs last week is +27%. KA's band is wider than the 98% cap, and TL's
peak error is still worse than last week. Verdict FAIL.

## Notebook (use this on a new dump)

[`notebooks/load_forecast.ipynb`](notebooks/load_forecast.ipynb) is the R&D model.

Set `DUMP` to any lifetime Excel/CSV and run all cells. It trains a direct 48 h gradient-boosting model, scores it against yesterday-copy and last-week-copy, and **issues the winner**.

Holdout on the bundled dumps (14 days, 145 origins × 48 h):

| Site | Winner | wMAPE | vs yesterday | vs last week | MAE |
|---|---|---|---|---|---|
| KA-1608 | **GBM** | **23.9%** | 29.3% | 32.5% | **0.10 kW** |
| TL-2362 | yesterday-copy | 57.4% | (best) | 61.8% | 1.78 kW |

KA is a regular house: the GBM cuts error ~18% vs copying yesterday. TL is occupancy spikes; copying yesterday is the honest forecast and the notebook selects it automatically (`burst = std/mean ≥ 1`).

```bash
source .venv/bin/activate
jupyter notebook notebooks/load_forecast.ipynb
```

## Run the Operate desk (research UI)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# optional: rebuild the hourly panel from the local lifetime workbook
python src/prepare.py
python src/run_all.py
PYTHONPATH=src python src/evaluate.py   # must print verdict PASS before go-live
PYTHONPATH=src python src/test_origin.py
python -m http.server 8765 --directory web
```

Inference: `PYTHONPATH=src python src/infer.py RSSEKA1608`

The lifetime workbook stays on the machine that owns it (`Downloads/…xlsx`).
It is not in git.

## Layout

```
docs/index.html        GitHub Page — the issued desk
docs/pravah-rea.md     Operate capture: contract, weather, what was copied
notebooks/load_forecast.ipynb   issuer — point DUMP at a new Excel/CSV
src/consumption.py house-load hour curves from load_p_load_total
src/model.py       TiDE, Indian hour-profile skip, Operate percentile head
src/train.py       per-site train, baselines
src/evaluate.py    production gates
src/infer.py       issue a 48 h research forecast
web/               source of the issued desk
outputs/tide.pt    India-profile checkpoint
MODEL_CARD.md      holdout numbers and the FAIL gates
```
