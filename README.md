# Rooftop load forecast (Pravāh Operate replica)

**Production verdict: FAIL.** See [MODEL_CARD.md](MODEL_CARD.md). Origin-safe 48 h
TiDE for two rooftop meters. Do not dispatch batteries or bid from this
checkpoint. The dashboard is a research Operate desk.

## What this copies

[Pravāh Operate](https://load.pravah.com/operate?iso=nyiso&region=nyiso-system&horizon=48)
issues hourly demand, a TiDE forecast (`*-tide-hpo1.1.n100`), a percentile band,
and weather on one time axis, horizon 48 h.

This repo does the same for two SolarSquare rooftop meters, not an ISO.

## Holdout (14 days, 456 origins)

| Site | TiDE wMAPE | Weekly naive | Coverage | Origin jump |
|---|---|---|---|---|
| KA-1608 | 31.5% | 31.6% | 0.79 | 0 |
| TL-2362 | 59.0% | 59.9% | 0.57 | 0 |

Skill over last week is ~1%. That is not production skill.

## Run

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
src/prepare.py     5-minute meters → hourly kW + Open-Meteo
src/model.py       TiDE, origin-locked similar-day skip, monotone quantiles
src/train.py       per-site train, baselines
src/evaluate.py    production gates
src/infer.py       issue a 48 h forecast
web/               Operate / Verify UI
outputs/tide.pt    checkpoint
```
