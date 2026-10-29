# What the Pravāh Operate capture showed

Captured 29 Oct 2026 from the live desk

[load.pravah.com/operate?iso=nyiso&region=nyiso-zonf&horizon=48](https://load.pravah.com/operate?iso=nyiso&region=nyiso-zonf&horizon=48)

Anonymous read of the public forecast API. This note records the contract we
copied into the house-load model. It is not Pravāh's training code, and the
checkpoint in this repo is not their weights.

## Product

The page is a Vite app titled "System Load Forecast by Pravāh". The forecast
comes from `GET /api/system_forecast/v1/public-access` with `iso`, `region`,
and `horizon`.

On that session the 48 h horizon (labeled 2D) was the live product. The 3D,
7D, and 15D controls were disabled. The region on the capture was NYISO ZONF
(Capital). The region clock is `America/New_York`. The browser was on UTC, so
UTC is not the site default.

## Model ids

The id on ZONF was `nyiso-zonf-tide-hpo1.1.n100`. The same pattern,
`{region}-tide-hpo1.1.n100`, covered the other NYISO zones in that response,
with one exception: `nyiso-zonj` used
`nyiso-zonj-tide-load-stations-openmeteo-era5-2d-tiny`.

`n100` is the only width the id gives. The rest of the `hpo1.1` search was
not in the payload. This repo's inference id is `{site}-tide-n100` so it does
not claim those weights. Hidden width 100 is the part we matched.

## Forecast contract

Each point is hourly. The series is `load_mw`, unit MW. A response carries an
`issue_token`, a `points` list, and an empty `ar_fill`.

A point has `timestamp`, `issue_time`, a central `value`, and six percentiles:
`p01`, `p05`, `p25`, `p75`, `p95`, `p99`. There is no `p10`, `p50`, or `p90`
on that desk. On all 60 ZONF chart hours, `value` sat inside `p25`–`p75`.

Actuals on the chart are EIA-930. In the capture they ended at
`2026-10-29T04:00:00Z`. The forecast starts the hour after that last actual
and runs through the issue plus 48 h. That is an ISO reporting lag. The house
model does not copy it. Lead 0 here is the last meter hour, because the meter
has already reported that hour.

## Weather

The desk charts four station-average series, in this order: temperature, GHI,
wind speed, relative humidity. The house model maps those to city Open-Meteo
`temperature_2m`, `shortwave_radiation`, `wind_speed_10m`, and
`relative_humidity_2m`. Apparent temperature, precipitation, and cloud cover
are not on that desk.

## What the house model took from this

- Horizon 48 h, lookback 168 h, hidden width 100.
- A point `value` plus `p01 p05 p25 p75 p95 p99`.
- `value` trained with a median pinball loss and forced inside `p25`–`p75`
  by nested softplus spreads.
- The coverage gate is the published 90% band, `p05`–`p95`, accepted in
  `[0.80, 0.98]`. Conformal offsets are not allowed to overwrite those bands.
- The four weather series above.
- Lead 0 of `value` locked to the last actual.

## What it did not take

- Pravāh's trained weights, or the rest of `hpo1.1`.
- An ISO megawatt target, EIA-930 actuals, or the reporting lag.
- Station-average weather. These two meters use one city point each
  (Bengaluru, Hyderabad).
- A weekly copy as the level. After the capture, the level was changed to
  each meter's own hour-of-day curve from `load_p_load_total`. See
  [MODEL_CARD.md](../MODEL_CARD.md).

## Why the same head is not a Pravāh-like forecast

Operate is forecasting a zone, a smooth sum of many customers, with years of
hourly actuals and a weather field. This repo forecasts two houses and a few
months of hourly load.

KA-1608 is an evening-peak house (full-file peak 19:00, about 0.71 kW).
TL-2362 is a midday spike (12:00, about 4.4 kW) whose high hours move off
the clock. City temperature does not see those spikes. Matching the Operate
head improved average error. It did not make the Hyderabad peak error beat
copying last week. The go-live verdict stays FAIL until `evaluate.py` prints
`PASS`.
