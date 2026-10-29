"""Sites, paths, and forecast geometry.

Mirrors Pravāh Operate: hourly load, 48 h horizon, weather covariates,
probabilistic bands. Two rooftop meters from the lifetime-daily workbook.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
WEB_DIR = ROOT / "web"

# Source workbook (Downloads). Copied into data/ on prepare.
SOURCE_XLSX = Path(
    "/Users/harshvardhan/Downloads/"
    "Load_data_lifetime_daily_RSSEKA1608_RSSETL2362 (1).xlsx"
)

LOOKBACK = 168  # 7 days of hourly history (TiDE encoder)
HORIZON = 48  # Operate 2D horizon
# Percentiles served by Operate (NYISO ZONF, issue 2026-10-29T16:00Z).
# The point forecast is a separate `value`, not one of these levels.
# On all 60 captured hours, value was inside p25–p75.
QUANTILES = (0.01, 0.05, 0.25, 0.75, 0.95, 0.99)
# Channel order of TideQuantile: the six percentiles with value in the middle.
OUTPUT_NAMES = ("p01", "p05", "p25", "value", "p75", "p95", "p99")
VALUE_INDEX = 3
# `n100` in the published id `{region}-tide-hpo1.1.n100`.
HIDDEN = 100
FREQ = "h"
TZ = "Asia/Kolkata"

# Hold out the most recent two weeks for Verify metrics.
TEST_DAYS = 14
VAL_DAYS = 7

SITES = {
    "RSSEKA1608": {
        "label": "KA-1608",
        "subtitle": "Karnataka rooftop residential",
        "timezone": TZ,
        "lat": 12.9716,
        "lon": 77.5946,
        "city": "Bengaluru",
        "sheet_5min": "RSSEKA1608_5min",
        "sheet_daily": "RSSEKA1608",
        "schema": "voltage_first",
    },
    "RSSETL2362": {
        "label": "TL-2362",
        "subtitle": "Telangana rooftop · 8 kWp residential",
        "timezone": TZ,
        "lat": 17.3850,
        "lon": 78.4867,
        "city": "Hyderabad",
        "sheet_5min": "RSSETL2362_5min",
        "sheet_daily": "RSSETL2362",
        "schema": "frequency_first",
    },
}

# Weather the Operate desk charts, in that order: temperature, GHI, wind, humidity.
# City Open-Meteo stands in for Operate's station-average. Apparent temperature,
# precipitation, and cloud cover are not on that desk.
WEATHER_HOURLY = [
    "temperature_2m",
    "shortwave_radiation",
    "wind_speed_10m",
    "relative_humidity_2m",
]

# india_kw is the hour-of-week mean of load_p_load_total from Battery_Data.xlsx.
COVARIATES = WEATHER_HOURLY + [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
    "is_weekend",
    "is_holiday",
    "india_kw",
]

BATTERY_XLSX = Path("/Users/harshvardhan/Downloads/Battery_Data.xlsx")

# Indian public holidays overlapping Apr–Oct 2026 on the meters.
HOLIDAYS = {
    "2026-04-14",  # Ambedkar Jayanti / Puthandu
    "2026-05-01",  # Labour Day
    "2026-05-27",  # Eid al-Adha (approx)
    "2026-08-15",  # Independence Day
    "2026-08-27",  # Ganesh Chaturthi (approx)
    "2026-10-02",  # Gandhi Jayanti
    "2026-10-20",  # Dussehra (approx)
}
