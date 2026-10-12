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
QUANTILES = (0.10, 0.50, 0.90)
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

# Open-Meteo fields used as future covariates (same role as Indus/NWP in Operate).
WEATHER_HOURLY = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "shortwave_radiation",
    "wind_speed_10m",
    "cloud_cover",
]

COVARIATES = WEATHER_HOURLY + [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
    "is_weekend",
    "is_holiday",
]

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
