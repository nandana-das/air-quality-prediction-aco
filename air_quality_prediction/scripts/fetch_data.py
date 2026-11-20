"""
Download the raw data used by this project.

1. Measured pollutant concentrations: XKDR Forum, India Air Quality Database
   (https://airquality.xkdr.org), compiled from the CPCB CAAQM continuous
   monitoring network. Licence CC BY 4.0. Credit: XKDR Forum, CPCB CAAQM
   network. Daily station means are downloaded per city / parameter / year.

2. Daily weather: Open-Meteo Historical Weather API (ERA5 reanalysis,
   https://open-meteo.com). Licence CC BY 4.0. No key needed.

Usage (from the air_quality_prediction/ directory):

    export XKDR_API_KEY=aqi_...        # free key from airquality.xkdr.org/signup
    python scripts/fetch_data.py

Files already downloaded are skipped, so the script can be re-run safely.
"""

import argparse
import io
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

XKDR_BASE = "https://airquality.xkdr.org/v1"
OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"
USER_AGENT = "air-quality-prediction-aco/1.0 (research script)"

# City name as used by XKDR -> (latitude, longitude) used for weather
CITIES = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Kolkata": (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
    "Chennai": (13.0827, 80.2707),
}

PARAMETERS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "SO2", "CO", "Ozone",
    "Benzene", "Toluene",
]

WEATHER_VARS = [
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
    "wind_direction_10m_dominant",
    "surface_pressure_mean",
    "precipitation_sum",
]


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    key = os.environ.get("XKDR_API_KEY")
    if key:
        s.headers["Authorization"] = f"Bearer {key}"
    return s


def _get(session, url, params, retries=4):
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=120)
            if r.status_code == 200:
                return r
            if r.status_code in (401, 403):
                sys.exit(f"Auth error {r.status_code} from {url}: {r.text[:200]}")
            print(f"  HTTP {r.status_code}, retrying...", flush=True)
        except requests.RequestException as e:
            print(f"  {e}, retrying...", flush=True)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed after {retries} attempts: {url} {params}")


def fetch_stations(session, out_dir: Path):
    path = out_dir / "stations.csv"
    if not path.exists():
        r = _get(session, f"{XKDR_BASE}/stations", {"format": "csv"})
        path.write_bytes(r.content)
    return pd.read_csv(path)


def fetch_pollutants(session, out_dir: Path, start_year: int, end_year: int):
    """Daily station-level means, one CSV per city/parameter/year."""
    part_dir = out_dir / "parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    for city in CITIES:
        for param in PARAMETERS:
            for year in range(start_year, end_year + 1):
                safe = param.replace(".", "")
                path = part_dir / f"{city}_{safe}_{year}.csv"
                if path.exists():
                    continue
                print(f"XKDR {city} {param} {year}", flush=True)
                r = _get(session, f"{XKDR_BASE}/measurements", {
                    "city": city, "parameter": param,
                    "start": f"{year}-01-01", "end": f"{year}-12-31",
                    "agg": "daily", "format": "csv",
                })
                df = pd.read_csv(io.BytesIO(r.content)) if r.content.strip() else pd.DataFrame()
                df["city"] = city
                df.to_csv(path, index=False)

    parts = [pd.read_csv(p) for p in sorted(part_dir.glob("*.csv"))]
    parts = [p for p in parts if "station_id" in p.columns and len(p)]
    daily = pd.concat(parts, ignore_index=True)
    daily = daily.rename(columns={"period_start": "date"})
    daily["date"] = pd.to_datetime(daily["date"]).dt.date
    daily.to_csv(out_dir / "xkdr_station_daily.csv", index=False)
    print(f"Saved {len(daily):,} station-day rows -> {out_dir / 'xkdr_station_daily.csv'}")


def fetch_weather(out_dir: Path, start_year: int, end_year: int):
    path = out_dir / "weather_daily.csv"
    frames = []
    for city, (lat, lon) in CITIES.items():
        print(f"Open-Meteo {city}", flush=True)
        r = _get(requests.Session(), OPEN_METEO, {
            "latitude": lat, "longitude": lon,
            "start_date": f"{start_year}-01-01", "end_date": f"{end_year}-12-31",
            "daily": ",".join(WEATHER_VARS),
            "timezone": "Asia/Kolkata",
        })
        d = pd.DataFrame(r.json()["daily"]).rename(columns={"time": "date"})
        d.insert(0, "city", city)
        frames.append(d)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    print(f"Saved weather -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start-year", type=int, default=2018)
    ap.add_argument("--end-year", type=int, default=2024,
                    help="CPCB gas data in XKDR ends 2024-12-31")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "raw"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = _session()
    fetch_stations(session, out_dir)
    fetch_pollutants(session, out_dir, args.start_year, args.end_year)
    fetch_weather(out_dir, args.start_year, args.end_year)


if __name__ == "__main__":
    main()
