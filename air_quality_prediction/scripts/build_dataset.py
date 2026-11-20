"""
Build the modelling table from the raw downloads made by fetch_data.py.

Steps
-----
1. Station-day means from XKDR -> keep CPCB stations only, keep days with at
   least 16 valid hours (the CPCB 24-hour completeness rule), drop
   non-physical values.
2. City-day value = mean over that city's stations.
3. CPCB National AQI computed from the city-day concentrations.
   Approximation: CO and O3 officially use the maximum 8-hour average; the
   daily mean is used here because only daily aggregates are downloaded.
4. Join daily weather (Open-Meteo / ERA5).
5. Features known at the end of day t (pollutants, AQI, weather, lags,
   rolling means, calendar). Target = AQI of day t+1 in the same city.

Outputs:
    data/processed/city_daily_pollutants.csv  (city-day means, committed)
    data/processed/aqi_daily_features.csv     (modelling table)
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "aqi_daily_features.csv"

# Column names used in the modelling table
RENAME = {"PM2.5": "PM2.5", "PM10": "PM10", "NO": "NO", "NO2": "NO2", "NOx": "NOx",
          "NH3": "NH3", "SO2": "SO2", "CO": "CO", "Ozone": "O3",
          "Benzene": "Benzene", "Toluene": "Toluene"}

# CPCB National AQI breakpoints: (concentration low, high) per AQI band
# Bands: 0-50, 51-100, 101-200, 201-300, 301-400, 401-500
AQI_BANDS = [(0, 50), (51, 100), (101, 200), (201, 300), (301, 400), (401, 500)]
BREAKPOINTS = {
    "PM10": [(0, 50), (51, 100), (101, 250), (251, 350), (351, 430), (431, 1000)],
    "PM2.5": [(0, 30), (31, 60), (61, 90), (91, 120), (121, 250), (251, 500)],
    "NO2": [(0, 40), (41, 80), (81, 180), (181, 280), (281, 400), (401, 1000)],
    "O3": [(0, 50), (51, 100), (101, 168), (169, 208), (209, 748), (749, 1000)],
    "CO": [(0, 1.0), (1.1, 2.0), (2.1, 10), (10.1, 17), (17.1, 34), (34.1, 50)],
    "SO2": [(0, 40), (41, 80), (81, 380), (381, 800), (801, 1600), (1601, 2000)],
    "NH3": [(0, 200), (201, 400), (401, 800), (801, 1200), (1201, 1800), (1801, 2500)],
}

# Plausible upper limits for daily means; larger values are sensor errors
MAX_VALID = {"PM2.5": 1000, "PM10": 1500, "NO": 1000, "NO2": 1000, "NOx": 1000,
             "NH3": 2000, "SO2": 1000, "CO": 50, "O3": 1000, "Benzene": 200,
             "Toluene": 1000}


def sub_index(conc: pd.Series, pollutant: str) -> pd.Series:
    """Piecewise-linear CPCB sub-index. Gaps between bands are bridged by
    interpolating on the band lower bounds, so the function is continuous."""
    lows = np.array([b[0] for b in BREAKPOINTS[pollutant]] + [BREAKPOINTS[pollutant][-1][1]])
    idx = np.array([b[0] for b in AQI_BANDS] + [500])
    idx[0] = 0
    return pd.Series(np.interp(conc, lows, idx), index=conc.index).where(conc.notna())


def compute_aqi(df: pd.DataFrame) -> pd.DataFrame:
    subs = pd.DataFrame({p: sub_index(df[p], p) for p in BREAKPOINTS}, index=df.index)
    n_valid = subs.notna().sum(axis=1)
    has_pm = subs[["PM2.5", "PM10"]].notna().any(axis=1)
    # CPCB rule: at least 3 sub-indices, one of them PM2.5 or PM10
    ok = (n_valid >= 3) & has_pm
    df["AQI"] = subs.max(axis=1).where(ok).round()
    df["prominent_pollutant"] = subs.idxmax(axis=1).where(ok)
    return df


def city_daily() -> pd.DataFrame:
    st = pd.read_csv(RAW / "stations.csv")
    cpcb = set(st.loc[st["source"] == "cpcb_caaqm", "station_id"])

    d = pd.read_csv(RAW / "xkdr_station_daily.csv")
    d = d[d["station_id"].isin(cpcb) & (d["n"] >= 16)]
    d["pollutant"] = d["parameter_name"].map(RENAME)
    d = d.dropna(subset=["pollutant"])
    limit = d["pollutant"].map(MAX_VALID)
    d = d[(d["mean"] >= 0) & (d["mean"] <= limit)]

    city = (d.groupby(["city", "date", "pollutant"])["mean"].mean()
              .unstack("pollutant").reset_index())
    n_st = d.groupby(["city", "date"])["station_id"].nunique().rename("n_stations").reset_index()
    city = city.merge(n_st, on=["city", "date"], how="left")
    city["date"] = pd.to_datetime(city["date"])
    return city


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    # Regular daily grid per city so shift(1) always means "previous day"
    df = (df.set_index("date").groupby("city", group_keys=False)
            .apply(lambda g: g.asfreq("D").assign(city=g["city"].iloc[0]))
            .reset_index())

    pollutants = [p for p in RENAME.values() if p in df.columns]
    g = df.groupby("city")
    for col in pollutants + ["AQI"]:
        for lag in (1, 2, 3):
            df[f"{col}_lag{lag}"] = g[col].shift(lag)
        for w in (3, 7):
            df[f"{col}_avg{w}"] = g[col].transform(lambda s: s.rolling(w, min_periods=max(2, w - 2)).mean())

    df["PM_ratio"] = df["PM2.5"] / df["PM10"].replace(0, np.nan)

    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    # Indian seasons (IMD): winter Jan-Feb, pre-monsoon Mar-May,
    # monsoon Jun-Sep, post-monsoon Oct-Dec
    df["season"] = df["month"].map({1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2,
                                    9: 2, 10: 3, 11: 3, 12: 3})
    for c in sorted(df["city"].unique()):
        df[f"city_{c}"] = (df["city"] == c).astype(int)

    # Target: next day's AQI in the same city
    df["target_AQI_next_day"] = df.groupby("city")["AQI"].shift(-1)
    return df


CITY_DAILY = ROOT / "data" / "processed" / "city_daily_pollutants.csv"


def main():
    # The station-level download is large and not committed; the city-level
    # table it produces is, so the pipeline can be rebuilt without an API key.
    if (RAW / "xkdr_station_daily.csv").exists():
        df = city_daily()
        CITY_DAILY.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CITY_DAILY, index=False)
        print(f"Saved city-day pollutant means -> {CITY_DAILY}")
    else:
        print(f"{RAW / 'xkdr_station_daily.csv'} not found, using {CITY_DAILY}")
        df = pd.read_csv(CITY_DAILY, parse_dates=["date"])
    df = compute_aqi(df)

    w = pd.read_csv(RAW / "weather_daily.csv", parse_dates=["date"])
    w = w.rename(columns={
        "temperature_2m_mean": "temperature",
        "relative_humidity_2m_mean": "humidity",
        "wind_speed_10m_mean": "wind_speed",
        "wind_direction_10m_dominant": "wind_direction",
        "surface_pressure_mean": "pressure",
        "precipitation_sum": "precipitation",
    })
    df = df.merge(w, on=["city", "date"], how="left")
    df = add_features(df)

    df = df.dropna(subset=["AQI", "target_AQI_next_day"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df):,} rows x {df.shape[1]} cols -> {OUT}")
    print(df.groupby("city")["date"].agg(["min", "max", "count"]))


if __name__ == "__main__":
    main()
