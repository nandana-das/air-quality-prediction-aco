# Air Quality Prediction with Ant Colony Optimization (ACO) + Decision Trees

Next-day AQI forecasting for six Indian cities (Delhi, Mumbai, Bengaluru, Kolkata, Hyderabad, Chennai) using **measured** CPCB pollutant data and ERA5 weather, 2018–2024.

An Ant Colony Optimization search picks a small feature subset for a Decision Tree. The search is biased toward features that emission policy can act on (PM2.5, PM10, NO2, SO2, ...) rather than weather.

## Results (test year 2024, 2,190 city-days, never seen during training or feature selection)

| Model | Features | RMSE | MAE | R² |
|---|---:|---:|---:|---:|
| Random Forest (all features) | 91 | **23.9** | **14.4** | **0.903** |
| ACO + Bagged Trees | 12 | 25.2 | 14.9 | 0.893 |
| Persistence (tomorrow = today) | 1 | 26.7 | 14.9 | 0.879 |
| Decision Tree (all features) | 91 | 27.9 | 16.6 | 0.868 |
| ACO + Decision Tree | 12 | 28.0 | 16.5 | 0.867 |

What the numbers say:
- ACO cut the feature set from 91 to 12 (−87%); 11 of 12 are policy-controllable pollutants (PM2.5, PM10, NO2, NO and their lags / rolling means), 0 weather.
- With those 12 features, a single tree matches a tree on all 91 features. A bagged ensemble on the same 12 features is about 5% behind the Random Forest on all features.
- AQI is very persistent from day to day, so "tomorrow = today" is a strong baseline. Only the ensembles beat it, by 6–10% RMSE. A single Decision Tree does not.
- Errors are largest in Delhi (MAE ~30 AQI points) and smallest in Bengaluru and Hyderabad (~9).

Full tables, per-city errors and plots: `air_quality_prediction/notebooks/02_aco_development.ipynb`, `air_quality_prediction/results/`.

## Data

| | Source | Licence |
|---|---|---|
| Pollutants (PM2.5, PM10, NO, NO2, NOx, NH3, SO2, CO, O3, Benzene, Toluene) | XKDR Forum, [India Air Quality Database](https://airquality.xkdr.org), compiled from the CPCB CAAQM continuous monitoring network. 111 stations in the six cities. | CC BY 4.0 (credit XKDR Forum and CPCB) |
| Weather (temperature, humidity, wind, pressure, precipitation) | [Open-Meteo](https://open-meteo.com) Historical Weather API (ERA5 reanalysis) | CC BY 4.0 |

Coverage ends 2024-12-31 because CPCB gas data in the XKDR database ends there (only PM2.5/PM10 continue into 2025).

Processing (`scripts/build_dataset.py`):
1. Station-day means with at least 16 valid hours (CPCB completeness rule), CPCB stations only, non-physical values dropped.
2. City-day value = mean over the city's stations.
3. CPCB National AQI from the city-day concentrations (max sub-index; needs ≥3 pollutants including PM2.5 or PM10). Approximation: CO and O3 use the daily mean instead of the maximum 8-hour mean.
4. Features known at the end of day *t*: pollutants, AQI, weather, lags 1–3, 3- and 7-day rolling means, calendar, city. Target: AQI on day *t*+1.

Split by time: train 2018–2022, validation 2023 (used by ACO to score subsets), test 2024.

## Project structure

```
air_quality_prediction/
├── data/
│   ├── raw/
│   │   ├── stations.csv                 # XKDR station catalogue
│   │   ├── weather_daily.csv            # Open-Meteo daily weather
│   │   └── xkdr_station_daily.csv       # station-level download (not committed, ~145 MB)
│   └── processed/
│       ├── city_daily_pollutants.csv    # city-day pollutant means (committed)
│       └── aqi_daily_features.csv       # modelling table (built locally)
├── scripts/
│   ├── fetch_data.py                    # download XKDR + Open-Meteo data
│   └── build_dataset.py                 # AQI, features, target
├── src/aco_optimizer.py                 # ACO feature selection
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_aco_development.ipynb
├── tests/test_aco_optimizer.py
└── results/{models,plots,reports}/
```

## Running

```bash
cd air_quality_prediction
pip install -r requirements.txt
```

Build the modelling table from the committed city-level data (no API key needed):

```bash
python scripts/build_dataset.py
```

Then run `notebooks/01_data_exploration.ipynb` and `notebooks/02_aco_development.ipynb`.

To re-download everything from the source (free key from https://airquality.xkdr.org/signup, about 30 minutes):

```bash
export XKDR_API_KEY=aqi_...
python scripts/fetch_data.py
python scripts/build_dataset.py
```

Tests:

```bash
python -m pytest tests
```

## ACO method

- Each ant builds a subset of 4–15 features. At each step it takes the best-scoring feature with probability `q0`, otherwise samples by `pheromone^alpha * heuristic^beta`.
- Heuristic = |correlation with target| × 1.5 for policy-controllable features, ÷ 1.5 for weather features.
- Fitness (lower is better), scored on the 2023 validation set:
  `val_RMSE × (1 + 0.05·k/n − 0.03·share_policy + 0.03·share_weather)`
- MAX-MIN pheromone bounds, iteration-best and global-best deposits, early stop after 15 iterations without improvement, seeded RNG.

Feature categories are matched on the base variable, so `PM2.5_lag1` and `NO2_avg7` count as policy-controllable and `wind_speed` as weather.

## Limitations

- City value is an average of stations, so local hotspots are smoothed out.
- Weather is ERA5 reanalysis of the same day, not a forecast; tomorrow's weather is not used.
- CO and O3 sub-indices use daily means (see above), so AQI on CO/O3-dominated days is slightly underestimated.
- The persistence baseline is hard to beat; most skill comes from recent PM levels.

## Credits

Pollutant data: XKDR Forum (2026), India Air Quality Database, https://airquality.xkdr.org; Central Pollution Control Board CAAQM network. Weather: Open-Meteo.com, ERA5 (Copernicus Climate Change Service).
