# Počasie vs. meškanie letov – TGCN vs. ConvLSTM vs. CNN-LSTM

Predikcia hodinového meškania odletov na 6 letiskách západného pobrežia (SEA → PDX → SFO → SJC → LAX → SAN)
v roku 2018, horizonty 1 / 3 / 6 h, porovnanie modelov po typoch počasia.

## Prostredie

Python 3.11 (`py -3.11`). Všetky príkazy sa spúšťajú z koreňa repozitára ako moduly (`-m`).

```bash
py -3.11 -m pip install pandas numpy pyarrow scipy scikit-learn statsmodels matplotlib seaborn pyyaml
```

## Dáta (osoba A) – stav: hotové na reálnych dátach

Surové súbory patria do `data/raw/` (zdieľané cez OneDrive, nie v Gite):

```
data/raw/WeatherEvents_Jan2016-Dec2022.csv
data/raw/flights/2018_flights_on_time.csv
data/raw/flights/airports_lookup.csv
```

Spracovanie (spolu ~2 min):

```bash
py -3.11 -m src.data.clean_flights          # data/interim/flights_clean.parquet, flights_hourly.csv
py -3.11 -m src.data.weather_hourly         # data/interim/weather_hourly.csv, weather_region_hourly.parquet
py -3.11 -m src.data.build_processed        # data/processed/* (dátový kontrakt)
py -3.11 -m src.baselines.baselines         # results/preds_<baseline>_<split>.npz + metriky
py -3.11 -m src.analysis.temporal.run_all   # tabuľky/obrázky kapitoly A
```

### Čo kto používa

| Kto | Súbory |
|---|---|
| **B** (priestor) | `data/interim/flights_hourly.csv` a `weather_hourly.csv` (8 760 h × 6 letísk, UTC), `flights_clean.parquet` (všetky lety cez 6 letísk vrátane `dep_utc`/`arr_utc`, `Tail_Number`, príčin meškania), `weather_events_6.csv` (2016–2022), `weather_stations_region.csv` |
| **C** (TGCN, ConvLSTM) | `WindowDataset(mode="graph" / "grid", ...)`, `data/processed/*`, formát a metriky v `src/baselines/evaluate.py` (`save_predictions`, `metrics_table`) |
| **D** (mriežka, CNN-LSTM) | `data/interim/weather_region_hourly.parquet` (227 staníc × hodina, riedko), `weather_stations_region.csv` (stanica → `grid_row`, `grid_col`), `data/processed/grid_index.json`; `X_grid.npy` [8760, C, 17, 9] uložiť do `data/processed/` – `WindowDataset(mode="grid")` ho načíta (+ voliteľne `scaler_grid.json`) |

### Dátový kontrakt (`data/processed/`)

| Súbor | Tvar | Poznámka |
|---|---|---|
| `X_nodes.npy` | [8760, 6, 22] float32 | **nenormalizované**; názvy vo `feature_names.json` |
| `Y.npy` | [8760, 6] float32 | priemerný `DepDelay` (min, orezaný na [−60, 360]); 0, kde mask = 0 |
| `mask.npy` | [8760, 6] uint8 | 1 = aspoň 3 uskutočnené odlety |
| `weather_cat.npy` | [8760, 6] int8 | 0 Jasno, 1 Dážď slabý, 2 Dážď stredný/silný, 3 Hmla, 4 Sneh |
| `timestamps.csv` | 8 760 riadkov | `t`, `time_utc`, `time_local`, časť každého rozdelenia |
| `split.json` | – | `main` + foldy `Q1`–`Q4` ako inkluzívne rozsahy indexov, medzera 30 h |
| `scaler.json` | – | μ, σ z tréningu pre každé rozdelenie zvlášť |
| `grid_index.json` | 6 × (riadok, stĺpec) | riadok 0 = 49° N, stĺpec 0 = 125° W |

Poradie uzlov všade: `[SEA, PDX, SFO, SJC, LAX, SAN]`.

### WindowDataset

```python
from src.data.window_dataset import WindowDataset

tr = WindowDataset(mode="graph", L=24, H=(1, 3, 6), split="main", part="train")   # PyTorch Dataset
item = tr[0]   # x [24,6,22], y [3,6], mask [3,6], weather_cat [3,6], t0
arr = WindowDataset(split="Q1", part="test").as_arrays()   # pre Keras: dict s osou S
# normalize_y=True -> cieľ normalizovaný, späť cez ds.inverse_y(...)
# oracle_weather=True -> navyše x_future_weather (počasie v T+h) – iba označený experiment
```

### Predikcie a hodnotenie

Každý model uloží test ako `results/preds_<model>_<variant>.npz`:
`y_true, y_pred, mask, weather_cat` [S, 3, 6], `t0` [S], `horizons` [3].

```python
from src.baselines.evaluate import save_predictions, load_predictions, metrics_table
p = load_predictions([f"results/preds_tgcn_E1_{q}.npz" for q in ["Q1","Q2","Q3","Q4"]])
metrics_table(p, by=("horizon", "weather_cat"))   # MAE/RMSE + n + 95 % blokový bootstrap
```

**Referenčné baseline (hlavný test, MAE min, h = 1/3/6):** lineárny VARX 6,76 / 7,52 / 8,31;
sezónny + AR(1) 7,53 / 7,88 / 8,19; persistence 8,29 / 9,40 / 10,51.

## Štruktúra

```
config.yaml        spoločné parametre (L, H, uzly, rozdelenia, kategórie, mriežka)
data/              raw / interim (A) / processed (A, B: adj_*, D: X_grid)
src/data/          čistenie, agregácia, kontrakt, WindowDataset              (A)
src/analysis/temporal/  obsah, časové vlastnosti, vplyv počasia, kontroly  (A)
src/baselines/     persistence, sezónny, sezónny+AR(1), VARX, hodnotenie    (A)
src/analysis/spatial/, src/graph/                                          (B)
src/models/tgcn/, src/models/convlstm/, src/eval/                           (C)
src/grid/, src/models/cnn_lstm/                                            (D)
notebooks/         01_temporal.ipynb (A)
results/           preds_*.npz, tables/A, figures/A
report/            01_data_a_casova_analyza.md (A)
```
