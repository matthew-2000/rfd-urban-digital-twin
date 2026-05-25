# AGENTS.md

Living spec for `rfd-urban-digital-twin-exam`. If implementation changes, update this file first or same commit.

## Project summary

Small exam project on urban air-quality consistency profiling with Relaxed Functional Dependencies (RFDs).

Focus:

- reproducible Python analysis;
- simplified Urban Digital Twin framing;
- interpretable approximate rules;
- lightweight discovery and validation;
- exported tables and figures for report/slides.

Do not add:

- dashboards;
- deep learning;
- real-time systems;
- causal claims;
- full DiMε implementation;
- complex GUIs.

## Dataset choice

Primary dataset: Beijing Multi-Site Air Quality Data Set.

Local raw files expected in:

- `data/raw/PRSA_Data_Aotizhongxin_20130301-20170228.csv`
- `data/raw/PRSA_Data_Changping_20130301-20170228.csv`
- remaining Beijing station files may also exist in `data/raw/`

Selected stations:

- `Aotizhongxin`
- `Changping`

Selected period:

- `2013-09-01 00:00:00` to `2013-12-31 23:00:00`

Reason:

- requested Sep-Nov window produced only `3996` cleaned rows after dropna;
- project target was roughly `5000-6000` rows;
- adding nearby month December produced `5426` cleaned rows, within target.

Processed dataset output:

- `data/processed/udt_rfd_dataset.csv`

Final kept variables:

- `datetime`
- `station`
- `hour`
- `PM2.5`
- `PM10`
- `NO2`
- `O3`
- `TEMP`
- `DEWP`
- `WSPM`
- `time_slot`

Dropped from final dataset:

- `SO2`
- `CO`
- `PRES`
- `RAIN`
- `wd`
- `No`

Derived variable:

- `time_slot`
  - `night`: 0-5
  - `morning`: 6-11
  - `afternoon`: 12-17
  - `evening`: 18-23

Missing-value policy:

1. compute missing summary on selected columns;
2. drop rows with missing values in selected columns;
3. no imputation.

## Repository layout

Required structure:

```text
AGENTS.md
README.md
requirements.txt
data/
  raw/
  processed/
notebooks/
src/
results/
figures/
```

Core code files:

- `src/preprocessing.py`
- `src/profiling.py`
- `src/rfd.py`
- `src/experiments.py`
- `src/visualization.py`
- `notebooks/01_rfd_udt_analysis.ipynb`

## Required outputs

Processed data:

- `data/processed/udt_rfd_dataset.csv`
- `data/processed/udt_rfd_sample.csv`

Results:

- `results/profile_shape_summary.csv`
- `results/profile_dtypes_summary.csv`
- `results/profile_missing_summary.csv`
- `results/profile_unique_values_summary.csv`
- `results/profile_numeric_summary.csv`
- `results/profile_station_distribution.csv`
- `results/profile_hour_distribution.csv`
- `results/profile_time_slot_distribution.csv`
- `results/profile_correlation_matrix.csv`
- `results/rfd_candidate_results.csv`
- `results/rfd_discovered_top10.csv`
- `results/rfd_threshold_comparison.csv`
- `results/rfd_station_comparison.csv`
- `results/violations_examples.csv`

Figures:

- `figures/missing_values.png`
- `figures/pm25_distribution.png`
- `figures/no2_distribution.png`
- `figures/pm25_timeseries.png`
- `figures/correlation_matrix.png`
- `figures/rfd_confidence_by_threshold.png`
- `figures/rfd_confidence_by_station.png`

## Profiling requirements

Notebook and modules must produce:

- row/column count;
- data types;
- missing values;
- unique values;
- numeric min/max/mean/std;
- PM2.5 distribution;
- NO2 distribution;
- PM2.5 time series;
- correlation matrix.

Implementation note:

- PM2.5 time-series figure uses daily mean by station for readability.

## RFD scope

Implement in `src/rfd.py`:

- `is_similar()`
- `validate_rfd()`
- lightweight discovery over LHS sizes 1-3

Threshold configs:

- `strict`
- `medium`
- `relaxed`

Default candidate RFDs:

- `station, hour, TEMP -> NO2`
- `station, TEMP, DEWP -> PM2.5`
- `station, PM2.5 -> PM10`
- `station, TEMP, WSPM -> O3`
- `hour, NO2 -> PM2.5`

Discovery search space:

LHS attributes:

- `station`
- `hour`
- `time_slot`
- `TEMP`
- `DEWP`
- `WSPM`
- `PM2.5`
- `NO2`

RHS attributes:

- `PM2.5`
- `PM10`
- `NO2`
- `O3`

Keep discovered rules only if:

- `support >= 0.01`
- `confidence >= 0.85`

RFD experiment runtime policy:

- keep full cleaned dataset for preprocessing and profiling;
- use deterministic balanced sample of `1500` rows (`750` per station) for quadratic RFD validation and discovery;
- save sample to `data/processed/udt_rfd_sample.csv`;
- if no discovered rule satisfies filtering thresholds, keep `results/rfd_discovered_top10.csv` as header-only and report that outcome explicitly.

## Interpretation constraints

Always state:

- UDT is conceptual, not 3D platform;
- RFDs are approximate regularities, not causal laws;
- violations may reflect anomalies, data-quality issues, or unobserved events;
- pairwise validation is quadratic, so dataset is intentionally reduced.

## Development order

1. preprocessing and cleaned dataset;
2. profiling and figures;
3. RFD validation;
4. lightweight discovery;
5. threshold comparison;
6. station comparison;
7. violation export;
8. notebook execution and README finalization.
