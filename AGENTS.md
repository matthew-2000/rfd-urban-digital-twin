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

- `2013-03-01 00:00:00` to `2017-02-28 23:00:00`

Reason:

- this is the complete common temporal coverage of the two locally available
  station files;
- the definitive experiment uses all available observations for preprocessing
  and profiling;
- the resulting cleaned dataset contains `66619` rows;
- quadratic RFD experiments remain tractable through the unchanged deterministic
  balanced sample of `1500` rows.

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

Available local raw files as of this revision:

- `Aotizhongxin`
- `Changping`

The two-station design provides a balanced comparison between sites while
containing the quadratic cost of pairwise validation. The locally available raw
files support this methodological selection. The code remains compatible with
additional station files, but the report must not claim an all-station
experiment unless those files are added and the pipeline is rerun.

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
- `results/rfd_candidate_metrics.csv`
- `results/rfd_discovered_top10.csv`
- `results/rfd_threshold_comparison.csv`
- `results/rfd_station_comparison.csv`
- `results/rfd_bootstrap_summary.csv`
- `results/rfd_bootstrap_iterations.csv`
- `results/rfd_train_test_validation.csv`
- `results/rfd_window_evolution.csv`
- `results/rfd_violations_summary.csv`
- `results/rfd_top_violating_pairs.csv`
- `results/violations_examples.csv`

Figures:

- `figures/missing_values.png`
- `figures/pm25_distribution.png`
- `figures/no2_distribution.png`
- `figures/pm25_timeseries.png`
- `figures/correlation_matrix.png`
- `figures/rfd_confidence_by_threshold.png`
- `figures/rfd_confidence_by_station.png`
- `figures/rfd_lift_vs_baseline.png`
- `figures/rfd_confidence_over_time.png`
- `figures/rfd_violations_by_month_station.png`

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

- `station, PM2.5 -> PM10`
- `station, time_slot, PM2.5 -> PM10`
- `station, PM2.5, NO2 -> PM10`
- `station, hour, TEMP, WSPM -> O3`
- `station, time_slot, TEMP, WSPM -> O3`
- `station, time_slot, NO2 -> O3`
- `station, TEMP, DEWP, WSPM -> PM2.5`

General validation metrics:

- `support`
- `confidence`
- `violation_rate`
- `baseline_confidence` as the mean over `30` seeded RHS permutations
- `baseline_confidence_std` across the same permutations
- `lift = confidence / baseline_confidence`
- `antecedent_pairs`
- `valid_pairs`

Robustness experiments:

- bootstrap validation uses at least 30 balanced resamples;
- temporal validation preserves the original 75/25 chronological design: it
  trains on `2013-03-01` through `2016-02-29` and evaluates on `2016-03-01`
  through `2017-02-28`;
- continuous profiling computes monthly rule metrics, including
  `antecedent_pairs`, and flags abrupt changes;
- binned validation creates low/medium/high quantile classes for pollutants and
  meteorological variables and compares raw versus binned rules;
- violation analysis exports strongest violating pairs for the top two raw RFDs
  and aggregates them by station, month, and time slot.

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
