# AGENTS.md

Living spec for `rfd-urban-digital-twin-exam`. If implementation changes, update this file first or same commit.

## Project summary

Small exam project on urban air-quality consistency profiling with Relaxed Functional Dependencies (RFDs).

Focus:

- reproducible Python analysis;
- simplified Urban Digital Twin framing;
- interpretable approximate rules;
- full DiMε discovery as taught in the KDIID course;
- post-discovery validation and robustness analysis;
- exported tables and figures for report/slides.

Do not add:

- dashboards;
- deep learning;
- real-time systems;
- causal claims;
- simplified, bounded, or "DiMε-inspired" discovery;
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
- the definitive experiment uses all available observations for preprocessing,
  profiling, and construction of the DiMε analytical relation;
- the resulting cleaned dataset contains `66619` rows;
- quadratic DiMε discovery remains tractable through a domain-driven weekly,
  station-by-time-slot median relation constructed from all cleaned rows.

Processed dataset output:

- `data/processed/udt_rfd_dataset.csv`
- `data/processed/udt_dime_projection.csv`

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
- `src/algorithms/dime.py`
- `src/experiments.py`
- `src/visualization.py`
- `notebooks/01_rfd_udt_analysis.ipynb`

## Required outputs

Processed data:

- `data/processed/udt_rfd_dataset.csv`
- `data/processed/udt_dime_projection.csv`
- `data/processed/udt_rfd_sample.csv` (legacy supplementary comparison only)

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
- `results/dime_discovered_all.csv`
- `results/dime_discovered_minimal.csv`
- `results/dime_discovered_metrics.csv`
- `results/dime_discovered_top.csv`
- `results/dime_top_violating_pairs.csv`
- `results/dime_discovery_summary.csv`
- `results/rfd_discovered_top10.csv` (compatibility alias of the DiMε top rules)
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

Figure readability:

- the monthly violation chart keeps every monthly bar but labels every sixth
  month to prevent unreadable axis-label overlap in the compiled report.

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

Core validation remains in `src/rfd.py`:

- `is_similar()`
- `validate_rfd()`

Implement the course algorithm in `src/algorithms/dime.py`:

- per-attribute difference matrices using absolute distance for numeric
  attributes and equality distance for categorical attributes;
- similar patterns and stripped similar pattern subsets;
- complete level-wise attribute lattice, without an LHS-size cap;
- TANE/DiMε `C+` candidate generation;
- dependency validation by stripped-subset cardinality/refinement for exact
  comparison RFDs;
- hybrid validation with the `g3` tuple-removal error;
- the official greedy vertex-cover approximation as the default `g3` mode;
- the official exact vertex-cover decision mode, limited only by the
  user-supplied extent threshold, as an optional reproducibility mode;
- DiMε bounds, key pruning, candidate pruning, and minimal-RFD output.

Threshold configs:

- `strict`
- `medium`
- `relaxed`

Default manually selected candidate RFDs are supplementary only:

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

Post-discovery robustness experiments:

- bootstrap validation uses at least 30 balanced resamples;
- temporal validation preserves the original 75/25 chronological design: it
  trains on `2013-03-01` through `2016-02-29` and evaluates on `2016-03-01`
  through `2017-02-28`;
- continuous profiling computes monthly rule metrics, including
  `antecedent_pairs`, and flags abrupt changes;
- binned validation creates low/medium/high quantile classes for pollutants and
  meteorological variables and compares raw versus binned rules;
- violation analysis exports strongest violating pairs for the top discovered RFDs
  and aggregates them by station, month, and time slot.

DiMε analytical relation:

- every cleaned observation contributes to weekly medians grouped by
  `week_start`, `station`, and `time_slot`;
- `week_start` and source-row counts are lineage fields, not discovery
  attributes;
- discovery uses the complete projected schema:
  `station`, `time_slot`, `PM2.5`, `PM10`, `NO2`, `O3`, `TEMP`, `WSPM`;
- this projection is domain-driven: it preserves site and intra-day context,
  retains the pollutants and meteorological covariates used in the study, and
  replaces arbitrary row sampling for the main discovery run;
- the full cleaned data remain the source for descriptive profiling.

DiMε discovery search space:

LHS attributes:

- `station`
- `time_slot`
- `TEMP`
- `WSPM`
- `PM2.5`
- `PM10`
- `NO2`
- `O3`

RHS attributes:

- `station`
- `time_slot`
- `PM2.5`
- `PM10`
- `NO2`
- `O3`
- `TEMP`
- `WSPM`

Default DiMε extent threshold:

- `g3 <= 0.10`

Ranking filters are post-discovery presentation filters, not discovery
conditions. All DiMε outputs, including vacuous key-derived dependencies, must
be exported. Ranked results require positive antecedent support and are ordered
by confidence, support, then permutation lift.

RFD experiment runtime policy:

- keep full cleaned dataset for preprocessing and profiling;
- run full DiMε on the complete domain-driven analytical relation;
- do not cap LHS size or replace the lattice with combination enumeration;
- retain the deterministic balanced sample only for legacy supplementary
  candidate comparisons.

## Interpretation constraints

Always state:

- UDT is conceptual, not 3D platform;
- RFDs are approximate regularities, not causal laws;
- violations may reflect anomalies, data-quality issues, or unobserved events;
- pairwise validation is quadratic, so dataset is intentionally reduced.

## Development order

1. preprocessing and cleaned dataset;
2. profiling and figures;
3. DiMε analytical relation;
4. full DiMε discovery;
5. post-discovery metrics and robustness analysis;
6. supplementary candidate comparisons;
7. violation export;
8. notebook, report, presentation, and README finalization.
