# RFD Urban Digital Twin Exam

Small Python exam project on Relaxed Functional Dependencies (RFDs) for urban air-quality consistency profiling in a simplified Urban Digital Twin (UDT).

## Dataset

Source: [Beijing Multi-Site Air Quality Data Set](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data)

Project subset:

- stations: `Aotizhongxin`, `Changping`
- period: `2013-03-01` to `2017-02-28`
- final variables: `datetime`, `station`, `hour`, `PM2.5`, `PM10`, `NO2`, `O3`, `TEMP`, `DEWP`, `WSPM`, `time_slot`

Dataset scope:

- the two-station design enables a balanced comparison between sites while
  reducing the quadratic pairwise cost;
- the pipeline remains extensible to additional stations, but this run is
  intentionally scoped to the selected pair;
- the definitive experiment uses the complete common temporal coverage of both files;
- the final cleaned dataset contains `66619` rows;
- full cleaned data are used for preprocessing and profiling, while pairwise RFD
  validation remains tractable through the deterministic balanced sample.

## Structure

```text
data/
  raw/
  processed/
figures/
notebooks/
results/
src/
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Raw Beijing CSV files must be in `data/raw/`.

## Run

Run the complete pipeline and overwrite the executed notebook and all generated outputs:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute notebooks/01_rfd_udt_analysis.ipynb --output 01_rfd_udt_analysis.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=0
```

Main generated dataset:

- `data/processed/udt_rfd_dataset.csv`
- `data/processed/udt_rfd_sample.csv`

RFD experiments use deterministic balanced sample of `1500` rows (`750` per station) because pairwise validation is quadratic.

## RFD metrics

For rule `LHS -> RHS`:

- `support = antecedent_pairs / total_pairs`
- `confidence = valid_pairs / antecedent_pairs`
- `violation_rate = 1 - confidence`
- `baseline_confidence` is the mean confidence over 30 seeded RHS permutations
- `baseline_confidence_std` measures variation across those permutations
- `lift = confidence / baseline_confidence`

RFD confidence measures how often similar tuples on LHS stay similar on RHS. Lift is used to avoid over-interpreting raw confidence: a rule is more informative when it performs clearly above the randomized RHS baseline. These metrics are not prediction accuracy and not causal evidence.

Additional robustness checks:

- 30 balanced bootstrap resamples with mean, standard deviation, and 95% intervals for support, confidence, and lift;
- train/test temporal validation: `2013-03-01`–`2016-02-29` as train and
  `2016-03-01`–`2017-02-28` as test;
- monthly continuous profiling with abrupt-change flags;
- raw versus quantile-binned low/medium/high RFD variants;
- strongest violation-pair export and aggregation by station, month, and time slot.

## Generated outputs

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

## Progress

- [x] preprocessing pipeline and cleaned dataset
- [x] profiling exports
- [x] RFD validation and discovery
- [x] threshold and station experiments
- [x] baseline/lift validation
- [x] bootstrap, temporal validation, and monthly continuous profiling
- [x] raw versus binned RFD comparison
- [x] strongest violation aggregation
- [x] notebook final run

If no discovered rule satisfies `support >= 0.01` and `confidence >= 0.85`, `results/rfd_discovered_top10.csv` is exported with headers only and discussed in notebook as negative result.

## Scope limits

- simplified conceptual UDT only
- no dashboard
- no deep learning
- no streaming backend
- no causal inference
- no full DiMε implementation
