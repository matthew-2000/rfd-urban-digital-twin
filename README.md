# RFD Urban Digital Twin Exam

Small Python exam project on Relaxed Functional Dependencies (RFDs) for urban air-quality consistency profiling in a simplified Urban Digital Twin (UDT).

## Dataset

Source: [Beijing Multi-Site Air Quality Data Set](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data)

Project subset:

- stations: `Aotizhongxin`, `Changping`
- period: `2013-09-01` to `2013-12-31`
- final variables: `datetime`, `station`, `hour`, `PM2.5`, `PM10`, `NO2`, `O3`, `TEMP`, `DEWP`, `WSPM`, `time_slot`

Why December included:

- requested Sep-Nov subset cleaned to `3996` rows after dropna;
- adding December yields `5426` cleaned rows;
- this keeps pairwise RFD validation feasible while staying near target size.

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

Run notebook:

```bash
jupyter notebook notebooks/01_rfd_udt_analysis.ipynb
```

Main generated dataset:

- `data/processed/udt_rfd_dataset.csv`

## RFD metrics

For rule `LHS -> RHS`:

- `support = antecedent_pairs / total_pairs`
- `confidence = valid_pairs / antecedent_pairs`
- `violation_rate = 1 - confidence`

RFD confidence measures how often similar tuples on LHS stay similar on RHS. It is not prediction accuracy and not causal evidence.

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

## Progress

- [x] preprocessing pipeline and cleaned dataset
- [x] profiling exports
- [ ] RFD validation and discovery
- [ ] threshold and station experiments
- [ ] notebook final run

## Scope limits

- simplified conceptual UDT only
- no dashboard
- no deep learning
- no streaming backend
- no causal inference
- no full DiMε implementation
