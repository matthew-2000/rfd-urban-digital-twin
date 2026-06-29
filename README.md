# RFD Urban Digital Twin Exam

Small Python exam project on Relaxed Functional Dependencies (RFDs) for
urban air-quality consistency profiling in a simplified Urban Digital Twin
(UDT). The main discovery method is the full DiMε algorithm taught in the
KDIID course and described by Caruccio, Deufemia, and Polese (2020).

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
- full cleaned data are used for preprocessing and profiling;
- every cleaned row contributes to the DiMε analytical relation, which contains
  weekly medians by station and time slot;
- the legacy deterministic sample is used only for supplementary manual-rule
  comparisons.

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
- `data/processed/udt_dime_projection.csv`
- `data/processed/udt_rfd_sample.csv` (supplementary comparison only)

## Course algorithm: DiMε

The implementation in `src/algorithms/dime.py` follows the course lecture,
the 2020 paper, and the authors' reference implementation:

- absolute-distance matrices for numeric attributes and equality distance for
  categorical attributes;
- similar patterns and stripped similar pattern subsets;
- complete level-wise attribute lattice with no LHS-size cap;
- TANE-style `C+` candidate generation;
- exact refinement/cardinality validation when `g3 = 0`;
- hybrid validation with the `g3` tuple-removal error;
- official greedy vertex-cover approximation by default, with the reference
  exact vertex-cover decision mode available for reproducibility;
- DiMε bounds, key pruning, candidate pruning, and minimal-RFD output.

Discovery uses all eight projected attributes:
`station`, `time_slot`, `PM2.5`, `PM10`, `NO2`, `O3`, `TEMP`, and `WSPM`.
The default extent threshold is `g3 <= 0.10`; similarity thresholds are the
domain-specific `medium` configuration in `src/rfd.py`.

The current reproducible run contains `1676` projected rows, visits all eight
lattice levels, validates `960` candidates, and discovers `5` minimal RFDs.
The main run remains the domain-specific `medium` threshold configuration.

## Threshold sensitivity

Similarity thresholds are part of the DiMε input, so the project keeps the
`medium` run as the primary result and uses a separate sensitivity analysis as
a robustness check rather than as an alternative main experiment.

The isolated sensitivity study in `outputs/sensitivity/` compares:

- three global settings: `strict_075x`, `medium`, `loose_150x`;
- three WSPM-only variants: `wspm_strict_05`, `wspm_loose_15`,
  `wspm_loose_20`;
- the same greedy validation with `g3 <= 0.10` used by the main run.

Interpretation is intentionally conservative:

- `medium` yields a compact set of `5` minimal RFDs, all with RHS `WSPM`;
- `strict_075x` expands the result to `19` RFDs and `6` distinct RHS, showing
  that WSPM dominance is threshold-dependent rather than absolute;
- looser settings collapse to trivial rules such as `∅ -> WSPM`, which have
  support `1` and lift `1` and are therefore not informative.

For this reason, the report and slides present sensitivity analysis as a
methodological robustness check that qualifies interpretation without changing
the main `medium` results.

The implementation was also checked on the `iris.csv` shipped with the
authors' official DiMε distribution (`maxthr=1`, `maxcovthr=1`, greedy mode).
The official JAR and this Python implementation return the same five
dependencies, with no missing or extra rule. The comparison is exported in
`results/dime_iris_validation.csv`; the JAR's verbatim rule list is retained in
`results/dime_iris_official_output.txt`.

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

- 30 balanced bootstrap resamples of the top DiMε RFDs with mean, standard
  deviation, and 95% intervals for support, confidence, and lift;
- train/test temporal validation: `2013-03-01`–`2016-02-29` as train and
  `2016-03-01`–`2017-02-28` as test;
- monthly continuous profiling with abrupt-change flags;
- raw versus quantile-binned low/medium/high variants as a supplementary
  manual-rule comparison;
- strongest violation-pair export and aggregation by station, month, and time slot.
- threshold sensitivity kept separate from the main pipeline outputs and used
  only to qualify the interpretation of the `medium` run.

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
- `results/dime_discovered_all.csv`
- `results/dime_discovered_minimal.csv`
- `results/dime_discovered_metrics.csv`
- `results/dime_discovered_top.csv`
- `results/dime_discovery_summary.csv`
- `results/dime_top_violating_pairs.csv`
- `results/rfd_discovered_top10.csv` (compatibility export)
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
- [x] full DiMε discovery and minimality pruning
- [x] threshold and station supplementary experiments
- [x] baseline/lift validation
- [x] bootstrap, temporal validation, and monthly continuous profiling
- [x] raw versus binned RFD comparison
- [x] strongest violation aggregation
- [x] notebook final run

## Scope limits

- simplified conceptual UDT only
- no dashboard
- no deep learning
- no streaming backend
- no causal inference

The UDT is conceptual rather than a 3D operational platform. RFDs are
approximate regularities, not causal laws. Violations may represent anomalies,
data-quality issues, or unobserved events. Pairwise discovery remains
quadratic, which motivates the documented analytical relation.
