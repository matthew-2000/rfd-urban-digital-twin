"""Experiment runners for profiling and RFD analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.algorithms.dime import discover_dime
from src.rfd import (
    THRESHOLD_CONFIGS,
    build_pair_index_cache,
    build_similarity_cache,
    rule_to_label,
    validate_rfd,
)


CANDIDATE_RULES = [
    {
        "lhs": ["station", "PM2.5"],
        "rhs": "PM10",
        "interpretation": "Within one station, particulate measures should move coherently.",
    },
    {
        "lhs": ["station", "time_slot", "PM2.5"],
        "rhs": "PM10",
        "interpretation": "Within one station and time slot, similar PM2.5 should imply similar PM10.",
    },
    {
        "lhs": ["station", "PM2.5", "NO2"],
        "rhs": "PM10",
        "interpretation": "Within one station, similar PM2.5 and NO2 should imply similar PM10.",
    },
    {
        "lhs": ["station", "hour", "TEMP", "WSPM"],
        "rhs": "O3",
        "interpretation": "Within one station and similar hour, temperature and wind should imply similar O3.",
    },
    {
        "lhs": ["station", "time_slot", "TEMP", "WSPM"],
        "rhs": "O3",
        "interpretation": "Within one station and time slot, similar temperature and wind should imply similar O3.",
    },
    {
        "lhs": ["station", "time_slot", "NO2"],
        "rhs": "O3",
        "interpretation": "Within one station and time slot, similar NO2 should imply similar O3.",
    },
    {
        "lhs": ["station", "TEMP", "DEWP", "WSPM"],
        "rhs": "PM2.5",
        "interpretation": "Within one station, similar meteorological conditions should imply similar PM2.5.",
    },
]

DISCOVERY_LHS_ATTRIBUTES = ["station", "hour", "time_slot", "TEMP", "DEWP", "WSPM", "PM2.5", "NO2"]
DISCOVERY_RHS_ATTRIBUTES = ["PM2.5", "PM10", "NO2", "O3"]
DEFAULT_RFD_SAMPLE_SIZE = 1500
DEFAULT_RANDOM_STATE = 42
DEFAULT_BASELINE_PERMUTATIONS = 30
BOOTSTRAP_BASELINE_PERMUTATIONS = 30
BOOTSTRAP_ITERATIONS = 30
DEFAULT_TRAIN_END = "2016-02-29 23:00:00"
DEFAULT_TEST_START = "2016-03-01 00:00:00"
BINNED_COLUMNS = ["PM2.5", "PM10", "NO2", "O3", "TEMP", "DEWP", "WSPM"]
DIME_ATTRIBUTES = ["station", "time_slot", "PM2.5", "PM10", "NO2", "O3", "TEMP", "WSPM"]
DIME_EXTENT_THRESHOLD = 0.10
DIME_G3_MODE = "greedy"

BINNED_CANDIDATE_RULES = [
    {"lhs": ["station", "PM2.5_bin"], "rhs": "PM10_bin"},
    {"lhs": ["station", "time_slot", "PM2.5_bin"], "rhs": "PM10_bin"},
    {"lhs": ["station", "PM2.5_bin", "NO2_bin"], "rhs": "PM10_bin"},
    {"lhs": ["station", "time_slot", "TEMP_bin", "WSPM_bin"], "rhs": "O3_bin"},
    {"lhs": ["station", "time_slot", "NO2_bin"], "rhs": "O3_bin"},
    {"lhs": ["station", "TEMP_bin", "DEWP_bin", "WSPM_bin"], "rhs": "PM2.5_bin"},
]


def prepare_dime_projection(df: pd.DataFrame) -> pd.DataFrame:
    """Build the domain-driven weekly station/time-slot DiMε relation."""

    projected = df.copy()
    projected["week_start"] = projected["datetime"].dt.to_period("W-MON").dt.start_time
    numeric = ["PM2.5", "PM10", "NO2", "O3", "TEMP", "WSPM"]
    relation = (
        projected.groupby(["week_start", "station", "time_slot"], observed=True, as_index=False)
        .agg(
            **{attribute: (attribute, "median") for attribute in numeric},
            source_rows=("datetime", "size"),
        )
        .sort_values(["week_start", "station", "time_slot"])
        .reset_index(drop=True)
    )
    relation["datetime"] = relation["week_start"]
    relation["source_index"] = relation.index
    return relation[
        ["datetime", "week_start", "station", "time_slot", *numeric, "source_rows", "source_index"]
    ]


def rules_from_discovery_frame(rules_df: pd.DataFrame) -> list[dict[str, object]]:
    """Convert exported discovery rows to validation-rule dictionaries."""

    rules: list[dict[str, object]] = []
    for row in rules_df.itertuples(index=False):
        lhs = [part.strip() for part in str(row.lhs).split(",") if part.strip()]
        rules.append({"lhs": lhs, "rhs": row.rhs})
    return rules


def run_dime_discovery(
    df: pd.DataFrame,
    results_dir: Path,
    thresholds_name: str = "medium",
    extent_threshold: float = DIME_EXTENT_THRESHOLD,
    g3_mode: str = DIME_G3_MODE,
    baseline_permutations: int = DEFAULT_BASELINE_PERMUTATIONS,
    top_k: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run full DiMε, validate every output, and export ranked results."""

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    thresholds = {attribute: THRESHOLD_CONFIGS[thresholds_name][attribute] for attribute in DIME_ATTRIBUTES}
    discovered_df, discoverer = discover_dime(
        df=df,
        thresholds=thresholds,
        extent_threshold=extent_threshold,
        g3_mode=g3_mode,
        attributes=DIME_ATTRIBUTES,
    )
    discovered_df.to_csv(results_dir / "dime_discovered_all.csv", index=False)
    minimal_df = discovered_df[discovered_df["is_minimal"]].copy()
    minimal_df.to_csv(results_dir / "dime_discovered_minimal.csv", index=False)

    rules = rules_from_discovery_frame(minimal_df)
    metrics_df, _ = _validate_rule_set(
        df=df,
        rules=rules,
        thresholds=thresholds,
        baseline_permutations=baseline_permutations,
        random_state=DEFAULT_RANDOM_STATE,
    )
    if metrics_df.empty:
        metrics_df = pd.DataFrame(
            columns=[
                "lhs", "rhs", "rule_label", "lhs_length", "total_pairs",
                "antecedent_pairs", "valid_pairs", "violations", "support",
                "confidence", "violation_rate", "baseline_confidence",
                "baseline_confidence_std", "lift",
            ]
        )
    metrics_df = minimal_df.merge(
        metrics_df,
        on=["lhs", "rhs", "rule_label", "lhs_length"],
        how="left",
    )
    metrics_df.to_csv(results_dir / "dime_discovered_metrics.csv", index=False)

    ranked = (
        metrics_df[metrics_df["antecedent_pairs"].fillna(0).gt(0)]
        .sort_values(
            ["confidence", "support", "lift", "g3_error", "lhs_length"],
            ascending=[False, False, False, True, True],
            na_position="last",
        )
        .head(top_k)
        .reset_index(drop=True)
    )
    ranked.to_csv(results_dir / "dime_discovered_top.csv", index=False)
    ranked.to_csv(results_dir / "rfd_discovered_top10.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "algorithm": "DiMε",
                "g3_mode": g3_mode,
                "extent_threshold": extent_threshold,
                "threshold_set": thresholds_name,
                "projection_rows": len(df),
                "projection_attributes": len(DIME_ATTRIBUTES),
                "lattice_levels_visited": discoverer.levels_visited_,
                "candidates_validated": discoverer.candidates_validated_,
                "discovered_rfds": len(discovered_df),
                "minimal_rfds": len(minimal_df),
                "positive_support_rfds": int(metrics_df["antecedent_pairs"].fillna(0).gt(0).sum()),
            }
        ]
    )
    summary.to_csv(results_dir / "dime_discovery_summary.csv", index=False)
    return minimal_df, metrics_df, ranked


def prepare_rfd_sample(
    df: pd.DataFrame,
    sample_size: int = DEFAULT_RFD_SAMPLE_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    replace: bool = False,
) -> pd.DataFrame:
    """Return deterministic balanced sample for quadratic pairwise experiments."""

    sampled_input = df.copy()
    if "source_index" not in sampled_input.columns:
        sampled_input["source_index"] = sampled_input.index

    if len(sampled_input) <= sample_size and not replace:
        return sampled_input.reset_index(drop=True)

    per_station = sample_size // sampled_input["station"].nunique()
    sampled_parts = []
    for _, station_part in sampled_input.groupby("station", sort=True):
        sampled_parts.append(
            station_part.sample(
                n=min(len(station_part), per_station),
                replace=replace,
                random_state=random_state,
            )
        )
    sampled = pd.concat(sampled_parts, ignore_index=False).sort_values(["datetime", "station"])
    if len(sampled) < sample_size and not replace:
        remaining = sample_size - len(sampled)
        leftovers = sampled_input.drop(sampled.index, errors="ignore")
        if remaining > 0 and not leftovers.empty:
            extra = leftovers.sample(n=min(len(leftovers), remaining), random_state=random_state)
            sampled = pd.concat([sampled, extra], ignore_index=False)
    return sampled.sort_values(["datetime", "station"]).reset_index(drop=True)


def _result_row(result: dict[str, object], threshold_set: str | None = None, station_scope: str | None = None) -> dict[str, object]:
    """Flatten validation result for CSV export."""

    row = {
        "lhs": ", ".join(result["lhs"]),
        "rhs": result["rhs"],
        "rule_label": rule_to_label(result["lhs"], result["rhs"]),
        "lhs_length": len(result["lhs"]),
        "total_pairs": result["total_pairs"],
        "antecedent_pairs": result["antecedent_pairs"],
        "valid_pairs": result["valid_pairs"],
        "violations": result["violations"],
        "support": result["support"],
        "confidence": result["confidence"],
        "violation_rate": result["violation_rate"],
        "baseline_confidence": result.get("baseline_confidence"),
        "baseline_confidence_std": result.get("baseline_confidence_std"),
        "lift": result.get("lift"),
    }
    if threshold_set is not None:
        row["threshold_set"] = threshold_set
    if station_scope is not None:
        row["station_scope"] = station_scope
    return row


def _candidate_attrs(rules: Iterable[dict[str, object]]) -> list[str]:
    """Collect attributes used by a list of candidate rules."""

    attrs: set[str] = set()
    for candidate in rules:
        attrs.update(candidate["lhs"])
        attrs.add(candidate["rhs"])
    return sorted(attrs)


def _validate_rule_set(
    df: pd.DataFrame,
    rules: Iterable[dict[str, object]],
    thresholds: dict[str, object],
    baseline_permutations: int,
    random_state: int,
    max_violations: int = 0,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Validate a list of RFD rules with shared pair and similarity caches."""

    rules = list(rules)
    pair_cache = build_pair_index_cache(df)
    similarity_cache = build_similarity_cache(df, thresholds, _candidate_attrs(rules), pair_cache)

    rows: list[dict[str, object]] = []
    violations: list[dict[str, object]] = []
    for idx, candidate in enumerate(rules):
        result = validate_rfd(
            df=df,
            lhs=candidate["lhs"],
            rhs=candidate["rhs"],
            thresholds=thresholds,
            pair_cache=pair_cache,
            similarity_cache=similarity_cache,
            max_violations=max_violations,
            baseline_permutations=baseline_permutations,
            random_state=random_state + idx,
        )
        row = _result_row(result)
        if "interpretation" in candidate:
            row["interpretation"] = candidate["interpretation"]
        rows.append(row)
        violations.extend(result["violation_examples"])

    return pd.DataFrame(rows), violations


def add_quantile_bins(
    df: pd.DataFrame,
    columns: Iterable[str] = BINNED_COLUMNS,
    labels: tuple[str, str, str] = ("low", "medium", "high"),
) -> pd.DataFrame:
    """Add low/medium/high quantile bins for selected numeric columns."""

    binned = df.copy()
    for column in columns:
        binned[f"{column}_bin"] = pd.qcut(
            binned[column],
            q=3,
            labels=labels,
            duplicates="drop",
        ).astype("object")
    return binned


def binned_thresholds(base_thresholds: dict[str, object]) -> dict[str, object]:
    """Extend a threshold config with exact matching for quantile-bin columns."""

    thresholds = dict(base_thresholds)
    for column in BINNED_COLUMNS:
        thresholds[f"{column}_bin"] = "equal"
    return thresholds


def run_candidate_metrics(
    df: pd.DataFrame,
    output_path: Path,
    figure_path: Path,
    thresholds_name: str = "medium",
    baseline_permutations: int = DEFAULT_BASELINE_PERMUTATIONS,
) -> pd.DataFrame:
    """Validate raw and binned candidate rules with permutation baselines."""

    from src.visualization import plot_lift_vs_baseline

    thresholds = THRESHOLD_CONFIGS[thresholds_name]
    raw_df, _ = _validate_rule_set(
        df=df,
        rules=CANDIDATE_RULES,
        thresholds=thresholds,
        baseline_permutations=baseline_permutations,
        random_state=DEFAULT_RANDOM_STATE,
    )
    raw_df["representation"] = "raw"
    raw_df["threshold_set"] = thresholds_name

    binned_df_input = add_quantile_bins(df)
    binned_df, _ = _validate_rule_set(
        df=binned_df_input,
        rules=BINNED_CANDIDATE_RULES,
        thresholds=binned_thresholds(thresholds),
        baseline_permutations=baseline_permutations,
        random_state=DEFAULT_RANDOM_STATE + 1000,
    )
    binned_df["representation"] = "binned"
    binned_df["threshold_set"] = thresholds_name
    binned_df["interpretation"] = "Quantile-binned low/medium/high variant."

    results_df = (
        pd.concat([raw_df, binned_df], ignore_index=True)
        .sort_values(["representation", "lift", "confidence"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    plot_lift_vs_baseline(results_df, figure_path)
    return results_df


def run_candidate_validation(
    df: pd.DataFrame,
    output_path: Path,
    thresholds_name: str = "medium",
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Validate manual candidate rules under one threshold configuration."""

    thresholds = THRESHOLD_CONFIGS[thresholds_name]
    pair_cache = build_pair_index_cache(df)
    similarity_cache = build_similarity_cache(df, thresholds, DISCOVERY_LHS_ATTRIBUTES + DISCOVERY_RHS_ATTRIBUTES, pair_cache)

    rows: list[dict[str, object]] = []
    violations: list[dict[str, object]] = []
    for candidate in CANDIDATE_RULES:
        result = validate_rfd(
            df=df,
            lhs=candidate["lhs"],
            rhs=candidate["rhs"],
            thresholds=thresholds,
            pair_cache=pair_cache,
            similarity_cache=similarity_cache,
            max_violations=5,
            baseline_permutations=DEFAULT_BASELINE_PERMUTATIONS,
        )
        row = _result_row(result)
        row["interpretation"] = candidate["interpretation"]
        rows.append(row)
        violations.extend(result["violation_examples"])

    results_df = pd.DataFrame(rows).sort_values(["confidence", "support"], ascending=[False, False])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    return results_df, violations


def run_threshold_comparison(df: pd.DataFrame, output_path: Path, figure_path: Path) -> pd.DataFrame:
    """Compare candidate-rule confidence under strict, medium, and relaxed thresholds."""

    from src.visualization import plot_confidence_by_threshold

    rows: list[dict[str, object]] = []
    for threshold_name, thresholds in THRESHOLD_CONFIGS.items():
        pair_cache = build_pair_index_cache(df)
        similarity_cache = build_similarity_cache(
            df,
            thresholds,
            DISCOVERY_LHS_ATTRIBUTES + DISCOVERY_RHS_ATTRIBUTES,
            pair_cache,
        )
        for candidate in CANDIDATE_RULES:
            result = validate_rfd(
                df=df,
                lhs=candidate["lhs"],
                rhs=candidate["rhs"],
                thresholds=thresholds,
                pair_cache=pair_cache,
                similarity_cache=similarity_cache,
                max_violations=0,
                baseline_permutations=0,
            )
            rows.append(_result_row(result, threshold_set=threshold_name))

    results_df = pd.DataFrame(rows).sort_values(["rule_label", "threshold_set"]).reset_index(drop=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    plot_confidence_by_threshold(results_df, figure_path)
    return results_df


def run_station_comparison(
    df: pd.DataFrame,
    output_path: Path,
    figure_path: Path,
    thresholds_name: str = "medium",
) -> pd.DataFrame:
    """Validate candidate rules separately for each station."""

    from src.visualization import plot_confidence_by_station

    thresholds = THRESHOLD_CONFIGS[thresholds_name]
    rows: list[dict[str, object]] = []
    for station_name, station_df in df.groupby("station"):
        station_df = station_df.reset_index(drop=True)
        pair_cache = build_pair_index_cache(station_df)
        similarity_cache = build_similarity_cache(
            station_df,
            thresholds,
            DISCOVERY_LHS_ATTRIBUTES + DISCOVERY_RHS_ATTRIBUTES,
            pair_cache,
        )
        for candidate in CANDIDATE_RULES:
            lhs = [attr for attr in candidate["lhs"] if attr != "station"]
            result = validate_rfd(
                df=station_df,
                lhs=lhs,
                rhs=candidate["rhs"],
                thresholds=thresholds,
                pair_cache=pair_cache,
                similarity_cache=similarity_cache,
                max_violations=0,
                baseline_permutations=0,
            )
            row = _result_row(result, station_scope=station_name)
            row["evaluated_lhs"] = ", ".join(lhs)
            rows.append(row)

    results_df = pd.DataFrame(rows).sort_values(["rule_label", "station_scope"]).reset_index(drop=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    plot_confidence_by_station(results_df, figure_path)
    return results_df


def run_bootstrap_validation(
    df: pd.DataFrame,
    output_path: Path,
    detail_output_path: Path | None = None,
    iterations: int = BOOTSTRAP_ITERATIONS,
    sample_size: int = DEFAULT_RFD_SAMPLE_SIZE,
    thresholds_name: str = "medium",
    rules: Iterable[dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Repeat balanced bootstrap sampling and summarize RFD metric uncertainty."""

    thresholds = THRESHOLD_CONFIGS[thresholds_name]
    evaluated_rules = list(rules) if rules is not None else CANDIDATE_RULES
    rows: list[dict[str, object]] = []
    for iteration in range(iterations):
        sample_df = prepare_rfd_sample(
            df,
            sample_size=sample_size,
            random_state=DEFAULT_RANDOM_STATE + iteration,
            replace=True,
        )
        iteration_df, _ = _validate_rule_set(
            df=sample_df,
            rules=evaluated_rules,
            thresholds=thresholds,
            baseline_permutations=BOOTSTRAP_BASELINE_PERMUTATIONS,
            random_state=DEFAULT_RANDOM_STATE + 10_000 + iteration,
        )
        iteration_df["iteration"] = iteration + 1
        rows.extend(iteration_df.to_dict("records"))

    detail_df = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for rule_label, rule_df in detail_df.groupby("rule_label", sort=True):
        row = {"rule_label": rule_label, "lhs": rule_df["lhs"].iloc[0], "rhs": rule_df["rhs"].iloc[0]}
        for metric in ["support", "confidence", "lift"]:
            values = rule_df[metric].dropna()
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1)
            row[f"{metric}_ci95_low"] = values.quantile(0.025)
            row[f"{metric}_ci95_high"] = values.quantile(0.975)
        row["iterations"] = int(rule_df["iteration"].nunique())
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("lift_mean", ascending=False).reset_index(drop=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    if detail_output_path is not None:
        detail_output_path = Path(detail_output_path)
        detail_output_path.parent.mkdir(parents=True, exist_ok=True)
        detail_df.to_csv(detail_output_path, index=False)
    return summary_df


def run_train_test_validation(
    df: pd.DataFrame,
    output_path: Path,
    train_end: str = DEFAULT_TRAIN_END,
    test_start: str = DEFAULT_TEST_START,
    thresholds_name: str = "medium",
    rules: Iterable[dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Validate supplied rules on chronological 75% train and 25% test windows."""

    thresholds = THRESHOLD_CONFIGS[thresholds_name]
    evaluated_rules = list(rules) if rules is not None else CANDIDATE_RULES
    train_df = df[df["datetime"] <= pd.Timestamp(train_end)].reset_index(drop=True)
    test_df = df[df["datetime"] >= pd.Timestamp(test_start)].reset_index(drop=True)
    train_sample = prepare_rfd_sample(train_df, random_state=DEFAULT_RANDOM_STATE)
    test_sample = prepare_rfd_sample(test_df, random_state=DEFAULT_RANDOM_STATE)

    train_metrics, _ = _validate_rule_set(
        train_sample,
        evaluated_rules,
        thresholds,
        baseline_permutations=DEFAULT_BASELINE_PERMUTATIONS,
        random_state=DEFAULT_RANDOM_STATE,
    )
    test_metrics, _ = _validate_rule_set(
        test_sample,
        evaluated_rules,
        thresholds,
        baseline_permutations=DEFAULT_BASELINE_PERMUTATIONS,
        random_state=DEFAULT_RANDOM_STATE + 2000,
    )

    keep = [
        "rule_label",
        "lhs",
        "rhs",
        "support",
        "confidence",
        "violation_rate",
        "baseline_confidence",
        "baseline_confidence_std",
        "lift",
    ]
    merged = train_metrics.loc[:, keep].merge(
        test_metrics.loc[:, keep],
        on=["rule_label", "lhs", "rhs"],
        suffixes=("_train", "_test"),
    )
    merged["confidence_delta_test_minus_train"] = merged["confidence_test"] - merged["confidence_train"]
    merged["lift_delta_test_minus_train"] = merged["lift_test"] - merged["lift_train"]
    merged = merged.sort_values(["lift_train", "confidence_train"], ascending=False).reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged


def run_window_evolution(
    df: pd.DataFrame,
    candidate_metrics_df: pd.DataFrame,
    output_path: Path,
    confidence_figure_path: Path,
    thresholds_name: str = "medium",
    window: str = "month",
    top_rules: int = 4,
    rules: Iterable[dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Compute RFD metrics over monthly or weekly temporal windows."""

    from src.visualization import plot_confidence_over_time

    thresholds = THRESHOLD_CONFIGS[thresholds_name]
    if "representation" in candidate_metrics_df.columns:
        ranked_metrics = candidate_metrics_df[candidate_metrics_df["representation"] == "raw"].copy()
    else:
        ranked_metrics = candidate_metrics_df.copy()
    top_labels = ranked_metrics.sort_values(["lift", "confidence"], ascending=False).head(top_rules)["rule_label"].tolist()
    available_rules = list(rules) if rules is not None else CANDIDATE_RULES
    selected_rules = [
        rule for rule in available_rules
        if rule_to_label(rule["lhs"], rule["rhs"]) in top_labels
    ]

    work_df = df.copy()
    freq = "W-MON" if window == "week" else "M"
    work_df["window_start"] = work_df["datetime"].dt.to_period(freq).dt.start_time
    rows: list[dict[str, object]] = []
    for window_start, window_df in work_df.groupby("window_start", sort=True):
        window_sample = prepare_rfd_sample(window_df.reset_index(drop=True), random_state=DEFAULT_RANDOM_STATE)
        if len(window_sample) < 2:
            continue
        metrics_df, _ = _validate_rule_set(
            window_sample,
            selected_rules,
            thresholds,
            baseline_permutations=DEFAULT_BASELINE_PERMUTATIONS,
            random_state=DEFAULT_RANDOM_STATE + int(pd.Timestamp(window_start).strftime("%m%d")),
        )
        metrics_df["window_start"] = window_start
        metrics_df["window"] = window
        metrics_df["rows"] = len(window_sample)
        rows.extend(metrics_df.to_dict("records"))

    evolution_df = pd.DataFrame(rows).sort_values(["rule_label", "window_start"]).reset_index(drop=True)
    if not evolution_df.empty:
        evolution_df["confidence_delta"] = evolution_df.groupby("rule_label")["confidence"].diff()
        evolution_df["lift_delta"] = evolution_df.groupby("rule_label")["lift"].diff()
        evolution_df["abrupt_change"] = (
            evolution_df["confidence_delta"].abs().ge(0.15) | evolution_df["lift_delta"].abs().ge(0.50)
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evolution_df.to_csv(output_path, index=False)
    plot_confidence_over_time(evolution_df, confidence_figure_path)
    return evolution_df


def _top_violating_pairs(
    df: pd.DataFrame,
    rule: dict[str, object],
    thresholds: dict[str, object],
    max_pairs: int,
) -> pd.DataFrame:
    """Return strongest RHS-difference violations for one rule."""

    pair_cache = build_pair_index_cache(df)
    attrs = list(rule["lhs"]) + [rule["rhs"]]
    similarity_cache = build_similarity_cache(df, thresholds, attrs, pair_cache)
    lhs_masks = [similarity_cache[attr] for attr in rule["lhs"]]
    antecedent_mask = np.logical_and.reduce(lhs_masks)
    violation_positions = np.flatnonzero(antecedent_mask & ~similarity_cache[rule["rhs"]])
    if len(violation_positions) == 0:
        return pd.DataFrame()

    rhs_values = df[rule["rhs"]].to_numpy(dtype=float)
    rhs_diff = np.abs(rhs_values[pair_cache.row_i[violation_positions]] - rhs_values[pair_cache.row_j[violation_positions]])
    order = np.argsort(-rhs_diff)[:max_pairs]
    rows: list[dict[str, object]] = []
    for position, diff in zip(violation_positions[order], rhs_diff[order]):
        idx1 = int(pair_cache.row_i[position])
        idx2 = int(pair_cache.row_j[position])
        row1 = df.iloc[idx1]
        row2 = df.iloc[idx2]
        row = {
            "rule_label": rule_to_label(rule["lhs"], rule["rhs"]),
            "lhs": ", ".join(rule["lhs"]),
            "rhs": rule["rhs"],
            "row_index_1": int(row1.get("source_index", idx1)),
            "row_index_2": int(row2.get("source_index", idx2)),
            "datetime_1": row1["datetime"],
            "datetime_2": row2["datetime"],
            "station_1": row1["station"],
            "station_2": row2["station"],
            "month_1": pd.Timestamp(row1["datetime"]).strftime("%Y-%m"),
            "month_2": pd.Timestamp(row2["datetime"]).strftime("%Y-%m"),
            "time_slot_1": row1["time_slot"],
            "time_slot_2": row2["time_slot"],
            "rhs_value_1": row1[rule["rhs"]],
            "rhs_value_2": row2[rule["rhs"]],
            "rhs_abs_diff": float(diff),
        }
        for attr in rule["lhs"]:
            row[f"{attr}_1"] = row1[attr]
            row[f"{attr}_2"] = row2[attr]
        rows.append(row)
    return pd.DataFrame(rows)


def run_violation_analysis(
    df: pd.DataFrame,
    candidate_metrics_df: pd.DataFrame,
    summary_output_path: Path,
    pairs_output_path: Path,
    figure_path: Path,
    thresholds_name: str = "medium",
    top_rules: int = 2,
    max_pairs_per_rule: int = 200,
    rules: Iterable[dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Export strongest violating pairs and aggregate their concentration."""

    from src.visualization import plot_violations_by_month_station

    if "representation" in candidate_metrics_df.columns:
        ranked_metrics = candidate_metrics_df[candidate_metrics_df["representation"] == "raw"].copy()
    else:
        ranked_metrics = candidate_metrics_df.copy()
    top_labels = ranked_metrics.sort_values(["lift", "confidence"], ascending=False).head(top_rules)["rule_label"].tolist()
    available_rules = list(rules) if rules is not None else CANDIDATE_RULES
    selected_rules = [
        rule for rule in available_rules
        if rule_to_label(rule["lhs"], rule["rhs"]) in top_labels
    ]
    thresholds = THRESHOLD_CONFIGS[thresholds_name]

    pairs = [
        _top_violating_pairs(df, rule, thresholds, max_pairs=max_pairs_per_rule)
        for rule in selected_rules
    ]
    pairs_df = pd.concat([item for item in pairs if not item.empty], ignore_index=True) if pairs else pd.DataFrame()

    pairs_output_path = Path(pairs_output_path)
    pairs_output_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_df.to_csv(pairs_output_path, index=False)

    if pairs_df.empty:
        summary_df = pd.DataFrame()
    else:
        summary_df = (
            pairs_df.groupby(["rule_label", "station_1", "month_1", "time_slot_1"], as_index=False)
            .agg(
                violation_pairs=("rhs_abs_diff", "size"),
                mean_rhs_abs_diff=("rhs_abs_diff", "mean"),
                max_rhs_abs_diff=("rhs_abs_diff", "max"),
            )
            .rename(columns={"station_1": "station", "month_1": "month", "time_slot_1": "time_slot"})
        )
        summary_df["share_within_rule"] = summary_df["violation_pairs"] / summary_df.groupby("rule_label")[
            "violation_pairs"
        ].transform("sum")
        summary_df = summary_df.sort_values(
            ["rule_label", "violation_pairs", "mean_rhs_abs_diff"],
            ascending=[True, False, False],
        ).reset_index(drop=True)

    summary_output_path = Path(summary_output_path)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_output_path, index=False)
    plot_violations_by_month_station(summary_df, figure_path)
    return summary_df


def export_violation_examples(
    violations: Iterable[dict[str, object]],
    output_path: Path,
    limit: int = 10,
) -> pd.DataFrame:
    """Export flattened violation examples."""

    violations_df = pd.DataFrame(list(violations)[:limit])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    violations_df.to_csv(output_path, index=False)
    return violations_df


def select_top_rule_violations(
    candidate_results_df: pd.DataFrame,
    violations: Iterable[dict[str, object]],
    top_rules: int = 2,
    per_rule: int = 5,
) -> list[dict[str, object]]:
    """Keep violations from highest-confidence candidate rules."""

    top_labels = candidate_results_df.head(top_rules)["rule_label"].tolist()
    selected: list[dict[str, object]] = []
    for label in top_labels:
        label_rows = [row for row in violations if row["rule_label"] == label][:per_rule]
        selected.extend(label_rows)
    return selected
