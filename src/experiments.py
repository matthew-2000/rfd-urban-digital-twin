"""Experiment runners for profiling and RFD analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.rfd import (
    THRESHOLD_CONFIGS,
    build_pair_index_cache,
    build_similarity_cache,
    discover_rfds,
    rule_to_label,
    validate_rfd,
)


CANDIDATE_RULES = [
    {
        "lhs": ["station", "hour", "TEMP"],
        "rhs": "NO2",
        "interpretation": "Same station, similar hour and temperature should imply similar NO2.",
    },
    {
        "lhs": ["station", "TEMP", "DEWP"],
        "rhs": "PM2.5",
        "interpretation": "Within one station, similar thermal conditions should imply similar PM2.5.",
    },
    {
        "lhs": ["station", "PM2.5"],
        "rhs": "PM10",
        "interpretation": "Within one station, particulate measures should move coherently.",
    },
    {
        "lhs": ["station", "TEMP", "WSPM"],
        "rhs": "O3",
        "interpretation": "Within one station, similar temperature and wind should imply similar O3.",
    },
    {
        "lhs": ["hour", "NO2"],
        "rhs": "PM2.5",
        "interpretation": "Similar hour and NO2 levels should often align with similar PM2.5.",
    },
]

DISCOVERY_LHS_ATTRIBUTES = ["station", "hour", "time_slot", "TEMP", "DEWP", "WSPM", "PM2.5", "NO2"]
DISCOVERY_RHS_ATTRIBUTES = ["PM2.5", "PM10", "NO2", "O3"]
DEFAULT_RFD_SAMPLE_SIZE = 1500
DEFAULT_RANDOM_STATE = 42


def prepare_rfd_sample(
    df: pd.DataFrame,
    sample_size: int = DEFAULT_RFD_SAMPLE_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Return deterministic balanced sample for quadratic pairwise experiments."""

    sampled_input = df.copy()
    sampled_input["source_index"] = sampled_input.index

    if len(sampled_input) <= sample_size:
        return sampled_input.reset_index(drop=True)

    per_station = sample_size // sampled_input["station"].nunique()
    sampled_parts = []
    for _, station_part in sampled_input.groupby("station", sort=True):
        sampled_parts.append(
            station_part.sample(
                n=min(len(station_part), per_station),
                random_state=random_state,
            )
        )
    sampled = pd.concat(sampled_parts, ignore_index=False).sort_values(["datetime", "station"])
    if len(sampled) < sample_size:
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
    }
    if threshold_set is not None:
        row["threshold_set"] = threshold_set
    if station_scope is not None:
        row["station_scope"] = station_scope
    return row


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


def run_lightweight_discovery(
    df: pd.DataFrame,
    output_path: Path,
    thresholds_name: str = "medium",
    top_k: int = 10,
) -> pd.DataFrame:
    """Run lightweight discovery and export top rules."""

    thresholds = THRESHOLD_CONFIGS[thresholds_name]
    pair_cache = build_pair_index_cache(df)
    similarity_cache = build_similarity_cache(
        df,
        thresholds,
        DISCOVERY_LHS_ATTRIBUTES + DISCOVERY_RHS_ATTRIBUTES,
        pair_cache,
    )
    results = discover_rfds(
        df=df,
        lhs_attributes=DISCOVERY_LHS_ATTRIBUTES,
        rhs_attributes=DISCOVERY_RHS_ATTRIBUTES,
        thresholds=thresholds,
        min_support=0.01,
        min_confidence=0.85,
        max_lhs_size=3,
        top_k=top_k,
        pair_cache=pair_cache,
        similarity_cache=similarity_cache,
    )
    columns = [
        "lhs",
        "rhs",
        "rule_label",
        "lhs_length",
        "total_pairs",
        "antecedent_pairs",
        "valid_pairs",
        "violations",
        "support",
        "confidence",
        "violation_rate",
    ]
    output_df = pd.DataFrame([_result_row(result) for result in results], columns=columns)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    return output_df


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
