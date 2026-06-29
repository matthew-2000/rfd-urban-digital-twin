#!/usr/bin/env python3
"""Run an isolated DiMε sensitivity analysis over similarity thresholds."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.dime import discover_dime  # noqa: E402
from src.experiments import (  # noqa: E402
    DEFAULT_BASELINE_PERMUTATIONS,
    DEFAULT_RANDOM_STATE,
    DIME_ATTRIBUTES,
    DIME_EXTENT_THRESHOLD,
    DIME_G3_MODE,
    _validate_rule_set,
    prepare_dime_projection,
    rules_from_discovery_frame,
)


CONFIGS: dict[str, dict[str, float | str]] = {
    "strict_075x": {
        "station": "equal",
        "time_slot": "equal",
        "PM2.5": 7.5,
        "PM10": 11.25,
        "NO2": 7.5,
        "O3": 7.5,
        "TEMP": 1.5,
        "WSPM": 0.75,
    },
    "medium": {
        "station": "equal",
        "time_slot": "equal",
        "PM2.5": 10.0,
        "PM10": 15.0,
        "NO2": 10.0,
        "O3": 10.0,
        "TEMP": 2.0,
        "WSPM": 1.0,
    },
    "loose_150x": {
        "station": "equal",
        "time_slot": "equal",
        "PM2.5": 15.0,
        "PM10": 22.5,
        "NO2": 15.0,
        "O3": 15.0,
        "TEMP": 3.0,
        "WSPM": 1.5,
    },
    "wspm_strict_05": {
        "station": "equal",
        "time_slot": "equal",
        "PM2.5": 10.0,
        "PM10": 15.0,
        "NO2": 10.0,
        "O3": 10.0,
        "TEMP": 2.0,
        "WSPM": 0.5,
    },
    "wspm_loose_15": {
        "station": "equal",
        "time_slot": "equal",
        "PM2.5": 10.0,
        "PM10": 15.0,
        "NO2": 10.0,
        "O3": 10.0,
        "TEMP": 2.0,
        "WSPM": 1.5,
    },
    "wspm_loose_20": {
        "station": "equal",
        "time_slot": "equal",
        "PM2.5": 10.0,
        "PM10": 15.0,
        "NO2": 10.0,
        "O3": 10.0,
        "TEMP": 2.0,
        "WSPM": 2.0,
    },
}

SUMMARY_COLUMNS = [
    "config_name",
    "threshold_PM25",
    "threshold_PM10",
    "threshold_NO2",
    "threshold_O3",
    "threshold_TEMP",
    "threshold_WSPM",
    "num_rfds",
    "num_distinct_rhs",
    "rhs_list",
    "num_rfds_rhs_WSPM",
    "share_rfds_rhs_WSPM",
    "best_rule_by_confidence",
    "best_rule_rhs",
    "best_rule_g3",
    "best_rule_support",
    "best_rule_confidence",
    "best_rule_lift",
    "min_g3",
    "max_confidence",
    "max_support",
]

FULL_REQUIRED_COLUMNS = [
    "lhs",
    "rhs",
    "g3",
    "support",
    "confidence",
    "lift",
    "antecedent_pairs",
    "valid_pairs",
    "violation_rate",
]


def _best_rule(metrics: pd.DataFrame) -> pd.Series | None:
    ranked = metrics[metrics["antecedent_pairs"].fillna(0).gt(0)].sort_values(
        ["confidence", "support", "lift", "g3_error", "lhs_length"],
        ascending=[False, False, False, True, True],
        na_position="last",
        kind="stable",
    )
    return None if ranked.empty else ranked.iloc[0]


def _summary_row(
    config_name: str,
    thresholds: dict[str, float | str],
    metrics: pd.DataFrame,
) -> dict[str, Any]:
    rhs_values = sorted(metrics["rhs"].dropna().astype(str).unique())
    wspm_count = int(metrics["rhs"].eq("WSPM").sum())
    best = _best_rule(metrics)

    row: dict[str, Any] = {
        "config_name": config_name,
        "threshold_PM25": thresholds["PM2.5"],
        "threshold_PM10": thresholds["PM10"],
        "threshold_NO2": thresholds["NO2"],
        "threshold_O3": thresholds["O3"],
        "threshold_TEMP": thresholds["TEMP"],
        "threshold_WSPM": thresholds["WSPM"],
        "num_rfds": len(metrics),
        "num_distinct_rhs": len(rhs_values),
        "rhs_list": ", ".join(rhs_values),
        "num_rfds_rhs_WSPM": wspm_count,
        "share_rfds_rhs_WSPM": wspm_count / len(metrics) if len(metrics) else 0.0,
        "best_rule_by_confidence": None,
        "best_rule_rhs": None,
        "best_rule_g3": None,
        "best_rule_support": None,
        "best_rule_confidence": None,
        "best_rule_lift": None,
        "min_g3": metrics["g3_error"].min() if len(metrics) else None,
        "max_confidence": metrics["confidence"].max() if len(metrics) else None,
        "max_support": metrics["support"].max() if len(metrics) else None,
    }
    if best is not None:
        best_label = best["rule_label"]
        if not str(best["lhs"]).strip():
            best_label = f"∅ -> {best['rhs']}"
        row.update(
            {
                "best_rule_by_confidence": best_label,
                "best_rule_rhs": best["rhs"],
                "best_rule_g3": best["g3_error"],
                "best_rule_support": best["support"],
                "best_rule_confidence": best["confidence"],
                "best_rule_lift": best["lift"],
            }
        )
    return row


def _format_number(value: Any, digits: int = 3) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _markdown_table(summary: pd.DataFrame) -> str:
    columns = [
        "config_name",
        "num_rfds",
        "rhs_list",
        "share_rfds_rhs_WSPM",
        "best_rule_by_confidence",
        "best_rule_g3",
        "best_rule_support",
        "best_rule_confidence",
        "best_rule_lift",
    ]
    labels = [
        "Configuration",
        "RFDs",
        "RHS",
        "WSPM share",
        "Best rule",
        "g3",
        "Support",
        "Confidence",
        "Lift",
    ]
    lines = [
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join(["---"] * len(labels)) + " |",
    ]
    numeric = {
        "share_rfds_rhs_WSPM",
        "best_rule_g3",
        "best_rule_support",
        "best_rule_confidence",
        "best_rule_lift",
    }
    for row in summary[columns].itertuples(index=False, name=None):
        values = []
        for column, value in zip(columns, row):
            text = _format_number(value) if column in numeric else str(value)
            values.append(text.replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _range_text(summary: pd.DataFrame, column: str) -> str:
    values = summary[column].dropna()
    if values.empty:
        return "n/a"
    return f"{values.min():.3f}–{values.max():.3f}"


def _interpretation_lines(summary: pd.DataFrame) -> list[str]:
    global_runs = summary[summary["config_name"].isin(["strict_075x", "medium", "loose_150x"])]
    wspm_runs = summary[summary["config_name"].str.startswith("wspm_")]
    alternative_rhs = sorted(
        {
            rhs.strip()
            for rhs_list in summary["rhs_list"]
            for rhs in str(rhs_list).split(",")
            if rhs.strip() and rhs.strip() != "WSPM"
        }
    )
    strict = global_runs.loc[global_runs["config_name"].eq("strict_075x")].iloc[0]
    loose = global_runs.loc[global_runs["config_name"].eq("loose_150x")].iloc[0]
    wspm_stable = wspm_runs["rhs_list"].eq("WSPM").all()
    wspm_count_stable = wspm_runs["num_rfds"].nunique() == 1

    medium = summary.loc[summary["config_name"].eq("medium")].iloc[0]
    confidence_delta = (summary["best_rule_confidence"] - medium["best_rule_confidence"]).abs().max()
    lift_delta = (summary["best_rule_lift"] - medium["best_rule_lift"]).abs().max()
    similar_metrics = confidence_delta <= 0.05 and lift_delta <= 0.10

    return [
        (
            "Global thresholds: the stricter run is not stable because it introduces "
            f"{strict['num_distinct_rhs']} RHS; the looser run retains only WSPM but "
            f"collapses to {loose['num_rfds']} RFD(s)."
        ),
        (
            "WSPM threshold: changing only WSPM leaves RHS identity stable, but "
            + (
                "also leaves the RFD count stable."
                if wspm_count_stable
                else "changes the discovered RFD count and therefore the rule set."
            )
            if wspm_stable
            else "WSPM threshold: changing only WSPM changes the discovered RHS composition."
        ),
        (
            "Alternative RHS: none."
            if not alternative_rhs
            else f"Alternative RHS: {', '.join(alternative_rhs)}."
        ),
        (
            f"Best-rule confidence ranges {_range_text(summary, 'best_rule_confidence')} and lift "
            f"ranges {_range_text(summary, 'best_rule_lift')}; maximum absolute differences from "
            f"medium are {confidence_delta:.3f} and {lift_delta:.3f}, respectively. "
            + (
                "They are similar under the stated 0.05 confidence / 0.10 lift tolerance."
                if similar_metrics
                else "They are not similar under the stated 0.05 confidence / 0.10 lift tolerance."
            )
        ),
    ]


def _write_markdown(
    summary: pd.DataFrame,
    output_path: Path,
    baseline_permutations: int,
) -> None:
    rhs_lines = [
        f"- `{row.config_name}`: {row.rhs_list or 'none'}"
        for row in summary.itertuples(index=False)
    ]
    dominance_lines = [
        (
            f"- `{row.config_name}`: WSPM is the only RHS "
            f"({row.num_rfds_rhs_WSPM}/{row.num_rfds} RFDs)."
            if row.rhs_list == "WSPM"
            else f"- `{row.config_name}`: other RHS emerge ({row.rhs_list or 'none'})."
        )
        for row in summary.itertuples(index=False)
    ]
    interpretation = _interpretation_lines(summary)
    text = "\n".join(
        [
            "# DiMε similarity-threshold sensitivity analysis",
            "",
            "All runs use the existing analytical relation, the complete eight-attribute "
            f"DiMε lattice, greedy `g3`, `g3 <= 0.10`, and {baseline_permutations} "
            "seeded RHS permutations.",
            "",
            "## Summary",
            "",
            _markdown_table(summary),
            "",
            "## RHS analysis",
            "",
            *rhs_lines,
            "",
            "## WSPM dominance",
            "",
            *dominance_lines,
            "",
            "## Interpretation",
            "",
            *[f"- {line}" for line in interpretation],
            "",
            "The UDT framing is conceptual. RFDs are approximate regularities, not causal "
            "laws; violations may reflect anomalies, data-quality issues, or unobserved "
            "events. Pairwise validation is quadratic, so the established analytical "
            "relation is intentionally reduced.",
            "",
        ]
    )
    output_path.write_text(text, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cleaned-input",
        type=Path,
        default=PROJECT_ROOT / "data/processed/udt_rfd_dataset.csv",
        help=(
            "Cleaned observation dataset used to reconstruct the analytical relation "
            "with the same function used by the main pipeline."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/sensitivity",
    )
    parser.add_argument(
        "--baseline-permutations",
        type=int,
        default=DEFAULT_BASELINE_PERMUTATIONS,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cleaned = pd.read_csv(args.cleaned_input, parse_dates=["datetime"])
    relation = prepare_dime_projection(cleaned)
    missing = [attribute for attribute in DIME_ATTRIBUTES if attribute not in relation.columns]
    if missing:
        raise ValueError(f"Analytical relation is missing attributes: {missing}")
    if relation[DIME_ATTRIBUTES].isna().any().any():
        raise ValueError("Analytical relation contains missing discovery values")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []

    for config_name, thresholds in CONFIGS.items():
        print(f"Running {config_name} ...", flush=True)
        discovered, discoverer = discover_dime(
            df=relation,
            thresholds=thresholds,
            extent_threshold=DIME_EXTENT_THRESHOLD,
            g3_mode=DIME_G3_MODE,
            attributes=DIME_ATTRIBUTES,
        )
        minimal = discovered[discovered["is_minimal"]].copy()
        metrics, _ = _validate_rule_set(
            df=relation,
            rules=rules_from_discovery_frame(minimal),
            thresholds=thresholds,
            baseline_permutations=args.baseline_permutations,
            random_state=DEFAULT_RANDOM_STATE,
        )
        merged = minimal.merge(
            metrics,
            on=["lhs", "rhs", "rule_label", "lhs_length"],
            how="left",
        )
        merged.insert(0, "config_name", config_name)
        merged.insert(1, "g3", merged.pop("g3_error"))
        output_columns = FULL_REQUIRED_COLUMNS + [
            column
            for column in merged.columns
            if column not in FULL_REQUIRED_COLUMNS
        ]
        merged[output_columns].to_csv(
            args.output_dir / f"rfds_{config_name}.csv",
            index=False,
        )

        metrics_with_g3 = merged.rename(columns={"g3": "g3_error"})
        summary_rows.append(_summary_row(config_name, thresholds, metrics_with_g3))
        print(
            f"Completed {config_name}: {len(merged)} RFDs; "
            f"RHS={', '.join(sorted(merged['rhs'].unique())) or 'none'}; "
            f"levels={discoverer.levels_visited_}; "
            f"candidates={discoverer.candidates_validated_}",
            flush=True,
        )
        del discovered, discoverer, minimal, metrics, merged, metrics_with_g3
        gc.collect()

    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    csv_path = args.output_dir / "sensitivity_summary.csv"
    markdown_path = args.output_dir / "sensitivity_summary.md"
    summary.to_csv(csv_path, index=False)
    _write_markdown(summary, markdown_path, args.baseline_permutations)

    print(f"\nSummary CSV: {csv_path.resolve()}")
    print(f"Markdown: {markdown_path.resolve()}")
    print("\nMain summary:")
    terminal_columns = [
        "config_name",
        "num_rfds",
        "rhs_list",
        "share_rfds_rhs_WSPM",
        "best_rule_by_confidence",
        "best_rule_confidence",
        "best_rule_lift",
    ]
    print(summary[terminal_columns].to_string(index=False))
    print("\nConclusion (5 lines):")
    for index, line in enumerate(_interpretation_lines(summary), start=1):
        print(f"{index}. {line}")
    all_wspm_only = summary["rhs_list"].eq("WSPM").all()
    print(
        "5. Overall: "
        + (
            "WSPM dominance is qualitatively stable across all tested thresholds."
            if all_wspm_only
            else "WSPM dominance is not qualitatively stable across all tested thresholds."
        )
    )


if __name__ == "__main__":
    main()
