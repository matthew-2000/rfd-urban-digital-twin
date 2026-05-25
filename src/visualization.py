"""Plotting helpers for the UDT RFD dataset and experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns


matplotlib.use("Agg")
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", context="talk")


def _prepare_output(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_missing_values(missing_summary: pd.DataFrame, output_path: Path) -> None:
    """Plot missing counts per column."""

    output_path = _prepare_output(output_path)
    plot_df = missing_summary.sort_values("missing_count", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=plot_df, x="missing_count", y="column", color="#2a9d8f", ax=ax)
    ax.set_title("Missing Values by Column")
    ax.set_xlabel("Missing count")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pollutant_distribution(df: pd.DataFrame, column: str, output_path: Path) -> None:
    """Plot histogram with KDE for pollutant distribution."""

    output_path = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x=column, bins=35, kde=True, color="#1f77b4", ax=ax)
    ax.set_title(f"{column} Distribution")
    ax.set_xlabel(column)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pm25_timeseries(df: pd.DataFrame, output_path: Path) -> None:
    """Plot daily average PM2.5 by station for readability."""

    output_path = _prepare_output(output_path)
    plot_df = df.copy()
    plot_df["date"] = plot_df["datetime"].dt.floor("D")
    daily = (
        plot_df.groupby(["date", "station"], as_index=False)["PM2.5"]
        .mean()
        .sort_values(["date", "station"])
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=daily, x="date", y="PM2.5", hue="station", marker="o", ax=ax)
    ax.set_title("Daily Mean PM2.5 by Station")
    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_matrix(corr_df: pd.DataFrame, output_path: Path) -> None:
    """Plot correlation matrix heatmap."""

    output_path = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, cmap="RdBu_r", center=0, fmt=".2f", ax=ax)
    ax.set_title("Numeric Correlation Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confidence_by_threshold(results_df: pd.DataFrame, output_path: Path) -> None:
    """Plot candidate-rule confidence under strict, medium, and relaxed thresholds."""

    output_path = _prepare_output(output_path)
    plot_df = results_df.copy()
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=plot_df, x="rule_label", y="confidence", hue="threshold_set", ax=ax)
    ax.set_title("RFD Confidence by Threshold Set")
    ax.set_xlabel("Rule")
    ax.set_ylabel("Confidence")
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confidence_by_station(results_df: pd.DataFrame, output_path: Path) -> None:
    """Plot per-station confidence for selected candidate rules."""

    output_path = _prepare_output(output_path)
    plot_df = results_df.copy()
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=plot_df, x="rule_label", y="confidence", hue="station_scope", ax=ax)
    ax.set_title("RFD Confidence by Station")
    ax.set_xlabel("Rule")
    ax.set_ylabel("Confidence")
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
