"""Profiling helpers for the UDT RFD dataset."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def dataset_shape_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return row and column counts."""

    return pd.DataFrame(
        {
            "metric": ["rows", "columns"],
            "value": [len(df), df.shape[1]],
        }
    )


def data_types_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return data types for each column."""

    return (
        df.dtypes.astype(str)
        .rename("dtype")
        .reset_index()
        .rename(columns={"index": "column"})
        .sort_values("column")
        .reset_index(drop=True)
    )


def missing_values_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing counts and rates for all columns."""

    summary = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": [int(df[column].isna().sum()) for column in df.columns],
        }
    )
    summary["missing_rate"] = summary["missing_count"] / len(df)
    return summary.sort_values(["missing_count", "column"], ascending=[False, True]).reset_index(
        drop=True
    )


def unique_values_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return unique counts for each column."""

    return pd.DataFrame(
        {
            "column": df.columns,
            "unique_values": [int(df[column].nunique(dropna=True)) for column in df.columns],
        }
    ).sort_values(["unique_values", "column"], ascending=[False, True]).reset_index(drop=True)


def numeric_summary(df: pd.DataFrame, numeric_columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Return min, max, mean, and standard deviation for numeric columns."""

    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include="number").columns.tolist()

    summary = (
        df.loc[:, list(numeric_columns)]
        .agg(["min", "max", "mean", "std"])
        .transpose()
        .reset_index()
        .rename(columns={"index": "column"})
    )
    return summary.sort_values("column").reset_index(drop=True)


def station_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return row counts by station."""

    return (
        df["station"]
        .value_counts(dropna=False)
        .rename_axis("station")
        .reset_index(name="rows")
        .sort_values("station")
        .reset_index(drop=True)
    )


def hour_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return row counts by hour."""

    return (
        df["hour"]
        .value_counts(dropna=False)
        .rename_axis("hour")
        .reset_index(name="rows")
        .sort_values("hour")
        .reset_index(drop=True)
    )


def time_slot_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return row counts by time slot."""

    order = ["night", "morning", "afternoon", "evening"]
    distribution = (
        df["time_slot"]
        .value_counts(dropna=False)
        .rename_axis("time_slot")
        .reset_index(name="rows")
    )
    distribution["time_slot"] = pd.Categorical(distribution["time_slot"], categories=order, ordered=True)
    return distribution.sort_values("time_slot").reset_index(drop=True)


def correlation_matrix(df: pd.DataFrame, numeric_columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Return Pearson correlation matrix for numeric columns."""

    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include="number").columns.tolist()
    return df.loc[:, list(numeric_columns)].corr(numeric_only=True)
