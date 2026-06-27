"""Data loading and preprocessing helpers for the UDT RFD project."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_STATIONS = ("Aotizhongxin", "Changping")
DEFAULT_START = "2013-03-01 00:00:00"
DEFAULT_END = "2017-02-28 23:00:00"
DEFAULT_COLUMNS = [
    "datetime",
    "station",
    "hour",
    "PM2.5",
    "PM10",
    "NO2",
    "O3",
    "TEMP",
    "DEWP",
    "WSPM",
    "time_slot",
]
RAW_USECOLS = [
    "year",
    "month",
    "day",
    "hour",
    "PM2.5",
    "PM10",
    "NO2",
    "O3",
    "TEMP",
    "DEWP",
    "WSPM",
    "station",
]


def load_station_files(raw_dir: Path, stations: Iterable[str] = DEFAULT_STATIONS) -> pd.DataFrame:
    """Load and combine Beijing station CSV files for selected stations."""

    raw_dir = Path(raw_dir)
    frames: list[pd.DataFrame] = []
    for station in stations:
        path = raw_dir / f"PRSA_Data_{station}_20130301-20170228.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing raw station file: {path}")
        frames.append(pd.read_csv(path, usecols=RAW_USECOLS))

    combined = pd.concat(frames, ignore_index=True)
    return combined


def create_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    """Create a timestamp column from year, month, day, and hour."""

    enriched = df.copy()
    enriched["datetime"] = pd.to_datetime(
        enriched[["year", "month", "day", "hour"]],
        errors="coerce",
    )
    return enriched


def add_time_slot(df: pd.DataFrame) -> pd.DataFrame:
    """Map each hour to one of four coarse-grained daily slots."""

    enriched = df.copy()
    conditions = [
        enriched["hour"].between(0, 5),
        enriched["hour"].between(6, 11),
        enriched["hour"].between(12, 17),
        enriched["hour"].between(18, 23),
    ]
    labels = ["night", "morning", "afternoon", "evening"]
    enriched["time_slot"] = pd.Series(pd.NA, index=enriched.index, dtype="object")
    for condition, label in zip(conditions, labels):
        enriched.loc[condition, "time_slot"] = label
    return enriched


def filter_period(
    df: pd.DataFrame,
    stations: Iterable[str] = DEFAULT_STATIONS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> pd.DataFrame:
    """Filter dataframe to selected stations and date window."""

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    mask = (
        df["station"].isin(list(stations))
        & df["datetime"].between(start_ts, end_ts, inclusive="both")
    )
    return df.loc[mask].copy()


def summarize_missing_values(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return missing-value counts and rates for selected columns."""

    summary = pd.DataFrame(
        {
            "column": list(columns),
            "missing_count": [int(df[column].isna().sum()) for column in columns],
        }
    )
    summary["missing_rate"] = summary["missing_count"] / len(df)
    return summary.sort_values(["missing_count", "column"], ascending=[False, True]).reset_index(
        drop=True
    )


def clean_dataset(df: pd.DataFrame, columns: Iterable[str] = DEFAULT_COLUMNS) -> pd.DataFrame:
    """Drop rows with missing values in selected columns and keep final schema."""

    selected_columns = list(columns)
    cleaned = df.dropna(subset=selected_columns).copy()
    cleaned = cleaned.loc[:, selected_columns].sort_values(["datetime", "station"]).reset_index(
        drop=True
    )
    return cleaned


def prepare_udt_dataset(
    raw_dir: Path,
    processed_path: Path,
    stations: Iterable[str] = DEFAULT_STATIONS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create cleaned UDT dataset and save it to disk.

    Returns cleaned dataframe and missing-value summary computed before row removal.
    """

    raw_df = load_station_files(raw_dir=raw_dir, stations=stations)
    filtered_df = filter_period(
        add_time_slot(create_datetime_column(raw_df)),
        stations=stations,
        start=start,
        end=end,
    )
    missing_summary = summarize_missing_values(filtered_df, DEFAULT_COLUMNS)
    cleaned_df = clean_dataset(filtered_df, DEFAULT_COLUMNS)

    processed_path = Path(processed_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(processed_path, index=False)
    return cleaned_df, missing_summary
