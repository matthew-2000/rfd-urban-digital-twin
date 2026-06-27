"""RFD similarity, validation, and discovery helpers."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd


THRESHOLD_CONFIGS = {
    "strict": {
        "station": "equal",
        "hour": 1,
        "time_slot": "equal",
        "TEMP": 1.0,
        "DEWP": 1.0,
        "WSPM": 0.5,
        "PM2.5": 5.0,
        "PM10": 10.0,
        "NO2": 5.0,
        "O3": 5.0,
    },
    "medium": {
        "station": "equal",
        "hour": 1,
        "time_slot": "equal",
        "TEMP": 2.0,
        "DEWP": 2.0,
        "WSPM": 1.0,
        "PM2.5": 10.0,
        "PM10": 15.0,
        "NO2": 10.0,
        "O3": 10.0,
    },
    "relaxed": {
        "station": "equal",
        "hour": 1,
        "time_slot": "equal",
        "TEMP": 3.0,
        "DEWP": 3.0,
        "WSPM": 2.0,
        "PM2.5": 15.0,
        "PM10": 25.0,
        "NO2": 15.0,
        "O3": 15.0,
    },
}


@dataclass(frozen=True)
class PairIndexCache:
    """Cache upper-triangle row indices for pairwise validation."""

    row_i: np.ndarray
    row_j: np.ndarray

    @property
    def total_pairs(self) -> int:
        return int(self.row_i.size)


def rule_to_label(lhs: Iterable[str], rhs: str) -> str:
    """Format a readable rule label."""

    return f"{', '.join(lhs)} -> {rhs}"


def is_similar(row1: pd.Series, row2: pd.Series, attr: str, threshold_config: dict[str, object]) -> bool:
    """Check whether two rows are similar on one attribute."""

    threshold = threshold_config[attr]
    value1 = row1[attr]
    value2 = row2[attr]

    if pd.isna(value1) or pd.isna(value2):
        return False
    if threshold == "equal":
        return value1 == value2
    if attr == "hour":
        diff = abs(int(value1) - int(value2))
        circular_diff = min(diff, 24 - diff)
        return circular_diff <= int(threshold)
    return abs(float(value1) - float(value2)) <= float(threshold)


def build_pair_index_cache(df: pd.DataFrame) -> PairIndexCache:
    """Create upper-triangle index arrays for pairwise comparisons."""

    row_i, row_j = np.triu_indices(len(df), k=1)
    return PairIndexCache(row_i=row_i, row_j=row_j)


def attribute_similarity_mask(
    df: pd.DataFrame,
    attr: str,
    thresholds: dict[str, object],
    pair_cache: PairIndexCache | None = None,
) -> np.ndarray:
    """Return boolean mask for pairwise similarity on one attribute."""

    if pair_cache is None:
        pair_cache = build_pair_index_cache(df)

    values = df[attr].to_numpy()
    left = values[pair_cache.row_i]
    right = values[pair_cache.row_j]
    threshold = thresholds[attr]

    if threshold == "equal":
        return left == right

    if attr == "hour":
        diff = np.abs(left.astype(int) - right.astype(int))
        circular_diff = np.minimum(diff, 24 - diff)
        return circular_diff <= int(threshold)

    diff = np.abs(left.astype(float) - right.astype(float))
    return diff <= float(threshold)


def values_similarity_mask(
    values: np.ndarray,
    attr: str,
    thresholds: dict[str, object],
    pair_cache: PairIndexCache,
) -> np.ndarray:
    """Return pairwise similarity mask for an explicit value array."""

    left = values[pair_cache.row_i]
    right = values[pair_cache.row_j]
    threshold = thresholds[attr]

    if threshold == "equal":
        return left == right

    if attr == "hour":
        diff = np.abs(left.astype(int) - right.astype(int))
        circular_diff = np.minimum(diff, 24 - diff)
        return circular_diff <= int(threshold)

    diff = np.abs(left.astype(float) - right.astype(float))
    return diff <= float(threshold)


def build_similarity_cache(
    df: pd.DataFrame,
    thresholds: dict[str, object],
    attrs: Iterable[str],
    pair_cache: PairIndexCache | None = None,
) -> dict[str, np.ndarray]:
    """Precompute pairwise similarity masks for a set of attributes."""

    if pair_cache is None:
        pair_cache = build_pair_index_cache(df)
    return {
        attr: attribute_similarity_mask(df=df, attr=attr, thresholds=thresholds, pair_cache=pair_cache)
        for attr in attrs
    }


def _build_violation_examples(
    df: pd.DataFrame,
    lhs: list[str],
    rhs: str,
    violation_positions: np.ndarray,
    pair_cache: PairIndexCache,
    max_violations: int,
) -> list[dict[str, object]]:
    """Create flattened violation examples for export."""

    examples: list[dict[str, object]] = []
    for position in violation_positions[:max_violations]:
        idx1 = int(pair_cache.row_i[position])
        idx2 = int(pair_cache.row_j[position])
        row1 = df.iloc[idx1]
        row2 = df.iloc[idx2]
        example: dict[str, object] = {
            "rule_label": rule_to_label(lhs, rhs),
            "lhs": ", ".join(lhs),
            "rhs": rhs,
            "row_index_1": int(row1.get("source_index", idx1)),
            "row_index_2": int(row2.get("source_index", idx2)),
            "datetime_1": row1["datetime"],
            "datetime_2": row2["datetime"],
            "station_1": row1["station"],
            "station_2": row2["station"],
            "rhs_value_1": row1[rhs],
            "rhs_value_2": row2[rhs],
            "rhs_abs_diff": abs(float(row1[rhs]) - float(row2[rhs])),
        }
        for attr in lhs:
            example[f"{attr}_1"] = row1[attr]
            example[f"{attr}_2"] = row2[attr]
        examples.append(example)
    return examples


def permutation_baseline_confidence(
    df: pd.DataFrame,
    rhs: str,
    thresholds: dict[str, object],
    antecedent_mask: np.ndarray,
    pair_cache: PairIndexCache,
    permutations: int = 30,
    random_state: int = 42,
) -> float | None:
    """Estimate RHS baseline confidence by permuting RHS values."""

    mean, _ = permutation_baseline_statistics(
        df=df,
        rhs=rhs,
        thresholds=thresholds,
        antecedent_mask=antecedent_mask,
        pair_cache=pair_cache,
        permutations=permutations,
        random_state=random_state,
    )
    return mean


def permutation_baseline_statistics(
    df: pd.DataFrame,
    rhs: str,
    thresholds: dict[str, object],
    antecedent_mask: np.ndarray,
    pair_cache: PairIndexCache,
    permutations: int = 30,
    random_state: int = 42,
) -> tuple[float | None, float | None]:
    """Estimate mean and standard deviation of the permuted-RHS confidence."""

    antecedent_pairs = int(antecedent_mask.sum())
    if antecedent_pairs == 0 or permutations <= 0:
        return None, None

    rng = np.random.default_rng(random_state)
    rhs_values = df[rhs].to_numpy()
    confidences: list[float] = []
    for _ in range(permutations):
        permuted_values = rng.permutation(rhs_values)
        permuted_rhs_mask = values_similarity_mask(
            values=permuted_values,
            attr=rhs,
            thresholds=thresholds,
            pair_cache=pair_cache,
        )
        confidences.append(float((antecedent_mask & permuted_rhs_mask).sum() / antecedent_pairs))
    return float(np.mean(confidences)), float(np.std(confidences, ddof=0))


def validate_rfd(
    df: pd.DataFrame,
    lhs: Iterable[str],
    rhs: str,
    thresholds: dict[str, object],
    max_violations: int = 10,
    pair_cache: PairIndexCache | None = None,
    similarity_cache: dict[str, np.ndarray] | None = None,
    baseline_permutations: int = 0,
    random_state: int = 42,
) -> dict[str, object]:
    """Validate one RFD and return observed metrics plus optional permutation baseline."""

    lhs = list(lhs)
    if pair_cache is None:
        pair_cache = build_pair_index_cache(df)

    required_attrs = set(lhs + [rhs])
    if similarity_cache is None:
        similarity_cache = build_similarity_cache(df, thresholds, required_attrs, pair_cache)

    lhs_masks = [similarity_cache[attr] for attr in lhs]
    antecedent_mask = np.logical_and.reduce(lhs_masks) if lhs_masks else np.ones(pair_cache.total_pairs, dtype=bool)
    rhs_mask = similarity_cache[rhs]

    total_pairs = pair_cache.total_pairs
    antecedent_pairs = int(antecedent_mask.sum())
    support = antecedent_pairs / total_pairs if total_pairs else 0.0

    if antecedent_pairs == 0:
        return {
            "lhs": lhs,
            "rhs": rhs,
            "total_pairs": total_pairs,
            "antecedent_pairs": 0,
            "valid_pairs": 0,
            "violations": 0,
            "support": support,
            "confidence": None,
            "violation_rate": None,
            "baseline_confidence": None,
            "baseline_confidence_std": None,
            "lift": None,
            "violation_examples": [],
        }

    valid_mask = antecedent_mask & rhs_mask
    valid_pairs = int(valid_mask.sum())
    violations = antecedent_pairs - valid_pairs
    confidence = valid_pairs / antecedent_pairs
    violation_rate = 1.0 - confidence
    baseline_confidence, baseline_confidence_std = permutation_baseline_statistics(
        df=df,
        rhs=rhs,
        thresholds=thresholds,
        antecedent_mask=antecedent_mask,
        pair_cache=pair_cache,
        permutations=baseline_permutations,
        random_state=random_state,
    )
    lift = (
        confidence / baseline_confidence
        if baseline_confidence is not None and baseline_confidence > 0
        else None
    )
    violation_positions = np.flatnonzero(antecedent_mask & ~rhs_mask)
    violation_examples = _build_violation_examples(
        df=df,
        lhs=lhs,
        rhs=rhs,
        violation_positions=violation_positions,
        pair_cache=pair_cache,
        max_violations=max_violations,
    )

    return {
        "lhs": lhs,
        "rhs": rhs,
        "total_pairs": total_pairs,
        "antecedent_pairs": antecedent_pairs,
        "valid_pairs": valid_pairs,
        "violations": violations,
        "support": support,
        "confidence": confidence,
        "violation_rate": violation_rate,
        "baseline_confidence": baseline_confidence,
        "baseline_confidence_std": baseline_confidence_std,
        "lift": lift,
        "violation_examples": violation_examples,
    }


def discover_rfds(
    df: pd.DataFrame,
    lhs_attributes: Iterable[str],
    rhs_attributes: Iterable[str],
    thresholds: dict[str, object],
    min_support: float = 0.01,
    min_confidence: float = 0.85,
    max_lhs_size: int = 3,
    top_k: int = 10,
    pair_cache: PairIndexCache | None = None,
    similarity_cache: dict[str, np.ndarray] | None = None,
) -> list[dict[str, object]]:
    """Run lightweight RFD discovery over small LHS combinations."""

    lhs_attributes = list(lhs_attributes)
    rhs_attributes = list(rhs_attributes)
    if pair_cache is None:
        pair_cache = build_pair_index_cache(df)

    required_attrs = set(lhs_attributes) | set(rhs_attributes)
    if similarity_cache is None:
        similarity_cache = build_similarity_cache(df, thresholds, required_attrs, pair_cache)

    discovered: list[dict[str, object]] = []
    for lhs_size in range(1, max_lhs_size + 1):
        for lhs in combinations(lhs_attributes, lhs_size):
            lhs_set = set(lhs)
            for rhs in rhs_attributes:
                if rhs in lhs_set:
                    continue
                result = validate_rfd(
                    df=df,
                    lhs=list(lhs),
                    rhs=rhs,
                    thresholds=thresholds,
                    max_violations=0,
                    pair_cache=pair_cache,
                    similarity_cache=similarity_cache,
                )
                confidence = result["confidence"]
                if confidence is None:
                    continue
                if result["support"] < min_support or confidence < min_confidence:
                    continue
                discovered.append(result)

    discovered.sort(
        key=lambda item: (
            -(item["confidence"] or 0.0),
            -item["support"],
            len(item["lhs"]),
            rule_to_label(item["lhs"], item["rhs"]),
        )
    )
    return discovered[:top_k]
