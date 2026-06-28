"""DiMε discovery for relaxed functional dependencies.

This module follows the level-wise algorithm presented by Caruccio, Deufemia,
and Polese (Data Mining and Knowledge Discovery, 2020) and the authors'
reference implementation. It uses difference matrices, stripped similar
pattern subsets, TANE-style C+ candidate sets, DiMε pruning, and g3 validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import math
from typing import Callable, Iterable, Literal, Mapping

import numpy as np
import pandas as pd


G3Mode = Literal["greedy", "exact"]
DistanceFunction = Callable[[object, object], float]


@dataclass(frozen=True)
class DifferenceMatrix:
    """Condensed upper-triangle difference and similarity matrices."""

    attribute: str
    distance: np.ndarray
    similar: np.ndarray
    threshold: float | str


@dataclass(frozen=True)
class SimilarPatternSubset:
    """Rows sharing one identical similarity-neighbourhood pattern."""

    pattern: tuple[int, ...]
    owners: tuple[int, ...]


@dataclass
class StrippedSimilarSubsets:
    """DiMε stripped similar pattern subsets for an attribute set."""

    attributes: frozenset[str]
    pair_mask: np.ndarray
    row_i: np.ndarray
    row_j: np.ndarray
    number_tuples: int
    _subsets: tuple[SimilarPatternSubset, ...] | None = field(default=None, repr=False)
    _max_cardinality: int | None = field(default=None, repr=False)

    @property
    def pair_count(self) -> int:
        """Return ||Î_X||, the number of similar unordered tuple pairs."""

        return int(self.pair_mask.sum())

    def _materialize(self) -> None:
        if self._subsets is not None:
            return

        adjacency = np.eye(self.number_tuples, dtype=bool)
        adjacency[self.row_i, self.row_j] = self.pair_mask
        adjacency[self.row_j, self.row_i] = self.pair_mask
        grouped: dict[bytes, list[int]] = {}
        patterns: dict[bytes, tuple[int, ...]] = {}
        max_cardinality = 0
        for row_index in range(self.number_tuples):
            key = adjacency[row_index].tobytes()
            grouped.setdefault(key, []).append(row_index)
            if key not in patterns:
                pattern = tuple(np.flatnonzero(adjacency[row_index]).tolist())
                patterns[key] = pattern
                max_cardinality = max(max_cardinality, len(pattern) - 1)

        self._subsets = tuple(
            SimilarPatternSubset(pattern=patterns[key], owners=tuple(owners))
            for key, owners in grouped.items()
            if len(patterns[key]) > 1
        )
        self._max_cardinality = max_cardinality

    @property
    def subsets(self) -> tuple[SimilarPatternSubset, ...]:
        """Return the stripped subsets (singleton patterns are omitted)."""

        self._materialize()
        assert self._subsets is not None
        return self._subsets

    @property
    def max_cardinality(self) -> int:
        """Largest number of neighbours of a tuple in this partition."""

        if self._max_cardinality is None:
            if not self.pair_mask.any():
                self._max_cardinality = 0
            else:
                degree = np.bincount(
                    np.concatenate((self.row_i[self.pair_mask], self.row_j[self.pair_mask])),
                    minlength=self.number_tuples,
                )
                self._max_cardinality = int(degree.max(initial=0))
        return self._max_cardinality

    def is_disjoint(self) -> bool:
        """Whether every stripped pattern is an equivalence class."""

        return all(subset.pattern == subset.owners for subset in self.subsets)

    def disjoint_g3_error(self, refined: "StrippedSimilarSubsets") -> float:
        """Polynomial g3 calculation used by DiMε for disjoint subsets."""

        refined_subsets = refined.subsets
        retained = self.number_tuples - sum(len(subset.owners) for subset in self.subsets)
        for subset in self.subsets:
            pattern = set(subset.pattern)
            owners = set(subset.owners)
            largest = 0
            for candidate in refined_subsets:
                if (
                    set(candidate.pattern).issubset(pattern)
                    and set(candidate.owners).issubset(owners)
                ):
                    largest = max(largest, len(candidate.pattern))
            retained += largest if largest else 1
        return 1.0 - retained / self.number_tuples


@dataclass(frozen=True)
class DimeRule:
    """One minimal RFD emitted by the DiMε lattice."""

    lhs: tuple[str, ...]
    rhs: str
    g3_error: float
    extent_threshold: float
    level: int
    exact_comparison: bool
    discovery_kind: str
    is_minimal: bool = True

    @property
    def rule_label(self) -> str:
        return f"{', '.join(self.lhs)} -> {self.rhs}"


@dataclass
class _LatticeNode:
    attributes: frozenset[str]
    partition: StrippedSimilarSubsets | None
    rhs_candidates: set[str] = field(default_factory=set)
    valid: bool = True


def _categorical_distance(values: np.ndarray, row_i: np.ndarray, row_j: np.ndarray) -> np.ndarray:
    return (values[row_i] != values[row_j]).astype(np.float32)


def _numeric_distance(values: np.ndarray, row_i: np.ndarray, row_j: np.ndarray) -> np.ndarray:
    numeric = values.astype(float)
    return np.abs(numeric[row_i] - numeric[row_j]).astype(np.float32)


def build_difference_matrices(
    df: pd.DataFrame,
    attributes: Iterable[str],
    thresholds: Mapping[str, float | str],
    distance_functions: Mapping[str, DistanceFunction] | None = None,
) -> tuple[dict[str, DifferenceMatrix], np.ndarray, np.ndarray]:
    """Build the per-attribute DiMε difference matrices."""

    row_i, row_j = np.triu_indices(len(df), k=1)
    matrices: dict[str, DifferenceMatrix] = {}
    distance_functions = dict(distance_functions or {})
    for attribute in attributes:
        threshold = thresholds[attribute]
        values = df[attribute].to_numpy()
        if attribute in distance_functions:
            function = distance_functions[attribute]
            distance = np.fromiter(
                (function(values[left], values[right]) for left, right in zip(row_i, row_j)),
                dtype=np.float32,
                count=len(row_i),
            )
            numeric_threshold = 0.0 if threshold == "equal" else float(threshold)
            similar = distance <= numeric_threshold
        elif threshold == "equal":
            distance = _categorical_distance(values, row_i, row_j)
            similar = distance == 0
        else:
            distance = _numeric_distance(values, row_i, row_j)
            similar = distance <= float(threshold)
        matrices[attribute] = DifferenceMatrix(
            attribute=attribute,
            distance=distance,
            similar=similar,
            threshold=threshold,
        )
    return matrices, row_i, row_j


def _official_greedy_vertex_cover(
    violation_mask: np.ndarray,
    row_i: np.ndarray,
    row_j: np.ndarray,
    number_tuples: int,
) -> tuple[int, tuple[int, ...]]:
    """Official DiMε greedy g3 approximation in deterministic tuple order."""

    edge_positions = np.flatnonzero(violation_mask)
    if edge_positions.size == 0:
        return 0, ()

    adjacency: dict[int, set[int]] = {}
    for u, v in zip(row_i[edge_positions], row_j[edge_positions]):
        left = int(u)
        right = int(v)
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)

    # The reference JAR stores the violation graph in fastutil's
    # Long2ObjectOpenHashMap and iterates over its key table. Reproducing that
    # order is necessary because the published greedy cover is order-sensitive.
    iteration_order = _fastutil_long_iteration_order(sorted(adjacency))
    removed: set[int] = set()
    selected: list[int] = []
    for vertex in iteration_order:
        if vertex in removed:
            continue
        for other in iteration_order:
            if (
                other in removed
                or other == vertex
                or vertex not in adjacency[other]
            ):
                continue
            adjacency[other].remove(vertex)
            if not adjacency[other]:
                removed.add(other)
        removed.add(vertex)
        selected.append(vertex)
    return len(selected), tuple(selected)


def _fastutil_long_mix(value: int) -> int:
    """fastutil 8.x HashCommon.mix(long), with Java unsigned shifts."""

    mask64 = (1 << 64) - 1
    long_phi = (-7046029254386353131) & mask64
    mixed = (value * long_phi) & mask64
    mixed ^= mixed >> 32
    return (mixed ^ (mixed >> 16)) & mask64


def _fastutil_array_size(expected: int, load_factor: float = 0.75) -> int:
    required = max(2, math.ceil(expected / load_factor))
    return 1 << (required - 1).bit_length()


def _fastutil_max_fill(size: int, load_factor: float = 0.75) -> int:
    return min(math.ceil(size * load_factor), size - 1)


def _fastutil_long_iteration_order(keys: Iterable[int]) -> list[int]:
    """Emulate Long2ObjectOpenHashMap key iteration in the reference JAR."""

    load_factor = 0.75
    table_size = _fastutil_array_size(16, load_factor)
    table = [0] * (table_size + 1)
    contains_zero = False
    stored = 0

    def rehash(old_table: list[int], old_size: int, new_size: int) -> list[int]:
        new_table = [0] * (new_size + 1)
        mask = new_size - 1
        for position in range(old_size - 1, -1, -1):
            key = old_table[position]
            if key == 0:
                continue
            new_position = _fastutil_long_mix(key) & mask
            while new_table[new_position] != 0:
                new_position = (new_position + 1) & mask
            new_table[new_position] = key
        return new_table

    for key in keys:
        if key == 0:
            contains_zero = True
        else:
            position = _fastutil_long_mix(key) & (table_size - 1)
            while table[position] != 0:
                position = (position + 1) & (table_size - 1)
            table[position] = key
        previous_size = stored
        stored += 1
        if previous_size >= _fastutil_max_fill(table_size, load_factor):
            new_size = _fastutil_array_size(stored + 1, load_factor)
            table = rehash(table, table_size, new_size)
            table_size = new_size

    order = [0] if contains_zero else []
    order.extend(
        table[position]
        for position in range(table_size - 1, -1, -1)
        if table[position] != 0
    )
    return order


def _exact_vertex_cover(
    violation_mask: np.ndarray,
    row_i: np.ndarray,
    row_j: np.ndarray,
    maximum_size: int,
) -> tuple[int | None, tuple[int, ...]]:
    """Exact vertex-cover decision procedure used by DiMε's optional exact mode."""

    positions = np.flatnonzero(violation_mask)
    edges = frozenset(
        (int(row_i[position]), int(row_j[position]))
        for position in positions
    )
    if not edges:
        return 0, ()

    best: tuple[int, ...] | None = None

    def search(remaining: frozenset[tuple[int, int]], chosen: tuple[int, ...]) -> None:
        nonlocal best
        if not remaining:
            if best is None or len(chosen) < len(best):
                best = chosen
            return
        if len(chosen) >= maximum_size:
            return
        if best is not None and len(chosen) >= len(best):
            return
        u, v = next(iter(remaining))
        for vertex in (u, v):
            reduced = frozenset(edge for edge in remaining if vertex not in edge)
            search(reduced, chosen + (vertex,))

    search(edges, ())
    if best is None:
        return None, ()
    return len(best), best


class DimeDiscovery:
    """Full level-wise DiMε discovery over a supplied relation projection."""

    def __init__(
        self,
        thresholds: Mapping[str, float | str],
        extent_threshold: float = 0.10,
        g3_mode: G3Mode = "greedy",
        distance_functions: Mapping[str, DistanceFunction] | None = None,
    ) -> None:
        if not 0.0 <= extent_threshold <= 1.0:
            raise ValueError("extent_threshold must be in [0, 1]")
        if g3_mode not in {"greedy", "exact"}:
            raise ValueError("g3_mode must be 'greedy' or 'exact'")
        self.thresholds = dict(thresholds)
        self.distance_functions = dict(distance_functions or {})
        self.extent_threshold = float(extent_threshold)
        self.g3_mode = g3_mode
        self.rules_: list[DimeRule] = []
        self.difference_matrices_: dict[str, DifferenceMatrix] = {}
        self.levels_visited_: int = 0
        self.candidates_validated_: int = 0
        self._row_i = np.array([], dtype=int)
        self._row_j = np.array([], dtype=int)
        self._number_tuples = 0
        self._attributes: tuple[str, ...] = ()

    def _partition(
        self,
        attributes: frozenset[str],
    ) -> StrippedSimilarSubsets:
        if not attributes:
            pair_mask = np.ones(self._row_i.size, dtype=bool)
        else:
            pair_mask = np.logical_and.reduce(
                [self.difference_matrices_[attribute].similar for attribute in attributes]
            )
        return StrippedSimilarSubsets(
            attributes=attributes,
            pair_mask=pair_mask,
            row_i=self._row_i,
            row_j=self._row_j,
            number_tuples=self._number_tuples,
        )

    def _g3_error(
        self,
        antecedent: StrippedSimilarSubsets,
        refined: StrippedSimilarSubsets,
    ) -> float:
        violation_mask = antecedent.pair_mask & ~refined.pair_mask
        if not violation_mask.any():
            return 0.0
        if antecedent.is_disjoint():
            return antecedent.disjoint_g3_error(refined)

        violating_vertices = np.unique(
            np.concatenate(
                (
                    self._row_i[violation_mask],
                    self._row_j[violation_mask],
                )
            )
        )
        if violating_vertices.size < self.extent_threshold * self._number_tuples:
            return violating_vertices.size / self._number_tuples

        if self.g3_mode == "greedy":
            count, _ = _official_greedy_vertex_cover(
                violation_mask,
                self._row_i,
                self._row_j,
                self._number_tuples,
            )
        else:
            maximum_size = int(np.floor(self.extent_threshold * self._number_tuples))
            count, _ = _exact_vertex_cover(
                violation_mask,
                self._row_i,
                self._row_j,
                maximum_size=maximum_size,
            )
            if count is None:
                return 1.0
        return count / self._number_tuples

    def _validate(
        self,
        antecedent: StrippedSimilarSubsets,
        refined: StrippedSimilarSubsets,
    ) -> tuple[bool, float, bool]:
        exact = antecedent.pair_count == refined.pair_count
        if self.extent_threshold == 0.0:
            return exact, 0.0 if exact else 1.0, exact
        if exact:
            return True, 0.0, True

        total_pairs = self._row_i.size
        pair_loss = antecedent.pair_count - refined.pair_count
        bound = (
            self.extent_threshold
            * self._number_tuples
            * antecedent.max_cardinality
        )
        if total_pairs and pair_loss > bound:
            return False, 1.0, False

        g3_error = self._g3_error(antecedent, refined)
        return g3_error <= self.extent_threshold, g3_error, False

    def _initialize_candidates(
        self,
        level: dict[frozenset[str], _LatticeNode],
        previous: dict[frozenset[str], _LatticeNode],
    ) -> None:
        universe = set(self._attributes)
        for attribute_set, node in level.items():
            candidates = universe.copy()
            for attribute in attribute_set:
                subset = attribute_set - {attribute}
                candidates &= previous[subset].rhs_candidates
            node.rhs_candidates = candidates

    def _compute_dependencies(
        self,
        level_number: int,
        level: dict[frozenset[str], _LatticeNode],
        previous: dict[frozenset[str], _LatticeNode],
    ) -> None:
        self._initialize_candidates(level, previous)
        universe = set(self._attributes)
        for attribute_set, node in list(level.items()):
            if not node.valid or node.partition is None:
                continue
            for rhs in sorted(attribute_set & node.rhs_candidates):
                lhs = attribute_set - {rhs}
                antecedent = previous[lhs].partition
                if antecedent is None:
                    continue
                self.candidates_validated_ += 1
                holds, g3_error, exact = self._validate(antecedent, node.partition)
                if not holds:
                    continue
                self.rules_.append(
                    DimeRule(
                        lhs=tuple(sorted(lhs, key=self._attributes.index)),
                        rhs=rhs,
                        g3_error=g3_error,
                        extent_threshold=self.extent_threshold,
                        level=level_number,
                        exact_comparison=exact,
                        discovery_kind="candidate",
                    )
                )
                node.rhs_candidates.discard(rhs)
                if exact:
                    node.rhs_candidates -= universe - set(attribute_set)

    def _prune_keys(
        self,
        level_number: int,
        level: dict[frozenset[str], _LatticeNode],
    ) -> None:
        universe = set(self._attributes)
        remove: list[frozenset[str]] = []
        for attribute_set, node in list(level.items()):
            if not node.rhs_candidates:
                remove.append(attribute_set)
                continue
            if not node.valid or node.partition is None or node.partition.pair_count != 0:
                continue
            for rhs in sorted(node.rhs_candidates - set(attribute_set)):
                intersection = universe.copy()
                for attribute in attribute_set:
                    related = (attribute_set | {rhs}) - {attribute}
                    related_node = level.get(related)
                    if related_node is None:
                        intersection.clear()
                        break
                    intersection &= related_node.rhs_candidates
                if rhs not in intersection:
                    continue
                self.rules_.append(
                    DimeRule(
                        lhs=tuple(sorted(attribute_set, key=self._attributes.index)),
                        rhs=rhs,
                        g3_error=0.0,
                        extent_threshold=self.extent_threshold,
                        level=level_number,
                        exact_comparison=True,
                        discovery_kind="key",
                    )
                )
                node.rhs_candidates.discard(rhs)
                node.valid = False
        for attribute_set in remove:
            level.pop(attribute_set, None)

    @staticmethod
    def _prefix(attribute_set: frozenset[str], order: Mapping[str, int]) -> tuple[str, ...]:
        ordered = sorted(attribute_set, key=order.__getitem__)
        return tuple(ordered[:-1])

    def _generate_next_level(
        self,
        level: dict[frozenset[str], _LatticeNode],
    ) -> dict[frozenset[str], _LatticeNode]:
        order = {attribute: index for index, attribute in enumerate(self._attributes)}
        blocks: dict[tuple[str, ...], list[frozenset[str]]] = {}
        for attribute_set in level:
            blocks.setdefault(self._prefix(attribute_set, order), []).append(attribute_set)

        next_level: dict[frozenset[str], _LatticeNode] = {}
        for block in blocks.values():
            for left, right in combinations(block, 2):
                joined = left | right
                if joined in next_level:
                    continue
                if any((joined - {attribute}) not in level for attribute in joined):
                    continue
                valid = level[left].valid and level[right].valid
                partition = self._partition(joined) if valid else None
                next_level[joined] = _LatticeNode(
                    attributes=joined,
                    partition=partition,
                    valid=valid,
                )
        return next_level

    def fit(self, df: pd.DataFrame, attributes: Iterable[str] | None = None) -> "DimeDiscovery":
        """Run complete DiMε discovery and retain all minimal emitted RFDs."""

        selected = tuple(attributes or self.thresholds.keys())
        if len(selected) < 2:
            raise ValueError("DiMε requires at least two discovery attributes")
        missing = [attribute for attribute in selected if attribute not in df.columns]
        if missing:
            raise KeyError(f"Missing discovery attributes: {missing}")
        missing_thresholds = [attribute for attribute in selected if attribute not in self.thresholds]
        if missing_thresholds:
            raise KeyError(f"Missing thresholds: {missing_thresholds}")
        if df[list(selected)].isna().any().any():
            raise ValueError("DiMε input must not contain missing discovery values")

        self.rules_ = []
        self.levels_visited_ = 0
        self.candidates_validated_ = 0
        self._attributes = selected
        self._number_tuples = len(df)
        self.difference_matrices_, self._row_i, self._row_j = build_difference_matrices(
            df,
            selected,
            self.thresholds,
            self.distance_functions,
        )

        empty = frozenset()
        previous: dict[frozenset[str], _LatticeNode] = {
            empty: _LatticeNode(
                attributes=empty,
                partition=self._partition(empty),
                rhs_candidates=set(selected),
            )
        }
        level = {
            frozenset({attribute}): _LatticeNode(
                attributes=frozenset({attribute}),
                partition=self._partition(frozenset({attribute})),
            )
            for attribute in selected
        }

        level_number = 1
        while level and level_number <= len(selected):
            self.levels_visited_ = level_number
            self._compute_dependencies(level_number, level, previous)
            self._prune_keys(level_number, level)
            previous, level = level, self._generate_next_level(level)
            level_number += 1
        return self

    def to_frame(self) -> pd.DataFrame:
        """Return the discovered minimal RFDs as a stable table."""

        rows = [
            {
                "lhs": ", ".join(rule.lhs),
                "rhs": rule.rhs,
                "rule_label": rule.rule_label,
                "lhs_length": len(rule.lhs),
                "g3_error": rule.g3_error,
                "extent_threshold": rule.extent_threshold,
                "level": rule.level,
                "exact_comparison": rule.exact_comparison,
                "discovery_kind": rule.discovery_kind,
                "is_minimal": rule.is_minimal,
                "g3_mode": self.g3_mode,
            }
            for rule in self.rules_
        ]
        columns = [
            "lhs",
            "rhs",
            "rule_label",
            "lhs_length",
            "g3_error",
            "extent_threshold",
            "level",
            "exact_comparison",
            "discovery_kind",
            "is_minimal",
            "g3_mode",
        ]
        return pd.DataFrame(rows, columns=columns).sort_values(
            ["lhs_length", "rhs", "lhs"],
            kind="stable",
        ).reset_index(drop=True)


def discover_dime(
    df: pd.DataFrame,
    thresholds: Mapping[str, float | str],
    extent_threshold: float = 0.10,
    g3_mode: G3Mode = "greedy",
    attributes: Iterable[str] | None = None,
    distance_functions: Mapping[str, DistanceFunction] | None = None,
) -> tuple[pd.DataFrame, DimeDiscovery]:
    """Convenience wrapper returning both results and the fitted discoverer."""

    discoverer = DimeDiscovery(
        thresholds=thresholds,
        extent_threshold=extent_threshold,
        g3_mode=g3_mode,
        distance_functions=distance_functions,
    ).fit(df, attributes=attributes)
    return discoverer.to_frame(), discoverer
