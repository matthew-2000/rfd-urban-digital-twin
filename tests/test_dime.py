"""Regression tests for the course DiMε implementation."""

import unittest

import pandas as pd

from src.algorithms.dime import build_difference_matrices, discover_dime


class DimeDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.paper_example = pd.DataFrame(
            {
                "Height": [175, 175, 175, 176, 178, 169, 170],
                "Weight": [70, 75, 69, 71, 81, 73, 62],
                "Shoe": [40, 39, 40, 40, 41, 37, 39],
            }
        )
        self.thresholds = {"Height": 1.0, "Weight": 10.0, "Shoe": 1.0}

    def test_difference_matrix_matches_lecture_example(self) -> None:
        matrices, row_i, row_j = build_difference_matrices(
            self.paper_example,
            self.thresholds,
            self.thresholds,
        )
        position = next(
            index
            for index, pair in enumerate(zip(row_i, row_j))
            if pair == (0, 3)
        )
        self.assertEqual(matrices["Height"].distance[position], 1.0)
        self.assertTrue(matrices["Height"].similar[position])

    def test_discovers_lecture_rfd_without_lhs_cap(self) -> None:
        result, discoverer = discover_dime(
            self.paper_example,
            self.thresholds,
            extent_threshold=0.0,
        )
        self.assertIn("Height, Weight -> Shoe", set(result["rule_label"]))
        self.assertEqual(discoverer.levels_visited_, 3)
        self.assertTrue(result["is_minimal"].all())

    def test_hybrid_discovery_records_g3_mode(self) -> None:
        result, _ = discover_dime(
            self.paper_example,
            self.thresholds,
            extent_threshold=0.2,
            g3_mode="greedy",
        )
        self.assertFalse(result.empty)
        self.assertTrue((result["g3_mode"] == "greedy").all())
        self.assertTrue((result["g3_error"] <= 0.2).all())

    def test_accepts_attribute_specific_distance_function(self) -> None:
        frame = pd.DataFrame({"Code": ["aa", "ab", "zz"], "Value": [1, 1, 2]})
        matrices, _, _ = build_difference_matrices(
            frame,
            ["Code", "Value"],
            {"Code": 1.0, "Value": 0.0},
            distance_functions={
                "Code": lambda left, right: sum(a != b for a, b in zip(left, right))
            },
        )
        self.assertEqual(int(matrices["Code"].similar.sum()), 1)


if __name__ == "__main__":
    unittest.main()
