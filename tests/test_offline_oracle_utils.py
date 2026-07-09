"""Focused tests for leakage-safe offline oracle utilities."""

from __future__ import annotations

import hashlib
import unittest

import numpy as np
import numpy.testing as npt

from llm_experiments.offline_oracle_utils import (
    DEFAULT_SPLIT_SEED,
    GROUP_SIGNAL_BLOCKS_V1,
    SPLIT_SIZES,
    audit_group_split,
    group_reward_summaries,
    group_signal_blocks_v1,
    pair_level_metrics,
)


def _synthetic_pair_rewards() -> np.ndarray:
    rewards = np.zeros((256, 32, 2), dtype=np.float32)
    for sample_id in range(256):
        informative_pairs = (sample_id * 13) % 33
        for pair_id in range(informative_pairs):
            amplitude = np.float32(0.01 * (1 + (sample_id + pair_id) % 10))
            rewards[sample_id, pair_id, (sample_id + pair_id) % 2] = amplitude
        if sample_id % 11 == 0:
            rewards[sample_id, 0, 0] = np.float32(1.1)
        if sample_id % 17 == 0:
            rewards[sample_id, 1, 1] = np.float32(1.05)
    return rewards


class GroupSplitTest(unittest.TestCase):
    def test_group_signal_blocks_v1_is_exact_and_deterministic(self):
        rewards = _synthetic_pair_rewards()
        first = group_signal_blocks_v1(rewards)
        second = group_signal_blocks_v1(rewards, seed=DEFAULT_SPLIT_SEED)

        self.assertEqual(set(first), set(SPLIT_SIZES))
        for name, expected_size in SPLIT_SIZES.items():
            self.assertEqual(first[name].shape, (expected_size,))
            npt.assert_array_equal(first[name], second[name])

        joined = np.concatenate([first[name] for name in SPLIT_SIZES])
        npt.assert_array_equal(np.sort(joined), np.arange(256))
        self.assertEqual(np.unique(joined).size, 256)
        fingerprint = hashlib.sha256(
            b"".join(first[name].tobytes() for name in SPLIT_SIZES)
        ).hexdigest()
        self.assertEqual(
            fingerprint,
            "ac4c1992ca2d9c39a22b9090a487b9536e7e533400b99210518a0eb1bd1b394b",
        )

    def test_summary_and_audit_report_independent_units(self):
        rewards = _synthetic_pair_rewards()
        summaries = group_reward_summaries(rewards)
        splits = group_signal_blocks_v1(rewards)
        audit = audit_group_split(rewards, splits)

        self.assertEqual(summaries.shape, (256,))
        self.assertEqual(audit["algorithm"], GROUP_SIGNAL_BLOCKS_V1)
        self.assertTrue(audit["group_disjoint"])
        self.assertTrue(audit["exhaustive"])
        self.assertEqual(audit["sample_groups"], 256)
        self.assertEqual(audit["pairs_per_group"], 32)
        self.assertEqual(audit["layers_per_pair"], 24)
        self.assertEqual(audit["splits"]["train"]["sample_groups"], 192)
        self.assertEqual(audit["splits"]["validation"]["independent_pairs"], 1024)
        self.assertEqual(audit["splits"]["test"]["logical_layer_rows"], 24576)
        self.assertGreater(audit["splits"]["test"]["informative_pairs"], 0)

    def test_audit_rejects_cross_split_group_leakage(self):
        rewards = _synthetic_pair_rewards()
        splits = {
            name: values.copy()
            for name, values in group_signal_blocks_v1(rewards).items()
        }
        splits["test"][0] = splits["train"][0]
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            audit_group_split(rewards, splits)

    def test_split_contract_rejects_non_dataset_shape(self):
        with self.assertRaisesRegex(ValueError, r"\[256, 32, 2\]"):
            group_signal_blocks_v1(np.zeros((255, 32, 2), dtype=np.float32))


class PairMetricTest(unittest.TestCase):
    def setUp(self):
        self.y = np.asarray(
            [
                [8, 7, 6, 5, 4, 3, 2, 1],
                [-1, -2, -3, -4, -5, -6, -7, -8],
            ],
            dtype=np.float64,
        )

    def test_perfect_predictions(self):
        metrics = pair_level_metrics(self.y, self.y)
        self.assertEqual(metrics["mse"], 0.0)
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["residual_ratio"], 0.0)
        self.assertEqual(metrics["r2_zero"], 1.0)
        self.assertAlmostEqual(metrics["pooled_pearson"], 1.0)
        self.assertAlmostEqual(metrics["macro_prompt_cosine"], 1.0)
        self.assertAlmostEqual(metrics["macro_prompt_spearman"], 1.0)
        self.assertEqual(metrics["nonzero_sign_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["calibration_slope"], 1.0)

        total_energy = float(np.sum(np.arange(1, 9, dtype=np.float64) ** 2))
        self.assertAlmostEqual(metrics["top1_energy_recall"], 64.0 / total_energy)
        self.assertAlmostEqual(
            metrics["top2_energy_recall"], (64.0 + 49.0) / total_energy
        )
        self.assertAlmostEqual(
            metrics["top4_energy_recall"],
            (64.0 + 49.0 + 36.0 + 25.0) / total_energy,
        )
        self.assertAlmostEqual(metrics["top8_energy_recall"], 1.0)

    def test_reversed_predictions_expose_update_harm(self):
        metrics = pair_level_metrics(self.y, -self.y)
        self.assertAlmostEqual(metrics["residual_ratio"], 4.0)
        self.assertAlmostEqual(metrics["r2_zero"], -3.0)
        self.assertAlmostEqual(metrics["pooled_pearson"], -1.0)
        self.assertAlmostEqual(metrics["macro_prompt_cosine"], -1.0)
        self.assertAlmostEqual(metrics["macro_prompt_spearman"], -1.0)
        self.assertEqual(metrics["nonzero_sign_accuracy"], 0.0)
        self.assertAlmostEqual(metrics["calibration_slope"], -1.0)
        self.assertAlmostEqual(metrics["top8_energy_recall"], 1.0)

    def test_calibration_slope_and_constant_prediction_behavior(self):
        scaled = pair_level_metrics(self.y, 2.0 * self.y)
        self.assertAlmostEqual(scaled["calibration_slope"], 0.5)
        self.assertAlmostEqual(scaled["macro_prompt_cosine"], 1.0)

        zero = pair_level_metrics(self.y, np.zeros_like(self.y))
        self.assertTrue(np.isnan(zero["pooled_pearson"]))
        self.assertTrue(np.isnan(zero["calibration_slope"]))
        self.assertEqual(zero["macro_prompt_cosine"], 0.0)
        self.assertEqual(zero["macro_prompt_spearman"], 0.0)
        self.assertEqual(zero["nonzero_sign_accuracy"], 0.0)

    def test_metrics_reject_layer_rows_or_nonfinite_values(self):
        with self.assertRaisesRegex(ValueError, r"\[prompts, pairs\]"):
            pair_level_metrics(self.y.ravel(), self.y.ravel())
        bad = self.y.copy()
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            pair_level_metrics(bad, self.y)


if __name__ == "__main__":
    unittest.main()
