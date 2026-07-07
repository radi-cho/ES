"""Tests for adaptive surrogate-fitness EGGROLL utilities."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import numpy.testing as npt

from hyperscalees.noiser.adaptive_surrogate_eggroll import (
    assemble_member_fitness,
    compute_trust_score,
    pair_differences_to_member_fitness,
    sample_surrogate_pair_mask,
)


def test_surrogate_fraction_controls_pair_counts():
    mask = sample_surrogate_pair_mask(
        num_prompts=8,
        pairs_per_prompt=4,
        surrogate_fraction=0.25,
        seed=0,
        epoch=3,
    )

    assert mask.shape == (32,)
    assert np.count_nonzero(mask) == 8
    grouped = mask.reshape(8, 4)
    assert np.all(grouped.sum(axis=1) == 1)


def test_assemble_member_fitness_marks_rollout_and_surrogate_members():
    predicted = pair_differences_to_member_fitness(
        np.asarray([1.0, -2.0, 0.5], dtype=np.float32)
    )
    rollout_ids = np.asarray([2, 3], dtype=np.int32)
    rollout_scores = np.asarray([0.4, -0.4], dtype=np.float32)
    mask = np.asarray([False, True, True], dtype=bool)

    combined, rollout_pairs, surrogate_pairs = assemble_member_fitness(
        total_members=6,
        predicted_member_fitness=predicted,
        rollout_member_fitness=rollout_scores,
        rollout_member_ids=rollout_ids,
        surrogate_pair_mask=mask,
    )

    npt.assert_allclose(combined[:2], predicted[:2])
    npt.assert_allclose(combined[2:4], rollout_scores)
    npt.assert_allclose(combined[4:], predicted[4:])
    assert rollout_pairs == 1
    assert surrogate_pairs == 2


def test_trust_score_is_clipped_correlation():
    score = compute_trust_score(
        np.asarray([1.0, 0.5, -0.25], dtype=np.float32),
        np.asarray([0.9, 0.4, -0.1], dtype=np.float32),
    )
    assert 0.0 < score <= 1.0


class AdaptiveSurrogateEggRollTest(unittest.TestCase):
    pass


def _as_unittest_method(test_function):
    def method(self):
        del self
        test_function()

    method.__name__ = test_function.__name__
    return method


for _name, _test in list(globals().items()):
    if _name.startswith("test_") and callable(_test):
        setattr(AdaptiveSurrogateEggRollTest, _name, _as_unittest_method(_test))


if __name__ == "__main__":
    unittest.main()
