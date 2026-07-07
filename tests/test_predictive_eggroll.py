"""Focused CPU tests for predictive virtual-population EGGROLL utilities."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from hyperscalees.noiser.eggroll import EggRoll, get_lora_update_params
from hyperscalees.noiser.predictive_eggroll import (
    OnlinePromptRidge,
    PredictiveEggRoll,
    audit_correct_pair_differences,
    build_mutation_feature_fn,
    pair_differences_to_member_utilities,
    pair_ids_to_member_ids,
    sample_stratified_audit_pairs,
)


def _observed_member_rewards(pair_differences: np.ndarray) -> np.ndarray:
    """Construct adjacent +/- rewards with the requested pair differences."""

    rewards = np.empty(2 * pair_differences.size, dtype=np.float32)
    rewards[0::2] = 0.5 * pair_differences
    rewards[1::2] = -0.5 * pair_differences
    return rewards


def test_one_in_sixteen_audits_complete_pairs_and_preserves_prompt_groups():
    num_prompts = 8
    physical_members_per_prompt = 8
    virtual_factor = 16
    physical_pairs = physical_members_per_prompt // 2
    virtual_pairs_per_prompt = physical_pairs * virtual_factor
    physical_population = num_prompts * physical_members_per_prompt
    virtual_population = physical_population * virtual_factor

    pair_ids = sample_stratified_audit_pairs(
        num_prompts=num_prompts,
        physical_members_per_prompt=physical_members_per_prompt,
        virtual_factor=virtual_factor,
        seed=7,
        epoch=11,
    )

    assert pair_ids.shape == (num_prompts * physical_pairs,)
    grouped = pair_ids.reshape(num_prompts, physical_pairs)
    for prompt_slot, prompt_pair_ids in enumerate(grouped):
        local = prompt_pair_ids - prompt_slot * virtual_pairs_per_prompt
        assert np.all((0 <= local) & (local < virtual_pairs_per_prompt))
        # Exactly one complete pair is selected from every size-16 stratum.
        npt.assert_array_equal(local // virtual_factor, np.arange(physical_pairs))
        assert np.unique(local).size == physical_pairs

    member_ids = pair_ids_to_member_ids(pair_ids).reshape(-1, 2)
    assert physical_population == 64
    assert virtual_population == 1024
    assert member_ids.size == physical_population
    assert member_ids.max() < virtual_population
    grouped_members = member_ids.reshape(num_prompts, physical_members_per_prompt)
    for prompt_slot, prompt_member_ids in enumerate(grouped_members):
        assert np.all(
            prompt_member_ids // (physical_members_per_prompt * virtual_factor)
            == prompt_slot
        )
    npt.assert_array_equal(member_ids[:, 0], 2 * pair_ids)
    npt.assert_array_equal(member_ids[:, 1], 2 * pair_ids + 1)
    assert np.all(member_ids[:, 0] % 2 == 0)
    assert np.all(member_ids[:, 1] % 2 == 1)


def test_audit_correction_is_monte_carlo_unbiased_at_one_in_sixteen():
    physical_members_per_prompt = 8
    virtual_factor = 16
    virtual_pairs = (physical_members_per_prompt // 2) * virtual_factor
    true_differences = np.linspace(-1.0, 1.0, virtual_pairs, dtype=np.float32)
    predictions = np.linspace(0.35, -0.15, virtual_pairs, dtype=np.float32)
    corrected_sum = np.zeros(virtual_pairs, dtype=np.float64)

    # Each epoch independently selects one of the sixteen entries in each of
    # four strata. Averaging the Horvitz--Thompson estimates recovers every
    # fixed virtual pair, not merely the population mean.
    trials = 8192
    for epoch in range(trials):
        audited = sample_stratified_audit_pairs(
            num_prompts=1,
            physical_members_per_prompt=physical_members_per_prompt,
            virtual_factor=virtual_factor,
            seed=123,
            epoch=epoch,
        )
        observed = _observed_member_rewards(true_differences[audited])
        corrected, observed_differences = audit_correct_pair_differences(
            predictions,
            audited,
            observed,
            virtual_factor=virtual_factor,
        )
        npt.assert_allclose(
            observed_differences, true_differences[audited], rtol=0.0, atol=1e-7
        )
        corrected_sum += corrected

    npt.assert_allclose(
        corrected_sum / trials,
        true_differences,
        rtol=0.0,
        atol=0.09,
    )


def test_zero_predictor_scaled_virtual_update_matches_physical_eggroll():
    frozen = {"rank": 1, "noise_reuse": 1, "freeze_nonlora": True}
    sigma = 0.03
    epoch = 5
    physical_population = 8
    virtual_factor = 16
    virtual_population = physical_population * virtual_factor
    virtual_pairs = virtual_population // 2
    param = jnp.asarray(
        [[0.2, -0.3], [0.7, 0.1], [-0.4, 0.9]], dtype=jnp.float32
    )
    matrix_key = jax.random.key(17)

    audited_pairs = sample_stratified_audit_pairs(
        num_prompts=1,
        physical_members_per_prompt=physical_population,
        virtual_factor=virtual_factor,
        seed=19,
        epoch=epoch,
    )
    observed_differences = np.asarray([0.7, -0.2, 1.1, -0.6], dtype=np.float32)
    observed_rewards = _observed_member_rewards(observed_differences)
    corrected, _ = audit_correct_pair_differences(
        np.zeros(virtual_pairs, dtype=np.float32),
        audited_pairs,
        observed_rewards,
        virtual_factor=virtual_factor,
    )
    virtual_utilities = pair_differences_to_member_utilities(
        corrected,
        physical_population=physical_population,
        virtual_population=virtual_population,
    )

    virtual_member_ids = jnp.arange(virtual_population, dtype=jnp.int32)
    virtual_iterinfo = (
        jnp.full((virtual_population,), epoch, dtype=jnp.int32),
        virtual_member_ids,
    )
    virtual_update = EggRoll._do_update(
        param,
        matrix_key,
        jnp.asarray(virtual_utilities),
        virtual_iterinfo,
        1,
        sigma,
        frozen,
    )

    physical_member_ids = jnp.asarray(pair_ids_to_member_ids(audited_pairs))
    physical_iterinfo = (
        jnp.full((physical_population,), epoch, dtype=jnp.int32),
        physical_member_ids,
    )
    physical_utilities = np.empty(physical_population, dtype=np.float32)
    physical_utilities[0::2] = 0.5 * observed_differences
    physical_utilities[1::2] = -0.5 * observed_differences
    physical_update = EggRoll._do_update(
        param,
        matrix_key,
        jnp.asarray(physical_utilities),
        physical_iterinfo,
        1,
        sigma,
        frozen,
    )

    npt.assert_allclose(
        np.asarray(virtual_update),
        np.asarray(physical_update),
        rtol=2e-5,
        atol=2e-6,
    )


def _manual_probe_features(
    matrices: list[jax.Array],
    keys: list[jax.Array],
    pair_ids: np.ndarray,
    epoch: int,
    frozen: dict,
    *,
    seed: int,
    probes_per_matrix: int,
) -> np.ndarray:
    columns = []
    for descriptor_index, (matrix, key) in enumerate(zip(matrices, keys)):
        descriptor_key = jax.random.fold_in(
            jax.random.key(seed), descriptor_index
        )
        out_key, in_key = jax.random.split(descriptor_key)
        out_probes = jax.random.normal(
            out_key, (matrix.shape[0], probes_per_matrix), dtype=jnp.float32
        )
        in_probes = jax.random.normal(
            in_key, (matrix.shape[1], probes_per_matrix), dtype=jnp.float32
        )
        out_probes /= jnp.linalg.norm(out_probes, axis=0, keepdims=True)
        in_probes /= jnp.linalg.norm(in_probes, axis=0, keepdims=True)
        values = []
        for pair_id in pair_ids:
            factors_out, factors_in = get_lora_update_params(
                frozen,
                1.0,
                (epoch, 2 * int(pair_id)),
                matrix,
                key,
            )
            projected_out = np.asarray(factors_out[:, 0] @ out_probes)
            projected_in = np.asarray(factors_in[:, 0] @ in_probes)
            values.append(
                (projected_out[:, None] * projected_in[None, :]).reshape(-1)
            )
        columns.append(np.asarray(values))
    return np.concatenate(columns, axis=1).astype(np.float32)


def test_mutation_features_are_exact_and_jittable_for_a_2d_matrix():
    frozen = {"rank": 1, "noise_reuse": 1}
    matrix = jnp.asarray(
        [[1.0, -0.5, 0.25], [0.75, 0.1, -0.3]], dtype=jnp.float32
    )
    matrix_key = jax.random.key(31)
    params = {"matrix": matrix}
    base_keys = {"matrix": matrix_key}
    es_map = {"matrix": 1}
    pair_ids = np.asarray([0, 3, 9], dtype=np.int32)
    epoch = 4

    feature_fn, info = build_mutation_feature_fn(
        params,
        base_keys,
        es_map,
        frozen,
        probes_per_matrix=2,
        seed=5,
    )
    expected = _manual_probe_features(
        [matrix],
        [matrix_key],
        pair_ids,
        epoch,
        frozen,
        seed=5,
        probes_per_matrix=2,
    )
    eager = np.asarray(feature_fn(params, pair_ids, epoch))
    compiled = np.asarray(jax.jit(feature_fn)(params, pair_ids, epoch))

    assert info.matrix_count == 1
    assert info.feature_dim == 4
    npt.assert_allclose(eager, expected, rtol=2e-5, atol=2e-6)
    npt.assert_allclose(compiled, expected, rtol=2e-5, atol=2e-6)


def test_mutation_features_expand_stacked_matrices_exactly_under_jit():
    frozen = {"rank": 1, "noise_reuse": 2}
    matrices = jnp.asarray(
        [
            [[1.0, 0.2, -0.1], [0.3, -0.4, 0.8]],
            [[-0.2, 0.7, 0.5], [0.6, 0.1, -0.9]],
        ],
        dtype=jnp.float32,
    )
    matrix_keys = jax.random.split(jax.random.key(41), 2)
    params = {"stacked": matrices}
    base_keys = {"stacked": matrix_keys}
    es_map = {"stacked": 1}
    pair_ids = np.asarray([1, 5], dtype=np.int32)
    epoch = 7

    feature_fn, info = build_mutation_feature_fn(
        params,
        base_keys,
        es_map,
        frozen,
        probes_per_matrix=1,
        seed=9,
    )
    expected = _manual_probe_features(
        [matrices[0], matrices[1]],
        [matrix_keys[0], matrix_keys[1]],
        pair_ids,
        epoch,
        frozen,
        seed=9,
        probes_per_matrix=1,
    )
    eager = np.asarray(feature_fn(params, pair_ids, epoch))
    compiled = np.asarray(jax.jit(feature_fn)(params, pair_ids, epoch))

    assert info.matrix_count == 2
    assert info.feature_dim == 2
    npt.assert_allclose(eager, expected, rtol=2e-5, atol=2e-6)
    npt.assert_allclose(compiled, expected, rtol=2e-5, atol=2e-6)


def test_prompt_ridge_is_frozen_by_call_order_then_becomes_ready_and_learns():
    predictor = OnlinePromptRidge(
        num_prompts=2,
        feature_dim=2,
        ridge=0.01,
        min_observations=2,
        prediction_clip=10.0,
    )
    evaluation_features = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    evaluation_prompts = np.asarray([0, 1], dtype=np.int32)

    frozen_predictions = predictor.predict(evaluation_features, evaluation_prompts)
    npt.assert_array_equal(frozen_predictions, np.zeros(2, dtype=np.float32))

    predictor.update(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.asarray([0], dtype=np.int32),
        np.asarray([2.0], dtype=np.float32),
    )
    # Prompt zero is not ready after only one observation; prompt one has none.
    npt.assert_array_equal(
        predictor.predict(evaluation_features, evaluation_prompts),
        np.zeros(2, dtype=np.float32),
    )

    predictor.update(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.asarray([0], dtype=np.int32),
        np.asarray([2.0], dtype=np.float32),
    )
    learned = predictor.predict(evaluation_features, evaluation_prompts)

    # A prediction captured before observing the generation remains an ordinary
    # immutable array; only a fresh call sees the newly fitted state.
    npt.assert_array_equal(frozen_predictions, np.zeros(2, dtype=np.float32))
    assert learned[0] > 1.9
    assert learned[1] == 0.0
    assert predictor.total_observations == 2


def test_predictive_eggroll_fitness_conversion_is_identity():
    raw_scores = jnp.asarray([0.25, -0.25, 3.0, -3.0], dtype=jnp.float32)
    converted = PredictiveEggRoll.convert_fitnesses(
        {"unused": True}, {"unused": True}, raw_scores
    )

    assert converted is raw_scores
    npt.assert_array_equal(np.asarray(converted), np.asarray(raw_scores))


class PredictiveEggRollTest(unittest.TestCase):
    """Expose the focused function tests through the repository's unittest style."""


def _as_unittest_method(test_function):
    def method(self):
        del self
        test_function()

    method.__name__ = test_function.__name__
    return method


for _name, _test in list(globals().items()):
    if _name.startswith("test_") and callable(_test):
        setattr(PredictiveEggRollTest, _name, _as_unittest_method(_test))


if __name__ == "__main__":
    unittest.main()
