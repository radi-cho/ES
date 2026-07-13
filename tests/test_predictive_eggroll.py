"""Focused CPU tests for preview-predicted virtual-population EGGROLL."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from hyperscalees.noiser.eggroll import EggRoll
from hyperscalees.noiser.predictive_eggroll import (
    OnlineRidgeSurrogate,
    PredictiveEggRoll,
    audit_correct_pair_differences,
    make_countsketch,
    make_online_surrogate,
    pair_differences_to_member_utilities,
    pair_ids_to_member_ids,
    prompt_center_features,
    sample_stratified_audit_pairs,
)


def _member_rewards(pair_differences: np.ndarray) -> np.ndarray:
    """Construct adjacent +/- member rewards with the requested differences."""

    pair_differences = np.asarray(pair_differences, dtype=np.float32)
    rewards = np.empty(2 * pair_differences.size, dtype=np.float32)
    rewards[0::2] = 0.5 * pair_differences
    rewards[1::2] = -0.5 * pair_differences
    return rewards


def test_target_layout_uses_global_virtual_pair_and_member_ids():
    num_prompts = 8
    physical_members_per_prompt = 8
    virtual_factor = 16
    physical_pairs_per_prompt = physical_members_per_prompt // 2
    virtual_pairs_per_prompt = physical_pairs_per_prompt * virtual_factor
    physical_population = num_prompts * physical_members_per_prompt
    virtual_population = physical_population * virtual_factor

    audited_pairs = sample_stratified_audit_pairs(
        num_prompts=num_prompts,
        physical_members_per_prompt=physical_members_per_prompt,
        virtual_factor=virtual_factor,
        seed=7,
        epoch=11,
    )

    assert physical_population == 64
    assert virtual_population == 1024
    assert virtual_population // 2 == 512
    assert audited_pairs.shape == (32,)

    prompt_pairs = audited_pairs.reshape(num_prompts, physical_pairs_per_prompt)
    for prompt_slot, global_pair_ids in enumerate(prompt_pairs):
        local_pair_ids = global_pair_ids - prompt_slot * virtual_pairs_per_prompt
        assert np.all((0 <= local_pair_ids) & (local_pair_ids < 64))
        # There is exactly one audited pair in each contiguous size-16 stratum.
        npt.assert_array_equal(
            local_pair_ids // virtual_factor,
            np.arange(physical_pairs_per_prompt),
        )

    audited_members = pair_ids_to_member_ids(audited_pairs).reshape(-1, 2)
    assert audited_members.size == physical_population
    assert audited_members.max() < virtual_population
    npt.assert_array_equal(audited_members[:, 0], 2 * audited_pairs)
    npt.assert_array_equal(audited_members[:, 1], 2 * audited_pairs + 1)
    assert np.all(audited_members[:, 0] % 2 == 0)
    assert np.all(audited_members[:, 1] % 2 == 1)

    prompt_members = audited_members.reshape(
        num_prompts, physical_members_per_prompt
    )
    for prompt_slot, global_member_ids in enumerate(prompt_members):
        assert np.all(global_member_ids // 128 == prompt_slot)


def test_exhaustive_ht_correction_recovers_every_pair():
    """Every pair is audited once across the 16 exhaustive stratum offsets."""

    virtual_factor = 16
    virtual_pairs = 4 * virtual_factor
    truth = np.linspace(-1.1, 1.1, virtual_pairs, dtype=np.float32)
    predictions = (
        0.35 * np.cos(np.arange(virtual_pairs, dtype=np.float32))
    ).astype(np.float32)
    corrected_sum = np.zeros(virtual_pairs, dtype=np.float64)

    for offset in range(virtual_factor):
        audited = (
            np.arange(4, dtype=np.int32) * virtual_factor + offset
        )
        corrected, observed = audit_correct_pair_differences(
            predictions,
            audited,
            _member_rewards(truth[audited]),
            audit_probability=1.0 / virtual_factor,
        )
        npt.assert_allclose(observed, truth[audited], rtol=0.0, atol=1e-7)
        corrected_sum += corrected

    npt.assert_allclose(
        corrected_sum / virtual_factor,
        truth,
        rtol=0.0,
        atol=2e-6,
    )


def test_sampled_ht_correction_is_monte_carlo_unbiased():
    virtual_factor = 16
    virtual_pairs = 4 * virtual_factor
    truth = np.linspace(-1.0, 1.0, virtual_pairs, dtype=np.float32)
    predictions = np.linspace(0.4, -0.2, virtual_pairs, dtype=np.float32)
    corrected_sum = np.zeros(virtual_pairs, dtype=np.float64)

    trials = 8192
    for epoch in range(trials):
        audited = sample_stratified_audit_pairs(
            num_prompts=1,
            physical_members_per_prompt=8,
            virtual_factor=virtual_factor,
            seed=123,
            epoch=epoch,
        )
        corrected, _ = audit_correct_pair_differences(
            predictions,
            audited,
            _member_rewards(truth[audited]),
            audit_probability=1.0 / virtual_factor,
        )
        corrected_sum += corrected

    npt.assert_allclose(
        corrected_sum / trials,
        truth,
        rtol=0.0,
        atol=0.09,
    )


def test_ht_residual_is_not_clipped_after_prediction():
    predictions = np.asarray([1.1, 0.0], dtype=np.float32)
    # Both rewards are valid Countdown rewards, but their inverse-probability
    # residual must be allowed outside the raw reward-difference range.
    observed_member_rewards = np.asarray([0.0, 1.1], dtype=np.float32)

    corrected, observed = audit_correct_pair_differences(
        predictions,
        np.asarray([0], dtype=np.int32),
        observed_member_rewards,
        audit_probability=1.0 / 16.0,
    )

    npt.assert_allclose(observed, [-1.1], rtol=0.0, atol=1e-7)
    npt.assert_allclose(corrected[0], 1.1 + 16.0 * (-2.2), atol=2e-6)
    assert abs(float(corrected[0])) > 1.1
    assert corrected[1] == predictions[1]


def test_zero_predictor_virtual_update_equals_selected_physical_update():
    frozen = {"rank": 1, "noise_reuse": 1, "freeze_nonlora": True}
    sigma = 0.03
    epoch = 5
    physical_population = 8
    virtual_factor = 16
    virtual_population = physical_population * virtual_factor
    virtual_pairs = virtual_population // 2
    reward_scale = 0.7
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
    corrected, _ = audit_correct_pair_differences(
        np.zeros(virtual_pairs, dtype=np.float32),
        audited_pairs,
        _member_rewards(observed_differences),
        audit_probability=1.0 / virtual_factor,
    )
    virtual_utilities = pair_differences_to_member_utilities(
        corrected,
        physical_population=physical_population,
        virtual_population=virtual_population,
        reward_scale=reward_scale,
    )

    virtual_iterinfo = (
        jnp.full((virtual_population,), epoch, dtype=jnp.int32),
        jnp.arange(virtual_population, dtype=jnp.int32),
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
    physical_utilities[0::2] = 0.5 * observed_differences / reward_scale
    physical_utilities[1::2] = -0.5 * observed_differences / reward_scale
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


def test_predictive_eggroll_fitness_conversion_is_identity():
    raw_scores = jnp.asarray([0.25, -0.25, 3.0, -3.0], dtype=jnp.float32)
    converted = PredictiveEggRoll.convert_fitnesses(
        {"unused": True}, {"unused": True}, raw_scores
    )

    assert converted is raw_scores
    npt.assert_array_equal(np.asarray(converted), np.asarray(raw_scores))


def test_countsketch_is_fixed_reproducible_and_layer_independent():
    first_buckets, first_signs = make_countsketch(
        num_layers=2, hidden_size=2048, sketch_size=128, seed=31
    )
    repeated_buckets, repeated_signs = make_countsketch(
        num_layers=2, hidden_size=2048, sketch_size=128, seed=31
    )
    other_buckets, other_signs = make_countsketch(
        num_layers=2, hidden_size=2048, sketch_size=128, seed=32
    )

    npt.assert_array_equal(first_buckets, repeated_buckets)
    npt.assert_array_equal(first_signs, repeated_signs)
    assert not np.array_equal(first_buckets[0], first_buckets[1])
    assert not np.array_equal(first_signs[0], first_signs[1])
    assert not np.array_equal(first_buckets, other_buckets)
    assert not np.array_equal(first_signs, other_signs)
    assert first_buckets.min() >= 0
    assert first_buckets.max() < 128
    npt.assert_array_equal(np.unique(first_signs), [-1.0, 1.0])


def test_prompt_centering_keeps_groups_separate_and_removes_offsets():
    features = np.asarray(
        [
            [11.0, 2.0],
            [13.0, 4.0],
            [15.0, 6.0],
            [-7.0, 10.0],
            [-5.0, 12.0],
            [-3.0, 14.0],
        ],
        dtype=np.float32,
    )
    centered = prompt_center_features(
        features, num_prompts=2, pairs_per_prompt=3
    ).reshape(2, 3, 2)

    npt.assert_allclose(np.mean(centered, axis=1), 0.0, atol=1e-7)
    npt.assert_allclose(centered[0, :, 0], [-2.0, 0.0, 2.0])
    npt.assert_allclose(centered[1, :, 0], [-2.0, 0.0, 2.0])

    shifted = features.copy()
    shifted[:3] += np.asarray([100.0, -50.0], dtype=np.float32)
    shifted[3:] += np.asarray([-20.0, 80.0], dtype=np.float32)
    npt.assert_allclose(
        prompt_center_features(shifted, num_prompts=2, pairs_per_prompt=3),
        centered.reshape(6, 2),
        atol=1e-6,
    )


def test_surrogate_factory_exposes_ridge_and_rejects_unknown_models():
    predictor = make_online_surrogate("ridge", feature_dim=2)
    assert isinstance(predictor, OnlineRidgeSurrogate)
    with npt.assert_raises_regex(ValueError, "unknown predictive surrogate"):
        make_online_surrogate("mlp", feature_dim=2)


def test_prequential_calibration_is_lagged_shrunk_and_bounded():
    predictor = OnlineRidgeSurrogate(
        feature_dim=1,
        ridge=0.01,
        min_observations=1,
        prediction_clip=10.0,
        calibration_decay=1.0,
        calibration_max_scale=1.0,
        calibration_min_observations=2,
        calibration_prior_observations=2.0,
        calibrate_predictions=True,
    )
    predictor.update(
        np.asarray([[1.0]], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
        rms_features=np.asarray([[1.0]], dtype=np.float32),
    )
    evaluation = np.asarray([[1.0], [-1.0]], dtype=np.float32)
    current, raw = predictor.predict_with_uncalibrated(evaluation)

    # No audited prediction/label pairs have calibrated this fitted model yet.
    assert predictor.calibration_scale == 0.0
    npt.assert_array_equal(current, np.zeros(2, dtype=np.float32))
    assert raw[0] > 0.0

    predictor.update(
        evaluation,
        np.asarray([1.0, -1.0], dtype=np.float32),
        rms_features=evaluation,
        evaluated_predictions=current,
        evaluated_uncalibrated_predictions=raw,
    )

    # The just-produced current array remains frozen; only later calls use the
    # positive, confidence-shrunk calibration coefficient.
    npt.assert_array_equal(current, np.zeros(2, dtype=np.float32))
    assert 0.0 < predictor.calibration_confidence < 1.0
    assert 0.0 < predictor.calibration_scale < 1.0
    later = predictor.predict(evaluation)
    assert later[0] > 0.0
    npt.assert_allclose(later[0], -later[1], atol=1e-7)


def test_negative_prequential_covariance_disables_control_term():
    predictor = OnlineRidgeSurrogate(
        feature_dim=1,
        min_observations=0,
        calibration_min_observations=2,
        calibration_prior_observations=0.0,
        calibrate_predictions=True,
    )
    features = np.asarray([[1.0], [-1.0]], dtype=np.float32)
    raw = np.asarray([1.0, -1.0], dtype=np.float32)
    predictor.update(
        features,
        np.asarray([-1.0, 1.0], dtype=np.float32),
        rms_features=features,
        evaluated_predictions=np.zeros(2, dtype=np.float32),
        evaluated_uncalibrated_predictions=raw,
    )

    assert predictor.calibration_slope == 0.0
    assert predictor.calibration_scale == 0.0


def test_zero_prequential_predictions_do_not_create_calibration_confidence():
    predictor = OnlineRidgeSurrogate(
        feature_dim=1,
        min_observations=0,
        calibration_min_observations=1,
        calibration_prior_observations=0.0,
        calibrate_predictions=True,
    )
    features = np.asarray([[1.0], [-1.0]], dtype=np.float32)
    zeros = np.zeros(2, dtype=np.float32)
    predictor.update(
        features,
        np.asarray([1.0, -1.0], dtype=np.float32),
        rms_features=features,
        evaluated_predictions=zeros,
        evaluated_uncalibrated_predictions=zeros,
    )

    assert predictor.calibration_observations == 0
    assert predictor.calibration_effective_observations == 0.0
    assert predictor.calibration_scale == 0.0


def test_ridge_gate_no_intercept_symmetric_clipping_and_decay():
    predictor = OnlineRidgeSurrogate(
        feature_dim=2,
        ridge=1e-3,
        decay=0.5,
        min_observations=2,
        prediction_clip=0.5,
        rms_floor=1e-6,
    )
    evaluation = np.asarray(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]], dtype=np.float32
    )

    npt.assert_array_equal(predictor.predict(evaluation), np.zeros(3))
    predictor.update(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.asarray([10.0], dtype=np.float32),
        rms_features=np.asarray([[1.0, 2.0]], dtype=np.float32),
    )
    assert not predictor.ready
    npt.assert_array_equal(predictor.predict(evaluation), np.zeros(3))

    predictor.update(
        np.asarray([[2.0, 0.0]], dtype=np.float32),
        np.asarray([20.0], dtype=np.float32),
        rms_features=np.asarray([[3.0, 4.0]], dtype=np.float32),
    )

    assert predictor.ready
    assert predictor.total_observations == 2
    npt.assert_allclose(predictor.gram, [[4.5, 0.0], [0.0, 0.0]])
    npt.assert_allclose(predictor.cross, [40.0 + 5.0, 0.0])
    npt.assert_allclose(predictor.effective_observations, 1.5)
    npt.assert_allclose(predictor.feature_squares, [9.5, 18.0])
    npt.assert_allclose(predictor.feature_weight, 1.5)

    predictions = predictor.predict(evaluation)
    npt.assert_allclose(predictions, [0.5, -0.5, 0.0], atol=1e-7)


def test_raw_decayed_statistics_match_direct_current_rms_ridge():
    ridge = 0.7
    decay = 0.6
    predictor = OnlineRidgeSurrogate(
        feature_dim=3,
        ridge=ridge,
        decay=decay,
        min_observations=0,
        prediction_clip=100.0,
        rms_floor=1e-8,
    )
    x1 = np.asarray(
        [[1.0, 2.0, -1.0], [0.5, -1.0, 3.0]], dtype=np.float64
    )
    y1 = np.asarray([0.75, -1.25], dtype=np.float64)
    rms1 = np.asarray(
        [[2.0, 1.0, 0.5], [1.0, 3.0, 2.0], [4.0, 2.0, 1.0]],
        dtype=np.float64,
    )
    x2 = np.asarray(
        [[-2.0, 0.25, 1.5], [1.25, -0.5, 0.75]], dtype=np.float64
    )
    y2 = np.asarray([2.0, -0.5], dtype=np.float64)
    rms2 = np.asarray(
        [[1.5, 2.5, 3.0], [0.75, 1.25, 2.25]], dtype=np.float64
    )

    predictor.update(x1, y1, rms_features=rms1)
    predictor.update(x2, y2, rms_features=rms2)

    gram = decay * (x1.T @ x1) + x2.T @ x2
    cross = decay * (x1.T @ y1) + x2.T @ y2
    label_weight = decay * x1.shape[0] + x2.shape[0]
    feature_squares = decay * np.sum(rms1 * rms1, axis=0) + np.sum(
        rms2 * rms2, axis=0
    )
    feature_weight = decay * rms1.shape[0] + rms2.shape[0]
    current_rms = np.sqrt(feature_squares / feature_weight)

    normalized_gram = (gram / label_weight) / (
        current_rms[:, None] * current_rms[None, :]
    )
    normalized_cross = (cross / label_weight) / current_rms
    expected_weights = np.linalg.solve(
        normalized_gram + ridge * np.eye(3), normalized_cross
    )

    npt.assert_allclose(predictor.gram, gram, rtol=0.0, atol=1e-12)
    npt.assert_allclose(predictor.cross, cross, rtol=0.0, atol=1e-12)
    npt.assert_allclose(predictor.feature_rms, current_rms, rtol=0.0, atol=1e-12)
    npt.assert_allclose(
        predictor.weights, expected_weights, rtol=1e-12, atol=1e-12
    )

    evaluation = np.asarray([[0.2, -0.4, 1.1], [-1.0, 0.5, 0.25]])
    expected_predictions = (evaluation / current_rms) @ expected_weights
    npt.assert_allclose(
        predictor.predict(evaluation),
        expected_predictions,
        rtol=1e-6,
        atol=1e-6,
    )


def test_predictions_are_frozen_until_an_explicit_post_step_update():
    predictor = OnlineRidgeSurrogate(
        feature_dim=1,
        ridge=0.01,
        decay=1.0,
        min_observations=1,
        prediction_clip=10.0,
    )
    evaluation = np.asarray([[1.0], [-1.0]], dtype=np.float32)

    current_iteration_predictions = predictor.predict(evaluation)
    npt.assert_array_equal(
        current_iteration_predictions, np.zeros(2, dtype=np.float32)
    )

    predictor.update(
        np.asarray([[1.0]], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
        rms_features=np.asarray([[1.0], [2.0]], dtype=np.float32),
    )
    next_iteration_predictions = predictor.predict(evaluation)

    # The array captured for the current update cannot be retroactively changed.
    npt.assert_array_equal(
        current_iteration_predictions, np.zeros(2, dtype=np.float32)
    )
    assert next_iteration_predictions[0] > 0.0
    npt.assert_allclose(
        next_iteration_predictions[0], -next_iteration_predictions[1], atol=1e-7
    )


def test_reward_scale_is_lagged_and_floored_for_the_next_iteration():
    predictor = OnlineRidgeSurrogate(
        feature_dim=1,
        initial_reward_scale=2.0,
        reward_scale_decay=0.5,
        minimum_reward_scale=0.2,
    )
    frozen_current_scale = predictor.reward_scale

    predictor.update_reward_scale(np.asarray([0.0, 2.0], dtype=np.float32))

    assert frozen_current_scale == 2.0
    npt.assert_allclose(predictor.reward_scale, 1.5)
    predictor.update_reward_scale(np.zeros(4, dtype=np.float32))
    npt.assert_allclose(predictor.reward_scale, 0.85)


class PredictiveEggRollTest(unittest.TestCase):
    """Expose function tests through the repository's unittest style."""


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
