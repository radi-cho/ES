"""Predictive virtual-population utilities for two-fidelity EGGROLL.

The expensive population stays unchanged: EGGROLL fully evaluates a fixed
number of complete antithetic pairs.  Those pairs are sampled from a larger
virtual population.  A predictor supplies pairwise reward differences for the
virtual population and a Horvitz--Thompson residual from the evaluated pairs
keeps the raw ES directional estimate unbiased.

This module deliberately leaves :class:`~hyperscalees.noiser.eggroll.EggRoll`
generation and parameter aggregation untouched.  It only provides the
fitness preprocessing, mutation features, and small online predictor needed
by the experiment driver.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_flatten

from .eggroll import EggRoll


LORA = 1


class PredictiveEggRoll(EggRoll):
    """Ordinary EGGROLL fed preconstructed, audit-corrected pair utilities."""

    @classmethod
    def convert_fitnesses(
        cls,
        frozen_noiser_params,
        noiser_params,
        raw_scores,
        num_episodes_list=None,
    ):
        # ``raw_scores`` are already centered within every antithetic pair.
        # Applying the ordinary current-population variance normalization here
        # would be nonlinear in the randomized audit estimator.
        del frozen_noiser_params, noiser_params, num_episodes_list
        return raw_scores


def sample_stratified_audit_pairs(
    *,
    num_prompts: int,
    physical_members_per_prompt: int,
    virtual_factor: int,
    seed: int,
    epoch: int,
) -> np.ndarray:
    """Choose complete pairs, one uniformly from every virtual stratum.

    With eight physical members and ``virtual_factor=16``, each prompt has four
    physical pairs and 64 virtual pairs.  The sampler partitions those 64 pairs
    into four strata of 16 and chooses one pair per stratum.  Every virtual pair
    therefore has known inclusion probability ``1 / virtual_factor`` while the
    physical batch shape remains static.
    """

    if num_prompts < 1:
        raise ValueError("num_prompts must be positive")
    if physical_members_per_prompt < 2 or physical_members_per_prompt % 2:
        raise ValueError("physical_members_per_prompt must be positive and even")
    if virtual_factor < 1:
        raise ValueError("virtual_factor must be positive")

    physical_pairs = physical_members_per_prompt // 2
    virtual_pairs_per_prompt = physical_pairs * virtual_factor
    selected: list[int] = []
    for prompt_slot in range(num_prompts):
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(epoch), int(prompt_slot)])
        )
        offsets = rng.integers(0, virtual_factor, size=physical_pairs)
        local_pairs = (
            np.arange(physical_pairs, dtype=np.int64) * virtual_factor + offsets
        )
        selected.extend(
            (prompt_slot * virtual_pairs_per_prompt + local_pairs).tolist()
        )
    return np.asarray(selected, dtype=np.int32)


def pair_ids_to_member_ids(pair_ids: np.ndarray) -> np.ndarray:
    """Expand pair IDs to adjacent even/odd EGGROLL member IDs."""

    pair_ids = np.asarray(pair_ids, dtype=np.int64)
    return np.stack((2 * pair_ids, 2 * pair_ids + 1), axis=-1).reshape(-1).astype(
        np.int32
    )


def audit_correct_pair_differences(
    predictions: np.ndarray,
    audited_pair_ids: np.ndarray,
    observed_member_rewards: np.ndarray,
    *,
    virtual_factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return debiased virtual pair differences and observed audit targets."""

    predictions = np.asarray(predictions, dtype=np.float32)
    audited_pair_ids = np.asarray(audited_pair_ids, dtype=np.int64)
    observed_member_rewards = np.asarray(observed_member_rewards, dtype=np.float32)
    if observed_member_rewards.shape != (2 * audited_pair_ids.size,):
        raise ValueError("observed rewards must contain adjacent +/- members")
    if virtual_factor < 1:
        raise ValueError("virtual_factor must be positive")
    if audited_pair_ids.size and (
        audited_pair_ids.min() < 0 or audited_pair_ids.max() >= predictions.size
    ):
        raise ValueError("audited pair ID is outside the virtual population")

    observed_differences = (
        observed_member_rewards[0::2] - observed_member_rewards[1::2]
    )
    corrected = predictions.copy()
    corrected[audited_pair_ids] = predictions[audited_pair_ids] + virtual_factor * (
        observed_differences - predictions[audited_pair_ids]
    )
    return corrected, observed_differences


def pair_differences_to_member_utilities(
    pair_differences: np.ndarray,
    *,
    physical_population: int,
    virtual_population: int,
    reward_scale: float = 1.0,
) -> np.ndarray:
    """Build centered +/- utilities with baseline-equivalent step scaling.

    EGGROLL multiplies its population mean by ``sqrt(population)``.  A virtual
    population that is ``F`` times larger would therefore make the same
    Horvitz--Thompson estimate ``sqrt(F)`` times larger.  Multiplication by
    ``sqrt(physical / virtual)`` keeps the original physical-population learning
    rate while allowing predictions from every virtual direction to contribute.
    """

    pair_differences = np.asarray(pair_differences, dtype=np.float32)
    if physical_population < 2 or physical_population % 2:
        raise ValueError("physical_population must be positive and even")
    if virtual_population != 2 * pair_differences.size:
        raise ValueError("virtual_population does not match pair_differences")
    if virtual_population < physical_population:
        raise ValueError("virtual_population cannot be smaller than physical")
    if not np.isfinite(reward_scale) or reward_scale <= 0.0:
        raise ValueError("reward_scale must be finite and positive")

    population_scale = np.sqrt(physical_population / virtual_population)
    pair_utility = population_scale * pair_differences / float(reward_scale)
    member_utilities = np.empty(virtual_population, dtype=np.float32)
    member_utilities[0::2] = 0.5 * pair_utility
    member_utilities[1::2] = -0.5 * pair_utility
    return member_utilities


def compute_audit_reliability_targets(
    observed_differences: np.ndarray,
    predicted_differences: np.ndarray,
    *,
    reward_scale: float,
    prediction_clip: float,
) -> np.ndarray:
    """Build supervised reliability labels for audited virtual pairs.

    Strong, prediction-consistent pair differences are treated as reliable
    training directions.  Weak or poorly predicted audits receive low scores.
    """

    observed_differences = np.asarray(observed_differences, dtype=np.float32)
    predicted_differences = np.asarray(predicted_differences, dtype=np.float32)
    if observed_differences.shape != predicted_differences.shape:
        raise ValueError("observed and predicted differences must match")
    if not np.isfinite(reward_scale) or reward_scale <= 0.0:
        raise ValueError("reward_scale must be finite and positive")
    if prediction_clip <= 0.0:
        raise ValueError("prediction_clip must be positive")

    signal_strength = np.clip(
        np.abs(observed_differences) / float(reward_scale), 0.0, 1.0
    )
    residual = np.abs(observed_differences - predicted_differences)
    prediction_accuracy = np.clip(
        1.0 - residual / max(2.0 * float(prediction_clip), 1e-8), 0.0, 1.0
    )
    return (signal_strength * prediction_accuracy).astype(np.float32)


def apply_reliability_gating(
    pair_differences: np.ndarray,
    reliability: np.ndarray,
    prompt_ids: np.ndarray,
    *,
    top_k_per_prompt: int | None = None,
    threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Weight virtual pair utilities by predicted reliability.

    Returns the gated pair differences and the effective reliability weights
    that were applied after optional thresholding and per-prompt top-K pruning.
    """

    pair_differences = np.asarray(pair_differences, dtype=np.float32)
    reliability = np.asarray(reliability, dtype=np.float32)
    prompt_ids = np.asarray(prompt_ids, dtype=np.int64)
    if pair_differences.shape != reliability.shape:
        raise ValueError("pair_differences and reliability must match")
    if prompt_ids.shape != pair_differences.shape:
        raise ValueError("prompt_ids must align with pair_differences")
    if top_k_per_prompt is not None and top_k_per_prompt < 1:
        raise ValueError("top_k_per_prompt must be positive when provided")

    effective = np.where(reliability >= threshold, reliability, 0.0).astype(
        np.float32
    )
    if top_k_per_prompt is not None:
        keep_mask = np.zeros(pair_differences.size, dtype=bool)
        for prompt_id in np.unique(prompt_ids):
            prompt_indices = np.flatnonzero(prompt_ids == prompt_id)
            if prompt_indices.size == 0:
                continue
            keep_count = min(int(top_k_per_prompt), prompt_indices.size)
            ranked = prompt_indices[
                np.argsort(-effective[prompt_indices], kind="stable")[:keep_count]
            ]
            keep_mask[ranked] = True
        effective = np.where(keep_mask, effective, 0.0).astype(np.float32)

    gated = (pair_differences * effective).astype(np.float32)
    return gated, effective


@dataclass(frozen=True)
class MutationFeatureInfo:
    feature_dim: int
    matrix_count: int
    probes_per_matrix: int


def build_mutation_feature_fn(
    params_example,
    base_keys,
    es_map,
    frozen_noiser_params,
    *,
    probes_per_matrix: int = 1,
    seed: int = 0,
) -> tuple[Callable, MutationFeatureInfo]:
    """Build pair features from the exact rank-one factors used by EGGROLL.

    Every trainable matrix contributes a small grid of fixed bilinear random
    projections ``(r_out.T @ A) * (r_in.T @ B)``.  These are signed linear
    measurements of the actual additive matrix ``A @ B.T``; unlike a numeric
    seed embedding, they can generalize to unseen perturbation IDs.  Their cost
    is linear in the two factor-vector lengths, not in the matrix size.

    Scanned parameter leaves are expanded over their leading axes, matching the
    per-layer keys used by the model's scan and EGGROLL's update reconstruction.
    """

    if probes_per_matrix < 1:
        raise ValueError("probes_per_matrix must be positive")
    if int(frozen_noiser_params.get("rank", 1)) != 1:
        raise ValueError("predictive mutation features currently require rank=1")

    flat_params, params_treedef = tree_flatten(params_example)
    flat_keys, _ = tree_flatten(base_keys)
    flat_es, _ = tree_flatten(es_map)
    descriptors: list[tuple[int, int, tuple[int, int], jax.Array, jax.Array]] = []
    projection_root = jax.random.key(seed)
    descriptor_index = 0

    for leaf_index, (param, key, classification) in enumerate(
        zip(flat_params, flat_keys, flat_es)
    ):
        if int(classification) != LORA:
            continue
        if param.ndim < 2:
            raise ValueError("rank-one EGGROLL matrix leaves must be at least 2D")
        leading_count = prod(param.shape[:-2]) if param.ndim > 2 else 1
        if key.size != leading_count:
            raise ValueError(
                "predictive features require one EGGROLL key per scanned matrix"
            )
        flat_key = key.reshape((leading_count,))
        matrix_shape = (int(param.shape[-2]), int(param.shape[-1]))
        for matrix_index in range(leading_count):
            probe_key = jax.random.fold_in(projection_root, descriptor_index)
            descriptors.append(
                (leaf_index, matrix_index, matrix_shape, flat_key[matrix_index], probe_key)
            )
            descriptor_index += 1

    if not descriptors:
        raise ValueError("no rank-one matrix parameters were found")

    noise_reuse = int(frozen_noiser_params["noise_reuse"])
    feature_dim = len(descriptors) * probes_per_matrix * probes_per_matrix

    def mutation_features(params, pair_ids, epoch):
        dynamic_flat_params, dynamic_treedef = tree_flatten(params)
        if dynamic_treedef != params_treedef:
            raise ValueError("parameter tree changed after feature compilation")
        pair_ids_array = jnp.asarray(pair_ids, dtype=jnp.int32)
        true_epoch = (
            jnp.asarray(0, dtype=jnp.int32)
            if noise_reuse == 0
            else jnp.asarray(epoch, dtype=jnp.int32) // noise_reuse
        )
        columns = []

        for leaf_index, _matrix_index, (out_dim, in_dim), matrix_key, probe_key in descriptors:
            leaf = dynamic_flat_params[leaf_index]

            def sample_factors(pair_id):
                key = jax.random.fold_in(
                    jax.random.fold_in(matrix_key, true_epoch), pair_id
                )
                factors = jax.random.normal(
                    key,
                    (out_dim + in_dim, 1),
                    dtype=leaf.dtype,
                ).astype(jnp.float32)
                return factors[in_dim:, 0], factors[:in_dim, 0]

            factors_out, factors_in = jax.vmap(sample_factors)(pair_ids_array)
            out_key, in_key = jax.random.split(probe_key)
            out_probes = jax.random.normal(
                out_key, (out_dim, probes_per_matrix), dtype=jnp.float32
            )
            in_probes = jax.random.normal(
                in_key, (in_dim, probes_per_matrix), dtype=jnp.float32
            )
            out_probes = out_probes / jnp.maximum(
                jnp.linalg.norm(out_probes, axis=0, keepdims=True), 1e-6
            )
            in_probes = in_probes / jnp.maximum(
                jnp.linalg.norm(in_probes, axis=0, keepdims=True), 1e-6
            )
            projected_out = factors_out @ out_probes
            projected_in = factors_in @ in_probes
            bilinear = (
                projected_out[:, :, None] * projected_in[:, None, :]
            ).reshape((pair_ids_array.size, -1))
            columns.extend(
                bilinear[:, feature_index]
                for feature_index in range(probes_per_matrix**2)
            )

        return jnp.stack(columns, axis=-1)

    return mutation_features, MutationFeatureInfo(
        feature_dim=feature_dim,
        matrix_count=len(descriptors),
        probes_per_matrix=probes_per_matrix,
    )


class OnlinePromptRidge:
    """Prompt-conditioned ridge predictor fitted on a bounded audit replay."""

    def __init__(
        self,
        *,
        num_prompts: int,
        feature_dim: int,
        ridge: float = 32.0,
        min_observations: int = 16,
        prediction_clip: float = 1.0,
        inactive_prediction: float = 0.0,
        warmup_ramp: bool = False,
        global_fallback: bool = False,
        initial_reward_scale: float = 0.5,
        reward_scale_decay: float = 0.9,
        minimum_reward_scale: float = 0.1,
        replay_capacity: int = 64,
    ):
        if num_prompts < 1 or feature_dim < 1:
            raise ValueError("num_prompts and feature_dim must be positive")
        if (
            ridge <= 0.0
            or min_observations < 0
            or prediction_clip <= 0.0
            or replay_capacity < 1
        ):
            raise ValueError("invalid predictor regularization")
        if not 0.0 <= reward_scale_decay < 1.0:
            raise ValueError("reward_scale_decay must be in [0, 1)")
        if initial_reward_scale <= 0.0 or minimum_reward_scale <= 0.0:
            raise ValueError("reward scales must be positive")

        self.num_prompts = int(num_prompts)
        self.feature_dim = int(feature_dim)
        self.ridge = float(ridge)
        self.min_observations = int(min_observations)
        self.prediction_clip = float(prediction_clip)
        self.inactive_prediction = float(inactive_prediction)
        self.warmup_ramp = bool(warmup_ramp)
        self.global_fallback = bool(global_fallback)
        self.reward_scale_decay = float(reward_scale_decay)
        self.minimum_reward_scale = float(minimum_reward_scale)
        self.replay_capacity = int(replay_capacity)
        self.reward_scale = float(initial_reward_scale)
        self.weights = np.zeros(
            (self.num_prompts, self.feature_dim), dtype=np.float32
        )
        self.replay_features = np.zeros(
            (self.num_prompts, self.replay_capacity, self.feature_dim),
            dtype=np.float32,
        )
        self.replay_targets = np.zeros(
            (self.num_prompts, self.replay_capacity), dtype=np.float32
        )
        self.observations = np.zeros(self.num_prompts, dtype=np.int64)
        if self.global_fallback:
            self.global_weights = np.zeros(self.feature_dim, dtype=np.float32)
            self.global_replay_features = np.zeros(
                (self.replay_capacity, self.feature_dim), dtype=np.float32
            )
            self.global_replay_targets = np.zeros(
                self.replay_capacity, dtype=np.float32
            )
            self.global_observations = 0
        else:
            self.global_weights = None
            self.global_replay_features = None
            self.global_replay_targets = None
            self.global_observations = 0

    def _prompt_ramp(self, prompt_ids: np.ndarray) -> np.ndarray:
        if self.min_observations <= 0:
            return np.ones(prompt_ids.shape, dtype=np.float64)
        return np.minimum(
            1.0,
            self.observations[prompt_ids].astype(np.float64)
            / float(self.min_observations),
        )

    def _fit_prompt_weights(self, prompt_id: int) -> None:
        replay_size = int(min(self.observations[prompt_id], self.replay_capacity))
        replay_x = self.replay_features[prompt_id, :replay_size].astype(np.float64)
        replay_y = self.replay_targets[prompt_id, :replay_size].astype(np.float64)
        kernel = replay_x @ replay_x.T
        kernel.flat[:: replay_size + 1] += self.ridge
        dual_weights = np.linalg.solve(kernel, replay_y)
        self.weights[prompt_id] = (replay_x.T @ dual_weights).astype(np.float32)

    def _fit_global_weights(self) -> None:
        if not self.global_fallback or self.global_observations <= 0:
            return
        replay_size = int(min(self.global_observations, self.replay_capacity))
        replay_x = self.global_replay_features[:replay_size].astype(np.float64)
        replay_y = self.global_replay_targets[:replay_size].astype(np.float64)
        kernel = replay_x @ replay_x.T
        kernel.flat[:: replay_size + 1] += self.ridge
        dual_weights = np.linalg.solve(kernel, replay_y)
        self.global_weights = (replay_x.T @ dual_weights).astype(np.float32)

    def _validate_inputs(self, features, prompt_ids):
        features = np.asarray(features, dtype=np.float64)
        prompt_ids = np.asarray(prompt_ids, dtype=np.int64)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("features have the wrong shape")
        if prompt_ids.shape != (features.shape[0],):
            raise ValueError("prompt_ids have the wrong shape")
        if prompt_ids.size and (
            prompt_ids.min() < 0 or prompt_ids.max() >= self.num_prompts
        ):
            raise ValueError("prompt ID is outside the predictor table")
        return features, prompt_ids

    def predict(self, features, prompt_ids) -> np.ndarray:
        features, prompt_ids = self._validate_inputs(features, prompt_ids)
        prompt_predictions = np.einsum("nf,nf->n", features, self.weights[prompt_ids])
        if self.global_fallback and self.global_weights is not None:
            global_predictions = features @ self.global_weights
            ramp = self._prompt_ramp(prompt_ids)
            predictions = ramp * prompt_predictions + (1.0 - ramp) * global_predictions
        elif self.warmup_ramp:
            ramp = self._prompt_ramp(prompt_ids)
            predictions = (
                ramp * prompt_predictions + (1.0 - ramp) * self.inactive_prediction
            )
        else:
            ready = self.observations[prompt_ids] >= self.min_observations
            predictions = np.where(ready, prompt_predictions, self.inactive_prediction)
        return np.clip(
            predictions, -self.prediction_clip, self.prediction_clip
        ).astype(np.float32)

    def update(self, features, prompt_ids, targets) -> None:
        features, prompt_ids = self._validate_inputs(features, prompt_ids)
        targets = np.asarray(targets, dtype=np.float64)
        if targets.shape != (features.shape[0],):
            raise ValueError("targets have the wrong shape")
        for prompt_id in np.unique(prompt_ids):
            mask = prompt_ids == prompt_id
            for feature, target in zip(features[mask], targets[mask]):
                slot = int(self.observations[prompt_id] % self.replay_capacity)
                self.replay_features[prompt_id, slot] = feature
                self.replay_targets[prompt_id, slot] = target
                self.observations[prompt_id] += 1
            self._fit_prompt_weights(int(prompt_id))

        if self.global_fallback:
            for feature, target in zip(features, targets):
                slot = int(self.global_observations % self.replay_capacity)
                self.global_replay_features[slot] = feature
                self.global_replay_targets[slot] = target
                self.global_observations += 1
            self._fit_global_weights()

    def update_reward_scale(self, observed_member_rewards) -> None:
        observed_member_rewards = np.asarray(
            observed_member_rewards, dtype=np.float64
        )
        observed_scale = max(
            float(np.std(observed_member_rewards)), self.minimum_reward_scale
        )
        self.reward_scale = max(
            self.minimum_reward_scale,
            self.reward_scale_decay * self.reward_scale
            + (1.0 - self.reward_scale_decay) * observed_scale,
        )

    @property
    def total_observations(self) -> int:
        return int(self.observations.sum())
