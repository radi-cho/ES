"""Preview-predicted virtual populations for EGGROLL.

Only a fixed, uniform subset of antithetic pairs receives a full rollout.
Prompt-prefill features predict the remaining pairwise reward differences and
a Horvitz--Thompson residual keeps the raw ES update unbiased.
"""

from __future__ import annotations

import numpy as np

from .eggroll import EggRoll


class PredictiveEggRoll(EggRoll):
    """EGGROLL fed already-centered, audit-corrected member utilities."""

    @classmethod
    def convert_fitnesses(
        cls,
        frozen_noiser_params,
        noiser_params,
        raw_scores,
        num_episodes_list=None,
    ):
        # Population standardization is nonlinear and would destroy the
        # conditional unbiasedness of the audit-corrected estimator.
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
    """Select complete pairs with marginal probability ``1 / virtual_factor``.

    Each prompt has one audited pair in every contiguous size-``virtual_factor``
    stratum.  This preserves a static physical rollout batch and gives every
    virtual pair the same, known inclusion probability.
    """

    if num_prompts < 1:
        raise ValueError("num_prompts must be positive")
    if physical_members_per_prompt < 2 or physical_members_per_prompt % 2:
        raise ValueError("physical_members_per_prompt must be positive and even")
    if virtual_factor < 1:
        raise ValueError("virtual_factor must be positive")

    physical_pairs = physical_members_per_prompt // 2
    virtual_pairs_per_prompt = physical_pairs * virtual_factor
    selected = np.empty(num_prompts * physical_pairs, dtype=np.int32)
    for prompt_slot in range(num_prompts):
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(epoch), int(prompt_slot)])
        )
        offsets = rng.integers(0, virtual_factor, size=physical_pairs)
        local_pairs = (
            np.arange(physical_pairs, dtype=np.int64) * virtual_factor + offsets
        )
        start = prompt_slot * physical_pairs
        selected[start : start + physical_pairs] = (
            prompt_slot * virtual_pairs_per_prompt + local_pairs
        )
    return selected


def pair_ids_to_member_ids(pair_ids: np.ndarray) -> np.ndarray:
    """Expand pair IDs to adjacent positive/negative EGGROLL member IDs."""

    pair_ids = np.asarray(pair_ids, dtype=np.int64)
    return np.stack((2 * pair_ids, 2 * pair_ids + 1), axis=-1).reshape(-1).astype(
        np.int32
    )


def audit_correct_pair_differences(
    predictions: np.ndarray,
    audited_pair_ids: np.ndarray,
    observed_member_rewards: np.ndarray,
    *,
    audit_probability: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the model-assisted Horvitz--Thompson residual correction."""

    predictions = np.asarray(predictions, dtype=np.float32)
    audited_pair_ids = np.asarray(audited_pair_ids, dtype=np.int64)
    observed_member_rewards = np.asarray(observed_member_rewards, dtype=np.float32)
    if predictions.ndim != 1:
        raise ValueError("predictions must be one-dimensional")
    if observed_member_rewards.shape != (2 * audited_pair_ids.size,):
        raise ValueError("observed rewards must contain adjacent +/- members")
    if not 0.0 < audit_probability <= 1.0:
        raise ValueError("audit_probability must be in (0, 1]")
    if np.unique(audited_pair_ids).size != audited_pair_ids.size:
        raise ValueError("audited_pair_ids must be unique")
    if audited_pair_ids.size and (
        audited_pair_ids.min() < 0 or audited_pair_ids.max() >= predictions.size
    ):
        raise ValueError("audited pair ID is outside the virtual population")

    observed_differences = (
        observed_member_rewards[0::2] - observed_member_rewards[1::2]
    )
    corrected = predictions.copy()
    residual = observed_differences - predictions[audited_pair_ids]
    corrected[audited_pair_ids] += residual / float(audit_probability)
    return corrected, observed_differences


def pair_differences_to_member_utilities(
    pair_differences: np.ndarray,
    *,
    physical_population: int,
    virtual_population: int,
    reward_scale: float,
) -> np.ndarray:
    """Map raw pair differences to +/- utilities for unchanged EGGROLL.

    EGGROLL multiplies its population mean by ``sqrt(population)``.  Scaling by
    ``sqrt(physical / virtual)`` therefore preserves the step magnitude of a
    physical estimator using the same fixed reward scale while allowing all
    virtual directions to contribute.
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

    pair_utility = (
        np.sqrt(physical_population / virtual_population)
        * pair_differences
        / float(reward_scale)
    )
    member_utilities = np.empty(virtual_population, dtype=np.float32)
    member_utilities[0::2] = 0.5 * pair_utility
    member_utilities[1::2] = -0.5 * pair_utility
    return member_utilities


def make_countsketch(
    *,
    num_layers: int,
    hidden_size: int,
    sketch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create independent fixed CountSketch bucket/sign maps per layer."""

    if num_layers < 1 or hidden_size < 1 or sketch_size < 1:
        raise ValueError("CountSketch dimensions must be positive")
    buckets = np.empty((num_layers, hidden_size), dtype=np.int32)
    signs = np.empty((num_layers, hidden_size), dtype=np.float32)
    for layer_slot in range(num_layers):
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(layer_slot)])
        )
        buckets[layer_slot] = rng.integers(0, sketch_size, size=hidden_size)
        signs[layer_slot] = rng.choice(
            np.asarray([-1.0, 1.0], dtype=np.float32), size=hidden_size
        )
    return buckets, signs


class OnlineRidgeSurrogate:
    """Task-level ridge probe with lagged RMS and decayed raw statistics.

    Gram and cross-product statistics stay in raw-sketch coordinates.  The
    system is transformed only when fitting, so changing the lagged feature RMS
    never mixes incompatible historical normalizations.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        ridge: float = 10.0,
        decay: float = 0.99,
        min_observations: int = 256,
        prediction_clip: float = 1.1,
        rms_floor: float = 1e-4,
        initial_reward_scale: float = 0.5,
        reward_scale_decay: float = 0.9,
        minimum_reward_scale: float = 0.1,
    ):
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if ridge <= 0.0:
            raise ValueError("ridge must be positive")
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        if min_observations < 0:
            raise ValueError("min_observations must be nonnegative")
        if prediction_clip <= 0.0 or rms_floor <= 0.0:
            raise ValueError("prediction_clip and rms_floor must be positive")
        if initial_reward_scale <= 0.0 or minimum_reward_scale <= 0.0:
            raise ValueError("reward scales must be positive")
        if not 0.0 <= reward_scale_decay <= 1.0:
            raise ValueError("reward_scale_decay must be in [0, 1]")

        self.feature_dim = int(feature_dim)
        self.ridge = float(ridge)
        self.decay = float(decay)
        self.min_observations = int(min_observations)
        self.prediction_clip = float(prediction_clip)
        self.rms_floor = float(rms_floor)
        self.reward_scale = max(
            float(initial_reward_scale), float(minimum_reward_scale)
        )
        self.reward_scale_decay = float(reward_scale_decay)
        self.minimum_reward_scale = float(minimum_reward_scale)

        self.gram = np.zeros((feature_dim, feature_dim), dtype=np.float64)
        self.cross = np.zeros(feature_dim, dtype=np.float64)
        self.label_weight = 0.0
        self.feature_squares = np.zeros(feature_dim, dtype=np.float64)
        self.feature_weight = 0.0
        self.feature_rms = np.ones(feature_dim, dtype=np.float64)
        self.weights = np.zeros(feature_dim, dtype=np.float64)
        self.total_observations = 0

    def _features(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("features have the wrong shape")
        if not np.all(np.isfinite(features)):
            raise ValueError("features must be finite")
        return features

    @property
    def ready(self) -> bool:
        return self.total_observations >= self.min_observations

    @property
    def effective_observations(self) -> float:
        return self.label_weight

    def predict(self, raw_features: np.ndarray) -> np.ndarray:
        """Predict with weights and RMS frozen from previous iterations."""

        raw_features = self._features(raw_features)
        if not self.ready:
            return np.zeros(raw_features.shape[0], dtype=np.float32)
        normalized = raw_features / self.feature_rms
        predictions = normalized @ self.weights
        return np.clip(
            predictions, -self.prediction_clip, self.prediction_clip
        ).astype(np.float32)

    def update(
        self,
        audited_features: np.ndarray,
        targets: np.ndarray,
        *,
        rms_features: np.ndarray,
    ) -> None:
        """Update after the ES step; new state is used next iteration."""

        audited_features = self._features(audited_features)
        rms_features = self._features(rms_features)
        targets = np.asarray(targets, dtype=np.float64)
        if targets.shape != (audited_features.shape[0],):
            raise ValueError("targets have the wrong shape")
        if not np.all(np.isfinite(targets)):
            raise ValueError("targets must be finite")

        self.gram *= self.decay
        self.cross *= self.decay
        self.label_weight *= self.decay
        self.gram += audited_features.T @ audited_features
        self.cross += audited_features.T @ targets
        self.label_weight += audited_features.shape[0]
        self.total_observations += audited_features.shape[0]

        self.feature_squares *= self.decay
        self.feature_weight *= self.decay
        self.feature_squares += np.sum(rms_features * rms_features, axis=0)
        self.feature_weight += rms_features.shape[0]
        if self.feature_weight > 0.0:
            self.feature_rms = np.maximum(
                np.sqrt(self.feature_squares / self.feature_weight), self.rms_floor
            )

        if not self.ready or self.label_weight <= 0.0:
            self.weights.fill(0.0)
            return

        mean_gram = self.gram / self.label_weight
        mean_cross = self.cross / self.label_weight
        scale_outer = self.feature_rms[:, None] * self.feature_rms[None, :]
        normalized_gram = mean_gram / scale_outer
        normalized_cross = mean_cross / self.feature_rms
        system = normalized_gram + self.ridge * np.eye(self.feature_dim)
        self.weights = np.linalg.solve(system, normalized_cross)

    def update_reward_scale(self, observed_member_rewards: np.ndarray) -> None:
        """Update the scale for the next iteration from current audit rewards."""

        rewards = np.asarray(observed_member_rewards, dtype=np.float64)
        if rewards.ndim != 1 or rewards.size == 0 or not np.all(np.isfinite(rewards)):
            raise ValueError("observed_member_rewards must be a finite vector")
        observed_scale = max(float(np.std(rewards)), self.minimum_reward_scale)
        self.reward_scale = max(
            self.minimum_reward_scale,
            self.reward_scale_decay * self.reward_scale
            + (1.0 - self.reward_scale_decay) * observed_scale,
        )
