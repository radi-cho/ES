"""Adaptive surrogate-fitness EGGROLL.

Each epoch uses a mixture of rollout fitness and model-predicted fitness.
The mixture fraction for epoch ``t`` is the trust score measured on epoch
``t - 1`` when both rollout and prediction were available on the audited
subset.
"""

from __future__ import annotations

import numpy as np

from .eggroll import EggRoll


class AdaptiveSurrogateEggRoll(EggRoll):
    """Ordinary EGGROLL with externally assembled member fitnesses."""

    @classmethod
    def convert_fitnesses(
        cls,
        frozen_noiser_params,
        noiser_params,
        raw_scores,
        num_episodes_list=None,
    ):
        del frozen_noiser_params, noiser_params, num_episodes_list
        return raw_scores


def sample_surrogate_pair_mask(
    *,
    num_prompts: int,
    pairs_per_prompt: int,
    surrogate_fraction: float,
    seed: int,
    epoch: int,
) -> np.ndarray:
    """Return a boolean mask over flattened pair IDs.

    ``True`` means the pair's fitness will come from the surrogate model.
    Selection is stratified per prompt so every prompt keeps the same
    surrogate fraction up to rounding.
    """

    if num_prompts < 1 or pairs_per_prompt < 1:
        raise ValueError("num_prompts and pairs_per_prompt must be positive")
    if not 0.0 <= surrogate_fraction <= 1.0:
        raise ValueError("surrogate_fraction must be in [0, 1]")

    total_pairs = num_prompts * pairs_per_prompt
    mask = np.zeros(total_pairs, dtype=bool)
    surrogate_per_prompt = int(
        np.floor(surrogate_fraction * pairs_per_prompt + 1e-8)
    )
    if surrogate_per_prompt == 0:
        return mask
    if surrogate_per_prompt >= pairs_per_prompt:
        return np.ones(total_pairs, dtype=bool)

    for prompt_slot in range(num_prompts):
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(epoch), int(prompt_slot)])
        )
        local = np.zeros(pairs_per_prompt, dtype=bool)
        chosen = rng.choice(
            pairs_per_prompt, size=surrogate_per_prompt, replace=False
        )
        local[chosen] = True
        start = prompt_slot * pairs_per_prompt
        mask[start : start + pairs_per_prompt] = local
    return mask


def pair_ids_to_member_ids(pair_ids: np.ndarray) -> np.ndarray:
    pair_ids = np.asarray(pair_ids, dtype=np.int64)
    return np.stack((2 * pair_ids, 2 * pair_ids + 1), axis=-1).reshape(-1).astype(
        np.int32
    )


def pair_differences_to_member_fitness(
    pair_differences: np.ndarray,
) -> np.ndarray:
    pair_differences = np.asarray(pair_differences, dtype=np.float32)
    member_fitness = np.empty(2 * pair_differences.size, dtype=np.float32)
    member_fitness[0::2] = 0.5 * pair_differences
    member_fitness[1::2] = -0.5 * pair_differences
    return member_fitness


def observed_member_fitness_to_pair_differences(
    member_fitness: np.ndarray,
) -> np.ndarray:
    member_fitness = np.asarray(member_fitness, dtype=np.float32)
    if member_fitness.size % 2:
        raise ValueError("member_fitness must contain antithetic pairs")
    return member_fitness[0::2] - member_fitness[1::2]


def compute_trust_score(
    predicted_pair_differences: np.ndarray,
    observed_pair_differences: np.ndarray,
) -> float:
    """Map last epoch's surrogate quality to next epoch's surrogate fraction."""

    predicted = np.asarray(predicted_pair_differences, dtype=np.float64)
    observed = np.asarray(observed_pair_differences, dtype=np.float64)
    if predicted.shape != observed.shape:
        raise ValueError("predicted and observed pair differences must match")
    if predicted.size < 2:
        return 0.0
    pred_std = float(np.std(predicted))
    obs_std = float(np.std(observed))
    if pred_std <= 1e-8 or obs_std <= 1e-8:
        return 0.0
    correlation = float(np.corrcoef(predicted, observed)[0, 1])
    if not np.isfinite(correlation):
        return 0.0
    return float(np.clip(max(0.0, correlation), 0.0, 1.0))


def assemble_member_fitness(
    *,
    total_members: int,
    predicted_member_fitness: np.ndarray,
    rollout_member_fitness: np.ndarray,
    rollout_member_ids: np.ndarray,
    surrogate_pair_mask: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    """Fill the full member fitness vector and count rollout vs surrogate pairs."""

    if total_members % 2:
        raise ValueError("total_members must be even")
    predicted_member_fitness = np.asarray(
        predicted_member_fitness, dtype=np.float32
    )
    rollout_member_fitness = np.asarray(rollout_member_fitness, dtype=np.float32)
    rollout_member_ids = np.asarray(rollout_member_ids, dtype=np.int64)
    surrogate_pair_mask = np.asarray(surrogate_pair_mask, dtype=bool)

    if predicted_member_fitness.shape != (total_members,):
        raise ValueError("predicted_member_fitness has the wrong shape")
    if rollout_member_fitness.shape != rollout_member_ids.shape:
        raise ValueError("rollout_member_fitness and rollout_member_ids must match")

    combined = predicted_member_fitness.copy()
    combined[rollout_member_ids] = rollout_member_fitness

    pairs_per_prompt_total = surrogate_pair_mask.size
    rollout_pairs = int(pairs_per_prompt_total - int(np.count_nonzero(surrogate_pair_mask)))
    surrogate_pairs = int(np.count_nonzero(surrogate_pair_mask))
    return combined, rollout_pairs, surrogate_pairs
