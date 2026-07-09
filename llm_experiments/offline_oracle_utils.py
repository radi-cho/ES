"""Leakage-safe splits and evaluation for the offline Countdown oracle data.

The independent statistical unit in the oracle dataset is a Countdown prompt.
All perturbation pairs and all layer views for a prompt must therefore remain
in the same split.  This module deliberately operates on pair-level arrays;
callers should collapse layer-level predictions before computing metrics.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import rankdata


GROUP_SIGNAL_BLOCKS_V1 = "group_signal_blocks_v1"
DEFAULT_SPLIT_SEED = 20260709
DEFAULT_NONZERO_THRESHOLD = 1e-7
SPLIT_SIZES = {"train": 192, "validation": 32, "test": 32}
_SPLIT_NAMES = tuple(SPLIT_SIZES)

GROUP_SUMMARY_DTYPE = np.dtype(
    [
        ("sample_id", "<i8"),
        ("n_informative", "<i4"),
        ("signal_mass", "<f8"),
        ("n_correct", "<i4"),
        ("mean_reward", "<f8"),
    ]
)


def _validate_pair_rewards(pair_rewards: np.ndarray) -> np.ndarray:
    rewards = np.asarray(pair_rewards, dtype=np.float64)
    if rewards.ndim != 3 or rewards.shape[-1] != 2:
        raise ValueError("pair_rewards must have shape [samples, pairs, 2]")
    if rewards.shape[0] < 1 or rewards.shape[1] < 1:
        raise ValueError("pair_rewards dimensions must be nonempty")
    if not np.all(np.isfinite(rewards)):
        raise ValueError("pair_rewards must contain only finite values")
    return rewards


def _validate_sample_ids(sample_ids: np.ndarray | None, count: int) -> np.ndarray:
    if sample_ids is None:
        ids = np.arange(count, dtype=np.int64)
    else:
        raw_ids = np.asarray(sample_ids)
        if raw_ids.ndim != 1 or raw_ids.shape[0] != count:
            raise ValueError("sample_ids must have one entry per sample")
        if not np.issubdtype(raw_ids.dtype, np.integer):
            raise ValueError("sample_ids must be integers")
        ids = raw_ids.astype(np.int64, copy=False)
    if np.unique(ids).size != ids.size:
        raise ValueError("sample_ids must be unique")
    return ids


def _sha256_tiebreak(domain: str, seed: int, sample_id: int) -> int:
    payload = f"{domain}|{int(seed)}|{int(sample_id)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def group_reward_summaries(
    pair_rewards: np.ndarray,
    sample_ids: np.ndarray | None = None,
    *,
    nonzero_threshold: float = DEFAULT_NONZERO_THRESHOLD,
) -> np.ndarray:
    """Return one reward-signal summary record per prompt group."""

    if nonzero_threshold < 0.0:
        raise ValueError("nonzero_threshold must be nonnegative")
    rewards = _validate_pair_rewards(pair_rewards)
    ids = _validate_sample_ids(sample_ids, rewards.shape[0])
    differences = rewards[..., 0] - rewards[..., 1]

    summaries = np.empty(rewards.shape[0], dtype=GROUP_SUMMARY_DTYPE)
    summaries["sample_id"] = ids
    summaries["n_informative"] = np.sum(
        np.abs(differences) > nonzero_threshold, axis=1
    ).astype(np.int32)
    summaries["signal_mass"] = np.sum(np.abs(differences), axis=1)
    summaries["n_correct"] = np.sum(rewards >= 1.0, axis=(1, 2)).astype(np.int32)
    summaries["mean_reward"] = np.mean(rewards, axis=(1, 2))
    return summaries


def group_signal_blocks_v1(
    pair_rewards: np.ndarray,
    sample_ids: np.ndarray | None = None,
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    nonzero_threshold: float = DEFAULT_NONZERO_THRESHOLD,
) -> dict[str, np.ndarray]:
    """Create the frozen 192/32/32 reward-balanced prompt-group split.

    Samples are sorted by oracle-relevant group statistics and divided into 32
    consecutive signal blocks of eight.  Within each block, an independent
    SHA-256 ordering chooses one validation and one test prompt; their order is
    alternated across blocks.  The remaining six prompts enter training.

    The returned arrays contain sample IDs, not row offsets, and are sorted for
    convenient indexing.  The hash payloads are exactly
    ``"rank|<seed>|<sample_id>"`` and ``"assign|<seed>|<sample_id>"``.
    """

    rewards = _validate_pair_rewards(pair_rewards)
    if rewards.shape[:2] != (256, 32):
        raise ValueError(
            f"{GROUP_SIGNAL_BLOCKS_V1} requires pair_rewards shape [256, 32, 2]"
        )
    summaries = group_reward_summaries(
        rewards,
        sample_ids,
        nonzero_threshold=nonzero_threshold,
    )
    by_id = {int(row["sample_id"]): row for row in summaries}
    ordered_ids = sorted(
        by_id,
        key=lambda sample_id: (
            int(by_id[sample_id]["n_informative"]),
            float(by_id[sample_id]["signal_mass"]),
            int(by_id[sample_id]["n_correct"]),
            float(by_id[sample_id]["mean_reward"]),
            _sha256_tiebreak("rank", seed, sample_id),
        ),
    )

    split_lists: dict[str, list[int]] = {name: [] for name in _SPLIT_NAMES}
    for block_index in range(32):
        block = ordered_ids[8 * block_index : 8 * (block_index + 1)]
        assigned = sorted(
            block,
            key=lambda sample_id: _sha256_tiebreak("assign", seed, sample_id),
        )
        if block_index % 2 == 0:
            validation_id, test_id = assigned[:2]
        else:
            test_id, validation_id = assigned[:2]
        split_lists["validation"].append(validation_id)
        split_lists["test"].append(test_id)
        split_lists["train"].extend(assigned[2:])

    splits = {
        name: np.sort(np.asarray(split_lists[name], dtype=np.int64))
        for name in _SPLIT_NAMES
    }
    expected_ids = summaries["sample_id"].astype(np.int64, copy=False)
    _validate_splits(splits, expected_ids, SPLIT_SIZES)
    return splits


def _validate_splits(
    splits: Mapping[str, np.ndarray],
    expected_ids: np.ndarray,
    expected_sizes: Mapping[str, int] | None,
) -> dict[str, np.ndarray]:
    missing_names = set(_SPLIT_NAMES) - set(splits)
    extra_names = set(splits) - set(_SPLIT_NAMES)
    if missing_names or extra_names:
        raise ValueError(
            f"split names must be {_SPLIT_NAMES}; missing={sorted(missing_names)}, "
            f"extra={sorted(extra_names)}"
        )

    normalized: dict[str, np.ndarray] = {}
    for name in _SPLIT_NAMES:
        raw = np.asarray(splits[name])
        if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError(f"{name} sample IDs must be a one-dimensional integer array")
        values = raw.astype(np.int64, copy=False)
        if np.unique(values).size != values.size:
            raise ValueError(f"{name} contains duplicate sample groups")
        if expected_sizes is not None and values.size != int(expected_sizes[name]):
            raise ValueError(
                f"{name} has {values.size} samples; expected {expected_sizes[name]}"
            )
        normalized[name] = values

    joined = np.concatenate([normalized[name] for name in _SPLIT_NAMES])
    unique, counts = np.unique(joined, return_counts=True)
    leaked = unique[counts > 1]
    if leaked.size:
        raise ValueError(f"sample groups occur in multiple splits: {leaked.tolist()}")
    expected = np.sort(np.asarray(expected_ids, dtype=np.int64))
    actual = np.sort(unique)
    missing = np.setdiff1d(expected, actual, assume_unique=True)
    unexpected = np.setdiff1d(actual, expected, assume_unique=True)
    if missing.size or unexpected.size:
        raise ValueError(
            f"split is not exhaustive: missing={missing.tolist()}, "
            f"unexpected={unexpected.tolist()}"
        )
    return normalized


def audit_group_split(
    pair_rewards: np.ndarray,
    splits: Mapping[str, np.ndarray],
    sample_ids: np.ndarray | None = None,
    *,
    nonzero_threshold: float = DEFAULT_NONZERO_THRESHOLD,
    layers_per_pair: int = 24,
    expected_sizes: Mapping[str, int] | None = SPLIT_SIZES,
) -> dict[str, Any]:
    """Validate group disjointness/exhaustiveness and summarize each split.

    A ``ValueError`` is raised on any duplicate group, cross-split leakage,
    missing group, unexpected group, or size mismatch.  Returned counts make
    the reward-signal balance auditable without exposing layer rows as if they
    were independent observations.
    """

    if layers_per_pair < 1:
        raise ValueError("layers_per_pair must be positive")
    rewards = _validate_pair_rewards(pair_rewards)
    ids = _validate_sample_ids(sample_ids, rewards.shape[0])
    normalized = _validate_splits(splits, ids, expected_sizes)
    summaries = group_reward_summaries(
        rewards,
        ids,
        nonzero_threshold=nonzero_threshold,
    )
    id_to_offset = {int(sample_id): offset for offset, sample_id in enumerate(ids)}

    split_audits: dict[str, dict[str, float | int]] = {}
    for name in _SPLIT_NAMES:
        offsets = np.asarray(
            [id_to_offset[int(sample_id)] for sample_id in normalized[name]],
            dtype=np.int64,
        )
        selected = summaries[offsets]
        selected_rewards = rewards[offsets]
        sample_count = int(offsets.size)
        pair_count = sample_count * rewards.shape[1]
        split_audits[name] = {
            "sample_groups": sample_count,
            "independent_pairs": pair_count,
            "logical_layer_rows": pair_count * int(layers_per_pair),
            "prompts_with_signal": int(np.sum(selected["n_informative"] > 0)),
            "informative_pairs": int(np.sum(selected["n_informative"])),
            "informative_pair_fraction": float(
                np.sum(selected["n_informative"]) / pair_count
            ),
            "mean_informative_pairs_per_prompt": float(
                np.mean(selected["n_informative"])
            ),
            "total_signal_mass": float(np.sum(selected["signal_mass"])),
            "mean_signal_mass_per_prompt": float(np.mean(selected["signal_mass"])),
            "prompts_with_correct_rollout": int(np.sum(selected["n_correct"] > 0)),
            "correct_member_rollouts": int(np.sum(selected["n_correct"])),
            "mean_member_reward": float(np.mean(selected_rewards)),
        }

    return {
        "algorithm": GROUP_SIGNAL_BLOCKS_V1,
        "group_disjoint": True,
        "exhaustive": True,
        "sample_groups": int(ids.size),
        "pairs_per_group": int(rewards.shape[1]),
        "layers_per_pair": int(layers_per_pair),
        "splits": split_audits,
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = np.asarray(x, dtype=np.float64) - np.mean(x)
    y_centered = np.asarray(y, dtype=np.float64) - np.mean(y)
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(x_centered, y_centered) / denominator)


def pair_level_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    nonzero_threshold: float = DEFAULT_NONZERO_THRESHOLD,
    top_ks: Sequence[int] = (1, 2, 4, 8),
) -> dict[str, float]:
    """Evaluate pair-difference predictions grouped by prompt.

    Inputs must have shape ``[prompts, pairs]``.  Macro cosine, Spearman, and
    top-k metrics average over prompts with nonzero true signal.  A zero
    prediction receives cosine/Spearman zero for a prompt with true variation;
    prompts whose true target is constant are omitted from macro Spearman.

    ``topK_energy_recall`` is the fraction of a prompt's total ``D**2`` energy
    found in the K pairs selected by largest ``abs(prediction)``.  Calibration
    slope is the ordinary least-squares slope with an intercept, i.e. the slope
    in a pooled regression of true values on predictions.
    """

    if nonzero_threshold < 0.0:
        raise ValueError("nonzero_threshold must be nonnegative")
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.shape != pred.shape or true.ndim != 2:
        raise ValueError("y_true and y_pred must share shape [prompts, pairs]")
    if true.shape[0] < 1 or true.shape[1] < 1:
        raise ValueError("metric inputs must be nonempty")
    if not np.all(np.isfinite(true)) or not np.all(np.isfinite(pred)):
        raise ValueError("metric inputs must contain only finite values")

    requested_ks = tuple(int(k) for k in top_ks)
    if (
        len(set(requested_ks)) != len(requested_ks)
        or any(k < 1 or k > true.shape[1] for k in requested_ks)
    ):
        raise ValueError("top_ks must be distinct and between 1 and pairs per prompt")

    residual = pred - true
    mse = float(np.mean(np.square(residual)))
    mae = float(np.mean(np.abs(residual)))
    zero_mse = float(np.mean(np.square(true)))
    residual_ratio = mse / zero_mse if zero_mse > 0.0 else float("nan")

    true_norms = np.linalg.norm(true, axis=1)
    pred_norms = np.linalg.norm(pred, axis=1)
    signal_groups = true_norms > nonzero_threshold
    prompt_cosines: list[float] = []
    for group in np.flatnonzero(signal_groups):
        if pred_norms[group] <= nonzero_threshold:
            prompt_cosines.append(0.0)
        else:
            prompt_cosines.append(
                float(
                    np.dot(true[group], pred[group])
                    / (true_norms[group] * pred_norms[group])
                )
            )

    prompt_spearman: list[float] = []
    for true_row, pred_row in zip(true, pred):
        true_ranks = rankdata(true_row, method="average")
        if np.ptp(true_ranks) == 0.0:
            continue
        pred_ranks = rankdata(pred_row, method="average")
        correlation = _pearson(true_ranks, pred_ranks)
        prompt_spearman.append(0.0 if np.isnan(correlation) else correlation)

    informative = np.abs(true) > nonzero_threshold
    nonzero_sign_accuracy = (
        float(np.mean(np.sign(pred[informative]) == np.sign(true[informative])))
        if np.any(informative)
        else float("nan")
    )

    metrics = {
        "mse": mse,
        "mae": mae,
        "residual_ratio": residual_ratio,
        "r2_zero": 1.0 - residual_ratio,
        "pooled_pearson": _pearson(true.ravel(), pred.ravel()),
        "macro_prompt_cosine": (
            float(np.mean(prompt_cosines)) if prompt_cosines else float("nan")
        ),
        "macro_prompt_spearman": (
            float(np.mean(prompt_spearman)) if prompt_spearman else float("nan")
        ),
        "nonzero_sign_accuracy": nonzero_sign_accuracy,
    }

    true_energy = np.square(true)
    total_energy = np.sum(true_energy, axis=1)
    energy_groups = total_energy > nonzero_threshold**2
    for k in requested_ks:
        recalls: list[float] = []
        for group in np.flatnonzero(energy_groups):
            selected = np.argsort(-np.abs(pred[group]), kind="stable")[:k]
            recalls.append(float(np.sum(true_energy[group, selected]) / total_energy[group]))
        metrics[f"top{k}_energy_recall"] = (
            float(np.mean(recalls)) if recalls else float("nan")
        )

    flat_true = true.ravel()
    flat_pred = pred.ravel()
    centered_pred = flat_pred - np.mean(flat_pred)
    prediction_variance_sum = float(np.dot(centered_pred, centered_pred))
    metrics["calibration_slope"] = (
        float(np.dot(centered_pred, flat_true - np.mean(flat_true)) / prediction_variance_sum)
        if prediction_variance_sum > 0.0
        else float("nan")
    )
    return metrics


__all__ = [
    "DEFAULT_NONZERO_THRESHOLD",
    "DEFAULT_SPLIT_SEED",
    "GROUP_SIGNAL_BLOCKS_V1",
    "GROUP_SUMMARY_DTYPE",
    "SPLIT_SIZES",
    "audit_group_split",
    "group_reward_summaries",
    "group_signal_blocks_v1",
    "pair_level_metrics",
]
