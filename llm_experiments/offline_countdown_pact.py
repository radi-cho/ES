"""Build and evaluate the 680-D PACT Countdown preview representation.

PACT is a split-aware feature builder: reward-derived directions and selected
layers are fitted only on the frozen training prompt IDs.  It consumes raw
question-only prefill artifacts, reuses the already-collected PR5 rollout
labels, and never performs generation.

The feature contract is exactly ``3 * 168 + 176 = 680``:

* 32 supervised-subspace tangent/context/curvature coordinates per layer;
* 64 residual tangent CountSketch coordinates per layer;
* 64 residual base--tangent TensorSketch coordinates per layer;
* 8 odd geometry scalars per layer; and
* a 176-D panel-conditional output-boundary block.

Every feature is odd under exchanging the positive and negative antithetic
members.  The module checks this invariant before committing a feature cache.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm

from llm_experiments.offline_oracle_classical import make_six_prompt_folds
from llm_experiments.offline_oracle_search import (
    load_and_verify_dataset,
    load_group_split,
)
from llm_experiments.offline_oracle_utils import pair_level_metrics


SCHEMA_VERSION = 1
SUBSPACE_DIM = 8
SKETCH_DIM = 64
LAYER_FEATURE_DIM = 168
POLICY_FEATURE_DIM = 176
PACT_FEATURE_DIM = 680
SELECTED_LAYERS = 3
TOP_PANEL = 64
PANEL_SIZE = 96
RMS_FLOOR = 1e-4
CURVATURE_CLIP = 10.0
PREDICTION_CLIP = 1.1
FEATURE_SEED = 20260709
CV_SEED = 20260709


@dataclass(frozen=True)
class LayerMap:
    layer: int
    base_mean: np.ndarray
    basis: np.ndarray
    coordinate_origin: np.ndarray
    curvature_scale: float


@dataclass(frozen=True)
class RawPrefills:
    member_hidden: np.ndarray
    clean_hidden: np.ndarray
    panel_token_ids: np.ndarray
    clean_panel_logits: np.ndarray
    clean_logsumexp: np.ndarray
    member_panel_logits: np.ndarray


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    temporary.replace(path)


def _load_array(path: Path, shape: tuple[int, ...], *, integer: bool = False) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = np.load(path, mmap_mode="r")
    if value.shape != shape:
        raise ValueError(f"{path.name} has shape {value.shape}; expected {shape}")
    if integer != np.issubdtype(value.dtype, np.integer):
        kind = "integer" if integer else "floating-point"
        raise ValueError(f"{path.name} must have a {kind} dtype, got {value.dtype}")
    return value


def load_raw_prefills(
    directory: str | Path, *, samples: int, pairs: int, layers: int, hidden: int
) -> RawPrefills:
    directory = Path(directory).expanduser().resolve()
    raw = RawPrefills(
        member_hidden=_load_array(
            directory / "member_hidden.npy", (samples, pairs, 2, layers, hidden)
        ),
        clean_hidden=_load_array(directory / "clean_hidden.npy", (samples, layers, hidden)),
        panel_token_ids=_load_array(
            directory / "panel_token_ids.npy", (samples, PANEL_SIZE), integer=True
        ),
        clean_panel_logits=_load_array(
            directory / "clean_panel_logits.npy", (samples, PANEL_SIZE)
        ),
        clean_logsumexp=_load_array(directory / "clean_logsumexp.npy", (samples,)),
        member_panel_logits=_load_array(
            directory / "member_panel_logits.npy",
            (samples, pairs, 2, PANEL_SIZE),
        ),
    )
    completed = _load_array(directory / "completed_samples.npy", (samples,), integer=True)
    if not np.all(completed == 1):
        raise ValueError("raw prefill collection is incomplete")
    if np.any(np.diff(np.asarray(raw.clean_panel_logits), axis=1) > 1e-5):
        raise ValueError("clean_panel_logits must be sorted from largest to smallest")
    if any(
        not np.all(np.isfinite(np.asarray(array)))
        for array in (
            raw.member_hidden,
            raw.clean_hidden,
            raw.clean_panel_logits,
            raw.clean_logsumexp,
            raw.member_panel_logits,
        )
    ):
        raise ValueError("raw prefill artifacts contain non-finite values")
    return raw


def _rms(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.maximum(
        np.sqrt(np.mean(np.square(values), axis=-1, keepdims=True, dtype=np.float32)),
        RMS_FLOOR,
    )


def _normalize_hidden(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / _rms(values)


def _orthogonal_append(
    columns: list[np.ndarray], candidate: np.ndarray, *, seed: int
) -> np.ndarray:
    value = np.asarray(candidate, dtype=np.float64).copy()
    reference = value.copy()
    for _ in range(2):
        for column in columns:
            value -= column * float(np.dot(column, value))
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-10:
        rng = np.random.default_rng(seed)
        value = rng.standard_normal(reference.size)
        for _ in range(2):
            for column in columns:
                value -= column * float(np.dot(column, value))
        norm = float(np.linalg.norm(value))
        if norm <= 1e-10:
            raise RuntimeError("could not construct a full-rank PACT subspace")
    value /= norm
    if float(np.dot(value, reference)) < 0.0:
        value *= -1.0
    return value


def _fit_layer_map(
    clean: np.ndarray,
    members: np.ndarray,
    tangent: np.ndarray,
    rewards: np.ndarray,
    *,
    sigma: float,
    layer: int,
    seed: int,
) -> LayerMap:
    """Fit four reward channels plus four label-free residual PCA directions."""

    clean = _normalize_hidden(clean)
    members = _normalize_hidden(members)
    tangent = np.asarray(tangent, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32)
    n, pairs, two, hidden = members.shape
    if (
        clean.shape != (n, hidden)
        or tangent.shape != (n, pairs, hidden)
        or rewards.shape != (n, pairs, 2)
        or two != 2
        or sigma <= 0.0
    ):
        raise ValueError("incompatible arrays passed to _fit_layer_map")

    plus, minus = members[:, :, 0], members[:, :, 1]
    prompt_reward = np.mean(rewards, axis=(1, 2), dtype=np.float32)
    base_mean = np.mean(clean, axis=0, dtype=np.float64)

    correct_plus = rewards[:, :, 0] >= 1.0
    correct_minus = rewards[:, :, 1] >= 1.0
    correct_count = np.sum(correct_plus, axis=1) + np.sum(correct_minus, axis=1)
    wrong_count = 2 * pairs - correct_count
    correct_prompts = correct_count > 0
    wrong_prompts = wrong_count > 0
    if np.any(correct_prompts) and np.any(wrong_prompts):
        # A prompt contributes at most one class centroid to either side,
        # irrespective of how many of its correlated perturbations succeeded.
        correct_sum = np.einsum(
            "nph,np->nh", plus, correct_plus.astype(np.float32), optimize=True
        ) + np.einsum(
            "nph,np->nh", minus, correct_minus.astype(np.float32), optimize=True
        )
        wrong_sum = np.einsum(
            "nph,np->nh", plus, (~correct_plus).astype(np.float32), optimize=True
        ) + np.einsum(
            "nph,np->nh", minus, (~correct_minus).astype(np.float32), optimize=True
        )
        correct_mean = np.mean(
            correct_sum[correct_prompts] / correct_count[correct_prompts, None],
            axis=0,
            dtype=np.float64,
        )
        wrong_mean = np.mean(
            wrong_sum[wrong_prompts] / wrong_count[wrong_prompts, None],
            axis=0,
            dtype=np.float64,
        )
        absolute_direction = correct_mean - wrong_mean
    else:
        correct_mean = wrong_mean = base_mean
        absolute_direction = np.zeros(hidden, dtype=np.float64)

    difficulty_direction = np.einsum(
        "nh,n->h",
        clean.astype(np.float64) - base_mean,
        prompt_reward.astype(np.float64) - float(np.mean(prompt_reward)),
        optimize=True,
    ) / n

    member_center = prompt_reward[:, None]
    plus_target = rewards[:, :, 0] - member_center
    minus_target = rewards[:, :, 1] - member_center
    effect_direction = (
        np.einsum("nph,np->h", plus - clean[:, None], plus_target, optimize=True)
        + np.einsum("nph,np->h", minus - clean[:, None], minus_target, optimize=True)
    ) / (2 * n * pairs)
    pair_target = rewards[:, :, 0] - rewards[:, :, 1]
    pair_direction = np.einsum(
        "nph,np->h", tangent, pair_target, optimize=True
    ) / (n * pairs)

    columns: list[np.ndarray] = []
    for slot, candidate in enumerate(
        (absolute_direction, difficulty_direction, effect_direction, pair_direction)
    ):
        columns.append(
            _orthogonal_append(columns, candidate, seed=seed + 101 * layer + slot)
        )
    reward_basis = np.stack(columns, axis=1)

    # Equal total mass for tangent and clean-state variation without repeating
    # every clean state P times.
    centered_base = clean.astype(np.float32) - base_mean.astype(np.float32)
    variation = np.concatenate(
        [tangent.reshape(-1, hidden), math.sqrt(pairs) * centered_base], axis=0
    )
    variation -= (variation @ reward_basis) @ reward_basis.T
    _, _, right = randomized_svd(
        variation,
        n_components=4,
        n_oversamples=4,
        n_iter=2,
        random_state=int(seed + 1009 * layer),
        flip_sign=True,
    )
    for slot, candidate in enumerate(right):
        columns.append(
            _orthogonal_append(columns, candidate, seed=seed + 10007 * layer + slot)
        )
    basis = np.stack(columns, axis=1).astype(np.float32)

    origin = base_mean @ basis.astype(np.float64)
    origin[0] = 0.5 * float((correct_mean + wrong_mean) @ basis[:, 0])

    raw_plus = np.asarray(members[:, :, 0], dtype=np.float32)
    raw_minus = np.asarray(members[:, :, 1], dtype=np.float32)
    # members are normalized above; curvature must be computed by the caller
    # from raw states.  The map stores a placeholder overwritten below.
    del raw_plus, raw_minus
    return LayerMap(
        layer=int(layer),
        base_mean=base_mean.astype(np.float32),
        basis=basis,
        coordinate_origin=origin.astype(np.float32),
        curvature_scale=1.0,
    )


def _with_curvature_scale(
    layer_map: LayerMap,
    clean_raw: np.ndarray,
    members_raw: np.ndarray,
    *,
    sigma: float,
) -> LayerMap:
    center = 0.5 * (
        np.asarray(members_raw[:, :, 0], dtype=np.float32)
        + np.asarray(members_raw[:, :, 1], dtype=np.float32)
    )
    curvature = (center - np.asarray(clean_raw, dtype=np.float32)[:, None]) / (
        float(sigma) ** 2 * _rms(clean_raw)[:, None]
    )
    scale = float(np.median(np.abs(curvature)))
    if not np.isfinite(scale) or scale < RMS_FLOOR:
        scale = RMS_FLOOR
    return LayerMap(
        layer=layer_map.layer,
        base_mean=layer_map.base_mean,
        basis=layer_map.basis,
        coordinate_origin=layer_map.coordinate_origin,
        curvature_scale=scale,
    )


def _fit_complete_layer_map(
    clean_raw: np.ndarray,
    members_raw: np.ndarray,
    tangent: np.ndarray,
    rewards: np.ndarray,
    *,
    sigma: float,
    layer: int,
    seed: int,
) -> LayerMap:
    fitted = _fit_layer_map(
        clean_raw,
        members_raw,
        tangent,
        rewards,
        sigma=sigma,
        layer=layer,
        seed=seed,
    )
    return _with_curvature_scale(fitted, clean_raw, members_raw, sigma=sigma)


def _countsketch_parameters(hidden: int, layer: int, stream: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(np.random.SeedSequence([FEATURE_SEED, layer, stream]))
    buckets = rng.integers(0, SKETCH_DIM, size=hidden, dtype=np.int32)
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=hidden)
    return buckets, signs


def _countsketch(values: np.ndarray, buckets: np.ndarray, signs: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    flat = values.reshape(-1, values.shape[-1])
    output = np.empty((flat.shape[0], SKETCH_DIM), dtype=np.float32)
    for bucket in range(SKETCH_DIM):
        selected = buckets == bucket
        output[:, bucket] = flat[:, selected] @ signs[selected]
    return output.reshape(values.shape[:-1] + (SKETCH_DIM,))


def _safe_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=-1)
    denominator = np.maximum(
        np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1), 1e-8
    )
    return numerator / denominator


def _layer_core_features(
    layer_map: LayerMap,
    clean_raw: np.ndarray,
    members_raw: np.ndarray,
    tangent: np.ndarray,
    *,
    sigma: float,
) -> tuple[np.ndarray, dict[str, float]]:
    clean = _normalize_hidden(clean_raw)
    members = _normalize_hidden(members_raw)
    tangent = np.asarray(tangent, dtype=np.float32)
    basis = layer_map.basis
    base_coordinates = clean @ basis - layer_map.coordinate_origin
    tangent_coordinates = np.einsum("nph,hk->npk", tangent, basis, optimize=True)

    center_raw = 0.5 * (
        np.asarray(members_raw[:, :, 0], dtype=np.float32)
        + np.asarray(members_raw[:, :, 1], dtype=np.float32)
    )
    curvature = (center_raw - np.asarray(clean_raw, dtype=np.float32)[:, None]) / (
        float(sigma) ** 2 * _rms(clean_raw)[:, None]
    )
    scaled_curvature = np.clip(
        curvature / layer_map.curvature_scale, -CURVATURE_CLIP, CURVATURE_CLIP
    )
    curvature_coordinates = np.einsum(
        "nph,hk->npk", scaled_curvature, basis, optimize=True
    )
    core = np.concatenate(
        [
            tangent_coordinates,
            base_coordinates[:, None] * tangent_coordinates,
            np.square(base_coordinates[:, None]) * tangent_coordinates,
            curvature_coordinates * tangent_coordinates,
        ],
        axis=-1,
    ).astype(np.float32)
    stats = {
        "curvature_scale": float(layer_map.curvature_scale),
        "curvature_rms_after_scale": float(np.sqrt(np.mean(scaled_curvature**2))),
        "curvature_clipped_fraction": float(
            np.mean(np.abs(curvature / layer_map.curvature_scale) > CURVATURE_CLIP)
        ),
        "tangent_rms": float(np.sqrt(np.mean(tangent**2))),
    }
    return core, stats


def _layer_features(
    layer_map: LayerMap,
    clean_raw: np.ndarray,
    members_raw: np.ndarray,
    tangent: np.ndarray,
    *,
    sigma: float,
) -> tuple[np.ndarray, dict[str, float]]:
    core, stats = _layer_core_features(
        layer_map, clean_raw, members_raw, tangent, sigma=sigma
    )
    clean = _normalize_hidden(clean_raw)
    members = _normalize_hidden(members_raw)
    plus, minus = members[:, :, 0], members[:, :, 1]
    basis = layer_map.basis
    tangent_coordinates = np.einsum("nph,hk->npk", tangent, basis, optimize=True)
    base_delta = clean - layer_map.base_mean
    base_coordinates = base_delta @ basis

    bucket_d, sign_d = _countsketch_parameters(tangent.shape[-1], layer_map.layer, 0)
    residual_tangent = _countsketch(tangent, bucket_d, sign_d)
    sketched_basis_d = _countsketch(basis.T, bucket_d, sign_d)
    residual_tangent -= np.einsum(
        "npk,km->npm", tangent_coordinates, sketched_basis_d, optimize=True
    )

    bucket_b, sign_b = _countsketch_parameters(tangent.shape[-1], layer_map.layer, 1)
    bucket_t, sign_t = _countsketch_parameters(tangent.shape[-1], layer_map.layer, 2)
    residual_base = _countsketch(base_delta, bucket_b, sign_b)
    residual_base -= base_coordinates @ _countsketch(basis.T, bucket_b, sign_b)
    tensor_tangent = _countsketch(tangent, bucket_t, sign_t)
    tensor_tangent -= np.einsum(
        "npk,km->npm",
        tangent_coordinates,
        _countsketch(basis.T, bucket_t, sign_t),
        optimize=True,
    )
    tensor_context = np.fft.irfft(
        np.fft.rfft(residual_base, axis=-1)[:, None]
        * np.fft.rfft(tensor_tangent, axis=-1),
        n=SKETCH_DIM,
        axis=-1,
    ).real.astype(np.float32)

    e_plus = plus - clean[:, None]
    e_minus = minus - clean[:, None]
    raw_plus = np.asarray(members_raw[:, :, 0], dtype=np.float32)
    raw_minus = np.asarray(members_raw[:, :, 1], dtype=np.float32)
    raw_clean = np.asarray(clean_raw, dtype=np.float32)[:, None]
    midpoint = 0.5 * (plus + minus) - clean[:, None]
    ep_coordinates = np.einsum("nph,hk->npk", e_plus, basis, optimize=True)
    em_coordinates = np.einsum("nph,hk->npk", e_minus, basis, optimize=True)
    ep_residual_norm = np.sqrt(
        np.maximum(np.sum(e_plus**2, axis=-1) - np.sum(ep_coordinates**2, axis=-1), 0)
    )
    em_residual_norm = np.sqrt(
        np.maximum(np.sum(e_minus**2, axis=-1) - np.sum(em_coordinates**2, axis=-1), 0)
    )
    geometry = np.stack(
        [
            np.linalg.norm(e_plus, axis=-1) - np.linalg.norm(e_minus, axis=-1),
            _rms(raw_plus)[..., 0] - _rms(raw_minus)[..., 0],
            _safe_cosine(clean[:, None], plus) - _safe_cosine(clean[:, None], minus),
            _safe_cosine(clean[:, None], tangent),
            _safe_cosine(midpoint, tangent),
            np.linalg.norm(ep_coordinates, axis=-1)
            - np.linalg.norm(em_coordinates, axis=-1),
            ep_residual_norm - em_residual_norm,
            np.max(np.abs(raw_plus - raw_clean), axis=-1)
            - np.max(np.abs(raw_minus - raw_clean), axis=-1),
        ],
        axis=-1,
    ).astype(np.float32)
    result = np.concatenate(
        [core, residual_tangent, tensor_context, geometry], axis=-1
    ).astype(np.float32)
    if result.shape[-1] != LAYER_FEATURE_DIM:
        raise AssertionError(f"layer feature contract drifted to {result.shape[-1]}")
    return result, stats


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.array(logits, dtype=np.float64, copy=True)
    shifted -= np.max(shifted, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=-1, keepdims=True)


def _kl(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(left * (np.log(np.maximum(left, 1e-30)) - np.log(np.maximum(right, 1e-30))), axis=-1)


def policy_features(
    clean_logits: np.ndarray,
    clean_logsumexp: np.ndarray,
    member_logits: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return 176 panel-conditional features, all exactly member-swap odd."""

    clean_logits = np.asarray(clean_logits, dtype=np.float64)
    clean_logsumexp = np.asarray(clean_logsumexp, dtype=np.float64)
    member_logits = np.asarray(member_logits, dtype=np.float64)
    if (
        clean_logits.ndim != 2
        or clean_logits.shape[1] != PANEL_SIZE
        or clean_logsumexp.shape != (clean_logits.shape[0],)
        or member_logits.shape[0] != clean_logits.shape[0]
        or member_logits.shape[2:] != (2, PANEL_SIZE)
    ):
        raise ValueError("policy panel arrays have incompatible shapes")
    plus, minus = member_logits[:, :, 0], member_logits[:, :, 1]
    delta = plus - minus
    full_base_probability = np.exp(clean_logits - clean_logsumexp[:, None])
    q0 = _softmax(clean_logits)
    q_plus = _softmax(plus)
    q_minus = _softmax(minus)
    ranks = np.arange(PANEL_SIZE, dtype=np.float64)

    margin_changes = delta[..., :1] - delta[..., 1:9]
    entropy_plus = -np.sum(q_plus * np.log(np.maximum(q_plus, 1e-30)), axis=-1)
    entropy_minus = -np.sum(q_minus * np.log(np.maximum(q_minus, 1e-30)), axis=-1)
    margin_plus = plus[..., 0] - np.max(plus[..., 1:], axis=-1)
    margin_minus = minus[..., 0] - np.max(minus[..., 1:], axis=-1)
    scalars = np.concatenate(
        [
            margin_changes,
            (entropy_minus - entropy_plus)[..., None],
            (q_plus[..., 0] - q_minus[..., 0])[..., None],
            (
                np.sum(q_plus[..., :TOP_PANEL], axis=-1)
                - np.sum(q_minus[..., :TOP_PANEL], axis=-1)
            )[..., None],
            (
                np.sum(q_minus * ranks, axis=-1)
                - np.sum(q_plus * ranks, axis=-1)
            )[..., None],
            (_kl(q0[:, None], q_minus) - _kl(q0[:, None], q_plus))[..., None],
            (_kl(q_minus, q0[:, None]) - _kl(q_plus, q0[:, None]))[..., None],
            (margin_plus - margin_minus)[..., None],
            (
                (np.argmax(plus, axis=-1) == 0).astype(np.float64)
                - (np.argmax(minus, axis=-1) == 0).astype(np.float64)
            )[..., None],
        ],
        axis=-1,
    )
    result = np.concatenate(
        [
            delta,
            full_base_probability[:, None, :TOP_PANEL] * delta[..., :TOP_PANEL],
            scalars,
        ],
        axis=-1,
    ).astype(np.float32)
    if result.shape[-1] != POLICY_FEATURE_DIM:
        raise AssertionError(f"policy feature contract drifted to {result.shape[-1]}")
    return result, {
        "mean_clean_top96_mass": float(np.mean(np.sum(full_base_probability, axis=-1))),
        "min_clean_top96_mass": float(np.min(np.sum(full_base_probability, axis=-1))),
        "max_clean_top96_mass": float(np.max(np.sum(full_base_probability, axis=-1))),
    }


def _fit_ridge10(features: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(features, dtype=np.float64).reshape(-1, features.shape[-1])
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    rms = np.maximum(np.sqrt(np.mean(x**2, axis=0)), RMS_FLOOR)
    normalized = x / rms
    count = float(x.shape[0])
    weights = np.linalg.solve(
        normalized.T @ normalized / count + 10.0 * np.eye(x.shape[1]),
        normalized.T @ y / count,
    )
    return {"rms": rms, "weights": weights}


def _predict_ridge(model: Mapping[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    flat = np.asarray(features, dtype=np.float64).reshape(-1, features.shape[-1])
    prediction = (flat / model["rms"]) @ model["weights"]
    return prediction.reshape(features.shape[:2])


def _prompt_center(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32).copy()
    values -= np.mean(values, axis=1, keepdims=True, dtype=np.float32)
    return values


def _fit_huber(features: np.ndarray, targets: np.ndarray) -> Any:
    model = make_pipeline(
        StandardScaler(),
        HuberRegressor(
            epsilon=1.35,
            alpha=1e-4,
            fit_intercept=True,
            max_iter=1000,
            tol=1e-5,
        ),
    )
    return model.fit(
        features.reshape(-1, features.shape[-1]), targets.reshape(-1)
    )


def _predict_huber(model: Any, features: np.ndarray) -> np.ndarray:
    prediction = model.predict(features.reshape(-1, features.shape[-1]))
    return np.asarray(prediction).reshape(features.shape[:2])


def _zero_intercept_scale(prediction: np.ndarray, target: np.ndarray) -> float:
    p = np.asarray(prediction, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    denominator = float(p @ p)
    return 1.0 if denominator <= 1e-12 else float((p @ y) / denominator)


def _layer_selection(
    raw: RawPrefills,
    tangent: np.ndarray,
    rewards: np.ndarray,
    train_ids: np.ndarray,
    *,
    sigma: float,
    candidate_layers: Sequence[int],
) -> tuple[list[int], dict[int, dict[str, float]]]:
    """Select three layers with a cheap train-only correctness direction.

    The final 8-D maps are fitted only after selection.  Refitting their PCA
    and curvature machinery in all 144 layer/fold combinations is both costly
    and an unnecessary source of selection noise.  This screen instead refits
    ``mean(tangent * reward_difference)`` in each prompt-group fold.
    """

    folds = make_six_prompt_folds(train_ids, seed=CV_SEED)
    id_to_offset = {int(sample_id): offset for offset, sample_id in enumerate(train_ids)}
    fold_offsets = [
        np.asarray([id_to_offset[int(sample_id)] for sample_id in fold], dtype=np.int64)
        for fold in folds
    ]
    results: dict[int, dict[str, float]] = {}
    train_rewards = np.asarray(rewards[train_ids], dtype=np.float32)
    train_targets = train_rewards[..., 0] - train_rewards[..., 1]
    for layer in tqdm(candidate_layers, desc="PACT train-only layer CV", unit="layer"):
        oof = np.empty_like(train_targets)
        layer_tangent = np.asarray(tangent[train_ids, :, layer], dtype=np.float32)
        for heldout in fold_offsets:
            fit_mask = np.ones(train_ids.size, dtype=bool)
            fit_mask[heldout] = False
            direction = np.einsum(
                "nph,np->h",
                layer_tangent[fit_mask],
                train_targets[fit_mask],
                optimize=True,
            )
            norm = float(np.linalg.norm(direction))
            if not np.isfinite(norm) or norm <= 1e-12:
                direction = np.zeros_like(direction)
            else:
                direction /= norm
            oof[heldout] = np.einsum(
                "nph,h->np", layer_tangent[heldout], direction, optimize=True
            )
        metrics = pair_level_metrics(train_targets, oof)
        results[int(layer)] = {
            "macro_prompt_cosine": float(metrics["macro_prompt_cosine"]),
            "r2_zero": float(metrics["r2_zero"]),
            "pooled_pearson": float(metrics["pooled_pearson"]),
        }
    ordered = sorted(
        candidate_layers,
        key=lambda layer: (
            results[int(layer)]["macro_prompt_cosine"],
            results[int(layer)]["r2_zero"],
            -int(layer),
        ),
        reverse=True,
    )
    return [int(layer) for layer in ordered[:SELECTED_LAYERS]], results


def build_features(
    *,
    dataset_directory: str | Path,
    raw_directory: str | Path,
    output_directory: str | Path,
    candidate_layers: Sequence[int] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    info = load_and_verify_dataset(dataset_directory, deep=False)
    search = info.directory / "offline_oracle_v1"
    split_path = search / "split.json"
    splits = load_group_split(split_path, info)
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    run_config = json.loads((info.directory / "run_config.json").read_text(encoding="utf-8"))
    sigma = float(run_config["perturbations"]["sigma"])
    raw_path = Path(raw_directory).expanduser().resolve()
    raw_run_config_path = raw_path / "run_config.json"
    raw_manifest_path = raw_path / "manifest.json"
    raw_metadata_path = raw_path / "metadata.json"
    for required in (raw_run_config_path, raw_manifest_path, raw_metadata_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    raw_run_config = json.loads(raw_run_config_path.read_text(encoding="utf-8"))
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    raw_metadata = json.loads(raw_metadata_path.read_text(encoding="utf-8"))
    raw_run_config_sha256 = _canonical_sha256(raw_run_config)
    if raw_metadata.get("status") != "complete":
        raise ValueError("raw PACT prefill collection is not complete")
    if any(
        payload.get("source_run_config_sha256") != info.run_config_sha256
        for payload in (raw_run_config, raw_manifest, raw_metadata)
    ):
        raise ValueError("raw prefills belong to another PR5 dataset")
    if any(
        payload.get("run_config_sha256") != raw_run_config_sha256
        for payload in (raw_manifest, raw_metadata)
    ):
        raise ValueError("raw prefill manifest/metadata does not match run_config.json")
    if int(raw_run_config["collection"]["panel_size"]) != PANEL_SIZE:
        raise ValueError("raw prefill policy panel does not have 96 tokens")
    if int(raw_run_config["collection"].get("full_rollouts", -1)) != 0:
        raise ValueError("PACT raw collection unexpectedly records new rollouts")
    raw = load_raw_prefills(
        raw_path,
        samples=info.samples,
        pairs=info.pairs,
        layers=info.layers,
        hidden=info.hidden,
    )
    output = Path(output_directory).expanduser().resolve()
    feature_path = output / "features.npy"
    if feature_path.exists() and not overwrite:
        raise FileExistsError(f"{feature_path} exists; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)

    tangent = np.load(info.predictor_inputs_path, mmap_mode="r").reshape(
        info.samples, info.pairs, info.layers, info.hidden
    )
    rewards = np.load(info.pair_rewards_path, mmap_mode="r")
    train_ids = np.asarray(splits["train"], dtype=np.int64)
    candidates = list(range(info.layers)) if candidate_layers is None else list(candidate_layers)
    if (
        len(set(candidates)) != len(candidates)
        or len(candidates) < SELECTED_LAYERS
        or any(layer < 0 or layer >= info.layers for layer in candidates)
    ):
        raise ValueError("candidate layers must contain at least three distinct valid layers")

    selected, selection_results = _layer_selection(
        raw,
        tangent,
        rewards,
        train_ids,
        sigma=sigma,
        candidate_layers=candidates,
    )
    maps: list[LayerMap] = []
    layer_blocks: list[np.ndarray] = []
    snr: dict[str, Any] = {}
    train_rewards = np.asarray(rewards[train_ids], dtype=np.float32)
    for layer in selected:
        layer_map = _fit_complete_layer_map(
            np.asarray(raw.clean_hidden[train_ids, layer]),
            np.asarray(raw.member_hidden[train_ids, :, :, layer]),
            np.asarray(tangent[train_ids, :, layer]),
            train_rewards,
            sigma=sigma,
            layer=layer,
            seed=FEATURE_SEED,
        )
        block, stats = _layer_features(
            layer_map,
            np.asarray(raw.clean_hidden[:, layer]),
            np.asarray(raw.member_hidden[:, :, :, layer]),
            np.asarray(tangent[:, :, layer]),
            sigma=sigma,
        )
        maps.append(layer_map)
        layer_blocks.append(block)
        snr[str(layer)] = stats

    panel, panel_stats = policy_features(
        raw.clean_panel_logits, raw.clean_logsumexp, raw.member_panel_logits
    )
    features = np.concatenate(layer_blocks + [panel], axis=-1).astype(np.float32)
    if features.shape != (info.samples, info.pairs, PACT_FEATURE_DIM):
        raise AssertionError(f"PACT feature shape is {features.shape}")

    swapped_panel, _ = policy_features(
        raw.clean_panel_logits,
        raw.clean_logsumexp,
        np.asarray(raw.member_panel_logits)[:, :, ::-1],
    )
    swapped_blocks: list[np.ndarray] = []
    audit_ids = np.arange(min(2, info.samples))
    for layer_map in maps:
        layer = layer_map.layer
        swapped, _ = _layer_features(
            layer_map,
            np.asarray(raw.clean_hidden[audit_ids, layer]),
            np.asarray(raw.member_hidden[audit_ids, :, ::-1, layer]),
            -np.asarray(tangent[audit_ids, :, layer]),
            sigma=sigma,
        )
        swapped_blocks.append(swapped)
    swapped_audit = np.concatenate(
        swapped_blocks + [swapped_panel[audit_ids]], axis=-1
    )
    swap_block_errors: dict[str, dict[str, float]] = {}
    named_blocks = [
        (f"layer_{layer_map.layer}", layer_blocks[index][audit_ids], swapped_blocks[index])
        for index, layer_map in enumerate(maps)
    ] + [("policy_panel", panel[audit_ids], swapped_panel[audit_ids])]
    for name, original_block, swapped_block in named_blocks:
        slices = (
            {"core32": slice(0, 32), "residual64": slice(32, 96),
             "tensor64": slice(96, 160), "geometry8": slice(160, 168)}
            if name.startswith("layer_")
            else {"all": slice(None)}
        )
        swap_block_errors[name] = {}
        for subname, section in slices.items():
            residual = swapped_block[..., section] + original_block[..., section]
            scale = np.maximum(
                np.maximum(
                    np.abs(swapped_block[..., section]),
                    np.abs(original_block[..., section]),
                ),
                1.0,
            )
            swap_block_errors[name][f"{subname}_max_abs"] = float(
                np.max(np.abs(residual))
            )
            swap_block_errors[name][f"{subname}_max_relative"] = float(
                np.max(np.abs(residual) / scale)
            )
    swap_residual = swapped_audit + features[audit_ids]
    swap_error = float(np.max(np.abs(swap_residual)))
    swap_scale = np.maximum(
        np.maximum(np.abs(swapped_audit), np.abs(features[audit_ids])), 1.0
    )
    swap_relative_error = float(np.max(np.abs(swap_residual) / swap_scale))
    # CountSketch/geometry/policy are exactly signed.  Re-evaluating the large
    # float32 matrix reductions and FFT TensorSketch after a swap accumulates
    # roundoff proportional to feature magnitude (audited below per block).
    if not np.allclose(
        swapped_audit,
        -features[audit_ids],
        rtol=1e-3,
        atol=2e-4,
    ):
        raise AssertionError(
            "member-swap oddness failed: "
            f"max_abs={swap_error}, max_relative={swap_relative_error}, "
            f"blocks={swap_block_errors}"
        )

    _atomic_npy(feature_path, features)
    _atomic_npy(output / "selected_layers.npy", np.asarray(selected, dtype=np.int32))
    _atomic_npy(output / "layer_basis.npy", np.stack([item.basis for item in maps]))
    _atomic_npy(output / "layer_base_mean.npy", np.stack([item.base_mean for item in maps]))
    _atomic_npy(
        output / "layer_coordinate_origin.npy",
        np.stack([item.coordinate_origin for item in maps]),
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method": "PACT_680_v1",
        "created_unix": time.time(),
        "dataset_run_config_sha256": info.run_config_sha256,
        "split_sha256": _canonical_sha256(split_payload),
        "source_split_path": split_path,
        "raw_directory": raw_path,
        "raw_run_config_sha256": raw_run_config_sha256,
        "raw_manifest_sha256": _canonical_sha256(raw_manifest),
        "train_sample_ids": train_ids,
        "validation_sample_ids": splits["validation"],
        "test_sample_ids": splits["test"],
        "test_labels_used_for_features": False,
        "validation_labels_used_for_features": False,
        "sigma": sigma,
        "candidate_layers": candidates,
        "selected_layers": selected,
        "layer_selection": "six_fold_train_prompt_cv_single_correctness_direction",
        "layer_selection_metrics": selection_results,
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "feature_blocks": {
            "subspace_odd_coordinates": 3 * 32,
            "residual_tangent_countsketch": 3 * 64,
            "residual_base_tangent_tensorsketch": 3 * 64,
            "odd_geometry": 3 * 8,
            "panel_conditional": 176,
            "total": 680,
        },
        "subspace": {
            "reward_channels": [
                "absolute_member_reward",
                "clean_prompt_difficulty",
                "within_prompt_member_effect",
                "pair_tangent_reward_difference",
            ],
            "label_free_channels": "four residual randomized-SVD directions",
        },
        "state_normalization": "clean/member RMS; tangent is exact PR5 predictor_inputs",
        "curvature": "((h+ + h-)/2 - h0)/(sigma^2*RMS(h0)); train-scale; winsor10",
        "snr_audit": snr,
        "panel_audit": panel_stats,
        "swap_negation_max_abs_error": swap_error,
        "swap_negation_max_relative_error": swap_relative_error,
        "swap_negation_rtol": 1e-3,
        "swap_negation_atol": 2e-4,
        "swap_negation_blocks": swap_block_errors,
        "features_sha256": _sha256_file(feature_path),
        "source_sha256": _sha256_file(Path(__file__)),
    }
    _atomic_json(output / "feature_manifest.json", manifest)
    return manifest


def evaluate_features(
    *,
    dataset_directory: str | Path,
    feature_directory: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    info = load_and_verify_dataset(dataset_directory, deep=False)
    search = info.directory / "offline_oracle_v1"
    split_path = search / "split.json"
    splits = load_group_split(split_path, info)
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    directory = Path(feature_directory).expanduser().resolve()
    result_path = directory / "evaluation.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"{result_path} exists; pass --overwrite")
    manifest = json.loads((directory / "feature_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_run_config_sha256") != info.run_config_sha256:
        raise ValueError("feature cache belongs to another source dataset")
    if manifest.get("split_sha256") != _canonical_sha256(split_payload):
        raise ValueError("feature cache belongs to another prompt split")
    features = np.load(directory / "features.npy", mmap_mode="r")
    if features.shape != (info.samples, info.pairs, PACT_FEATURE_DIM):
        raise ValueError("PACT feature cache has the wrong shape")
    train_ids = np.asarray(splits["train"], dtype=np.int64)
    validation_ids = np.asarray(splits["validation"], dtype=np.int64)
    test_ids = np.asarray(splits["test"], dtype=np.int64)
    train_x = np.asarray(features[train_ids], dtype=np.float32)
    validation_x = np.asarray(features[validation_ids], dtype=np.float32)
    test_x = np.asarray(features[test_ids], dtype=np.float32)

    targets = np.load(info.reward_differences_path, mmap_mode="r")
    train_y = np.asarray(targets[train_ids], dtype=np.float32)
    validation_y = np.asarray(targets[validation_ids], dtype=np.float32)

    configurations = (
        ("faithful_ridge10", train_x, validation_x, test_x),
        (
            "prompt_centered_huber_epsilon1p35_alpha1e-4",
            _prompt_center(train_x),
            _prompt_center(validation_x),
            _prompt_center(test_x),
        ),
    )
    predictions: dict[str, dict[str, Any]] = {}
    for identifier, fit_x, val_x, heldout_x in configurations:
        if identifier == "faithful_ridge10":
            model = _fit_ridge10(fit_x, train_y)
            raw_val = _predict_ridge(model, val_x)
            raw_test = _predict_ridge(model, heldout_x)
            fit_diagnostics = {"converged": True, "iterations": None}
        else:
            model = _fit_huber(fit_x, train_y)
            raw_val = _predict_huber(model, val_x)
            raw_test = _predict_huber(model, heldout_x)
            huber = model.named_steps["huberregressor"]
            fit_diagnostics = {
                "iterations": int(huber.n_iter_),
                "max_iterations": int(huber.max_iter),
                "converged": bool(huber.n_iter_ < huber.max_iter),
            }
        # Match the historical protocols exactly: faithful ridge uses identity
        # calibration; the prior-best Huber freezes one scale fitted on val.
        scale = (
            1.0
            if identifier == "faithful_ridge10"
            else _zero_intercept_scale(raw_val, validation_y)
        )
        predictions[identifier] = {
            "raw_validation": np.clip(raw_val, -PREDICTION_CLIP, PREDICTION_CLIP),
            "raw_test": np.clip(raw_test, -PREDICTION_CLIP, PREDICTION_CLIP),
            "legacy_validation": np.clip(
                scale * raw_val, -PREDICTION_CLIP, PREDICTION_CLIP
            ),
            "legacy_test": np.clip(
                scale * raw_test, -PREDICTION_CLIP, PREDICTION_CLIP
            ),
            "validation_scale": float(scale),
            "fit_diagnostics": fit_diagnostics,
        }
        _atomic_npy(
            directory / f"predictions_{identifier}.npy",
            np.stack(
                [predictions[identifier]["raw_test"], predictions[identifier]["legacy_test"]]
            ).astype(np.float32),
        )

    # Test predictions are frozen on disk before test labels are materialized.
    test_y = np.asarray(targets[test_ids], dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for identifier, prediction in predictions.items():
        rows.append(
            {
                "id": identifier,
                "validation_scale": prediction["validation_scale"],
                "fit_diagnostics": prediction["fit_diagnostics"],
                "validation_raw_metrics": pair_level_metrics(
                    validation_y, prediction["raw_validation"]
                ),
                "validation_legacy_calibrated_metrics": pair_level_metrics(
                    validation_y, prediction["legacy_validation"]
                ),
                "test_raw_metrics": pair_level_metrics(test_y, prediction["raw_test"]),
                "test_legacy_calibrated_metrics": pair_level_metrics(
                    test_y, prediction["legacy_test"]
                ),
            }
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "method": "PACT_680_v1",
        "feature_manifest_sha256": _canonical_sha256(manifest),
        "split_sha256": _canonical_sha256(split_payload),
        "test_predictions_written_before_test_labels_read": True,
        "validation_legacy_metrics_use_validation_label_scale": True,
        "test_status": "exploratory: this test split was inspected in earlier experiments",
        "models": rows,
    }
    _atomic_json(result_path, result)
    metric_names = (
        "r2_zero",
        "macro_prompt_cosine",
        "pooled_pearson",
        "mse",
        "nonzero_sign_accuracy",
        "top1_energy_recall",
        "top4_energy_recall",
        "top8_energy_recall",
    )
    with (directory / "evaluation.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["id", "view"] + list(metric_names)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for view in (
                "validation_raw_metrics",
                "validation_legacy_calibrated_metrics",
                "test_raw_metrics",
                "test_legacy_calibrated_metrics",
            ):
                writer.writerow(
                    {"id": row["id"], "view": view}
                    | {name: row[view][name] for name in metric_names}
                )
    return result


def _self_test() -> None:
    rng = np.random.default_rng(7)
    n, pairs, hidden = 10, 4, 32
    sigma = 0.01
    clean = rng.normal(size=(n, hidden)).astype(np.float32)
    displacement = rng.normal(scale=0.02, size=(n, pairs, hidden)).astype(np.float32)
    curvature = rng.normal(scale=0.001, size=(n, pairs, hidden)).astype(np.float32)
    members = np.stack(
        [clean[:, None] + displacement + curvature, clean[:, None] - displacement + curvature],
        axis=2,
    )
    center_rms = _rms(0.5 * (members[:, :, 0] + members[:, :, 1]))
    tangent = (members[:, :, 0] - members[:, :, 1]) / (2 * sigma * center_rms)
    rewards = rng.integers(0, 2, size=(n, pairs, 2)).astype(np.float32)
    layer_map = _fit_complete_layer_map(
        clean, members, tangent, rewards, sigma=sigma, layer=3, seed=11
    )
    block, _ = _layer_features(layer_map, clean, members, tangent, sigma=sigma)
    swapped, _ = _layer_features(
        layer_map, clean, members[:, :, ::-1], -tangent, sigma=sigma
    )
    np.testing.assert_allclose(swapped, -block, rtol=2e-5, atol=2e-5)
    assert block.shape == (n, pairs, LAYER_FEATURE_DIM)

    clean_logits = np.sort(rng.normal(size=(n, PANEL_SIZE)), axis=-1)[:, ::-1]
    clean_lse = np.log(np.sum(np.exp(clean_logits), axis=-1)) + 2.0
    member_logits = rng.normal(size=(n, pairs, 2, PANEL_SIZE))
    panel, stats = policy_features(clean_logits, clean_lse, member_logits)
    panel_swapped, _ = policy_features(clean_logits, clean_lse, member_logits[:, :, ::-1])
    np.testing.assert_allclose(panel_swapped, -panel, rtol=1e-5, atol=1e-5)
    assert panel.shape == (n, pairs, POLICY_FEATURE_DIM)
    assert 0.0 < stats["mean_clean_top96_mass"] < 1.0
    complete = np.concatenate([block, block, block, panel], axis=-1)
    assert complete.shape == (n, pairs, PACT_FEATURE_DIM)

    targets = rewards[..., 0] - rewards[..., 1]
    ridge = _fit_ridge10(complete[:8], targets[:8])
    assert _predict_ridge(ridge, complete[8:]).shape == (2, pairs)
    huber = _fit_huber(_prompt_center(complete[:8]), targets[:8])
    assert _predict_huber(huber, _prompt_center(complete[8:])).shape == (2, pairs)
    print("offline_countdown_pact self-test passed")


def _parse_layers(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    subparsers = parser.add_subparsers(dest="command")
    for name in ("build", "run"):
        child = subparsers.add_parser(name)
        child.add_argument("--dataset-directory", required=True)
        child.add_argument("--raw-directory", required=True)
        child.add_argument("--output-directory", required=True)
        child.add_argument("--candidate-layers")
        child.add_argument("--overwrite", action="store_true")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--dataset-directory", required=True)
    evaluate.add_argument("--feature-directory", required=True)
    evaluate.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        _self_test()
        return
    if args.command in ("build", "run"):
        build_features(
            dataset_directory=args.dataset_directory,
            raw_directory=args.raw_directory,
            output_directory=args.output_directory,
            candidate_layers=_parse_layers(args.candidate_layers),
            overwrite=args.overwrite,
        )
        if args.command == "run":
            evaluate_features(
                dataset_directory=args.dataset_directory,
                feature_directory=args.output_directory,
                overwrite=args.overwrite,
            )
        return
    if args.command == "evaluate":
        evaluate_features(
            dataset_directory=args.dataset_directory,
            feature_directory=args.feature_directory,
            overwrite=args.overwrite,
        )
        return
    parser.error("choose build, evaluate, run, or --self-test")


if __name__ == "__main__":
    main()
