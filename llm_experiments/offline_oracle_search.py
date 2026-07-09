"""Offline search for prompt-preview reward-difference oracles.

This module consumes the fixed dataset produced by
``collect_countdown_oracle_dataset.py``.  Model selection is grouped by
Countdown prompt: ``screen`` reads labels from only the train and validation
groups, writes frozen hyperparameter configurations, and ``test`` is the only
stage that indexes test labels.

For exact reproduction of the original analysis, prompt groups are
reward-stratified before the split and screen-time scalar calibration is fitted
on validation itself.  The groups are disjoint, but split assignment is not
label-blind and calibrated validation MSE/R2 are calibration diagnostics rather
than unbiased held-out estimates.  Only the later test metrics use predictions
fixed without test labels.

The faithful PR4 baseline is a no-intercept, coordinate-RMS-normalized ridge
regression over a fixed 256-dimensional CountSketch of layers 18 and 23.  The
ridge system uses mean Gram/cross statistics, matching
``OnlineRidgeSurrogate`` rather than sklearn's sum-Gram convention.
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
from tqdm.auto import tqdm

from llm_experiments.offline_oracle_utils import (
    DEFAULT_SPLIT_SEED,
    GROUP_SIGNAL_BLOCKS_V1,
    audit_group_split,
    group_signal_blocks_v1,
    pair_level_metrics,
)


SCHEMA_VERSION = 1
DEFAULT_LAMBDAS = (
    1e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
    3e-1,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
)
DEFAULT_CLIP = 1.1
DEFAULT_RMS_FLOOR = 1e-4
DEFAULT_FEATURE_SEED = 0
DEFAULT_RANDOM_FEATURE_SEED = 20260709
ODD_RANDOM_DIMENSION = 512
ODD_TANH_SCALES = (0.25, 0.5, 1.0, 2.0)


@dataclass(frozen=True)
class DatasetInfo:
    directory: Path
    samples: int
    pairs: int
    layers: int
    hidden: int
    run_config_sha256: str
    predictor_inputs_path: Path
    pair_rewards_path: Path
    reward_differences_path: Path


@dataclass(frozen=True)
class RidgeSystem:
    """One train-only RMS normalization and eigendecomposed mean Gram."""

    feature_rms: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    projected_cross: np.ndarray

    def weights(self, ridge: float) -> np.ndarray:
        if not np.isfinite(ridge) or ridge <= 0.0:
            raise ValueError("ridge must be finite and positive")
        return self.eigenvectors @ (
            self.projected_cross / (self.eigenvalues + float(ridge))
        )


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
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_header(path: Path, shape: tuple[int, ...], dtype: Any) -> np.memmap:
    if not path.is_file():
        raise FileNotFoundError(path)
    array = np.load(path, mmap_mode="r")
    expected_dtype = np.dtype(dtype)
    if array.shape != shape or array.dtype != expected_dtype:
        raise ValueError(
            f"{path.name} has {array.shape}/{array.dtype}; expected "
            f"{shape}/{expected_dtype}"
        )
    return array


def load_and_verify_dataset(directory: str | Path, *, deep: bool = False) -> DatasetInfo:
    """Validate the collector contract without eagerly reading reward labels."""

    directory = Path(directory).expanduser().resolve()
    run_config_path = directory / "run_config.json"
    manifest_path = directory / "manifest.json"
    metadata_path = directory / "metadata.json"
    for path in (run_config_path, manifest_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise ValueError(f"Dataset is not complete: status={metadata.get('status')!r}")

    samples = int(run_config["dataset"]["dataset_size"])
    pairs = int(run_config["perturbations"]["directions_per_prompt"])
    layers = int(run_config["model"]["num_layers"])
    hidden = int(run_config["model"]["hidden_size"])
    run_hash = str(manifest["run_config_sha256"])
    if metadata.get("run_config_sha256") != run_hash:
        raise ValueError("manifest and metadata run-config fingerprints disagree")
    if _sha256_json(run_config) != run_hash:
        raise ValueError("run_config.json does not match its recorded fingerprint")
    predictor_contract = run_config.get("predictor_input", {})
    if predictor_contract.get("countsketch_applied") is not False:
        raise ValueError("Expected normalized pre-CountSketch predictor inputs")
    if predictor_contract.get("layers") not in (None, list(range(layers))):
        raise ValueError("predictor inputs do not contain every model layer in order")
    recorded_floor = predictor_contract.get("center_rms_floor")
    if recorded_floor is not None and not np.isclose(
        float(recorded_floor), DEFAULT_RMS_FLOOR, rtol=0.0, atol=1e-12
    ):
        raise ValueError("Dataset center-RMS floor does not match the PR4 baseline")

    rows = samples * pairs * layers
    predictor_path = directory / "predictor_inputs.npy"
    rewards_path = directory / "pair_rewards.npy"
    differences_path = directory / "reward_differences.npy"
    _array_header(predictor_path, (rows, hidden), np.float32)
    _array_header(rewards_path, (samples, pairs, 2), np.float32)
    _array_header(differences_path, (samples, pairs), np.float32)
    completed = _array_header(
        directory / "completed_samples.npy", (samples,), np.uint8
    )
    if not np.all(np.asarray(completed) == 1):
        raise ValueError("completed_samples.npy is not all ones")

    row_index = np.load(directory / "row_index.npy", mmap_mode="r")
    required_fields = {
        "sample_id",
        "local_pair_id",
        "global_pair_id",
        "positive_member_id",
        "negative_member_id",
        "layer_id",
    }
    if row_index.shape != (rows,) or not required_fields.issubset(
        row_index.dtype.names or ()
    ):
        raise ValueError("row_index.npy has an incompatible schema")
    # These checks establish the reshape contract while touching only small
    # integer metadata, not test rewards.
    expected_samples = np.repeat(
        np.arange(samples, dtype=np.int32), pairs * layers
    )
    expected_pairs = np.tile(
        np.repeat(np.arange(pairs, dtype=np.int32), layers), samples
    )
    expected_layers = np.tile(np.arange(layers, dtype=np.int32), samples * pairs)
    if not np.array_equal(row_index["sample_id"], expected_samples):
        raise ValueError("row index is not sample-major")
    if not np.array_equal(row_index["local_pair_id"], expected_pairs):
        raise ValueError("row index is not pair-major within sample")
    if not np.array_equal(row_index["layer_id"], expected_layers):
        raise ValueError("row index is not layer-major within pair")

    if deep:
        rewards = np.asarray(np.load(rewards_path, mmap_mode="r"))
        differences = np.asarray(np.load(differences_path, mmap_mode="r"))
        if not np.all(np.isfinite(rewards)) or not np.all(np.isfinite(differences)):
            raise ValueError("reward arrays contain non-finite values")
        np.testing.assert_allclose(
            differences, rewards[..., 0] - rewards[..., 1], rtol=0.0, atol=1e-7
        )

    return DatasetInfo(
        directory=directory,
        samples=samples,
        pairs=pairs,
        layers=layers,
        hidden=hidden,
        run_config_sha256=run_hash,
        predictor_inputs_path=predictor_path,
        pair_rewards_path=rewards_path,
        reward_differences_path=differences_path,
    )


def make_countsketch(
    *, num_layers: int, hidden_size: int, sketch_size: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Exact NumPy reconstruction of PR4's fixed per-slot CountSketch maps."""

    if min(num_layers, hidden_size, sketch_size) < 1:
        raise ValueError("CountSketch dimensions must be positive")
    buckets = np.empty((num_layers, hidden_size), dtype=np.int32)
    signs = np.empty((num_layers, hidden_size), dtype=np.float32)
    for slot in range(num_layers):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), slot]))
        buckets[slot] = rng.integers(0, sketch_size, size=hidden_size)
        signs[slot] = rng.choice(
            np.asarray([-1.0, 1.0], dtype=np.float32), size=hidden_size
        )
    return buckets, signs


def _countsketch_values(
    values: np.ndarray,
    buckets: np.ndarray,
    signs: np.ndarray,
    sketch_size: int,
) -> np.ndarray:
    """Sketch ``[..., slots, hidden]`` normalized central differences."""

    values = np.asarray(values, dtype=np.float32)
    if values.ndim < 2:
        raise ValueError("values must end in [slots, hidden]")
    slots, hidden = values.shape[-2:]
    if buckets.shape != (slots, hidden) or signs.shape != buckets.shape:
        raise ValueError("CountSketch maps do not match values")
    output = np.empty(values.shape[:-1] + (sketch_size,), dtype=np.float32)
    for slot in range(slots):
        slot_values = values[..., slot, :]
        for bucket in range(sketch_size):
            selected = buckets[slot] == bucket
            output[..., slot, bucket] = np.sum(
                slot_values[..., selected] * signs[slot, selected],
                axis=-1,
                dtype=np.float32,
            )
    return output


def _cache_specs(info: DatasetInfo, seed: int) -> dict[str, dict[str, Any]]:
    late = min(info.layers - 1, int(math.floor(0.75 * info.layers)))
    final = info.layers - 1
    late_eight = list(range(max(0, info.layers - 8), info.layers))
    return {
        "pr4_late_final_cs128": {
            "layers": [late, final],
            "sketch_size": 128,
            "seed": seed,
            "shape": [info.samples, info.pairs, 256],
        },
        "single_layers_cs128": {
            "layers": list(range(info.layers)),
            "sketch_size": 128,
            "seed": seed,
            # A single-layer candidate has one feature slot, so recreating
            # make_countsketch(num_layers=1) for every candidate uses slot 0.
            # Sharing that map also avoids confounding the layer comparison
            # with 24 different random projections.
            "shared_map": True,
            "shape": [info.samples, info.pairs, info.layers, 128],
        },
        "all_layers_cs32": {
            "layers": list(range(info.layers)),
            "sketch_size": 32,
            "seed": seed,
            "shape": [info.samples, info.pairs, info.layers * 32],
        },
        "late8_cs64": {
            "layers": late_eight,
            "sketch_size": 64,
            "seed": seed,
            "shape": [info.samples, info.pairs, len(late_eight) * 64],
        },
    }


def _feature_path(output_directory: Path, name: str) -> Path:
    return output_directory / "features" / f"{name}.npy"


def _valid_cache(path: Path, shape: Sequence[int]) -> bool:
    if not path.is_file():
        return False
    array = np.load(path, mmap_mode="r")
    return array.shape == tuple(shape) and array.dtype == np.float32


def _build_sketch_cache(
    *,
    raw: np.ndarray,
    path: Path,
    spec: Mapping[str, Any],
    chunk_samples: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.npy")
    temporary.unlink(missing_ok=True)
    output = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=tuple(spec["shape"])
    )
    layers = np.asarray(spec["layers"], dtype=np.int32)
    map_count = 1 if bool(spec.get("shared_map", False)) else layers.size
    buckets, signs = make_countsketch(
        num_layers=map_count,
        hidden_size=raw.shape[-1],
        sketch_size=int(spec["sketch_size"]),
        seed=int(spec["seed"]),
    )
    if map_count == 1 and layers.size > 1:
        buckets = np.broadcast_to(buckets, (layers.size, raw.shape[-1]))
        signs = np.broadcast_to(signs, (layers.size, raw.shape[-1]))
    starts = tqdm(
        range(0, raw.shape[0], chunk_samples),
        desc=f"prepare {path.stem}",
        unit="chunk",
    )
    for start in starts:
        stop = min(raw.shape[0], start + chunk_samples)
        selected = np.asarray(raw[start:stop][:, :, layers, :], dtype=np.float32)
        if not np.all(np.isfinite(selected)):
            raise ValueError("predictor inputs contain non-finite values")
        sketched = _countsketch_values(
            selected, buckets, signs, int(spec["sketch_size"])
        )
        if len(spec["shape"]) == 3:
            sketched = sketched.reshape(stop - start, raw.shape[1], -1)
        output[start:stop] = sketched
    output.flush()
    del output
    temporary.replace(path)


def _build_odd_projection_cache(
    baseline_path: Path,
    path: Path,
    *,
    random_seed: int,
    random_dimension: int,
    chunk_samples: int,
) -> dict[str, Any]:
    baseline = np.load(baseline_path, mmap_mode="r")
    input_dim = baseline.shape[-1]
    rng = np.random.default_rng(
        np.random.SeedSequence([int(random_seed), int(input_dim), random_dimension])
    )
    projection = rng.choice(
        np.asarray([-1.0, 1.0], dtype=np.float32),
        size=(input_dim, random_dimension),
    ) / np.float32(np.sqrt(input_dim))
    shape = baseline.shape[:-1] + (random_dimension,)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.npy")
    temporary.unlink(missing_ok=True)
    output = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=shape
    )
    for start in range(0, baseline.shape[0], chunk_samples):
        stop = min(baseline.shape[0], start + chunk_samples)
        output[start:stop] = np.asarray(baseline[start:stop]) @ projection
    output.flush()
    del output
    temporary.replace(path)
    return {
        "source": "pr4_late_final_cs128",
        "random_seed": random_seed,
        "random_dimension": random_dimension,
        "projection": "Rademacher/sqrt(input_dimension), no bias",
        "shape": list(shape),
    }


def _split_payload(info: DatasetInfo, split_path: Path, *, force: bool) -> dict[str, Any]:
    if split_path.is_file() and not force:
        return json.loads(split_path.read_text(encoding="utf-8"))
    # Split construction is an explicit prepare-time operation.  Screen itself
    # later indexes only train/validation labels from this frozen JSON.
    pair_rewards = np.asarray(np.load(info.pair_rewards_path, mmap_mode="r"))
    sample_ids = np.arange(info.samples, dtype=np.int32)
    split_arrays = group_signal_blocks_v1(
        pair_rewards,
        sample_ids,
        seed=DEFAULT_SPLIT_SEED,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": GROUP_SIGNAL_BLOCKS_V1,
        "seed": DEFAULT_SPLIT_SEED,
        "dataset_run_config_sha256": info.run_config_sha256,
        "assignment_uses_reward_labels": True,
        "test_assignment_label_blind": False,
        "train": np.asarray(split_arrays["train"], dtype=np.int32).tolist(),
        "validation": np.asarray(
            split_arrays["validation"], dtype=np.int32
        ).tolist(),
        "test": np.asarray(split_arrays["test"], dtype=np.int32).tolist(),
    }
    _atomic_json(split_path, payload)
    # Keep label-derived balance diagnostics separate.  Screen reads only
    # split.json and consequently cannot even observe aggregate test labels.
    _atomic_json(
        split_path.with_name("split_audit.json"),
        {
            "schema_version": SCHEMA_VERSION,
            "dataset_run_config_sha256": info.run_config_sha256,
            "audit": audit_group_split(pair_rewards, split_arrays, sample_ids),
        },
    )
    return payload


def load_group_split(
    split_path: str | Path, info: DatasetInfo
) -> dict[str, np.ndarray]:
    payload = json.loads(Path(split_path).read_text(encoding="utf-8"))
    if payload.get("algorithm") != GROUP_SIGNAL_BLOCKS_V1:
        raise ValueError("Unexpected split algorithm")
    if payload.get("dataset_run_config_sha256") != info.run_config_sha256:
        raise ValueError("Split belongs to a different dataset")
    splits: dict[str, np.ndarray] = {}
    for name in ("train", "validation", "test"):
        values = np.asarray(payload[name], dtype=np.int32)
        if values.ndim != 1 or values.size == 0 or np.unique(values).size != values.size:
            raise ValueError(f"Invalid {name} sample IDs")
        if values.min() < 0 or values.max() >= info.samples:
            raise ValueError(f"{name} sample ID outside dataset")
        splits[name] = values
    if (
        np.intersect1d(splits["train"], splits["validation"]).size
        or np.intersect1d(splits["train"], splits["test"]).size
        or np.intersect1d(splits["validation"], splits["test"]).size
    ):
        raise ValueError("Prompt groups overlap")
    return splits


def prepare(
    info: DatasetInfo,
    output_directory: Path,
    *,
    feature_seed: int,
    random_feature_seed: int,
    chunk_samples: int,
    force: bool,
) -> dict[str, Any]:
    if chunk_samples < 1:
        raise ValueError("chunk_samples must be positive")
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "feature_manifest.json"
    expected_identity = {
        "schema_version": SCHEMA_VERSION,
        "dataset_run_config_sha256": info.run_config_sha256,
        "feature_seed": feature_seed,
        "random_feature_seed": random_feature_seed,
    }
    if manifest_path.is_file() and not force:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in expected_identity.items():
            if previous.get(key) != value:
                raise ValueError(
                    "Existing feature cache has different settings; use --force-prepare"
                )

    raw = np.load(info.predictor_inputs_path, mmap_mode="r").reshape(
        info.samples, info.pairs, info.layers, info.hidden
    )
    specs = _cache_specs(info, feature_seed)
    for name, spec in specs.items():
        path = _feature_path(output_directory, name)
        if force:
            path.unlink(missing_ok=True)
        if not _valid_cache(path, spec["shape"]):
            _build_sketch_cache(
                raw=raw,
                path=path,
                spec=spec,
                chunk_samples=chunk_samples,
            )

    odd_path = _feature_path(output_directory, "odd_pr4_projection512")
    odd_shape = (info.samples, info.pairs, ODD_RANDOM_DIMENSION)
    if force:
        odd_path.unlink(missing_ok=True)
    if not _valid_cache(odd_path, odd_shape):
        odd_spec = _build_odd_projection_cache(
            _feature_path(output_directory, "pr4_late_final_cs128"),
            odd_path,
            random_seed=random_feature_seed,
            random_dimension=ODD_RANDOM_DIMENSION,
            chunk_samples=chunk_samples,
        )
    else:
        odd_spec = {
            "source": "pr4_late_final_cs128",
            "random_seed": random_feature_seed,
            "random_dimension": ODD_RANDOM_DIMENSION,
            "projection": "Rademacher/sqrt(input_dimension), no bias",
            "shape": list(odd_shape),
        }

    split_path = output_directory / "split.json"
    split = _split_payload(info, split_path, force=force)
    manifest = expected_identity | {
        "created_at_unix": time.time(),
        "dataset_directory": str(info.directory),
        "samples": info.samples,
        "pairs": info.pairs,
        "layers": info.layers,
        "hidden": info.hidden,
        "features": specs | {"odd_pr4_projection512": odd_spec},
        "split_json": str(split_path),
        "split_sha256": _sha256_json(split),
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def fit_ridge_system(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    rms_floor: float = DEFAULT_RMS_FLOOR,
) -> RidgeSystem:
    """Fit the shared portion of a repository-equivalent one-shot ridge path."""

    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if features.ndim != 2 or targets.shape != (features.shape[0],):
        raise ValueError("Ridge features/targets have incompatible shapes")
    if features.shape[0] == 0 or not np.all(np.isfinite(features)):
        raise ValueError("Ridge features must be nonempty and finite")
    if not np.all(np.isfinite(targets)) or rms_floor <= 0.0:
        raise ValueError("Ridge targets and RMS floor must be finite/valid")
    rms = np.maximum(np.sqrt(np.mean(features * features, axis=0)), rms_floor)
    normalized = features / rms
    count = float(features.shape[0])
    gram = (normalized.T @ normalized) / count
    cross = (normalized.T @ targets) / count
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    return RidgeSystem(
        feature_rms=rms,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        projected_cross=eigenvectors.T @ cross,
    )


def fit_one_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    ridge: float,
    *,
    rms_floor: float = DEFAULT_RMS_FLOOR,
) -> tuple[np.ndarray, np.ndarray]:
    """Direct mean-Gram solve used when a frozen test config is refit once."""

    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    rms = np.maximum(np.sqrt(np.mean(features * features, axis=0)), rms_floor)
    normalized = features / rms
    count = float(features.shape[0])
    gram = (normalized.T @ normalized) / count
    cross = (normalized.T @ targets) / count
    weights = np.linalg.solve(
        gram + float(ridge) * np.eye(features.shape[1], dtype=np.float64), cross
    )
    return rms, weights


def _predict(
    features: np.ndarray,
    rms: np.ndarray,
    weights: np.ndarray,
    clip: float | None,
) -> np.ndarray:
    predictions = (np.asarray(features, dtype=np.float64) / rms) @ weights
    if clip is not None:
        predictions = np.clip(predictions, -float(clip), float(clip))
    return predictions.astype(np.float32)


def validation_calibration_scale(
    predictions: np.ndarray, targets: np.ndarray
) -> float:
    """Return the zero-intercept least-squares scale fit on validation only."""

    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    if predictions.shape != targets.shape or not np.all(np.isfinite(predictions)):
        raise ValueError("Calibration predictions/targets are invalid")
    if not np.all(np.isfinite(targets)):
        raise ValueError("Calibration targets are invalid")
    denominator = float(predictions @ predictions)
    if denominator <= 1e-12:
        # Any scale gives the same all-zero prediction; identity is the least
        # surprising frozen value and preserves the faithful baseline meaning.
        return 1.0
    scale = float((predictions @ targets) / denominator)
    return scale if np.isfinite(scale) else 1.0


def _apply_calibration_and_clip(
    predictions: np.ndarray, scale: float, clip: float | None
) -> np.ndarray:
    calibrated = np.asarray(predictions, dtype=np.float64) * float(scale)
    if clip is not None:
        calibrated = np.clip(calibrated, -float(clip), float(clip))
    return calibrated.astype(np.float32)


def _feature_array(
    output_directory: Path,
    config: Mapping[str, Any],
    sample_ids: np.ndarray,
) -> np.ndarray:
    name = str(config["feature_set"])
    array = np.load(_feature_path(output_directory, name), mmap_mode="r")
    if name == "single_layers_cs128":
        # Slice before materializing so a single-layer trial does not read the
        # full 96 MiB layer bank on every sweep iteration.
        selected = np.asarray(
            array[sample_ids, :, int(config["layer"]), :], dtype=np.float32
        )
    else:
        selected = np.asarray(array[sample_ids], dtype=np.float32)
    if name == "odd_pr4_projection512":
        selected = np.tanh(np.float32(config["tanh_scale"]) * selected)
    return selected.reshape(-1, selected.shape[-1])


def _config(
    *,
    identifier: str,
    family: str,
    feature_set: str,
    ridge: float,
    clip: float | None = DEFAULT_CLIP,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "family": family,
        "feature_set": feature_set,
        "ridge": float(ridge),
        "clip": None if clip is None else float(clip),
        "rms_floor": DEFAULT_RMS_FLOOR,
    } | extra


def _evaluate_path(
    *,
    output_directory: Path,
    base_config: Mapping[str, Any],
    lambdas: Sequence[float],
    train_ids: np.ndarray,
    validation_ids: np.ndarray,
    y_train: np.ndarray,
    y_validation: np.ndarray,
) -> list[dict[str, Any]]:
    train_x = _feature_array(output_directory, base_config, train_ids)
    validation_x = _feature_array(output_directory, base_config, validation_ids)
    system = fit_ridge_system(
        train_x, y_train.reshape(-1), rms_floor=float(base_config["rms_floor"])
    )
    results: list[dict[str, Any]] = []
    for ridge in lambdas:
        config = dict(base_config) | {
            "id": f"{base_config['id']}_ridge{float(ridge):g}",
            "ridge": float(ridge),
        }
        raw_predictions = _predict(
            validation_x,
            system.feature_rms,
            system.weights(float(ridge)),
            None,
        ).reshape(y_validation.shape)
        calibration_scale = (
            validation_calibration_scale(raw_predictions, y_validation)
            if bool(config.get("fit_validation_calibration", True))
            else 1.0
        )
        config["calibration_scale"] = calibration_scale
        predictions = _apply_calibration_and_clip(
            raw_predictions, calibration_scale, config["clip"]
        )
        metrics = pair_level_metrics(y_validation, predictions)
        results.append({"config": config, "metrics": metrics})
    return results


def _rank_key(result: Mapping[str, Any]) -> tuple[float, float, float]:
    metrics = result["metrics"]
    primary = float(metrics.get("macro_prompt_cosine", float("nan")))
    secondary = float(metrics.get("r2_zero", float("nan")))
    if not np.isfinite(primary):
        primary = -float("inf")
    if not np.isfinite(secondary):
        secondary = -float("inf")
    # Deterministically prefer less regularization after metric ties.
    return primary, secondary, -float(result["config"]["ridge"])


def _write_results_csv(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    metric_names = sorted(
        {name for result in results for name in result["metrics"].keys()}
    )
    fields = [
        "id",
        "family",
        "feature_set",
        "ridge",
        "clip",
        "calibration_scale",
        "config_json",
    ] + metric_names
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            config = result["config"]
            writer.writerow(
                {
                    "id": config["id"],
                    "family": config["family"],
                    "feature_set": config["feature_set"],
                    "ridge": config["ridge"],
                    "clip": config["clip"],
                    "calibration_scale": config.get("calibration_scale"),
                    "config_json": json.dumps(config, sort_keys=True),
                }
                | {name: result["metrics"].get(name) for name in metric_names}
            )
    temporary.replace(path)


def screen(
    info: DatasetInfo,
    output_directory: Path,
    *,
    lambdas: Sequence[float],
) -> dict[str, Any]:
    """Select hyperparameters without ever indexing ``y[test]``."""

    manifest_path = output_directory / "feature_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Run prepare before screen")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_run_config_sha256") != info.run_config_sha256:
        raise ValueError("Feature cache belongs to a different dataset")
    splits = load_group_split(output_directory / "split.json", info)
    feature_seed = int(manifest["feature_seed"])
    random_feature_seed = int(manifest["random_feature_seed"])

    # Keep this memmapped.  These are the only label-indexing operations in
    # screen; the test IDs are intentionally never applied to this array.
    all_targets = np.load(info.reward_differences_path, mmap_mode="r")
    y_train = np.asarray(all_targets[splits["train"]], dtype=np.float32)
    y_validation = np.asarray(
        all_targets[splits["validation"]], dtype=np.float32
    )

    late = min(info.layers - 1, int(math.floor(0.75 * info.layers)))
    results: list[dict[str, Any]] = []

    pr4_base = _config(
        identifier="pr4_late_final_cs128",
        family="pr4_tuned",
        feature_set="pr4_late_final_cs128",
        ridge=10.0,
        layers=[late, info.layers - 1],
        sketch_size=128,
        feature_seed=feature_seed,
        fit_validation_calibration=True,
    )
    pr4_path = _evaluate_path(
        output_directory=output_directory,
        base_config=pr4_base,
        lambdas=tuple(sorted(set(float(value) for value in lambdas) | {10.0})),
        train_ids=splits["train"],
        validation_ids=splits["validation"],
        y_train=y_train,
        y_validation=y_validation,
    )
    results.extend(pr4_path)
    faithful_base = _config(
        identifier="faithful_pr4_l18_l23_b128",
        family="faithful_pr4_baseline",
        feature_set="pr4_late_final_cs128",
        ridge=10.0,
        layers=[late, info.layers - 1],
        sketch_size=128,
        feature_seed=feature_seed,
        fit_validation_calibration=False,
    )
    faithful = _evaluate_path(
        output_directory=output_directory,
        base_config=faithful_base,
        lambdas=(10.0,),
        train_ids=splits["train"],
        validation_ids=splits["validation"],
        y_train=y_train,
        y_validation=y_validation,
    )[0]
    results.append(faithful)

    for layer in range(info.layers):
        base = _config(
            identifier=f"single_layer{layer}_cs128",
            family="single_layer",
            feature_set="single_layers_cs128",
            ridge=10.0,
            layer=layer,
            sketch_size=128,
            feature_seed=feature_seed,
            fit_validation_calibration=True,
        )
        results.extend(
            _evaluate_path(
                output_directory=output_directory,
                base_config=base,
                lambdas=lambdas,
                train_ids=splits["train"],
                validation_ids=splits["validation"],
                y_train=y_train,
                y_validation=y_validation,
            )
        )

    for family, feature_set, sketch_size, layers in (
        ("all_layers_cs32", "all_layers_cs32", 32, list(range(info.layers))),
        (
            "late8_cs64",
            "late8_cs64",
            64,
            list(range(max(0, info.layers - 8), info.layers)),
        ),
    ):
        base = _config(
            identifier=feature_set,
            family=family,
            feature_set=feature_set,
            ridge=10.0,
            layers=layers,
            sketch_size=sketch_size,
            feature_seed=feature_seed,
            fit_validation_calibration=True,
        )
        results.extend(
            _evaluate_path(
                output_directory=output_directory,
                base_config=base,
                lambdas=lambdas,
                train_ids=splits["train"],
                validation_ids=splits["validation"],
                y_train=y_train,
                y_validation=y_validation,
            )
        )

    for scale in ODD_TANH_SCALES:
        base = _config(
            identifier=f"odd_tanh512_scale{scale:g}",
            family="odd_tanh_ridge",
            feature_set="odd_pr4_projection512",
            ridge=10.0,
            tanh_scale=scale,
            random_dimension=ODD_RANDOM_DIMENSION,
            random_feature_seed=random_feature_seed,
            no_bias=True,
            fit_validation_calibration=True,
        )
        results.extend(
            _evaluate_path(
                output_directory=output_directory,
                base_config=base,
                lambdas=lambdas,
                train_ids=splits["train"],
                validation_ids=splits["validation"],
                y_train=y_train,
                y_validation=y_validation,
            )
        )

    families = (
        "pr4_tuned",
        "single_layer",
        "all_layers_cs32",
        "late8_cs64",
        "odd_tanh_ridge",
    )
    selected = [faithful]
    for family in families:
        selected.append(
            max(
                (result for result in results if result["config"]["family"] == family),
                key=_rank_key,
            )
        )

    result_payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": "screen",
        "dataset_run_config_sha256": info.run_config_sha256,
        "split_sha256": _sha256_json(
            json.loads((output_directory / "split.json").read_text(encoding="utf-8"))
        ),
        "selection_metric": "macro_prompt_cosine",
        "tie_break_metric": "r2_zero",
        "train_sample_ids": splits["train"],
        "validation_sample_ids": splits["validation"],
        "test_labels_read": False,
        "validation_calibration_uses_validation_labels": True,
        "validation_metrics_strictly_heldout": False,
        "lambda_grid": list(lambdas),
        "results": results,
        "selected": selected,
    }
    _atomic_json(output_directory / "screen_results.json", result_payload)
    _write_results_csv(output_directory / "screen_results.csv", results)
    frozen = {
        "schema_version": SCHEMA_VERSION,
        "dataset_run_config_sha256": info.run_config_sha256,
        "split_sha256": result_payload["split_sha256"],
        "selection_metric": "macro_prompt_cosine",
        "tie_break_metric": "r2_zero",
        "test_protocol": (
            "fit ridge on train only; apply the validation-frozen zero-intercept "
            "calibration scale; then evaluate test"
        ),
        "configs": [result["config"] for result in selected],
    }
    _atomic_json(output_directory / "frozen_configs.json", frozen)
    return result_payload


def test_frozen(info: DatasetInfo, output_directory: Path) -> dict[str, Any]:
    """Fit train-only frozen winners, then and only then read test labels."""

    frozen_path = output_directory / "frozen_configs.json"
    if not frozen_path.is_file():
        raise FileNotFoundError("Run screen before test")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    splits = load_group_split(output_directory / "split.json", info)
    split_payload = json.loads(
        (output_directory / "split.json").read_text(encoding="utf-8")
    )
    if frozen.get("split_sha256") != _sha256_json(split_payload):
        raise ValueError("Frozen configurations do not match the split")
    configs = list(frozen["configs"])
    all_targets = np.load(info.reward_differences_path, mmap_mode="r")
    y_fit = np.asarray(all_targets[splits["train"]], dtype=np.float32)

    # Fit every model before the first test-label indexing operation.
    fitted: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
    for config in configs:
        if "calibration_scale" not in config:
            raise ValueError("Frozen config is missing validation calibration scale")
        fit_x = _feature_array(output_directory, config, splits["train"])
        rms, weights = fit_one_ridge(
            fit_x,
            y_fit.reshape(-1),
            float(config["ridge"]),
            rms_floor=float(config["rms_floor"]),
        )
        fitted.append((config, rms, weights))

    y_test = np.asarray(all_targets[splits["test"]], dtype=np.float32)
    models_directory = output_directory / "models"
    models_directory.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for config, rms, weights in fitted:
        test_x = _feature_array(output_directory, config, splits["test"])
        raw_predictions = _predict(test_x, rms, weights, None).reshape(y_test.shape)
        predictions = _apply_calibration_and_clip(
            raw_predictions,
            float(config["calibration_scale"]),
            config["clip"],
        )
        metrics = pair_level_metrics(y_test, predictions)
        artifact = models_directory / f"{config['id']}.npz"
        np.savez_compressed(
            artifact,
            feature_rms=rms,
            weights=weights,
            config_json=np.asarray(json.dumps(config, sort_keys=True)),
        )
        results.append(
            {"config": config, "metrics": metrics, "model_artifact": str(artifact)}
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": "test",
        "dataset_run_config_sha256": info.run_config_sha256,
        "test_sample_ids": splits["test"],
        "fit_sample_ids": splits["train"],
        "calibration_sample_ids": splits["validation"],
        "configs_frozen_before_test_labels": True,
        "results": results,
    }
    _atomic_json(output_directory / "test_results.json", payload)
    _write_results_csv(output_directory / "test_results.csv", results)
    return payload


def _parse_lambdas(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(not np.isfinite(item) or item <= 0.0 for item in values):
        raise argparse.ArgumentTypeError("lambda grid must contain positive numbers")
    return tuple(sorted(set(values)))


def self_test() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(40, 6)).astype(np.float32)
    y = rng.normal(size=40).astype(np.float32)
    ridge = 0.7
    system = fit_ridge_system(x, y, rms_floor=1e-8)
    path_weights = system.weights(ridge)
    direct_rms, direct_weights = fit_one_ridge(x, y, ridge, rms_floor=1e-8)
    np.testing.assert_allclose(system.feature_rms, direct_rms, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(path_weights, direct_weights, rtol=1e-9, atol=1e-9)
    calibration = validation_calibration_scale(
        np.asarray([1.0, -2.0]), np.asarray([2.0, -4.0])
    )
    np.testing.assert_allclose(calibration, 2.0)
    np.testing.assert_allclose(
        _apply_calibration_and_clip(np.asarray([1.0, -2.0]), calibration, 1.1),
        [1.1, -1.1],
    )

    values = rng.normal(size=(3, 2, 5)).astype(np.float32)
    buckets, signs = make_countsketch(
        num_layers=2, hidden_size=5, sketch_size=3, seed=11
    )
    sketched = _countsketch_values(values, buckets, signs, 3)
    expected = np.zeros((3, 2, 3), dtype=np.float32)
    for row in range(3):
        for slot in range(2):
            np.add.at(
                expected[row, slot], buckets[slot], values[row, slot] * signs[slot]
            )
    np.testing.assert_allclose(sketched, expected, rtol=0.0, atol=1e-6)
    odd_projection = rng.normal(size=(6, 9)).astype(np.float32)
    np.testing.assert_allclose(
        np.tanh((-x) @ odd_projection),
        -np.tanh(x @ odd_projection),
        rtol=1e-6,
        atol=1e-6,
    )
    print("offline_oracle_search self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "screen", "test", "all", "self-test")
    )
    parser.add_argument(
        "--dataset-directory",
        default="outputs/countdown_oracle_q35_2b_D256_P32_seed0",
    )
    parser.add_argument(
        "--output-directory",
        default=None,
        help="Default: DATASET_DIRECTORY/offline_oracle_v1",
    )
    parser.add_argument("--feature-seed", type=int, default=DEFAULT_FEATURE_SEED)
    parser.add_argument(
        "--random-feature-seed", type=int, default=DEFAULT_RANDOM_FEATURE_SEED
    )
    parser.add_argument("--chunk-samples", type=int, default=16)
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument(
        "--lambda-grid",
        type=_parse_lambdas,
        default=DEFAULT_LAMBDAS,
        help="Comma-separated positive ridge coefficients",
    )
    args = parser.parse_args()
    if args.stage == "self-test":
        self_test()
        return

    info = load_and_verify_dataset(
        args.dataset_directory, deep=args.stage in ("prepare", "all")
    )
    output_directory = (
        Path(args.output_directory).expanduser().resolve()
        if args.output_directory is not None
        else info.directory / "offline_oracle_v1"
    )
    if args.stage in ("prepare", "all"):
        prepare(
            info,
            output_directory,
            feature_seed=args.feature_seed,
            random_feature_seed=args.random_feature_seed,
            chunk_samples=args.chunk_samples,
            force=args.force_prepare,
        )
    if args.stage in ("screen", "all"):
        screen(info, output_directory, lambdas=args.lambda_grid)
    if args.stage in ("test", "all"):
        test_frozen(info, output_directory)


if __name__ == "__main__":
    main()
