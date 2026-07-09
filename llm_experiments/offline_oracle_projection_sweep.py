"""Validation-only CountSketch projection sweep for the Countdown oracle.

The sweep reconstructs PR4's exact two-slot CountSketch on normalized layer-18
and layer-23 central differences for bucket sizes 64/128/256/512 and seeds
0/1/2/3.  Identity and prompt-centered views are evaluated with the exact six
prompt folds used by ``offline_oracle_classical`` and an external frozen
validation split.  Test labels are never indexed in this module.

For historical reproducibility, each fold and validation receive a scalar
calibration fitted on that same evaluation partition.  Calibrated MSE/R2 are
therefore diagnostics, not strictly held-out estimates; the separate test
driver never calibrates on test labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from tqdm.auto import tqdm

from llm_experiments.offline_oracle_classical import (
    CV_ALGORITHM,
    CV_SEED,
    _load_frozen_data,
    make_six_prompt_folds,
)
from llm_experiments.offline_oracle_search import make_countsketch
from llm_experiments.offline_oracle_utils import pair_level_metrics


SCHEMA_VERSION = 1
LAYERS = (18, 23)
SKETCH_SIZES = (64, 128, 256, 512)
SKETCH_SEEDS = (0, 1, 2, 3)
PREPROCESSINGS = ("identity", "prompt_center")
RIDGES = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
RMS_FLOOR = 1e-4
PREDICTION_CLIP = 1.1
SELECTION_METRIC = "macro_prompt_cosine"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
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
            _json_safe(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _feature_name(sketch_size: int, seed: int) -> str:
    return f"l18_l23_cs{int(sketch_size)}_seed{int(seed)}"


def _feature_path(output: Path, sketch_size: int, seed: int) -> Path:
    return output / "features" / f"{_feature_name(sketch_size, seed)}.npy"


def _countsketch_two_layers(
    values: np.ndarray,
    buckets: np.ndarray,
    signs: np.ndarray,
    sketch_size: int,
) -> np.ndarray:
    """Apply exact signed bucket sums to ``[..., 2, hidden]`` values."""

    values = np.asarray(values, dtype=np.float32)
    if values.shape[-2] != 2 or buckets.shape != values.shape[-2:]:
        raise ValueError("CountSketch inputs have incompatible shapes")
    if signs.shape != buckets.shape:
        raise ValueError("CountSketch signs have an incompatible shape")
    output = np.empty(values.shape[:-1] + (sketch_size,), dtype=np.float32)
    for slot in range(2):
        slot_values = values[..., slot, :]
        for bucket in range(sketch_size):
            selected = buckets[slot] == bucket
            output[..., slot, bucket] = np.sum(
                slot_values[..., selected] * signs[slot, selected],
                axis=-1,
                dtype=np.float32,
            )
    return output.reshape(values.shape[:-2] + (2 * sketch_size,))


def prepare_projection_caches(
    *,
    raw_inputs_path: Path,
    output: Path,
    samples: int,
    pairs: int,
    layers: int,
    hidden: int,
    chunk_samples: int,
    overwrite: bool,
) -> dict[str, Any]:
    if layers != 24 or hidden < 1 or chunk_samples < 1:
        raise ValueError("Projection sweep expects 24 layers and a positive chunk size")
    raw = np.load(raw_inputs_path, mmap_mode="r")
    if raw.shape != (samples * pairs * layers, hidden) or raw.dtype != np.float32:
        raise ValueError("predictor_inputs.npy has an incompatible contract")
    raw = raw.reshape(samples, pairs, layers, hidden)
    (output / "features").mkdir(parents=True, exist_ok=True)

    specs: list[dict[str, Any]] = []
    pending: list[tuple[dict[str, Any], Path, Path, np.memmap]] = []
    for sketch_size in SKETCH_SIZES:
        for seed in SKETCH_SEEDS:
            path = _feature_path(output, sketch_size, seed)
            shape = (samples, pairs, 2 * sketch_size)
            spec = {
                "name": _feature_name(sketch_size, seed),
                "layers": list(LAYERS),
                "sketch_size": sketch_size,
                "seed": seed,
                "shape": list(shape),
                "path": path,
            }
            specs.append(spec)
            if path.is_file() and not overwrite:
                cached = np.load(path, mmap_mode="r")
                if cached.shape != shape or cached.dtype != np.float32:
                    raise ValueError(f"Incompatible existing cache {path}")
                continue
            path.unlink(missing_ok=True)
            temporary = path.with_name(f".{path.stem}.tmp.npy")
            temporary.unlink(missing_ok=True)
            array = np.lib.format.open_memmap(
                temporary, mode="w+", dtype=np.float32, shape=shape
            )
            pending.append((spec, path, temporary, array))

    if pending:
        maps = {
            (int(spec["sketch_size"]), int(spec["seed"])): make_countsketch(
                num_layers=2,
                hidden_size=hidden,
                sketch_size=int(spec["sketch_size"]),
                seed=int(spec["seed"]),
            )
            for spec, _, _, _ in pending
        }
        for start in tqdm(
            range(0, samples, chunk_samples),
            desc="projection feature caches",
            unit="chunk",
        ):
            stop = min(samples, start + chunk_samples)
            selected = np.asarray(raw[start:stop, :, LAYERS, :], dtype=np.float32)
            if not np.all(np.isfinite(selected)):
                raise ValueError("selected predictor inputs contain non-finite values")
            for spec, _, _, array in pending:
                size = int(spec["sketch_size"])
                seed = int(spec["seed"])
                buckets, signs = maps[(size, seed)]
                array[start:stop] = _countsketch_two_layers(
                    selected, buckets, signs, size
                )
        for _, path, temporary, array in pending:
            array.flush()
            # Release the mapping retained by ``pending`` before publishing.
            array._mmap.close()  # type: ignore[attr-defined]
            temporary.replace(path)
    return {spec["name"]: spec for spec in specs}


def _transform(features: np.ndarray, preprocessing: str) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 3 or not np.all(np.isfinite(values)):
        raise ValueError("features must be finite [prompts,pairs,features]")
    if preprocessing == "identity":
        return values.copy()
    if preprocessing == "prompt_center":
        output = values.copy()
        output -= np.mean(output, axis=1, keepdims=True, dtype=np.float32)
        return output
    raise ValueError(f"Unknown preprocessing {preprocessing!r}")


def _fit_ridge_path(
    fit_features: np.ndarray,
    fit_targets: np.ndarray,
    predict_features: np.ndarray,
    ridges: Sequence[float] = RIDGES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit all mean-Gram ridge values with sklearn's equivalent sum alphas."""

    x = np.asarray(fit_features, dtype=np.float64).reshape(
        -1, fit_features.shape[-1]
    )
    y = np.asarray(fit_targets, dtype=np.float64).reshape(-1)
    prediction_x = np.asarray(predict_features, dtype=np.float64).reshape(
        -1, predict_features.shape[-1]
    )
    if x.shape[0] != y.size or x.shape[0] == 0:
        raise ValueError("ridge path features/targets are incompatible")
    rms = np.maximum(np.sqrt(np.mean(x * x, axis=0)), RMS_FLOOR)
    normalized = x / rms
    prediction_normalized = prediction_x / rms
    ridge_values = np.asarray(ridges, dtype=np.float64)
    if np.any(~np.isfinite(ridge_values)) or np.any(ridge_values <= 0.0):
        raise ValueError("ridge values must be finite and positive")
    # OnlineRidge solves (X'X/n + lambda I)w=X'y/n.  sklearn solves the
    # sum-Gram system, hence alpha=n*lambda.  Repeating y as seven output
    # columns lets Ridge handle the entire path in one API call.
    targets = np.broadcast_to(y[:, None], (y.size, ridge_values.size)).copy()
    estimator = Ridge(
        alpha=x.shape[0] * ridge_values,
        fit_intercept=False,
        solver="lsqr",
        tol=1e-8,
        max_iter=5_000,
    ).fit(normalized, targets)
    predictions = estimator.predict(prediction_normalized)
    if predictions.shape != (prediction_x.shape[0], ridge_values.size):
        raise RuntimeError("unexpected ridge path prediction shape")
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("ridge path produced non-finite predictions")
    return predictions.reshape(
        predict_features.shape[0], predict_features.shape[1], ridge_values.size
    ), rms, np.asarray(estimator.coef_, dtype=np.float64)


def _calibrate_path(
    raw_predictions: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw_predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if raw.shape[:2] != target.shape:
        raise ValueError("calibration predictions/targets do not match")
    flat_raw = raw.reshape(-1, raw.shape[-1])
    flat_target = target.reshape(-1)
    denominator = np.sum(flat_raw * flat_raw, axis=0)
    numerator = flat_raw.T @ flat_target
    scales = np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator),
        where=denominator > 1e-12,
    )
    scales = np.where(np.isfinite(scales), scales, 1.0)
    predictions = np.clip(
        raw * scales.reshape(1, 1, -1),
        -PREDICTION_CLIP,
        PREDICTION_CLIP,
    ).astype(np.float32)
    return predictions, scales


def _rank_key(result: Mapping[str, Any]) -> tuple[float, float, float, float]:
    cv = result["cv_metrics"]
    validation = result["validation_metrics"]

    def finite(value: Any) -> float:
        number = float(value)
        return number if np.isfinite(number) else -float("inf")

    return (
        finite(cv[SELECTION_METRIC]),
        finite(validation[SELECTION_METRIC]),
        finite(validation["r2_zero"]),
        -float(result["config"]["candidate_order"]),
    )


def _write_csv(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    metric_names = sorted(
        {
            name
            for result in results
            for name in set(result["cv_metrics"]) | set(result["validation_metrics"])
        }
    )
    fields = [
        "rank",
        "id",
        "sketch_size",
        "sketch_seed",
        "preprocessing",
        "ridge",
        "validation_calibration_scale",
        "mean_cv_calibration_scale",
        "config_json",
    ] + [f"cv_{name}" for name in metric_names] + [
        f"validation_{name}" for name in metric_names
    ]
    ranked = sorted(results, key=_rank_key, reverse=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, result in enumerate(ranked, 1):
            config = result["config"]
            row = {
                "rank": rank,
                "id": config["id"],
                "sketch_size": config["sketch_size"],
                "sketch_seed": config["sketch_seed"],
                "preprocessing": config["preprocessing"],
                "ridge": config["ridge"],
                "validation_calibration_scale": result[
                    "validation_calibration_scale"
                ],
                "mean_cv_calibration_scale": float(
                    np.mean(result["cv_calibration_scales"])
                ),
                "config_json": json.dumps(config, sort_keys=True),
            }
            for name in metric_names:
                row[f"cv_{name}"] = result["cv_metrics"].get(name)
                row[f"validation_{name}"] = result["validation_metrics"].get(name)
            writer.writerow(row)
    temporary.replace(path)


def run_sweep(
    *,
    dataset_directory: str | Path,
    search_directory: str | Path | None = None,
    output_directory: str | Path | None = None,
    chunk_samples: int = 16,
    cv_seed: int = CV_SEED,
    overwrite: bool = False,
) -> dict[str, Any]:
    (
        info,
        search,
        search_manifest,
        splits,
        _,
        _,
        train_y,
        validation_y,
    ) = _load_frozen_data(dataset_directory, search_directory)
    if info.layers != 24 or info.pairs != 32:
        raise ValueError("Projection sweep requires Qwen3.5-2B's 24 layers and 32 pairs")
    output = (
        Path(output_directory).expanduser().resolve()
        if output_directory is not None
        else search / "projection_sweep_v1"
    )
    result_path = output / "projection_sweep_results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"{result_path} exists; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions").mkdir(exist_ok=True)

    feature_specs = prepare_projection_caches(
        raw_inputs_path=info.predictor_inputs_path,
        output=output,
        samples=info.samples,
        pairs=info.pairs,
        layers=info.layers,
        hidden=info.hidden,
        chunk_samples=chunk_samples,
        # ``--overwrite`` controls result artifacts.  Valid deterministic
        # feature caches are deliberately reused across repeated sweeps.
        overwrite=False,
    )
    reference_cache = np.load(
        search / "features" / "pr4_late_final_cs128.npy", mmap_mode="r"
    )
    reconstructed_reference = np.load(
        _feature_path(output, 128, 0), mmap_mode="r"
    )
    if reference_cache.shape != reconstructed_reference.shape:
        raise ValueError("faithful PR4 reference cache shape mismatch")
    reference_max_abs_error = 0.0
    for start in range(0, info.samples, chunk_samples):
        stop = min(info.samples, start + chunk_samples)
        error = float(
            np.max(
                np.abs(
                    np.asarray(reference_cache[start:stop], dtype=np.float32)
                    - np.asarray(
                        reconstructed_reference[start:stop], dtype=np.float32
                    )
                )
            )
        )
        reference_max_abs_error = max(reference_max_abs_error, error)
    if reference_max_abs_error > 1e-4:
        raise ValueError(
            "reconstructed B128/seed0 projection disagrees with the PR4 cache: "
            f"max_abs_error={reference_max_abs_error}"
        )
    feature_manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_run_config_sha256": info.run_config_sha256,
        "source_normalized_predictor_inputs": info.predictor_inputs_path,
        "layers": list(LAYERS),
        "features": feature_specs,
        "pr4_b128_seed0_max_abs_error": reference_max_abs_error,
    }
    _atomic_json(output / "projection_feature_manifest.json", feature_manifest)

    train_ids = np.asarray(splits["train"], dtype=np.int64)
    validation_ids = np.asarray(splits["validation"], dtype=np.int64)
    folds = make_six_prompt_folds(train_ids, seed=cv_seed)
    id_to_offset = {int(sample_id): offset for offset, sample_id in enumerate(train_ids)}
    fold_offsets = tuple(
        np.asarray([id_to_offset[int(sample_id)] for sample_id in fold], dtype=np.int64)
        for fold in folds
    )
    fold_index_by_offset = np.full(train_ids.size, -1, dtype=np.int8)
    for fold_index, offsets in enumerate(fold_offsets):
        fold_index_by_offset[offsets] = fold_index
    if np.any(fold_index_by_offset < 0):
        raise AssertionError("CV folds are incomplete")

    results: list[dict[str, Any]] = []
    prediction_store: dict[str, dict[str, np.ndarray]] = {}
    candidate_order = 0
    projections = [
        (sketch_size, seed)
        for sketch_size in SKETCH_SIZES
        for seed in SKETCH_SEEDS
    ]
    progress = tqdm(projections, desc="projection sweep", unit="projection")
    for sketch_size, seed in progress:
        progress.set_postfix(size=sketch_size, seed=seed)
        cache = np.load(_feature_path(output, sketch_size, seed), mmap_mode="r")
        train_raw = np.asarray(cache[train_ids], dtype=np.float32)
        validation_raw = np.asarray(cache[validation_ids], dtype=np.float32)
        for preprocessing in PREPROCESSINGS:
            train_x = _transform(train_raw, preprocessing)
            validation_x = _transform(validation_raw, preprocessing)
            oof = np.empty(
                (len(RIDGES), train_ids.size, info.pairs), dtype=np.float32
            )
            fold_scales = np.empty((len(folds), len(RIDGES)), dtype=np.float64)
            fold_metrics: list[list[dict[str, float]]] = []
            for fold_index, holdout in enumerate(fold_offsets):
                fit_mask = np.ones(train_ids.size, dtype=bool)
                fit_mask[holdout] = False
                raw_path, _, _ = _fit_ridge_path(
                    train_x[fit_mask],
                    train_y[fit_mask],
                    train_x[holdout],
                )
                calibrated, scales = _calibrate_path(raw_path, train_y[holdout])
                oof[:, holdout, :] = np.moveaxis(calibrated, -1, 0)
                fold_scales[fold_index] = scales
                fold_metrics.append(
                    [
                        pair_level_metrics(train_y[holdout], calibrated[..., index])
                        for index in range(len(RIDGES))
                    ]
                )

            validation_raw_path, final_rms, final_weights = _fit_ridge_path(
                train_x, train_y, validation_x
            )
            validation_predictions, validation_scales = _calibrate_path(
                validation_raw_path, validation_y
            )
            for ridge_index, ridge in enumerate(RIDGES):
                identifier = (
                    f"cs{sketch_size}_seed{seed}_{preprocessing}_ridge{ridge:g}"
                )
                config = {
                    "id": identifier,
                    "candidate_order": candidate_order,
                    "layers": list(LAYERS),
                    "sketch_size": sketch_size,
                    "sketch_seed": seed,
                    "preprocessing": preprocessing,
                    "estimator": "mean_gram_ridge_via_sklearn_lsqr",
                    "ridge": ridge,
                    "sklearn_alpha_rule": "n_samples * ridge",
                    "rms_floor": RMS_FLOOR,
                    "calibration": "heldout_zero_intercept",
                    "prediction_clip": PREDICTION_CLIP,
                }
                candidate_order += 1
                oof_prediction = oof[ridge_index]
                validation_prediction = validation_predictions[..., ridge_index]
                result = {
                    "config": config,
                    "cv_metrics": pair_level_metrics(train_y, oof_prediction),
                    "validation_metrics": pair_level_metrics(
                        validation_y, validation_prediction
                    ),
                    "cv_calibration_scales": fold_scales[:, ridge_index],
                    "validation_calibration_scale": validation_scales[ridge_index],
                    "fold_metrics": [
                        metrics[ridge_index] for metrics in fold_metrics
                    ],
                }
                results.append(result)
                prediction_store[identifier] = {
                    "cv_oof_predictions": oof_prediction.copy(),
                    "validation_raw_predictions": validation_raw_path[
                        ..., ridge_index
                    ].astype(np.float32),
                    "validation_predictions": validation_prediction.copy(),
                    "cv_calibration_scales": fold_scales[:, ridge_index].copy(),
                    "validation_calibration_scale": np.asarray(
                        validation_scales[ridge_index]
                    ),
                    "feature_rms": final_rms,
                    "weights": final_weights[ridge_index],
                }

    ranked = sorted(results, key=_rank_key, reverse=True)
    for rank, result in enumerate(ranked, 1):
        result["rank"] = rank
    winners: list[dict[str, Any]] = [ranked[0]]
    for preprocessing in PREPROCESSINGS:
        winners.append(
            max(
                (
                    result
                    for result in results
                    if result["config"]["preprocessing"] == preprocessing
                ),
                key=_rank_key,
            )
        )
    for sketch_size in SKETCH_SIZES:
        winners.append(
            max(
                (
                    result
                    for result in results
                    if result["config"]["sketch_size"] == sketch_size
                ),
                key=_rank_key,
            )
        )
    unique_winners = {
        str(result["config"]["id"]): result for result in winners
    }
    for identifier, winner in unique_winners.items():
        arrays = prediction_store[identifier]
        _atomic_npz(
            output / "predictions" / f"{identifier}.npz",
            config_json=np.asarray(
                json.dumps(winner["config"], sort_keys=True)
            ),
            train_sample_ids=train_ids.astype(np.int32),
            validation_sample_ids=validation_ids.astype(np.int32),
            cv_fold_index=fold_index_by_offset,
            **arrays,
        )
        winner["prediction_artifact"] = output / "predictions" / f"{identifier}.npz"

    split_payload = json.loads((search / "split.json").read_text(encoding="utf-8"))
    identity = {
        "schema_version": SCHEMA_VERSION,
        "stage": "validation_only_projection_sweep",
        "dataset_run_config_sha256": info.run_config_sha256,
        "search_feature_manifest_sha256": _canonical_sha256(search_manifest),
        "projection_feature_manifest_sha256": _canonical_sha256(feature_manifest),
        "split_sha256": _canonical_sha256(split_payload),
        "cv_algorithm": CV_ALGORITHM,
        "cv_seed": cv_seed,
        "folds": [fold.tolist() for fold in folds],
        "train_sample_ids": train_ids,
        "validation_sample_ids": validation_ids,
        "test_labels_read": False,
        "cv_calibration_uses_heldout_fold_labels": True,
        "cv_metrics_strictly_out_of_fold": False,
        "validation_calibration_uses_validation_labels": True,
        "validation_metrics_strictly_heldout": False,
        "selection_order": [
            "cv_macro_prompt_cosine",
            "validation_macro_prompt_cosine",
            "validation_r2_zero",
            "candidate_order",
        ],
        "software": {
            "numpy": _package_version("numpy"),
            "scikit_learn": _package_version("scikit-learn"),
            "source_sha256": _sha256_file(Path(__file__)),
        },
    }
    payload = identity | {
        "completed_at_unix": time.time(),
        "candidate_count": len(results),
        "winner_id": ranked[0]["config"]["id"],
        "winner_config": ranked[0]["config"],
        "winner_cv_metrics": ranked[0]["cv_metrics"],
        "winner_validation_metrics": ranked[0]["validation_metrics"],
        "winner_prediction_artifact": ranked[0].get("prediction_artifact"),
        "saved_winners": list(unique_winners),
        "results": results,
    }
    _atomic_json(result_path, payload)
    _write_csv(output / "projection_sweep_results.csv", results)
    _atomic_json(
        output / "selected_config.json",
        identity
        | {
            "winner_id": ranked[0]["config"]["id"],
            "winner_config": ranked[0]["config"],
            "winner_cv_metrics": ranked[0]["cv_metrics"],
            "winner_validation_metrics": ranked[0]["validation_metrics"],
            "validation_calibration_scale": ranked[0][
                "validation_calibration_scale"
            ],
            "prediction_artifact": ranked[0].get("prediction_artifact"),
        },
    )
    return payload


def self_test() -> None:
    folds = make_six_prompt_folds(np.arange(192, dtype=np.int32), seed=CV_SEED)
    assert len(folds) == 6 and all(fold.size == 32 for fold in folds)
    rng = np.random.default_rng(9)
    values = rng.normal(size=(4, 3, 2, 7)).astype(np.float32)
    buckets, signs = make_countsketch(
        num_layers=2, hidden_size=7, sketch_size=5, seed=2
    )
    actual = _countsketch_two_layers(values, buckets, signs, 5)
    expected = np.zeros((4, 3, 2, 5), dtype=np.float32)
    for row in range(4):
        for pair in range(3):
            for slot in range(2):
                np.add.at(
                    expected[row, pair, slot],
                    buckets[slot],
                    values[row, pair, slot] * signs[slot],
                )
    np.testing.assert_allclose(actual, expected.reshape(4, 3, 10), atol=1e-6)

    features = rng.normal(size=(10, 4, 6)).astype(np.float32)
    centered = _transform(features, "prompt_center")
    np.testing.assert_allclose(centered.mean(axis=1), 0.0, atol=2e-7)
    targets = np.einsum("npf,f->np", features, np.arange(1, 7), optimize=True)
    raw_path, _, _ = _fit_ridge_path(
        features[:8], targets[:8], features[8:], ridges=(0.1, 1.0)
    )
    for index, ridge in enumerate((0.1, 1.0)):
        # Direct mean-Gram reference.
        x = features[:8].reshape(-1, 6).astype(np.float64)
        y = targets[:8].reshape(-1).astype(np.float64)
        rms = np.maximum(np.sqrt(np.mean(x * x, axis=0)), RMS_FLOOR)
        z = x / rms
        weights = np.linalg.solve(
            z.T @ z / z.shape[0] + ridge * np.eye(6),
            z.T @ y / z.shape[0],
        )
        expected_prediction = (
            features[8:].reshape(-1, 6) / rms @ weights
        ).reshape(2, 4)
        np.testing.assert_allclose(
            raw_path[..., index], expected_prediction, rtol=2e-5, atol=2e-5
        )
    calibrated, scales = _calibrate_path(raw_path, targets[8:])
    assert calibrated.shape == raw_path.shape and scales.shape == (2,)
    print("offline_oracle_projection_sweep self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-directory",
        default="outputs/countdown_oracle_q35_2b_D256_P32_seed0",
    )
    parser.add_argument(
        "--search-directory",
        default=None,
        help="Default: DATASET_DIRECTORY/offline_oracle_v1",
    )
    parser.add_argument(
        "--output-directory",
        default=None,
        help="Default: SEARCH_DIRECTORY/projection_sweep_v1",
    )
    parser.add_argument("--chunk-samples", type=int, default=16)
    parser.add_argument("--cv-seed", type=int, default=CV_SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    run_sweep(
        dataset_directory=args.dataset_directory,
        search_directory=args.search_directory,
        output_directory=args.output_directory,
        chunk_samples=args.chunk_samples,
        cv_seed=args.cv_seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
