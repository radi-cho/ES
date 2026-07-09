"""Validation-only classical oracle screen on the frozen Countdown dataset.

This runner consumes the split and feature cache created by
``offline_oracle_search.py prepare``.  Every model is evaluated with six
deterministic prompt-group folds inside the 192 training prompts and once on
the 32 validation prompts.  The test IDs are never applied to a label array.

The faithful PR4 candidate uses the repository's mean-Gram ridge convention,
coordinate-RMS normalization, ridge 10, identity calibration, and clipping at
1.1.  Every non-faithful candidate receives a zero-intercept scalar
calibration fitted independently on each held-out fold and on validation.
Consequently, the stored CV and validation MSE/R2 values reproduce the
historical calibration diagnostics; they are not strictly out-of-fold metrics.
The frozen test evaluator fits no scale with test labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from tqdm.auto import tqdm

from llm_experiments.offline_oracle_search import (
    DatasetInfo,
    load_and_verify_dataset,
    load_group_split,
)
from llm_experiments.offline_oracle_utils import pair_level_metrics


SCHEMA_VERSION = 1
CV_ALGORITHM = "sha256_prompt_groups_6x32_v1"
CV_SEED = 20260709
FEATURE_SET = "pr4_late_final_cs128"
RMS_FLOOR = 1e-4
PREDICTION_CLIP = 1.1
SELECTION_METRIC = "macro_prompt_cosine"
TREE_PREPROCESSING = "signed_power_0.5_then_prompt_center"


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
    encoded = json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _atomic_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    joblib.dump(value, temporary, compress=3)
    temporary.replace(path)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _hash_order_value(domain: str, seed: int, sample_id: int) -> int:
    value = f"{domain}|{int(seed)}|{int(sample_id)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest(), "big")


def make_six_prompt_folds(
    train_sample_ids: np.ndarray,
    *,
    seed: int = CV_SEED,
) -> tuple[np.ndarray, ...]:
    """Return six exact held-out folds of 32 prompt groups via SHA ordering."""

    raw = np.asarray(train_sample_ids)
    if raw.ndim != 1 or raw.size != 192 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("train_sample_ids must contain exactly 192 integer groups")
    sample_ids = raw.astype(np.int64, copy=False)
    if np.unique(sample_ids).size != sample_ids.size:
        raise ValueError("train_sample_ids must be unique")
    ordered = sorted(
        sample_ids.tolist(),
        key=lambda sample_id: _hash_order_value(
            "classical_cv", seed, int(sample_id)
        ),
    )
    folds = tuple(
        np.sort(np.asarray(ordered[start : start + 32], dtype=np.int64))
        for start in range(0, 192, 32)
    )
    joined = np.concatenate(folds)
    if len(folds) != 6 or any(fold.size != 32 for fold in folds):
        raise AssertionError("internal fold construction error")
    if not np.array_equal(np.sort(joined), np.sort(sample_ids)):
        raise AssertionError("folds are not exhaustive and disjoint")
    return folds


def _candidate_configs() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [
        {
            "id": "faithful_pr4_ridge10",
            "family": "faithful_pr4_ridge",
            "estimator": "mean_gram_ridge",
            "preprocessing": "identity",
            "ridge": 10.0,
            "calibration": "identity",
        },
        {
            "id": "tuned_ridge1e-5",
            "family": "tuned_ridge",
            "estimator": "mean_gram_ridge",
            "preprocessing": "identity",
            "ridge": 1e-5,
            "calibration": "heldout_zero_intercept",
        },
        {
            "id": "prompt_centered_ridge1e-5",
            "family": "prompt_centered_ridge",
            "estimator": "mean_gram_ridge",
            "preprocessing": "prompt_center",
            "ridge": 1e-5,
            "calibration": "heldout_zero_intercept",
        },
        {
            "id": "centered_signed_power0p5_ridge1e-5",
            "family": "centered_signed_power_ridge",
            "estimator": "mean_gram_ridge",
            "preprocessing": TREE_PREPROCESSING,
            "signed_power": 0.5,
            "ridge": 1e-5,
            "calibration": "heldout_zero_intercept",
        },
    ]
    for leaves in (15, 31):
        for l2_regularization in (1.0, 10.0):
            for iterations in (100, 200, 400):
                candidates.append(
                    {
                        "id": (
                            f"hgb_leaves{leaves}_lr0p1_l2{int(l2_regularization)}_"
                            f"iter{iterations}"
                        ),
                        "family": "hist_gradient_boosting",
                        "estimator": "hist_gradient_boosting",
                        "preprocessing": TREE_PREPROCESSING,
                        "max_leaf_nodes": leaves,
                        "learning_rate": 0.1,
                        "l2_regularization": l2_regularization,
                        "max_iter": iterations,
                        "min_samples_leaf": 20,
                        "calibration": "heldout_zero_intercept",
                    }
                )
    # The initial validation screen found its strongest HGB result on the
    # untouched PR4 sketch.  Keep these raw/centered controls separate from
    # the signed-power family above so grouped CV can distinguish a real gain
    # from a preprocessing-specific validation fluctuation.
    for preprocessing, prefix in (
        ("identity", "raw"),
        ("prompt_center", "centered"),
    ):
        for leaves in (15, 31):
            for l2_regularization in (1.0, 10.0):
                candidates.append(
                    {
                        "id": (
                            f"hgb_{prefix}_leaves{leaves}_lr0p1_"
                            f"l2{int(l2_regularization)}_iter200"
                        ),
                        "family": f"hist_gradient_boosting_{prefix}",
                        "estimator": "hist_gradient_boosting",
                        "preprocessing": preprocessing,
                        "max_leaf_nodes": leaves,
                        "learning_rate": 0.1,
                        "l2_regularization": l2_regularization,
                        "max_iter": 200,
                        "min_samples_leaf": 20,
                        "calibration": "heldout_zero_intercept",
                    }
                )
    candidates.append(
        {
            "id": "extratrees_leaf20_maxfeat0p5",
            "family": "extra_trees",
            "estimator": "extra_trees",
            "preprocessing": TREE_PREPROCESSING,
            "n_estimators": 400,
            "min_samples_leaf": 20,
            "max_features": 0.5,
            "calibration": "heldout_zero_intercept",
        }
    )
    candidates.append(
        {
            "id": "extratrees_raw_leaf20_maxfeat0p5",
            "family": "extra_trees_raw",
            "estimator": "extra_trees",
            "preprocessing": "identity",
            "n_estimators": 400,
            "min_samples_leaf": 20,
            "max_features": 0.5,
            "calibration": "heldout_zero_intercept",
        }
    )
    for order, candidate in enumerate(candidates):
        candidate.update(
            {
                "candidate_order": order,
                "feature_set": FEATURE_SET,
                "rms_floor": RMS_FLOOR,
                "prediction_clip": PREDICTION_CLIP,
            }
        )
    return candidates


def _transform_features(features: np.ndarray, preprocessing: str) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("features must have shape [prompts, pairs, features]")
    if not np.all(np.isfinite(values)):
        raise ValueError("features contain non-finite values")
    if preprocessing == "identity":
        return values.copy()
    if preprocessing == "prompt_center":
        output = values.copy()
        output -= np.mean(output, axis=1, keepdims=True, dtype=np.float32)
        return output
    if preprocessing == TREE_PREPROCESSING:
        output = np.sign(values) * np.sqrt(np.abs(values))
        output -= np.mean(output, axis=1, keepdims=True, dtype=np.float32)
        return output.astype(np.float32, copy=False)
    raise ValueError(f"Unknown preprocessing {preprocessing!r}")


def _fit_mean_gram_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    ridge: float,
) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float64).reshape(-1, features.shape[-1])
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    if x.shape[0] != y.size or x.shape[0] == 0:
        raise ValueError("ridge features and targets are incompatible")
    rms = np.maximum(np.sqrt(np.mean(np.square(x), axis=0)), RMS_FLOOR)
    normalized = x / rms
    count = float(x.shape[0])
    gram = (normalized.T @ normalized) / count
    cross = (normalized.T @ y) / count
    weights = np.linalg.solve(
        gram + float(ridge) * np.eye(x.shape[1], dtype=np.float64),
        cross,
    )
    return {
        "kind": "mean_gram_ridge",
        "ridge": float(ridge),
        "feature_rms": rms,
        "weights": weights,
    }


def _estimator_seed(base_seed: int, candidate_id: str, fold: int) -> int:
    payload = f"estimator|{int(base_seed)}|{candidate_id}|{int(fold)}".encode(
        "utf-8"
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _fit_estimator(
    config: Mapping[str, Any],
    features: np.ndarray,
    targets: np.ndarray,
    *,
    random_seed: int,
    n_jobs: int,
) -> Any:
    x = np.asarray(features, dtype=np.float32).reshape(-1, features.shape[-1])
    y = np.asarray(targets, dtype=np.float32).reshape(-1)
    if x.shape[0] != y.size or not np.all(np.isfinite(y)):
        raise ValueError("estimator features and targets are incompatible")
    kind = str(config["estimator"])
    if kind == "mean_gram_ridge":
        return _fit_mean_gram_ridge(x, y, float(config["ridge"]))
    if kind == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=float(config["learning_rate"]),
            max_iter=int(config["max_iter"]),
            max_leaf_nodes=int(config["max_leaf_nodes"]),
            min_samples_leaf=int(config["min_samples_leaf"]),
            l2_regularization=float(config["l2_regularization"]),
            early_stopping=False,
            random_state=int(random_seed),
        )
    elif kind == "extra_trees":
        estimator = ExtraTreesRegressor(
            n_estimators=int(config["n_estimators"]),
            min_samples_leaf=int(config["min_samples_leaf"]),
            max_features=float(config["max_features"]),
            bootstrap=False,
            n_jobs=int(n_jobs),
            random_state=int(random_seed),
        )
    else:
        raise ValueError(f"Unknown estimator {kind!r}")
    return estimator.fit(x, y)


def _predict_estimator(estimator: Any, features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32).reshape(-1, features.shape[-1])
    if isinstance(estimator, dict) and estimator.get("kind") == "mean_gram_ridge":
        prediction = (
            x.astype(np.float64) / np.asarray(estimator["feature_rms"])
        ) @ np.asarray(estimator["weights"])
    else:
        prediction = estimator.predict(x)
    prediction = np.asarray(prediction, dtype=np.float64)
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("estimator produced non-finite predictions")
    return prediction.reshape(features.shape[:2])


def _zero_intercept_calibration(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    prediction = np.asarray(predictions, dtype=np.float64).reshape(-1)
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    if prediction.shape != target.shape:
        raise ValueError("calibration predictions and targets do not match")
    denominator = float(prediction @ prediction)
    if denominator <= 1e-12:
        return 1.0
    scale = float((prediction @ target) / denominator)
    return scale if np.isfinite(scale) else 1.0


def _calibrate_and_clip(
    raw_predictions: np.ndarray,
    targets: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, float]:
    scale = (
        1.0
        if config["calibration"] == "identity"
        else _zero_intercept_calibration(raw_predictions, targets)
    )
    prediction = np.asarray(raw_predictions, dtype=np.float64) * scale
    clip = config.get("prediction_clip")
    if clip is not None:
        prediction = np.clip(prediction, -float(clip), float(clip))
    return prediction.astype(np.float32), float(scale)


def _load_frozen_data(
    dataset_directory: str | Path,
    search_directory: str | Path | None,
) -> tuple[
    DatasetInfo,
    Path,
    dict[str, Any],
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    info = load_and_verify_dataset(dataset_directory, deep=False)
    search = (
        Path(search_directory).expanduser().resolve()
        if search_directory is not None
        else info.directory / "offline_oracle_v1"
    )
    manifest_path = search / "feature_manifest.json"
    split_path = search / "split.json"
    if not manifest_path.is_file() or not split_path.is_file():
        raise FileNotFoundError(
            "Run `python -m llm_experiments.offline_oracle_search prepare` first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_run_config_sha256") != info.run_config_sha256:
        raise ValueError("feature manifest belongs to a different dataset")
    if manifest.get("split_sha256") != _canonical_sha256(split_payload):
        raise ValueError("feature manifest and split fingerprint disagree")
    splits = load_group_split(split_path, info)
    if (
        tuple(splits["train"].shape) != (192,)
        or tuple(splits["validation"].shape) != (32,)
        or tuple(splits["test"].shape) != (32,)
        or not np.array_equal(
            np.sort(np.concatenate(tuple(splits.values()))),
            np.arange(info.samples),
        )
    ):
        raise ValueError("classical screen requires the frozen 192/32/32 split")

    spec = manifest.get("features", {}).get(FEATURE_SET)
    expected_layers = [min(info.layers - 1, int(np.floor(0.75 * info.layers))), info.layers - 1]
    if not isinstance(spec, Mapping):
        raise ValueError(f"feature manifest does not contain {FEATURE_SET}")
    if (
        spec.get("layers") != expected_layers
        or int(spec.get("sketch_size", -1)) != 128
        or int(spec.get("seed", -1)) != 0
        or tuple(spec.get("shape", ())) != (info.samples, info.pairs, 256)
    ):
        raise ValueError("cached PR4 features do not match the faithful contract")
    feature_path = search / "features" / f"{FEATURE_SET}.npy"
    feature_cache = np.load(feature_path, mmap_mode="r")
    if feature_cache.shape != (info.samples, info.pairs, 256) or feature_cache.dtype != np.float32:
        raise ValueError("PR4 feature cache has incompatible shape or dtype")

    train_ids = np.asarray(splits["train"], dtype=np.int64)
    validation_ids = np.asarray(splits["validation"], dtype=np.int64)
    train_x = np.asarray(feature_cache[train_ids], dtype=np.float32)
    validation_x = np.asarray(feature_cache[validation_ids], dtype=np.float32)
    if not np.all(np.isfinite(train_x)) or not np.all(np.isfinite(validation_x)):
        raise ValueError("selected feature cache contains non-finite values")

    # LABEL SAFETY BOUNDARY: no other target indexing is permitted in this file.
    # In particular, splits["test"] is never applied to this memmap.
    all_targets = np.load(info.reward_differences_path, mmap_mode="r")
    train_y = np.asarray(all_targets[train_ids], dtype=np.float32)
    validation_y = np.asarray(all_targets[validation_ids], dtype=np.float32)
    if train_y.shape != (192, info.pairs) or validation_y.shape != (32, info.pairs):
        raise ValueError("train/validation target shape disagrees with the split")
    if not np.all(np.isfinite(train_y)) or not np.all(np.isfinite(validation_y)):
        raise ValueError("train/validation targets contain non-finite values")
    return (
        info,
        search,
        manifest,
        splits,
        train_x,
        validation_x,
        train_y,
        validation_y,
    )


def _rank_key(result: Mapping[str, Any]) -> tuple[float, float, float, float]:
    validation = result["validation_metrics"]
    cross_validation = result["cv_metrics"]

    def finite_or_negative_infinity(value: Any) -> float:
        numeric = float(value)
        return numeric if np.isfinite(numeric) else -float("inf")

    return (
        finite_or_negative_infinity(validation[SELECTION_METRIC]),
        finite_or_negative_infinity(cross_validation[SELECTION_METRIC]),
        finite_or_negative_infinity(validation["r2_zero"]),
        -float(result["config"]["candidate_order"]),
    )


def _write_results_csv(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    metric_names = sorted(
        {
            metric
            for result in results
            for metric in (
                set(result["validation_metrics"]) | set(result["cv_metrics"])
            )
        }
    )
    fields = [
        "rank",
        "id",
        "family",
        "preprocessing",
        "estimator",
        "validation_calibration_scale",
        "mean_cv_calibration_scale",
        "config_json",
    ] + [f"validation_{name}" for name in metric_names] + [
        f"cv_{name}" for name in metric_names
    ]
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, result in enumerate(sorted(results, key=_rank_key, reverse=True), 1):
            config = result["config"]
            row = {
                "rank": rank,
                "id": config["id"],
                "family": config["family"],
                "preprocessing": config["preprocessing"],
                "estimator": config["estimator"],
                "validation_calibration_scale": result[
                    "validation_calibration_scale"
                ],
                "mean_cv_calibration_scale": float(
                    np.mean(result["cv_calibration_scales"])
                ),
                "config_json": json.dumps(config, sort_keys=True),
            }
            for name in metric_names:
                row[f"validation_{name}"] = result["validation_metrics"].get(name)
                row[f"cv_{name}"] = result["cv_metrics"].get(name)
            writer.writerow(row)
    temporary.replace(path)


def run_screen(
    *,
    dataset_directory: str | Path,
    search_directory: str | Path | None = None,
    output_directory: str | Path | None = None,
    cv_seed: int = CV_SEED,
    estimator_seed: int = 0,
    n_jobs: int = -1,
    overwrite: bool = False,
) -> dict[str, Any]:
    (
        info,
        search,
        manifest,
        splits,
        train_x_raw,
        validation_x_raw,
        train_y,
        validation_y,
    ) = _load_frozen_data(dataset_directory, search_directory)
    output = (
        Path(output_directory).expanduser().resolve()
        if output_directory is not None
        else search / "classical_v1"
    )
    result_path = output / "classical_results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"{result_path} already exists; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    (output / "predictions").mkdir(exist_ok=True)

    candidates = _candidate_configs()
    folds = make_six_prompt_folds(splits["train"], seed=cv_seed)
    id_to_train_offset = {
        int(sample_id): offset for offset, sample_id in enumerate(splits["train"])
    }
    fold_offsets = tuple(
        np.asarray([id_to_train_offset[int(sample_id)] for sample_id in fold])
        for fold in folds
    )
    fold_index_by_offset = np.full(192, -1, dtype=np.int8)
    for fold_index, offsets in enumerate(fold_offsets):
        fold_index_by_offset[offsets] = fold_index
    if np.any(fold_index_by_offset < 0):
        raise AssertionError("CV fold assignment is incomplete")

    split_payload = json.loads((search / "split.json").read_text(encoding="utf-8"))
    identity = {
        "schema_version": SCHEMA_VERSION,
        "stage": "validation_only_classical_screen",
        "dataset_run_config_sha256": info.run_config_sha256,
        "feature_manifest_sha256": _canonical_sha256(manifest),
        "split_sha256": _canonical_sha256(split_payload),
        "feature_set": FEATURE_SET,
        "cv_algorithm": CV_ALGORITHM,
        "cv_seed": int(cv_seed),
        "cv_hash_payload": "classical_cv|<seed>|<sample_id>",
        "estimator_seed": int(estimator_seed),
        "dataset_directory": info.directory,
        "search_directory": search,
        "output_directory": output,
        "extra_trees_n_jobs": int(n_jobs),
        "selection_metric": SELECTION_METRIC,
        "selection_tie_breakers": [
            "cv_macro_prompt_cosine",
            "validation_r2_zero",
            "candidate_order",
        ],
        "train_sample_ids": splits["train"],
        "validation_sample_ids": splits["validation"],
        "test_labels_read": False,
        "cv_calibration_uses_heldout_fold_labels": True,
        "cv_metrics_strictly_out_of_fold": False,
        "validation_calibration_uses_validation_labels": True,
        "validation_metrics_strictly_heldout": False,
        "folds": [fold.tolist() for fold in folds],
        "software": {
            "numpy": _package_version("numpy"),
            "scikit_learn": _package_version("scikit-learn"),
            "joblib": _package_version("joblib"),
            "source_sha256": _sha256_file(Path(__file__)),
        },
    }
    _atomic_json(output / "classical_configs.json", identity | {"candidates": candidates})
    _atomic_json(
        output / "cv_folds.json",
        {
            "algorithm": CV_ALGORITHM,
            "seed": cv_seed,
            "hash_payload": "classical_cv|<seed>|<sample_id>",
            "train_sample_ids": splits["train"],
            "folds": [fold.tolist() for fold in folds],
        },
    )

    preprocessings = sorted({str(config["preprocessing"]) for config in candidates})
    train_views = {
        name: _transform_features(train_x_raw, name) for name in preprocessings
    }
    validation_views = {
        name: _transform_features(validation_x_raw, name) for name in preprocessings
    }

    results: list[dict[str, Any]] = []
    progress = tqdm(candidates, desc="classical oracle screen", unit="candidate")
    for config in progress:
        progress.set_postfix(candidate=config["id"])
        train_x = train_views[str(config["preprocessing"])]
        validation_x = validation_views[str(config["preprocessing"])]
        oof_predictions = np.empty_like(train_y, dtype=np.float32)
        fold_results: list[dict[str, Any]] = []
        fold_scales: list[float] = []

        for fold_index, holdout_offsets in enumerate(fold_offsets):
            fit_mask = np.ones(192, dtype=bool)
            fit_mask[holdout_offsets] = False
            seed = _estimator_seed(estimator_seed, str(config["id"]), fold_index)
            estimator = _fit_estimator(
                config,
                train_x[fit_mask],
                train_y[fit_mask],
                random_seed=seed,
                n_jobs=n_jobs,
            )
            raw_holdout = _predict_estimator(estimator, train_x[holdout_offsets])
            heldout_prediction, calibration_scale = _calibrate_and_clip(
                raw_holdout,
                train_y[holdout_offsets],
                config,
            )
            oof_predictions[holdout_offsets] = heldout_prediction
            fold_scales.append(calibration_scale)
            fold_results.append(
                {
                    "fold": fold_index,
                    "fit_sample_ids": np.asarray(splits["train"])[fit_mask],
                    "heldout_sample_ids": np.asarray(splits["train"])[
                        holdout_offsets
                    ],
                    "estimator_seed": seed,
                    "calibration_scale": calibration_scale,
                    "metrics": pair_level_metrics(
                        train_y[holdout_offsets], heldout_prediction
                    ),
                }
            )

        cv_metrics = pair_level_metrics(train_y, oof_predictions)
        final_seed = _estimator_seed(estimator_seed, str(config["id"]), 6)
        final_estimator = _fit_estimator(
            config,
            train_x,
            train_y,
            random_seed=final_seed,
            n_jobs=n_jobs,
        )
        validation_raw = _predict_estimator(final_estimator, validation_x)
        validation_prediction, validation_scale = _calibrate_and_clip(
            validation_raw,
            validation_y,
            config,
        )
        validation_metrics = pair_level_metrics(
            validation_y, validation_prediction
        )

        model_path = output / "models" / f"{config['id']}.joblib"
        prediction_path = output / "predictions" / f"{config['id']}.npz"
        _atomic_joblib(
            model_path,
            {
                "schema_version": SCHEMA_VERSION,
                "config": dict(config),
                "feature_set": FEATURE_SET,
                "feature_manifest_sha256": identity["feature_manifest_sha256"],
                "split_sha256": identity["split_sha256"],
                "fit_sample_ids": np.asarray(splits["train"], dtype=np.int32),
                "estimator_seed": final_seed,
                "validation_calibration_scale": validation_scale,
                "estimator": final_estimator,
            },
        )
        _atomic_npz(
            prediction_path,
            train_sample_ids=np.asarray(splits["train"], dtype=np.int32),
            validation_sample_ids=np.asarray(splits["validation"], dtype=np.int32),
            cv_fold_index=fold_index_by_offset,
            cv_oof_predictions=oof_predictions,
            validation_raw_predictions=validation_raw.astype(np.float32),
            validation_predictions=validation_prediction,
            cv_calibration_scales=np.asarray(fold_scales, dtype=np.float64),
            validation_calibration_scale=np.asarray(validation_scale),
        )
        results.append(
            {
                "config": dict(config),
                "cv_metrics": cv_metrics,
                "cv_calibration_scales": fold_scales,
                "cv_folds": fold_results,
                "validation_metrics": validation_metrics,
                "validation_calibration_scale": validation_scale,
                "model_artifact": model_path,
                "prediction_artifact": prediction_path,
            }
        )

    ranked = sorted(results, key=_rank_key, reverse=True)
    for rank, result in enumerate(ranked, 1):
        result["validation_rank"] = rank
    payload = identity | {
        "completed_at_unix": time.time(),
        "candidate_count": len(candidates),
        "winner_id": ranked[0]["config"]["id"],
        "winner_config": ranked[0]["config"],
        "results": results,
    }
    _atomic_json(result_path, payload)
    _write_results_csv(output / "classical_results.csv", results)
    _atomic_json(
        output / "selected_config.json",
        identity
        | {
            "winner_id": ranked[0]["config"]["id"],
            "winner_config": ranked[0]["config"],
            "winner_validation_metrics": ranked[0]["validation_metrics"],
            "winner_cv_metrics": ranked[0]["cv_metrics"],
            "model_artifact": ranked[0]["model_artifact"],
            "validation_calibration_scale": ranked[0][
                "validation_calibration_scale"
            ],
        },
    )
    return payload


def self_test() -> None:
    folds = make_six_prompt_folds(np.arange(192, dtype=np.int32))
    assert len(folds) == 6 and all(fold.size == 32 for fold in folds)
    np.testing.assert_array_equal(np.sort(np.concatenate(folds)), np.arange(192))
    fold_fingerprint = hashlib.sha256(
        b"".join(fold.astype("<i8", copy=False).tobytes() for fold in folds)
    ).hexdigest()
    assert (
        fold_fingerprint
        == "cda16a09df4780e5a8723146d21207a480bd4e660647b01c1bfabcf3e59d5ffa"
    )

    rng = np.random.default_rng(17)
    features = rng.normal(size=(8, 32, 6)).astype(np.float32)
    centered = _transform_features(features, "prompt_center")
    signed_centered = _transform_features(features, TREE_PREPROCESSING)
    np.testing.assert_allclose(centered.mean(axis=1), 0.0, atol=2e-7)
    np.testing.assert_allclose(signed_centered.mean(axis=1), 0.0, atol=2e-7)

    targets = np.einsum("npf,f->np", features, np.arange(1, 7), optimize=True)
    ridge = _fit_mean_gram_ridge(features[:6], targets[:6], 1e-8)
    prediction = _predict_estimator(ridge, features[6:])
    np.testing.assert_allclose(prediction, targets[6:], rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(
        _zero_intercept_calibration(np.asarray([1.0, -2.0]), np.asarray([2.0, -4.0])),
        2.0,
    )
    candidates = _candidate_configs()
    assert len(candidates) == 26
    assert sum(config["family"] == "hist_gradient_boosting" for config in candidates) == 12
    print("offline_oracle_classical self-test passed")


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
        help="Default: SEARCH_DIRECTORY/classical_v1",
    )
    parser.add_argument("--cv-seed", type=int, default=CV_SEED)
    parser.add_argument("--estimator-seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    run_screen(
        dataset_directory=args.dataset_directory,
        search_directory=args.search_directory,
        output_directory=args.output_directory,
        cv_seed=args.cv_seed,
        estimator_seed=args.estimator_seed,
        n_jobs=args.n_jobs,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
