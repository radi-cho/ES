"""Validation-only latent linear screen for the offline Countdown oracle.

The script compares supervised PLS and train-only PCA-plus-ridge models on the
frozen 256-dimensional PR4 feature cache.  Raw and within-prompt-centered
feature views are evaluated with the exact six prompt-group folds used by
``offline_oracle_classical``.  Test sample IDs are never applied to labels.

PCA ridge strength is chosen exclusively from six-fold training OOF metrics.
Every held-out fold, and the external validation set, receives its own
zero-intercept scalar calibration before clipping and evaluation.
These calibrated CV and validation MSE/R2 values reproduce the historical
calibration diagnostics and are not strictly held-out estimates.  Test labels
are never used to fit a scale.
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

import joblib
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from llm_experiments.offline_oracle_classical import (
    CV_ALGORITHM,
    CV_SEED,
    FEATURE_SET,
    make_six_prompt_folds,
)
from llm_experiments.offline_oracle_search import (
    DatasetInfo,
    load_and_verify_dataset,
    load_group_split,
)
from llm_experiments.offline_oracle_utils import pair_level_metrics


SCHEMA_VERSION = 1
SELECTION_METRIC = "macro_prompt_cosine"
RMS_FLOOR = 1e-4
PREDICTION_CLIP = 1.1
PLS_COMPONENTS = (1, 2, 4, 8, 16, 32)
PCA_COMPONENTS = (16, 32, 64, 128)
PCA_SOLVERS = ("randomized", "full")
RIDGE_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
PREPROCESSINGS = ("raw", "prompt_centered")


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


def _estimator_seed(base_seed: int, identifier: str, fold: int) -> int:
    payload = f"latent|{int(base_seed)}|{identifier}|{int(fold)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _preprocess(features: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 3 or not np.all(np.isfinite(values)):
        raise ValueError("features must be finite [prompts, pairs, features]")
    if name == "raw":
        return values.copy()
    if name == "prompt_centered":
        centered = values.copy()
        centered -= np.mean(centered, axis=1, keepdims=True, dtype=np.float32)
        return centered
    raise ValueError(f"Unknown preprocessing {name!r}")


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
    predictions: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, float]:
    scale = _zero_intercept_calibration(predictions, targets)
    calibrated = np.asarray(predictions, dtype=np.float64) * scale
    calibrated = np.clip(calibrated, -PREDICTION_CLIP, PREDICTION_CLIP)
    return calibrated.astype(np.float32), scale


def _fit_mean_gram_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    ridge: float,
) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.size or x.shape[0] == 0:
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


def _predict_ridge(model: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    prediction = (x / np.asarray(model["feature_rms"])) @ np.asarray(
        model["weights"]
    )
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("ridge produced non-finite predictions")
    return prediction


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
        raise ValueError("feature manifest and frozen split disagree")
    splits = load_group_split(split_path, info)
    if (
        splits["train"].shape != (192,)
        or splits["validation"].shape != (32,)
        or splits["test"].shape != (32,)
        or not np.array_equal(
            np.sort(np.concatenate(tuple(splits.values()))), np.arange(info.samples)
        )
    ):
        raise ValueError("latent screen requires the frozen 192/32/32 split")

    spec = manifest.get("features", {}).get(FEATURE_SET)
    expected_layers = [
        min(info.layers - 1, int(np.floor(0.75 * info.layers))),
        info.layers - 1,
    ]
    if not isinstance(spec, Mapping) or (
        spec.get("layers") != expected_layers
        or int(spec.get("sketch_size", -1)) != 128
        or int(spec.get("seed", -1)) != 0
        or tuple(spec.get("shape", ())) != (info.samples, info.pairs, 256)
    ):
        raise ValueError("cached PR4 features do not match the frozen contract")
    feature_path = search / "features" / f"{FEATURE_SET}.npy"
    feature_cache = np.load(feature_path, mmap_mode="r")
    if feature_cache.shape != (info.samples, info.pairs, 256):
        raise ValueError("PR4 feature cache has the wrong shape")
    if feature_cache.dtype != np.float32:
        raise ValueError("PR4 feature cache must be float32")

    train_ids = np.asarray(splits["train"], dtype=np.int64)
    validation_ids = np.asarray(splits["validation"], dtype=np.int64)
    train_x = np.asarray(feature_cache[train_ids], dtype=np.float32)
    validation_x = np.asarray(feature_cache[validation_ids], dtype=np.float32)
    if not np.all(np.isfinite(train_x)) or not np.all(np.isfinite(validation_x)):
        raise ValueError("selected feature groups contain non-finite values")

    # LABEL SAFETY BOUNDARY: the test IDs are never applied to this memmap.
    all_targets = np.load(info.reward_differences_path, mmap_mode="r")
    train_y = np.asarray(all_targets[train_ids], dtype=np.float32)
    validation_y = np.asarray(all_targets[validation_ids], dtype=np.float32)
    if train_y.shape != (192, 32) or validation_y.shape != (32, 32):
        raise ValueError("target arrays disagree with the frozen split")
    if not np.all(np.isfinite(train_y)) or not np.all(np.isfinite(validation_y)):
        raise ValueError("train/validation labels contain non-finite values")
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


def _candidate_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for preprocessing in PREPROCESSINGS:
        for components in PLS_COMPONENTS:
            configs.append(
                {
                    "id": f"pls_{preprocessing}_components{components}",
                    "family": "pls_regression",
                    "preprocessing": preprocessing,
                    "components": components,
                    "scale": True,
                    "max_iter": 500,
                    "tol": 1e-6,
                    "prediction_clip": PREDICTION_CLIP,
                }
            )
        for solver in PCA_SOLVERS:
            for components in PCA_COMPONENTS:
                configs.append(
                    {
                        "id": f"pca_{solver}_{preprocessing}_components{components}",
                        "family": "pca_ridge_cv",
                        "preprocessing": preprocessing,
                        "components": components,
                        "svd_solver": solver,
                        "whiten": False,
                        "ridge_grid": list(RIDGE_GRID),
                        "prediction_clip": PREDICTION_CLIP,
                    }
                )
        configs.extend(
            [
                {
                    "id": f"huber_{preprocessing}_epsilon1p35_alpha1e-4",
                    "family": "huber_regression",
                    "preprocessing": preprocessing,
                    "epsilon": 1.35,
                    "alpha": 1e-4,
                    "max_iter": 200,
                    "tol": 1e-5,
                    "standardize": True,
                    "prediction_clip": PREDICTION_CLIP,
                },
                {
                    "id": f"elasticnet_{preprocessing}_alpha1e-4_l1ratio0p1",
                    "family": "elastic_net",
                    "preprocessing": preprocessing,
                    "alpha": 1e-4,
                    "l1_ratio": 0.1,
                    "max_iter": 3000,
                    "tol": 1e-5,
                    "standardize": True,
                    "prediction_clip": PREDICTION_CLIP,
                },
            ]
        )
    for order, config in enumerate(configs):
        config.update(
            {
                "candidate_order": order,
                "feature_set": FEATURE_SET,
                "calibration": "heldout_zero_intercept",
            }
        )
    return configs


def _fold_offsets(
    train_ids: np.ndarray,
    folds: Sequence[np.ndarray],
) -> tuple[np.ndarray, ...]:
    id_to_offset = {
        int(sample_id): offset for offset, sample_id in enumerate(train_ids)
    }
    offsets = tuple(
        np.asarray([id_to_offset[int(sample_id)] for sample_id in fold])
        for fold in folds
    )
    joined = np.concatenate(offsets)
    if np.unique(joined).size != 192 or not np.array_equal(
        np.sort(joined), np.arange(192)
    ):
        raise AssertionError("CV offsets are not exhaustive and disjoint")
    return offsets


def _fit_pls(
    config: Mapping[str, Any],
    train_x: np.ndarray,
    train_y: np.ndarray,
) -> PLSRegression:
    model = PLSRegression(
        n_components=int(config["components"]),
        scale=bool(config["scale"]),
        max_iter=int(config["max_iter"]),
        tol=float(config["tol"]),
        copy=True,
    )
    model.fit(
        train_x.reshape(-1, train_x.shape[-1]),
        train_y.reshape(-1),
    )
    return model


def _predict_pls(model: PLSRegression, features: np.ndarray) -> np.ndarray:
    prediction = model.predict(features.reshape(-1, features.shape[-1]))
    prediction = np.asarray(prediction, dtype=np.float64).reshape(features.shape[:2])
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("PLS produced non-finite predictions")
    return prediction


def _fit_robust_linear(
    config: Mapping[str, Any],
    features: np.ndarray,
    targets: np.ndarray,
    *,
    random_seed: int,
) -> Any:
    family = str(config["family"])
    if family == "huber_regression":
        estimator: Any = HuberRegressor(
            epsilon=float(config["epsilon"]),
            alpha=float(config["alpha"]),
            max_iter=int(config["max_iter"]),
            tol=float(config["tol"]),
            fit_intercept=True,
        )
    elif family == "elastic_net":
        estimator = ElasticNet(
            alpha=float(config["alpha"]),
            l1_ratio=float(config["l1_ratio"]),
            max_iter=int(config["max_iter"]),
            tol=float(config["tol"]),
            fit_intercept=True,
            selection="cyclic",
            random_state=int(random_seed),
        )
    else:
        raise ValueError(f"Unknown robust linear family {family!r}")
    model = make_pipeline(StandardScaler(), estimator)
    model.fit(
        features.reshape(-1, features.shape[-1]),
        targets.reshape(-1),
    )
    return model


def _predict_robust_linear(model: Any, features: np.ndarray) -> np.ndarray:
    prediction = model.predict(features.reshape(-1, features.shape[-1]))
    prediction = np.asarray(prediction, dtype=np.float64).reshape(features.shape[:2])
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("robust linear model produced non-finite predictions")
    return prediction


def _evaluate_pls(
    config: Mapping[str, Any],
    train_x: np.ndarray,
    validation_x: np.ndarray,
    train_y: np.ndarray,
    validation_y: np.ndarray,
    train_ids: np.ndarray,
    fold_offsets: Sequence[np.ndarray],
) -> tuple[dict[str, Any], PLSRegression, np.ndarray, np.ndarray, np.ndarray]:
    oof = np.empty_like(train_y, dtype=np.float32)
    fold_results: list[dict[str, Any]] = []
    fold_scales: list[float] = []
    for fold_index, heldout in enumerate(fold_offsets):
        fit_mask = np.ones(192, dtype=bool)
        fit_mask[heldout] = False
        model = _fit_pls(config, train_x[fit_mask], train_y[fit_mask])
        raw = _predict_pls(model, train_x[heldout])
        prediction, scale = _calibrate_and_clip(raw, train_y[heldout])
        oof[heldout] = prediction
        fold_scales.append(scale)
        fold_results.append(
            {
                "fold": fold_index,
                "fit_sample_ids": train_ids[fit_mask],
                "heldout_sample_ids": train_ids[heldout],
                "calibration_scale": scale,
                "metrics": pair_level_metrics(train_y[heldout], prediction),
            }
        )

    cv_metrics = pair_level_metrics(train_y, oof)
    final_model = _fit_pls(config, train_x, train_y)
    validation_raw = _predict_pls(final_model, validation_x)
    validation_prediction, validation_scale = _calibrate_and_clip(
        validation_raw, validation_y
    )
    result = {
        "config": dict(config),
        "selected_ridge": None,
        "cv_metrics": cv_metrics,
        "cv_calibration_scales": fold_scales,
        "cv_folds": fold_results,
        "validation_metrics": pair_level_metrics(
            validation_y, validation_prediction
        ),
        "validation_calibration_scale": validation_scale,
    }
    return result, final_model, oof, validation_raw, validation_prediction


def _evaluate_robust_linear(
    config: Mapping[str, Any],
    train_x: np.ndarray,
    validation_x: np.ndarray,
    train_y: np.ndarray,
    validation_y: np.ndarray,
    train_ids: np.ndarray,
    fold_offsets: Sequence[np.ndarray],
    *,
    estimator_seed: int,
) -> tuple[dict[str, Any], Any, np.ndarray, np.ndarray, np.ndarray]:
    oof = np.empty_like(train_y, dtype=np.float32)
    fold_results: list[dict[str, Any]] = []
    fold_scales: list[float] = []
    for fold_index, heldout in enumerate(fold_offsets):
        fit_mask = np.ones(192, dtype=bool)
        fit_mask[heldout] = False
        seed = _estimator_seed(estimator_seed, str(config["id"]), fold_index)
        model = _fit_robust_linear(
            config,
            train_x[fit_mask],
            train_y[fit_mask],
            random_seed=seed,
        )
        raw = _predict_robust_linear(model, train_x[heldout])
        prediction, scale = _calibrate_and_clip(raw, train_y[heldout])
        oof[heldout] = prediction
        fold_scales.append(scale)
        fold_results.append(
            {
                "fold": fold_index,
                "fit_sample_ids": train_ids[fit_mask],
                "heldout_sample_ids": train_ids[heldout],
                "estimator_seed": seed,
                "calibration_scale": scale,
                "metrics": pair_level_metrics(train_y[heldout], prediction),
            }
        )

    final_seed = _estimator_seed(estimator_seed, str(config["id"]), 6)
    final_model = _fit_robust_linear(
        config, train_x, train_y, random_seed=final_seed
    )
    validation_raw = _predict_robust_linear(final_model, validation_x)
    validation_prediction, validation_scale = _calibrate_and_clip(
        validation_raw, validation_y
    )
    result = {
        "config": dict(config),
        "selected_ridge": None,
        "cv_metrics": pair_level_metrics(train_y, oof),
        "cv_calibration_scales": fold_scales,
        "cv_folds": fold_results,
        "validation_metrics": pair_level_metrics(
            validation_y, validation_prediction
        ),
        "validation_calibration_scale": validation_scale,
        "final_estimator_seed": final_seed,
    }
    return result, final_model, oof, validation_raw, validation_prediction


def _ridge_rank_key(item: Mapping[str, Any]) -> tuple[float, float, float]:
    metrics = item["cv_metrics"]

    def finite(value: Any) -> float:
        number = float(value)
        return number if np.isfinite(number) else -float("inf")

    return (
        finite(metrics[SELECTION_METRIC]),
        finite(metrics["r2_zero"]),
        -float(item["ridge"]),
    )


def _evaluate_pca(
    config: Mapping[str, Any],
    train_x: np.ndarray,
    validation_x: np.ndarray,
    train_y: np.ndarray,
    validation_y: np.ndarray,
    train_ids: np.ndarray,
    fold_offsets: Sequence[np.ndarray],
    *,
    estimator_seed: int,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    ridge_grid = tuple(float(value) for value in config["ridge_grid"])
    oof_by_ridge = {
        ridge: np.empty_like(train_y, dtype=np.float32) for ridge in ridge_grid
    }
    fold_results_by_ridge: dict[float, list[dict[str, Any]]] = {
        ridge: [] for ridge in ridge_grid
    }
    fold_scales_by_ridge: dict[float, list[float]] = {
        ridge: [] for ridge in ridge_grid
    }

    for fold_index, heldout in enumerate(fold_offsets):
        fit_mask = np.ones(192, dtype=bool)
        fit_mask[heldout] = False
        fit_flat = train_x[fit_mask].reshape(-1, train_x.shape[-1])
        heldout_flat = train_x[heldout].reshape(-1, train_x.shape[-1])
        seed = _estimator_seed(estimator_seed, str(config["id"]), fold_index)
        pca = PCA(
            n_components=int(config["components"]),
            svd_solver=str(config["svd_solver"]),
            whiten=bool(config["whiten"]),
            random_state=seed,
        )
        fit_latent = pca.fit_transform(fit_flat)
        heldout_latent = pca.transform(heldout_flat)
        for ridge in ridge_grid:
            model = _fit_mean_gram_ridge(
                fit_latent, train_y[fit_mask].reshape(-1), ridge
            )
            raw = _predict_ridge(model, heldout_latent).reshape(
                heldout.size, train_y.shape[1]
            )
            prediction, scale = _calibrate_and_clip(raw, train_y[heldout])
            oof_by_ridge[ridge][heldout] = prediction
            fold_scales_by_ridge[ridge].append(scale)
            fold_results_by_ridge[ridge].append(
                {
                    "fold": fold_index,
                    "fit_sample_ids": train_ids[fit_mask],
                    "heldout_sample_ids": train_ids[heldout],
                    "pca_random_state": seed,
                    "calibration_scale": scale,
                    "metrics": pair_level_metrics(train_y[heldout], prediction),
                }
            )

    ridge_results = [
        {
            "ridge": ridge,
            "cv_metrics": pair_level_metrics(train_y, oof_by_ridge[ridge]),
            "cv_calibration_scales": fold_scales_by_ridge[ridge],
            "cv_folds": fold_results_by_ridge[ridge],
        }
        for ridge in ridge_grid
    ]
    selected = max(ridge_results, key=_ridge_rank_key)
    selected_ridge = float(selected["ridge"])

    final_seed = _estimator_seed(estimator_seed, str(config["id"]), 6)
    final_pca = PCA(
        n_components=int(config["components"]),
        svd_solver=str(config["svd_solver"]),
        whiten=bool(config["whiten"]),
        random_state=final_seed,
    )
    train_flat = train_x.reshape(-1, train_x.shape[-1])
    validation_flat = validation_x.reshape(-1, validation_x.shape[-1])
    train_latent = final_pca.fit_transform(train_flat)
    validation_latent = final_pca.transform(validation_flat)
    final_ridge = _fit_mean_gram_ridge(
        train_latent, train_y.reshape(-1), selected_ridge
    )
    validation_raw = _predict_ridge(final_ridge, validation_latent).reshape(
        validation_y.shape
    )
    validation_prediction, validation_scale = _calibrate_and_clip(
        validation_raw, validation_y
    )
    result = {
        "config": dict(config),
        "selected_ridge": selected_ridge,
        "ridge_grid_cv": ridge_results,
        "cv_metrics": selected["cv_metrics"],
        "cv_calibration_scales": selected["cv_calibration_scales"],
        "cv_folds": selected["cv_folds"],
        "validation_metrics": pair_level_metrics(
            validation_y, validation_prediction
        ),
        "validation_calibration_scale": validation_scale,
    }
    artifact = {
        "pca": final_pca,
        "ridge": final_ridge,
        "pca_random_state": final_seed,
    }
    return (
        result,
        artifact,
        oof_by_ridge[selected_ridge],
        validation_raw,
        validation_prediction,
    )


def _result_rank_key(result: Mapping[str, Any]) -> tuple[float, float, float, float]:
    cv = result["cv_metrics"]
    validation = result["validation_metrics"]

    def finite(value: Any) -> float:
        number = float(value)
        return number if np.isfinite(number) else -float("inf")

    return (
        finite(cv[SELECTION_METRIC]),
        finite(validation[SELECTION_METRIC]),
        finite(cv["r2_zero"]),
        -float(result["config"]["candidate_order"]),
    )


def _write_results_csv(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    metric_names = sorted(
        {
            metric
            for result in results
            for metric in set(result["cv_metrics"])
            | set(result["validation_metrics"])
        }
    )
    fields = [
        "rank",
        "id",
        "family",
        "preprocessing",
        "components",
        "svd_solver",
        "selected_ridge",
        "validation_calibration_scale",
        "config_json",
    ] + [f"cv_{name}" for name in metric_names] + [
        f"validation_{name}" for name in metric_names
    ]
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, result in enumerate(
            sorted(results, key=_result_rank_key, reverse=True), 1
        ):
            config = result["config"]
            row = {
                "rank": rank,
                "id": config["id"],
                "family": config["family"],
                "preprocessing": config["preprocessing"],
                "components": config.get("components"),
                "svd_solver": config.get("svd_solver"),
                "selected_ridge": result["selected_ridge"],
                "validation_calibration_scale": result[
                    "validation_calibration_scale"
                ],
                "config_json": json.dumps(config, sort_keys=True),
            }
            for name in metric_names:
                row[f"cv_{name}"] = result["cv_metrics"].get(name)
                row[f"validation_{name}"] = result["validation_metrics"].get(name)
            writer.writerow(row)
    temporary.replace(path)


def run_screen(
    *,
    dataset_directory: str | Path,
    search_directory: str | Path | None = None,
    output_directory: str | Path | None = None,
    cv_seed: int = CV_SEED,
    estimator_seed: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    screen_start = time.time()
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
        else search / "latent_v1"
    )
    result_path = output / "latent_results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"{result_path} already exists; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    (output / "predictions").mkdir(exist_ok=True)

    candidates = _candidate_configs()
    folds = make_six_prompt_folds(splits["train"], seed=cv_seed)
    offsets = _fold_offsets(splits["train"], folds)
    fold_index = np.full(192, -1, dtype=np.int8)
    for index, heldout in enumerate(offsets):
        fold_index[heldout] = index
    if np.any(fold_index < 0):
        raise AssertionError("fold index is incomplete")

    split_payload = json.loads((search / "split.json").read_text(encoding="utf-8"))
    identity = {
        "schema_version": SCHEMA_VERSION,
        "stage": "validation_only_latent_screen",
        "dataset_run_config_sha256": info.run_config_sha256,
        "feature_manifest_sha256": _canonical_sha256(manifest),
        "split_sha256": _canonical_sha256(split_payload),
        "feature_set": FEATURE_SET,
        "cv_algorithm": CV_ALGORITHM,
        "cv_seed": int(cv_seed),
        "cv_hash_payload": "classical_cv|<seed>|<sample_id>",
        "estimator_seed": int(estimator_seed),
        "selection_metric_order": [
            "cv_macro_prompt_cosine",
            "validation_macro_prompt_cosine",
            "cv_r2_zero",
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
        "dataset_directory": info.directory,
        "search_directory": search,
        "output_directory": output,
        "runtime_budget_note": (
            "One fixed Huber and one fixed ElasticNet configuration per feature "
            "view are included; their measured fit cost is small relative to PLS/PCA."
        ),
        "software": {
            "numpy": _package_version("numpy"),
            "scikit_learn": _package_version("scikit-learn"),
            "joblib": _package_version("joblib"),
            "source_sha256": _sha256_file(Path(__file__)),
        },
    }
    _atomic_json(output / "latent_configs.json", identity | {"candidates": candidates})
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

    train_views = {
        name: _preprocess(train_x_raw, name) for name in PREPROCESSINGS
    }
    validation_views = {
        name: _preprocess(validation_x_raw, name) for name in PREPROCESSINGS
    }
    results: list[dict[str, Any]] = []
    progress = tqdm(candidates, desc="latent oracle screen", unit="candidate")
    for config in progress:
        progress.set_postfix(candidate=config["id"])
        train_x = train_views[str(config["preprocessing"])]
        validation_x = validation_views[str(config["preprocessing"])]
        if config["family"] == "pls_regression":
            result, model, oof, validation_raw, validation_prediction = _evaluate_pls(
                config,
                train_x,
                validation_x,
                train_y,
                validation_y,
                np.asarray(splits["train"]),
                offsets,
            )
            artifact_model: Any = {"pls": model}
        elif config["family"] == "pca_ridge_cv":
            (
                result,
                artifact_model,
                oof,
                validation_raw,
                validation_prediction,
            ) = _evaluate_pca(
                config,
                train_x,
                validation_x,
                train_y,
                validation_y,
                np.asarray(splits["train"]),
                offsets,
                estimator_seed=estimator_seed,
            )
        else:
            (
                result,
                robust_model,
                oof,
                validation_raw,
                validation_prediction,
            ) = _evaluate_robust_linear(
                config,
                train_x,
                validation_x,
                train_y,
                validation_y,
                np.asarray(splits["train"]),
                offsets,
                estimator_seed=estimator_seed,
            )
            artifact_model = {"robust_linear": robust_model}

        model_path = output / "models" / f"{config['id']}.joblib"
        prediction_path = output / "predictions" / f"{config['id']}.npz"
        _atomic_joblib(
            model_path,
            {
                "schema_version": SCHEMA_VERSION,
                "config": dict(config),
                "selected_ridge": result["selected_ridge"],
                "feature_manifest_sha256": identity["feature_manifest_sha256"],
                "split_sha256": identity["split_sha256"],
                "fit_sample_ids": np.asarray(splits["train"], dtype=np.int32),
                "validation_calibration_scale": result[
                    "validation_calibration_scale"
                ],
                "model": artifact_model,
            },
        )
        _atomic_npz(
            prediction_path,
            train_sample_ids=np.asarray(splits["train"], dtype=np.int32),
            validation_sample_ids=np.asarray(splits["validation"], dtype=np.int32),
            cv_fold_index=fold_index,
            cv_oof_predictions=oof,
            validation_raw_predictions=np.asarray(validation_raw, dtype=np.float32),
            validation_predictions=validation_prediction,
            cv_calibration_scales=np.asarray(
                result["cv_calibration_scales"], dtype=np.float64
            ),
            validation_calibration_scale=np.asarray(
                result["validation_calibration_scale"]
            ),
        )
        result["model_artifact"] = model_path
        result["prediction_artifact"] = prediction_path
        results.append(result)

    ranked = sorted(results, key=_result_rank_key, reverse=True)
    for rank, result in enumerate(ranked, 1):
        result["cv_rank"] = rank
    payload = identity | {
        "completed_at_unix": time.time(),
        "runtime_seconds": time.time() - screen_start,
        "candidate_count": len(candidates),
        "winner_id": ranked[0]["config"]["id"],
        "winner_config": ranked[0]["config"],
        "winner_selected_ridge": ranked[0]["selected_ridge"],
        "results": results,
    }
    _atomic_json(result_path, payload)
    _write_results_csv(output / "latent_results.csv", results)
    _atomic_json(
        output / "selected_config.json",
        identity
        | {
            "winner_id": ranked[0]["config"]["id"],
            "winner_config": ranked[0]["config"],
            "winner_selected_ridge": ranked[0]["selected_ridge"],
            "winner_cv_metrics": ranked[0]["cv_metrics"],
            "winner_validation_metrics": ranked[0]["validation_metrics"],
            "validation_calibration_scale": ranked[0][
                "validation_calibration_scale"
            ],
            "model_artifact": ranked[0]["model_artifact"],
        },
    )
    return payload


def self_test() -> None:
    folds = make_six_prompt_folds(np.arange(192, dtype=np.int32))
    np.testing.assert_array_equal(np.sort(np.concatenate(folds)), np.arange(192))

    rng = np.random.default_rng(23)
    features = rng.normal(size=(8, 32, 12)).astype(np.float32)
    centered = _preprocess(features, "prompt_centered")
    np.testing.assert_allclose(centered.mean(axis=1), 0.0, atol=2e-7)

    targets = np.einsum("npf,f->np", features, np.arange(1, 13), optimize=True)
    ridge = _fit_mean_gram_ridge(
        features[:6].reshape(-1, 12), targets[:6].reshape(-1), 1e-8
    )
    prediction = _predict_ridge(ridge, features[6:].reshape(-1, 12)).reshape(2, 32)
    np.testing.assert_allclose(prediction, targets[6:], rtol=3e-4, atol=3e-4)
    np.testing.assert_allclose(
        _zero_intercept_calibration(np.asarray([1.0, -2.0]), np.asarray([3.0, -6.0])),
        3.0,
    )

    small_x = features[:6].reshape(-1, 12)
    small_y = targets[:6].reshape(-1)
    pls = PLSRegression(n_components=2, scale=True).fit(small_x, small_y)
    assert np.asarray(pls.predict(small_x)).reshape(-1).shape == small_y.shape
    pca = PCA(n_components=4, svd_solver="full").fit(small_x)
    assert pca.transform(small_x).shape == (small_x.shape[0], 4)
    configs = _candidate_configs()
    assert len(configs) == 32
    assert sum(config["family"] == "huber_regression" for config in configs) == 2
    robust_config = next(
        config for config in configs if config["family"] == "huber_regression"
    )
    robust = _fit_robust_linear(
        robust_config, features[:6], targets[:6], random_seed=0
    )
    assert _predict_robust_linear(robust, features[6:]).shape == (2, 32)
    print("offline_oracle_latent self-test passed")


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
        help="Default: SEARCH_DIRECTORY/latent_v1",
    )
    parser.add_argument("--cv-seed", type=int, default=CV_SEED)
    parser.add_argument("--estimator-seed", type=int, default=0)
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
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
