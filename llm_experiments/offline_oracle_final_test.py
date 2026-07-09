"""Frozen validation/test evaluation for the offline Countdown oracle.

This script is intentionally not a model search.  It consumes artifacts chosen
by the validation-only screens, reconstructs validation and test predictions,
and only then reads test labels for metrics.  The statistical split remains
prompt-grouped: every row reported here is evaluated at the 32-perturbation
pair level for each held-out prompt.

The benchmark split is reward-stratified before this stage, so group membership
is disjoint but not label-blind.  Validation metrics use a scale fitted on
validation and are calibration diagnostics.  Test predictions use that frozen
validation scale and are materialized before any test label is read.
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

from llm_experiments.offline_oracle_classical import (
    FEATURE_SET,
    PREDICTION_CLIP,
    _predict_estimator,
    _transform_features,
)
from llm_experiments.offline_oracle_latent import (
    _predict_pls,
    _predict_ridge,
    _predict_robust_linear,
    _preprocess as _latent_preprocess,
)
from llm_experiments.offline_oracle_projection_sweep import (
    _feature_path as _projection_feature_path,
    _transform as _projection_transform,
)
from llm_experiments.offline_oracle_ranker import _predict_ranker
from llm_experiments.offline_oracle_search import (
    load_and_verify_dataset,
    load_group_split,
)
from llm_experiments.offline_oracle_stack import _predict as _stack_predict
from llm_experiments.offline_oracle_utils import pair_level_metrics


SCHEMA_VERSION = 1
DEFAULT_CLASSICAL_IDS = (
    "faithful_pr4_ridge10",
    "tuned_ridge1e-5",
    "prompt_centered_ridge1e-5",
    "extratrees_raw_leaf20_maxfeat0p5",
    "hgb_raw_leaves15_lr0p1_l210_iter200",
    "hgb_leaves15_lr0p1_l21_iter200",
)
DEFAULT_PROJECTION_IDS = (
    "cs256_seed0_identity_ridge30",
    "cs512_seed2_identity_ridge100",
)
DEFAULT_LATENT_IDS = (
    "pls_prompt_centered_components1",
    "huber_prompt_centered_epsilon1p35_alpha1e-4",
)
DEFAULT_RANKER_IDS: tuple[str, ...] = ()
SUMMARY_METRICS = (
    "macro_prompt_cosine",
    "pooled_pearson",
    "r2_zero",
    "mse",
    "nonzero_sign_accuracy",
    "top1_energy_recall",
    "top4_energy_recall",
    "top8_energy_recall",
)


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


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "id",
        "family",
        "artifact",
        "validation_calibration_scale",
        "split_assignment_uses_reward_labels",
        "validation_calibration_uses_validation_labels",
        "test_calibration_uses_test_labels",
    ]
    for split in ("validation", "test"):
        fields.extend(f"{split}_{metric}" for metric in SUMMARY_METRICS)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {
                "id": row["id"],
                "family": row["family"],
                "artifact": row["artifact"],
                "validation_calibration_scale": row.get(
                    "validation_calibration_scale"
                ),
                "split_assignment_uses_reward_labels": True,
                "validation_calibration_uses_validation_labels": True,
                "test_calibration_uses_test_labels": False,
            }
            for split in ("validation", "test"):
                metrics = row[f"{split}_metrics"]
                for metric in SUMMARY_METRICS:
                    output[f"{split}_{metric}"] = metrics.get(metric)
            writer.writerow(output)
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _result_by_id(path: Path) -> dict[str, Mapping[str, Any]]:
    payload = _read_json(path)
    return {str(row["config"]["id"]): row for row in payload["results"]}


def _apply_scale_clip(
    raw: np.ndarray,
    scale: float,
    clip: float | None = PREDICTION_CLIP,
) -> np.ndarray:
    prediction = np.asarray(raw, dtype=np.float64) * float(scale)
    if clip is not None:
        prediction = np.clip(prediction, -float(clip), float(clip))
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("prediction contains non-finite values")
    return prediction.astype(np.float32)


def _load_pr4_feature_cache(search: Path, samples: int, pairs: int) -> np.memmap:
    path = search / "features" / f"{FEATURE_SET}.npy"
    features = np.load(path, mmap_mode="r")
    if features.shape != (samples, pairs, 256) or features.dtype != np.float32:
        raise ValueError(f"{path} has incompatible PR4 feature shape/dtype")
    return features


def _load_targets(info: Any, sample_ids: np.ndarray) -> np.ndarray:
    targets = np.load(info.reward_differences_path, mmap_mode="r")
    selected = np.asarray(targets[np.asarray(sample_ids, dtype=np.int64)], dtype=np.float32)
    if selected.shape != (len(sample_ids), info.pairs) or not np.all(np.isfinite(selected)):
        raise ValueError("selected target labels have an invalid shape or non-finite values")
    return selected


def _predict_classical(
    *,
    model_path: Path,
    pr4_features: np.memmap,
    validation_ids: np.ndarray,
    test_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], float]:
    artifact = joblib.load(model_path)
    config = dict(artifact["config"])
    if np.asarray(artifact["fit_sample_ids"]).shape != (192,):
        raise ValueError(f"{model_path.name} was not fit on 192 training prompts")
    preprocessing = str(config["preprocessing"])
    validation_x = _transform_features(
        np.asarray(pr4_features[validation_ids], dtype=np.float32), preprocessing
    )
    test_x = _transform_features(
        np.asarray(pr4_features[test_ids], dtype=np.float32), preprocessing
    )
    scale = float(artifact["validation_calibration_scale"])
    clip = config.get("prediction_clip", PREDICTION_CLIP)
    validation_raw = _predict_estimator(artifact["estimator"], validation_x)
    test_raw = _predict_estimator(artifact["estimator"], test_x)
    return (
        _apply_scale_clip(validation_raw, scale, clip),
        _apply_scale_clip(test_raw, scale, clip),
        config,
        scale,
    )


def _predict_projection(
    *,
    prediction_path: Path,
    projection_directory: Path,
    validation_ids: np.ndarray,
    test_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], float]:
    with np.load(prediction_path, allow_pickle=False) as artifact:
        config = json.loads(str(artifact["config_json"]))
        feature_rms = np.asarray(artifact["feature_rms"], dtype=np.float64)
        weights = np.asarray(artifact["weights"], dtype=np.float64)
        scale = float(np.asarray(artifact["validation_calibration_scale"]).item())
    cache = np.load(
        _projection_feature_path(
            projection_directory,
            int(config["sketch_size"]),
            int(config["sketch_seed"]),
        ),
        mmap_mode="r",
    )
    expected_features = int(config["sketch_size"]) * 2
    if cache.ndim != 3 or cache.shape[-1] != expected_features:
        raise ValueError(f"{prediction_path.name} projection cache shape mismatch")
    preprocessing = str(config["preprocessing"])
    clip = config.get("prediction_clip", PREDICTION_CLIP)

    def predict(sample_ids: np.ndarray) -> np.ndarray:
        x = _projection_transform(
            np.asarray(cache[sample_ids], dtype=np.float32), preprocessing
        )
        flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
        raw = (flat / feature_rms) @ weights
        return _apply_scale_clip(raw.reshape(x.shape[:2]), scale, clip)

    return predict(validation_ids), predict(test_ids), config, scale


def _predict_latent_model(model: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    if "pls" in model:
        return _predict_pls(model["pls"], features)
    if "robust_linear" in model:
        return _predict_robust_linear(model["robust_linear"], features)
    if "pca" in model and "ridge" in model:
        flat = features.reshape(-1, features.shape[-1])
        latent = model["pca"].transform(flat)
        prediction = _predict_ridge(model["ridge"], latent)
        return prediction.reshape(features.shape[:2])
    raise ValueError("latent model artifact has an unknown model family")


def _predict_latent(
    *,
    model_path: Path,
    pr4_features: np.memmap,
    validation_ids: np.ndarray,
    test_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], float]:
    artifact = joblib.load(model_path)
    config = dict(artifact["config"])
    preprocessing = str(config["preprocessing"])
    validation_x = _latent_preprocess(
        np.asarray(pr4_features[validation_ids], dtype=np.float32), preprocessing
    )
    test_x = _latent_preprocess(
        np.asarray(pr4_features[test_ids], dtype=np.float32), preprocessing
    )
    scale = float(artifact["validation_calibration_scale"])
    clip = config.get("prediction_clip", PREDICTION_CLIP)
    validation_raw = _predict_latent_model(artifact["model"], validation_x)
    test_raw = _predict_latent_model(artifact["model"], test_x)
    return (
        _apply_scale_clip(validation_raw, scale, clip),
        _apply_scale_clip(test_raw, scale, clip),
        config,
        scale,
    )


def _predict_ranker_artifact(
    *,
    model_path: Path,
    pr4_features: np.memmap,
    validation_ids: np.ndarray,
    test_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], float]:
    artifact = joblib.load(model_path)
    config = dict(artifact["config"])
    preprocessing = str(config["preprocessing"])
    validation_x = _transform_features(
        np.asarray(pr4_features[validation_ids], dtype=np.float32), preprocessing
    )
    test_x = _transform_features(
        np.asarray(pr4_features[test_ids], dtype=np.float32), preprocessing
    )
    scale = float(artifact["validation_calibration_scale"])
    clip = config.get("prediction_clip", PREDICTION_CLIP)
    validation_raw = _predict_ranker(artifact["model"], validation_x)
    test_raw = _predict_ranker(artifact["model"], test_x)
    return (
        _apply_scale_clip(validation_raw, scale, clip),
        _apply_scale_clip(test_raw, scale, clip),
        config,
        scale,
    )


class PredictionResolver:
    def __init__(
        self,
        *,
        search: Path,
        classical_directory: Path,
        projection_directory: Path,
        latent_directory: Path,
        ranker_directory: Path,
        stack_directory: Path,
        pr4_features: np.memmap,
        validation_ids: np.ndarray,
        test_ids: np.ndarray,
    ) -> None:
        self.search = search
        self.classical_directory = classical_directory
        self.projection_directory = projection_directory
        self.latent_directory = latent_directory
        self.ranker_directory = ranker_directory
        self.stack_directory = stack_directory
        self.pr4_features = pr4_features
        self.validation_ids = validation_ids
        self.test_ids = test_ids
        self.cache: dict[str, dict[str, Any]] = {}
        self.stack_config = _read_json(stack_directory / "stack_config.json")

    def classical(self, identifier: str) -> dict[str, Any]:
        key = f"classical:{identifier}"
        if key not in self.cache:
            path = self.classical_directory / "models" / f"{identifier}.joblib"
            validation, test, config, scale = _predict_classical(
                model_path=path,
                pr4_features=self.pr4_features,
                validation_ids=self.validation_ids,
                test_ids=self.test_ids,
            )
            self.cache[key] = {
                "id": identifier,
                "family": config.get("family", "classical"),
                "artifact": path,
                "validation": validation,
                "test": test,
                "config": config,
                "validation_calibration_scale": scale,
            }
        return self.cache[key]

    def projection_from_path(self, alias: str, path: Path) -> dict[str, Any]:
        key = f"projection:{alias}"
        if key not in self.cache:
            validation, test, config, scale = _predict_projection(
                prediction_path=path,
                projection_directory=self.projection_directory,
                validation_ids=self.validation_ids,
                test_ids=self.test_ids,
            )
            self.cache[key] = {
                "id": alias,
                "family": "projection_ridge",
                "artifact": path,
                "validation": validation,
                "test": test,
                "config": config,
                "validation_calibration_scale": scale,
            }
        return self.cache[key]

    def projection(self, identifier: str) -> dict[str, Any]:
        return self.projection_from_path(
            identifier,
            self.projection_directory / "predictions" / f"{identifier}.npz",
        )

    def latent_from_path(self, alias: str, path: Path) -> dict[str, Any]:
        key = f"latent:{alias}"
        if key not in self.cache:
            model_path = self.latent_directory / "models" / f"{path.stem}.joblib"
            validation, test, config, scale = _predict_latent(
                model_path=model_path,
                pr4_features=self.pr4_features,
                validation_ids=self.validation_ids,
                test_ids=self.test_ids,
            )
            self.cache[key] = {
                "id": alias,
                "family": config.get("family", "latent"),
                "artifact": model_path,
                "validation": validation,
                "test": test,
                "config": config,
                "validation_calibration_scale": scale,
            }
        return self.cache[key]

    def latent(self, identifier: str) -> dict[str, Any]:
        return self.latent_from_path(
            identifier,
            self.latent_directory / "predictions" / f"{identifier}.npz",
        )

    def ranker_from_path(self, alias: str, path: Path) -> dict[str, Any]:
        key = f"ranker:{alias}"
        if key not in self.cache:
            model_path = self.ranker_directory / "models" / f"{path.stem}.joblib"
            validation, test, config, scale = _predict_ranker_artifact(
                model_path=model_path,
                pr4_features=self.pr4_features,
                validation_ids=self.validation_ids,
                test_ids=self.test_ids,
            )
            self.cache[key] = {
                "id": alias,
                "family": config.get("family", "pairwise_ranker"),
                "artifact": model_path,
                "validation": validation,
                "test": test,
                "config": config,
                "validation_calibration_scale": scale,
            }
        return self.cache[key]

    def ranker(self, identifier: str) -> dict[str, Any]:
        return self.ranker_from_path(
            identifier,
            self.ranker_directory / "predictions" / f"{identifier}.npz",
        )

    def extra(self, alias: str) -> dict[str, Any]:
        extras = self.stack_config.get("extra_prediction_artifacts", {})
        if alias not in extras:
            raise KeyError(f"Stack extra alias {alias!r} is not recorded")
        path = Path(extras[alias]).expanduser().resolve()
        parts = set(path.parts)
        if "projection_sweep_v1" in parts:
            return self.projection_from_path(alias, path)
        if "latent_v1" in parts:
            return self.latent_from_path(alias, path)
        if "ranker_v1" in parts:
            return self.ranker_from_path(alias, path)
        raise ValueError(f"Cannot infer predictor family for extra artifact {path}")

    def by_stack_base_id(self, identifier: str) -> dict[str, Any]:
        classical_ids = tuple(self.stack_config.get("classical_base_ids", ()))
        if identifier in classical_ids:
            return self.classical(identifier)
        return self.extra(identifier)


def _stack_candidate(stack_directory: Path, identifier: str | None) -> Mapping[str, Any]:
    payload = _read_json(stack_directory / "stack_results.json")
    chosen = str(identifier or payload["winner_id"])
    for row in payload["results"]:
        if str(row["id"]) == chosen:
            return row
    raise KeyError(f"Stack candidate {chosen!r} not found")


def _predict_stack(
    *,
    resolver: PredictionResolver,
    stack_directory: Path,
    stack_id: str | None,
) -> dict[str, Any]:
    candidate = _stack_candidate(stack_directory, stack_id)
    stack_config = _read_json(stack_directory / "stack_config.json")
    base_ids = tuple(str(identifier) for identifier in stack_config["base_ids"])
    bases = [resolver.by_stack_base_id(identifier) for identifier in base_ids]
    validation_stack = np.stack([base["validation"] for base in bases], axis=-1)
    test_stack = np.stack([base["test"] for base in bases], axis=-1)
    weights = np.asarray(candidate["final_weights"], dtype=np.float64)
    if weights.shape != (len(base_ids),):
        raise ValueError("Stack weight vector does not match base IDs")
    scale = float(candidate["validation_calibration_scale"])
    clip = stack_config.get("prediction_clip", PREDICTION_CLIP)
    validation = _apply_scale_clip(_stack_predict(validation_stack, weights), scale, clip)
    test = _apply_scale_clip(_stack_predict(test_stack, weights), scale, clip)
    return {
        "id": f"stack_extended_{candidate['id']}",
        "family": "stack_extended",
        "artifact": stack_directory / "stack_results.json",
        "validation": validation,
        "test": test,
        "config": {
            "stack_candidate_id": candidate["id"],
            "kind": candidate["kind"],
            "alpha": candidate["alpha"],
            "base_ids": base_ids,
            "weights": weights,
            "selection_uses_external_validation": False,
        },
        "validation_calibration_scale": scale,
        "base_ids": base_ids,
        "base_artifacts": [base["artifact"] for base in bases],
    }


def _resolve_direct_models(
    *,
    resolver: PredictionResolver,
    classical_ids: Sequence[str],
    projection_ids: Sequence[str],
    latent_ids: Sequence[str],
    ranker_ids: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "id": "zero",
            "family": "constant_zero",
            "artifact": None,
            "validation": np.zeros(
                (resolver.validation_ids.size, resolver.pr4_features.shape[1]),
                dtype=np.float32,
            ),
            "test": np.zeros(
                (resolver.test_ids.size, resolver.pr4_features.shape[1]),
                dtype=np.float32,
            ),
            "config": {"kind": "constant_zero"},
            "validation_calibration_scale": 0.0,
        }
    ]
    rows.extend(resolver.classical(identifier) for identifier in classical_ids)
    rows.extend(resolver.projection(identifier) for identifier in projection_ids)
    rows.extend(resolver.latent(identifier) for identifier in latent_ids)
    rows.extend(resolver.ranker(identifier) for identifier in ranker_ids)
    return rows


def _prediction_rows(
    *,
    models: Sequence[Mapping[str, Any]],
    validation_y: np.ndarray,
    test_y: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        validation_prediction = np.asarray(model["validation"], dtype=np.float32)
        test_prediction = np.asarray(model["test"], dtype=np.float32)
        row = {
            "id": str(model["id"]),
            "family": str(model["family"]),
            "artifact": model.get("artifact"),
            "config": model.get("config", {}),
            "validation_calibration_scale": model.get("validation_calibration_scale"),
            "validation_metrics": pair_level_metrics(validation_y, validation_prediction),
            "test_metrics": pair_level_metrics(test_y, test_prediction),
        }
        rows.append(row)
    return rows


def _export_full_prediction(
    *,
    path: Path,
    info: Any,
    splits: Mapping[str, np.ndarray],
    validation_predictions: np.ndarray,
    test_predictions: np.ndarray,
    validation_targets: np.ndarray,
    test_targets: np.ndarray,
    model_id: str,
) -> None:
    full = np.full((info.samples, info.pairs), np.nan, dtype=np.float32)
    full[np.asarray(splits["validation"], dtype=np.int64)] = validation_predictions
    full[np.asarray(splits["test"], dtype=np.int64)] = test_predictions
    targets = np.full((info.samples, info.pairs), np.nan, dtype=np.float32)
    targets[np.asarray(splits["validation"], dtype=np.int64)] = validation_targets
    targets[np.asarray(splits["test"], dtype=np.int64)] = test_targets
    _atomic_npz(
        path,
        model_id=np.asarray(model_id),
        split_names=np.asarray(["validation", "test"]),
        validation_sample_ids=np.asarray(splits["validation"], dtype=np.int32),
        test_sample_ids=np.asarray(splits["test"], dtype=np.int32),
        predictions=full,
        targets=targets,
    )


def run_final_test(
    *,
    dataset_directory: str | Path,
    search_directory: str | Path | None = None,
    output_directory: str | Path | None = None,
    classical_directory: str | Path | None = None,
    projection_directory: str | Path | None = None,
    latent_directory: str | Path | None = None,
    ranker_directory: str | Path | None = None,
    stack_directory: str | Path | None = None,
    classical_ids: Sequence[str] = DEFAULT_CLASSICAL_IDS,
    projection_ids: Sequence[str] = DEFAULT_PROJECTION_IDS,
    latent_ids: Sequence[str] = DEFAULT_LATENT_IDS,
    ranker_ids: Sequence[str] = DEFAULT_RANKER_IDS,
    stack_id: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    info = load_and_verify_dataset(dataset_directory, deep=False)
    search = (
        Path(search_directory).expanduser().resolve()
        if search_directory is not None
        else info.directory / "offline_oracle_v1"
    )
    output = (
        Path(output_directory).expanduser().resolve()
        if output_directory is not None
        else search / "final_test_v1"
    )
    result_path = output / "final_test_results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"{result_path} exists; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)

    split_path = search / "split.json"
    split_payload = _read_json(split_path)
    splits = load_group_split(split_path, info)
    for name, expected in (("train", 192), ("validation", 32), ("test", 32)):
        if np.asarray(splits[name]).shape != (expected,):
            raise ValueError(f"Unexpected {name} split shape")
    pr4_features = _load_pr4_feature_cache(search, info.samples, info.pairs)
    validation_ids = np.asarray(splits["validation"], dtype=np.int64)
    test_ids = np.asarray(splits["test"], dtype=np.int64)
    # Prediction artifacts and validation labels are loaded before this point.
    # The only test-label indexing in the script happens below.
    validation_y = _load_targets(info, validation_ids)

    resolver = PredictionResolver(
        search=search,
        classical_directory=(
            Path(classical_directory).expanduser().resolve()
            if classical_directory is not None
            else search / "classical_v2_raw_controls"
        ),
        projection_directory=(
            Path(projection_directory).expanduser().resolve()
            if projection_directory is not None
            else search / "projection_sweep_v1"
        ),
        latent_directory=(
            Path(latent_directory).expanduser().resolve()
            if latent_directory is not None
            else search / "latent_v1"
        ),
        ranker_directory=(
            Path(ranker_directory).expanduser().resolve()
            if ranker_directory is not None
            else search / "ranker_v1"
        ),
        stack_directory=(
            Path(stack_directory).expanduser().resolve()
            if stack_directory is not None
            else search / "stack_extended_v1"
        ),
        pr4_features=pr4_features,
        validation_ids=validation_ids,
        test_ids=test_ids,
    )
    direct_models = _resolve_direct_models(
        resolver=resolver,
        classical_ids=classical_ids,
        projection_ids=projection_ids,
        latent_ids=latent_ids,
        ranker_ids=ranker_ids,
    )
    stack_model = _predict_stack(
        resolver=resolver,
        stack_directory=resolver.stack_directory,
        stack_id=stack_id,
    )
    models = direct_models + [stack_model]

    # LABEL SAFETY BOUNDARY: test labels are not read until all model choices
    # and predictions have been materialized.
    test_y = _load_targets(info, test_ids)
    rows = _prediction_rows(
        models=models,
        validation_y=validation_y,
        test_y=test_y,
    )
    ranked_by_validation = sorted(
        rows,
        key=lambda row: float(row["validation_metrics"]["macro_prompt_cosine"]),
        reverse=True,
    )
    ranked_by_test = sorted(
        rows,
        key=lambda row: float(row["test_metrics"]["macro_prompt_cosine"]),
        reverse=True,
    )
    for rank, row in enumerate(ranked_by_validation, 1):
        row["validation_rank"] = rank
    for rank, row in enumerate(ranked_by_test, 1):
        row["test_rank"] = rank

    best_stack = stack_model
    _export_full_prediction(
        path=output / "final_stack_predictions_full.npz",
        info=info,
        splits=splits,
        validation_predictions=np.asarray(best_stack["validation"], dtype=np.float32),
        test_predictions=np.asarray(best_stack["test"], dtype=np.float32),
        validation_targets=validation_y,
        test_targets=test_y,
        model_id=str(best_stack["id"]),
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "frozen_validation_and_test",
        "dataset_directory": info.directory,
        "dataset_run_config_sha256": info.run_config_sha256,
        "search_directory": search,
        "split_sha256": _canonical_sha256(split_payload),
        "train_sample_ids": np.asarray(splits["train"], dtype=np.int32),
        "validation_sample_ids": np.asarray(splits["validation"], dtype=np.int32),
        "test_sample_ids": np.asarray(splits["test"], dtype=np.int32),
        "test_labels_read": True,
        "test_labels_read_after_predictions_materialized": True,
        "split_assignment_uses_reward_labels": True,
        "split_assignment_label_blind": False,
        "validation_calibration_uses_validation_labels": True,
        "validation_metrics_strictly_heldout": False,
        "test_calibration_uses_test_labels": False,
        "classical_ids": tuple(classical_ids),
        "projection_ids": tuple(projection_ids),
        "latent_ids": tuple(latent_ids),
        "ranker_ids": tuple(ranker_ids),
        "stack_id": stack_model["config"]["stack_candidate_id"],
        "software": {
            "numpy": _package_version("numpy"),
            "scikit_learn": _package_version("scikit-learn"),
            "joblib": _package_version("joblib"),
            "source_sha256": _sha256_file(Path(__file__)),
        },
        "completed_at_unix": time.time(),
    }
    payload = manifest | {
        "models": rows,
        "validation_winner_id": ranked_by_validation[0]["id"],
        "test_winner_id": ranked_by_test[0]["id"],
        "exported_full_prediction": output / "final_stack_predictions_full.npz",
    }
    _atomic_json(output / "final_test_manifest.json", manifest)
    _atomic_json(result_path, payload)
    _write_csv(output / "final_test_results.csv", rows)
    return payload


def _parse_ids(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("IDs must be unique")
    return values


def self_test() -> None:
    raw = np.asarray([[2.0, -3.0, 0.5]], dtype=np.float32)
    np.testing.assert_allclose(
        _apply_scale_clip(raw, 2.0, 1.1),
        np.asarray([[1.1, -1.1, 1.0]], dtype=np.float32),
    )
    payload = {"b": np.asarray([2, 1]), "a": {"x": np.float32(1.0)}}
    assert _canonical_sha256(payload) == _canonical_sha256(payload)
    print("offline_oracle_final_test self-test passed")


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
        help="Default: SEARCH_DIRECTORY/final_test_v1",
    )
    parser.add_argument("--classical-directory", default=None)
    parser.add_argument("--projection-directory", default=None)
    parser.add_argument("--latent-directory", default=None)
    parser.add_argument("--ranker-directory", default=None)
    parser.add_argument("--stack-directory", default=None)
    parser.add_argument("--classical-ids", type=_parse_ids, default=DEFAULT_CLASSICAL_IDS)
    parser.add_argument("--projection-ids", type=_parse_ids, default=DEFAULT_PROJECTION_IDS)
    parser.add_argument("--latent-ids", type=_parse_ids, default=DEFAULT_LATENT_IDS)
    parser.add_argument("--ranker-ids", type=_parse_ids, default=DEFAULT_RANKER_IDS)
    parser.add_argument("--stack-id", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    run_final_test(
        dataset_directory=args.dataset_directory,
        search_directory=args.search_directory,
        output_directory=args.output_directory,
        classical_directory=args.classical_directory,
        projection_directory=args.projection_directory,
        latent_directory=args.latent_directory,
        ranker_directory=args.ranker_directory,
        stack_directory=args.stack_directory,
        classical_ids=args.classical_ids,
        projection_ids=args.projection_ids,
        latent_ids=args.latent_ids,
        ranker_ids=args.ranker_ids,
        stack_id=args.stack_id,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
