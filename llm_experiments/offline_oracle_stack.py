"""Validation-only stacking of classical offline Countdown oracle models.

The script consumes the six-fold OOF and external-validation predictions
written by ``offline_oracle_classical.py``.  Meta models are themselves
cross-fitted with the same prompt folds.  Only the frozen train and validation
prompt IDs are ever applied to the reward-difference memmap; test IDs are
validated for disjointness but are never used to index labels.

The meta-model's own cross-fit calibration is fit-partition-only.  Historical
base artifacts can nevertheless contain fold-label-calibrated predictions;
stack CV metrics inherit that limitation and are calibration diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize, nnls

from llm_experiments.offline_oracle_search import (
    DatasetInfo,
    load_and_verify_dataset,
    load_group_split,
)
from llm_experiments.offline_oracle_utils import pair_level_metrics


SCHEMA_VERSION = 1
SELECTION_METRIC = "macro_prompt_cosine"
DEFAULT_BASE_IDS = (
    "faithful_pr4_ridge10",
    "prompt_centered_ridge1e-5",
    "extratrees_raw_leaf20_maxfeat0p5",
    "hgb_raw_leaves15_lr0p1_l210_iter200",
    "hgb_leaves15_lr0p1_l21_iter200",
)
DEFAULT_ALPHA_GRID = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


@dataclass(frozen=True)
class MetaSpec:
    identifier: str
    kind: str
    alpha: float | None = None


@dataclass(frozen=True)
class StackData:
    train_predictions: np.ndarray
    validation_predictions: np.ndarray
    train_targets: np.ndarray
    validation_targets: np.ndarray
    train_ids: np.ndarray
    validation_ids: np.ndarray
    test_ids: np.ndarray
    fold_index: np.ndarray
    base_ids: tuple[str, ...]
    artifact_paths: tuple[Path, ...]


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


def _parse_base_ids(value: str) -> tuple[str, ...]:
    identifiers = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(identifiers) < 2 or len(set(identifiers)) != len(identifiers):
        raise argparse.ArgumentTypeError(
            "base IDs must contain at least two distinct comma-separated names"
        )
    return identifiers


def _parse_alpha_grid(value: str) -> tuple[float, ...]:
    try:
        alphas = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("alpha grid must contain numbers") from error
    if not alphas or any(not np.isfinite(alpha) or alpha < 0.0 for alpha in alphas):
        raise argparse.ArgumentTypeError("alpha values must be finite and nonnegative")
    return tuple(sorted(set(alphas)))


def _parse_extra_prediction(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("extra prediction must use ID=PATH")
    identifier, raw_path = value.split("=", 1)
    identifier = identifier.strip()
    raw_path = raw_path.strip()
    if not identifier or not raw_path:
        raise argparse.ArgumentTypeError("extra prediction must have nonempty ID and PATH")
    if any(character.isspace() for character in identifier):
        raise argparse.ArgumentTypeError("extra prediction ID may not contain whitespace")
    return identifier, raw_path


def _normalize_extra_predictions(
    values: Sequence[tuple[str, str | Path]],
) -> tuple[tuple[str, Path], ...]:
    normalized: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for raw_identifier, raw_path in values:
        identifier = str(raw_identifier).strip()
        if not identifier or identifier in seen:
            raise ValueError("extra prediction IDs must be nonempty and unique")
        if any(character.isspace() for character in identifier):
            raise ValueError("extra prediction ID may not contain whitespace")
        normalized.append((identifier, Path(raw_path).expanduser().resolve()))
        seen.add(identifier)
    return tuple(normalized)


def _default_classical_directory(search_directory: Path) -> Path:
    """Resolve the v2 directory name used by the classical raw-control run."""

    candidates = (
        search_directory / "classical_v2",
        search_directory / "classical_v2_raw_controls",
    )
    for candidate in candidates:
        if (candidate / "predictions").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find classical_v2 prediction artifacts in "
        f"{', '.join(str(path) for path in candidates)}"
    )


def _expected_fold_index(
    classical_directory: Path, train_sample_ids: np.ndarray
) -> tuple[np.ndarray, str]:
    fold_path = classical_directory / "cv_folds.json"
    if not fold_path.is_file():
        raise FileNotFoundError(fold_path)
    payload = json.loads(fold_path.read_text(encoding="utf-8"))
    recorded_train = np.asarray(payload.get("train_sample_ids"), dtype=np.int64)
    if not np.array_equal(recorded_train, train_sample_ids):
        raise ValueError("Classical CV train IDs disagree with the frozen split")
    folds = payload.get("folds")
    if not isinstance(folds, list) or len(folds) != 6:
        raise ValueError("Classical CV must contain exactly six folds")
    offset = {int(sample_id): index for index, sample_id in enumerate(train_sample_ids)}
    fold_index = np.full(train_sample_ids.size, -1, dtype=np.int8)
    for index, fold in enumerate(folds):
        fold_ids = np.asarray(fold, dtype=np.int64)
        if fold_ids.shape != (32,) or np.unique(fold_ids).size != 32:
            raise ValueError("Every classical CV fold must contain 32 prompt IDs")
        try:
            fold_offsets = np.asarray(
                [offset[int(sample_id)] for sample_id in fold_ids], dtype=np.int64
            )
        except KeyError as error:
            raise ValueError("CV fold contains an ID outside frozen train") from error
        if np.any(fold_index[fold_offsets] >= 0):
            raise ValueError("Classical CV folds overlap")
        fold_index[fold_offsets] = index
    if np.any(fold_index < 0) or not np.array_equal(
        np.bincount(fold_index, minlength=6), np.full(6, 32)
    ):
        raise ValueError("Classical CV folds are not exhaustive 6x32 groups")
    return fold_index, _canonical_sha256(payload)


def _load_prediction_artifact(
    path: Path,
    *,
    train_ids: np.ndarray,
    validation_ids: np.ndarray,
    expected_fold_index: np.ndarray,
    pairs: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as artifact:
        required = {
            "train_sample_ids",
            "validation_sample_ids",
            "cv_fold_index",
            "cv_oof_predictions",
            "validation_predictions",
        }
        missing = required - set(artifact.files)
        if missing:
            raise ValueError(f"{path.name} is missing arrays {sorted(missing)}")
        # Reject any accidentally exported held-out labels or predictions.
        if any("test" in name.lower() for name in artifact.files):
            raise ValueError(f"{path.name} unexpectedly contains a test array")
        if not np.array_equal(artifact["train_sample_ids"], train_ids):
            raise ValueError(f"{path.name} train IDs disagree with frozen split")
        if not np.array_equal(artifact["validation_sample_ids"], validation_ids):
            raise ValueError(f"{path.name} validation IDs disagree with frozen split")
        if not np.array_equal(artifact["cv_fold_index"], expected_fold_index):
            raise ValueError(f"{path.name} has a different CV fold assignment")
        oof = np.asarray(artifact["cv_oof_predictions"], dtype=np.float32)
        validation = np.asarray(artifact["validation_predictions"], dtype=np.float32)
    if oof.shape != (train_ids.size, pairs):
        raise ValueError(f"{path.name} has OOF shape {oof.shape}")
    if validation.shape != (validation_ids.size, pairs):
        raise ValueError(f"{path.name} has validation shape {validation.shape}")
    if not np.all(np.isfinite(oof)) or not np.all(np.isfinite(validation)):
        raise ValueError(f"{path.name} contains non-finite predictions")
    return oof, validation


def load_stack_data(
    *,
    dataset_directory: str | Path,
    search_directory: str | Path | None,
    classical_directory: str | Path | None,
    base_ids: Sequence[str],
    extra_predictions: Sequence[tuple[str, str | Path]] = (),
) -> tuple[DatasetInfo, Path, Path, StackData, dict[str, str]]:
    """Load base predictions and only frozen train/validation labels."""

    info = load_and_verify_dataset(dataset_directory, deep=False)
    search = (
        Path(search_directory).expanduser().resolve()
        if search_directory is not None
        else info.directory / "offline_oracle_v1"
    )
    split_path = search / "split.json"
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    splits = load_group_split(split_path, info)
    train_ids = np.asarray(splits["train"], dtype=np.int32)
    validation_ids = np.asarray(splits["validation"], dtype=np.int32)
    test_ids = np.asarray(splits["test"], dtype=np.int32)
    if (train_ids.size, validation_ids.size, test_ids.size) != (192, 32, 32):
        raise ValueError("Stacker requires the frozen 192/32/32 prompt split")
    classical = (
        Path(classical_directory).expanduser().resolve()
        if classical_directory is not None
        else _default_classical_directory(search)
    )
    config_path = classical / "classical_configs.json"
    classical_config = json.loads(config_path.read_text(encoding="utf-8"))
    if classical_config.get("dataset_run_config_sha256") != info.run_config_sha256:
        raise ValueError("Classical predictions belong to a different dataset")
    split_sha256 = _canonical_sha256(split_payload)
    if classical_config.get("split_sha256") != split_sha256:
        raise ValueError("Classical predictions belong to a different frozen split")
    if classical_config.get("test_labels_read") is not False:
        raise ValueError("Classical artifact does not certify test-label isolation")
    candidate_ids = {
        str(candidate.get("id")) for candidate in classical_config.get("candidates", [])
    }
    unknown = set(base_ids) - candidate_ids
    if unknown:
        raise ValueError(f"Requested base IDs are absent from classical_v2: {sorted(unknown)}")
    extras = _normalize_extra_predictions(extra_predictions)
    collisions = set(base_ids) & {identifier for identifier, _ in extras}
    if collisions:
        raise ValueError(
            f"Extra prediction IDs collide with classical base IDs: {sorted(collisions)}"
        )

    common_fold_index, folds_sha256 = _expected_fold_index(classical, train_ids)
    train_predictions: list[np.ndarray] = []
    validation_predictions: list[np.ndarray] = []
    artifact_paths: list[Path] = []
    artifact_sha256: dict[str, str] = {}
    for identifier in base_ids:
        path = classical / "predictions" / f"{identifier}.npz"
        oof, validation = _load_prediction_artifact(
            path,
            train_ids=train_ids,
            validation_ids=validation_ids,
            expected_fold_index=common_fold_index,
            pairs=info.pairs,
        )
        train_predictions.append(oof)
        validation_predictions.append(validation)
        artifact_paths.append(path)
        artifact_sha256[str(identifier)] = _sha256_file(path)
    for identifier, path in extras:
        oof, validation = _load_prediction_artifact(
            path,
            train_ids=train_ids,
            validation_ids=validation_ids,
            expected_fold_index=common_fold_index,
            pairs=info.pairs,
        )
        train_predictions.append(oof)
        validation_predictions.append(validation)
        artifact_paths.append(path)
        artifact_sha256[identifier] = _sha256_file(path)

    combined_ids = tuple(str(identifier) for identifier in base_ids) + tuple(
        identifier for identifier, _ in extras
    )

    # LABEL SAFETY BOUNDARY: these are the only target indexing operations.
    # test_ids is retained for the audit trail and is never applied here.
    target_memmap = np.load(info.reward_differences_path, mmap_mode="r")
    train_targets = np.asarray(target_memmap[train_ids], dtype=np.float32)
    validation_targets = np.asarray(target_memmap[validation_ids], dtype=np.float32)
    if train_targets.shape != (192, info.pairs):
        raise ValueError("Unexpected frozen train-label shape")
    if validation_targets.shape != (32, info.pairs):
        raise ValueError("Unexpected frozen validation-label shape")
    if not np.all(np.isfinite(train_targets)) or not np.all(
        np.isfinite(validation_targets)
    ):
        raise ValueError("Train/validation labels contain non-finite values")

    return info, search, classical, StackData(
        train_predictions=np.stack(train_predictions, axis=-1),
        validation_predictions=np.stack(validation_predictions, axis=-1),
        train_targets=train_targets,
        validation_targets=validation_targets,
        train_ids=train_ids,
        validation_ids=validation_ids,
        test_ids=test_ids,
        fold_index=common_fold_index,
        base_ids=combined_ids,
        artifact_paths=tuple(artifact_paths),
    ), {
        "split_sha256": split_sha256,
        "cv_folds_sha256": folds_sha256,
        **{f"prediction:{key}": value for key, value in artifact_sha256.items()},
    }


def _flatten(features: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float64).reshape(-1, features.shape[-1])
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    if x.shape[0] != y.size or x.shape[0] == 0:
        raise ValueError("Meta features and targets are incompatible")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Meta fitting arrays contain non-finite values")
    return x, y


def _simplex_weights(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
    x, y = _flatten(features, targets)
    dimension = x.shape[1]

    def objective(weights: np.ndarray) -> float:
        residual = x @ weights - y
        return float(np.mean(np.square(residual)))

    def gradient(weights: np.ndarray) -> np.ndarray:
        return (2.0 / x.shape[0]) * (x.T @ (x @ weights - y))

    result = minimize(
        objective,
        np.full(dimension, 1.0 / dimension, dtype=np.float64),
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, None)] * dimension,
        constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
        options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"Convex stack optimization failed: {result.message}")
    weights = np.maximum(np.asarray(result.x, dtype=np.float64), 0.0)
    total = float(np.sum(weights))
    if total <= 0.0:
        raise RuntimeError("Convex stack returned zero total weight")
    return weights / total


def fit_meta_weights(
    spec: MetaSpec, features: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    x, y = _flatten(features, targets)
    dimension = x.shape[1]
    if spec.kind == "mean":
        return np.full(dimension, 1.0 / dimension, dtype=np.float64)
    if spec.kind == "ridge":
        if spec.alpha is None or spec.alpha < 0.0:
            raise ValueError("Ridge spec requires nonnegative alpha")
        count = float(x.shape[0])
        gram = (x.T @ x) / count
        cross = (x.T @ y) / count
        system = gram + float(spec.alpha) * np.eye(dimension, dtype=np.float64)
        try:
            return np.linalg.solve(system, cross)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(system, cross, rcond=None)[0]
    if spec.kind == "nnls":
        weights, _ = nnls(x, y, maxiter=max(100, 20 * dimension))
        return np.asarray(weights, dtype=np.float64)
    if spec.kind == "convex":
        return _simplex_weights(features, targets)
    raise ValueError(f"Unknown meta model kind {spec.kind!r}")


def _predict(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    prediction = np.einsum(
        "npm,m->np",
        np.asarray(features, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        optimize=True,
    )
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("Meta model produced non-finite predictions")
    return prediction


def zero_intercept_calibration(predictions: np.ndarray, targets: np.ndarray) -> float:
    prediction = np.asarray(predictions, dtype=np.float64).reshape(-1)
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    if prediction.shape != target.shape or prediction.size == 0:
        raise ValueError("Calibration arrays must be nonempty and have equal shape")
    denominator = float(prediction @ prediction)
    if denominator <= np.finfo(np.float64).tiny:
        return 0.0
    scale = float((prediction @ target) / denominator)
    if not np.isfinite(scale):
        raise RuntimeError("Calibration produced a non-finite scale")
    return scale


def _calibrated_prediction(
    raw: np.ndarray, scale: float, prediction_clip: float | None
) -> np.ndarray:
    prediction = np.asarray(raw, dtype=np.float64) * float(scale)
    if prediction_clip is not None:
        prediction = np.clip(
            prediction, -float(prediction_clip), float(prediction_clip)
        )
    return prediction.astype(np.float32)


def _meta_specs(
    alpha_grid: Sequence[float], include_convex: bool
) -> tuple[MetaSpec, ...]:
    specs: list[MetaSpec] = [MetaSpec("simple_mean", "mean")]
    specs.extend(
        MetaSpec(f"ridge_alpha{float(alpha):g}", "ridge", float(alpha))
        for alpha in alpha_grid
    )
    specs.append(MetaSpec("nnls_nonnegative", "nnls"))
    if include_convex:
        specs.append(MetaSpec("convex_sum_to_one", "convex"))
    return tuple(specs)


def cross_fit_meta(
    spec: MetaSpec,
    features: np.ndarray,
    targets: np.ndarray,
    fold_index: np.ndarray,
    *,
    prediction_clip: float | None,
) -> dict[str, Any]:
    if features.shape[:2] != targets.shape or features.shape[0] != fold_index.size:
        raise ValueError("Cross-fit arrays have incompatible prompt dimensions")
    folds = np.unique(fold_index)
    if not np.array_equal(folds, np.arange(6)):
        raise ValueError("Cross-fitting requires common fold indices 0..5")
    raw_oof = np.empty_like(targets, dtype=np.float64)
    calibrated_oof = np.empty_like(targets, dtype=np.float32)
    fold_results: list[dict[str, Any]] = []
    for fold in folds:
        heldout = fold_index == fold
        fitted = ~heldout
        weights = fit_meta_weights(spec, features[fitted], targets[fitted])
        raw_fit = _predict(features[fitted], weights)
        scale = zero_intercept_calibration(raw_fit, targets[fitted])
        raw_heldout = _predict(features[heldout], weights)
        heldout_prediction = _calibrated_prediction(
            raw_heldout, scale, prediction_clip
        )
        raw_oof[heldout] = raw_heldout
        calibrated_oof[heldout] = heldout_prediction
        fold_results.append(
            {
                "fold": int(fold),
                "fit_prompt_count": int(np.sum(fitted)),
                "heldout_prompt_count": int(np.sum(heldout)),
                "weights": weights,
                "calibration_scale": scale,
                "metrics": pair_level_metrics(targets[heldout], heldout_prediction),
            }
        )
    return {
        "raw_predictions": raw_oof.astype(np.float32),
        "predictions": calibrated_oof,
        "raw_metrics": pair_level_metrics(targets, raw_oof),
        "metrics": pair_level_metrics(targets, calibrated_oof),
        "folds": fold_results,
    }


def _rank_key(result: Mapping[str, Any]) -> tuple[float, float, float, int]:
    metrics = result["cv_metrics"]

    def finite(value: Any) -> float:
        number = float(value)
        return number if np.isfinite(number) else -float("inf")

    # Selection is CV-only; external validation is never a model-selection key.
    return (
        finite(metrics[SELECTION_METRIC]),
        finite(metrics["r2_zero"]),
        finite(metrics["pooled_pearson"]),
        -int(result["candidate_order"]),
    )


def _write_results_csv(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    metric_names = sorted(
        {
            name
            for result in results
            for name in (
                set(result["cv_metrics"])
                | set(result["validation_metrics"])
                | set(result["validation_raw_metrics"])
            )
        }
    )
    fields = [
        "rank",
        "id",
        "kind",
        "alpha",
        "weights_json",
        "validation_calibration_scale",
    ] + [f"cv_{name}" for name in metric_names] + [
        f"validation_{name}" for name in metric_names
    ] + [f"validation_raw_{name}" for name in metric_names]
    temporary = path.with_name(f".{path.name}.tmp")
    ranked = sorted(results, key=_rank_key, reverse=True)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, result in enumerate(ranked, 1):
            row = {
                "rank": rank,
                "id": result["id"],
                "kind": result["kind"],
                "alpha": result["alpha"],
                "weights_json": json.dumps(
                    np.asarray(result["final_weights"]).tolist()
                ),
                "validation_calibration_scale": result[
                    "validation_calibration_scale"
                ],
            }
            for name in metric_names:
                row[f"cv_{name}"] = result["cv_metrics"].get(name)
                row[f"validation_{name}"] = result["validation_metrics"].get(name)
                row[f"validation_raw_{name}"] = result[
                    "validation_raw_metrics"
                ].get(name)
            writer.writerow(row)
    temporary.replace(path)


def run_stack(
    *,
    dataset_directory: str | Path,
    search_directory: str | Path | None = None,
    classical_directory: str | Path | None = None,
    output_directory: str | Path | None = None,
    base_ids: Sequence[str] = DEFAULT_BASE_IDS,
    extra_predictions: Sequence[tuple[str, str | Path]] = (),
    alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
    include_convex: bool = True,
    prediction_clip: float | None = 1.1,
    overwrite: bool = False,
) -> dict[str, Any]:
    identifiers = tuple(str(identifier) for identifier in base_ids)
    if len(identifiers) < 2 or len(set(identifiers)) != len(identifiers):
        raise ValueError("base_ids must contain at least two distinct IDs")
    alphas = tuple(sorted(set(float(alpha) for alpha in alpha_grid)))
    if not alphas or any(not np.isfinite(alpha) or alpha < 0.0 for alpha in alphas):
        raise ValueError("alpha_grid must contain finite nonnegative values")
    if prediction_clip is not None and prediction_clip <= 0.0:
        raise ValueError("prediction_clip must be positive when provided")
    extras = _normalize_extra_predictions(extra_predictions)
    collisions = set(identifiers) & {identifier for identifier, _ in extras}
    if collisions:
        raise ValueError(
            f"Extra prediction IDs collide with classical base IDs: {sorted(collisions)}"
        )

    info, search, classical, data, hashes = load_stack_data(
        dataset_directory=dataset_directory,
        search_directory=search_directory,
        classical_directory=classical_directory,
        base_ids=identifiers,
        extra_predictions=extras,
    )
    output = (
        Path(output_directory).expanduser().resolve()
        if output_directory is not None
        else search / "stack_v1"
    )
    result_path = output / "stack_results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"{result_path} already exists; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)

    specs = _meta_specs(alphas, include_convex)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "stage": "validation_only_classical_stack",
        "dataset_directory": info.directory,
        "dataset_run_config_sha256": info.run_config_sha256,
        "search_directory": search,
        "classical_directory": classical,
        "classical_configs_sha256": _sha256_file(
            classical / "classical_configs.json"
        ),
        "split_sha256": hashes["split_sha256"],
        "cv_folds_sha256": hashes["cv_folds_sha256"],
        "prediction_artifact_sha256": {
            identifier: hashes[f"prediction:{identifier}"]
            for identifier in data.base_ids
        },
        "classical_base_ids": identifiers,
        "extra_prediction_artifacts": {
            identifier: path for identifier, path in extras
        },
        "base_ids": data.base_ids,
        "alpha_grid": alphas,
        "include_convex": bool(include_convex),
        "prediction_clip": prediction_clip,
        "selection_metric": SELECTION_METRIC,
        "selection_uses_external_validation": False,
        "train_sample_ids": data.train_ids,
        "validation_sample_ids": data.validation_ids,
        "test_sample_ids": data.test_ids,
        "test_labels_read": False,
        "base_oof_predictions_may_use_heldout_calibration": True,
        "cv_metrics_strictly_out_of_fold": False,
        "common_cv_fold_index": data.fold_index,
        "meta_candidates": [
            {"id": spec.identifier, "kind": spec.kind, "alpha": spec.alpha}
            for spec in specs
        ],
        "started_at_unix": time.time(),
        "source_sha256": _sha256_file(Path(__file__)),
    }
    _atomic_json(output / "stack_config.json", identity)

    results: list[dict[str, Any]] = []
    weight_arrays: dict[str, Any] = {
        "base_ids": np.asarray(data.base_ids),
        "candidate_ids": np.asarray([spec.identifier for spec in specs]),
    }
    prediction_arrays: dict[str, Any] = {
        "validation_sample_ids": data.validation_ids,
        "pair_ids": np.arange(info.pairs, dtype=np.int32),
        "targets": data.validation_targets,
        "base_ids": np.asarray(data.base_ids),
        "base_validation_predictions": data.validation_predictions,
    }
    for candidate_order, spec in enumerate(specs):
        cross_fitted = cross_fit_meta(
            spec,
            data.train_predictions,
            data.train_targets,
            data.fold_index,
            prediction_clip=prediction_clip,
        )
        final_weights = fit_meta_weights(
            spec, data.train_predictions, data.train_targets
        )
        raw_validation = _predict(data.validation_predictions, final_weights)
        validation_scale = zero_intercept_calibration(
            raw_validation, data.validation_targets
        )
        validation_prediction = _calibrated_prediction(
            raw_validation, validation_scale, prediction_clip
        )
        result = {
            "candidate_order": candidate_order,
            "id": spec.identifier,
            "kind": spec.kind,
            "alpha": spec.alpha,
            "cv_metrics": cross_fitted["metrics"],
            "cv_raw_metrics": cross_fitted["raw_metrics"],
            "cv_folds": cross_fitted["folds"],
            "final_weights": final_weights,
            "validation_calibration_scale": validation_scale,
            "validation_raw_metrics": pair_level_metrics(
                data.validation_targets, raw_validation
            ),
            "validation_metrics": pair_level_metrics(
                data.validation_targets, validation_prediction
            ),
        }
        results.append(result)
        key = f"candidate_{candidate_order}"
        weight_arrays[f"{key}_weights"] = final_weights.astype(np.float64)
        weight_arrays[f"{key}_validation_calibration_scale"] = np.asarray(
            validation_scale, dtype=np.float64
        )
        prediction_arrays[f"{key}_raw_predictions"] = raw_validation.astype(
            np.float32
        )
        prediction_arrays[f"{key}_predictions"] = validation_prediction

    ranked = sorted(results, key=_rank_key, reverse=True)
    for rank, result in enumerate(ranked, 1):
        result["cv_rank"] = rank
    winner = ranked[0]
    payload = identity | {
        "completed_at_unix": time.time(),
        "candidate_count": len(results),
        "winner_id": winner["id"],
        "winner_selected_by": [
            "cv_macro_prompt_cosine",
            "cv_r2_zero",
            "cv_pooled_pearson",
            "candidate_order",
        ],
        "winner_final_weights": winner["final_weights"],
        "winner_validation_calibration_scale": winner[
            "validation_calibration_scale"
        ],
        "winner_cv_metrics": winner["cv_metrics"],
        "winner_validation_metrics": winner["validation_metrics"],
        "results": results,
        "test_labels_read": False,
    }
    _atomic_npz(output / "stack_weights.npz", **weight_arrays)
    _atomic_npz(
        output / "validation_predictions.npz", **prediction_arrays
    )
    _write_results_csv(output / "stack_results.csv", results)
    _atomic_json(result_path, payload)
    print(f"Saved validation-only stack search to {output}")
    print(
        f"CV winner {winner['id']}: "
        f"CV {SELECTION_METRIC}={winner['cv_metrics'][SELECTION_METRIC]:.6f}, "
        f"validation={winner['validation_metrics'][SELECTION_METRIC]:.6f}"
    )
    return payload


def self_test() -> None:
    rng = np.random.default_rng(41)
    prompts, pairs, models = 192, 8, 5
    latent = rng.normal(size=(prompts, pairs))
    base = np.stack(
        [latent + rng.normal(scale=scale, size=latent.shape) for scale in (0.2, 0.3, 0.5, 0.8, 1.0)],
        axis=-1,
    ).astype(np.float32)
    true_weights = np.asarray([0.55, 0.30, 0.15, 0.0, 0.0])
    targets = _predict(base, true_weights).astype(np.float32)
    fold_index = np.repeat(np.arange(6, dtype=np.int8), 32)

    mean_spec = MetaSpec("simple_mean", "mean")
    ridge_spec = MetaSpec("ridge_alpha1e-6", "ridge", 1e-6)
    nnls_spec = MetaSpec("nnls_nonnegative", "nnls")
    convex_spec = MetaSpec("convex_sum_to_one", "convex")
    for spec in (mean_spec, ridge_spec, nnls_spec, convex_spec):
        cross_fitted = cross_fit_meta(
            spec,
            base,
            targets,
            fold_index,
            prediction_clip=None,
        )
        assert cross_fitted["predictions"].shape == targets.shape
        assert np.all(np.isfinite(cross_fitted["predictions"]))
        assert len(cross_fitted["folds"]) == 6

    ridge_weights = fit_meta_weights(ridge_spec, base, targets)
    np.testing.assert_allclose(ridge_weights, true_weights, atol=2e-3)
    nonnegative = fit_meta_weights(nnls_spec, base, targets)
    assert np.all(nonnegative >= 0.0)
    convex = fit_meta_weights(convex_spec, base, targets)
    assert np.all(convex >= 0.0)
    np.testing.assert_allclose(np.sum(convex), 1.0, atol=1e-8)
    np.testing.assert_allclose(
        zero_intercept_calibration(
            np.asarray([1.0, -2.0]), np.asarray([2.0, -4.0])
        ),
        2.0,
    )
    assert (
        cross_fit_meta(
            ridge_spec,
            base,
            targets,
            fold_index,
            prediction_clip=None,
        )["metrics"][SELECTION_METRIC]
        > 0.99
    )
    assert _parse_extra_prediction("latent=/tmp/latent.npz") == (
        "latent",
        "/tmp/latent.npz",
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        path = Path(temporary_directory) / "compatible.npz"
        train_ids = np.arange(192, dtype=np.int32)
        validation_ids = np.arange(192, 224, dtype=np.int32)
        compatible_folds = np.repeat(np.arange(6, dtype=np.int8), 32)
        np.savez_compressed(
            path,
            train_sample_ids=train_ids,
            validation_sample_ids=validation_ids,
            cv_fold_index=compatible_folds,
            cv_oof_predictions=np.zeros((192, 8), dtype=np.float32),
            validation_predictions=np.zeros((32, 8), dtype=np.float32),
        )
        loaded_train, loaded_validation = _load_prediction_artifact(
            path,
            train_ids=train_ids,
            validation_ids=validation_ids,
            expected_fold_index=compatible_folds,
            pairs=8,
        )
        assert loaded_train.shape == (192, 8)
        assert loaded_validation.shape == (32, 8)
    print("offline_oracle_stack self-test passed")


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
        "--classical-directory",
        default=None,
        help="Default: SEARCH_DIRECTORY/classical_v2, then classical_v2_raw_controls",
    )
    parser.add_argument(
        "--output-directory",
        default=None,
        help="Default: SEARCH_DIRECTORY/stack_v1",
    )
    parser.add_argument(
        "--base-ids",
        type=_parse_base_ids,
        default=DEFAULT_BASE_IDS,
        help="Comma-separated classical-v2 candidate IDs",
    )
    parser.add_argument(
        "--extra-prediction",
        action="append",
        type=_parse_extra_prediction,
        default=[],
        metavar="ID=PATH",
        help=(
            "Add a compatible NPZ with common train/validation IDs, "
            "cv_fold_index, cv_oof_predictions, and validation_predictions. "
            "Repeat for multiple extras."
        ),
    )
    parser.add_argument(
        "--alpha-grid",
        type=_parse_alpha_grid,
        default=DEFAULT_ALPHA_GRID,
        help="Comma-separated nonnegative ridge coefficients",
    )
    parser.add_argument(
        "--convex-sum-to-one",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--prediction-clip",
        type=float,
        default=1.1,
        help="Symmetric post-calibration clip; pass 0 to disable.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    prediction_clip = None if args.prediction_clip == 0.0 else args.prediction_clip
    run_stack(
        dataset_directory=args.dataset_directory,
        search_directory=args.search_directory,
        classical_directory=args.classical_directory,
        output_directory=args.output_directory,
        base_ids=args.base_ids,
        extra_predictions=args.extra_prediction,
        alpha_grid=args.alpha_grid,
        include_convex=args.convex_sum_to_one,
        prediction_clip=prediction_clip,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
