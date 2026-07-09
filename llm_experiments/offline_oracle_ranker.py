"""Validation-only pairwise ranking oracle for Countdown perturbations.

The direct regressors try to predict reward differences in absolute units.
For ES selection we often care first about relative ordering inside a prompt:
which perturbations look more promising than their siblings?  This runner fits
linear pairwise classifiers on within-prompt feature differences, then uses the
learned linear score as a reward-difference surrogate with validation-only
zero-intercept calibration.

For exact historical reproduction, every CV fold and validation are calibrated
on their own labels.  The resulting calibrated MSE/R2 values are diagnostics,
not strictly held-out estimates; test labels are never used for calibration.
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
from sklearn.svm import LinearSVC
from tqdm.auto import tqdm

from llm_experiments.offline_oracle_classical import (
    CV_ALGORITHM,
    CV_SEED,
    FEATURE_SET,
    PREDICTION_CLIP,
    _calibrate_and_clip,
    _transform_features,
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
NONZERO_THRESHOLD = 1e-7
PREPROCESSINGS = ("identity", "prompt_center")
C_VALUES = (0.01, 0.1, 1.0, 10.0)


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
    payload = f"ranker|{int(base_seed)}|{identifier}|{int(fold)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _fold_offsets(
    train_ids: np.ndarray,
    folds: Sequence[np.ndarray],
) -> tuple[np.ndarray, ...]:
    offset = {int(sample_id): index for index, sample_id in enumerate(train_ids)}
    offsets = tuple(
        np.asarray([offset[int(sample_id)] for sample_id in fold], dtype=np.int64)
        for fold in folds
    )
    joined = np.concatenate(offsets)
    if np.unique(joined).size != train_ids.size:
        raise ValueError("CV folds overlap")
    return offsets


def _candidate_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for preprocessing in PREPROCESSINGS:
        for c_value in C_VALUES:
            configs.append(
                {
                    "id": (
                        f"pairwise_svc_{preprocessing}_C"
                        f"{str(c_value).replace('.', 'p')}"
                    ),
                    "family": "pairwise_linear_svc",
                    "preprocessing": preprocessing,
                    "C": float(c_value),
                    "class_weight": "balanced",
                    "fit_intercept": False,
                    "max_iter": 10000,
                    "tol": 1e-4,
                    "calibration": "heldout_zero_intercept",
                    "prediction_clip": PREDICTION_CLIP,
                }
            )
    for order, config in enumerate(configs):
        config.update(
            {
                "candidate_order": order,
                "feature_set": FEATURE_SET,
                "pairwise_threshold": NONZERO_THRESHOLD,
            }
        )
    return configs


def _pairwise_training_data(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    threshold: float = NONZERO_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    if x.ndim != 3 or y.shape != x.shape[:2]:
        raise ValueError("features/targets must have shapes [prompts,pairs,d] and [prompts,pairs]")
    pairs = x.shape[1]
    rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for prompt_x, prompt_y in zip(x, y):
        prompt_rows = []
        prompt_labels = []
        for i in range(pairs):
            for j in range(i + 1, pairs):
                delta = float(prompt_y[i] - prompt_y[j])
                if abs(delta) <= threshold:
                    continue
                diff = prompt_x[i] - prompt_x[j]
                label = 1 if delta > 0.0 else 0
                prompt_rows.append(diff)
                prompt_labels.append(label)
                prompt_rows.append(-diff)
                prompt_labels.append(1 - label)
        if prompt_rows:
            rows.append(np.asarray(prompt_rows, dtype=np.float32))
            labels.append(np.asarray(prompt_labels, dtype=np.int8))
    if not rows:
        raise ValueError("no non-tied pairwise comparisons were available")
    return np.concatenate(rows, axis=0), np.concatenate(labels, axis=0)


def _fit_ranker(
    config: Mapping[str, Any],
    features: np.ndarray,
    targets: np.ndarray,
    *,
    random_seed: int,
) -> dict[str, Any]:
    x_pair, y_pair = _pairwise_training_data(
        features, targets, threshold=float(config["pairwise_threshold"])
    )
    estimator = LinearSVC(
        C=float(config["C"]),
        class_weight=str(config["class_weight"]),
        fit_intercept=bool(config["fit_intercept"]),
        max_iter=int(config["max_iter"]),
        tol=float(config["tol"]),
        dual=False,
        random_state=int(random_seed),
    )
    estimator.fit(x_pair, y_pair)
    coef = np.asarray(estimator.coef_, dtype=np.float64).reshape(-1)
    if coef.shape != (features.shape[-1],) or not np.all(np.isfinite(coef)):
        raise RuntimeError("ranker produced an invalid coefficient vector")
    return {
        "kind": "pairwise_linear_svc",
        "coef": coef,
        "classes": np.asarray(estimator.classes_),
        "n_pairwise_examples": int(y_pair.size),
        "random_seed": int(random_seed),
        "n_iter": np.asarray(estimator.n_iter_).tolist(),
    }


def _predict_ranker(model: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    coef = np.asarray(model["coef"], dtype=np.float64)
    raw = x.reshape(-1, x.shape[-1]).astype(np.float64) @ coef
    if not np.all(np.isfinite(raw)):
        raise RuntimeError("ranker produced non-finite predictions")
    return raw.reshape(x.shape[:2])


def _rank_key(result: Mapping[str, Any]) -> tuple[float, float, float, float]:
    cv = result["cv_metrics"]
    validation = result["validation_metrics"]

    def finite(value: Any) -> float:
        number = float(value)
        return number if np.isfinite(number) else -float("inf")

    return (
        finite(cv[SELECTION_METRIC]),
        finite(validation[SELECTION_METRIC]),
        finite(cv["top8_energy_recall"]),
        -float(result["config"]["candidate_order"]),
    )


def _write_csv(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    metric_names = sorted(
        {
            metric
            for result in results
            for metric in set(result["cv_metrics"]) | set(result["validation_metrics"])
        }
    )
    fields = [
        "rank",
        "id",
        "preprocessing",
        "C",
        "validation_calibration_scale",
        "mean_cv_calibration_scale",
        "pairwise_examples_final",
    ] + [f"cv_{name}" for name in metric_names] + [
        f"validation_{name}" for name in metric_names
    ]
    ranked = sorted(results, key=_rank_key, reverse=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, result in enumerate(ranked, 1):
            row = {
                "rank": rank,
                "id": result["config"]["id"],
                "preprocessing": result["config"]["preprocessing"],
                "C": result["config"]["C"],
                "validation_calibration_scale": result[
                    "validation_calibration_scale"
                ],
                "mean_cv_calibration_scale": float(
                    np.mean(result["cv_calibration_scales"])
                ),
                "pairwise_examples_final": result["model"]["n_pairwise_examples"],
            }
            for name in metric_names:
                row[f"cv_{name}"] = result["cv_metrics"].get(name)
                row[f"validation_{name}"] = result["validation_metrics"].get(name)
            writer.writerow(row)
    temporary.replace(path)


def _load_frozen_data(
    dataset_directory: str | Path,
    search_directory: str | Path | None,
) -> tuple[DatasetInfo, Path, dict[str, Any], dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    info = load_and_verify_dataset(dataset_directory, deep=False)
    search = (
        Path(search_directory).expanduser().resolve()
        if search_directory is not None
        else info.directory / "offline_oracle_v1"
    )
    manifest_path = search / "feature_manifest.json"
    split_path = search / "split.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_run_config_sha256") != info.run_config_sha256:
        raise ValueError("feature manifest belongs to another dataset")
    if manifest.get("split_sha256") != _canonical_sha256(split_payload):
        raise ValueError("feature manifest and split fingerprint disagree")
    splits = load_group_split(split_path, info)
    feature_path = search / "features" / f"{FEATURE_SET}.npy"
    feature_cache = np.load(feature_path, mmap_mode="r")
    if feature_cache.shape != (info.samples, info.pairs, 256):
        raise ValueError("PR4 feature cache has an unexpected shape")
    train_ids = np.asarray(splits["train"], dtype=np.int64)
    validation_ids = np.asarray(splits["validation"], dtype=np.int64)
    train_x = np.asarray(feature_cache[train_ids], dtype=np.float32)
    validation_x = np.asarray(feature_cache[validation_ids], dtype=np.float32)
    targets = np.load(info.reward_differences_path, mmap_mode="r")
    train_y = np.asarray(targets[train_ids], dtype=np.float32)
    validation_y = np.asarray(targets[validation_ids], dtype=np.float32)
    return info, search, manifest, splits, train_x, validation_x, train_y, validation_y


def run_screen(
    *,
    dataset_directory: str | Path,
    search_directory: str | Path | None = None,
    output_directory: str | Path | None = None,
    cv_seed: int = CV_SEED,
    estimator_seed: int = 0,
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
        else search / "ranker_v1"
    )
    result_path = output / "ranker_results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"{result_path} exists; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    (output / "predictions").mkdir(exist_ok=True)

    configs = _candidate_configs()
    folds = make_six_prompt_folds(np.asarray(splits["train"], dtype=np.int64), seed=cv_seed)
    offsets = _fold_offsets(np.asarray(splits["train"], dtype=np.int64), folds)
    fold_index = np.full(192, -1, dtype=np.int8)
    for index, heldout in enumerate(offsets):
        fold_index[heldout] = index
    if np.any(fold_index < 0):
        raise AssertionError("fold index is incomplete")

    train_views = {
        name: _transform_features(train_x_raw, name) for name in PREPROCESSINGS
    }
    validation_views = {
        name: _transform_features(validation_x_raw, name) for name in PREPROCESSINGS
    }
    split_payload = json.loads((search / "split.json").read_text(encoding="utf-8"))
    identity = {
        "schema_version": SCHEMA_VERSION,
        "stage": "validation_only_pairwise_ranker",
        "dataset_run_config_sha256": info.run_config_sha256,
        "feature_manifest_sha256": _canonical_sha256(manifest),
        "split_sha256": _canonical_sha256(split_payload),
        "feature_set": FEATURE_SET,
        "cv_algorithm": CV_ALGORITHM,
        "cv_seed": int(cv_seed),
        "estimator_seed": int(estimator_seed),
        "train_sample_ids": splits["train"],
        "validation_sample_ids": splits["validation"],
        "test_labels_read": False,
        "cv_calibration_uses_heldout_fold_labels": True,
        "cv_metrics_strictly_out_of_fold": False,
        "validation_calibration_uses_validation_labels": True,
        "validation_metrics_strictly_heldout": False,
        "selection_metric_order": [
            "cv_macro_prompt_cosine",
            "validation_macro_prompt_cosine",
            "cv_top8_energy_recall",
            "candidate_order",
        ],
        "software": {
            "numpy": _package_version("numpy"),
            "scikit_learn": _package_version("scikit-learn"),
            "joblib": _package_version("joblib"),
            "source_sha256": _sha256_file(Path(__file__)),
        },
    }
    _atomic_json(output / "ranker_configs.json", identity | {"candidates": configs})
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

    results: list[dict[str, Any]] = []
    for config in tqdm(configs, desc="pairwise ranker screen", unit="candidate"):
        train_x = train_views[str(config["preprocessing"])]
        validation_x = validation_views[str(config["preprocessing"])]
        oof = np.empty_like(train_y, dtype=np.float32)
        fold_scales: list[float] = []
        fold_results: list[dict[str, Any]] = []
        for fold, heldout in enumerate(offsets):
            fit_mask = np.ones(192, dtype=bool)
            fit_mask[heldout] = False
            seed = _estimator_seed(estimator_seed, str(config["id"]), fold)
            model = _fit_ranker(config, train_x[fit_mask], train_y[fit_mask], random_seed=seed)
            raw = _predict_ranker(model, train_x[heldout])
            prediction, scale = _calibrate_and_clip(raw, train_y[heldout], config)
            oof[heldout] = prediction
            fold_scales.append(scale)
            fold_results.append(
                {
                    "fold": fold,
                    "fit_sample_ids": np.asarray(splits["train"])[fit_mask],
                    "heldout_sample_ids": np.asarray(splits["train"])[heldout],
                    "estimator_seed": seed,
                    "calibration_scale": scale,
                    "pairwise_examples": model["n_pairwise_examples"],
                    "metrics": pair_level_metrics(train_y[heldout], prediction),
                }
            )
        final_seed = _estimator_seed(estimator_seed, str(config["id"]), 6)
        final_model = _fit_ranker(config, train_x, train_y, random_seed=final_seed)
        validation_raw = _predict_ranker(final_model, validation_x)
        validation_prediction, validation_scale = _calibrate_and_clip(
            validation_raw, validation_y, config
        )
        result = {
            "config": dict(config),
            "model": final_model,
            "cv_metrics": pair_level_metrics(train_y, oof),
            "cv_calibration_scales": fold_scales,
            "cv_folds": fold_results,
            "validation_metrics": pair_level_metrics(validation_y, validation_prediction),
            "validation_calibration_scale": validation_scale,
            "final_estimator_seed": final_seed,
        }
        model_path = output / "models" / f"{config['id']}.joblib"
        prediction_path = output / "predictions" / f"{config['id']}.npz"
        _atomic_joblib(
            model_path,
            {
                "schema_version": SCHEMA_VERSION,
                "config": dict(config),
                "feature_manifest_sha256": identity["feature_manifest_sha256"],
                "split_sha256": identity["split_sha256"],
                "fit_sample_ids": np.asarray(splits["train"], dtype=np.int32),
                "validation_calibration_scale": validation_scale,
                "model": final_model,
            },
        )
        _atomic_npz(
            prediction_path,
            train_sample_ids=np.asarray(splits["train"], dtype=np.int32),
            validation_sample_ids=np.asarray(splits["validation"], dtype=np.int32),
            cv_fold_index=fold_index,
            cv_oof_predictions=oof,
            validation_raw_predictions=validation_raw.astype(np.float32),
            validation_predictions=validation_prediction,
            cv_calibration_scales=np.asarray(fold_scales, dtype=np.float64),
            validation_calibration_scale=np.asarray(validation_scale),
        )
        result["model_artifact"] = model_path
        result["prediction_artifact"] = prediction_path
        results.append(result)

    ranked = sorted(results, key=_rank_key, reverse=True)
    for rank, result in enumerate(ranked, 1):
        result["cv_rank"] = rank
    payload = identity | {
        "completed_at_unix": time.time(),
        "candidate_count": len(results),
        "winner_id": ranked[0]["config"]["id"],
        "winner_config": ranked[0]["config"],
        "results": results,
    }
    _atomic_json(result_path, payload)
    _write_csv(output / "ranker_results.csv", results)
    _atomic_json(
        output / "selected_config.json",
        identity
        | {
            "winner_id": ranked[0]["config"]["id"],
            "winner_config": ranked[0]["config"],
            "winner_cv_metrics": ranked[0]["cv_metrics"],
            "winner_validation_metrics": ranked[0]["validation_metrics"],
            "validation_calibration_scale": ranked[0]["validation_calibration_scale"],
            "model_artifact": ranked[0]["model_artifact"],
            "prediction_artifact": ranked[0]["prediction_artifact"],
        },
    )
    return payload


def self_test() -> None:
    rng = np.random.default_rng(3)
    features = rng.normal(size=(3, 4, 5)).astype(np.float32)
    targets = np.asarray(
        [[0.0, 1.0, -1.0, 0.0], [0.2, 0.3, 0.2, 0.7], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    x_pair, y_pair = _pairwise_training_data(features, targets)
    assert x_pair.shape[0] == y_pair.size and x_pair.shape[1] == 5
    assert set(y_pair.tolist()) == {0, 1}
    model = _fit_ranker(_candidate_configs()[0], features, targets, random_seed=1)
    prediction = _predict_ranker(model, features)
    assert prediction.shape == targets.shape
    print("offline_oracle_ranker self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-directory",
        default="outputs/countdown_oracle_q35_2b_D256_P32_seed0",
    )
    parser.add_argument("--search-directory", default=None)
    parser.add_argument("--output-directory", default=None)
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
