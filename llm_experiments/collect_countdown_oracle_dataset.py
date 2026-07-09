"""Collect a fixed, fully labelled Countdown perturbation dataset.

Each logical example is one EGGROLL antithetic direction at one model layer.
The input is the same center-normalized prompt-hidden difference used before
PR4's CountSketch, and the labels are the two raw final rollout fitnesses plus
their difference.  Collection is streamed to resumable ``.npy`` memmaps.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

import numpy as np


MODEL_REPOSITORY = "Qwen/Qwen3.5-2B"
LABEL_COLUMNS = ("reward_positive", "reward_negative", "reward_difference")


@dataclass(frozen=True)
class Args:
    output_directory: str = "outputs/countdown_oracle_q35_2b_D256_P32_seed0"
    dataset_size: int = 256
    directions_per_prompt: int = 32
    generation_length: int = 1024
    seed: int = 0
    sigma: float = 1e-3
    center_rms_floor: float = 1e-4
    val_holdout_size: int = 256
    dtype: Optional[str] = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
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


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def zero_padded_prompt_lengths(prompts: np.ndarray) -> np.ndarray:
    prompts = np.asarray(prompts)
    if prompts.ndim != 2:
        raise ValueError("prompts must have shape [samples, sequence]")
    nonzero = prompts != 0
    has_token = np.any(nonzero, axis=1)
    last_from_end = np.argmax(nonzero[:, ::-1], axis=1)
    return np.where(has_token, prompts.shape[1] - last_from_end, 0).astype(np.int32)


ROW_INDEX_DTYPE = np.dtype(
    [
        ("sample_id", "<i4"),
        ("local_pair_id", "<i2"),
        ("global_pair_id", "<i4"),
        ("positive_member_id", "<i4"),
        ("negative_member_id", "<i4"),
        ("layer_id", "<i2"),
    ]
)


def build_row_index(num_samples: int, pairs_per_sample: int, num_layers: int) -> np.ndarray:
    if min(num_samples, pairs_per_sample, num_layers) < 1:
        raise ValueError("row-index dimensions must be positive")
    row_count = num_samples * pairs_per_sample * num_layers
    sample_ids = np.repeat(
        np.arange(num_samples, dtype=np.int32), pairs_per_sample * num_layers
    )
    local_pair_ids = np.tile(
        np.repeat(np.arange(pairs_per_sample, dtype=np.int16), num_layers),
        num_samples,
    )
    global_pair_ids = sample_ids * pairs_per_sample + local_pair_ids.astype(np.int32)
    rows = np.empty(row_count, dtype=ROW_INDEX_DTYPE)
    rows["sample_id"] = sample_ids
    rows["local_pair_id"] = local_pair_ids
    rows["global_pair_id"] = global_pair_ids
    rows["positive_member_id"] = 2 * global_pair_ids
    rows["negative_member_id"] = 2 * global_pair_ids + 1
    rows["layer_id"] = np.tile(
        np.arange(num_layers, dtype=np.int16), num_samples * pairs_per_sample
    )
    return rows


def expand_row_labels(pair_rewards: np.ndarray, num_layers: int) -> np.ndarray:
    pair_rewards = np.asarray(pair_rewards, dtype=np.float32)
    if pair_rewards.ndim != 2 or pair_rewards.shape[1] != 2:
        raise ValueError("pair_rewards must have shape [pairs, 2]")
    if num_layers < 1:
        raise ValueError("num_layers must be positive")
    pair_labels = np.column_stack(
        (
            pair_rewards[:, 0],
            pair_rewards[:, 1],
            pair_rewards[:, 0] - pair_rewards[:, 1],
        )
    ).astype(np.float32)
    return np.repeat(pair_labels, num_layers, axis=0)


def shard_member_ids(member_ids: np.ndarray, num_devices: int) -> np.ndarray:
    """Split ordered complete antithetic pairs evenly across local devices."""

    member_ids = np.asarray(member_ids, dtype=np.int32)
    if member_ids.ndim != 1 or member_ids.size == 0 or member_ids.size % 2:
        raise ValueError("member_ids must be a nonempty vector of complete pairs")
    if num_devices < 1 or (member_ids.size // 2) % num_devices:
        raise ValueError("antithetic pairs must divide evenly across local GPUs")
    return member_ids.reshape(num_devices, member_ids.size // num_devices)


def _open_memmap(
    path: Path,
    *,
    shape: tuple[int, ...],
    dtype: Any,
    fill: Optional[float] = None,
) -> np.memmap:
    expected_dtype = np.dtype(dtype)
    if path.exists():
        array = np.load(path, mmap_mode="r+")
        if array.shape != shape or array.dtype != expected_dtype:
            raise ValueError(
                f"Existing {path} has shape/dtype {array.shape}/{array.dtype}; "
                f"expected {shape}/{expected_dtype}"
            )
        return array
    array = np.lib.format.open_memmap(path, mode="w+", dtype=expected_dtype, shape=shape)
    if fill is not None:
        array[...] = fill
        array.flush()
    return array


def _source_hashes(repo_root: Path) -> dict[str, str]:
    paths = (
        "llm_experiments/collect_countdown_oracle_dataset.py",
        "llm_experiments/utils.py",
        "src/hyperscalees/models/base_model.py",
        "src/hyperscalees/models/common.py",
        "src/hyperscalees/models/llm/auto.py",
        "src/hyperscalees/models/llm/llm.py",
        "src/hyperscalees/noiser/eggroll.py",
        "src/hyperscalees/models/llm/qrwkv6.py",
        "src/hyperscalees/models/llm/tokenizer.py",
        "src/hyperscalees/environments/llm_bandits.py",
    )
    return {
        relative: _sha256_file(repo_root / relative)
        for relative in paths
        if (repo_root / relative).is_file()
    }


def _package_versions() -> dict[str, Optional[str]]:
    names = (
        "jax",
        "jaxlib",
        "numpy",
        "transformers",
        "tokenizers",
        "huggingface_hub",
        "datasets",
        "optax",
        "tyro",
        "tqdm",
    )
    versions: dict[str, Optional[str]] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _ensure_run_config(output_directory: Path, config: dict[str, Any]) -> str:
    config_text = json.dumps(config, indent=2, sort_keys=True) + "\n"
    config_path = output_directory / "run_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError(
                f"Refusing to resume {output_directory}: run_config.json does not "
                "match the requested/model/dataset configuration. Use a new output directory."
            )
    else:
        _atomic_write_text(config_path, config_text)
    return _sha256_bytes(_canonical_json(config).encode("utf-8"))


def _dataset_files_size(output_directory: Path) -> int:
    return sum(path.stat().st_size for path in output_directory.glob("*.npy"))


def _already_complete(output_directory: Path, args: Args) -> bool:
    metadata_path = output_directory / "metadata.json"
    config_path = output_directory / "run_config.json"
    completed_path = output_directory / "completed_samples.npy"
    if completed_path.is_file() and not config_path.is_file():
        completed_without_config = np.load(completed_path, mmap_mode="r")
        if np.any(completed_without_config == 1):
            raise ValueError(
                "Cannot resume committed samples without their immutable run_config.json"
            )
    if not (metadata_path.is_file() and config_path.is_file() and completed_path.is_file()):
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        return False
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_sha256 = _sha256_bytes(_canonical_json(config).encode("utf-8"))
    if metadata.get("run_config_sha256") != config_sha256:
        raise ValueError("metadata.json does not match the immutable run_config.json")
    requested = {
        "dataset_size": args.dataset_size,
        "dataset_seed": args.seed,
        "validation_holdout_size": args.val_holdout_size,
        "directions_per_prompt": args.directions_per_prompt,
        "sigma": args.sigma,
        "generation_length_including_prompt": args.generation_length,
        "center_rms_floor": args.center_rms_floor,
        "dtype": args.dtype or "bfloat16",
    }
    actual = {
        "dataset_size": config["dataset"]["dataset_size"],
        "dataset_seed": config["dataset"]["dataset_seed"],
        "validation_holdout_size": config["dataset"]["validation_holdout_size"],
        "directions_per_prompt": config["perturbations"]["directions_per_prompt"],
        "sigma": config["perturbations"]["sigma"],
        "generation_length_including_prompt": config["rollout"][
            "generation_length_including_prompt"
        ],
        "center_rms_floor": config["predictor_input"]["center_rms_floor"],
        "dtype": config["model"]["dtype"],
    }
    if actual != requested:
        raise ValueError(
            f"{output_directory} already contains a completed dataset with different settings"
        )
    completed = np.load(completed_path, mmap_mode="r")
    if completed.shape != (args.dataset_size,) or not np.all(completed == 1):
        raise ValueError("Completed metadata disagrees with completed_samples.npy")
    num_layers = int(config["model"]["num_layers"])
    hidden_size = int(config["model"]["hidden_size"])
    row_count = args.dataset_size * args.directions_per_prompt * num_layers
    expected_arrays = {
        "predictor_inputs.npy": ((row_count, hidden_size), np.dtype(np.float32)),
        "center_rms.npy": ((row_count,), np.dtype(np.float32)),
        "row_labels.npy": ((row_count, len(LABEL_COLUMNS)), np.dtype(np.float32)),
        "row_index.npy": ((row_count,), ROW_INDEX_DTYPE),
        "pair_rewards.npy": (
            (args.dataset_size, args.directions_per_prompt, 2),
            np.dtype(np.float32),
        ),
        "reward_differences.npy": (
            (args.dataset_size, args.directions_per_prompt),
            np.dtype(np.float32),
        ),
        "output_tokens.npy": (
            (
                args.dataset_size,
                args.directions_per_prompt,
                2,
                args.generation_length,
            ),
            np.dtype(np.int32),
        ),
        "prompt_tokens.npy": (
            (args.dataset_size, args.generation_length),
            np.dtype(np.int32),
        ),
        "prompt_lengths.npy": ((args.dataset_size,), np.dtype(np.int32)),
        "completed_samples.npy": ((args.dataset_size,), np.dtype(np.uint8)),
    }
    required_files = tuple(expected_arrays) + ("manifest.json", "samples.jsonl")
    missing = [name for name in required_files if not (output_directory / name).is_file()]
    if missing:
        raise ValueError(f"Completed dataset is missing files: {missing}")
    for name, (shape, dtype) in expected_arrays.items():
        array = np.load(output_directory / name, mmap_mode="r")
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(
                f"Completed {name} has shape/dtype {array.shape}/{array.dtype}; "
                f"expected {shape}/{dtype}"
            )
    return True


def collect(args: Args) -> None:
    if args.dataset_size < 1 or args.directions_per_prompt < 1:
        raise ValueError("dataset_size and directions_per_prompt must be positive")
    if args.generation_length < 2:
        raise ValueError("generation_length must be at least two")
    if args.sigma <= 0.0 or args.center_rms_floor <= 0.0:
        raise ValueError("sigma and center_rms_floor must be positive")
    if args.dtype not in (None, "bfloat16", "float32"):
        raise ValueError("dtype must be bfloat16, float32, or omitted")

    output_directory = Path(args.output_directory).expanduser().resolve()
    if _already_complete(output_directory, args):
        metadata = json.loads(
            (output_directory / "metadata.json").read_text(encoding="utf-8")
        )
        print(f"Dataset is already complete: {output_directory}")
        print(f"Full rollouts: {metadata['completed_rollouts']:,}")
        print(f"Logical rows: {metadata['logical_rows']:,}")
        return

    import jax
    import jax.numpy as jnp
    from huggingface_hub.constants import HF_HOME
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from tqdm.auto import tqdm

    jax.config.update(
        "jax_compilation_cache_dir", os.path.join(HF_HOME, "hyperscaleescomp")
    )
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    from hyperscalees.environments.llm_bandits import CountdownChatTrain
    from hyperscalees.models.common import simple_es_tree_key
    from hyperscalees.models.llm.auto import get_model
    from hyperscalees.noiser.eggroll import EggRoll
    from llm_experiments.utils import build_generate_batch_with_preview

    gpu_devices = [device for device in jax.local_devices() if device.platform == "gpu"]
    if not gpu_devices:
        raise RuntimeError(f"Expected at least one visible GPU, found {jax.local_devices()}")
    num_devices = len(gpu_devices)
    if args.directions_per_prompt % num_devices:
        raise ValueError(
            "directions_per_prompt must divide evenly across visible GPUs"
        )
    print(f"Using {num_devices} JAX GPU(s): {gpu_devices}")

    output_directory.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]

    print(f"Loading {MODEL_REPOSITORY}...")
    model, full_params, tokenizer = get_model(
        "q35_2B", rwkv_type="Qwen35RWKV", verbose=True, dtype=args.dtype
    )
    config, params, scan_map, _ = full_params
    if "layer_types" not in config:
        raise ValueError("The collector requires the instrumented Qwen3.5 model")
    num_layers = len(config["layer_types"])
    hidden_size = int(config["hidden_size"])

    task = CountdownChatTrain(
        tokenizer,
        tokenizer,
        args.generation_length,
        dataset_size=args.dataset_size,
        seed=args.seed,
        val_holdout_size=args.val_holdout_size,
    )
    if len(task) != args.dataset_size:
        raise ValueError(f"Requested {args.dataset_size} samples but loaded {len(task)}")

    sample_ids = np.arange(args.dataset_size, dtype=np.int32)
    prompts = np.asarray(task.get_input(jnp.asarray(sample_ids)), dtype=np.int32)
    prompt_lengths = zero_padded_prompt_lengths(prompts)
    if np.any(prompt_lengths <= 0) or np.any(prompt_lengths >= args.generation_length):
        raise ValueError("Every prompt must be nonempty and leave room for generation")
    for prompt, length in zip(prompts, prompt_lengths):
        if np.any(prompt[: int(length)] == 0) or np.any(prompt[int(length) :] != 0):
            raise ValueError("Token 0 must occur only in prompt padding")

    sample_records = []
    for sample_id in range(args.dataset_size):
        sample_records.append(
            {
                "sample_id": sample_id,
                "prompt_length": int(prompt_lengths[sample_id]),
                "source_example": _json_safe(dict(task.dataset[sample_id])),
            }
        )
    samples_text = "".join(
        json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
        for record in sample_records
    )
    samples_path = output_directory / "samples.jsonl"
    if samples_path.exists() and samples_path.read_text(encoding="utf-8") != samples_text:
        raise ValueError("Existing samples.jsonl does not match the selected dataset")
    if not samples_path.exists():
        _atomic_write_text(samples_path, samples_text)

    rollout_config = dict(config)
    rollout_config["attn_cache_len"] = args.generation_length
    model_config_json = _canonical_json(_json_safe(config))
    model_dtype = str(params["embed_tokens"]["weight"].dtype)
    model_cache_path = Path(HF_HOME) / "hyperscalees_cache" / (
        f"q35_2B_{model_dtype}.model"
    )
    model_cache_metadata: dict[str, Any] = {"filename": model_cache_path.name}
    if model_cache_path.is_file():
        print(f"Hashing cached model for reproducibility: {model_cache_path}")
        model_cache_metadata |= {
            "bytes": model_cache_path.stat().st_size,
            "sha256": _sha256_file(model_cache_path),
        }
    git_metadata = _git_metadata(repo_root)
    backend_tokenizer = getattr(tokenizer.tok, "backend_tokenizer", None)
    tokenizer_backend_json = (
        backend_tokenizer.to_str() if backend_tokenizer is not None else None
    )
    run_config = {
        "schema_version": 1,
        "model": {
            "choice": "q35_2B",
            "repository": MODEL_REPOSITORY,
            "revision": config.get("_commit_hash"),
            "dtype": model_dtype,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "config_sha256": _sha256_bytes(model_config_json.encode("utf-8")),
            "local_cache": model_cache_metadata,
        },
        "tokenizer": {
            "name_or_path": getattr(tokenizer.tok, "name_or_path", MODEL_REPOSITORY),
            "revision": getattr(tokenizer.tok, "init_kwargs", {}).get("_commit_hash"),
            "vocab_size": int(len(tokenizer.tok)),
            "backend_sha256": (
                _sha256_bytes(tokenizer_backend_json.encode("utf-8"))
                if tokenizer_backend_json is not None
                else None
            ),
            "special_tokens_map": _json_safe(tokenizer.tok.special_tokens_map),
            "prompt_tokens_sha256": _sha256_bytes(
                np.ascontiguousarray(prompts).tobytes()
            ),
        },
        "dataset": {
            "task": "countdown_chat",
            "split": "disjoint_train_after_validation_holdout",
            "dataset_size": args.dataset_size,
            "dataset_seed": args.seed,
            "validation_holdout_size": args.val_holdout_size,
            "samples_sha256": _sha256_bytes(samples_text.encode("utf-8")),
        },
        "perturbations": {
            "directions_per_prompt": args.directions_per_prompt,
            "rank": 1,
            "sigma": args.sigma,
            "freeze_nonlora": True,
            "noise_reuse": 1,
            "epoch": 0,
            "master_seed": args.seed,
            "global_pair_id": "sample_id * directions_per_prompt + local_pair_id",
            "member_ids": "[2 * global_pair_id, 2 * global_pair_id + 1]",
        },
        "rollout": {
            "temperature": 0.0,
            "generation_length_including_prompt": args.generation_length,
            "batch_members": 2 * args.directions_per_prompt,
            "eos_early_stopping": False,
            "fitness": "raw CountdownChatTrain.get_batch_fitness output",
        },
        "predictor_input": {
            "capture": "post-block residual at final forced prompt token",
            "layers": list(range(num_layers)),
            "normalization": "(h_plus-h_minus)/(2*sigma*max(RMS((h_plus+h_minus)/2),floor))",
            "center_rms_floor": args.center_rms_floor,
            "countsketch_applied": False,
            "capture_mode": "fused_with_labelled_full_rollout",
            "reference_pr4_projection": {
                "type": "CountSketch",
                "layers": [
                    min(num_layers - 1, int(np.floor(0.75 * num_layers))),
                    num_layers - 1,
                ],
                "buckets_per_layer": 128,
                "seed": 0,
            },
        },
        "row_order": "sample_id, local_pair_id, layer_id",
        "label_columns": list(LABEL_COLUMNS),
        "software": {
            "git_commit": git_metadata["commit"],
            "source_sha256": _source_hashes(repo_root),
            "packages": _package_versions(),
        },
    }
    run_config_sha256 = _ensure_run_config(output_directory, run_config)

    row_count = args.dataset_size * args.directions_per_prompt * num_layers
    expected_new_bytes = (
        row_count * hidden_size * np.dtype(np.float32).itemsize
        + row_count * np.dtype(np.float32).itemsize
        + row_count * len(LABEL_COLUMNS) * np.dtype(np.float32).itemsize
        + args.dataset_size
        * args.directions_per_prompt
        * 2
        * args.generation_length
        * np.dtype(np.int32).itemsize
    )
    core_payload_paths = (
        output_directory / "predictor_inputs.npy",
        output_directory / "center_rms.npy",
        output_directory / "row_labels.npy",
        output_directory / "pair_rewards.npy",
        output_directory / "reward_differences.npy",
        output_directory / "output_tokens.npy",
    )
    completed_path = output_directory / "completed_samples.npy"
    if completed_path.exists():
        previous_completed = np.load(completed_path, mmap_mode="r")
        if np.any(previous_completed == 1):
            missing = [path.name for path in core_payload_paths if not path.exists()]
            if missing:
                raise ValueError(
                    "Cannot safely resume: completed_samples.npy contains committed "
                    f"samples but payload files are missing: {missing}"
                )
    missing_payload_bytes = (
        0 if all(path.exists() for path in core_payload_paths) else expected_new_bytes
    )
    free_bytes = shutil.disk_usage(output_directory).free
    if free_bytes < missing_payload_bytes + 512 * 1024**2:
        raise OSError(
            f"Need approximately {missing_payload_bytes / 1024**3:.2f} GiB plus headroom; "
            f"only {free_bytes / 1024**3:.2f} GiB is free"
        )

    predictor_inputs = _open_memmap(
        output_directory / "predictor_inputs.npy",
        shape=(row_count, hidden_size),
        dtype=np.float32,
    )
    center_rms_values = _open_memmap(
        output_directory / "center_rms.npy",
        shape=(row_count,),
        dtype=np.float32,
        fill=np.nan,
    )
    row_labels = _open_memmap(
        output_directory / "row_labels.npy",
        shape=(row_count, len(LABEL_COLUMNS)),
        dtype=np.float32,
        fill=np.nan,
    )
    pair_rewards = _open_memmap(
        output_directory / "pair_rewards.npy",
        shape=(args.dataset_size, args.directions_per_prompt, 2),
        dtype=np.float32,
        fill=np.nan,
    )
    reward_differences = _open_memmap(
        output_directory / "reward_differences.npy",
        shape=(args.dataset_size, args.directions_per_prompt),
        dtype=np.float32,
        fill=np.nan,
    )
    output_tokens = _open_memmap(
        output_directory / "output_tokens.npy",
        shape=(
            args.dataset_size,
            args.directions_per_prompt,
            2,
            args.generation_length,
        ),
        dtype=np.int32,
    )
    saved_prompts = _open_memmap(
        output_directory / "prompt_tokens.npy",
        shape=prompts.shape,
        dtype=np.int32,
    )
    saved_prompt_lengths = _open_memmap(
        output_directory / "prompt_lengths.npy",
        shape=prompt_lengths.shape,
        dtype=np.int32,
    )
    completed = _open_memmap(
        output_directory / "completed_samples.npy",
        shape=(args.dataset_size,),
        dtype=np.uint8,
        fill=0,
    )
    row_index_path = output_directory / "row_index.npy"
    if row_index_path.exists():
        row_index = np.load(row_index_path, mmap_mode="r")
        if row_index.shape != (row_count,) or row_index.dtype != ROW_INDEX_DTYPE:
            raise ValueError("Existing row_index.npy has an incompatible schema")
    else:
        row_index = np.lib.format.open_memmap(
            row_index_path, mode="w+", dtype=ROW_INDEX_DTYPE, shape=(row_count,)
        )
        row_index[:] = build_row_index(
            args.dataset_size, args.directions_per_prompt, num_layers
        )
        row_index.flush()

    if np.any(completed > 1):
        raise ValueError("completed_samples.npy contains invalid values")
    saved_prompts[:] = prompts
    saved_prompt_lengths[:] = prompt_lengths
    saved_prompts.flush()
    saved_prompt_lengths.flush()

    manifest = {
        "schema_version": 1,
        "logical_rows": row_count,
        "independent_pair_examples": args.dataset_size * args.directions_per_prompt,
        "full_rollouts": args.dataset_size * args.directions_per_prompt * 2,
        "arrays": {
            "predictor_inputs.npy": [row_count, hidden_size],
            "center_rms.npy": [row_count],
            "row_labels.npy": [row_count, len(LABEL_COLUMNS)],
            "row_index.npy": [row_count],
            "pair_rewards.npy": [args.dataset_size, args.directions_per_prompt, 2],
            "reward_differences.npy": [args.dataset_size, args.directions_per_prompt],
            "output_tokens.npy": [
                args.dataset_size,
                args.directions_per_prompt,
                2,
                args.generation_length,
            ],
            "prompt_tokens.npy": list(prompts.shape),
            "prompt_lengths.npy": list(prompt_lengths.shape),
            "completed_samples.npy": [args.dataset_size],
        },
        "label_columns": list(LABEL_COLUMNS),
        "row_index_fields": list(ROW_INDEX_DTYPE.names or ()),
        "notes": [
            "All layer rows for a pair share one pair-level rollout label.",
            "Split offline experiments by sample_id, never by individual layer row.",
            "Only rows belonging to completed_samples == 1 are valid during a partial run.",
        ],
        "run_config_sha256": run_config_sha256,
    }
    _atomic_write_json(output_directory / "manifest.json", manifest)

    print(
        f"Dataset: {args.dataset_size} prompts x {args.directions_per_prompt} pairs "
        f"x {num_layers} layers = {row_count:,} rows"
    )
    print(f"Full rollouts: {args.dataset_size * args.directions_per_prompt * 2:,}")
    print(f"Expected predictor input payload: {row_count * hidden_size * 4 / 1024**3:.2f} GiB")
    print(f"Output directory: {output_directory}")

    replica_mesh = Mesh(np.asarray(gpu_devices), ("replica",))
    replicated_sharding = NamedSharding(replica_mesh, P())
    params = jax.tree.map(
        lambda value: jax.device_put(value, replicated_sharding), params
    )
    master_key = jax.random.key(args.seed)
    base_model_key = jax.random.fold_in(master_key, 0)
    base_gen_key = jax.random.fold_in(master_key, 1)
    base_evo_keys = simple_es_tree_key(params, base_model_key, scan_map)
    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        params,
        args.sigma,
        0.0,
        group_size=2 * args.directions_per_prompt,
        freeze_nonlora=True,
        noise_reuse=1,
        rank=1,
    )

    fused_generate = build_generate_batch_with_preview(
        model,
        EggRoll,
        frozen_noiser_params,
        rollout_config,
        base_evo_keys,
        base_gen_key,
        tuple(range(num_layers)),
        temperature=0.0,
        suppress_eos_token=None,
        center_rms_floor=args.center_rms_floor,
    )
    member_count = 2 * args.directions_per_prompt
    members_per_device = member_count // num_devices
    parallel_generate = jax.pmap(
        fused_generate,
        in_axes=(None, None, None, None, 0, None),
        devices=gpu_devices,
    )
    print(
        f"Compiling {members_per_device} members/GPU across {num_devices} GPU(s)..."
    )
    compile_start = time.time()
    compiled_generate = parallel_generate.lower(
        noiser_params,
        params,
        jax.ShapeDtypeStruct((args.generation_length,), jnp.dtype("int32")),
        jax.ShapeDtypeStruct((), jnp.dtype("int32")),
        jax.ShapeDtypeStruct(
            (num_devices, members_per_device), jnp.dtype("int32")
        ),
        jnp.asarray(0, dtype=jnp.int32),
    ).compile()
    print(f"Compilation finished in {time.time() - compile_start:.1f}s")
    print(compiled_generate.memory_analysis())

    completed_mask = np.asarray(completed, dtype=bool)
    initial_completed = int(completed_mask.sum())
    session_start = time.time()
    metadata_path = output_directory / "metadata.json"
    base_metadata = {
        "schema_version": 1,
        "status": "running",
        "started_or_resumed_at": _utc_now(),
        "run_config_sha256": run_config_sha256,
        "git": git_metadata,
        "devices": [str(device) for device in gpu_devices],
        "device_count": num_devices,
        "members_per_device": members_per_device,
        "completed_samples": initial_completed,
        "total_samples": args.dataset_size,
    }
    _atomic_write_json(metadata_path, base_metadata)

    cpu_device = jax.local_devices(backend="cpu")[0]
    progress = tqdm(
        total=args.dataset_size,
        initial=initial_completed,
        desc="Countdown oracle dataset",
        unit="prompt",
        dynamic_ncols=True,
    )
    try:
        for sample_id in np.flatnonzero(~completed_mask):
            batch_start = time.time()
            first_pair = int(sample_id) * args.directions_per_prompt
            global_pair_ids = np.arange(
                first_pair, first_pair + args.directions_per_prompt, dtype=np.int32
            )
            member_ids = np.stack(
                (2 * global_pair_ids, 2 * global_pair_ids + 1), axis=-1
            ).reshape(-1)

            generated_tokens, pair_inputs, pair_center_rms = jax.block_until_ready(
                compiled_generate(
                    noiser_params,
                    params,
                    jnp.asarray(prompts[sample_id]),
                    jnp.asarray(prompt_lengths[sample_id], dtype=jnp.int32),
                    jnp.asarray(shard_member_ids(member_ids, num_devices)),
                    jnp.asarray(0, dtype=jnp.int32),
                )
            )
            tokens_np = np.asarray(
                jax.device_get(generated_tokens), dtype=np.int32
            ).reshape(member_count, args.generation_length)
            inputs_np = np.asarray(
                jax.device_get(pair_inputs), dtype=np.float32
            ).reshape(args.directions_per_prompt, num_layers, hidden_size)
            center_rms_np = np.asarray(
                jax.device_get(pair_center_rms), dtype=np.float32
            ).reshape(args.directions_per_prompt, num_layers)
            if tokens_np.shape != (member_count, args.generation_length):
                raise RuntimeError(f"Unexpected generated-token shape {tokens_np.shape}")
            expected_input_shape = (
                args.directions_per_prompt,
                num_layers,
                hidden_size,
            )
            if inputs_np.shape != expected_input_shape:
                raise RuntimeError(
                    f"Unexpected predictor input shape {inputs_np.shape}; "
                    f"expected {expected_input_shape}"
                )
            if not np.all(np.isfinite(inputs_np)):
                raise RuntimeError("Predictor inputs contain non-finite values")
            if center_rms_np.shape != (args.directions_per_prompt, num_layers):
                raise RuntimeError(f"Unexpected center-RMS shape {center_rms_np.shape}")
            if not np.all(np.isfinite(center_rms_np)):
                raise RuntimeError("Center RMS contains non-finite values")

            with jax.default_device(cpu_device):
                fitness = task.get_batch_fitness(
                    jax.device_put(
                        np.full(member_count, sample_id, dtype=np.int32), cpu_device
                    ),
                    jax.device_put(tokens_np, cpu_device),
                )
            rewards_np = np.asarray(jax.device_get(fitness), dtype=np.float32).reshape(
                args.directions_per_prompt, 2
            )
            if not np.all(np.isfinite(rewards_np)):
                raise RuntimeError("Raw rollout fitness contains non-finite values")
            differences_np = rewards_np[:, 0] - rewards_np[:, 1]
            row_start = int(sample_id) * args.directions_per_prompt * num_layers
            row_stop = row_start + args.directions_per_prompt * num_layers

            predictor_inputs[row_start:row_stop] = inputs_np.reshape(
                args.directions_per_prompt * num_layers, hidden_size
            )
            center_rms_values[row_start:row_stop] = center_rms_np.reshape(-1)
            row_labels[row_start:row_stop] = expand_row_labels(rewards_np, num_layers)
            pair_rewards[sample_id] = rewards_np
            reward_differences[sample_id] = differences_np
            output_tokens[sample_id] = tokens_np.reshape(
                args.directions_per_prompt, 2, args.generation_length
            )
            predictor_inputs.flush()
            center_rms_values.flush()
            row_labels.flush()
            pair_rewards.flush()
            reward_differences.flush()
            output_tokens.flush()
            completed[sample_id] = 1
            completed.flush()
            completed_mask[sample_id] = True

            progress.update(1)
            observed_rewards = np.asarray(pair_rewards[completed_mask])
            progress.set_postfix(
                rollouts=int(completed_mask.sum()) * member_count,
                mean_reward=f"{float(np.nanmean(observed_rewards)):.4f}",
                batch_s=f"{time.time() - batch_start:.1f}",
            )
            _atomic_write_json(
                metadata_path,
                base_metadata
                | {
                    "status": "running",
                    "updated_at": _utc_now(),
                    "completed_samples": int(completed_mask.sum()),
                    "completed_rollouts": int(completed_mask.sum()) * member_count,
                },
            )
    except BaseException as error:
        _atomic_write_json(
            metadata_path,
            base_metadata
            | {
                "status": "interrupted",
                "updated_at": _utc_now(),
                "completed_samples": int(completed_mask.sum()),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
    finally:
        progress.close()

    final_rewards = np.asarray(pair_rewards, dtype=np.float32)
    final_metadata = base_metadata | {
        "status": "complete",
        "completed_at": _utc_now(),
        "completed_samples": args.dataset_size,
        "completed_pairs": args.dataset_size * args.directions_per_prompt,
        "completed_rollouts": args.dataset_size * args.directions_per_prompt * 2,
        "logical_rows": row_count,
        "mean_member_reward": float(np.mean(final_rewards)),
        "nonzero_member_reward_fraction": float(np.mean(final_rewards != 0.0)),
        "session_seconds": time.time() - session_start,
        "npy_payload_bytes": _dataset_files_size(output_directory),
    }
    _atomic_write_json(metadata_path, final_metadata)
    print(f"Complete: {output_directory}")
    print(f"Mean raw rollout fitness: {final_metadata['mean_member_reward']:.6f}")
    print(f"Saved .npy payload: {final_metadata['npy_payload_bytes'] / 1024**3:.2f} GiB")


def main() -> None:
    import tyro

    collect(tyro.cli(Args))


if __name__ == "__main__":
    main()
