"""Replay PR5 perturbations and collect richer prompt-only PACT states.

This collector deliberately performs no decoding, scorer call, or new rollout.
It reuses the immutable prompts, labels, perturbation IDs, and seed recorded by
``collect_countdown_oracle_dataset.py``.  For every prompt it saves the clean
last-real-token residual, both antithetic members' residuals at every layer,
and a compact policy panel selected by the clean model.

The large arrays are resumable ``.npy`` memmaps.  A sample is committed only
after its newly replayed antithetic states reconstruct the original PR5
pre-CountSketch predictor input within the configured tolerance.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

import numpy as np


@dataclass(frozen=True)
class Args:
    source_directory: str = "outputs/countdown_oracle_q35_2b_D256_P32_seed0"
    output_directory: Optional[str] = None
    panel_size: int = 96
    min_replay_cosine: float = 0.995
    max_replay_relative_rmse: float = 0.10
    min_center_cosine: float = 0.999
    max_center_relative_rmse: float = 0.01
    max_new_samples: Optional[int] = None
    self_test: bool = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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
                f"Existing {path.name} has {array.shape}/{array.dtype}; "
                f"expected {shape}/{expected_dtype}"
            )
        return array
    array = np.lib.format.open_memmap(
        path, mode="w+", dtype=expected_dtype, shape=shape
    )
    if fill is not None:
        array[...] = fill
        array.flush()
    return array


def global_member_ids(
    sample_id: int, directions_per_prompt: int
) -> np.ndarray:
    """Return PR5's adjacent even/odd global member IDs for one prompt."""

    if sample_id < 0 or directions_per_prompt < 1:
        raise ValueError("sample_id must be nonnegative and directions positive")
    first_pair = int(sample_id) * int(directions_per_prompt)
    pair_ids = np.arange(
        first_pair, first_pair + directions_per_prompt, dtype=np.int32
    )
    return np.stack((2 * pair_ids, 2 * pair_ids + 1), axis=-1).reshape(-1)


def shard_complete_pairs(member_ids: np.ndarray, num_devices: int) -> np.ndarray:
    """Place contiguous complete antithetic pairs on each local device."""

    member_ids = np.asarray(member_ids, dtype=np.int32)
    if member_ids.ndim != 1 or member_ids.size == 0 or member_ids.size % 2:
        raise ValueError("member_ids must be a nonempty vector of complete pairs")
    if num_devices < 1 or (member_ids.size // 2) % num_devices:
        raise ValueError("pairs must divide evenly across visible devices")
    sharded = member_ids.reshape(num_devices, member_ids.size // num_devices)
    if np.any(sharded[:, 0] % 2):
        raise ValueError("a device shard starts at the negative member of a pair")
    return sharded


def antithetic_replay_statistics(
    member_hidden: np.ndarray,
    sigma: float,
    center_rms_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct PR5 normalized directions from raw member residuals.

    ``member_hidden`` is ``[pairs, 2, layers, hidden]`` in positive/negative
    order.  The return values have shapes ``[pairs, layers, hidden]`` and
    ``[pairs, layers]``.
    """

    values = np.asarray(member_hidden, dtype=np.float32)
    if values.ndim != 4 or values.shape[1] != 2:
        raise ValueError("member_hidden must have shape [pairs, 2, layers, hidden]")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    if not np.isfinite(center_rms_floor) or center_rms_floor <= 0.0:
        raise ValueError("center_rms_floor must be finite and positive")
    positive, negative = values[:, 0], values[:, 1]
    center = 0.5 * (positive + negative)
    center_rms = np.sqrt(np.mean(np.square(center), axis=-1, dtype=np.float32))
    normalized = (positive - negative) / (
        2.0 * np.float32(sigma) * np.maximum(center_rms[..., None], center_rms_floor)
    )
    return normalized.astype(np.float32), center_rms.astype(np.float32)


def replay_error_summary(
    actual: np.ndarray, expected: np.ndarray
) -> dict[str, float]:
    """Return finite elementwise error diagnostics without fitting anything."""

    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {actual.shape} != {expected.shape}")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        raise ValueError("replay comparison contains non-finite values")
    difference = actual - expected
    denominator = np.maximum(np.abs(expected), np.float32(1e-6))
    actual_flat = actual.astype(np.float64, copy=False).reshape(-1)
    expected_flat = expected.astype(np.float64, copy=False).reshape(-1)
    cosine_denominator = float(
        np.linalg.norm(actual_flat) * np.linalg.norm(expected_flat)
    )
    expected_rms = float(np.sqrt(np.mean(np.square(expected_flat))))
    rmse = float(np.sqrt(np.mean(np.square(difference), dtype=np.float64)))
    return {
        "max_abs": float(np.max(np.abs(difference), initial=0.0)),
        "rmse": rmse,
        "relative_rmse": rmse / max(expected_rms, 1e-12),
        "cosine": (
            float(actual_flat @ expected_flat / cosine_denominator)
            if cosine_denominator > 0.0
            else 1.0
        ),
        "max_relative": float(
            np.max(np.abs(difference) / denominator, initial=0.0)
        ),
    }


def qwen35_rms_norm_numpy(
    hidden: np.ndarray, weight: np.ndarray, eps: float
) -> np.ndarray:
    """Reference Qwen3.5 final RMSNorm, including its ``1 + weight`` rule."""

    hidden = np.asarray(hidden, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    variance = np.mean(np.square(hidden), axis=-1, keepdims=True, dtype=np.float32)
    return hidden / np.sqrt(variance + np.float32(eps)) * (1.0 + weight)


def _self_test() -> None:
    ids = global_member_ids(3, 2)
    np.testing.assert_array_equal(ids, [12, 13, 14, 15])
    np.testing.assert_array_equal(shard_complete_pairs(ids, 2), [[12, 13], [14, 15]])

    sigma = 0.1
    center = np.asarray([[[2.0, 4.0]], [[3.0, 5.0]]], dtype=np.float32)
    tangent = np.asarray([[[1.0, -2.0]], [[-1.5, 0.5]]], dtype=np.float32)
    positive = center + sigma * tangent
    negative = center - sigma * tangent
    replay, center_rms = antithetic_replay_statistics(
        np.stack((positive, negative), axis=1), sigma, 1e-6
    )
    expected_rms = np.sqrt(np.mean(np.square(center), axis=-1))
    np.testing.assert_allclose(center_rms, expected_rms, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        replay, tangent / expected_rms[..., None], rtol=2e-6, atol=2e-6
    )
    summary = replay_error_summary(replay, replay.copy())
    assert summary["max_abs"] == 0.0 and summary["rmse"] == 0.0

    normalized = qwen35_rms_norm_numpy(
        np.asarray([[3.0, 4.0]], dtype=np.float32),
        np.asarray([0.0, 1.0], dtype=np.float32),
        0.0,
    )
    np.testing.assert_allclose(
        normalized, [[3.0 / np.sqrt(12.5), 8.0 / np.sqrt(12.5)]], atol=1e-6
    )
    print("collect_countdown_pact_prefills self-test passed")


def build_pact_prefill_batch(
    model,
    noiser,
    frozen_noiser_params: dict[str, Any],
    config: dict[str, Any],
    base_evo_keys: Any,
    *,
    preview_layers: tuple[int, ...],
    prompt_width: int,
    attention_cache_width: int,
    panel_size: int,
):
    """Build one replica's clean plus memberwise hidden-only prefill kernel."""

    import jax
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp

    preview_layers = tuple(int(layer) for layer in preview_layers)
    num_layers = len(config["layer_types"])
    if preview_layers != tuple(range(num_layers)):
        raise ValueError("PACT raw collection expects every model layer in order")
    if prompt_width < 1 or attention_cache_width < prompt_width or panel_size < 1:
        raise ValueError("prompt/cache widths and panel_size are invalid")
    if panel_size > int(config["vocab_size"]):
        raise ValueError("panel_size exceeds model vocabulary")

    preview_config = {
        **config,
        # Cache geometry affects BF16 reduction order in the six full-attention
        # blocks.  Match PR5's rollout cache exactly even though this kernel
        # stops after the real prompt.
        "attn_cache_len": int(attention_cache_width),
        "preview_layers": preview_layers,
    }
    eps = jnp.asarray(config["rms_norm_eps"], dtype=jnp.float32)

    def run_prefill(noiser_params, params, prompt, prompt_length, iterinfo):
        initial_state = model.default_state(params, preview_config)
        _, final_state = model.forward(
            noiser,
            frozen_noiser_params,
            noiser_params,
            preview_config,
            params,
            base_evo_keys,
            iterinfo,
            prompt,
            initial_state,
            length=prompt_length,
            return_hidden=True,
        )
        return final_state["preview_hidden"]

    def final_norm(params, hidden):
        hidden = hidden.astype(jnp.float32)
        variance = jnp.mean(jnp.square(hidden), axis=-1, keepdims=True)
        weight = params["norm"]["weight"].astype(jnp.float32)
        return hidden * jax.lax.rsqrt(variance + eps) * (1.0 + weight)

    def prefill_batch(noiser_params, params, prompt, prompt_length, member_ids, epoch):
        # ``None`` is a static Python value.  Every EggRoll matrix operation
        # consequently takes its exact clean branch, rather than approximating
        # the clean state with the antithetic midpoint.
        clean_hidden = run_prefill(
            noiser_params, params, prompt, prompt_length, None
        )

        def one_member(member_id):
            return run_prefill(
                noiser_params,
                params,
                prompt,
                prompt_length,
                (epoch, member_id),
            )

        member_hidden = jax.vmap(one_member)(member_ids)

        head = params["lm_head"]["weight"]
        clean_output = final_norm(params, clean_hidden[-1])
        # BF16 operands with FP32 accumulation avoid materializing a second
        # full FP32 copy of the vocabulary head on each 24 GiB GPU.
        full_clean_logits = jnp.matmul(
            clean_output.astype(head.dtype),
            head.T,
            preferred_element_type=jnp.float32,
        )
        clean_panel_logits, panel_token_ids = jax.lax.top_k(
            full_clean_logits, panel_size
        )
        panel_head = head[panel_token_ids]
        member_output = final_norm(params, member_hidden[:, -1])
        member_panel_logits = jnp.matmul(
            member_output.astype(panel_head.dtype),
            panel_head.T,
            preferred_element_type=jnp.float32,
        )
        return (
            clean_hidden,
            member_hidden,
            panel_token_ids,
            clean_panel_logits,
            logsumexp(full_clean_logits.astype(jnp.float32)),
            member_panel_logits,
        )

    return prefill_batch


def _require_array(
    path: Path, shape: tuple[int, ...], dtype: Any
) -> np.memmap:
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


def collect(args: Args) -> None:
    if args.panel_size < 1:
        raise ValueError("panel_size must be positive")
    if not 0.0 <= args.min_replay_cosine <= 1.0:
        raise ValueError("min_replay_cosine must lie in [0, 1]")
    if not 0.0 <= args.min_center_cosine <= 1.0:
        raise ValueError("min_center_cosine must lie in [0, 1]")
    if args.max_replay_relative_rmse < 0.0 or args.max_center_relative_rmse < 0.0:
        raise ValueError("relative-RMSE thresholds must be nonnegative")
    if args.max_new_samples is not None and args.max_new_samples < 1:
        raise ValueError("max_new_samples must be positive when provided")

    source_directory = Path(args.source_directory).expanduser().resolve()
    output_directory = (
        Path(args.output_directory).expanduser().resolve()
        if args.output_directory is not None
        else source_directory / "pact_prefills_v1"
    )
    if output_directory == source_directory:
        raise ValueError("output_directory must differ from source_directory")

    run_config_path = source_directory / "run_config.json"
    source_metadata_path = source_directory / "metadata.json"
    if not run_config_path.is_file() or not source_metadata_path.is_file():
        raise FileNotFoundError("source dataset is missing run_config.json or metadata.json")
    source_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    source_config_sha256 = _sha256_bytes(
        _canonical_json(source_config).encode("utf-8")
    )
    if source_metadata.get("status") != "complete":
        raise ValueError("source oracle dataset is not complete")
    if source_metadata.get("run_config_sha256") != source_config_sha256:
        raise ValueError("source metadata does not match source run_config.json")

    model_info = source_config["model"]
    dataset_info = source_config["dataset"]
    perturbation_info = source_config["perturbations"]
    predictor_info = source_config["predictor_input"]
    samples = int(dataset_info["dataset_size"])
    pairs = int(perturbation_info["directions_per_prompt"])
    layers = int(model_info["num_layers"])
    hidden = int(model_info["hidden_size"])
    sigma = float(perturbation_info["sigma"])
    center_rms_floor = float(predictor_info["center_rms_floor"])
    generation_length = int(
        source_config["rollout"]["generation_length_including_prompt"]
    )
    if perturbation_info != (
        perturbation_info
        | {
            "rank": 1,
            "freeze_nonlora": True,
            "noise_reuse": 1,
            "epoch": 0,
        }
    ):
        raise ValueError("source perturbation configuration is not PR5 rank-one replay")

    prompts = _require_array(
        source_directory / "prompt_tokens.npy",
        (samples, generation_length),
        np.int32,
    )
    prompt_lengths = _require_array(
        source_directory / "prompt_lengths.npy", (samples,), np.int32
    )
    source_completed = _require_array(
        source_directory / "completed_samples.npy", (samples,), np.uint8
    )
    if not np.all(source_completed == 1):
        raise ValueError("source completed_samples.npy is not fully committed")
    source_predictor_inputs = _require_array(
        source_directory / "predictor_inputs.npy",
        (samples * pairs * layers, hidden),
        np.float32,
    )
    source_center_rms = _require_array(
        source_directory / "center_rms.npy",
        (samples * pairs * layers,),
        np.float32,
    )
    _require_array(
        source_directory / "pair_rewards.npy", (samples, pairs, 2), np.float32
    )
    _require_array(
        source_directory / "reward_differences.npy", (samples, pairs), np.float32
    )
    prompt_width = int(np.max(prompt_lengths))
    if prompt_width < 1 or prompt_width > generation_length:
        raise ValueError("source prompt lengths are invalid")
    trimmed_prompts = np.asarray(prompts[:, :prompt_width], dtype=np.int32)
    for prompt, length in zip(trimmed_prompts, prompt_lengths):
        length = int(length)
        if np.any(prompt[:length] == 0) or np.any(prompt[length:] != 0):
            raise ValueError("token zero must occur only in trailing prompt padding")

    output_directory.mkdir(parents=True, exist_ok=True)
    collection_config = {
        "schema_version": 1,
        "source_directory": str(source_directory),
        "source_run_config_sha256": source_config_sha256,
        "source_git_commit": source_config["software"]["git_commit"],
        "model": {
            "choice": model_info["choice"],
            "dtype": model_info["dtype"],
            "config_sha256": model_info["config_sha256"],
            "num_layers": layers,
            "hidden_size": hidden,
        },
        "perturbations": {
            "master_seed": int(perturbation_info["master_seed"]),
            "epoch": int(perturbation_info["epoch"]),
            "sigma": sigma,
            "rank": int(perturbation_info["rank"]),
            "freeze_nonlora": bool(perturbation_info["freeze_nonlora"]),
            "noise_reuse": int(perturbation_info["noise_reuse"]),
            "directions_per_prompt": pairs,
            "global_pair_id": perturbation_info["global_pair_id"],
            "member_ids": perturbation_info["member_ids"],
        },
        "collection": {
            "samples": samples,
            "prompt_width": prompt_width,
            "attention_cache_width": generation_length,
            "layers": list(range(layers)),
            "panel_size": args.panel_size,
            "panel_selection": "top clean-model logits at the final real prompt token",
            "capture": "post-block residual at the final real prompt token",
            "member_hidden_dtype": "float32",
            "clean_hidden_dtype": "float32",
            "decoded_tokens": 0,
            "full_rollouts": 0,
            "scorer_calls": 0,
        },
        "validation": {
            "reference": "source predictor_inputs.npy and center_rms.npy",
            "formula": "(h_plus-h_minus)/(2*sigma*max(RMS((h_plus+h_minus)/2),floor))",
            "acceptance": "pooled cosine and relative RMSE",
            "min_replay_cosine": args.min_replay_cosine,
            "max_replay_relative_rmse": args.max_replay_relative_rmse,
            "min_center_cosine": args.min_center_cosine,
            "max_center_relative_rmse": args.max_center_relative_rmse,
            "note": (
                "BF16 compiler fusion changes the amplified finite difference; "
                "PACT response features reuse the exact saved PR5 tangent."
            ),
        },
    }
    collection_config_path = output_directory / "run_config.json"
    if collection_config_path.exists():
        existing = json.loads(collection_config_path.read_text(encoding="utf-8"))
        if existing != collection_config:
            raise ValueError(
                "existing PACT run_config.json differs; use a new output directory"
            )
    else:
        _atomic_write_json(collection_config_path, collection_config)
    collection_config_sha256 = _sha256_bytes(
        _canonical_json(collection_config).encode("utf-8")
    )

    array_specs: dict[str, tuple[tuple[int, ...], Any, Optional[float]]] = {
        "member_hidden.npy": ((samples, pairs, 2, layers, hidden), np.float32, None),
        "clean_hidden.npy": ((samples, layers, hidden), np.float32, None),
        "panel_token_ids.npy": ((samples, args.panel_size), np.int32, None),
        "clean_panel_logits.npy": (
            (samples, args.panel_size),
            np.float32,
            np.nan,
        ),
        "clean_logsumexp.npy": ((samples,), np.float32, np.nan),
        "member_panel_logits.npy": (
            (samples, pairs, 2, args.panel_size),
            np.float32,
            None,
        ),
        "replay_max_abs_error.npy": ((samples,), np.float32, np.nan),
        "replay_rmse.npy": ((samples,), np.float32, np.nan),
        "replay_relative_rmse.npy": ((samples,), np.float32, np.nan),
        "replay_cosine.npy": ((samples,), np.float32, np.nan),
        "replay_center_max_abs_error.npy": ((samples,), np.float32, np.nan),
        "replay_center_relative_rmse.npy": ((samples,), np.float32, np.nan),
        "replay_center_cosine.npy": ((samples,), np.float32, np.nan),
        "completed_samples.npy": ((samples,), np.uint8, 0),
    }
    missing_bytes = sum(
        int(np.prod(shape)) * np.dtype(dtype).itemsize
        for name, (shape, dtype, _) in array_specs.items()
        if not (output_directory / name).exists()
    )
    free_bytes = shutil.disk_usage(output_directory).free
    if free_bytes < missing_bytes + 512 * 1024**2:
        raise OSError(
            f"Need {missing_bytes / 1024**3:.2f} GiB plus headroom; "
            f"only {free_bytes / 1024**3:.2f} GiB is free"
        )
    arrays = {
        name: _open_memmap(
            output_directory / name, shape=shape, dtype=dtype, fill=fill
        )
        for name, (shape, dtype, fill) in array_specs.items()
    }
    completed = arrays["completed_samples.npy"]
    if np.any((completed != 0) & (completed != 1)):
        raise ValueError("completed_samples.npy contains values other than zero/one")

    manifest = {
        "schema_version": 1,
        "run_config_sha256": collection_config_sha256,
        "source_run_config_sha256": source_config_sha256,
        "arrays": {name: list(shape) for name, (shape, _, _) in array_specs.items()},
        "array_dtypes": {
            name: str(np.dtype(dtype)) for name, (_, dtype, _) in array_specs.items()
        },
        "labels": {
            "pair_rewards": str(source_directory / "pair_rewards.npy"),
            "reward_differences": str(source_directory / "reward_differences.npy"),
        },
        "logical_clean_prefills": samples,
        "perturbed_prefills": samples * pairs * 2,
        "full_rollouts": 0,
        "decoded_tokens": 0,
    }
    _atomic_write_json(output_directory / "manifest.json", manifest)

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

    from hyperscalees.models.common import simple_es_tree_key
    from hyperscalees.models.llm.auto import get_model
    from hyperscalees.noiser.eggroll import EggRoll

    gpu_devices = [device for device in jax.local_devices() if device.platform == "gpu"]
    if not gpu_devices:
        raise RuntimeError(f"Expected a visible GPU, found {jax.local_devices()}")
    num_devices = len(gpu_devices)
    if pairs % num_devices:
        raise ValueError("directions_per_prompt must divide evenly across GPUs")

    print(f"Loading {model_info['choice']} as {model_info['dtype']}...")
    model, full_params, _ = get_model(
        model_info["choice"],
        rwkv_type="Qwen35RWKV",
        verbose=True,
        dtype=model_info["dtype"],
    )
    config, params, scan_map, _ = full_params
    loaded_config_sha256 = _sha256_bytes(
        _canonical_json(config).encode("utf-8")
    )
    if loaded_config_sha256 != model_info["config_sha256"]:
        raise ValueError("loaded model config does not match the source dataset")
    if len(config["layer_types"]) != layers or int(config["hidden_size"]) != hidden:
        raise ValueError("loaded model dimensions do not match source metadata")
    if args.panel_size > int(config["vocab_size"]):
        raise ValueError("panel_size exceeds loaded vocabulary")

    replica_mesh = Mesh(np.asarray(gpu_devices), ("replica",))
    replicated_sharding = NamedSharding(replica_mesh, P())
    params = jax.tree.map(
        lambda value: jax.device_put(value, replicated_sharding), params
    )
    master_key = jax.random.key(int(perturbation_info["master_seed"]))
    base_model_key = jax.random.fold_in(master_key, 0)
    base_evo_keys = simple_es_tree_key(params, base_model_key, scan_map)
    # The prefill path needs only these EggRoll fields and sigma.  Avoid
    # constructing an unused optimizer state as large as the 2B model.
    frozen_noiser_params = {
        "group_size": 2 * pairs,
        "freeze_nonlora": True,
        "noise_reuse": 1,
        "rank": 1,
    }
    noiser_params = {"sigma": jnp.asarray(sigma, dtype=jnp.float32)}

    one_replica = build_pact_prefill_batch(
        model,
        EggRoll,
        frozen_noiser_params,
        config,
        base_evo_keys,
        preview_layers=tuple(range(layers)),
        prompt_width=prompt_width,
        attention_cache_width=generation_length,
        panel_size=args.panel_size,
    )
    parallel_prefill = jax.pmap(
        one_replica,
        in_axes=(None, None, None, None, 0, None),
        devices=gpu_devices,
    )
    members_per_device = 2 * pairs // num_devices
    print(
        f"Compiling {members_per_device} perturbed members/GPU, "
        f"prompt width {prompt_width}, attention cache {generation_length}, "
        f"all {layers} layers..."
    )
    compile_start = time.time()
    compiled_prefill = parallel_prefill.lower(
        noiser_params,
        params,
        jax.ShapeDtypeStruct((prompt_width,), jnp.dtype("int32")),
        jax.ShapeDtypeStruct((), jnp.dtype("int32")),
        jax.ShapeDtypeStruct(
            (num_devices, members_per_device), jnp.dtype("int32")
        ),
        jnp.asarray(0, dtype=jnp.int32),
    ).compile()
    print(f"Compilation finished in {time.time() - compile_start:.1f}s")
    print(compiled_prefill.memory_analysis())

    completed_mask = np.asarray(completed, dtype=bool)
    metadata_path = output_directory / "metadata.json"
    started = time.time()
    base_metadata = {
        "schema_version": 1,
        "status": "running",
        "started_or_resumed_at": _utc_now(),
        "run_config_sha256": collection_config_sha256,
        "source_run_config_sha256": source_config_sha256,
        "devices": [str(device) for device in gpu_devices],
        "device_count": num_devices,
        "members_per_device": members_per_device,
        "prompt_width": prompt_width,
        "completed_samples": int(completed_mask.sum()),
        "total_samples": samples,
        "full_rollouts": 0,
        "decoded_tokens": 0,
        "scorer_calls": 0,
    }
    _atomic_write_json(metadata_path, base_metadata)

    pending_sample_ids = np.flatnonzero(~completed_mask)
    if args.max_new_samples is not None:
        pending_sample_ids = pending_sample_ids[: args.max_new_samples]
    progress = tqdm(
        total=samples,
        initial=int(completed_mask.sum()),
        desc="PACT prompt prefills",
        unit="prompt",
        dynamic_ncols=True,
    )
    try:
        for sample_id in pending_sample_ids:
            sample_start = time.time()
            member_ids = global_member_ids(int(sample_id), pairs)
            result = jax.block_until_ready(
                compiled_prefill(
                    noiser_params,
                    params,
                    jnp.asarray(trimmed_prompts[sample_id]),
                    jnp.asarray(prompt_lengths[sample_id], dtype=jnp.int32),
                    jnp.asarray(shard_complete_pairs(member_ids, num_devices)),
                    jnp.asarray(0, dtype=jnp.int32),
                )
            )
            (
                clean_by_device,
                member_by_device,
                panel_ids_by_device,
                clean_logits_by_device,
                clean_lse_by_device,
                member_logits_by_device,
            ) = (np.asarray(jax.device_get(value)) for value in result)

            # Clean work is intentionally replicated inside pmap to avoid an
            # additional model copy or cross-sharding path.  It must agree.
            for replica in range(1, num_devices):
                np.testing.assert_allclose(
                    clean_by_device[replica],
                    clean_by_device[0],
                    rtol=1e-6,
                    atol=1e-6,
                )
                np.testing.assert_array_equal(
                    panel_ids_by_device[replica], panel_ids_by_device[0]
                )
                np.testing.assert_allclose(
                    clean_logits_by_device[replica],
                    clean_logits_by_device[0],
                    rtol=1e-6,
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    clean_lse_by_device[replica],
                    clean_lse_by_device[0],
                    rtol=1e-6,
                    atol=1e-6,
                )

            clean_np = np.asarray(clean_by_device[0], dtype=np.float32)
            members_np = np.asarray(member_by_device, dtype=np.float32).reshape(
                pairs, 2, layers, hidden
            )
            panel_ids_np = np.asarray(panel_ids_by_device[0], dtype=np.int32)
            clean_logits_np = np.asarray(
                clean_logits_by_device[0], dtype=np.float32
            )
            clean_lse_np = np.float32(clean_lse_by_device[0])
            member_logits_np = np.asarray(
                member_logits_by_device, dtype=np.float32
            ).reshape(pairs, 2, args.panel_size)
            if clean_np.shape != (layers, hidden):
                raise RuntimeError(f"unexpected clean hidden shape {clean_np.shape}")
            if members_np.shape != (pairs, 2, layers, hidden):
                raise RuntimeError(f"unexpected member hidden shape {members_np.shape}")
            if np.unique(panel_ids_np).size != args.panel_size:
                raise RuntimeError("clean policy panel contains duplicate token IDs")
            if not all(
                np.all(np.isfinite(value))
                for value in (
                    clean_np,
                    members_np,
                    clean_logits_np,
                    member_logits_np,
                    np.asarray(clean_lse_np),
                )
            ):
                raise RuntimeError("prefill outputs contain non-finite values")

            replay_inputs, replay_center = antithetic_replay_statistics(
                members_np, sigma, center_rms_floor
            )
            row_start = int(sample_id) * pairs * layers
            row_stop = row_start + pairs * layers
            expected_inputs = np.asarray(
                source_predictor_inputs[row_start:row_stop], dtype=np.float32
            ).reshape(pairs, layers, hidden)
            expected_center = np.asarray(
                source_center_rms[row_start:row_stop], dtype=np.float32
            ).reshape(pairs, layers)
            input_errors = replay_error_summary(replay_inputs, expected_inputs)
            center_errors = replay_error_summary(replay_center, expected_center)
            if (
                input_errors["cosine"] < args.min_replay_cosine
                or input_errors["relative_rmse"] > args.max_replay_relative_rmse
            ):
                raise RuntimeError(
                    "replayed perturbations fail PR5 direction alignment: "
                    f"sample={sample_id}, input={input_errors}, center={center_errors}"
                )
            if (
                center_errors["cosine"] < args.min_center_cosine
                or center_errors["relative_rmse"] > args.max_center_relative_rmse
            ):
                raise RuntimeError(
                    "replayed centers fail PR5 alignment: "
                    f"sample={sample_id}, {center_errors}"
                )

            arrays["member_hidden.npy"][sample_id] = members_np
            arrays["clean_hidden.npy"][sample_id] = clean_np
            arrays["panel_token_ids.npy"][sample_id] = panel_ids_np
            arrays["clean_panel_logits.npy"][sample_id] = clean_logits_np
            arrays["clean_logsumexp.npy"][sample_id] = clean_lse_np
            arrays["member_panel_logits.npy"][sample_id] = member_logits_np
            arrays["replay_max_abs_error.npy"][sample_id] = input_errors["max_abs"]
            arrays["replay_rmse.npy"][sample_id] = input_errors["rmse"]
            arrays["replay_relative_rmse.npy"][sample_id] = input_errors[
                "relative_rmse"
            ]
            arrays["replay_cosine.npy"][sample_id] = input_errors["cosine"]
            arrays["replay_center_max_abs_error.npy"][sample_id] = center_errors[
                "max_abs"
            ]
            arrays["replay_center_relative_rmse.npy"][sample_id] = center_errors[
                "relative_rmse"
            ]
            arrays["replay_center_cosine.npy"][sample_id] = center_errors["cosine"]
            for name, array in arrays.items():
                if name != "completed_samples.npy":
                    array.flush()
            completed[sample_id] = 1
            completed.flush()
            completed_mask[sample_id] = True

            progress.update(1)
            progress.set_postfix(
                max_err=f"{input_errors['max_abs']:.2e}",
                cosine=f"{input_errors['cosine']:.5f}",
                batch_s=f"{time.time() - sample_start:.1f}",
                rollouts=0,
            )
            _atomic_write_json(
                metadata_path,
                base_metadata
                | {
                    "status": "running",
                    "updated_at": _utc_now(),
                    "completed_samples": int(completed_mask.sum()),
                    "perturbed_prefills": int(completed_mask.sum()) * pairs * 2,
                    "logical_clean_prefills": int(completed_mask.sum()),
                    "physical_clean_prefills": int(completed_mask.sum())
                    * num_devices,
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

    if not np.all(completed_mask):
        partial_metadata = base_metadata | {
            "status": "partial",
            "updated_at": _utc_now(),
            "completed_samples": int(completed_mask.sum()),
            "perturbed_prefills": int(completed_mask.sum()) * pairs * 2,
            "logical_clean_prefills": int(completed_mask.sum()),
            "physical_clean_prefills": int(completed_mask.sum()) * num_devices,
            "full_rollouts": 0,
            "decoded_tokens": 0,
            "scorer_calls": 0,
            "session_seconds": time.time() - started,
        }
        _atomic_write_json(metadata_path, partial_metadata)
        print(
            f"Partial collection: {int(completed_mask.sum())}/{samples} prompts "
            f"committed in {output_directory}"
        )
        print("Run the same command without --max-new-samples to resume.")
        return

    final_metadata = base_metadata | {
        "status": "complete",
        "completed_at": _utc_now(),
        "completed_samples": samples,
        "perturbed_prefills": samples * pairs * 2,
        "logical_clean_prefills": samples,
        "physical_clean_prefills": samples * num_devices,
        "full_rollouts": 0,
        "decoded_tokens": 0,
        "scorer_calls": 0,
        "session_seconds": time.time() - started,
        "max_replay_abs_error": float(
            np.max(np.asarray(arrays["replay_max_abs_error.npy"]))
        ),
        "max_replay_center_abs_error": float(
            np.max(np.asarray(arrays["replay_center_max_abs_error.npy"]))
        ),
        "min_replay_cosine": float(
            np.min(np.asarray(arrays["replay_cosine.npy"]))
        ),
        "max_replay_relative_rmse": float(
            np.max(np.asarray(arrays["replay_relative_rmse.npy"]))
        ),
        "min_replay_center_cosine": float(
            np.min(np.asarray(arrays["replay_center_cosine.npy"]))
        ),
        "max_replay_center_relative_rmse": float(
            np.max(np.asarray(arrays["replay_center_relative_rmse.npy"]))
        ),
        "npy_payload_bytes": sum(
            path.stat().st_size for path in output_directory.glob("*.npy")
        ),
    }
    _atomic_write_json(metadata_path, final_metadata)
    print(f"Complete: {output_directory}")
    print("Full rollouts: 0; decoded tokens: 0; scorer calls: 0")
    print(
        "PR5 replay alignment: "
        f"min cosine={final_metadata['min_replay_cosine']:.6f}, "
        f"max relative RMSE={final_metadata['max_replay_relative_rmse']:.6f}"
    )


def main() -> None:
    import tyro

    args = tyro.cli(Args)
    if args.self_test:
        _self_test()
        return
    collect(args)


if __name__ == "__main__":
    main()
