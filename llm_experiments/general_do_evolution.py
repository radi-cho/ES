import os
import sys
import csv
import jax
from huggingface_hub.constants import HF_HOME

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

jax.config.update("jax_compilation_cache_dir", os.path.join(HF_HOME, "hyperscaleescomp"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jax.numpy as jnp

import numpy as np

import hyperscalees as hs
from hyperscalees.models.llm.auto import get_model, models
from hyperscalees.models.llm.tokenizer import LegacyWorldTokenizer
from hyperscalees.models.common import simple_es_tree_key

from hyperscalees.noiser import all_noisers
from hyperscalees.noiser.predictive_eggroll import (
    audit_correct_pair_differences,
    make_countsketch,
    make_online_surrogate,
    pair_differences_to_member_utilities,
    pair_ids_to_member_ids,
    prompt_center_features,
    sample_stratified_audit_pairs,
)
from hyperscalees.environments.llm_bandits import all_tasks, validation_tasks

import tyro
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Literal
from pathlib import Path

from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.experimental.multihost_utils import process_allgather

from omegaconf import DictConfig, OmegaConf   
from hydra import initialize, compose, initialize_config_dir
from hydra.utils import instantiate

from .utils import (
    build_generate_thread,
    build_preview_pair_thread,
    build_validate,
    build_hellaswag_validate,
    safe_decode
)

import time

import tqdm

import operator

import wandb

@dataclass
class Args:
    seed: int = 0
    model_choice: Literal[tuple(models.keys())] =  "7g0.1B"
    output_directory: Optional[str] = "."
    wandb_directory: Optional[str] = "."

    rwkv_type: str = "BaseRWKV"
    dtype: Optional[str] = None

    parallel_generations_per_gpu: int = 1024

    generation_length: int = 100
    thinking_length: int = 100
    answer_length: int = 100

    num_epochs: int = 100
    log_output_every: int = 10

    lr_scale: float = 1.0
    sigma: float = 1e-3
    noise_reuse: int = 1
    freeze_nonlora: bool = True
    temperature: float = 0.0

    validate_every: int = 10
    parallel_validations: int = 128
    validation_iterations: int = 10

    task: Literal[tuple(all_tasks.keys())] = "fastzero"
    noiser: Literal[tuple(all_noisers.keys())] = "eggroll"

    wandb_mode: Literal["online", "offline"] = "online"
    wandb_project: str = "HyperscaleExp"
    wandb_name: str = "full"
    track: bool = False

    generations_per_prompt: int = 8

    # Prompt-preview surrogate for a larger virtual EGGROLL population.
    predictive_virtual_factor: int = 16
    predictive_preview_microbatch_pairs: int = 8
    predictive_sketch_size: int = 128
    predictive_feature_kind: Literal["sketch", "sketch_summary"] = "sketch"
    predictive_prompt_center: bool = False
    predictive_surrogate: Literal["ridge"] = "ridge"
    predictive_ridge: float = 10.0
    predictive_decay: float = 0.99
    predictive_min_observations: int = 256
    predictive_prediction_clip: float = 1.1
    predictive_reward_scale: float = 0.5
    predictive_reward_scale_decay: float = 0.9
    predictive_minimum_reward_scale: float = 0.1
    predictive_feature_seed: int = 0
    predictive_audit_seed: int = 1
    predictive_rms_floor: float = 1e-4
    predictive_use_predictions: bool = True
    # Deprecated PR5 compatibility flags. PR6 replaces the one-step binary
    # gate with continuously shrunk, lagged calibration.
    predictive_max_residual_ratio: float = 0.9
    predictive_min_quality_nonzero_labels: int = 4
    predictive_calibration_decay: float = 0.9
    predictive_calibration_max_scale: float = 1.0
    predictive_calibration_min_observations: int = 32
    predictive_calibration_prior_observations: float = 64.0

    train_dataset_size: Optional[int] = None
    val_dataset_size: Optional[int] = None
    time_budget_seconds: Optional[float] = None
    random_train_prompts: bool = False
    aux_validation_task: Optional[Literal["hellaswag"]] = None
    hellaswag_val_size: int = 256
    hellaswag_val_seed: int = 42

    coord_addr: Optional[str] = None
    num_procs: Optional[int] = None
    proc_id: Optional[int] = None


args = tyro.cli(Args)
profile = os.getenv("PROFILE", "default")
CONFIG_DIR = (Path(__file__).resolve().parents[1] / "configs").as_posix()

if args.model_choice.startswith("q35_") and args.rwkv_type == "BaseRWKV":
    args.rwkv_type = "Qwen35RWKV"

suppress_eos_token = 0 if args.model_choice[0] == "7" else None

if profile !=  "default":
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        user_cfg = compose(config_name=profile) 
        
    # Override config with vals from yaml
    user_overrides = OmegaConf.to_container(user_cfg, resolve=True)
    for k, v in user_overrides.items():
        if hasattr(args, k) and v is not None:
            setattr(args, k, v) 

print()
print(f"Using config: {profile}")
print()
args.generation_length = args.thinking_length + args.answer_length

master_key = jax.random.key(args.seed)

base_model_key = jax.random.fold_in(master_key, 0)
base_gen_key = jax.random.fold_in(master_key, 1)
base_valid_key = jax.random.fold_in(master_key, 2)

NOISER = all_noisers[args.noiser]
# NOISER = hs.noiser.eggroll.EggRoll # TODO: make this a parameter
# NOISER = hs.noiser.base_noiser.Noiser

print("starting distributed init")
if args.coord_addr is not None:
    jax.distributed.initialize(args.coord_addr, args.num_procs, args.proc_id)
else:
    print("NOT DISTRIBUTED CONTEXT")

total_num_devices = len(jax.devices())
print("global devices", jax.devices())
print("local devices", jax.local_devices())
print("process id", jax.process_index())
args.proc_id = jax.process_index()
args.total_parallel_generations = total_num_devices * args.parallel_generations_per_gpu

# args.lr = args.lr_scale * (args.sigma ** 2) * np.sqrt(args.total_parallel_generations)
USE_SHARD_MAP = total_num_devices > 1
mesh = jax.make_mesh((len(jax.devices()),), ("data",)) if USE_SHARD_MAP else None

print()
print("per-device generations is", args.parallel_generations_per_gpu)
print("physical number of generations is", args.total_parallel_generations)

if args.generations_per_prompt < 1:
    raise ValueError("generations_per_prompt must be positive")
if args.total_parallel_generations % args.generations_per_prompt:
    raise ValueError("physical population must divide evenly into prompt groups")
args.prompts_per_epoch = args.total_parallel_generations // args.generations_per_prompt

PREDICTIVE_MODE = args.noiser == "predictive_eggroll"
if PREDICTIVE_MODE:
    if args.generations_per_prompt < 2 or args.generations_per_prompt % 2:
        raise ValueError(
            "predictive_eggroll needs positive, even generations_per_prompt"
        )
    if total_num_devices != 1:
        raise ValueError("predictive_eggroll currently supports one visible device")
    if not args.freeze_nonlora:
        raise ValueError("predictive_eggroll currently requires freeze_nonlora")
    if args.predictive_virtual_factor < 1:
        raise ValueError("predictive_virtual_factor must be positive")
    if args.predictive_preview_microbatch_pairs < 1:
        raise ValueError("predictive_preview_microbatch_pairs must be positive")
    if args.sigma <= 0.0:
        raise ValueError("predictive_eggroll requires sigma > 0")
    if args.predictive_reward_scale <= 0.0:
        raise ValueError("predictive_reward_scale must be positive")
    if not 0.0 <= args.predictive_reward_scale_decay <= 1.0:
        raise ValueError("predictive_reward_scale_decay must be in [0, 1]")
    if args.predictive_minimum_reward_scale <= 0.0:
        raise ValueError("predictive_minimum_reward_scale must be positive")
    if args.predictive_sketch_size < 1:
        raise ValueError("predictive_sketch_size must be positive")
    if not 0.0 < args.predictive_calibration_decay <= 1.0:
        raise ValueError("predictive_calibration_decay must be in (0, 1]")
    if args.predictive_calibration_max_scale <= 0.0:
        raise ValueError("predictive_calibration_max_scale must be positive")
    if args.predictive_calibration_min_observations < 0:
        raise ValueError("predictive_calibration_min_observations must be nonnegative")
    if args.predictive_calibration_prior_observations < 0.0:
        raise ValueError("predictive_calibration_prior_observations must be nonnegative")
    if args.predictive_max_residual_ratio <= 0.0:
        raise ValueError("predictive_max_residual_ratio must be positive")
    if args.predictive_min_quality_nonzero_labels < 1:
        raise ValueError("predictive_min_quality_nonzero_labels must be positive")
    args.virtual_generations_per_prompt = (
        args.generations_per_prompt * args.predictive_virtual_factor
    )
    args.virtual_total_parallel_generations = (
        args.total_parallel_generations * args.predictive_virtual_factor
    )
    if args.temperature != 0.0:
        print("WARNING: stochastic rollout noise is not represented by prompt previews")
else:
    args.virtual_generations_per_prompt = args.generations_per_prompt
    args.virtual_total_parallel_generations = args.total_parallel_generations

if PREDICTIVE_MODE:
    print(
        "virtual number of generations is",
        args.virtual_total_parallel_generations,
        f"({args.virtual_generations_per_prompt} per prompt; "
        f"1/{args.predictive_virtual_factor} fully evaluated)",
    )

RWKV, full_params, tokenizer = get_model(args.model_choice, rwkv_type=args.rwkv_type, verbose=True, dtype=args.dtype)
legacy_tokenizer = LegacyWorldTokenizer() if args.model_choice[0] == "7" else tokenizer

config, params, scan_map, es_map = full_params

def _task_kwargs(task_name: str, *, dataset_size: Optional[int], seed: Optional[int], val_holdout_size: Optional[int] = None) -> dict:
    if task_name != "countdown_chat":
        return {}
    kwargs = {}
    if dataset_size is not None:
        kwargs["dataset_size"] = dataset_size
    if seed is not None:
        kwargs["seed"] = seed
    if val_holdout_size is not None:
        kwargs["val_holdout_size"] = val_holdout_size
    return kwargs

train_ds = args.train_dataset_size if args.train_dataset_size is not None else 256
val_holdout = args.val_dataset_size if args.val_dataset_size is not None else 256
Task = all_tasks[args.task](
    tokenizer,
    legacy_tokenizer,
    args.generation_length,
    **_task_kwargs(args.task, dataset_size=train_ds, seed=args.seed, val_holdout_size=val_holdout),
)
print(f"Train dataset size: {len(Task)}")
if args.task == "countdown_chat":
    from hyperscalees.environments.llm_bandits import countdown_train_val_overlap
    overlap = countdown_train_val_overlap(train_ds, val_holdout, train_seed=args.seed)
    print(f"Countdown train∩val overlap: {overlap} (disjoint split, seed={42})")
if args.random_train_prompts:
    if args.prompts_per_epoch > len(Task):
        raise ValueError(
            f"random_train_prompts needs prompts_per_epoch ({args.prompts_per_epoch}) "
            f"<= train dataset size ({len(Task)})"
        )
    print(f"Random train prompt sampling: {args.prompts_per_epoch} unique examples per epoch")

def replicate_matrix(x):
    if not USE_SHARD_MAP:
        return x
    return jax.make_array_from_single_device_arrays(
        x.shape, NamedSharding(mesh, P()), [jax.device_put(x, d) for d in jax.local_devices()]
    )

def _data_sharding(x):
    # shard_map expects P('data', None) for 2D batch args, not P('data',)
    if x.ndim == 1:
        return NamedSharding(mesh, P("data"))
    return NamedSharding(mesh, P("data", None))


def shard_on_data(x):
    if not USE_SHARD_MAP:
        return jnp.asarray(x)
    x = np.asarray(x)
    sharding = _data_sharding(x)
    arr = jax.make_array_from_single_device_arrays(
        x.shape,
        sharding,
        [jax.device_put(x, d) for d in jax.local_devices()],
    )
    return jax.sharding.reshard(arr, sharding)

params = jax.tree.map(replicate_matrix, params)
frozen_noiser_params, noiser_params = NOISER.init_noiser(
    params,
    args.sigma,
    args.lr_scale,
    group_size=args.virtual_generations_per_prompt,
    freeze_nonlora=args.freeze_nonlora,
    noise_reuse=args.noise_reuse,
)
base_evo_keys = simple_es_tree_key(params, base_model_key, scan_map)


physical_thread_idxes = shard_on_data(np.arange(args.total_parallel_generations))
global_indices = shard_on_data(np.arange(args.virtual_total_parallel_generations))

_generate_thread = build_generate_thread(
    RWKV,
    NOISER,
    frozen_noiser_params,
    config,
    base_evo_keys,
    base_gen_key,
    args.temperature,
    for_shard_map=USE_SHARD_MAP,
    suppress_eos_token=suppress_eos_token,
)

print("Compiling generate batch")
start_time = time.time()
if USE_SHARD_MAP:
    generate_batch = jax.jit(
        shard_map(
            jax.vmap(_generate_thread, in_axes=(None, None, 0, 0, None)),
            mesh=mesh,
            in_specs=(P(), P(), P("data"), P("data"), P()),
            out_specs=P("data"),
        )
    ).lower(
        noiser_params,
        params,
        shard_on_data(
            np.zeros(
                (args.total_parallel_generations, args.generation_length),
                dtype=np.int32,
            )
        ),
        physical_thread_idxes,
        0,
    ).compile()
else:
    generate_batch = jax.jit(
        jax.vmap(_generate_thread, in_axes=(None, None, 0, 0, None))
    ).lower(
        noiser_params,
        params,
        jax.ShapeDtypeStruct(
            (args.total_parallel_generations, args.generation_length),
            jnp.dtype("int32"),
        ),
        jnp.arange(args.total_parallel_generations, dtype=jnp.int32),
        0,
    ).compile()
print("Compile time", time.time() - start_time)
print("memory info")
print(generate_batch.memory_analysis())

validate = build_validate(RWKV, config, params, base_evo_keys, base_valid_key, tokenizer, legacy_tokenizer, args, args.temperature, suppress_eos_token=suppress_eos_token)

hellaswag_validate = None
hellaswag_csv_path = None
if args.aux_validation_task == "hellaswag":
    hellaswag_validate = build_hellaswag_validate(
        RWKV,
        config,
        params,
        base_evo_keys,
        base_valid_key,
        tokenizer,
        NOISER=NOISER,
        val_size=args.hellaswag_val_size,
        seed=args.hellaswag_val_seed,
        suppress_eos_token=suppress_eos_token,
    )

def _do_update(noiser_params, params, raw_scores, epoch_num):
    iterinfos = (jnp.full_like(raw_scores, epoch_num, dtype=jnp.int32), global_indices)

    fitnesses = NOISER.convert_fitnesses(frozen_noiser_params, noiser_params, raw_scores)
    noiser_params, new_params = NOISER.do_updates(frozen_noiser_params, noiser_params, params, base_evo_keys, fitnesses, iterinfos, es_map)

    return noiser_params, new_params, jax.tree.map(lambda x, y: jnp.sqrt(jnp.mean((x - y) ** 2)), params, new_params)


print()
print("Compiling do update")
start_time = time.time()
if USE_SHARD_MAP:
    do_update = jax.jit(
        shard_map(
            _do_update,
            mesh=mesh,
            in_specs=(P(), P(), P("data"), P()),
            out_specs=(P(), P(), P()),
        ),
        donate_argnums=(0, 1),
    ).lower(
        noiser_params,
        params,
        shard_on_data(
            np.zeros(args.virtual_total_parallel_generations, dtype=np.float32)
        ),
        0,
    ).compile()
else:
    do_update = jax.jit(_do_update, donate_argnums=(0, 1)).lower(
        noiser_params,
        params,
        jnp.zeros(args.virtual_total_parallel_generations, dtype=jnp.float32),
        0,
    ).compile()
print("Compile time", time.time() - start_time)
print("memory info")
print(do_update.memory_analysis())

predictive_predictor = None
preview_feature_batch = None
preview_prompt_width = None
preview_layers = None
countsketch_buckets = None
countsketch_signs = None


def _zero_padded_prompt_lengths(prompts: np.ndarray) -> np.ndarray:
    """Return the last nonzero token position plus one for each prompt."""

    nonzero = np.asarray(prompts) != 0
    has_token = np.any(nonzero, axis=1)
    last_from_end = np.argmax(nonzero[:, ::-1], axis=1)
    return np.where(has_token, prompts.shape[1] - last_from_end, 0).astype(np.int32)


if PREDICTIVE_MODE:
    if RWKV.__name__ != "Qwen35RWKV" or "layer_types" not in config:
        raise ValueError("predictive_eggroll requires the instrumented Qwen35 model")
    num_layers = len(config["layer_types"])
    late_layer = min(num_layers - 1, int(np.floor(0.75 * num_layers)))
    preview_layers = (late_layer, num_layers - 1)
    if len(set(preview_layers)) != 2:
        raise ValueError("predictive_eggroll requires at least two distinct layers")

    task_prompts = np.asarray(
        Task.get_input(jnp.arange(len(Task), dtype=jnp.int32)), dtype=np.int32
    )
    task_prompt_lengths = _zero_padded_prompt_lengths(task_prompts)
    if not np.all(task_prompt_lengths > 0):
        raise ValueError("predictive_eggroll requires nonempty, zero-padded prompts")
    if any(
        np.any(prompt[:length] == 0)
        for prompt, length in zip(task_prompts, task_prompt_lengths)
    ):
        raise ValueError(
            "predictive_eggroll requires token 0 to occur only in prompt padding"
        )
    preview_prompt_width = int(task_prompt_lengths.max())

    countsketch_buckets, countsketch_signs = make_countsketch(
        num_layers=2,
        hidden_size=int(config["hidden_size"]),
        sketch_size=args.predictive_sketch_size,
        seed=args.predictive_feature_seed,
    )
    summary_dim = (
        6 * len(preview_layers)
        if args.predictive_feature_kind == "sketch_summary"
        else 0
    )
    feature_dim = len(preview_layers) * args.predictive_sketch_size + summary_dim
    predictive_predictor = make_online_surrogate(
        args.predictive_surrogate,
        feature_dim=feature_dim,
        ridge=args.predictive_ridge,
        decay=args.predictive_decay,
        min_observations=args.predictive_min_observations,
        prediction_clip=args.predictive_prediction_clip,
        rms_floor=args.predictive_rms_floor,
        initial_reward_scale=args.predictive_reward_scale,
        reward_scale_decay=args.predictive_reward_scale_decay,
        minimum_reward_scale=args.predictive_minimum_reward_scale,
        calibration_decay=args.predictive_calibration_decay,
        calibration_max_scale=args.predictive_calibration_max_scale,
        calibration_min_observations=args.predictive_calibration_min_observations,
        calibration_prior_observations=args.predictive_calibration_prior_observations,
        calibrate_predictions=True,
    )

    preview_pair = build_preview_pair_thread(
        RWKV,
        NOISER,
        frozen_noiser_params,
        config,
        base_evo_keys,
        preview_layers,
        preview_prompt_width,
        countsketch_buckets,
        countsketch_signs,
        num_buckets=args.predictive_sketch_size,
        center_rms_floor=args.predictive_rms_floor,
        feature_kind=args.predictive_feature_kind,
    )
    print(
        "Compiling hidden-only preview batch:",
        f"pairs={args.predictive_preview_microbatch_pairs},",
        f"prompt_width={preview_prompt_width}, layers={preview_layers},",
        f"feature_kind={args.predictive_feature_kind}, features={feature_dim}",
    )
    start_time = time.time()
    preview_feature_batch = jax.jit(
        jax.vmap(preview_pair, in_axes=(None, None, 0, 0, 0, None))
    ).lower(
        noiser_params,
        params,
        jax.ShapeDtypeStruct(
            (args.predictive_preview_microbatch_pairs, preview_prompt_width),
            jnp.dtype("int32"),
        ),
        jax.ShapeDtypeStruct(
            (args.predictive_preview_microbatch_pairs,), jnp.dtype("int32")
        ),
        jax.ShapeDtypeStruct(
            (args.predictive_preview_microbatch_pairs,), jnp.dtype("int32")
        ),
        0,
    ).compile()
    print("Compile time", time.time() - start_time)
    print("memory info")
    print(preview_feature_batch.memory_analysis())

    print(
        "Predictive EGGROLL enabled:",
        f"physical={args.total_parallel_generations},",
        f"virtual={args.virtual_total_parallel_generations},",
        f"audit_probability={1.0 / args.predictive_virtual_factor:.4f},",
        f"audit_seed={args.predictive_audit_seed},",
        f"feature_seed={args.predictive_feature_seed},",
        f"surrogate={args.predictive_surrogate},",
        f"prompt_center={args.predictive_prompt_center},",
        f"initial_reward_scale={args.predictive_reward_scale},",
        f"warmup_labels={args.predictive_min_observations}",
    )

true_train_fitness_sum = 0.0

FULL = 0
LORA = 1

full_name = f"{args.task}_{args.noiser}_{args.wandb_name}_lr={args.lr_scale}_sigma={args.sigma:.2e}_bs={args.total_parallel_generations}"
if PREDICTIVE_MODE:
    full_name += f"_virtual={args.virtual_total_parallel_generations}"
if args.train_dataset_size is not None:
    full_name += f"_trainD={args.train_dataset_size}"
experiment_id = f"{full_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

base_out_dir = Path(args.output_directory) if args.output_directory else (Path.cwd() / "outputs")
run_out_dir = base_out_dir / f"{experiment_id}"
run_out_dir.mkdir(parents=True, exist_ok=True)

fitness_csv_path = run_out_dir / "fitness.csv"
validation_csv_path = run_out_dir / "validation.csv"
predictive_csv_path = run_out_dir / "predictive_surrogate.csv"
hellaswag_csv_path = run_out_dir / "hellaswag_validation.csv"
figure_4b_path = run_out_dir / "figure_4b.png"

print("Run name", full_name)
print("Output directory:", run_out_dir)
if args.track:
    if args.wandb_mode == "offline":
        os.environ["WANDB_MODE"] = "offline" 
    
    wandb_dir = (Path(args.wandb_directory) / "wandb_runs").resolve()
    wandb_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project=args.wandb_project,
        config=args,
        name=full_name,
        dir=str(wandb_dir),
    )

def _epoch_train_indices(epoch: int) -> np.ndarray:
    """Dataset row indices for this epoch's unique train prompts."""
    if args.random_train_prompts:
        rng = np.random.default_rng(args.seed + epoch)
        return rng.choice(len(Task), size=args.prompts_per_epoch, replace=False)
    start = epoch * args.prompts_per_epoch
    return np.arange(start, start + args.prompts_per_epoch, dtype=np.int32)


def _maybe_log_outputs(epoch, output_batch, prompt_batch, sample_ids=None):
    if not (
        args.track
        and args.log_output_every > 0
        and epoch % args.log_output_every == 0
        and jax.process_index() == 0
    ):
        return
    sample_count = min(8, args.total_parallel_generations)
    if USE_SHARD_MAP:
        local_gen = np.asarray(output_batch.addressable_shards[0].data)[:sample_count]
        local_prompts = np.asarray(prompt_batch.addressable_shards[0].data)[:sample_count]
    else:
        local_gen = np.asarray(output_batch)[:sample_count]
        local_prompts = np.asarray(prompt_batch)[:sample_count]
    local_ids = (
        np.arange(local_gen.shape[0], dtype=np.int32)
        if sample_ids is None
        else np.asarray(sample_ids)[: local_gen.shape[0]]
    )
    rows = [
        [
            epoch,
            int(local_ids[i]),
            safe_decode(local_prompts[i], tokenizer),
            safe_decode(local_gen[i], tokenizer),
        ]
        for i in range(local_gen.shape[0])
    ]
    wandb.log(
        {"text_samples": wandb.Table(
            columns=["epoch", "sample_id", "prompt", "generation"], rows=rows
        )},
        step=epoch,
    )
    epoch_dir = run_out_dir / f"epoch_{epoch:05d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    with open(
        epoch_dir / f"outputs_rank{args.proc_id}.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "global_idx", "prompt", "generation"])
        writer.writerows(rows)


def single_epoch(
    noiser_params,
    params,
    true_train_fitness_sum,
    epoch,
):
    validation_score = None
    hellaswag_score = None
    if epoch % args.validate_every == 0:
        print("VALIDATION")
        validation_score = validate(params, epoch)
        print("VALIDATION SCORE=", validation_score)
        if hellaswag_validate is not None:
            print("HELLASWAG VALIDATION")
            hellaswag_score = hellaswag_validate(params, epoch)
            print("HELLASWAG SCORE=", hellaswag_score)

    start_time = time.time()
    train_indices = _epoch_train_indices(epoch)
    if args.random_train_prompts and epoch % args.validate_every == 0:
        print(f"Epoch {epoch} train indices: {train_indices.tolist()}")
    unique_prompts_np = np.asarray(
        Task.get_input(jnp.asarray(train_indices, dtype=jnp.int32)), dtype=np.int32
    )
    prompt_processing_time = time.time() - start_time

    preview_time = 0.0
    prediction_time = 0.0
    gather_time = 0.0
    predictive_stats = {}

    if PREDICTIVE_MODE:
        virtual_pairs_per_prompt = args.virtual_generations_per_prompt // 2
        total_virtual_pairs = args.virtual_total_parallel_generations // 2
        audit_probability = 1.0 / args.predictive_virtual_factor
        frozen_reward_scale = predictive_predictor.reward_scale

        # Selection happens before either kernel and never depends on predictions.
        audited_pair_ids = sample_stratified_audit_pairs(
            num_prompts=args.prompts_per_epoch,
            physical_members_per_prompt=args.generations_per_prompt,
            virtual_factor=args.predictive_virtual_factor,
            seed=args.predictive_audit_seed,
            epoch=epoch,
        )
        audited_member_ids = pair_ids_to_member_ids(audited_pair_ids)
        audited_pair_prompt_slots = audited_pair_ids // virtual_pairs_per_prompt
        audited_member_prompt_slots = np.repeat(audited_pair_prompt_slots, 2)
        audited_prompts = unique_prompts_np[audited_member_prompt_slots]
        unique_prompt_lengths = _zero_padded_prompt_lengths(unique_prompts_np)
        if np.any(unique_prompt_lengths > preview_prompt_width):
            raise ValueError("encountered a prompt longer than the compiled preview width")
        audited_dataset_indices = np.asarray(train_indices, dtype=np.int32)[
            audited_member_prompt_slots
        ]

        if epoch == 0:
            print("generating audited full rollouts")
        start_time = time.time()
        output_batch = jax.block_until_ready(
            generate_batch(
                noiser_params,
                params,
                jnp.asarray(audited_prompts),
                jnp.asarray(audited_member_ids),
                epoch,
            )
        )
        token_generation_time = time.time() - start_time
        _maybe_log_outputs(
            epoch, output_batch, audited_prompts, audited_member_ids
        )

        start_time = time.time()
        observed_scores = np.asarray(
            Task.get_batch_fitness(
                jax.device_put(
                    audited_dataset_indices, jax.local_devices(backend="cpu")[0]
                ),
                jax.device_put(
                    output_batch, jax.local_devices(backend="cpu")[0]
                ),
            ),
            dtype=np.float32,
        )
        fitness_time = time.time() - start_time

        start_time = time.time()
        pair_features = np.empty((total_virtual_pairs, feature_dim), dtype=np.float32)
        # Every pair uses this exact same feature kernel.  In particular, the
        # feature/prediction cannot reveal whether a pair was selected for the
        # audit, which is required by the HT unbiasedness argument.
        preview_pair_ids = np.arange(total_virtual_pairs, dtype=np.int32)
        microbatch = args.predictive_preview_microbatch_pairs
        for offset in range(0, preview_pair_ids.size, microbatch):
            pair_ids = preview_pair_ids[offset : offset + microbatch]
            valid_count = pair_ids.size
            if valid_count < microbatch:
                pair_ids = np.pad(pair_ids, (0, microbatch - valid_count), mode="edge")
            prompt_slots = pair_ids // virtual_pairs_per_prompt
            preview_prompts = unique_prompts_np[
                prompt_slots, :preview_prompt_width
            ]
            features = jax.block_until_ready(
                preview_feature_batch(
                    noiser_params,
                    params,
                    jnp.asarray(preview_prompts),
                    jnp.asarray(unique_prompt_lengths[prompt_slots]),
                    jnp.asarray(pair_ids),
                    epoch,
                )
            )
            pair_features[pair_ids[:valid_count]] = np.asarray(
                features[:valid_count], dtype=np.float32
            )
        preview_time = time.time() - start_time

        if args.predictive_prompt_center:
            pair_features = prompt_center_features(
                pair_features,
                num_prompts=args.prompts_per_epoch,
                pairs_per_prompt=virtual_pairs_per_prompt,
            )

        start_time = time.time()
        predictor_ready_current = predictive_predictor.ready
        (
            candidate_pair_differences,
            uncalibrated_pair_differences,
        ) = predictive_predictor.predict_with_uncalibrated(pair_features)
        predictor_enabled_current = (
            args.predictive_use_predictions
            and predictor_ready_current
            and predictive_predictor.calibration_scale > 0.0
        )
        predicted_pair_differences = candidate_pair_differences.copy()
        if not predictor_enabled_current:
            predicted_pair_differences.fill(0.0)
        corrected_pair_differences, observed_pair_differences = (
            audit_correct_pair_differences(
                predicted_pair_differences,
                audited_pair_ids,
                observed_scores,
                audit_probability=audit_probability,
            )
        )
        output_scores = jnp.asarray(
            pair_differences_to_member_utilities(
                corrected_pair_differences,
                physical_population=args.total_parallel_generations,
                virtual_population=args.virtual_total_parallel_generations,
                reward_scale=frozen_reward_scale,
            )
        )
        prediction_time = time.time() - start_time
        reported_scores = observed_scores
    else:
        indices_np = np.repeat(
            np.asarray(train_indices, dtype=np.int32), args.generations_per_prompt
        )
        batch_prompts_np = np.repeat(
            unique_prompts_np, args.generations_per_prompt, axis=0
        )
        indices = shard_on_data(indices_np)
        batch_prompts = shard_on_data(batch_prompts_np)
        thread_idxes = (
            physical_thread_idxes
            if USE_SHARD_MAP
            else jnp.arange(args.total_parallel_generations, dtype=jnp.int32)
        )
        if epoch == 0:
            print("generating batch")
        start_time = time.time()
        output_batch = jax.block_until_ready(
            generate_batch(noiser_params, params, batch_prompts, thread_idxes, epoch)
        )
        token_generation_time = time.time() - start_time
        _maybe_log_outputs(epoch, output_batch, batch_prompts)

        start_time = time.time()
        if USE_SHARD_MAP:
            local_shards = [
                jax.device_put(
                    Task.get_batch_fitness(
                        jax.device_put(
                            index_shard.data, jax.local_devices(backend="cpu")[0]
                        ),
                        jax.device_put(
                            output_shard.data, jax.local_devices(backend="cpu")[0]
                        ),
                    ),
                    index_shard.device,
                )
                for index_shard, output_shard in zip(
                    indices.addressable_shards, output_batch.addressable_shards
                )
            ]
            local_fitness = jax.make_array_from_single_device_arrays(
                (args.total_parallel_generations,),
                NamedSharding(mesh, P("data")),
                local_shards,
            )
        else:
            local_fitness = jax.device_put(
                Task.get_batch_fitness(
                    jax.device_put(indices, jax.local_devices(backend="cpu")[0]),
                    jax.device_put(output_batch, jax.local_devices(backend="cpu")[0]),
                ),
                jax.local_devices()[0],
            )
        fitness_time = time.time() - start_time

        start_time = time.time()
        output_scores = (
            process_allgather(local_fitness, True) if USE_SHARD_MAP else local_fitness
        )
        if USE_SHARD_MAP:
            output_scores = jax.sharding.reshard(
                output_scores, NamedSharding(mesh, P("data"))
            )
        gather_time = time.time() - start_time
        reported_scores = np.asarray(jax.device_get(output_scores), dtype=np.float32)

    if epoch == 0:
        print("updating params")
    start_time = time.time()
    noiser_params, params, parameter_differences = jax.block_until_ready(
        do_update(noiser_params, params, output_scores, epoch)
    )
    parameter_update_time = time.time() - start_time

    if PREDICTIVE_MODE:
        audited_predictions = candidate_pair_differences[audited_pair_ids]
        audit_mse = float(
            np.mean((audited_predictions - observed_pair_differences) ** 2)
        )
        zero_mse = float(np.mean(observed_pair_differences**2))
        residual_ratio = (
            audit_mse / zero_mse if zero_mse > 1e-12 else float("nan")
        )
        if (
            observed_pair_differences.size > 1
            and np.std(audited_predictions) > 1e-8
            and np.std(observed_pair_differences) > 1e-8
        ):
            audit_correlation = float(
                np.corrcoef(audited_predictions, observed_pair_differences)[0, 1]
            )
        else:
            audit_correlation = 0.0
        nonzero = observed_pair_differences != 0.0
        nonzero_count = int(np.count_nonzero(nonzero))
        sign_accuracy = float(
            np.mean(
                np.sign(audited_predictions[nonzero])
                == np.sign(observed_pair_differences[nonzero])
            )
        ) if np.any(nonzero) else 0.0

        # This update occurs strictly after the model update, so current labels
        # can only affect the next ES iteration.
        start_time = time.time()
        predictive_predictor.update(
            pair_features[audited_pair_ids],
            observed_pair_differences,
            rms_features=pair_features,
            evaluated_predictions=(
                audited_predictions if predictor_ready_current else None
            ),
            evaluated_uncalibrated_predictions=(
                uncalibrated_pair_differences[audited_pair_ids]
                if predictor_ready_current
                else None
            ),
        )
        predictive_predictor.update_reward_scale(observed_scores)
        predictor_fit_time = time.time() - start_time
        rolling_residual_ratio = predictive_predictor.prequential_residual_ratio
        predictor_enabled_next = bool(
            args.predictive_use_predictions
            and predictive_predictor.ready
            and predictive_predictor.calibration_scale > 0.0
        )
        predictive_stats = {
            "physical_population": args.total_parallel_generations,
            "virtual_population": args.virtual_total_parallel_generations,
            "audit_probability": audit_probability,
            "predictor_reward_scale": frozen_reward_scale,
            "predictor_reward_scale_next": predictive_predictor.reward_scale,
            "audited_pairs": audited_pair_ids.size,
            "preview_pairs_evaluated": preview_pair_ids.size,
            "preview_only_pairs": preview_pair_ids.size - audited_pair_ids.size,
            "preview_time": preview_time,
            "prediction_time": prediction_time,
            "predictor_fit_time": predictor_fit_time,
            "predictor_observations": predictive_predictor.total_observations,
            "predictor_effective_observations": predictive_predictor.effective_observations,
            "predictor_ready_current": float(predictor_ready_current),
            "predictor_ready_next": float(predictive_predictor.ready),
            "predictor_enabled_current": float(predictor_enabled_current),
            "predictor_enabled_next": float(predictor_enabled_next),
            "predictor_candidate_rms": float(
                np.sqrt(np.mean(candidate_pair_differences**2))
            ),
            "predictor_used_rms": float(
                np.sqrt(np.mean(predicted_pair_differences**2))
            ),
            "predictor_audit_mse": audit_mse,
            "predictor_zero_mse": zero_mse,
            "predictor_residual_ratio": residual_ratio,
            "predictor_audit_r2_vs_zero": 1.0 - residual_ratio,
            "predictor_rolling_residual_ratio": rolling_residual_ratio,
            "predictor_calibration_scale": predictive_predictor.calibration_scale,
            "predictor_calibration_slope": predictive_predictor.calibration_slope,
            "predictor_calibration_confidence": predictive_predictor.calibration_confidence,
            "predictor_calibration_observations": predictive_predictor.calibration_observations,
            "predictor_audit_correlation": audit_correlation,
            "predictor_nonzero_labels": nonzero_count,
            "predictor_nonzero_label_fraction": nonzero_count
            / observed_pair_differences.size,
            "predictor_nonzero_sign_accuracy": sign_accuracy,
            "predictor_corrected_rms": float(
                np.sqrt(np.mean(corrected_pair_differences**2))
            ),
            "preview_feature_zero_fraction": float(np.mean(pair_features == 0.0)),
            "predictor_feature_rms_min": float(predictive_predictor.feature_rms.min()),
            "predictor_feature_rms_max": float(predictive_predictor.feature_rms.max()),
        }
        print(
            f"Epoch {epoch} predictive audit: labels={audited_pair_ids.size}, "
            f"total_labels={predictive_predictor.total_observations}, "
            f"residual_ratio={residual_ratio:.3f}, "
            f"rolling_ratio={rolling_residual_ratio:.3f}, "
            f"calibration={predictive_predictor.calibration_scale:.3f}, "
            f"corr={audit_correlation:.3f}, "
            f"enabled_next={predictor_enabled_next}, "
            f"preview={preview_time:.2f}s, rollout={token_generation_time:.2f}s"
        )

    lora_updates = jax.tree.reduce(
        operator.add,
        jax.tree.map(
            lambda x, y: x if y == LORA else 0.0, parameter_differences, es_map
        ),
    ) / jax.tree.reduce(
        operator.add, jax.tree.map(lambda y: 1.0 if y == LORA else 0.0, es_map)
    )
    nonlora_updates = jax.tree.reduce(
        operator.add,
        jax.tree.map(
            lambda x, y: x if y == FULL else 0.0, parameter_differences, es_map
        ),
    ) / jax.tree.reduce(
        operator.add, jax.tree.map(lambda y: 1.0 if y == FULL else 0.0, es_map)
    )

    reported_scores_jax = jnp.asarray(reported_scores)
    true_train_fitness_sum += float(np.sum(reported_scores))
    stats = {
        "avg_fitness": jnp.mean(reported_scores_jax),
        "std_fitness": jnp.std(reported_scores_jax),
        "max_fitness": jnp.max(reported_scores_jax),
        "min_fitness": jnp.min(reported_scores_jax),
        "median_fitness": jnp.median(reported_scores_jax),
        "lora_updates": lora_updates,
        "nonlora_updates": nonlora_updates,
        "prompt_preproc_time": prompt_processing_time,
        "token_gen_time": token_generation_time,
        "fitness_time": fitness_time,
        "gather_time": gather_time,
        "update_time": parameter_update_time,
        "true_train_avg_fitness": true_train_fitness_sum
        / ((epoch + 1) * args.total_parallel_generations),
    }
    stats.update(predictive_stats)

    if validation_score is not None:
        stats["validation_score"] = validation_score
        elapsed = time.time() - run_start_time
        with open(validation_csv_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{float(validation_score)},{elapsed:.3f}\n")
    if hellaswag_score is not None:
        stats["hellaswag_validation_score"] = hellaswag_score
        elapsed = time.time() - run_start_time
        with open(hellaswag_csv_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{float(hellaswag_score)},{elapsed:.3f}\n")

    with open(fitness_csv_path, "a", encoding="utf-8") as f:
        f.write(f"{epoch},{float(jnp.mean(reported_scores_jax))}\n")
    if PREDICTIVE_MODE:
        with open(predictive_csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{predictive_stats['predictor_observations']},"
                f"{predictive_stats['predictor_audit_mse']:.8f},"
                f"{predictive_stats['predictor_zero_mse']:.8f},"
                f"{predictive_stats['predictor_residual_ratio']:.8f},"
                f"{predictive_stats['predictor_rolling_residual_ratio']:.8f},"
                f"{predictive_stats['predictor_calibration_scale']:.8f},"
                f"{predictive_stats['predictor_calibration_slope']:.8f},"
                f"{predictive_stats['predictor_calibration_confidence']:.8f},"
                f"{predictive_stats['predictor_calibration_observations']},"
                f"{predictive_stats['predictor_audit_correlation']:.8f},"
                f"{predictive_stats['predictor_nonzero_labels']},"
                f"{int(predictive_stats['predictor_enabled_current'])},"
                f"{int(predictive_stats['predictor_enabled_next'])},"
                f"{predictive_stats['predictor_reward_scale']:.8f},"
                f"{predictive_stats['predictor_reward_scale_next']:.8f},"
                f"{preview_time:.6f},{token_generation_time:.6f},"
                f"{prediction_time:.6f},{predictor_fit_time:.6f},"
                f"{parameter_update_time:.6f}\n"
            )

    if args.track and jax.process_index() == 0:
        run.log(stats)
    else:
        print(
            f"Mean fitness: {jnp.mean(reported_scores_jax)}; "
            f"std fitness: {jnp.std(reported_scores_jax)}; "
            f"max fitness: {jnp.max(reported_scores_jax)}; "
            f"min fitness: {jnp.min(reported_scores_jax)}; "
            f"median fitness: {jnp.median(reported_scores_jax)}"
        )
        print("mean parameter diffs")
        print("Lora modules:", lora_updates)
        print("Full modules:", nonlora_updates)
        print("Stats:")
        for key, value in stats.items():
            print(f"\t{key}: {value}")

    return noiser_params, params, true_train_fitness_sum

with open(validation_csv_path, "w", encoding="utf-8") as f:
    f.write("epoch,validation_score,time_seconds\n")
if PREDICTIVE_MODE:
    with open(predictive_csv_path, "w", encoding="utf-8") as f:
        f.write(
            "epoch,predictor_observations,audit_mse,zero_mse,residual_ratio,"
            "rolling_residual_ratio,calibration_scale,calibration_slope,"
            "calibration_confidence,calibration_observations,"
            "audit_correlation,nonzero_labels,predictor_enabled_current,"
            "predictor_enabled_next,reward_scale,reward_scale_next,preview_time,"
            "rollout_time,prediction_time,predictor_fit_time,update_time\n"
        )
if hellaswag_validate is not None:
    with open(hellaswag_csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,validation_score,time_seconds\n")

run_start_time = time.time()


def _effective_time_budget_seconds() -> Optional[float]:
    """Runtime budget override via env or a file in the run output directory."""
    env_val = os.environ.get("HYPERSCALEES_TIME_BUDGET_SECONDS")
    if env_val is not None:
        try:
            return float(env_val)
        except ValueError:
            pass
    override_path = run_out_dir / "time_budget_override.txt"
    if override_path.exists():
        try:
            return float(override_path.read_text(encoding="utf-8").strip())
        except ValueError:
            pass
    return args.time_budget_seconds


for epoch in tqdm.trange(args.num_epochs):
    budget = _effective_time_budget_seconds()
    if budget is not None and (time.time() - run_start_time) >= budget:
        print(f"Time budget ({budget}s) reached before epoch {epoch}. Stopping.")
        break
    noiser_params, params, true_train_fitness_sum = single_epoch(
        noiser_params,
        params,
        true_train_fitness_sum,
        epoch,
    )
    budget = _effective_time_budget_seconds()
    if budget is not None and (time.time() - run_start_time) >= budget:
        print(f"Time budget ({budget}s) reached after epoch {epoch}. Stopping.")
        break

if validation_csv_path.exists() and validation_csv_path.stat().st_size > len("epoch,validation_score,time_seconds\n"):
    from .plot_figure_4b import plot_figure_4b

    plot_figure_4b(validation_csv_path, figure_4b_path, model=args.model_choice)
    print(f"Saved validation log: {validation_csv_path}")
    print(f"Saved figure: {figure_4b_path}")
    print(f"Saved figure: {figure_4b_path.with_suffix('.pdf')}")
else:
    print(f"No validation points logged (validate_every={args.validate_every}).")
    print(f"Training fitness log: {fitness_csv_path}")

if args.track:
    run.finish()
