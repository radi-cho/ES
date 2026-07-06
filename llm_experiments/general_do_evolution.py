import os
import sys
import csv
import json
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
    build_validate,
    build_train_eval,
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
    eval_train_every: Optional[int] = None
    parallel_validations: int = 128
    validation_iterations: int = 10

    task: Literal[tuple(all_tasks.keys())] = "fastzero"
    noiser: Literal[tuple(all_noisers.keys())] = "eggroll"

    wandb_mode: Literal["online", "offline"] = "online"
    wandb_project: str = "HyperscaleExp"
    wandb_name: str = "full"
    track: bool = False

    generations_per_prompt: int = 8

    # Used only by diag_eggroll; the baseline path below is unchanged.
    diag_geometry_lr: float = 0.02
    diag_geometry_ema_decay: float = 0.9
    diag_geometry_warmup_pairs: int = 64
    diag_geometry_update_every: int = 1
    diag_geometry_condition_cap: float = 2.0
    diag_geometry_utility_clip: float = 3.0

    # Used only by product_space_eggroll; EGGROLL and diag_eggroll keep their
    # existing initialization paths below.
    product_space_rank: int = 8
    product_space_scout_pairs: int = 2
    product_space_warmup_pairs: int = 256
    product_space_geometry_lr: float = 0.02
    product_space_geometry_ema_decay: float = 0.9
    product_space_geometry_update_every: int = 1
    product_space_control_variate: bool = True

    train_dataset_size: Optional[int] = None
    val_dataset_size: Optional[int] = None
    time_budget_seconds: Optional[float] = None
    random_train_prompts: bool = False
    train_phased_subset_sizes: Optional[str] = None
    train_phased_durations_seconds: Optional[str] = None
    train_phased_split_seed: Optional[int] = None
    train_phased_adaptive: bool = False
    train_phased_adaptive_pool_threshold: float = 0.90
    train_phased_adaptive_val_drop: float = 0.03
    train_phased_adaptive_early_stop_val_drop: float = 0.03
    train_phased_adaptive_min_seconds: float = 600.0
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
print("full number of generations is", args.total_parallel_generations)

RWKV, full_params, tokenizer = get_model(args.model_choice, rwkv_type=args.rwkv_type, verbose=True, dtype=args.dtype)
legacy_tokenizer = LegacyWorldTokenizer() if args.model_choice[0] == "7" else tokenizer

config, params, scan_map, es_map = full_params

args.prompts_per_epoch = args.total_parallel_generations // args.generations_per_prompt

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

_train_phased_subsets: Optional[list[np.ndarray]] = None
_train_phased_durations: Optional[list[float]] = None
_train_phased_active_idx: int = -1
_train_phased_adaptive_idx: int = 0
_train_phased_phase_start_seconds: float = 0.0
_train_phased_phase_peak_val: float = float("-inf")
_train_phased_global_best_val: float = float("-inf")
_train_phased_val_eval_history: list[float] = []
_adaptive_stop_training: bool = False


def _init_train_phased_subsets(run_dir: Path) -> None:
    global _train_phased_subsets, _train_phased_durations, _train_phased_active_idx
    if args.train_phased_subset_sizes is None:
        _train_phased_subsets = None
        _train_phased_durations = None
        return

    sizes = [int(x.strip()) for x in args.train_phased_subset_sizes.split(",") if x.strip()]
    if not sizes:
        raise ValueError("train_phased_subset_sizes must list at least one subset size")
    if sum(sizes) != len(Task):
        raise ValueError(
            f"train_phased_subset_sizes ({sizes}) must sum to train dataset size ({len(Task)})"
        )
    if args.train_phased_durations_seconds is None:
        raise ValueError("train_phased_durations_seconds is required with train_phased_subset_sizes")
    durations = [
        float(x.strip())
        for x in args.train_phased_durations_seconds.split(",")
        if x.strip()
    ]
    if len(durations) != len(sizes):
        raise ValueError(
            f"train_phased_durations_seconds ({durations}) must match "
            f"train_phased_subset_sizes ({sizes})"
        )

    split_seed = args.train_phased_split_seed if args.train_phased_split_seed is not None else args.seed
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(len(Task)).astype(np.int32)
    subsets: list[np.ndarray] = []
    offset = 0
    for size in sizes:
        subsets.append(perm[offset : offset + size])
        offset += size

    _train_phased_subsets = subsets
    _train_phased_durations = durations
    _train_phased_active_idx = -1

    mode = "adaptive" if args.train_phased_adaptive else "fixed wall-clock"
    print(
        f"Phased train pools ({mode}): {len(subsets)} phases, "
        f"durations={durations}s, split_seed={split_seed}"
    )
    if args.train_phased_adaptive:
        print(
            f"  Adaptive switch: pool>={args.train_phased_adaptive_pool_threshold}, "
            f"val drop>={args.train_phased_adaptive_val_drop}, "
            f"min_phase={args.train_phased_adaptive_min_seconds}s, "
            f"early_stop_val_drop={args.train_phased_adaptive_early_stop_val_drop}"
        )
    for phase_idx, (subset, duration) in enumerate(zip(subsets, durations)):
        cap = f"max {duration}s" if args.train_phased_adaptive else f"{duration}s"
        print(
            f"  Phase {phase_idx + 1}: {len(subset)} examples for {cap}, "
            f"indices={subset.tolist()}"
        )

    phase_meta = {
        "split_seed": split_seed,
        "subset_sizes": sizes,
        "durations_seconds": durations,
        "adaptive": args.train_phased_adaptive,
        "adaptive_pool_threshold": args.train_phased_adaptive_pool_threshold,
        "adaptive_val_drop": args.train_phased_adaptive_val_drop,
        "adaptive_early_stop_val_drop": args.train_phased_adaptive_early_stop_val_drop,
        "adaptive_min_seconds": args.train_phased_adaptive_min_seconds,
        "subsets": [subset.tolist() for subset in subsets],
    }
    phase_path = run_dir / "train_phased_subsets.json"
    phase_path.write_text(json.dumps(phase_meta, indent=2), encoding="utf-8")
    print(f"Saved phased train split: {phase_path}")


def _advance_phased_pool(elapsed_seconds: float, reason: str) -> None:
    global _train_phased_adaptive_idx, _train_phased_phase_start_seconds
    global _train_phased_phase_peak_val, _train_phased_val_eval_history
    assert _train_phased_subsets is not None
    if _train_phased_adaptive_idx >= len(_train_phased_subsets) - 1:
        return
    _train_phased_adaptive_idx += 1
    _train_phased_phase_start_seconds = elapsed_seconds
    _train_phased_phase_peak_val = float("-inf")
    _train_phased_val_eval_history = []
    pool = _train_phased_subsets[_train_phased_adaptive_idx]
    print(
        f"Adaptive phase switch ({reason}) -> "
        f"phase {_train_phased_adaptive_idx + 1}/{len(_train_phased_subsets)}: "
        f"{len(pool)} examples, indices={pool.tolist()}"
    )


def _maybe_force_adaptive_phase_cap(elapsed_seconds: float) -> None:
    """Force-advance when the current phase exceeds its max duration cap."""
    assert _train_phased_subsets is not None and _train_phased_durations is not None
    while _train_phased_adaptive_idx < len(_train_phased_subsets) - 1:
        phase_elapsed = elapsed_seconds - _train_phased_phase_start_seconds
        if phase_elapsed < _train_phased_durations[_train_phased_adaptive_idx]:
            break
        _advance_phased_pool(elapsed_seconds, reason="max_phase_duration")


def _check_adaptive_phased_switch(
    elapsed_seconds: float,
    validation_score: Optional[float],
    active_pool_score: Optional[float],
) -> None:
    global _train_phased_global_best_val, _train_phased_phase_peak_val
    global _train_phased_val_eval_history, _adaptive_stop_training
    if not args.train_phased_adaptive or validation_score is None or active_pool_score is None:
        return

    val = float(validation_score)
    pool_score = float(active_pool_score)
    _train_phased_global_best_val = max(_train_phased_global_best_val, val)
    _train_phased_phase_peak_val = max(_train_phased_phase_peak_val, val)
    _train_phased_val_eval_history.append(val)

    if (
        len(_train_phased_val_eval_history) >= 2
        and _train_phased_global_best_val > 0.0
        and all(
            v <= _train_phased_global_best_val - args.train_phased_adaptive_early_stop_val_drop
            for v in _train_phased_val_eval_history[-2:]
        )
    ):
        _adaptive_stop_training = True
        print(
            "Adaptive early stop: validation at or below "
            f"{_train_phased_global_best_val - args.train_phased_adaptive_early_stop_val_drop:.4f} "
            "for 2 consecutive evals"
        )
        return

    if _train_phased_adaptive_idx >= len(_train_phased_subsets) - 1:
        return

    phase_elapsed = elapsed_seconds - _train_phased_phase_start_seconds
    if phase_elapsed < args.train_phased_adaptive_min_seconds:
        return

    pool_learned = pool_score >= args.train_phased_adaptive_pool_threshold
    val_drop = val <= _train_phased_phase_peak_val - args.train_phased_adaptive_val_drop
    no_improve_twice = (
        len(_train_phased_val_eval_history) >= 3
        and _train_phased_val_eval_history[-1] <= _train_phased_val_eval_history[-2]
        and _train_phased_val_eval_history[-2] <= _train_phased_val_eval_history[-3]
    )
    if pool_learned and (val_drop or no_improve_twice):
        trigger = "val_drop" if val_drop else "val_stagnation"
        _advance_phased_pool(elapsed_seconds, reason=f"pool>={pool_score:.3f}, {trigger}")


def _active_phased_pool(elapsed_seconds: float) -> tuple[int, np.ndarray]:
    assert _train_phased_subsets is not None and _train_phased_durations is not None
    if args.train_phased_adaptive:
        _maybe_force_adaptive_phase_cap(elapsed_seconds)
        idx = min(_train_phased_adaptive_idx, len(_train_phased_subsets) - 1)
        return idx, _train_phased_subsets[idx]
    cumulative = 0.0
    for phase_idx, duration in enumerate(_train_phased_durations):
        cumulative += duration
        if elapsed_seconds < cumulative:
            return phase_idx, _train_phased_subsets[phase_idx]
    return len(_train_phased_subsets) - 1, _train_phased_subsets[-1]


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
if args.noiser == "diag_eggroll":
    frozen_noiser_params, noiser_params = NOISER.init_noiser(
        params,
        args.sigma,
        args.lr_scale,
        group_size=args.generations_per_prompt,
        freeze_nonlora=args.freeze_nonlora,
        noise_reuse=args.noise_reuse,
        es_map=es_map,
        diag_geometry_lr=args.diag_geometry_lr,
        diag_geometry_ema_decay=args.diag_geometry_ema_decay,
        diag_geometry_warmup_pairs=args.diag_geometry_warmup_pairs,
        diag_geometry_update_every=args.diag_geometry_update_every,
        diag_geometry_condition_cap=args.diag_geometry_condition_cap,
        diag_geometry_utility_clip=args.diag_geometry_utility_clip,
    )
elif args.noiser == "product_space_eggroll":
    frozen_noiser_params, noiser_params = NOISER.init_noiser(
        params,
        args.sigma,
        args.lr_scale,
        group_size=args.generations_per_prompt,
        freeze_nonlora=args.freeze_nonlora,
        noise_reuse=args.noise_reuse,
        es_map=es_map,
        product_space_rank=args.product_space_rank,
        product_space_scout_pairs=args.product_space_scout_pairs,
        product_space_warmup_pairs=args.product_space_warmup_pairs,
        product_space_geometry_lr=args.product_space_geometry_lr,
        product_space_geometry_ema_decay=args.product_space_geometry_ema_decay,
        product_space_geometry_update_every=args.product_space_geometry_update_every,
        product_space_control_variate=args.product_space_control_variate,
        product_space_seed=args.seed,
    )
else:
    frozen_noiser_params, noiser_params = NOISER.init_noiser(params, args.sigma, args.lr_scale, group_size=args.generations_per_prompt, freeze_nonlora=args.freeze_nonlora, noise_reuse=args.noise_reuse)
base_evo_keys = simple_es_tree_key(params, base_model_key, scan_map)


all_thread_idxes = shard_on_data(np.arange(args.total_parallel_generations))
global_indices = all_thread_idxes

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
        shard_on_data(np.zeros((args.total_parallel_generations, args.generation_length), dtype=np.int32)),
        all_thread_idxes,
        0,
    ).compile()
else:
    generate_batch = jax.jit(
        jax.vmap(_generate_thread, in_axes=(None, None, 0, 0, None))
    ).lower(
        noiser_params,
        params,
        jax.ShapeDtypeStruct(
            (args.total_parallel_generations, args.generation_length), jnp.dtype("int32")
        ),
        jnp.arange(args.total_parallel_generations, dtype=jnp.int32),
        0,
    ).compile()
print("Compile time", time.time() - start_time)
print("memory info")
print(generate_batch.memory_analysis())

validate = build_validate(RWKV, config, params, base_evo_keys, base_valid_key, tokenizer, legacy_tokenizer, args, args.temperature, suppress_eos_token=suppress_eos_token)

train_eval = None
if args.eval_train_every is not None and args.eval_train_every > 0:
    train_eval = build_train_eval(
        RWKV,
        config,
        params,
        base_evo_keys,
        base_gen_key,
        Task,
        args,
        NOISER=NOISER,
        temperature=args.temperature,
        suppress_eos_token=suppress_eos_token,
    )

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
        shard_on_data(np.zeros(args.total_parallel_generations, dtype=np.float32)),
        0,
    ).compile()
else:
    do_update = jax.jit(_do_update, donate_argnums=(0, 1)).lower(
        noiser_params, params, jnp.zeros(args.total_parallel_generations, dtype=jnp.float32), 0
    ).compile()
print("Compile time", time.time() - start_time)
print("memory info")
print(do_update.memory_analysis())

true_train_fitness_sum = 0.0

FULL = 0
LORA = 1

full_name = f"{args.task}_{args.noiser}_{args.wandb_name}_lr={args.lr_scale}_sigma={args.sigma:.2e}_bs={args.total_parallel_generations}"
if args.train_dataset_size is not None:
    full_name += f"_trainD={args.train_dataset_size}"
experiment_id = f"{full_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

base_out_dir = Path(args.output_directory) if args.output_directory else (Path.cwd() / "outputs")
run_out_dir = base_out_dir / f"{experiment_id}"
run_out_dir.mkdir(parents=True, exist_ok=True)

fitness_csv_path = run_out_dir / "fitness.csv"
validation_csv_path = run_out_dir / "validation.csv"
metrics_csv_path = run_out_dir / "metrics.csv"
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

_init_train_phased_subsets(run_out_dir)
if args.train_phased_subset_sizes is not None and not args.random_train_prompts:
    raise ValueError("train_phased_subset_sizes requires --random-train-prompts")
if args.train_phased_adaptive:
    if args.train_phased_subset_sizes is None:
        raise ValueError("train_phased_adaptive requires train_phased_subset_sizes")
    if args.eval_train_every is None or args.eval_train_every <= 0:
        raise ValueError("train_phased_adaptive requires --eval-train-every")
    if args.eval_train_every != args.validate_every:
        raise ValueError("train_phased_adaptive requires eval_train_every == validate_every")


def _epoch_train_indices(epoch: int, elapsed_seconds: float) -> np.ndarray:
    """Dataset row indices for this epoch's unique train prompts."""
    global _train_phased_active_idx

    if _train_phased_subsets is not None:
        phase_idx, pool = _active_phased_pool(elapsed_seconds)
        if phase_idx != _train_phased_active_idx:
            _train_phased_active_idx = phase_idx
            print(
                f"Train phase {phase_idx + 1}/{len(_train_phased_subsets)} active: "
                f"{len(pool)} examples, indices={pool.tolist()}"
            )
        if args.random_train_prompts:
            if args.prompts_per_epoch > len(pool):
                raise ValueError(
                    f"random_train_prompts needs prompts_per_epoch ({args.prompts_per_epoch}) "
                    f"<= active phased pool size ({len(pool)})"
                )
            rng = np.random.default_rng(args.seed + epoch)
            return rng.choice(pool, size=args.prompts_per_epoch, replace=False)
        start = epoch * args.prompts_per_epoch
        return pool[start : start + args.prompts_per_epoch]

    if args.random_train_prompts:
        rng = np.random.default_rng(args.seed + epoch)
        return rng.choice(len(Task), size=args.prompts_per_epoch, replace=False)
    start = epoch * args.prompts_per_epoch
    return np.arange(start, start + args.prompts_per_epoch, dtype=np.int32)


def _current_train_phase(elapsed_seconds: float) -> int:
    if _train_phased_subsets is None:
        return -1
    phase_idx, _ = _active_phased_pool(elapsed_seconds)
    return phase_idx


def _active_train_pool_indices(elapsed_seconds: float) -> np.ndarray:
    if _train_phased_subsets is None:
        return np.arange(len(Task), dtype=np.int32)
    _, pool = _active_phased_pool(elapsed_seconds)
    return pool


def _mean_tree_rms(tree) -> float:
    leaves = jax.tree.leaves(tree)
    if not leaves:
        return 0.0
    return float(jnp.mean(jnp.array([jnp.mean(x) for x in leaves])))


def single_epoch(noiser_params, params, true_train_fitness_sum, epoch, elapsed_seconds: float):
    validation_score = None
    train_eval_score = None
    active_pool_train_eval_score = None
    hellaswag_score = None
    if epoch % args.validate_every == 0:
        print("VALIDATION")
        validation_score = validate(params, epoch)
        print("VALIDATION SCORE=", validation_score)
    if train_eval is not None and epoch % args.eval_train_every == 0:
        active_pool = _active_train_pool_indices(elapsed_seconds)
        print("TRAIN EVAL (all 30)")
        train_eval_score = train_eval(params, epoch)
        print("TRAIN EVAL SCORE=", train_eval_score)
        print(f"TRAIN EVAL (active pool, n={len(active_pool)})")
        active_pool_train_eval_score = train_eval(params, epoch, active_pool)
        print("ACTIVE POOL TRAIN EVAL SCORE=", active_pool_train_eval_score)
        _check_adaptive_phased_switch(
            elapsed_seconds, validation_score, active_pool_train_eval_score
        )
    if epoch % args.validate_every == 0:
        if hellaswag_validate is not None:
            print("HELLASWAG VALIDATION")
            hellaswag_score = hellaswag_validate(params, epoch)
            print("HELLASWAG SCORE=", hellaswag_score)
    # print("CURRENT MEMORY start of epoch", jax.local_devices()[0].memory_stats())
    start_time = time.time()
    train_indices = _epoch_train_indices(epoch, elapsed_seconds)
    if args.random_train_prompts and epoch % args.validate_every == 0:
        print(f"Epoch {epoch} train indices: {train_indices.tolist()}")
    if USE_SHARD_MAP:
        unique_indices = jax.device_put(
            replicate_matrix(jnp.asarray(train_indices, dtype=jnp.int32)),
            NamedSharding(mesh, P("data")),
        )
        indices = jnp.repeat(unique_indices, args.generations_per_prompt, axis=0)
        unique_prompts_np = np.asarray(Task.get_input(jnp.asarray(train_indices, dtype=jnp.int32)))
        batch_prompts = shard_on_data(
            np.repeat(unique_prompts_np, args.generations_per_prompt, axis=0)
        )
    else:
        unique_indices = jnp.asarray(train_indices, dtype=jnp.int32)
        indices = jnp.repeat(unique_indices, args.generations_per_prompt, axis=0)
        unique_prompts = Task.get_input(unique_indices)
        batch_prompts = jnp.repeat(unique_prompts, args.generations_per_prompt, axis=0)
    prompt_processing_time = time.time() - start_time

    # print("CURRENT MEMORY start of batch", jax.local_devices()[0].memory_stats())
    start_time = time.time()
    if epoch == 0:
        print("generating batch")
    thread_idxes = all_thread_idxes if USE_SHARD_MAP else jnp.arange(args.total_parallel_generations, dtype=jnp.int32)
    output_batch = jax.block_until_ready(
        generate_batch(noiser_params, params, batch_prompts, thread_idxes, epoch)
    )
    token_generation_time = time.time() - start_time

    if (
        args.track
        and args.log_output_every > 0
        and (epoch % args.log_output_every == 0)
        and jax.process_index() == 0
    ):
        # Take a small sample from the first local shard to minimize overhead
        K = min(8, args.total_parallel_generations)

        if USE_SHARD_MAP:
            local_gen = np.array(output_batch.addressable_shards[0].data)[:K]
            local_prompts = np.array(batch_prompts.addressable_shards[0].data)[:K]
        else:
            local_gen = np.array(output_batch)[:K]
            local_prompts = np.array(batch_prompts)[:K]

        rows = []
        for i in range(local_gen.shape[0]):
            prompt_txt = safe_decode(local_prompts[i], tokenizer)
            gen_txt = safe_decode(local_gen[i], tokenizer)
            rows.append([epoch, i, prompt_txt, gen_txt])

        table = wandb.Table(columns=["epoch", "sample_id", "prompt", "generation"], rows=rows)
        wandb.log({"text_samples": table}, step=epoch)
        
        epoch_dir = run_out_dir / f"epoch_{epoch:05d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        csv_path = epoch_dir / f"outputs_rank{args.proc_id}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "global_idx", "prompt", "generation"])
            writer.writerows(rows)
    
    start_time = time.time()
    if epoch == 0:
        print("calculating fitness")
    # local_output_scores = jax.block_until_ready(Task.get_batch_fitness(indices, output_batch))
    if USE_SHARD_MAP:
        _local_fitness = [
            jax.device_put(
                Task.get_batch_fitness(
                    jax.device_put(shard1.data, jax.local_devices(backend="cpu")[0]),
                    jax.device_put(shard2.data, jax.local_devices(backend="cpu")[0]),
                ),
                shard1.device,
            )
            for shard1, shard2 in zip(indices.addressable_shards, output_batch.addressable_shards)
        ]
        local_fitness = jax.make_array_from_single_device_arrays(
            (args.total_parallel_generations,), NamedSharding(mesh, P("data")), _local_fitness
        )
    else:
        idx_cpu = jax.device_put(indices, jax.local_devices(backend="cpu")[0])
        out_cpu = jax.device_put(output_batch, jax.local_devices(backend="cpu")[0])
        local_fitness = jax.device_put(
            Task.get_batch_fitness(idx_cpu, out_cpu), jax.local_devices()[0]
        )

    fitness_time = time.time() - start_time

    # print("CURRENT MEMORY start of update", jax.local_devices()[0].memory_stats())
    start_time = time.time()
    if epoch == 0:
        print("gathering")
    output_scores = process_allgather(local_fitness, True) if USE_SHARD_MAP else local_fitness
    if USE_SHARD_MAP:
        output_scores = jax.sharding.reshard(output_scores, NamedSharding(mesh, P("data")))
    gather_time = time.time() - start_time


    start_time = time.time()
    if epoch == 0:
        print("updating params")
    noiser_params, params, parameter_differences = jax.block_until_ready(do_update(noiser_params, params, output_scores, epoch))
    parameter_update_time = time.time() - start_time

    # print("CURRENT MEMORY start of stats", jax.local_devices()[0].memory_stats())
    # parameter_differences = jax.tree.map(lambda x, y:jnp.mean(jnp.abs(x-y)), params, updated_params)
    lora_updates = jax.tree.reduce(operator.add, jax.tree.map(lambda x, y: x if y == LORA else 0.0, parameter_differences, es_map)) / jax.tree.reduce(operator.add, jax.tree.map(lambda y: 1.0 if y == LORA else 0.0, es_map))
    nonlora_updates = jax.tree.reduce(operator.add, jax.tree.map(lambda x, y: x if y == FULL else 0.0, parameter_differences, es_map)) / jax.tree.reduce(operator.add, jax.tree.map(lambda y: 1.0 if y == FULL else 0.0, es_map))
    total_update_rms = _mean_tree_rms(parameter_differences)

    # params = updated_params

    true_train_fitness_sum += jnp.sum(output_scores).item()

    stats = {
        "avg_fitness": jnp.mean(output_scores),
        "std_fitness": jnp.std(output_scores),
        "max_fitness": jnp.max(output_scores),
        "min_fitness": jnp.min(output_scores),
        "median_fitness": jnp.median(output_scores),
        "lora_updates": lora_updates,
        "nonlora_updates": nonlora_updates,
        "total_update_rms": total_update_rms,
        "train_phase": _current_train_phase(elapsed_seconds),
        # "total_lora_updates": total_lora_updates,
        # "total_nonlora_updates": total_nonlora_updates,
        "prompt_preproc_time": prompt_processing_time,
        "token_gen_time": token_generation_time,
        "fitness_time": fitness_time,
        "gather_time": gather_time,
        "update_time": parameter_update_time,
        "true_train_avg_fitness": true_train_fitness_sum / ((epoch + 1) * args.total_parallel_generations)
    }
    if args.noiser == "diag_eggroll":
        from hyperscalees.noiser.diag_eggroll import geometry_diagnostics

        stats.update(geometry_diagnostics(noiser_params))
    elif args.noiser == "product_space_eggroll":
        from hyperscalees.noiser.product_space_eggroll import geometry_diagnostics

        stats.update(geometry_diagnostics(noiser_params))

    if validation_score is not None:
        stats["validation_score"] = validation_score
        elapsed = time.time() - run_start_time
        with open(validation_csv_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{float(validation_score)},{elapsed:.3f}\n")
    if train_eval_score is not None:
        stats["train_eval_score"] = train_eval_score
    if active_pool_train_eval_score is not None:
        stats["active_pool_train_eval_score"] = active_pool_train_eval_score
    if args.eval_train_every is not None and args.eval_train_every > 0:
        elapsed = time.time() - run_start_time
        phase = _current_train_phase(elapsed_seconds)
        val_field = "" if validation_score is None else f"{float(validation_score):.9g}"
        train_field = "" if train_eval_score is None else f"{float(train_eval_score):.9g}"
        active_field = (
            "" if active_pool_train_eval_score is None else f"{float(active_pool_train_eval_score):.9g}"
        )
        with open(metrics_csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{elapsed:.3f},{phase},{val_field},{train_field},{active_field},"
                f"{float(jnp.mean(output_scores)):.9g},"
                f"{float(lora_updates):.9g},{float(nonlora_updates):.9g},{total_update_rms:.9g}\n"
            )
    if hellaswag_score is not None:
        stats["hellaswag_validation_score"] = hellaswag_score
        elapsed = time.time() - run_start_time
        with open(hellaswag_csv_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{float(hellaswag_score)},{elapsed:.3f}\n")

    with open(fitness_csv_path, "a", encoding="utf-8") as f:
        f.write(f"{epoch},{float(jnp.mean(output_scores))}\n")
    
    if args.track and jax.process_index() == 0:
        run.log(stats)
    else:
        print(f"Mean fitness: {jnp.mean(output_scores)}; std fitness: {jnp.std(output_scores)}; max fitness: {jnp.max(output_scores)}; min fitness: {jnp.min(output_scores)}; median fitness: {jnp.median(output_scores)}")
        print("mean parameter diffs")
        print("Lora modules:", lora_updates)
        print("Full modules:", nonlora_updates)
        print("Stats:")
        for k in stats:
            print(f"\t{k}: {stats[k]}")

    return noiser_params, params, true_train_fitness_sum

with open(validation_csv_path, "w", encoding="utf-8") as f:
    f.write("epoch,validation_score,time_seconds\n")
if args.eval_train_every is not None and args.eval_train_every > 0:
    with open(metrics_csv_path, "w", encoding="utf-8") as f:
        f.write(
            "epoch,time_seconds,train_phase,validation_score,train_eval_score,"
            "active_pool_train_eval_score,batch_fitness,lora_update_rms,nonlora_update_rms,total_update_rms\n"
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
        noiser_params, params, true_train_fitness_sum, epoch, time.time() - run_start_time
    )
    if _adaptive_stop_training:
        print(f"Adaptive policy stop after epoch {epoch}.")
        break
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
