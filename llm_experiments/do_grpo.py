import os
import csv
import jax
from huggingface_hub.constants import HF_HOME

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.88"

jax.config.update("jax_compilation_cache_dir", os.path.join(HF_HOME, "hyperscaleescomp"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jax.numpy as jnp

import optax

import numpy as np

from hyperscalees.noiser import all_noisers
from hyperscalees.models.llm.auto import get_model, models
from hyperscalees.models.llm.tokenizer import LegacyWorldTokenizer
from hyperscalees.models.common import simple_es_tree_key

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


from .utils import build_generate_thread, build_validate

import time

import gc

import tqdm

import operator

import wandb

@dataclass
class Args:
    seed: int = 0
    model_choice: Literal[tuple(models.keys())] =  "7g0.1B"
    output_directory: Optional[str] = "."
    wandb_directory: Optional[str] = "."

    rwkv_type: str = "AssociativeScanRWKV"
    dtype: Optional[str] = "bfloat16"

    parallel_generations_per_gpu: int = 128
    generation_length: int = 101
    thinking_length: int = 100
    answer_length: int = 100

    num_epochs: int = 100

    # lr_scale: float = 1.0
    lr: float = 5e-6
    sigma: float = 0.0#1e-3
    num_minibatches: int = 4
    clip_eps: float = 0.2
    train_temp: float = 0.6
    freeze_nonlora: bool = False
    rank: int = 32
    grpo_max_seq_len: int = 128

    validate_every: int = 10
    parallel_validations: int = 128
    validation_iterations: int = 10

    task: Literal[tuple(all_tasks.keys())] = "fastzero"

    dataset_size: Optional[int] = None
    dataset_seed: int = 42
    dataset_pool_size: Optional[int] = None
    dataset_subset_mode: Literal["prefix", "random"] = "prefix"
    dataset_full_epoch: bool = False
    dataset_resample_epoch: bool = False
    train_dataset_size: Optional[int] = None
    val_dataset_size: Optional[int] = None
    random_train_prompts: bool = False
    time_budget_seconds: Optional[float] = None
    max_time_hours: Optional[float] = None
    initial_eval_examples: int = 0
    debug_reward: bool = False
    debug_reward_samples: int = 5
    kl_samples: int = 8

    wandb_mode: Literal["online", "offline"] = "online"
    wandb_project: str = "HyperscaleExp"
    wandb_name: str = "full"
    tag: str = ""
    track: bool = False

    generations_per_prompt: int = 4

    coord_addr: Optional[str] = None
    num_procs: Optional[int] = None
    proc_id: Optional[int] = None


def _task_init_kwargs(args: Args, seed_override: Optional[int] = None) -> dict:
    if args.task == "countdown_chat":
        train_ds = (
            args.train_dataset_size
            if args.train_dataset_size is not None
            else (args.dataset_size if args.dataset_size is not None else 256)
        )
        val_holdout = args.val_dataset_size if args.val_dataset_size is not None else 256
        seed = args.seed if seed_override is None else seed_override
        return {
            "dataset_size": train_ds,
            "seed": seed,
            "val_holdout_size": val_holdout,
        }
    if args.dataset_size is None:
        return {}
    seed = args.dataset_seed if seed_override is None else seed_override
    kwargs = {
        "dataset_size": args.dataset_size,
        "seed": seed,
        "subset_mode": args.dataset_subset_mode,
    }
    if args.dataset_pool_size is not None:
        kwargs["pool_size"] = args.dataset_pool_size
    if args.debug_reward:
        kwargs["debug_reward"] = True
        kwargs["debug_reward_samples"] = args.debug_reward_samples
    return kwargs


defaults = Args()
profile = os.getenv("PROFILE", "default")
CONFIG_DIR = (Path(__file__).resolve().parents[1] / "configs").as_posix()

if profile != "default":
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        user_cfg = compose(config_name=profile)
    user_overrides = OmegaConf.to_container(user_cfg, resolve=True)
    for k, v in user_overrides.items():
        if hasattr(defaults, k) and v is not None:
            setattr(defaults, k, v)

args = tyro.cli(Args, default=defaults)
print()
print(f"Using config: {profile}")
print()
if args.model_choice.startswith("q35_") and args.rwkv_type == "BaseRWKV":
    args.rwkv_type = "Qwen35RWKV"
suppress_eos_token = 0 if args.model_choice[0] == "7" else None
args.generation_length = args.thinking_length + args.answer_length
if args.grpo_max_seq_len <= 0 or args.grpo_max_seq_len > args.generation_length:
    args.grpo_max_seq_len = min(args.answer_length + 32, args.generation_length)

print("starting distributed init")
if args.coord_addr is not None:
    jax.distributed.initialize(args.coord_addr, args.num_procs, args.proc_id)
else:
    print("NOT DISTRIBUTED CONTEXT")

master_key = jax.random.key(args.seed)

base_model_key = jax.random.fold_in(master_key, 0)
base_gen_key = jax.random.fold_in(master_key, 1)
base_valid_key = jax.random.fold_in(master_key, 2)

NOISER = all_noisers["eggroll"]

total_num_devices = len(jax.devices())
print("global devices", jax.devices())
print("local devices", jax.local_devices())
print("process id", jax.process_index())
args.proc_id = jax.process_index()
args.total_parallel_generations = total_num_devices * args.parallel_generations_per_gpu
if args.total_parallel_generations % args.num_minibatches != 0:
    raise ValueError(
        f"total_parallel_generations ({args.total_parallel_generations}) must be divisible by "
        f"num_minibatches ({args.num_minibatches})"
    )

USE_SHARD_MAP = total_num_devices > 1
mesh = jax.make_mesh((len(jax.devices()),), ("data",)) if USE_SHARD_MAP else None

print()
print("per-device generations is", args.parallel_generations_per_gpu)
print("full number of generations is", args.total_parallel_generations)

RWKV, full_params, tokenizer = get_model(args.model_choice, rwkv_type=args.rwkv_type, verbose=True, dtype=args.dtype)
legacy_tokenizer = LegacyWorldTokenizer() if args.model_choice[0] == "7" else tokenizer

config, params, scan_map, es_map = full_params

args.prompts_per_epoch = args.total_parallel_generations // args.generations_per_prompt

Task = all_tasks[args.task](
    tokenizer, legacy_tokenizer, args.generation_length, **_task_init_kwargs(args)
)
train_ds = args.train_dataset_size if args.train_dataset_size is not None else args.dataset_size
if args.task == "countdown_chat":
    val_holdout = args.val_dataset_size if args.val_dataset_size is not None else 256
    train_size = train_ds if train_ds is not None else len(Task)
    from hyperscalees.environments.llm_bandits import countdown_train_val_overlap
    overlap = countdown_train_val_overlap(train_size, val_holdout, train_seed=args.seed)
    print(f"Train dataset size: {len(Task)}")
    print(f"Countdown train∩val overlap: {overlap} (disjoint split, seed={42})")
elif args.dataset_size is not None:
    print(f"Training dataset size: {args.dataset_size} (seed={args.dataset_seed})")
if args.random_train_prompts:
    if args.prompts_per_epoch > len(Task):
        raise ValueError(
            f"random_train_prompts needs prompts_per_epoch ({args.prompts_per_epoch}) "
            f"<= train dataset size ({len(Task)})"
        )
    print(f"Random train prompt sampling: {args.prompts_per_epoch} unique examples per epoch")
if args.grpo_max_seq_len and args.grpo_max_seq_len > 0:
    print(f"GRPO policy loss uses last {args.grpo_max_seq_len} tokens of each trajectory")
print(f"GRPO rollout temperature: {args.train_temp}")

pad_token_id = tokenizer.pad_token_id() if hasattr(tokenizer, "pad_token_id") else 0

def replicate_matrix(x):
    if not USE_SHARD_MAP:
        return x
    return jax.make_array_from_single_device_arrays(
        x.shape, NamedSharding(mesh, P()), [jax.device_put(x, d) for d in jax.local_devices()]
    )


def _data_sharding(x):
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

FULL = 0
LORA = 1

solver = optax.adam(args.lr)
lora_mask = jax.tree.map(lambda y: y == 1, es_map)


def extract_lora_params(full_params):
    return jax.tree.map(lambda p, m: p if m == LORA else None, full_params, es_map)


def merge_lora_params(base_params, lora_params):
    frozen_base = jax.tree.map(jax.lax.stop_gradient, base_params)
    return jax.tree.map(
        lambda base, lora, m: lora if m == LORA else base,
        frozen_base,
        lora_params,
        es_map,
    )


if args.freeze_nonlora:
    print(f"GRPO training LoRA-only (rank={args.rank}, freeze_nonlora=True)")
    trainable_params = extract_lora_params(params)
    optimizer = solver.init(trainable_params)
else:
    trainable_params = params
    optimizer = solver.init(params)
frozen_noiser_params, noiser_params = NOISER.init_noiser(
    params,
    args.sigma,
    args.lr,
    group_size=args.generations_per_prompt,
    freeze_nonlora=args.freeze_nonlora,
    rank=args.rank,
)
base_evo_keys = simple_es_tree_key(params, base_model_key, scan_map)


global_indices = shard_on_data(np.arange(args.total_parallel_generations))
all_thread_idxes = global_indices

_generate_thread = build_generate_thread(
    RWKV,
    NOISER,
    frozen_noiser_params,
    config,
    base_evo_keys,
    base_gen_key,
    args.train_temp,
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
            check_rep=False,
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

validate = build_validate(
    RWKV,
    config,
    params,
    base_evo_keys,
    base_valid_key,
    tokenizer,
    legacy_tokenizer,
    args,
    0.0,
    NOISER=NOISER,
    suppress_eos_token=suppress_eos_token,
)
if args.initial_eval_examples > 0 and jax.process_index() == 0:
    from dataclasses import replace

    init_parallel = min(args.parallel_validations, args.initial_eval_examples)
    init_iters = int(np.ceil(args.initial_eval_examples / init_parallel))
    init_args = replace(
        args,
        parallel_validations=init_parallel,
        validation_iterations=init_iters,
    )
    initial_validate = build_validate(
        RWKV,
        config,
        params,
        base_evo_keys,
        base_valid_key,
        tokenizer,
        legacy_tokenizer,
        init_args,
        0.0,
        use_validation_set=False,
        NOISER=NOISER,
        suppress_eos_token=suppress_eos_token,
    )
    print("Initial evaluation (train set subset)")
    init_score = initial_validate(params, 0)
    print("Initial average reward:", float(init_score))


"""
Input tokens: [1, 2, 3, 0, 0, 0, 0, 0]
Gener tokens: [1, 2, 3, 7, 7, 7, 7, 7]


"""


def _truncate_grpo_sequence(is_input_token, tokens):
    max_len = args.grpo_max_seq_len
    if max_len is None or max_len <= 0 or tokens.shape[0] <= max_len:
        return is_input_token, tokens
    return is_input_token[-max_len:], tokens[-max_len:]


def _prompt_lens_from_batch(batch_prompts):
    is_pad = (batch_prompts == pad_token_id) | (batch_prompts == 0)
    first_pad = jnp.argmax(is_pad, axis=1)
    any_pad = jnp.any(is_pad, axis=1)
    return jnp.where(any_pad, first_pad, batch_prompts.shape[1])


def _is_input_token_from_prompt_lens(prompt_lens, generation_length):
    positions = jnp.arange(generation_length, dtype=jnp.int32)
    return positions[None, :] < prompt_lens[:, None]


def _grpo_logprob_len():
    max_len = args.grpo_max_seq_len
    if max_len is None or max_len <= 0 or args.generation_length <= max_len:
        return args.generation_length - 1
    return max_len - 1


def _forward_token_logprobs_impl(params, noiser_params, is_input_token, tokens):
    T = tokens.shape[0]
    token_padding = (16 - T % 16) % 16
    input_tokens = jnp.concatenate((tokens, jnp.zeros_like(tokens[:token_padding])))

    input_state = RWKV.default_state(params, config)
    pi, _ = RWKV.forward(
        NOISER,
        frozen_noiser_params,
        noiser_params,
        config,
        params,
        base_evo_keys,
        None,
        input_tokens,
        input_state,
    )
    return jax.nn.log_softmax(pi[: T - 1])[jnp.arange(T - 1), tokens[1:]]


forward_token_logprobs = jax.checkpoint(_forward_token_logprobs_impl)


def single_example_loss(params, old_logprobs, noiser_params, is_input_token, tokens, advantage):
    is_input_token, tokens = _truncate_grpo_sequence(is_input_token, tokens)
    seq_logprobs = tokens.shape[0] - 1
    old_logprobs = old_logprobs[-seq_logprobs:]
    pi_logprob = forward_token_logprobs(params, noiser_params, is_input_token, tokens)
    ratio = jnp.exp(pi_logprob - old_logprobs)
    token_loss = -jnp.minimum(
        ratio * advantage,
        jnp.clip(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * advantage,
    )
    return jnp.mean(jnp.where(is_input_token[1:], 0.0, token_loss))


def compute_old_logprobs(params, noiser_params, is_input_token, generation):
    is_input_token, generation = _truncate_grpo_sequence(is_input_token, generation)
    return forward_token_logprobs(params, noiser_params, is_input_token, generation)

def normalize_advantages(raw_scores, epoch):
    group_scores = raw_scores.reshape((-1, args.generations_per_prompt))
    mean = jnp.mean(group_scores, axis=-1, keepdims=True)
    std = jnp.std(group_scores, axis=-1, keepdims=True)
    normalized = (group_scores - mean) / (std + 1e-5)
    # If variance collapses, add tiny noise to break ties.
    use_raw = std < 1e-5
    if jnp.any(use_raw):
        key = jax.random.fold_in(jax.random.key(args.seed), epoch)
        noise = 1e-3 * jax.random.normal(key, shape=group_scores.shape)
        safe = jnp.where(use_raw, group_scores + noise, normalized)
    else:
        safe = normalized
    return safe.ravel()


def _single_example_update(
    optimizer,
    noiser_params,
    trainable_params,
    frozen_base_params,
    old_logprobs,
    is_input_token,
    generation,
    advantage,
):
    def example_loss(trainable):
        merged_params = (
            merge_lora_params(frozen_base_params, trainable)
            if args.freeze_nonlora
            else trainable
        )
        return single_example_loss(
            merged_params, old_logprobs, noiser_params, is_input_token, generation, advantage
        )

    loss, grad = jax.value_and_grad(example_loss)(trainable_params)
    updates, optimizer = solver.update(grad, optimizer, trainable_params)
    trainable_params = optax.apply_updates(trainable_params, updates)
    merged_params = (
        merge_lora_params(frozen_base_params, trainable_params)
        if args.freeze_nonlora
        else trainable_params
    )
    return optimizer, trainable_params, merged_params, loss


_seq_len = min(args.generation_length, args.grpo_max_seq_len or args.generation_length)
print()
print("Compiling old-logprob forward")
start_time = time.time()
compute_old_logprobs_jit = jax.jit(compute_old_logprobs).lower(
    params,
    noiser_params,
    jnp.zeros((_seq_len,), dtype=jnp.bool),
    jnp.zeros((_seq_len,), dtype=jnp.int32),
).compile()
print("Compile time", time.time() - start_time)

print("Compiling single-example GRPO update")
start_time = time.time()
single_example_update = jax.jit(_single_example_update).lower(
    optimizer,
    noiser_params,
    trainable_params,
    params,
    jnp.zeros((_grpo_logprob_len(),), dtype=jnp.float32),
    jnp.zeros((_seq_len,), dtype=jnp.bool),
    jnp.zeros((_seq_len,), dtype=jnp.int32),
    jnp.array(0.0, dtype=jnp.float32),
).compile()
print("Compile time", time.time() - start_time)
print("memory info")
print(single_example_update.memory_analysis())

true_train_fitness_sum = 0.0

ds_tag = ""
if args.train_dataset_size is not None:
    ds_tag = f"_trainD={args.train_dataset_size}"
elif args.dataset_size is not None:
    ds_tag = f"_ds{args.dataset_size}"
full_name = f"{args.task}_grpo_{args.wandb_name}{ds_tag}_lr={args.lr}_bs={args.total_parallel_generations}"
experiment_id = f"{full_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

base_out_dir = Path(args.output_directory) if args.output_directory else (Path.cwd() / "outputs")
run_out_dir = base_out_dir / f"{experiment_id}"
if jax.process_index() == 0:
    run_out_dir.mkdir(parents=True, exist_ok=True)

validation_csv_path = run_out_dir / "validation.csv"
fitness_csv_path = run_out_dir / "fitness.csv"

print("Run name", full_name)
print("Output directory:", run_out_dir)
if args.track:
    print("Tracking run ...")
    if args.wandb_mode == "offline":
        print("Initializing wandb in offline mode")
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
    if args.random_train_prompts:
        rng = np.random.default_rng(args.seed + epoch)
        return rng.choice(len(Task), size=args.prompts_per_epoch, replace=False)
    start = epoch * args.prompts_per_epoch
    return np.arange(start, start + args.prompts_per_epoch, dtype=np.int32)


def _epoch_batches(epoch: int):
    if args.random_train_prompts:
        yield _epoch_train_indices(epoch)
        return
    if args.dataset_size is None or not args.dataset_full_epoch:
        yield None
        return
    rng = np.random.default_rng(args.dataset_seed + epoch)
    order = rng.permutation(args.dataset_size)
    total = args.prompts_per_epoch
    num_batches = int(np.ceil(args.dataset_size / total))
    for batch_idx in range(num_batches):
        start = batch_idx * total
        end = start + total
        batch = order[start:end]
        if batch.size < total:
            pad = order[: total - batch.size]
            batch = np.concatenate([batch, pad])
        yield batch


def single_epoch(optimizer, noiser_params, params, trainable_params, true_train_fitness_sum, epoch):
    if args.dataset_resample_epoch and args.dataset_size is not None:
        Task = all_tasks[args.task](
            tokenizer,
            legacy_tokenizer,
            args.generation_length,
            **_task_init_kwargs(args, seed_override=args.dataset_seed + epoch),
        )
    else:
        Task = all_tasks[args.task](
            tokenizer,
            legacy_tokenizer,
            args.generation_length,
            **_task_init_kwargs(args),
        )
    if epoch % args.validate_every == 0:
        print("VALIDATION", flush=True)
        validation_score = validate(params, epoch)
        print("VALIDATION SCORE=", validation_score, flush=True)
        if jax.process_index() == 0:
            elapsed = time.time() - run_start_time
            with open(validation_csv_path, "a", encoding="utf-8") as f:
                f.write(f"{epoch},{float(validation_score)},{elapsed:.3f}\n")
                f.flush()
            print(f"Saved validation epoch {epoch} -> {validation_csv_path}", flush=True)
    else:
        validation_score = None
    prompt_processing_time = 0.0
    token_generation_time = 0.0
    fitness_time = 0.0
    gather_time = 0.0
    parameter_update_time = 0.0
    parameter_differences = None
    stats = None
    output_scores = None

    loss_sum = 0.0
    loss_count = 0
    kl_sum = 0.0
    kl_count = 0
    for batch_indices in _epoch_batches(epoch):
        if args.random_train_prompts and batch_indices is not None and epoch % args.validate_every == 0:
            print(f"Epoch {epoch} train indices: {batch_indices.tolist()}")
        start_time = time.time()
        if USE_SHARD_MAP:
            if batch_indices is None:
                unique_indices = (
                    jax.device_put(replicate_matrix(jnp.arange(args.prompts_per_epoch)), NamedSharding(mesh, P("data")))
                    + epoch * args.prompts_per_epoch
                )
                base_idx = epoch * args.prompts_per_epoch
                unique_prompts_np = np.stack(
                    [np.asarray(Task.get_input(base_idx + i)) for i in range(args.prompts_per_epoch)]
                )
            else:
                unique_indices = shard_on_data(np.asarray(batch_indices, dtype=np.int32))
                unique_prompts_np = np.stack(
                    [np.asarray(Task.get_input(i)) for i in batch_indices]
                )
            indices = jnp.repeat(unique_indices, args.generations_per_prompt, axis=0)
            batch_prompts = shard_on_data(
                np.repeat(unique_prompts_np, args.generations_per_prompt, axis=0)
            )
        else:
            if batch_indices is None:
                unique_indices = jnp.arange(args.prompts_per_epoch, dtype=jnp.int32) + epoch * args.prompts_per_epoch
            else:
                unique_indices = jnp.asarray(batch_indices, dtype=jnp.int32)
            indices = jnp.repeat(unique_indices, args.generations_per_prompt, axis=0)
            unique_prompts = Task.get_input(unique_indices)
            batch_prompts = jnp.repeat(unique_prompts, args.generations_per_prompt, axis=0)
        all_is_input_token = _is_input_token_from_prompt_lens(
            _prompt_lens_from_batch(batch_prompts), args.generation_length
        )
        prompt_processing_time += time.time() - start_time

        start_time = time.time()
        if epoch == 0:
            print("generating batch")
        thread_idxes = all_thread_idxes if USE_SHARD_MAP else jnp.arange(args.total_parallel_generations, dtype=jnp.int32)
        output_batch = jax.block_until_ready(
            generate_batch(noiser_params, params, batch_prompts, thread_idxes, epoch)
        )
        all_generations = output_batch
        token_generation_time += time.time() - start_time

        start_time = time.time()
        if epoch == 0:
            print("calculating fitness")
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
        fitness_time += time.time() - start_time

        start_time = time.time()
        if epoch == 0:
            print("gathering")
        output_scores = process_allgather(local_fitness, True) if USE_SHARD_MAP else local_fitness
        if USE_SHARD_MAP:
            output_scores = jax.sharding.reshard(output_scores, NamedSharding(mesh, P("data")))
        gather_time += time.time() - start_time

        start_time = time.time()
        if epoch == 0:
            print("updating params", flush=True)
        old_params = params
        frozen_base_params = params
        true_scores = normalize_advantages(output_scores, epoch)
        true_scores_host = np.asarray(jax.device_get(true_scores))
        n_updates = args.total_parallel_generations
        for i in range(n_updates):
            grpo_is_input, grpo_generation = _truncate_grpo_sequence(
                all_is_input_token[i], all_generations[i]
            )
            old_logprobs = jax.block_until_ready(
                compute_old_logprobs_jit(
                    frozen_base_params,
                    noiser_params,
                    grpo_is_input,
                    grpo_generation,
                )
            )
            optimizer, trainable_params, params, loss_val = jax.block_until_ready(
                single_example_update(
                    optimizer,
                    noiser_params,
                    trainable_params,
                    frozen_base_params,
                    old_logprobs,
                    grpo_is_input,
                    grpo_generation,
                    jnp.asarray(true_scores_host[i], dtype=jnp.float32),
                )
            )
            loss_sum += float(loss_val)
            loss_count += 1
            if (i + 1) % 8 == 0 or i + 1 == n_updates:
                print(f"  GRPO update {i + 1}/{n_updates}", flush=True)
        parameter_differences = jax.tree.map(
            lambda x, y: jnp.sqrt(jnp.mean((x - y) ** 2)), old_params, params
        )
        parameter_update_time += time.time() - start_time

        # print("CURRENT MEMORY start of stats", jax.local_devices()[0].memory_stats())
        # parameter_differences = jax.tree.map(lambda x, y:jnp.mean(jnp.abs(x-y)), params, updated_params)
        lora_updates = jax.tree.reduce(operator.add, jax.tree.map(lambda x, y: x if y == LORA else 0.0, parameter_differences, es_map)) / jax.tree.reduce(operator.add, jax.tree.map(lambda y: 1.0 if y == LORA else 0.0, es_map))
        nonlora_updates = jax.tree.reduce(operator.add, jax.tree.map(lambda x, y: x if y == FULL else 0.0, parameter_differences, es_map)) / jax.tree.reduce(operator.add, jax.tree.map(lambda y: 1.0 if y == FULL else 0.0, es_map))

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
            # "total_lora_updates": total_lora_updates,
            # "total_nonlora_updates": total_nonlora_updates,
            "prompt_preproc_time": prompt_processing_time,
            "token_gen_time": token_generation_time,
            "fitness_time": fitness_time,
            "gather_time": gather_time,
            "update_time": parameter_update_time,
            "true_train_avg_fitness": true_train_fitness_sum / ((epoch + 1) * args.total_parallel_generations)
        }
        if loss_count > 0:
            stats["loss"] = loss_sum / loss_count
        if kl_count > 0:
            stats["kl_divergence"] = kl_sum / kl_count

        if validation_score is not None:
            stats["validation_score"] = validation_score

        if jax.process_index() == 0:
            with open(fitness_csv_path, "a", encoding="utf-8") as f:
                f.write(f"{epoch},{float(jnp.mean(output_scores)):.12g}\n")
                f.flush()
            print(f"Saved fitness epoch {epoch} -> {fitness_csv_path}", flush=True)
        
        if args.track:
            run.log(stats)
        else:
            print(f"Mean fitness: {jnp.mean(output_scores)}; std fitness: {jnp.std(output_scores)}; max fitness: {jnp.max(output_scores)}; min fitness: {jnp.min(output_scores)}; median fitness: {jnp.median(output_scores)}")
            print("mean parameter diffs")
            print("Lora modules:", lora_updates)
            print("Full modules:", nonlora_updates)
            print("Stats:")
            for k in stats:
                print(f"\t{k}: {stats[k]}")

        del output_batch, all_generations, batch_prompts, all_is_input_token
        del local_fitness, output_scores, old_params, frozen_base_params, parameter_differences
        gc.collect()

    return optimizer, noiser_params, params, trainable_params, true_train_fitness_sum

if jax.process_index() == 0:
    with open(validation_csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,validation_score,time_seconds\n")
    with open(fitness_csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,avg_fitness\n")

run_start_time = time.time()


def _effective_time_budget_seconds() -> Optional[float]:
    if args.time_budget_seconds is not None:
        return args.time_budget_seconds
    if args.max_time_hours is not None:
        return args.max_time_hours * 3600.0
    return None


for epoch in tqdm.trange(args.num_epochs):
    budget = _effective_time_budget_seconds()
    if budget is not None and jax.process_index() == 0 and (time.time() - run_start_time) >= budget:
        print(f"Time budget ({budget}s) reached before epoch {epoch}. Stopping.")
        break
    optimizer, noiser_params, params, trainable_params, true_train_fitness_sum = single_epoch(
        optimizer, noiser_params, params, trainable_params, true_train_fitness_sum, epoch
    )
    budget = _effective_time_budget_seconds()
    if budget is not None and jax.process_index() == 0 and (time.time() - run_start_time) >= budget:
        print(f"Time budget ({budget}s) reached after epoch {epoch}. Stopping.")
        break
    gc.collect()

if jax.process_index() == 0 and validation_csv_path.exists() and validation_csv_path.stat().st_size > len("epoch,validation_score,time_seconds\n"):
    from .plot_figure_4b import plot_figure_4b

    figure_path = run_out_dir / "figure_4b.png"
    plot_figure_4b(
        validation_csv_path,
        figure_path,
        title=f"Countdown — {args.model_choice} (GRPO)",
        model=args.model_choice,
    )
    print(f"Saved validation log: {validation_csv_path}")
    print(f"Saved figure: {figure_path}")

if args.track:
    run.finish()
