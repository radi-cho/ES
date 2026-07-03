import jax
import jax.numpy as jnp
from typing import NamedTuple

import hyperscalees as hs
from hyperscalees.environments.llm_bandits import all_tasks, validation_tasks

import tqdm
import time

import numpy as np
from datasets import load_dataset

from jax.sharding import NamedSharding, PartitionSpec as P
from jax.experimental.multihost_utils import process_allgather

def as_named(x, mesh, spec):
    return jax.device_put(x, NamedSharding(mesh, spec))

def fold_in_helper(key, epoch, true_thread_idx):
    return jax.random.fold_in(jax.random.fold_in(key, epoch), true_thread_idx)

def safe_decode(tokens, tokenizer):
    try:
        stop_tokens = np.flatnonzero(tokens==0)
        if stop_tokens.size > 0:
            tokens = tokens[:stop_tokens[0]]
        return tokenizer.decode(tokens)
    except BaseException as e:
        return ""

def build_generate_thread(MODEL, NOISER, frozen_noiser_params, config, base_evo_keys, master_gen_key, temperature=1.0, for_shard_map=False, suppress_eos_token=0):

    def forward_and_sample(noiser_params, params, input_token, input_state, generation_key, iterinfo):
        print("compiling forward and sample")
        gen_key, _gen_key = jax.random.split(generation_key)
        generated_outs, generated_state = MODEL.forward(NOISER, frozen_noiser_params, noiser_params, config, params, base_evo_keys, iterinfo, input_token, input_state)
        logits = generated_outs[-1]
        if suppress_eos_token is not None:
            logits = logits.at[suppress_eos_token].set(-jnp.inf)
        if temperature != 0.0:
            sampled_tok = jax.random.categorical(_gen_key, logits / temperature)
        else:
            sampled_tok = jnp.argmax(logits)
        return sampled_tok, generated_state, gen_key
    
    def generate_thread(noiser_params, params, prompt, thread_idx, epoch_num):
        print("Compiling generate_batch")

        start_gen_key = fold_in_helper(master_gen_key, epoch_num, thread_idx)

        iterinfo = (epoch_num, thread_idx)
        def inner_scan(carry, input_token):
            tok, state, gen_key = carry
            true_input = jnp.where(input_token == 0, tok, input_token)
            tok, state, gen_key = forward_and_sample(noiser_params, params, true_input, state, gen_key, iterinfo)
            return (tok, state, gen_key), true_input

        if for_shard_map:
            init_token = jax.lax.pcast(0, "data", to="varying")
            init_state = jax.lax.pcast(MODEL.default_state(params, config), "data", to="varying")
        else:
            init_token = jnp.array(0, dtype=jnp.int32)
            init_state = MODEL.default_state(params, config)

        _, out_tokens = jax.lax.scan(inner_scan, (init_token, init_state, start_gen_key), prompt)
        return out_tokens

    return generate_thread


def build_generate_sft_thread(MODEL, NOISER, frozen_noiser_params, config, base_evo_keys, master_gen_key, temperature=1.0):

    def forward_and_sample(noiser_params, params, input_token, input_state, generation_key, iterinfo, target_token):
        print("compiling forward and sample")
        gen_key, _gen_key = jax.random.split(generation_key)
        generated_outs, generated_state = MODEL.forward(NOISER, frozen_noiser_params, noiser_params, config, params, base_evo_keys, iterinfo, input_token, input_state)
        logits = generated_outs[-1]
        if temperature != 0.0:
            sampled_tok = jax.random.categorical(_gen_key, logits / temperature)
        else:
            sampled_tok = jnp.argmax(logits)
        log_probs = jax.nn.log_softmax(logits)
        sampled_log_prob = log_probs[target_token]
        sampled_log_prob = jnp.where(jnp.logical_and(target_token == 0, input_token == 0), 0.0, sampled_log_prob)
        return sampled_tok, generated_state, gen_key, sampled_log_prob
    
    def generate_thread(noiser_params, params, inputs, targets, thread_idx, epoch_num, init_token, init_state):
        print("Compiling generate_batch")

        start_gen_key = fold_in_helper(master_gen_key, epoch_num, thread_idx)

        iterinfo = (epoch_num, thread_idx)
        def inner_scan(carry, input_tuple):
            tok, state, gen_key = carry
            input_token, target_token = input_tuple
            true_input = jnp.where(input_token == 0, tok, input_token)
            tok, state, gen_key, log_prob = forward_and_sample(noiser_params, params, true_input, state, gen_key, iterinfo, target_token)
            return (target_token, state, gen_key), {'true_inputs': true_input, 'log_probs': log_prob}

        (last_token, last_state, _), out_dict = jax.lax.scan(inner_scan, (init_token, init_state, start_gen_key), (inputs, targets))
        out_tokens = out_dict['true_inputs']
        log_probs = out_dict['log_probs']
        return out_tokens, log_probs, last_state, last_token

    return generate_thread


def build_validate(MODEL, config, params_example, base_evo_keys, master_gen_key, tokenizer, legacy_tokenizer, args, temperature=1.0, use_validation_set=True, NOISER=hs.noiser.base_noiser.Noiser, sigma=0.0, suppress_eos_token=0):
    frozen_noiser_params, noiser_params = NOISER.init_noiser(params_example, sigma, 0.0)

    if use_validation_set:
        val_cls = validation_tasks[args.task]
        if args.task == "countdown_chat":
            val_ds = getattr(args, "val_dataset_size", None) or 256
            validation_task = val_cls(
                tokenizer, legacy_tokenizer, args.generation_length,
                dataset_size=val_ds, val_holdout_size=val_ds,
            )
        else:
            validation_task = val_cls(tokenizer, legacy_tokenizer, args.generation_length)
    else:
        validation_task = all_tasks[args.task](tokenizer, legacy_tokenizer, args.generation_length)

    _generate_thread = build_generate_thread(MODEL, NOISER, frozen_noiser_params, config, base_evo_keys, master_gen_key, temperature, suppress_eos_token=suppress_eos_token)

    print("Compiling generate validation batch")
    start_time = time.time()
    generate_batch = jax.jit(jax.vmap(_generate_thread, in_axes=(None, None, 0, 0, None))).lower(noiser_params, params_example, jax.ShapeDtypeStruct((args.parallel_validations, args.generation_length), jnp.dtype('int32')), jnp.arange(args.parallel_validations), 0).compile()
    print("Compile time", time.time() - start_time)
    print("memory info")
    print(generate_batch.memory_analysis())
    
    def validate(params, epoch):
        sum_scores = 0.0

        for i in tqdm.trange(args.validation_iterations):
            unique_indices = jnp.arange(args.parallel_validations) + (i * args.parallel_validations)
            unique_prompts = validation_task.get_input(unique_indices)

            output_batch = jax.block_until_ready(generate_batch(noiser_params, params, unique_prompts, unique_indices, epoch))
            fitnesses = jax.device_put(validation_task.get_batch_fitness(jax.device_put(unique_indices, jax.local_devices(backend='cpu')[0]), jax.device_put(output_batch, jax.local_devices(backend='cpu')[0])), output_batch.device)

            sum_scores += jnp.sum(fitnesses)
        
        return sum_scores / (args.parallel_validations * args.validation_iterations)
    
    return validate


HELLASWAG_SEQ_LEN = 128


def _pad_token_ids(ids, max_len: int):
    ids = list(ids)[:max_len]
    n = len(ids)
    arr = np.zeros(max_len, dtype=np.int32)
    if n:
        arr[:n] = np.asarray(ids, dtype=np.int32)
    return arr, n


def _prepare_hellaswag_examples(tokenizer, val_size: int, seed: int = 42):
    """Load HellaSwag validation split (disjoint from countdown train data)."""
    ds = load_dataset("Rowan/hellaswag", split="validation")
    ds = ds.shuffle(seed=seed).select(range(min(val_size, len(ds))))
    examples = []
    for ex in ds:
        ctx_ids = list(tokenizer.encode(ex["ctx"]))
        max_ctx = max(1, HELLASWAG_SEQ_LEN // 2)
        ctx_ids = ctx_ids[:max_ctx]
        ctx_len = len(ctx_ids)
        tokens = []
        ending_lens = []
        for ending in ex["endings"]:
            ending_ids = list(tokenizer.encode(ending))
            max_ending = max(1, HELLASWAG_SEQ_LEN - ctx_len)
            ending_ids = ending_ids[:max_ending]
            combined = ctx_ids + ending_ids
            padded, _ = _pad_token_ids(combined, HELLASWAG_SEQ_LEN)
            tokens.append(padded)
            ending_lens.append(len(ending_ids))
        examples.append(
            {
                "tokens": np.stack(tokens, axis=0),
                "ctx_len": ctx_len,
                "ending_lens": np.asarray(ending_lens, dtype=np.int32),
                "label": int(ex["label"]),
            }
        )
    return examples


def build_hellaswag_validate(
    MODEL,
    config,
    params_example,
    base_evo_keys,
    master_gen_key,
    tokenizer,
    NOISER=hs.noiser.base_noiser.Noiser,
    val_size: int = 256,
    seed: int = 42,
    suppress_eos_token=None,
):
    """Multiple-choice HellaSwag accuracy via length-normalized teacher-forced log-likelihood."""
    frozen_noiser_params, noiser_params = NOISER.init_noiser(params_example, 0.0, 0.0)
    examples = _prepare_hellaswag_examples(tokenizer, val_size, seed=seed)
    print(f"HellaSwag validation examples: {len(examples)} (HF validation split, seed={seed})")

    def forward_logit(noiser_params, params, input_token, state, gen_key, iterinfo):
        generated_outs, generated_state = MODEL.forward(
            NOISER,
            frozen_noiser_params,
            noiser_params,
            config,
            params,
            base_evo_keys,
            iterinfo,
            input_token,
            state,
        )
        logits = generated_outs[-1]
        if suppress_eos_token is not None:
            logits = logits.at[suppress_eos_token].set(-jnp.inf)
        return logits, generated_state, gen_key

    def score_endings(noiser_params, params, tokens, ctx_len, ending_lens, epoch_num, thread_idx):
        init_state = MODEL.default_state(params, config)
        start_gen_key = fold_in_helper(master_gen_key, epoch_num, thread_idx)
        iterinfo = (epoch_num, thread_idx)

        def step(carry, target_pos):
            tok, state, gen_key = carry
            target = tokens[target_pos]
            logits, state, gen_key = forward_logit(
                noiser_params, params, tok, state, gen_key, iterinfo
            )
            log_probs = jax.nn.log_softmax(logits)
            logp = log_probs[target]
            in_suffix = jnp.logical_and(
                target_pos >= ctx_len,
                target_pos < ctx_len + ending_lens,
            )
            return (target, state, gen_key), jnp.where(in_suffix, logp, 0.0)

        init_carry = (tokens[0], init_state, start_gen_key)
        positions = jnp.arange(1, HELLASWAG_SEQ_LEN, dtype=jnp.int32)
        _, logps = jax.lax.scan(step, init_carry, positions)
        denom = jnp.maximum(ending_lens, 1)
        return jnp.sum(logps) / denom

    score_batch_fn = jax.jit(
        jax.vmap(
            score_endings,
            in_axes=(None, None, 0, None, 0, None, None),
        )
    )

    def validate(params, epoch):
        correct = 0
        for ex_idx, ex in enumerate(tqdm.tqdm(examples, desc="HellaSwag")):
            scores = score_batch_fn(
                noiser_params,
                params,
                jnp.asarray(ex["tokens"]),
                jnp.int32(ex["ctx_len"]),
                jnp.asarray(ex["ending_lens"]),
                epoch,
                ex_idx,
            )
            pred = int(np.argmax(np.asarray(scores)))
            if pred == ex["label"]:
                correct += 1
        return correct / len(examples)

    return validate
