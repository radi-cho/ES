import jax
import jax.numpy as jnp
from typing import NamedTuple

import hyperscalees as hs
from hyperscalees.environments.llm_bandits import all_tasks, validation_tasks

import tqdm
import time

import numpy as np

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
