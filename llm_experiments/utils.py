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


def antithetic_hidden_state_statistics(
    hidden_states,
    sigma,
    *,
    center_rms_floor=1e-6,
):
    """Return normalized differences and center RMS for adjacent +/- members."""

    hidden_states = jnp.asarray(hidden_states)
    if hidden_states.ndim != 3 or hidden_states.shape[0] % 2:
        raise ValueError("hidden_states must have shape [2 * pairs, layers, hidden]")
    if center_rms_floor <= 0.0:
        raise ValueError("center_rms_floor must be positive")

    pairs = hidden_states.reshape(
        hidden_states.shape[0] // 2, 2, *hidden_states.shape[1:]
    ).astype(jnp.float32)
    positive, negative = pairs[:, 0], pairs[:, 1]
    center = 0.5 * (positive + negative)
    center_rms = jnp.sqrt(jnp.mean(jnp.square(center), axis=-1, keepdims=True))
    normalized = (positive - negative) / (
        2.0
        * jnp.asarray(sigma, dtype=jnp.float32)
        * jnp.maximum(center_rms, jnp.asarray(center_rms_floor, jnp.float32))
    )
    return normalized, center_rms[..., 0]


def normalize_antithetic_hidden_states(
    hidden_states,
    sigma,
    *,
    center_rms_floor=1e-6,
):
    """Return center-normalized hidden differences for adjacent +/- members."""

    normalized, _ = antithetic_hidden_state_statistics(
        hidden_states,
        sigma,
        center_rms_floor=center_rms_floor,
    )
    return normalized


def countsketch_antithetic_hidden_states(
    hidden_states,
    sigma,
    bucket_ids,
    bucket_signs,
    *,
    num_buckets=128,
    center_rms_floor=1e-6,
):
    """Map adjacent +/- prompt states to center-normalized CountSketch blocks.

    ``hidden_states`` has shape ``[2 * pairs, layers, hidden]`` with each
    positive member immediately followed by its negative member. Bucket IDs
    and signs have shape ``[layers, hidden]``; independent rows therefore give
    each captured layer an independent fixed sketch.
    """

    hidden_states = jnp.asarray(hidden_states)
    bucket_ids = jnp.asarray(bucket_ids, dtype=jnp.int32)
    bucket_signs = jnp.asarray(bucket_signs, dtype=jnp.float32)
    if hidden_states.ndim != 3 or hidden_states.shape[0] % 2:
        raise ValueError("hidden_states must have shape [2 * pairs, layers, hidden]")
    if bucket_ids.shape != hidden_states.shape[1:]:
        raise ValueError("bucket_ids must have shape [layers, hidden]")
    if bucket_signs.shape != bucket_ids.shape:
        raise ValueError("bucket_signs must match bucket_ids")
    if num_buckets < 1 or center_rms_floor <= 0.0:
        raise ValueError("num_buckets and center_rms_floor must be positive")

    direction = normalize_antithetic_hidden_states(
        hidden_states,
        sigma,
        center_rms_floor=center_rms_floor,
    )

    def sketch_layer(layer_values, layer_buckets, layer_signs):
        def sketch_row(values):
            return jnp.zeros((num_buckets,), dtype=jnp.float32).at[
                layer_buckets
            ].add(values * layer_signs)

        return jax.vmap(sketch_row)(layer_values)

    blocks = jax.vmap(
        sketch_layer, in_axes=(1, 0, 0), out_axes=1
    )(direction, bucket_ids, bucket_signs)
    return blocks.reshape((blocks.shape[0], -1))


def summarize_antithetic_hidden_states(
    hidden_states,
    sigma,
    *,
    center_rms_floor=1e-6,
    feature_clip=10.0,
):
    """Return six cheap signed response statistics for every captured layer.

    Unlike an absolute hidden representation, every statistic is odd under
    swapping the positive and negative members.  The summaries retain global
    response shape that a bucketed sketch can discard, while adding no model
    forward work and only a handful of reductions over the hidden dimension.
    """

    values = jnp.asarray(hidden_states)
    if values.ndim != 3 or values.shape[0] % 2:
        raise ValueError("hidden_states must have shape [2 * pairs, layers, hidden]")
    if feature_clip <= 0.0:
        raise ValueError("feature_clip must be positive")
    direction, _ = antithetic_hidden_state_statistics(
        values,
        sigma,
        center_rms_floor=center_rms_floor,
    )
    pairs = values.reshape(values.shape[0] // 2, 2, *values.shape[1:]).astype(
        jnp.float32
    )
    center = 0.5 * (pairs[:, 0] + pairs[:, 1])
    center_rms = jnp.sqrt(jnp.mean(jnp.square(center), axis=-1, keepdims=True))
    center_unit_rms = center / jnp.maximum(
        center_rms, jnp.asarray(center_rms_floor, dtype=jnp.float32)
    )
    direction_rms = jnp.sqrt(
        jnp.maximum(jnp.mean(jnp.square(direction), axis=-1, keepdims=True), 1e-12)
    )
    standardized_direction = jnp.clip(
        direction / direction_rms,
        -jnp.asarray(feature_clip, dtype=jnp.float32),
        jnp.asarray(feature_clip, dtype=jnp.float32),
    )

    summaries = jnp.stack(
        (
            jnp.mean(direction * center_unit_rms, axis=-1),
            jnp.mean(direction, axis=-1),
            0.5 * (
                jnp.max(direction, axis=-1) + jnp.min(direction, axis=-1)
            ),
            jnp.mean(jnp.sign(direction), axis=-1),
            jnp.mean(jnp.power(standardized_direction, 3), axis=-1),
            jnp.mean(direction * jnp.sign(center_unit_rms), axis=-1),
        ),
        axis=-1,
    )
    return jnp.clip(summaries, -feature_clip, feature_clip).reshape(
        (summaries.shape[0], -1)
    )


def sketch_summary_antithetic_hidden_states(
    hidden_states,
    sigma,
    bucket_ids,
    bucket_signs,
    *,
    num_buckets=128,
    center_rms_floor=1e-6,
    summary_clip=10.0,
):
    """Concatenate the PR5 CountSketch with signed layer summaries."""

    sketch = countsketch_antithetic_hidden_states(
        hidden_states,
        sigma,
        bucket_ids,
        bucket_signs,
        num_buckets=num_buckets,
        center_rms_floor=center_rms_floor,
    )
    summary = summarize_antithetic_hidden_states(
        hidden_states,
        sigma,
        center_rms_floor=center_rms_floor,
        feature_clip=summary_clip,
    )
    return jnp.concatenate((sketch, summary), axis=-1)

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


def build_generate_batch_with_preview(
    MODEL,
    NOISER,
    frozen_noiser_params,
    config,
    base_evo_keys,
    master_gen_key,
    preview_layers,
    *,
    temperature=0.0,
    suppress_eos_token=None,
    center_rms_floor=1e-6,
):
    """Build a fused rollout/all-layer-preview kernel for one shared prompt.

    The returned function evaluates adjacent antithetic members and returns
    their complete token sequences, normalized hidden difference, and center
    RMS at the final forced prompt token. Prompt-prefix calls bypass the LM
    head; the key is still advanced exactly as in :func:`build_generate_thread`.
    """

    preview_layers = tuple(int(layer) for layer in preview_layers)
    num_layers = len(config["layer_types"])
    if not preview_layers or len(set(preview_layers)) != len(preview_layers):
        raise ValueError("preview_layers must contain distinct layer indices")
    if any(layer < 0 or layer >= num_layers for layer in preview_layers):
        raise ValueError("preview layer index is outside the model")

    preview_config = {**config, "preview_layers": preview_layers}
    decode_config = dict(config)
    decode_config.pop("preview_layers", None)

    def generate_batch(
        noiser_params,
        params,
        prompt,
        prompt_length,
        member_ids,
        epoch_num,
    ):
        batch_size = member_ids.shape[0]
        if batch_size % 2:
            raise ValueError("member_ids must contain complete antithetic pairs")

        initial_state = MODEL.default_state(params, preview_config)
        states = jax.tree.map(
            lambda value: jnp.broadcast_to(value, (batch_size,) + value.shape),
            initial_state,
        )
        tokens = jnp.zeros((batch_size,), dtype=jnp.int32)
        generation_keys = jax.vmap(
            lambda member_id: fold_in_helper(
                master_gen_key, epoch_num, member_id
            )
        )(member_ids)

        def prefix_only(input_tokens, input_states, input_keys):
            """Advance model state without computing unused prompt logits."""

            def one(input_token, input_state, member_id):
                _, output_state = MODEL.forward(
                    NOISER,
                    frozen_noiser_params,
                    noiser_params,
                    decode_config,
                    params,
                    base_evo_keys,
                    (epoch_num, member_id),
                    input_token,
                    input_state,
                    length=prompt_length,
                    return_hidden=True,
                )
                return output_state

            output_states = jax.vmap(one)(input_tokens, input_states, member_ids)
            # The ordinary rollout splits once at every position, including
            # forced-prefix positions whose sampled token is never consumed.
            output_keys = jax.vmap(lambda key: jax.random.split(key)[0])(
                input_keys
            )
            return tokens, output_states, output_keys

        def forward_and_sample(step_config, input_tokens, input_states, input_keys):
            def one(input_token, input_state, input_key, member_id):
                next_key, sample_key = jax.random.split(input_key)
                generated, output_state = MODEL.forward(
                    NOISER,
                    frozen_noiser_params,
                    noiser_params,
                    step_config,
                    params,
                    base_evo_keys,
                    (epoch_num, member_id),
                    input_token,
                    input_state,
                    length=prompt_length,
                )
                logits = generated[-1]
                if suppress_eos_token is not None:
                    logits = logits.at[suppress_eos_token].set(-jnp.inf)
                if temperature == 0.0:
                    sampled = jnp.argmax(logits)
                else:
                    sampled = jax.random.categorical(sample_key, logits / temperature)
                return sampled, output_state, next_key

            return jax.vmap(one)(input_tokens, input_states, input_keys, member_ids)

        def scan_step(carry, step_input):
            previous_tokens, input_states, input_keys = carry
            step, prompt_token = step_input
            true_inputs = jnp.where(prompt_token == 0, previous_tokens, prompt_token)

            def prefix_branch(args):
                return prefix_only(*args)

            def head_branch(args):
                def capture_branch(inner_args):
                    return forward_and_sample(preview_config, *inner_args)

                def decode_branch(inner_args):
                    return forward_and_sample(decode_config, *inner_args)

                return jax.lax.cond(
                    step == prompt_length - 1,
                    capture_branch,
                    decode_branch,
                    args,
                )

            output_carry = jax.lax.cond(
                step < prompt_length - 1,
                prefix_branch,
                head_branch,
                (true_inputs, input_states, input_keys),
            )
            return output_carry, true_inputs

        (_, final_states, _), output_tokens = jax.lax.scan(
            scan_step,
            (tokens, states, generation_keys),
            (jnp.arange(prompt.shape[0], dtype=jnp.int32), prompt),
        )
        pair_inputs, center_rms = antithetic_hidden_state_statistics(
            final_states["preview_hidden"],
            noiser_params["sigma"],
            center_rms_floor=center_rms_floor,
        )
        return jnp.swapaxes(output_tokens, 0, 1), pair_inputs, center_rms

    return generate_batch


def build_preview_pair_thread(
    MODEL,
    NOISER,
    frozen_noiser_params,
    config,
    base_evo_keys,
    preview_layers,
    prompt_width,
    bucket_ids,
    bucket_signs,
    num_buckets=128,
    center_rms_floor=1e-6,
    feature_kind="sketch",
):
    """Build one global pair's hidden-only prefill feature."""

    preview_layers = tuple(int(layer) for layer in preview_layers)
    n_layers = len(config["layer_types"])
    if not preview_layers or len(set(preview_layers)) != len(preview_layers):
        raise ValueError("preview_layers must contain distinct layer indices")
    if any(layer < 0 or layer >= n_layers for layer in preview_layers):
        raise ValueError("preview layer index is outside the model")
    if prompt_width < 1:
        raise ValueError("prompt_width must be positive")
    if feature_kind not in ("sketch", "sketch_summary"):
        raise ValueError("feature_kind must be 'sketch' or 'sketch_summary'")
    bucket_ids = jnp.asarray(bucket_ids, dtype=jnp.int32)
    bucket_signs = jnp.asarray(bucket_signs, dtype=jnp.float32)
    expected_sketch_shape = (len(preview_layers), int(config["hidden_size"]))
    if bucket_ids.shape != expected_sketch_shape:
        raise ValueError(
            f"bucket_ids must have shape {expected_sketch_shape}"
        )
    if bucket_signs.shape != expected_sketch_shape:
        raise ValueError(
            f"bucket_signs must have shape {expected_sketch_shape}"
        )
    preview_config = {
        **config,
        "attn_cache_len": int(prompt_width),
        "preview_layers": preview_layers,
    }

    def preview_pair(
        noiser_params,
        params,
        prompt,
        prompt_length,
        global_pair_id,
        epoch_num,
    ):
        global_member_ids = (
            2 * jnp.asarray(global_pair_id, dtype=jnp.int32)
            + jnp.arange(2, dtype=jnp.int32)
        )

        def preview_member(global_member_id):
            iterinfo = (epoch_num, global_member_id)
            init_state = MODEL.default_state(params, preview_config)
            _, final_state = MODEL.forward(
                NOISER,
                frozen_noiser_params,
                noiser_params,
                preview_config,
                params,
                base_evo_keys,
                iterinfo,
                prompt,
                init_state,
                length=prompt_length,
                return_hidden=True,
            )
            return final_state["preview_hidden"]

        pair_hidden = jax.vmap(preview_member)(global_member_ids)
        feature_fn = (
            countsketch_antithetic_hidden_states
            if feature_kind == "sketch"
            else sketch_summary_antithetic_hidden_states
        )
        return feature_fn(
            pair_hidden,
            noiser_params["sigma"],
            bucket_ids,
            bucket_signs,
            num_buckets=num_buckets,
            center_rms_floor=center_rms_floor,
        )[0]

    return preview_pair


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


def build_train_eval(
    MODEL,
    config,
    params_example,
    base_evo_keys,
    master_gen_key,
    train_task,
    args,
    NOISER=hs.noiser.base_noiser.Noiser,
    sigma=0.0,
    temperature=0.0,
    suppress_eos_token=0,
):
    """Greedy fitness eval on all examples in the train task (in-distribution)."""
    frozen_noiser_params, noiser_params = NOISER.init_noiser(params_example, sigma, 0.0)
    _generate_thread = build_generate_thread(
        MODEL,
        NOISER,
        frozen_noiser_params,
        config,
        base_evo_keys,
        master_gen_key,
        temperature,
        suppress_eos_token=suppress_eos_token,
    )

    num_train = len(train_task)
    compile_batch_size = num_train

    print(f"Compiling train eval batch ({num_train} examples, compile_batch={compile_batch_size})")
    start_time = time.time()
    generate_batch = jax.jit(
        jax.vmap(_generate_thread, in_axes=(None, None, 0, 0, None))
    ).lower(
        noiser_params,
        params_example,
        jax.ShapeDtypeStruct((compile_batch_size, args.generation_length), jnp.dtype("int32")),
        jnp.arange(compile_batch_size),
        0,
    ).compile()
    print("Compile time", time.time() - start_time)
    print("memory info")
    print(generate_batch.memory_analysis())

    def evaluate_train(params, epoch, indices=None):
        eval_indices = (
            np.arange(num_train, dtype=np.int32)
            if indices is None
            else np.asarray(indices, dtype=np.int32)
        )
        sum_scores = 0.0
        count = 0
        for start in range(0, len(eval_indices), compile_batch_size):
            chunk = eval_indices[start : start + compile_batch_size]
            n = len(chunk)
            if n < compile_batch_size:
                chunk = np.pad(chunk, (0, compile_batch_size - n), mode="edge")
            unique_indices = jnp.asarray(chunk, dtype=jnp.int32)
            unique_prompts = train_task.get_input(unique_indices)
            thread_idxes = jnp.arange(compile_batch_size, dtype=jnp.int32)
            output_batch = jax.block_until_ready(
                generate_batch(noiser_params, params, unique_prompts, thread_idxes, epoch)
            )
            fitnesses = jax.device_put(
                train_task.get_batch_fitness(
                    jax.device_put(unique_indices, jax.local_devices(backend="cpu")[0]),
                    jax.device_put(output_batch, jax.local_devices(backend="cpu")[0]),
                ),
                output_batch.device,
            )
            sum_scores += float(jnp.sum(fitnesses[:n]))
            count += n
        return sum_scores / count

    return evaluate_train


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
