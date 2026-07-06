"""EGGROLL with a learned, shape-shared diagonal Kronecker geometry.

This is deliberately a small extension of :mod:`hyperscalees.noiser.eggroll`.
The model update, antithetic pairing, fitness shaping, and rank-one implicit
forward are unchanged.  The only difference is that the row and column
Gaussian factors are multiplied by bounded learned standard deviations.

Geometry is learned from the pair-even component of the existing rewards, so
there are no center evaluations and no extra model generations.
"""

from __future__ import annotations

from collections import defaultdict
from functools import partial

import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten

from .eggroll import EggRoll, _noop_update, _simple_full_update


LORA = 1


def _shape_name(shape) -> str:
    return f"{int(shape[-2])}x{int(shape[-1])}"


def _project_log_std(log_std, condition_cap):
    """Project to RMS(std)=1 and a bounded max/min standard-deviation ratio."""
    half_range = 0.5 * jnp.log(jnp.asarray(condition_cap, jnp.float32))
    centered = log_std - 0.5 * (
        jax.nn.logsumexp(2.0 * log_std) - jnp.log(log_std.size)
    )
    centered = jnp.clip(centered, -half_range, half_range)
    return centered - 0.5 * (
        jax.nn.logsumexp(2.0 * centered) - jnp.log(centered.size)
    )


def _base_lora_factors(frozen_noiser_params, iterinfo, param, key):
    """Return the exact unscaled Gaussian factors used by ordinary EGGROLL."""
    epoch, thread_id = iterinfo
    true_epoch = (
        0
        if frozen_noiser_params["noise_reuse"] == 0
        else epoch // frozen_noiser_params["noise_reuse"]
    )
    true_thread_idx = thread_id // 2
    out_features, in_features = param.shape
    factors = jax.random.normal(
        jax.random.fold_in(jax.random.fold_in(key, true_epoch), true_thread_idx),
        (out_features + in_features, 1),
        dtype=param.dtype,
    )
    return factors[in_features:], factors[:in_features]


def get_lora_update_params(
    frozen_noiser_params, noiser_params, base_sigma, iterinfo, param, key
):
    base_out, base_in = _base_lora_factors(
        frozen_noiser_params, iterinfo, param, key
    )
    geometry = noiser_params["geometry"][_shape_name(param.shape)]
    std_out = jnp.exp(geometry["log_std_out"]).astype(param.dtype)[:, None]
    std_in = jnp.exp(geometry["log_std_in"]).astype(param.dtype)[:, None]
    sign = jnp.where(iterinfo[1] % 2 == 0, base_sigma, -base_sigma)
    return base_out * std_out * sign, base_in * std_in


def _simple_lora_update(
    base_sigma,
    param,
    key,
    scores,
    iterinfo,
    frozen_noiser_params,
    noiser_params,
):
    factors_out, factors_in = jax.vmap(
        partial(get_lora_update_params, frozen_noiser_params, noiser_params),
        in_axes=(None, 0, None, None),
    )(base_sigma, iterinfo, param, key)
    weighted_out = scores[:, None, None] * factors_out
    return jnp.einsum("nir,njr->ij", weighted_out, factors_in) / scores.size


def _pair_even_utilities(fitnesses, group_size, utility_clip, minimum_std):
    """Standardize pair-even values within each prompt candidate group."""
    pair_even = fitnesses.reshape((-1, group_size // 2, 2)).mean(axis=-1)
    centered = pair_even - pair_even.mean(axis=-1, keepdims=True)
    group_scale = jnp.sqrt(jnp.mean(jnp.square(centered), axis=-1, keepdims=True))
    valid = group_scale > minimum_std
    standardized = jnp.where(valid, centered / jnp.maximum(group_scale, minimum_std), 0.0)
    utility = standardized.reshape(-1)
    utility = utility - utility.mean()
    utility_scale = jnp.sqrt(jnp.mean(jnp.square(utility)))
    has_signal = utility_scale > minimum_std
    utility = jnp.where(has_signal, utility / jnp.maximum(utility_scale, minimum_std), 0.0)
    utility = jnp.clip(utility, -utility_clip, utility_clip)
    # Clipping can reintroduce a non-zero score-function baseline.
    utility = utility - utility.mean()
    return utility, has_signal


class DiagEggRoll(EggRoll):
    """Rank-one EGGROLL with adaptive diagonal row/column factor scales."""

    @classmethod
    def init_noiser(
        cls,
        params,
        sigma,
        lr,
        *args,
        es_map=None,
        diag_geometry_lr=0.02,
        diag_geometry_ema_decay=0.9,
        diag_geometry_warmup_pairs=64,
        diag_geometry_update_every=1,
        diag_geometry_condition_cap=2.0,
        diag_geometry_utility_clip=3.0,
        diag_geometry_minimum_std=1e-6,
        rank=1,
        **kwargs,
    ):
        group_size = int(kwargs.get("group_size", 0))
        # Unperturbed validators initialize the selected noiser with sigma=0,
        # group_size=0, and no es_map. Some still pass a non-null iterinfo.
        if group_size == 0 and es_map is None:
            if float(sigma) != 0.0 or float(lr) != 0.0:
                raise ValueError(
                    "diag_eggroll needs es_map and a nonzero group for training"
                )
            return super().init_noiser(
                params, sigma, lr, *args, rank=rank, **kwargs
            )
        if rank != 1:
            raise ValueError("diag_eggroll currently supports rank=1 only")
        if group_size < 4 or group_size % 4:
            raise ValueError(
                "diag_eggroll needs generations_per_prompt divisible by four "
                "(at least two antithetic directions per prompt)"
            )
        if not 0.0 <= diag_geometry_ema_decay < 1.0:
            raise ValueError("diag geometry EMA decay must be in [0, 1)")
        if diag_geometry_lr < 0.0 or diag_geometry_warmup_pairs < 0:
            raise ValueError("diag geometry learning rate/warmup must be nonnegative")
        if diag_geometry_update_every < 1:
            raise ValueError("diag geometry update interval must be positive")
        if diag_geometry_condition_cap < 1.0:
            raise ValueError("diag geometry condition cap must be at least one")
        if es_map is None:
            raise ValueError("diag_eggroll training requires es_map")
        if len(jax.devices()) != 1:
            raise ValueError("diag_eggroll currently supports one visible device")

        frozen, dynamic = super().init_noiser(
            params, sigma, lr, *args, rank=rank, **kwargs
        )
        flat_params, _ = tree_flatten(params)
        flat_es, _ = tree_flatten(es_map)
        shapes = sorted(
            {
                (int(param.shape[-2]), int(param.shape[-1]))
                for param, classification in zip(flat_params, flat_es)
                if int(classification) == LORA
            }
        )
        if not shapes:
            raise ValueError("diag_eggroll found no rank-one matrix parameters")
        geometry = {
            _shape_name(shape): {
                "log_std_out": jnp.zeros(shape[0], dtype=jnp.float32),
                "log_std_in": jnp.zeros(shape[1], dtype=jnp.float32),
                "ema_out": jnp.zeros(shape[0], dtype=jnp.float32),
                "ema_in": jnp.zeros(shape[1], dtype=jnp.float32),
            }
            for shape in shapes
        }
        frozen = frozen | {
            "diag_geometry_lr": float(diag_geometry_lr),
            "diag_geometry_ema_decay": float(diag_geometry_ema_decay),
            "diag_geometry_warmup_pairs": int(diag_geometry_warmup_pairs),
            "diag_geometry_update_every": int(diag_geometry_update_every),
            "diag_geometry_condition_cap": float(diag_geometry_condition_cap),
            "diag_geometry_utility_clip": float(diag_geometry_utility_clip),
            "diag_geometry_minimum_std": float(diag_geometry_minimum_std),
        }
        dynamic = dynamic | {
            "geometry": geometry,
            "geometry_seen_pairs": jnp.asarray(0, dtype=jnp.int32),
            "geometry_update_count": jnp.asarray(0, dtype=jnp.int32),
        }
        return frozen, dynamic

    @classmethod
    def do_mm(
        cls,
        frozen_noiser_params,
        noiser_params,
        param,
        base_key,
        iterinfo,
        x,
    ):
        base = x @ param.T
        if iterinfo is None or "geometry" not in noiser_params:
            return base
        factor_out, factor_in = get_lora_update_params(
            frozen_noiser_params,
            noiser_params,
            noiser_params["sigma"],
            iterinfo,
            param,
            base_key,
        )
        return base + x @ factor_in @ factor_out.T

    @classmethod
    def do_Tmm(
        cls,
        frozen_noiser_params,
        noiser_params,
        param,
        base_key,
        iterinfo,
        x,
    ):
        base = x @ param
        if iterinfo is None or "geometry" not in noiser_params:
            return base
        factor_out, factor_in = get_lora_update_params(
            frozen_noiser_params,
            noiser_params,
            noiser_params["sigma"],
            iterinfo,
            param,
            base_key,
        )
        return base + x @ factor_out @ factor_in.T

    @classmethod
    def _do_update(
        cls,
        param,
        base_key,
        fitnesses,
        iterinfos,
        map_classification,
        sigma,
        frozen_noiser_params,
        noiser_params,
    ):
        if map_classification == LORA:
            update_fn = partial(_simple_lora_update, noiser_params=noiser_params)
        elif map_classification == 0:
            update_fn = _simple_full_update
        else:
            update_fn = _noop_update

        if len(base_key.shape) == 0:
            new_grad = update_fn(
                sigma,
                param,
                base_key,
                fitnesses,
                iterinfos,
                frozen_noiser_params,
            )
        else:
            new_grad = jax.lax.scan(
                lambda _, values: (
                    0,
                    update_fn(
                        sigma,
                        values[0],
                        values[1],
                        fitnesses,
                        iterinfos,
                        frozen_noiser_params,
                    ),
                ),
                0,
                xs=(param, base_key),
            )[1]
        return -(new_grad * jnp.sqrt(fitnesses.size)).astype(param.dtype)

    @classmethod
    def _geometry_scores(
        cls,
        frozen_noiser_params,
        noiser_params,
        params,
        base_keys,
        utilities,
        epoch,
        es_map,
    ):
        flat_params, _ = tree_flatten(params)
        flat_keys, _ = tree_flatten(base_keys)
        flat_es, _ = tree_flatten(es_map)
        pair_ids = jnp.arange(utilities.size, dtype=jnp.int32)
        true_epoch = (
            0
            if frozen_noiser_params["noise_reuse"] == 0
            else epoch // frozen_noiser_params["noise_reuse"]
        )
        grouped = defaultdict(list)

        for param, base_key, classification in zip(
            flat_params, flat_keys, flat_es
        ):
            if int(classification) != LORA:
                continue
            out_features, in_features = param.shape[-2:]
            module_keys = base_key.reshape((-1,))

            def score_module(module_key):
                def sample(pair_id):
                    factors = jax.random.normal(
                        jax.random.fold_in(
                            jax.random.fold_in(module_key, true_epoch), pair_id
                        ),
                        (out_features + in_features, 1),
                        dtype=param.dtype,
                    )
                    return factors[in_features:, 0], factors[:in_features, 0]

                base_out, base_in = jax.vmap(sample)(pair_ids)
                utility = utilities[:, None]
                # Natural gradient in log(std): the ordinary score is
                # (z**2 - 1), whose diagonal Fisher information is 2.
                score_out = 0.5 * jnp.mean(
                    utility * (jnp.square(base_out.astype(jnp.float32)) - 1.0),
                    axis=0,
                )
                score_in = 0.5 * jnp.mean(
                    utility * (jnp.square(base_in.astype(jnp.float32)) - 1.0),
                    axis=0,
                )
                return score_out, score_in

            module_out, module_in = jax.vmap(score_module)(module_keys)
            grouped[_shape_name(param.shape)].append(
                (module_out, module_in)
            )

        result = {}
        for name, pieces in grouped.items():
            all_out = jnp.concatenate([piece[0] for piece in pieces], axis=0)
            all_in = jnp.concatenate([piece[1] for piece in pieces], axis=0)
            result[name] = (all_out.mean(axis=0), all_in.mean(axis=0))
        return result

    @classmethod
    def _update_geometry(
        cls,
        frozen_noiser_params,
        noiser_params,
        params,
        base_keys,
        fitnesses,
        iterinfos,
        es_map,
    ):
        group_size = frozen_noiser_params["group_size"]
        utilities, has_signal = _pair_even_utilities(
            fitnesses,
            group_size,
            frozen_noiser_params["diag_geometry_utility_clip"],
            frozen_noiser_params["diag_geometry_minimum_std"],
        )
        epoch = iterinfos[0][0]
        gradients = cls._geometry_scores(
            frozen_noiser_params,
            noiser_params,
            params,
            base_keys,
            utilities,
            epoch,
            es_map,
        )
        pair_count = fitnesses.size // 2
        seen_pairs = noiser_params["geometry_seen_pairs"] + pair_count
        should_update = jnp.logical_and(
            has_signal,
            jnp.logical_and(
                seen_pairs >= frozen_noiser_params["diag_geometry_warmup_pairs"],
                jnp.logical_and(
                    (epoch + 1)
                    % frozen_noiser_params["diag_geometry_update_every"]
                    == 0,
                    frozen_noiser_params["diag_geometry_lr"] > 0.0,
                ),
            ),
        )
        decay = frozen_noiser_params["diag_geometry_ema_decay"]
        learning_rate = frozen_noiser_params["diag_geometry_lr"]
        cap = frozen_noiser_params["diag_geometry_condition_cap"]
        new_geometry = {}
        for name, state in noiser_params["geometry"].items():
            gradient_out, gradient_in = gradients[name]
            ema_out = decay * state["ema_out"] + (1.0 - decay) * gradient_out
            ema_in = decay * state["ema_in"] + (1.0 - decay) * gradient_in
            candidate_out = _project_log_std(
                state["log_std_out"] + learning_rate * ema_out, cap
            )
            candidate_in = _project_log_std(
                state["log_std_in"] + learning_rate * ema_in, cap
            )
            new_geometry[name] = {
                "log_std_out": jnp.where(
                    should_update, candidate_out, state["log_std_out"]
                ),
                "log_std_in": jnp.where(
                    should_update, candidate_in, state["log_std_in"]
                ),
                "ema_out": ema_out,
                "ema_in": ema_in,
            }
        return noiser_params | {
            "geometry": new_geometry,
            "geometry_seen_pairs": seen_pairs,
            "geometry_update_count": noiser_params["geometry_update_count"]
            + should_update.astype(jnp.int32),
        }

    @classmethod
    def do_updates(
        cls,
        frozen_noiser_params,
        noiser_params,
        params,
        base_keys,
        fitnesses,
        iterinfos,
        es_map,
    ):
        # Keep the model update algebra exactly equal to ordinary EGGROLL, but
        # use the same scaled factors that produced the current rollouts.
        new_grad = jax.tree.map(
            lambda param, key, classification: cls._do_update(
                param,
                key,
                fitnesses,
                iterinfos,
                classification,
                noiser_params["sigma"],
                frozen_noiser_params,
                noiser_params,
            ),
            params,
            base_keys,
            es_map,
        )
        updates, opt_state = frozen_noiser_params["solver"].update(
            new_grad, noiser_params["opt_state"], params
        )
        new_params = jax.tree.map(lambda value, update: value + update, params, updates)
        geometry_state = cls._update_geometry(
            frozen_noiser_params,
            noiser_params,
            params,
            base_keys,
            fitnesses,
            iterinfos,
            es_map,
        )
        return geometry_state | {"opt_state": opt_state}, new_params


def geometry_diagnostics(noiser_params):
    ratios = []
    for state in noiser_params["geometry"].values():
        std_out = jnp.exp(state["log_std_out"])
        std_in = jnp.exp(state["log_std_in"])
        ratios.extend(
            [std_out.max() / std_out.min(), std_in.max() / std_in.min()]
        )
    return {
        "geometry_seen_pairs": noiser_params["geometry_seen_pairs"],
        "geometry_update_count": noiser_params["geometry_update_count"],
        "geometry_max_std_ratio": jnp.max(jnp.stack(ratios)),
    }
