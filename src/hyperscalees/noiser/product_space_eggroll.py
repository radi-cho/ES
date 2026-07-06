"""Shape-shared product-subspace EGGROLL experiment.

The implementation is intentionally isolated from :mod:`eggroll`.  It keeps
EGGROLL's implicit rank-one forward pass and exact warm-up update, then splits
each prompt group into isotropic scouts and probes in a learned row/column
product space.  The bases are estimated from squared antithetic score
differences; no extra generations are required.

This is an experimental retrofit.  Geometry is shared by matrix shape because
the current model/noiser interface does not expose a stable module identifier
during ``do_mm``.
"""

from __future__ import annotations

from collections import defaultdict

import jax
import jax.numpy as jnp
import optax
from jax.tree_util import tree_flatten, tree_unflatten

from .eggroll import EggRoll


LORA = 1


def _shape_name(shape) -> str:
    return f"{int(shape[-2])}x{int(shape[-1])}"


def _true_epoch(frozen_noiser_params, epoch):
    reuse = frozen_noiser_params["noise_reuse"]
    return jnp.asarray(0, dtype=jnp.int32) if reuse == 0 else epoch // reuse


def _direction_key(frozen_noiser_params, iterinfo, key):
    epoch, thread_id = iterinfo
    return jax.random.fold_in(
        jax.random.fold_in(key, _true_epoch(frozen_noiser_params, epoch)),
        thread_id // 2,
    )


def _base_lora_factors(frozen_noiser_params, iterinfo, param, key):
    """Reproduce the exact rank-one factors used by ordinary EGGROLL."""
    out_features, in_features = param.shape[-2:]
    factors = jax.random.normal(
        _direction_key(frozen_noiser_params, iterinfo, key),
        (out_features + in_features, 1),
        dtype=param.dtype,
    )
    return factors[in_features:, 0], factors[:in_features, 0]


def _active_lora_factors(frozen_noiser_params, noiser_params, param, base_factors):
    out_features, in_features = param.shape[-2:]
    rank = frozen_noiser_params["product_space_rank"]
    geometry = noiser_params["geometry"][_shape_name(param.shape)]
    base_out, base_in = base_factors
    alpha = geometry["u"].T @ base_out.astype(jnp.float32)
    beta = geometry["v"].T @ base_in.astype(jnp.float32)
    factor_out = jnp.sqrt(out_features / rank) * (geometry["u"] @ alpha)
    factor_in = jnp.sqrt(in_features / rank) * (geometry["v"] @ beta)
    return factor_out.astype(param.dtype), factor_in.astype(param.dtype)


def _is_active_pair(frozen_noiser_params, noiser_params, thread_id):
    warmed_up = (
        noiser_params["geometry_seen_pairs"]
        >= frozen_noiser_params["product_space_warmup_pairs"]
    )
    pair_slot = (thread_id % frozen_noiser_params["group_size"]) // 2
    return jnp.logical_and(
        warmed_up,
        pair_slot >= frozen_noiser_params["product_space_scout_pairs"],
    )


def get_lora_direction(
    frozen_noiser_params, noiser_params, iterinfo, param, key
):
    """Return the unsigned factors used by a scout or active pair."""
    base_factors = _base_lora_factors(
        frozen_noiser_params, iterinfo, param, key
    )
    return jax.lax.cond(
        _is_active_pair(frozen_noiser_params, noiser_params, iterinfo[1]),
        lambda _: _active_lora_factors(
            frozen_noiser_params, noiser_params, param, base_factors
        ),
        lambda _: base_factors,
        operand=None,
    )


def get_lora_update_params(
    frozen_noiser_params, noiser_params, base_sigma, iterinfo, param, key
):
    factor_out, factor_in = get_lora_direction(
        frozen_noiser_params, noiser_params, iterinfo, param, key
    )
    sign = jnp.where(iterinfo[1] % 2 == 0, base_sigma, -base_sigma)
    return factor_out * sign, factor_in


def _pair_data(fitnesses, iterinfos):
    # ``fitnesses`` use EGGROLL's in-population standardization.  The moment
    # and control-variate identities are exact for raw (or past-normalized)
    # differences and are self-normalized approximations on this retrofit path.
    if fitnesses.shape[0] % 2:
        raise ValueError("product_space_eggroll requires an even population")
    epochs, thread_ids = iterinfos
    return (
        fitnesses[::2] - fitnesses[1::2],
        epochs[::2],
        thread_ids[::2],
    )


def _pair_masks(frozen_noiser_params, noiser_params, thread_ids):
    active = jax.vmap(
        lambda thread_id: _is_active_pair(
            frozen_noiser_params, noiser_params, thread_id
        )
    )(thread_ids)
    return (~active).astype(jnp.float32), active.astype(jnp.float32)


def _module_factors(
    frozen_noiser_params,
    noiser_params,
    param,
    module_keys,
    pair_epochs,
    pair_threads,
):
    """Regenerate factors as ``[modules, pairs, dimension]`` arrays."""

    def factors_for_module(module_key):
        return jax.vmap(
            lambda epoch, thread: get_lora_direction(
                frozen_noiser_params,
                noiser_params,
                (epoch, thread),
                param,
                module_key,
            )
        )(pair_epochs, pair_threads)

    return jax.vmap(factors_for_module)(module_keys)


def _module_base_factors(
    frozen_noiser_params, param, module_keys, pair_epochs, pair_threads
):
    """Regenerate isotropic factors, including during all-scout warm-up."""

    def factors_for_module(module_key):
        return jax.vmap(
            lambda epoch, thread: _base_lora_factors(
                frozen_noiser_params, (epoch, thread), param, module_key
            )
        )(pair_epochs, pair_threads)

    return jax.vmap(factors_for_module)(module_keys)


def _orthonormalize(value):
    q, r = jnp.linalg.qr(value.astype(jnp.float32), mode="reduced")
    # Fix QR's arbitrary signs so diagnostics do not report artificial churn.
    signs = jnp.where(jnp.diag(r) < 0.0, -1.0, 1.0)
    return q * signs[None, :]


class ProductSpaceEggRoll(EggRoll):
    """EGGROLL with fitness-only product-space tomography and exploration."""

    @classmethod
    def init_noiser(
        cls,
        params,
        sigma,
        lr,
        *args,
        es_map=None,
        product_space_rank=8,
        product_space_scout_pairs=2,
        product_space_warmup_pairs=256,
        product_space_geometry_lr=0.02,
        product_space_geometry_ema_decay=0.9,
        product_space_geometry_update_every=1,
        product_space_control_variate=True,
        product_space_seed=0,
        rank=1,
        **kwargs,
    ):
        group_size = int(kwargs.get("group_size", 0))
        if group_size == 0 and es_map is None:
            if float(sigma) != 0.0 or float(lr) != 0.0:
                raise ValueError(
                    "product_space_eggroll training requires es_map and a group"
                )
            return super().init_noiser(
                params, sigma, lr, *args, rank=rank, **kwargs
            )
        if rank != 1:
            raise ValueError("product_space_eggroll supports rank-one probes only")
        if group_size < 4 or group_size % 2:
            raise ValueError("product_space_eggroll needs an even group of at least 4")
        pairs_per_prompt = group_size // 2
        if not 0 < product_space_scout_pairs < pairs_per_prompt:
            raise ValueError(
                "product_space_scout_pairs must leave both scouts and active pairs"
            )
        if product_space_rank < 1:
            raise ValueError("product_space_rank must be positive")
        if product_space_warmup_pairs < 0:
            raise ValueError("product_space_warmup_pairs must be nonnegative")
        if product_space_geometry_lr < 0.0:
            raise ValueError("product_space_geometry_lr must be nonnegative")
        if not 0.0 <= product_space_geometry_ema_decay < 1.0:
            raise ValueError("product-space EMA decay must be in [0, 1)")
        if product_space_geometry_update_every < 1:
            raise ValueError("product-space update interval must be positive")
        if es_map is None:
            raise ValueError("product_space_eggroll training requires es_map")
        if not bool(kwargs.get("freeze_nonlora", False)):
            raise ValueError(
                "product_space_eggroll currently requires freeze_nonlora=True"
            )
        if int(kwargs.get("noise_reuse", 0)) != 1:
            raise ValueError(
                "product_space_eggroll requires noise_reuse=1 so scouts are fresh"
            )
        if len(jax.devices()) != 1:
            raise ValueError("product_space_eggroll currently supports one device")

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
            raise ValueError("product_space_eggroll found no matrix parameters")
        if any(product_space_rank > min(shape) for shape in shapes):
            raise ValueError("product_space_rank exceeds a matrix dimension")

        geometry = {}
        for shape_index, (out_features, in_features) in enumerate(shapes):
            key = jax.random.fold_in(
                jax.random.key(int(product_space_seed)), 0x5EED + shape_index
            )
            key_u, key_v = jax.random.split(key)
            u = _orthonormalize(
                jax.random.normal(
                    key_u, (out_features, product_space_rank), jnp.float32
                )
            )
            v = _orthonormalize(
                jax.random.normal(
                    key_v, (in_features, product_space_rank), jnp.float32
                )
            )
            geometry[_shape_name((out_features, in_features))] = {
                "u": u,
                "v": v,
                "ema_u": jnp.zeros_like(u),
                "ema_v": jnp.zeros_like(v),
            }

        frozen = frozen | {
            "product_space_rank": int(product_space_rank),
            "product_space_scout_pairs": int(product_space_scout_pairs),
            "product_space_warmup_pairs": int(product_space_warmup_pairs),
            "product_space_geometry_lr": float(product_space_geometry_lr),
            "product_space_geometry_ema_decay": float(
                product_space_geometry_ema_decay
            ),
            "product_space_geometry_update_every": int(
                product_space_geometry_update_every
            ),
            "product_space_control_variate": bool(product_space_control_variate),
            "product_space_seed": int(product_space_seed),
        }
        dynamic = dynamic | {
            "geometry": geometry,
            "geometry_seen_pairs": jnp.asarray(0, jnp.int32),
            "geometry_update_count": jnp.asarray(0, jnp.int32),
            "basis_churn": jnp.asarray(0.0, jnp.float32),
            "active_to_scout_energy": jnp.asarray(0.0, jnp.float32),
            "residual_variance_ratio": jnp.asarray(1.0, jnp.float32),
            "scout_prediction_correlation": jnp.asarray(0.0, jnp.float32),
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
        return base + x @ factor_in[:, None] @ factor_out[None, :]

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
        return base + x @ factor_out[:, None] @ factor_in[None, :]

    @classmethod
    def _active_coefficients_and_predictions(
        cls,
        frozen_noiser_params,
        noiser_params,
        params,
        base_keys,
        es_map,
        differences,
        pair_epochs,
        pair_threads,
        active_weights,
    ):
        """First pass: estimate active matrices and predict scout scores."""
        flat_params, _ = tree_flatten(params)
        flat_keys, _ = tree_flatten(base_keys)
        flat_es, _ = tree_flatten(es_map)
        active_count = jnp.maximum(active_weights.sum(), 1.0)
        prediction = jnp.zeros_like(differences, dtype=jnp.float32)
        coefficients = []

        for param, keys, classification in zip(flat_params, flat_keys, flat_es):
            if int(classification) != LORA:
                coefficients.append(None)
                continue
            out_features, in_features = param.shape[-2:]
            module_keys = keys.reshape((-1,))
            module_param = param.reshape((-1, out_features, in_features))[0]
            factor_out, factor_in = _module_factors(
                frozen_noiser_params,
                noiser_params,
                module_param,
                module_keys,
                pair_epochs,
                pair_threads,
            )
            geometry = noiser_params["geometry"][_shape_name(param.shape)]
            scale_out = jnp.sqrt(out_features / frozen_noiser_params["product_space_rank"])
            scale_in = jnp.sqrt(in_features / frozen_noiser_params["product_space_rank"])
            alpha = jnp.einsum(
                "mpo,ok->mpk", factor_out.astype(jnp.float32), geometry["u"]
            ) / scale_out
            beta = jnp.einsum(
                "mpi,ik->mpk", factor_in.astype(jnp.float32), geometry["v"]
            ) / scale_in
            radius = scale_out * scale_in
            weights = active_weights * differences.astype(jnp.float32) / radius
            coefficient = jnp.einsum(
                "p,mpk,mpl->mkl", weights, alpha, beta
            ) / active_count
            coefficients.append(coefficient)

            scout_u = jnp.einsum(
                "mpo,ok->mpk", factor_out.astype(jnp.float32), geometry["u"]
            )
            scout_v = jnp.einsum(
                "mpi,ik->mpk", factor_in.astype(jnp.float32), geometry["v"]
            )
            prediction = prediction + jnp.einsum(
                "mpk,mkl,mpl->p", scout_u, coefficient, scout_v
            )

        return coefficients, prediction

    @classmethod
    def _product_update(
        cls,
        frozen_noiser_params,
        noiser_params,
        params,
        base_keys,
        fitnesses,
        iterinfos,
        es_map,
    ):
        differences, pair_epochs, pair_threads = _pair_data(fitnesses, iterinfos)
        scout_weights, active_weights = _pair_masks(
            frozen_noiser_params, noiser_params, pair_threads
        )
        scout_count = jnp.maximum(scout_weights.sum(), 1.0)
        pair_count = differences.size

        coefficients, prediction = cls._active_coefficients_and_predictions(
            frozen_noiser_params,
            noiser_params,
            params,
            base_keys,
            es_map,
            differences,
            pair_epochs,
            pair_threads,
            active_weights,
        )
        residual = differences.astype(jnp.float32) - prediction

        flat_params, treedef = tree_flatten(params)
        flat_keys, _ = tree_flatten(base_keys)
        flat_es, _ = tree_flatten(es_map)
        flat_updates = []
        # Regenerate factors instead of retaining every population factor from
        # pass one.  This trades cheap deterministic RNG for bounded Qwen HBM.
        for param, keys, classification, coefficient in zip(
            flat_params, flat_keys, flat_es, coefficients
        ):
            if int(classification) != LORA:
                flat_updates.append(jnp.zeros_like(param))
                continue
            out_features, in_features = param.shape[-2:]
            module_keys = keys.reshape((-1,))
            module_param = param.reshape((-1, out_features, in_features))[0]
            factor_out, factor_in = _module_factors(
                frozen_noiser_params,
                noiser_params,
                module_param,
                module_keys,
                pair_epochs,
                pair_threads,
            )
            geometry = noiser_params["geometry"][_shape_name(param.shape)]
            active_matrix = jnp.einsum(
                "ok,mkl,il->moi", geometry["u"], coefficient, geometry["v"]
            )
            if frozen_noiser_params["product_space_control_variate"]:
                scout_scores = scout_weights * residual
                scout_matrix = jnp.einsum(
                    "p,mpo,mpi->moi",
                    scout_scores,
                    factor_out.astype(jnp.float32),
                    factor_in.astype(jnp.float32),
                ) / scout_count
                estimate = active_matrix + scout_matrix
            else:
                scout_matrix = jnp.einsum(
                    "p,mpo,mpi->moi",
                    scout_weights * differences.astype(jnp.float32),
                    factor_out.astype(jnp.float32),
                    factor_in.astype(jnp.float32),
                ) / scout_count
                active_fraction = active_weights.sum() / pair_count
                scout_fraction = scout_weights.sum() / pair_count
                estimate = (
                    active_fraction * active_matrix + scout_fraction * scout_matrix
                )

            # EGGROLL's pair-level scale is -sigma*sqrt(N)/2.
            update = -(
                noiser_params["sigma"]
                * jnp.sqrt(jnp.asarray(fitnesses.size, jnp.float32))
                / 2.0
                * estimate
            )
            flat_updates.append(update.reshape(param.shape).astype(param.dtype))

        gradient = tree_unflatten(treedef, flat_updates)
        optimizer_updates, opt_state = frozen_noiser_params["solver"].update(
            gradient, noiser_params["opt_state"], params
        )
        new_params = optax.apply_updates(params, optimizer_updates)

        scout_energy = jnp.sum(
            scout_weights * jnp.square(differences.astype(jnp.float32))
        ) / scout_count
        active_count = jnp.maximum(active_weights.sum(), 1.0)
        active_energy = jnp.sum(
            active_weights * jnp.square(differences.astype(jnp.float32))
        ) / active_count
        scout_mean = jnp.sum(scout_weights * differences) / scout_count
        pred_mean = jnp.sum(scout_weights * prediction) / scout_count
        residual_mean = jnp.sum(scout_weights * residual) / scout_count
        covariance = jnp.sum(
            scout_weights
            * (differences - scout_mean)
            * (prediction - pred_mean)
        ) / scout_count
        scout_var = jnp.sum(
            scout_weights * jnp.square(differences - scout_mean)
        ) / scout_count
        pred_var = jnp.sum(
            scout_weights * jnp.square(prediction - pred_mean)
        ) / scout_count
        residual_var = jnp.sum(
            scout_weights * jnp.square(residual - residual_mean)
        ) / scout_count
        correlation = covariance / jnp.sqrt(scout_var * pred_var + 1e-8)
        return (
            noiser_params
            | {
                "opt_state": opt_state,
                "active_to_scout_energy": active_energy / (scout_energy + 1e-8),
                "residual_variance_ratio": residual_var / (scout_var + 1e-8),
                "scout_prediction_correlation": correlation,
            },
            new_params,
        )

    @classmethod
    def _geometry_tangents(
        cls,
        frozen_noiser_params,
        noiser_params,
        params,
        base_keys,
        es_map,
        differences,
        pair_epochs,
        pair_threads,
        scout_weights,
    ):
        flat_params, _ = tree_flatten(params)
        flat_keys, _ = tree_flatten(base_keys)
        flat_es, _ = tree_flatten(es_map)
        scout_count = jnp.maximum(scout_weights.sum(), 1.0)
        energy = jnp.sum(
            scout_weights * jnp.square(differences.astype(jnp.float32))
        ) / scout_count
        normalized_energy = (
            scout_weights
            * jnp.square(differences.astype(jnp.float32))
            / (energy + 1e-8)
        )
        normalized_mean = normalized_energy.sum() / scout_count
        grouped = defaultdict(list)

        for param, keys, classification in zip(flat_params, flat_keys, flat_es):
            if int(classification) != LORA:
                continue
            out_features, in_features = param.shape[-2:]
            module_keys = keys.reshape((-1,))
            module_param = param.reshape((-1, out_features, in_features))[0]
            factor_out, factor_in = _module_base_factors(
                frozen_noiser_params,
                module_param,
                module_keys,
                pair_epochs,
                pair_threads,
            )
            state = noiser_params["geometry"][_shape_name(param.shape)]
            out32 = factor_out.astype(jnp.float32)
            in32 = factor_in.astype(jnp.float32)
            projected_out = jnp.einsum("mpo,ok->mpk", out32, state["u"])
            projected_in = jnp.einsum("mpi,ik->mpk", in32, state["v"])
            tangent_u = jnp.einsum(
                "p,mpo,mpk->mok", normalized_energy, out32, projected_out
            ) / scout_count - normalized_mean * state["u"][None, :, :]
            tangent_v = jnp.einsum(
                "p,mpi,mpk->mik", normalized_energy, in32, projected_in
            ) / scout_count - normalized_mean * state["v"][None, :, :]
            grouped[_shape_name(param.shape)].append((tangent_u, tangent_v))

        result = {}
        for name, pieces in grouped.items():
            all_u = jnp.concatenate([piece[0] for piece in pieces], axis=0)
            all_v = jnp.concatenate([piece[1] for piece in pieces], axis=0)
            # The shared basis represents the aggregate operator over modules.
            # Sum module moments; averaging would dilute the signal by the
            # number of same-shaped matrices after global-energy normalization.
            result[name] = (all_u.sum(axis=0), all_v.sum(axis=0), energy)
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
        differences, pair_epochs, pair_threads = _pair_data(fitnesses, iterinfos)
        scout_weights, _ = _pair_masks(
            frozen_noiser_params, noiser_params, pair_threads
        )
        tangents = cls._geometry_tangents(
            frozen_noiser_params,
            noiser_params,
            params,
            base_keys,
            es_map,
            differences,
            pair_epochs,
            pair_threads,
            scout_weights,
        )
        scout_count = scout_weights.sum().astype(jnp.int32)
        seen_pairs = noiser_params["geometry_seen_pairs"] + scout_count
        epoch = pair_epochs[0]
        geometry_energy = next(iter(tangents.values()))[2]
        should_update = jnp.logical_and(
            geometry_energy > 1e-8,
            jnp.logical_and(
                frozen_noiser_params["product_space_geometry_lr"] > 0.0,
                (epoch + 1)
                % frozen_noiser_params["product_space_geometry_update_every"]
                == 0,
            ),
        )
        decay = frozen_noiser_params["product_space_geometry_ema_decay"]
        learning_rate = frozen_noiser_params["product_space_geometry_lr"]
        new_geometry = {}
        churn_values = []
        for name, state in noiser_params["geometry"].items():
            tangent_u, tangent_v, energy = tangents[name]
            ema_u = decay * state["ema_u"] + (1.0 - decay) * tangent_u
            ema_v = decay * state["ema_v"] + (1.0 - decay) * tangent_v
            candidate_u = _orthonormalize(state["u"] + learning_rate * ema_u)
            candidate_v = _orthonormalize(state["v"] + learning_rate * ema_v)
            new_u = jnp.where(should_update, candidate_u, state["u"])
            new_v = jnp.where(should_update, candidate_v, state["v"])
            rank = frozen_noiser_params["product_space_rank"]
            overlap_u = jnp.sum(jnp.square(state["u"].T @ new_u)) / rank
            overlap_v = jnp.sum(jnp.square(state["v"].T @ new_v)) / rank
            churn_values.append(1.0 - 0.5 * (overlap_u + overlap_v))
            new_geometry[name] = {
                "u": new_u,
                "v": new_v,
                "ema_u": ema_u,
                "ema_v": ema_v,
            }
        return noiser_params | {
            "geometry": new_geometry,
            "geometry_seen_pairs": seen_pairs,
            "geometry_update_count": noiser_params["geometry_update_count"]
            + should_update.astype(jnp.int32),
            "basis_churn": jnp.mean(jnp.stack(churn_values)),
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
        warmup = (
            noiser_params["geometry_seen_pairs"]
            < frozen_noiser_params["product_space_warmup_pairs"]
        )

        def baseline_branch(values):
            dynamic, model = values
            return EggRoll.do_updates(
                frozen_noiser_params,
                dynamic,
                model,
                base_keys,
                fitnesses,
                iterinfos,
                es_map,
            )

        def product_branch(values):
            dynamic, model = values
            return cls._product_update(
                frozen_noiser_params,
                dynamic,
                model,
                base_keys,
                fitnesses,
                iterinfos,
                es_map,
            )

        updated_state, new_params = jax.lax.cond(
            warmup, baseline_branch, product_branch, (noiser_params, params)
        )
        geometry_state = cls._update_geometry(
            frozen_noiser_params,
            updated_state,
            params,
            base_keys,
            fitnesses,
            iterinfos,
            es_map,
        )
        return geometry_state, new_params


def geometry_diagnostics(noiser_params):
    return {
        "geometry_seen_pairs": noiser_params["geometry_seen_pairs"],
        "geometry_update_count": noiser_params["geometry_update_count"],
        "geometry_basis_churn": noiser_params["basis_churn"],
        "geometry_active_to_scout_energy": noiser_params[
            "active_to_scout_energy"
        ],
        "geometry_residual_variance_ratio": noiser_params[
            "residual_variance_ratio"
        ],
        "geometry_scout_prediction_correlation": noiser_params[
            "scout_prediction_correlation"
        ],
    }
