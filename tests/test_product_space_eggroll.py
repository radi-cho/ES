"""CPU correctness tests for the product-space EGGROLL experiment."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from hyperscalees.noiser.eggroll import EggRoll
from hyperscalees.noiser.product_space_eggroll import (
    ProductSpaceEggRoll,
    _base_lora_factors,
    get_lora_direction,
)


class ProductSpaceEggRollTest(unittest.TestCase):
    def _state(
        self,
        *,
        shape=(4, 5),
        rank=2,
        warmup=256,
        geometry_lr=0.02,
        model_lr=0.2,
        dtype=jnp.float32,
    ):
        params = {"weight": jnp.zeros(shape, dtype=dtype)}
        es_map = {"weight": 1}
        keys = {"weight": jax.random.key(17)}
        frozen, state = ProductSpaceEggRoll.init_noiser(
            params,
            sigma=0.1,
            lr=model_lr,
            group_size=8,
            freeze_nonlora=True,
            noise_reuse=1,
            es_map=es_map,
            product_space_rank=rank,
            product_space_scout_pairs=2,
            product_space_warmup_pairs=warmup,
            product_space_geometry_lr=geometry_lr,
            product_space_geometry_ema_decay=0.0,
        )
        return params, es_map, keys, frozen, state

    def test_00_warmup_forward_and_update_match_eggroll(self):
        params, es_map, keys, frozen, state = self._state()
        baseline_frozen, baseline_state = EggRoll.init_noiser(
            params,
            sigma=0.1,
            lr=0.2,
            group_size=8,
            freeze_nonlora=True,
            noise_reuse=1,
        )
        x = jnp.asarray([[0.5, -1.0, 2.0, 0.3, -0.7]], jnp.float32)
        for thread in range(8):
            info = (jnp.int32(0), jnp.int32(thread))
            expected = EggRoll.do_mm(
                baseline_frozen,
                baseline_state,
                params["weight"],
                keys["weight"],
                info,
                x,
            )
            actual = ProductSpaceEggRoll.do_mm(
                frozen, state, params["weight"], keys["weight"], info, x
            )
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

        scores = jnp.asarray([0.4, -0.2, 1.1, 0.0, -0.7, 0.9, 0.3, -0.5])
        infos = (jnp.zeros(8, jnp.int32), jnp.arange(8, dtype=jnp.int32))
        _, expected_params = EggRoll.do_updates(
            baseline_frozen,
            baseline_state,
            params,
            keys,
            scores,
            infos,
            es_map,
        )
        state, actual_params = ProductSpaceEggRoll.do_updates(
            frozen, state, params, keys, scores, infos, es_map
        )
        np.testing.assert_array_equal(
            np.asarray(actual_params["weight"]),
            np.asarray(expected_params["weight"]),
        )
        self.assertEqual(int(state["geometry_seen_pairs"]), 4)

    def test_active_and_scout_schedule_and_subspace(self):
        params, _, keys, frozen, state = self._state(warmup=0)
        geometry = state["geometry"]["4x5"]
        for thread in range(8):
            info = (jnp.int32(0), jnp.int32(thread))
            out, inn = get_lora_direction(
                frozen, state, info, params["weight"], keys["weight"]
            )
            if thread < 4:
                expected_out, expected_in = _base_lora_factors(
                    frozen, info, params["weight"], keys["weight"]
                )
                np.testing.assert_array_equal(out, expected_out)
                np.testing.assert_array_equal(inn, expected_in)
            else:
                out_residual = out - geometry["u"] @ (geometry["u"].T @ out)
                in_residual = inn - geometry["v"] @ (geometry["v"].T @ inn)
                self.assertLess(float(jnp.linalg.norm(out_residual)), 2e-5)
                self.assertLess(float(jnp.linalg.norm(in_residual)), 2e-5)

        plus = get_lora_direction(
            frozen,
            state,
            (jnp.int32(0), jnp.int32(4)),
            params["weight"],
            keys["weight"],
        )
        minus = get_lora_direction(
            frozen,
            state,
            (jnp.int32(0), jnp.int32(5)),
            params["weight"],
            keys["weight"],
        )
        np.testing.assert_array_equal(plus[0], minus[0])
        np.testing.assert_array_equal(plus[1], minus[1])

    def test_constant_scores_do_not_rotate_geometry(self):
        params, es_map, keys, frozen, state = self._state(
            warmup=128, geometry_lr=0.2, model_lr=0.0
        )
        old_u = np.asarray(state["geometry"]["4x5"]["u"])
        old_v = np.asarray(state["geometry"]["4x5"]["v"])
        infos = (jnp.zeros(8, jnp.int32), jnp.arange(8, dtype=jnp.int32))
        state, _ = ProductSpaceEggRoll.do_updates(
            frozen,
            state,
            params,
            keys,
            jnp.ones(8, jnp.float32),
            infos,
            es_map,
        )
        np.testing.assert_array_equal(state["geometry"]["4x5"]["u"], old_u)
        np.testing.assert_array_equal(state["geometry"]["4x5"]["v"], old_v)

    def test_tomography_moves_toward_rank_one_signal(self):
        pair_count = 128
        params = {"weight": jnp.zeros((2, 3, 3), jnp.float32)}
        es_map = {"weight": 1}
        keys = {"weight": jax.random.split(jax.random.key(29), 2)}
        frozen, state = ProductSpaceEggRoll.init_noiser(
            params,
            sigma=0.1,
            lr=0.0,
            group_size=8,
            freeze_nonlora=True,
            noise_reuse=1,
            es_map=es_map,
            product_space_rank=1,
            product_space_scout_pairs=2,
            product_space_warmup_pairs=pair_count + 1,
            product_space_geometry_lr=0.5,
            product_space_geometry_ema_decay=0.0,
        )
        target_u = jnp.asarray([1.0, 0.0, 0.0])
        target_v = jnp.asarray([0.0, 1.0, 0.0])
        before_u = float(
            jnp.square(state["geometry"]["3x3"]["u"].T @ target_u).squeeze()
        )
        before_v = float(
            jnp.square(state["geometry"]["3x3"]["v"].T @ target_v).squeeze()
        )

        pair_ids = jnp.arange(pair_count, dtype=jnp.int32)

        def response(pair_id):
            def module_response(key):
                out, inn = _base_lora_factors(
                    frozen,
                    (jnp.int32(0), pair_id * 2),
                    params["weight"][0],
                    key,
                )
                return (out @ target_u) * (inn @ target_v)

            return jax.vmap(module_response)(keys["weight"]).sum()

        differences = jax.vmap(response)(pair_ids)
        scores = jnp.stack((0.5 * differences, -0.5 * differences), axis=1).reshape(-1)
        infos = (
            jnp.zeros(2 * pair_count, jnp.int32),
            jnp.arange(2 * pair_count, dtype=jnp.int32),
        )
        state, _ = ProductSpaceEggRoll.do_updates(
            frozen, state, params, keys, scores, infos, es_map
        )
        after_u = float(
            jnp.square(state["geometry"]["3x3"]["u"].T @ target_u).squeeze()
        )
        after_v = float(
            jnp.square(state["geometry"]["3x3"]["v"].T @ target_v).squeeze()
        )
        self.assertGreater(after_u, before_u)
        self.assertGreater(after_v, before_v)

    def test_control_variate_recovers_full_gradient_in_expectation(self):
        pair_count = 256
        params = {"weight": jnp.zeros((2, 3, 3), jnp.float32)}
        es_map = {"weight": 1}
        keys = {"weight": jax.random.split(jax.random.key(17), 2)}
        frozen, state = ProductSpaceEggRoll.init_noiser(
            params,
            sigma=0.1,
            lr=1.0,
            group_size=8,
            freeze_nonlora=True,
            noise_reuse=1,
            es_map=es_map,
            product_space_rank=1,
            product_space_scout_pairs=2,
            product_space_warmup_pairs=0,
            product_space_geometry_lr=0.0,
        )
        u = jnp.asarray([[1.0], [0.0], [0.0]])
        v = jnp.asarray([[0.0], [1.0], [0.0]])
        state["geometry"]["3x3"] = state["geometry"]["3x3"] | {"u": u, "v": v}
        gradients = jnp.asarray(
            [
                [[0.2, 1.0, -0.1], [0.0, 0.1, 0.3], [-0.2, 0.0, 0.15]],
                [[-0.1, 0.6, 0.2], [0.1, -0.2, 0.1], [0.25, 0.0, -0.1]],
            ],
            jnp.float32,
        )
        pair_ids = jnp.arange(pair_count, dtype=jnp.int32)

        def response(pair_id):
            def module_response(gradient, key):
                out, inn = get_lora_direction(
                    frozen,
                    state,
                    (jnp.int32(0), pair_id * 2),
                    params["weight"][0],
                    key,
                )
                return out @ gradient @ inn

            return jax.vmap(module_response)(gradients, keys["weight"]).sum()

        differences = jax.vmap(response)(pair_ids)
        scores = jnp.stack((0.5 * differences, -0.5 * differences), axis=1).reshape(-1)
        infos = (
            jnp.zeros(2 * pair_count, jnp.int32),
            jnp.arange(2 * pair_count, dtype=jnp.int32),
        )
        _, result = ProductSpaceEggRoll.do_updates(
            frozen, state, params, keys, scores, infos, es_map
        )
        estimate = result["weight"] / (
            float(state["sigma"]) * np.sqrt(2 * pair_count) / 2.0
        )
        cosine = jnp.vdot(estimate, gradients) / (
            jnp.linalg.norm(estimate) * jnp.linalg.norm(gradients)
        )
        relative_error = jnp.linalg.norm(estimate - gradients) / jnp.linalg.norm(
            gradients
        )
        self.assertGreater(float(cosine), 0.9)
        self.assertLess(float(relative_error), 0.55)

    def test_adapted_stacked_bfloat16_update_jits(self):
        params = {"weight": jnp.zeros((3, 4, 5), dtype=jnp.bfloat16)}
        es_map = {"weight": 1}
        keys = {"weight": jax.random.split(jax.random.key(43), 3)}
        frozen, state = ProductSpaceEggRoll.init_noiser(
            params,
            sigma=0.1,
            lr=0.2,
            group_size=8,
            freeze_nonlora=True,
            noise_reuse=1,
            es_map=es_map,
            product_space_rank=2,
            product_space_scout_pairs=2,
            product_space_warmup_pairs=0,
            product_space_geometry_ema_decay=0.0,
        )
        scores = jnp.asarray([0.3, -0.4, 0.9, 0.1, -0.2, 0.8, 0.5, -0.7])
        infos = (jnp.zeros(8, jnp.int32), jnp.arange(8, dtype=jnp.int32))
        update = jax.jit(
            lambda dynamic, model: ProductSpaceEggRoll.do_updates(
                frozen, dynamic, model, keys, scores, infos, es_map
            )
        )
        state, actual = update(state, params)
        self.assertEqual(actual["weight"].dtype, jnp.bfloat16)
        self.assertTrue(np.isfinite(np.asarray(actual["weight"], np.float32)).all())
        self.assertEqual(int(state["geometry_update_count"]), 1)

    def test_no_control_variate_keeps_active_covariance_scale(self):
        pair_count = 128
        params = {"weight": jnp.zeros((3, 3), jnp.float32)}
        es_map = {"weight": 1}
        keys = {"weight": jax.random.key(9)}
        frozen, state = ProductSpaceEggRoll.init_noiser(
            params,
            sigma=0.1,
            lr=1.0,
            group_size=8,
            freeze_nonlora=True,
            noise_reuse=1,
            es_map=es_map,
            product_space_rank=1,
            product_space_scout_pairs=2,
            product_space_warmup_pairs=0,
            product_space_geometry_lr=0.0,
            product_space_control_variate=False,
        )
        u = jnp.asarray([[1.0], [0.0], [0.0]])
        v = jnp.asarray([[0.0], [1.0], [0.0]])
        state["geometry"]["3x3"] = state["geometry"]["3x3"] | {"u": u, "v": v}
        gradient = u @ v.T

        def response(pair_id):
            out, inn = get_lora_direction(
                frozen,
                state,
                (jnp.int32(0), pair_id * 2),
                params["weight"],
                keys["weight"],
            )
            return out @ gradient @ inn

        differences = jax.vmap(response)(jnp.arange(pair_count, dtype=jnp.int32))
        scores = jnp.stack((0.5 * differences, -0.5 * differences), axis=1).reshape(-1)
        infos = (
            jnp.zeros(2 * pair_count, jnp.int32),
            jnp.arange(2 * pair_count, dtype=jnp.int32),
        )
        _, result = ProductSpaceEggRoll.do_updates(
            frozen, state, params, keys, scores, infos, es_map
        )
        estimate = result["weight"] / (
            float(state["sigma"]) * np.sqrt(2 * pair_count) / 2.0
        )
        cosine = jnp.vdot(estimate, gradient) / (
            jnp.linalg.norm(estimate) * jnp.linalg.norm(gradient)
        )
        relative_error = jnp.linalg.norm(estimate - gradient) / jnp.linalg.norm(
            gradient
        )
        self.assertGreater(float(cosine), 0.9)
        self.assertLess(float(relative_error), 0.55)

    def test_unperturbed_validator_mode(self):
        params = {"weight": jnp.zeros((4, 5), jnp.float32)}
        frozen, state = ProductSpaceEggRoll.init_noiser(
            params, sigma=0.0, lr=0.0, group_size=0
        )
        self.assertNotIn("geometry", state)
        actual = ProductSpaceEggRoll.do_mm(
            frozen,
            state,
            params["weight"],
            jax.random.key(0),
            (jnp.int32(0), jnp.int32(0)),
            jnp.ones((1, 5)),
        )
        np.testing.assert_array_equal(np.asarray(actual), np.zeros((1, 4)))

    def test_rejects_reused_training_noise(self):
        params = {"weight": jnp.zeros((4, 5), jnp.float32)}
        es_map = {"weight": 1}
        for reuse in (0, 2):
            with self.assertRaisesRegex(ValueError, "noise_reuse=1"):
                ProductSpaceEggRoll.init_noiser(
                    params,
                    sigma=0.1,
                    lr=0.2,
                    group_size=8,
                    freeze_nonlora=True,
                    noise_reuse=reuse,
                    es_map=es_map,
                    product_space_rank=2,
                )


if __name__ == "__main__":
    unittest.main()
