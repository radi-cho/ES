"""CPU-only correctness tests for adaptive diagonal EGGROLL."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from hyperscalees.noiser.diag_eggroll import (
    DiagEggRoll,
    _base_lora_factors,
    _pair_even_utilities,
    _project_log_std,
)
from hyperscalees.noiser.eggroll import EggRoll


class DiagEggRollTest(unittest.TestCase):
    def _tiny_state(self, *, warmup_pairs=64, geometry_lr=0.02):
        params = {"weight": jnp.zeros((2, 3), dtype=jnp.float32)}
        es_map = {"weight": 1}
        keys = {"weight": jax.random.key(17)}
        frozen, state = DiagEggRoll.init_noiser(
            params,
            sigma=0.1,
            lr=0.2,
            group_size=8,
            freeze_nonlora=True,
            noise_reuse=1,
            es_map=es_map,
            diag_geometry_warmup_pairs=warmup_pairs,
            diag_geometry_lr=geometry_lr,
            diag_geometry_ema_decay=0.0,
        )
        return params, es_map, keys, frozen, state

    def test_projection_preserves_scale_and_condition_cap(self):
        projected = _project_log_std(
            jnp.asarray([-100.0, -3.0, 7.0, 100.0]), 2.0
        )
        std = np.exp(np.asarray(projected))
        self.assertAlmostEqual(float(np.mean(std**2)), 1.0, places=6)
        self.assertLessEqual(float(std.max() / std.min()), 2.0 + 1e-6)

    def test_warmup_is_bitwise_equal_to_eggroll(self):
        params, es_map, keys, frozen, state = self._tiny_state()
        baseline_frozen, baseline_state = EggRoll.init_noiser(
            params,
            sigma=0.1,
            lr=0.2,
            group_size=8,
            freeze_nonlora=True,
            noise_reuse=1,
        )
        x = jnp.asarray([[0.5, -1.0, 2.0]], dtype=jnp.float32)
        for thread_id in range(8):
            iterinfo = (jnp.int32(0), jnp.int32(thread_id))
            expected = EggRoll.do_mm(
                baseline_frozen,
                baseline_state,
                params["weight"],
                keys["weight"],
                iterinfo,
                x,
            )
            actual = DiagEggRoll.do_mm(
                frozen,
                state,
                params["weight"],
                keys["weight"],
                iterinfo,
                x,
            )
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

        fitnesses = jnp.asarray([0.4, -0.2, 1.1, 0.0, -0.7, 0.9, 0.3, -0.5])
        iterinfos = (
            jnp.zeros(8, dtype=jnp.int32),
            jnp.arange(8, dtype=jnp.int32),
        )
        _, baseline_params = EggRoll.do_updates(
            baseline_frozen,
            baseline_state,
            params,
            keys,
            fitnesses,
            iterinfos,
            es_map,
        )
        state, adaptive_params = DiagEggRoll.do_updates(
            frozen, state, params, keys, fitnesses, iterinfos, es_map
        )
        np.testing.assert_array_equal(
            np.asarray(adaptive_params["weight"]),
            np.asarray(baseline_params["weight"]),
        )
        self.assertEqual(int(state["geometry_update_count"]), 0)

    def test_constant_fitness_does_not_change_geometry(self):
        params, es_map, keys, frozen, state = self._tiny_state(warmup_pairs=0)
        iterinfos = (
            jnp.zeros(8, dtype=jnp.int32),
            jnp.arange(8, dtype=jnp.int32),
        )
        state, _ = DiagEggRoll.do_updates(
            frozen,
            state,
            params,
            keys,
            jnp.ones(8, dtype=jnp.float32),
            iterinfos,
            es_map,
        )
        geometry = state["geometry"]["2x3"]
        np.testing.assert_array_equal(np.asarray(geometry["log_std_out"]), 0.0)
        np.testing.assert_array_equal(np.asarray(geometry["log_std_in"]), 0.0)
        self.assertEqual(int(state["geometry_update_count"]), 0)

    def test_prompt_offsets_do_not_change_pair_utilities(self):
        fitnesses = jnp.asarray(
            [
                1.0, 0.5, -0.2, 0.1, 0.7, -0.4, 0.3, -0.1,
                -1.0, 0.2, 0.5, 0.9, -0.3, 0.4, 0.8, -0.6,
            ],
            dtype=jnp.float32,
        )
        shifted = fitnesses.at[:8].add(100.0).at[8:].add(-37.0)
        left, left_valid = _pair_even_utilities(fitnesses, 8, 3.0, 1e-6)
        right, right_valid = _pair_even_utilities(shifted, 8, 3.0, 1e-6)
        self.assertTrue(bool(left_valid))
        self.assertTrue(bool(right_valid))
        np.testing.assert_allclose(np.asarray(left), np.asarray(right), atol=5e-5)

    def test_adapted_stacked_bfloat16_update_jits(self):
        params = {"weight": jnp.zeros((3, 2, 3), dtype=jnp.bfloat16)}
        es_map = {"weight": 1}
        keys = {"weight": jax.random.split(jax.random.key(43), 3)}
        frozen, state = DiagEggRoll.init_noiser(
            params,
            sigma=0.1,
            lr=0.2,
            group_size=8,
            freeze_nonlora=True,
            noise_reuse=1,
            es_map=es_map,
            diag_geometry_warmup_pairs=0,
            diag_geometry_ema_decay=0.0,
        )
        raw = jnp.asarray([0.3, -0.4, 0.9, 0.1, -0.2, 0.8, 0.5, -0.7])
        fitnesses = DiagEggRoll.convert_fitnesses(frozen, state, raw)
        iterinfos = (
            jnp.zeros(8, dtype=jnp.int32),
            jnp.arange(8, dtype=jnp.int32),
        )
        update = jax.jit(
            lambda dynamic, model: DiagEggRoll.do_updates(
                frozen, dynamic, model, keys, fitnesses, iterinfos, es_map
            )
        )
        state, actual = update(state, params)
        self.assertEqual(actual["weight"].dtype, jnp.bfloat16)
        self.assertEqual(int(state["geometry_update_count"]), 1)
        self.assertTrue(
            np.isfinite(np.asarray(actual["weight"], dtype=np.float32)).all()
        )

    def test_quadratic_signal_shrinks_sharp_row(self):
        pair_count = 4096
        population = pair_count * 2
        params = {"weight": jnp.zeros((2, 2), dtype=jnp.float32)}
        es_map = {"weight": 1}
        keys = {"weight": jax.random.key(29)}
        frozen, state = DiagEggRoll.init_noiser(
            params,
            sigma=0.1,
            lr=0.0,
            group_size=8,
            freeze_nonlora=True,
            noise_reuse=1,
            es_map=es_map,
            diag_geometry_warmup_pairs=0,
            diag_geometry_lr=0.25,
            diag_geometry_ema_decay=0.0,
        )

        def factors(pair_id):
            return _base_lora_factors(
                frozen,
                (jnp.int32(0), pair_id * 2),
                params["weight"],
                keys["weight"],
            )

        out, inn = jax.vmap(factors)(jnp.arange(pair_count, dtype=jnp.int32))
        out = out[..., 0]
        inn = inn[..., 0]
        pair_even = -(
            10.0 * jnp.square(out[:, 0]) + jnp.square(out[:, 1])
        ) * jnp.sum(jnp.square(inn), axis=1)
        raw = jnp.repeat(pair_even[:, None], 2, axis=1).reshape(population)
        fitnesses = DiagEggRoll.convert_fitnesses(frozen, state, raw)
        iterinfos = (
            jnp.zeros(population, dtype=jnp.int32),
            jnp.arange(population, dtype=jnp.int32),
        )
        state, _ = DiagEggRoll.do_updates(
            frozen, state, params, keys, fitnesses, iterinfos, es_map
        )
        std_out = np.exp(np.asarray(state["geometry"]["2x2"]["log_std_out"]))
        self.assertLess(float(std_out[0]), float(std_out[1]))
        self.assertAlmostEqual(float(np.mean(std_out**2)), 1.0, places=5)

    def test_unperturbed_validator_mode_needs_no_geometry(self):
        params = {"weight": jnp.zeros((2, 3), dtype=jnp.float32)}
        frozen, state = DiagEggRoll.init_noiser(
            params, sigma=0.0, lr=0.0, group_size=0
        )
        self.assertNotIn("geometry", state)
        x = jnp.ones((1, 3), dtype=jnp.float32)
        actual = DiagEggRoll.do_mm(
            frozen,
            state,
            params["weight"],
            jax.random.key(0),
            (jnp.int32(0), jnp.int32(0)),
            x,
        )
        np.testing.assert_array_equal(np.asarray(actual), np.zeros((1, 2)))


if __name__ == "__main__":
    unittest.main()
