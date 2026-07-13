"""Tests for prompt-hidden preview features and pair indexing."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from llm_experiments.utils import (
    build_preview_pair_thread,
    countsketch_antithetic_hidden_states,
    sketch_summary_antithetic_hidden_states,
    summarize_antithetic_hidden_states,
)


def test_countsketch_central_difference_and_swap_antisymmetry():
    sigma = 0.5
    base = np.asarray(
        [[2.0, 2.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    directions = np.asarray(
        [[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, 2.0, -2.0]],
        dtype=np.float32,
    )
    # Choose symmetric offsets so the pair center is exactly ``base`` and the
    # normalized central difference is exactly ``directions``.
    offsets = np.stack((
        directions[0] * 1.0,
        directions[1] * 0.5,
    ))
    positive = base + offsets
    negative = base - offsets
    hidden = jnp.asarray(np.stack((positive, negative)))
    buckets = jnp.asarray([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=jnp.int32)
    signs = jnp.asarray([[1, 1, -1, 1], [1, -1, 1, 1]], dtype=jnp.float32)

    sketch_fn = jax.jit(
        lambda states: countsketch_antithetic_hidden_states(
            states,
            sigma,
            buckets,
            signs,
            num_buckets=2,
            center_rms_floor=1e-6,
        )
    )
    feature = np.asarray(sketch_fn(hidden))
    swapped = np.asarray(sketch_fn(hidden[::-1]))
    zero = np.asarray(sketch_fn(jnp.stack((hidden[0], hidden[0]))))

    npt.assert_allclose(feature, [[-2.0, 6.0, -2.5, 1.0]], atol=1e-6)
    npt.assert_allclose(swapped, -feature, atol=1e-6)
    npt.assert_array_equal(zero, np.zeros_like(zero))


def test_signed_hidden_summaries_are_finite_and_swap_antisymmetric():
    sigma = 0.25
    base = jnp.asarray(
        [[2.0, -1.0, 0.5, 3.0], [1.5, 2.0, -2.0, 0.25]],
        dtype=jnp.float32,
    )
    response = jnp.asarray(
        [[0.5, -1.0, 2.0, 1.5], [-2.0, 0.25, 1.0, 3.0]],
        dtype=jnp.float32,
    )
    hidden = jnp.stack((base + sigma * response, base - sigma * response))

    feature = np.asarray(summarize_antithetic_hidden_states(hidden, sigma))
    swapped = np.asarray(summarize_antithetic_hidden_states(hidden[::-1], sigma))
    zero = np.asarray(
        summarize_antithetic_hidden_states(jnp.stack((hidden[0], hidden[0])), sigma)
    )

    assert feature.shape == (1, 12)
    assert np.all(np.isfinite(feature))
    npt.assert_allclose(swapped, -feature, rtol=1e-6, atol=1e-6)
    npt.assert_array_equal(zero, np.zeros_like(zero))


def test_sketch_summary_appends_six_statistics_per_layer():
    sigma = 0.5
    base = jnp.ones((2, 4), dtype=jnp.float32)
    response = jnp.asarray(
        [[1.0, 2.0, -1.0, 0.5], [0.25, -0.5, 1.5, -2.0]],
        dtype=jnp.float32,
    )
    hidden = jnp.stack((base + sigma * response, base - sigma * response))
    buckets = jnp.asarray([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=jnp.int32)
    signs = jnp.ones((2, 4), dtype=jnp.float32)

    feature = np.asarray(
        sketch_summary_antithetic_hidden_states(
            hidden, sigma, buckets, signs, num_buckets=2
        )
    )
    swapped = np.asarray(
        sketch_summary_antithetic_hidden_states(
            hidden[::-1], sigma, buckets, signs, num_buckets=2
        )
    )

    assert feature.shape == (1, 4 + 12)
    npt.assert_allclose(swapped, -feature, rtol=1e-6, atol=1e-6)


class _FakePreviewModel:
    @classmethod
    def default_state(cls, params, config):
        del params
        return {
            "preview_hidden": jnp.zeros(
                (len(config["preview_layers"]), config["hidden_size"]),
                dtype=jnp.float32,
            )
        }

    @classmethod
    def forward(
        cls,
        noiser,
        frozen_noiser_params,
        noiser_params,
        config,
        params,
        base_evo_keys,
        iterinfo,
        prompt,
        state,
        *,
        length,
        return_hidden,
    ):
        del (
            noiser,
            frozen_noiser_params,
            config,
            params,
            base_evo_keys,
            prompt,
            length,
            return_hidden,
        )
        _, member_id = iterinfo
        pair_scale = member_id // 2 + 1
        sign = jnp.where(member_id % 2 == 0, 1.0, -1.0)
        base = jnp.asarray(
            [[2.0, 2.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0]],
            dtype=jnp.float32,
        )
        response = jnp.asarray(
            [[1.0, -1.0, 0.5, 2.0], [0.25, 1.0, -0.5, 1.5]],
            dtype=jnp.float32,
        )
        state = state | {
            "preview_hidden": base
            + sign * noiser_params["sigma"] * pair_scale * response
        }
        return jnp.zeros((1, 4), dtype=jnp.float32), state


def test_preview_pair_uses_global_pair_ids_and_returns_sketch_only():
    buckets = np.asarray([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.int32)
    signs = np.ones((2, 4), dtype=np.float32)
    preview_pair = build_preview_pair_thread(
        _FakePreviewModel,
        object(),
        {},
        {"layer_types": ["x", "x"], "hidden_size": 4},
        {},
        (0, 1),
        3,
        buckets,
        signs,
        num_buckets=2,
    )
    compiled = jax.jit(preview_pair)
    prompt = jnp.asarray([7, 8, 0], dtype=jnp.int32)
    noiser_params = {"sigma": jnp.asarray(0.01, dtype=jnp.float32)}

    pair_zero = np.asarray(compiled(noiser_params, {}, prompt, 2, 0, 5))
    pair_three = np.asarray(compiled(noiser_params, {}, prompt, 2, 3, 5))

    assert pair_zero.shape == (4,)
    # Pair 3 must use global members 6/7, making its synthetic response 4x
    # pair 0 (global members 0/1). Local microbatch IDs would fail this check.
    npt.assert_allclose(pair_three, 4.0 * pair_zero, rtol=2e-5, atol=2e-5)


def test_preview_pair_can_append_structured_summaries():
    buckets = np.asarray([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.int32)
    signs = np.ones((2, 4), dtype=np.float32)
    preview_pair = build_preview_pair_thread(
        _FakePreviewModel,
        object(),
        {},
        {"layer_types": ["x", "x"], "hidden_size": 4},
        {},
        (0, 1),
        3,
        buckets,
        signs,
        num_buckets=2,
        feature_kind="sketch_summary",
    )
    compiled = jax.jit(preview_pair)
    feature = np.asarray(
        compiled(
            {"sigma": jnp.asarray(0.01, dtype=jnp.float32)},
            {},
            jnp.asarray([7, 8, 0], dtype=jnp.int32),
            2,
            0,
            5,
        )
    )

    assert feature.shape == (16,)
    assert np.all(np.isfinite(feature))


class PreviewFeatureTest(unittest.TestCase):
    pass


def _as_unittest_method(test_function):
    def method(self):
        del self
        test_function()

    method.__name__ = test_function.__name__
    return method


for _name, _test in list(globals().items()):
    if _name.startswith("test_") and callable(_test):
        setattr(PreviewFeatureTest, _name, _as_unittest_method(_test))


if __name__ == "__main__":
    unittest.main()
