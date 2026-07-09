"""Tests for the standalone fully labelled Countdown collector."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from llm_experiments.collect_countdown_oracle_dataset import (
    Args,
    _already_complete,
    build_row_index,
    expand_row_labels,
    shard_member_ids,
    zero_padded_prompt_lengths,
)
from llm_experiments.utils import build_generate_batch_with_preview


class _FakeRolloutModel:
    @classmethod
    def default_state(cls, params, config):
        del params
        return {
            "position": jnp.asarray(0, dtype=jnp.int32),
            "preview_hidden": jnp.zeros(
                (len(config["preview_layers"]), config["hidden_size"]),
                dtype=jnp.float32,
            ),
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
        input_token,
        state,
        *,
        length,
        return_hidden=False,
    ):
        del noiser, frozen_noiser_params, params, base_evo_keys, input_token
        _, member_id = iterinfo
        position = state["position"]
        sign = jnp.where(member_id % 2 == 0, 1.0, -1.0)
        pair_scale = member_id // 2 + 1
        layer_scale = jnp.arange(1, 3, dtype=jnp.float32)[:, None]
        candidate = jnp.ones((2, 3), dtype=jnp.float32) + (
            sign
            * noiser_params["sigma"]
            * (position + 1).astype(jnp.float32)
            * pair_scale
            * layer_scale
        )
        should_capture = jnp.asarray(bool(config.get("preview_layers"))) & (
            position == length - 1
        )
        state = state | {
            "position": position + 1,
            "preview_hidden": jnp.where(
                should_capture, candidate, state["preview_hidden"]
            ),
        }
        hidden = jnp.zeros((1, 3), dtype=jnp.float32)
        if return_hidden:
            return hidden, state
        logits = jnp.full((1, 8), -1.0, dtype=jnp.float32).at[0, 6].set(1.0)
        return logits, state


def test_fused_rollout_captures_the_final_prompt_token_without_second_prefill():
    generator = build_generate_batch_with_preview(
        _FakeRolloutModel,
        object(),
        {},
        {"layer_types": ["x", "x"], "hidden_size": 3},
        {},
        jax.random.key(9),
        (0, 1),
        temperature=0.0,
        center_rms_floor=1e-6,
    )
    compiled = jax.jit(generator)
    tokens, inputs, center_rms = compiled(
        {"sigma": jnp.asarray(0.01, dtype=jnp.float32)},
        {},
        jnp.asarray([3, 4, 0, 0, 0], dtype=jnp.int32),
        jnp.asarray(2, dtype=jnp.int32),
        jnp.asarray([0, 1, 2, 3], dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    tokens = np.asarray(tokens)
    inputs = np.asarray(inputs)
    center_rms = np.asarray(center_rms)

    assert tokens.shape == (4, 5)
    npt.assert_array_equal(tokens[:, :2], np.asarray([[3, 4]] * 4))
    npt.assert_array_equal(tokens[:, 2:], np.asarray([[6, 6, 6]] * 4))
    assert inputs.shape == (2, 2, 3)
    npt.assert_allclose(center_rms, np.ones((2, 2)), rtol=2e-5, atol=2e-5)
    # Prompt length two means the captured position is one, hence factor two.
    npt.assert_allclose(inputs[0, 0], 2.0, rtol=2e-5, atol=2e-5)
    npt.assert_allclose(inputs[0, 1], 4.0, rtol=2e-5, atol=2e-5)
    # Global pair 1 has twice pair 0's synthetic perturbation response.
    npt.assert_allclose(inputs[1], 2.0 * inputs[0], rtol=2e-5, atol=2e-5)


def test_row_index_and_raw_fitness_labels_are_traceable():
    rows = build_row_index(2, 2, 3)
    assert rows.shape == (12,)
    npt.assert_array_equal(rows["sample_id"], [0] * 6 + [1] * 6)
    npt.assert_array_equal(rows["layer_id"], [0, 1, 2] * 4)
    npt.assert_array_equal(rows["global_pair_id"], [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3)
    npt.assert_array_equal(rows["positive_member_id"], 2 * rows["global_pair_id"])
    npt.assert_array_equal(rows["negative_member_id"], 2 * rows["global_pair_id"] + 1)

    labels = expand_row_labels(np.asarray([[1.1, 0.1], [0.0, 0.06]]), 3)
    assert labels.shape == (6, 3)
    npt.assert_allclose(labels[:3], [[1.1, 0.1, 1.0]] * 3, atol=1e-7)
    npt.assert_allclose(labels[3:], [[0.0, 0.06, -0.06]] * 3, atol=1e-7)


def test_prompt_lengths_use_only_trailing_zero_padding():
    prompts = np.asarray([[4, 5, 0, 0], [7, 8, 9, 0]], dtype=np.int32)
    npt.assert_array_equal(zero_padded_prompt_lengths(prompts), [2, 3])


def test_member_sharding_keeps_complete_pairs_and_global_order():
    members = np.arange(64, dtype=np.int32)
    sharded = shard_member_ids(members, 2)
    assert sharded.shape == (2, 32)
    npt.assert_array_equal(sharded.reshape(-1), members)
    npt.assert_array_equal(sharded[:, 0] % 2, 0)
    with npt.assert_raises(ValueError):
        shard_member_ids(members, 3)


def test_resume_refuses_committed_samples_without_run_config():
    with tempfile.TemporaryDirectory() as temporary_directory:
        output_directory = Path(temporary_directory)
        np.save(
            output_directory / "completed_samples.npy",
            np.asarray([1, 0], dtype=np.uint8),
        )
        with npt.assert_raises(ValueError):
            _already_complete(
                output_directory,
                Args(
                    output_directory=str(output_directory),
                    dataset_size=2,
                    directions_per_prompt=1,
                ),
            )


class CountdownOracleDatasetTest(unittest.TestCase):
    pass


def _as_unittest_method(test_function):
    def method(self):
        del self
        test_function()

    method.__name__ = test_function.__name__
    return method


for _name, _test in list(globals().items()):
    if _name.startswith("test_") and callable(_test):
        setattr(CountdownOracleDatasetTest, _name, _as_unittest_method(_test))


if __name__ == "__main__":
    unittest.main()
