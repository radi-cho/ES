"""Focused test for the hidden-only LLM execution switch."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy.testing as npt

from hyperscalees.models.llm.llm import LLM


class _HeadMustNotRun(LLM):
    @classmethod
    def embed(cls, common_params, tokens):
        del common_params
        return tokens.ravel()[:, None].astype(jnp.float32)

    @classmethod
    def forward_seq(cls, common_params, x, state, length, new_starts):
        del common_params, length, new_starts
        return 2.0 * x, state | {"visited": jnp.asarray(True)}

    @classmethod
    def outhead(cls, common_params, x):
        del common_params, x
        raise AssertionError("hidden-only preview executed the LM head")


class HiddenBypassTest(unittest.TestCase):
    def test_return_hidden_bypasses_outhead(self):
        hidden, state = _HeadMustNotRun._forward(
            None,
            jnp.asarray([2, 3], dtype=jnp.int32),
            {},
            return_hidden=True,
        )

        npt.assert_array_equal(hidden, [[4.0], [6.0]])
        self.assertTrue(bool(state["visited"]))

    def test_normal_forward_still_calls_outhead(self):
        with self.assertRaisesRegex(AssertionError, "executed the LM head"):
            _HeadMustNotRun._forward(
                None,
                jnp.asarray([2], dtype=jnp.int32),
                {},
            )


if __name__ == "__main__":
    unittest.main()
