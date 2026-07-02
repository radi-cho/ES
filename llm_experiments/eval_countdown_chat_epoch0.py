"""Run epoch-0 countdown_chat validation only (pipeline smoke test)."""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import tyro
from dataclasses import dataclass

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hyperscalees.models.llm.auto import get_model
from hyperscalees.models.common import simple_es_tree_key
from hyperscalees.models.llm.tokenizer import LegacyWorldTokenizer
import hyperscalees as hs
from llm_experiments.utils import build_validate


@dataclass
class Args:
    model_choice: str = "q35_2B"
    rwkv_type: str = "Qwen35RWKV"
    task: str = "countdown_chat"
    noiser: str = "eggroll"
    seed: int = 0
    dtype: str = "bfloat16"
    temperature: float = 0.0
    thinking_length: int = 1024
    answer_length: int = 0
    parallel_validations: int = 64
    validation_iterations: int = 10
    sigma: float = 1e-3
    lr_scale: float = 0.2
    generations_per_prompt: int = 8
    parallel_generations_per_gpu: int = 64


def main():
    args = tyro.cli(Args)
    args.generation_length = args.thinking_length + args.answer_length
    args.total_parallel_generations = len(jax.devices()) * args.parallel_generations_per_gpu

    if args.model_choice.startswith("q35_") and args.rwkv_type == "BaseRWKV":
        args.rwkv_type = "Qwen35RWKV"

    suppress_eos_token = 0 if args.model_choice[0] == "7" else None
    master_key = jax.random.key(args.seed)
    base_model_key = jax.random.fold_in(master_key, 0)
    base_valid_key = jax.random.fold_in(master_key, 2)

    NOISER = hs.noiser.eggroll.EggRoll
    RWKV, full_params, tokenizer = get_model(
        args.model_choice, rwkv_type=args.rwkv_type, verbose=True, dtype=args.dtype
    )
    legacy_tokenizer = LegacyWorldTokenizer() if args.model_choice[0] == "7" else tokenizer
    config, params, scan_map, _es_map = full_params
    frozen_noiser_params, noiser_params = NOISER.init_noiser(
        params, args.sigma, args.lr_scale, group_size=args.generations_per_prompt
    )
    base_evo_keys = simple_es_tree_key(params, base_model_key, scan_map)
    validate = build_validate(
        RWKV,
        config,
        params,
        base_evo_keys,
        base_valid_key,
        tokenizer,
        legacy_tokenizer,
        args,
        args.temperature,
        suppress_eos_token=suppress_eos_token,
    )
    score = float(validate(params, 0))
    print(f"EPOCH 0 VALIDATION SCORE = {score:.4f}")


if __name__ == "__main__":
    main()
