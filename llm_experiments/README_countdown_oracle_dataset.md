# Fully labelled Countdown oracle dataset

This standalone collector creates a fixed offline dataset from the exact
Countdown prompt/scorer and rank-one antithetic EGGROLL perturbations used by
the preview-ridge experiment. Defaults are:

- 256 disjoint training prompts;
- 32 antithetic directions (64 full rollouts) per prompt;
- all 24 Qwen3.5-2B post-block hidden states;
- 196,608 logical layer rows and 16,384 raw final-fitness labels.

Hidden capture is fused into the full rollout. Forced prefix tokens before the
last prompt token bypass the otherwise-unused vocabulary head, and the model
uses a 1,024-token attention cache. The saved predictor input is the float32
pre-CountSketch feature

```text
(h_positive - h_negative)
-----------------------------------------------
2 * sigma * RMS((h_positive + h_negative) / 2)
```

with the configured RMS floor. `row_labels.npy` stores the raw positive
fitness, raw negative fitness, and their difference. These are final outputs
of `CountdownChatTrain.get_batch_fitness`; they are not standardized EGGROLL
utilities. `center_rms.npy` retains the normalization scale, so the unnormalized
hidden difference can also be reconstructed without storing both member states.

The output directory contains memory-mapped `.npy` arrays, `row_index.npy`,
the generated token sequences, source examples, an immutable run config, and
a manifest. A prompt is committed only after all its arrays have been flushed.
Running the same command again resumes incomplete prompts.

When multiple GPUs are visible, complete antithetic pairs are split evenly
across them. The default launcher exposes GPUs 0 and 1, so the 64 trajectories
for each prompt run as 32 members per device while preserving global pair IDs.

Do not randomly split the 196,608 rows: all 24 rows from a prompt/pair share
one label. Split by `sample_id` from `row_index.npy`.

## Run

```bash
bash llm_experiments/run_countdown_oracle_dataset.sh
```

Set `OUTPUT_DIRECTORY`, `VENV_PYTHON`, or `CUDA_VISIBLE_DEVICES` in the
environment if the machine differs from the existing experiment setup.

## Offline oracle baselines

The offline analysis keeps all perturbations from one Countdown prompt in the
same split.  The frozen split contains 192 training prompts, 32 validation
prompts, and 32 test prompts.  Prompt groups are reward-stratified using the
complete labelled dataset: groups never cross splits, but test assignment is
not label-blind.  This is a balanced benchmark split, not a prospective random
holdout.  Install the analysis-only dependencies with:

```bash
python -m pip install -e '.[oracle-analysis]'
```

After the labelled dataset has completed, prepare the grouped split and fixed
feature caches, then fit the model families used in the comparison:

```bash
DATASET=outputs/countdown_oracle_q35_2b_D256_P32_seed0
SEARCH="$DATASET/offline_oracle_v1"

python -m llm_experiments.offline_oracle_search prepare \
  --dataset-directory "$DATASET" \
  --output-directory "$SEARCH"

python -m llm_experiments.offline_oracle_classical \
  --dataset-directory "$DATASET" \
  --search-directory "$SEARCH" \
  --output-directory "$SEARCH/classical_v2_raw_controls"

python -m llm_experiments.offline_oracle_projection_sweep \
  --dataset-directory "$DATASET" \
  --search-directory "$SEARCH" \
  --output-directory "$SEARCH/projection_sweep_v1"

python -m llm_experiments.offline_oracle_latent \
  --dataset-directory "$DATASET" \
  --search-directory "$SEARCH" \
  --output-directory "$SEARCH/latent_v1"

python -m llm_experiments.offline_oracle_ranker \
  --dataset-directory "$DATASET" \
  --search-directory "$SEARCH" \
  --output-directory "$SEARCH/ranker_v1"
```

The five direct models highlighted in the report are:

- `huber_prompt_centered_epsilon1p35_alpha1e-4`;
- `cs512_seed2_identity_ridge100`;
- `pairwise_svc_prompt_center_C1p0`;
- `cs256_seed0_identity_ridge30`;
- `extratrees_raw_leaf20_maxfeat0p5`.

The final evaluator also expects a validation-only stack artifact.  The
following commands create the default stack and evaluate the frozen direct
models, explicitly including the pairwise ranker:

```bash
python -m llm_experiments.offline_oracle_stack \
  --dataset-directory "$DATASET" \
  --search-directory "$SEARCH" \
  --classical-directory "$SEARCH/classical_v2_raw_controls" \
  --output-directory "$SEARCH/stack_v1"

python -m llm_experiments.offline_oracle_final_test \
  --dataset-directory "$DATASET" \
  --search-directory "$SEARCH" \
  --classical-directory "$SEARCH/classical_v2_raw_controls" \
  --projection-directory "$SEARCH/projection_sweep_v1" \
  --latent-directory "$SEARCH/latent_v1" \
  --ranker-directory "$SEARCH/ranker_v1" \
  --ranker-ids pairwise_svc_prompt_center_C1p0 \
  --stack-directory "$SEARCH/stack_v1"
```

`final_test_results.csv` reports prompt-level cosine, pooled Pearson
correlation, MSE, nonzero sign accuracy, and top-k signal-energy recall.
`r2_zero` is `1 - predictor_mse / zero_predictor_mse`: variance reduction
relative to always predicting a zero antithetic reward difference, not the
usual mean-baseline R-squared.  To reproduce the historical run exactly, each
CV fold and validation split fits a scalar calibration on its own labels.
Their calibrated MSE and `r2_zero` values are therefore calibration diagnostics,
not unbiased held-out estimates.  Test predictions use only the scale fitted
on validation and never fit anything to test labels.  The test split is a
one-time evaluation; do not select models or hyperparameters from its results.
