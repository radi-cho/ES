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
