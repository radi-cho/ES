# Preview-predicted EGGROLL

This experiment keeps EGGROLL's rank-one perturbations and aggregation but
uses full rollouts for a fixed uniform audit sample of a larger virtual
population. The remaining candidates run only the forced prompt prefix.

For the D8 launch script, each epoch has:

- 8 prompts and 8 physical members per prompt (64 full trajectories);
- a virtual factor of 16 (1,024 members / 512 antithetic pairs);
- 32 uniformly audited pairs, one from each size-16 stratum;
- 480 preview-only pairs (960 hidden-only prompt prefills).

## Preview feature

For each positive/negative pair, the model captures the post-block residual at
the final prompt token from two zero-based layers:

1. `floor(0.75 * number_of_layers)`;
2. the final block.

The hidden states are cast to float32. Each layer uses the normalized central
difference

`(h_plus - h_minus) / (2 * sigma * RMS((h_plus + h_minus) / 2))`.

Independent fixed CountSketch projections reduce each layer to 128 signed
bucket sums, producing one 256-dimensional feature. Preview-only execution
bypasses the language-model head and uses an attention cache sized to the
maximum prompt width. Every virtual pair, including audited pairs, uses this
same kernel so the prediction is independent of the random audit indicator.
Audited pairs additionally run through the ordinary full-rollout kernel. This
duplicates 6.25% of prefills in the default experiment; removing that cost
requires a future shared-prefill/decode kernel with bitwise-identical features.

## Predictor and unbiased update

One task-level, no-intercept ridge regressor predicts the raw antithetic reward
difference. It returns zero until 256 audited labels have accumulated and clips
only its predictions to the configured reward-difference range. Even after
warmup, predictions enter the ES update only when the previous iteration's
held-out audits beat the zero predictor (default residual-MSE ratio below 0.9)
with at least four nonzero labels. Candidate predictions are still scored while
the gate is closed, so the gate can open on a later iteration.

The audit correction for pair `i` is:

`corrected_i = prediction_i + audit_i / p * (observed_i - prediction_i)`.

Here `p=1/16`. Corrected values are never clipped. They are mapped to adjacent
positive/negative EGGROLL utilities and scaled by `sqrt(physical / virtual)` so
that the zero-predictor update exactly matches a physical EGGROLL update using
the same scale. The scale is frozen within an iteration, then its lagged EMA is
updated from audited member rewards for the next iteration. This does not
reproduce ordinary EGGROLL's current-population standardization; the initial
scale and decay must be calibrated or compared against a same-scale physical
control. Fitness conversion is the identity because current-population
normalization would make the randomized estimator nonlinear.

The predictor is updated only after the model update. It keeps exponentially
decayed raw-coordinate Gram/cross-product statistics and a lagged coordinate
RMS computed from every virtual preview. This allows the RMS to change without
mixing historical examples expressed in incompatible normalized coordinates.

`predictive_surrogate.csv` records audit MSE, the zero-predictor MSE, their
ratio, the number of nonzero audit labels, audit correlation, and
preview/rollout/update timing. The ratio is undefined (`nan`) when all audited
differences are zero. `fitness.csv` continues to report only actual
full-rollout rewards, never virtual utilities.

## Run

```bash
bash llm_experiments/run_qwen35_countdown_predictive_preview_2h.sh
```

Set `PREDICTIVE_USE_PREDICTIONS=0` for the zero-predictor control. It retains
the identical virtual population, audit design, preview cost, and lagged reward
scale, isolating the learned probe's contribution.

The initial reward scale, its decay, and the ridge coefficient are experimental
calibration constants, exposed as `PREDICTIVE_REWARD_SCALE`,
`PREDICTIVE_REWARD_SCALE_DECAY`, and `PREDICTIVE_RIDGE`. The primary run is
greedy (`temperature=0`); stochastic decoding adds rollout noise that
prompt-only features cannot predict.
