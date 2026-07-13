# Preview-control EGGROLL

This experiment keeps EGGROLL's rank-one perturbations and aggregation but
uses full rollouts for a fixed uniform audit sample of a larger virtual
population. The remaining candidates run only the forced prompt prefix.

For the D8 launch script, each epoch has:

- 8 prompts and 8 physical members per prompt (64 full trajectories);
- a virtual factor of 16 (1,024 members / 512 antithetic pairs);
- 32 uniformly audited pairs, one from each size-16 stratum;
- 480 preview-only pairs (960 hidden-only prompt prefills).

## PR6 preview feature

For each positive/negative pair, the model captures the post-block residual at
the final prompt token from two zero-based layers:

1. `floor(0.75 * number_of_layers)`;
2. the final block.

The hidden states are cast to float32. Each layer uses the normalized central
difference

`(h_plus - h_minus) / (2 * sigma * RMS((h_plus + h_minus) / 2))`.

Independent fixed CountSketch projections reduce each layer to 128 signed
bucket sums. PR6 appends six signed response summaries per layer: alignment
with the pair center, mean, midpoint of the extrema, sign balance, bounded
standardized skew, and alignment with the center's sign. The resulting feature
has 268 dimensions (256 sketch + 12 summaries). It is then centered over the
64 candidate pairs belonging to the same prompt. The centering uses neither
rollout labels nor the audit selection, so it does not leak which candidates
will be evaluated.

The summaries add reductions over already-produced hidden states, not another
transformer pass. Preview-only execution bypasses the language-model head and
uses an attention cache sized to the maximum prompt width. Every virtual pair,
including audited pairs, uses this same kernel so the prediction is independent
of the random audit indicator. Audited pairs additionally run through the
ordinary full-rollout kernel. This duplicates 6.25% of prefills in the default
experiment; removing that cost requires a future shared-prefill/decode kernel
with bitwise-identical features.

## Predictor and unbiased update

One task-level, no-intercept ridge regressor predicts the raw antithetic reward
difference. The PR6 launcher returns zero until 128 audited labels have
accumulated. It clips only raw predictions to the configured reward-difference
range. The old one-epoch binary quality gate is replaced by a lagged continuous
control coefficient. On predictions made before their labels are observed, the
coefficient estimates the positive least-squares scale, clips it to `[0, 1]`,
and shrinks it toward zero according to the effective audit count. Negative or
zero prequential alignment therefore disables the learned term; weak evidence
uses only a small term instead of switching the whole virtual population on.

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

Both ridge and calibration statistics are updated only after the model update,
so a label cannot affect its own ES step. The ridge keeps exponentially decayed
raw-coordinate Gram/cross-product statistics and a lagged coordinate RMS
computed from every virtual preview. This allows the RMS to change without
mixing historical examples expressed in incompatible normalized coordinates.

`predictive_surrogate.csv` records current and rolling residual ratios,
calibration scale/slope/confidence, audit correlation, nonzero labels, and
preview/rollout/update timing. The ratio is undefined (`nan`) when all audited
differences are zero. `fitness.csv` continues to report only actual full-rollout
rewards, never virtual utilities.

## Run

```bash
bash llm_experiments/run_qwen35_countdown_preview_control_2h.sh
```

Set `PREDICTIVE_USE_PREDICTIONS=0` for the zero-predictor control. It retains
the identical virtual population, audit design, preview cost, and lagged reward
scale, isolating the learned probe's contribution.

The launcher matches the attached D8 experiment: Qwen3.5-2B, 64 physical
rollouts, 1,024 virtual members, eight generations per prompt, greedy 1,024
token generations, validation every five epochs, and seed 0. Its default
training timer is 6,300 seconds, leaving startup headroom on the target A100;
override `TIME_BUDGET_SECONDS` if compilation on a new machine needs a
different allowance.

The generic CLI keeps PR5's `sketch` and non-centered feature defaults. The
PR6 launcher explicitly opts into `sketch_summary` and prompt centering, so an
unrelated predictive experiment does not silently change feature shape.

To add the resulting line to the old comparison plot:

```bash
python -m llm_experiments.plot_preview_control_compare \
  --pr6 outputs/pr6/.../validation.csv \
  --preview-ridge outputs/pr5/.../validation.csv \
  --baseline outputs/baseline/.../validation.csv \
  --output outputs/preview_control_compare.png
```

The initial reward scale, its decay, and the ridge coefficient are experimental
calibration constants, exposed as `PREDICTIVE_REWARD_SCALE`,
`PREDICTIVE_REWARD_SCALE_DECAY`, and `PREDICTIVE_RIDGE`. The primary run is
greedy (`temperature=0`); stochastic decoding adds rollout noise that
prompt-only features cannot predict.
