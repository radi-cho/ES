# Predictive virtual-population EGGROLL

This experiment keeps EGGROLL's complete-rollout batch and rank-one kernels,
but estimates an update from a 16-times-larger virtual population.

For the D=8 comparison, every epoch uses eight prompts.  Each prompt has:

- 128 virtual trajectories (64 antithetic pairs),
- four uniformly stratified audited pairs, and
- eight fully generated trajectories in the existing EGGROLL batch.

Thus the accelerator still generates 64 trajectories per epoch, while the
parameter updater receives 1,024 audit-corrected virtual utilities.  Here
"eight rollouts" means eight individual trajectories, or four complete
antithetic pairs.

The online predictor uses signed bilinear measurements of the exact EGGROLL
rank-one factors.  Predictions are frozen before each audit.  The observed
pair residuals receive inverse-probability weight 16, and only then are the
labels added to the predictor for the following epoch.  The virtual utilities
are additionally multiplied by `sqrt(64 / 1024) = 1/4` so EGGROLL's internal
population scaling matches the physical-population baseline.  Reward scaling
uses an EMA frozen from earlier randomized audits rather than a nonlinear
current-virtual-population normalization.

Run the two-hour experiment on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
TIME_BUDGET_SECONDS=7200 \
bash llm_experiments/run_qwen35_countdown_predictive_2h.sh
```

Set `VENV_PYTHON=/path/to/python` if the checkout does not contain `.venv`.
Before the two-hour run, compile and execute one epoch on the target GPU:

```bash
CUDA_VISIBLE_DEVICES=0 RUN_PREFLIGHT=0 NUM_EPOCHS=1 \
TIME_BUDGET_SECONDS=600 \
bash llm_experiments/run_qwen35_countdown_predictive_2h.sh
```

Run the audit-only control, which keeps the same virtual bookkeeping but sets
all predictions to zero:

```bash
CUDA_VISIBLE_DEVICES=0 \
PREDICTIVE_USE_PREDICTIONS=0 \
TIME_BUDGET_SECONDS=7200 \
bash llm_experiments/run_qwen35_countdown_predictive_2h.sh
```

The standard EGGROLL comparator remains:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m llm_experiments.general_do_evolution \
  --task countdown_chat --noiser eggroll \
  --model-choice q35_2B --rwkv-type Qwen35RWKV \
  --parallel-generations-per-gpu 64 --generations-per-prompt 8 \
  --sigma 1e-3 --lr-scale 0.2 --temperature 0.0 \
  --parallel-validations 64 --validation-iterations 10 \
  --thinking-length 1024 --answer-length 0 --validate-every 5 \
  --train-dataset-size 8 --val-dataset-size 256 \
  --time-budget-seconds 7200 --random-train-prompts
```

Important diagnostics in the training log include `predictor_audit_mse`,
`predictor_zero_mse`, `predictor_residual_ratio`, audit correlation, prediction
RMS, corrected-pair RMS, and feature-generation time.  A residual ratio below
one means that the learned prediction improves on the zero control variate on
the randomized audited pairs.
