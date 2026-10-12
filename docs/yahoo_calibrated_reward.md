# Yahoo survey-assisted conservative reward experiment

Branch: `codex/yahoo-survey-calibrated-reward`.

DeepFM is a regression model trained on 1–5 ratings, not a bounded-output model. The current base matrix ranges from about -0.556 to 5.978. All new conditions consistently clip predictions to [1,5]. The reward is:

```
reward = (clip(prediction, 1, 5) - 1) / 4 - lambda * risk
risk target = max(clip(prediction, 1, 5) - observed_rating, 0) / 4
```

Risk is a fitted positive-error estimate, clipped to [0,1], not a dropout probability. Lambda must be finite and nonnegative. DeepFM stays fixed. There is no additional leave penalty or survey sensitivity multiplier in this mode. The original rule-based termination and pseudo-GT evaluation reward are unchanged.

The calibrator uses a small ridge regression with five triangular prediction-rating basis functions crossed with user descriptors. Historical descriptors are log rating count, mean, and standard deviation from the original self-selected training ratings. Survey descriptors are all seven ordinal answers, normalized; they are not treated as actual feedback probabilities. The variants are:

* `normalized`: clipped/normalized rating only, no penalty (scale-matched baseline).
* `no_survey`: risk fitted from prediction and history only.
* `survey`: risk fitted with the additional actual survey answers.
* `shuffled`: identical survey model with permuted user-to-survey assignment.

The artifact uses seed 2022 to split users into 3240 fitting, 1080 validation, and 1080 final-test users. The risk model uses random-exposure labels from fitting users only; validation risk metrics are reported, final labels are not inspected. DQN trains only on fitting users and its epoch evaluation samples validation users. Fix lambda and select the checkpoint on validation before final evaluation. Changing the DQN seed does not change the artifact split. Using random-exposure data for fitting means this is a supervised calibration experiment using additional labels, not survey-only debiasing; `no_survey` has the same labels and fitting budget.

Final evaluation is pseudo-GT simulation on reserved users. It does not establish real satisfaction/retention improvements. Genuine random-exposure ranking on final users remains a separate evaluation task; it is not implemented by this runner. The default 100 sampled evaluation episodes are not an exhaustive evaluation of all 1080 users. Existing model/previous experiments may already have been selected using the Yahoo test data; these splits cannot retroactively undo such selection.

## Server setup

```bash
cd ~/rl-rec-sim
git fetch origin
git switch codex/yahoo-survey-calibrated-reward
git pull --ff-only
conda activate easyrl4rec
```

Keep the complete BASE DeepFM artifacts under `saved_models/YahooEnv-v0/DeepFM`: params, all five model weights, prediction matrix, and the val-user/val-item embeddings needed by the existing state tracker. Do not use the trait model for this controlled experiment. No DeepFM retraining is necessary if these artifacts already exist. The baseline calibration name defaults to `pointneg-main-rerun`; override using `export YAHOO_BASE_MESSAGE=pointneg` if that is your complete base model's message. A params/matrix pair alone is insufficient to train this runner.

## Fit once, then run matched conditions

```bash
bash scripts/run_yahoo_calibrated.sh fit
bash scripts/run_yahoo_calibrated.sh train normalized 2022 1
bash scripts/run_yahoo_calibrated.sh train no_survey 2022 1
bash scripts/run_yahoo_calibrated.sh train survey 2022 1
bash scripts/run_yahoo_calibrated.sh train shuffled 2022 1
```

The normalized condition has identically zero risk, so lambda 1 still gives rating-only reward and matches the state-normalization bounds of the penalty conditions. Repeat controlled conditions at policy seeds 2023/2024 if progressing beyond the pilot. Tune a small common lambda grid on validation users, with equal search budgets.

Training logs/checkpoints are under `saved_models/YahooEnv-v0/DQN/`. Main evaluation comparison should use `NX_0` (no repeat, original termination); `FB` permits repeated evaluation actions and `NX_10` forces length 10. The experiment also repairs the wrapper's exploratory-action mask enforcement so training exploration respects the no-repeat constraint.

After selecting an epoch on validation, evaluate it on the reserved test pool:

```bash
bash scripts/run_yahoo_calibrated.sh eval survey 2022 1 saved_models/YahooEnv-v0/DQN/DQN_cal-survey-l1-s2022-e8.pt
```

Epoch 8 is only a filename example, not a recommended checkpoint. Use the actual selected checkpoint. Train-time final-test evaluation is rejected. Final JSON summarizes mean cumulative reward, length and pooled reward per step for FB/NX_0/NX_10. Do not compare modified training rewards across methods.

Calibration artifacts store the input matrix SHA256 and split membership; loading an artifact with a different matrix is rejected. Generated data and model files are ignored by Git and must be generated on the server. Only experiment code, tests and this guide are committed.
