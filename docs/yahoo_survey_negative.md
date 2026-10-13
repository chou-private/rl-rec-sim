# Quick Yahoo questionnaire penalty experiment

Run the three arms with one training seed, shared negative sampling and identical average negative weight. Default is a quick pilot with one ensemble member and five epochs in every arm. It must not be compared directly against previous five-member ensembles as if only the questionnaire changed.

```bash
git switch codex/yahoo-survey-calibrated-reward
git pull --ff-only origin codex/yahoo-survey-calibrated-reward
conda activate easyrl4rec
bash scripts/run_yahoo_survey_negative.sh
```

Optional full five-member experiment: `N_MODELS=5 bash scripts/run_yahoo_survey_negative.sh`. If the previous 0.1 matrix is stored elsewhere, set `YAHOO_SURVEY_PREDICTION` to its quoted path. The default is `saved_models/YahooEnv-v0/DeepFM/matsPre/[pointneg-weak01-s2022]_matPre.pickle`.

The arms are uniform, survey and shuffled. No questionnaire columns enter DeepFM features. A frozen existing 0.1 model supplies a clipped score in [1,5] for each sampled-negative user/item pair. Q2–Q6 supply the user's ordinal willingness to rate hated through loved songs. Linearly interpolate between score categories, yielding q in [1,5]. Raw strength is `a = 0.25 + 0.75*(q-1)/4`: infrequent rating of a category weakens its sampled-negative zero-target penalty. The constants specify a bounded heuristic sensitivity, not learned probabilities or validated propensities. Q1 and Q7 are not used in this pilot.

For sampled candidates belonging to questionnaire respondents, the multiplier is a divided by its mean across those candidates. Other users receive multiplier 1. Final penalty is `0.1 * multiplier`. Thus respondent and overall sample average are both 0.1, while the nonrespondent weight stays 0.1. This scales both negative squared error and negative log variance. It does not change observed rating targets, impute missing ratings, or change the existing sampler (which can also sample observed pairs).

Shuffled permutes complete respondent Q2–Q6 rows using a private RNG. All arms fix NumPy seed before sampling. Check the positive/negative SHA256 audit fields are identical across arms; the frozen matrix hashes must agree between survey and shuffled. Normalization uses training sampled candidates only and neither reads actual random-exposure test labels nor updates the frozen matrix. The frozen model and existing hyperparameters have already been explored on the test set, so evaluation is exploratory, not untouched final validation.

Interpretation: survey beating uniform and shuffled supports this questionnaire heuristic in this run. Survey and shuffled both improving similarly supports generic heterogeneous weighting, not necessarily the correspondence between answers and users. No improvement rejects neither all questionnaire methods nor metacognition research; this is one bounded mapping with one seed. Report real rating MAE/RMSE, candidate ranking, head diagnostics and simulated greedy separately; no DQN is trained here.

Return for each message `surveyneg-{uniform,survey,shuffled}-m1-s2022`: complete log, matPre pickle, params pickle, and `params/[message]_negative_audit.json`. For N_MODELS=5 the message contains m5. Keep old results. Expected locations are DeepFM/logs, matsPre, params under saved_models/YahooEnv-v0.

Validation: existing negative-weight tests, interpolation/direction, mean-penalty matching, nonrespondent handling, private shuffle RNG, weighted Gaussian penalty gradients, and a real-data dataset construction check. No full server training was performed locally.

## Per-user normalization follow-up

`--survey_negative_normalization user` divides each raw penalty by its mean across that user's training sampled candidates, instead of the overall respondent mean. Every user's average weight stays 0.1; constant answer profiles automatically produce a constant 0.1 penalty. The original global rule remains the default. Frozen predictions, sampler, observed targets and other loss terms stay the same.

```bash
bash scripts/run_yahoo_survey_negative_user.sh
```

This runs only the two new survey/shuffled arms with messages `surveyneg-user-{survey,shuffled}-m1-s2022`. Reuse the previous uniform m1 group only in the same environment with matching sampling hashes. To run all three in a new environment, use `INCLUDE_UNIFORM=1 bash scripts/run_yahoo_survey_negative_user.sh`; the new baseline is `surveyneg-user-uniform-m1-s2022`. The same N_MODELS, PYTHON_BIN and YAHOO_SURVEY_PREDICTION overrides apply. Audit files now include the minimum and maximum of per-user average weights; these should both be 0.1 within floating-point precision for the new rule.

Return complete logs, matPre, params and negative_audit JSON as before. This is a separate normalization ablation, not a strict missing-only sampler or ranking-loss experiment.

## Half-strength questionnaire follow-up (server training only)

`--survey_negative_mix_alpha 0.5` applies `w_mix = 0.1 + 0.5*(w_user - 0.1)`. Alpha defaults to 1 for backward compatibility; 0 recovers uniform weights. Alpha must be finite and in [0,1]. Every user's average remains 0.1, constant profiles and nonrespondents remain uniform, and both negative squared error and log-variance penalties use the mixed weights. The audit records mix_alpha.

```bash
git switch codex/yahoo-survey-calibrated-reward
git pull --ff-only origin codex/yahoo-survey-calibrated-reward
conda activate easyrl4rec
bash scripts/run_yahoo_survey_negative_mix.sh
```

Default: four single-member, five-epoch arms, full/half strength crossed with survey/shuffled. Full-strength runs are included because the previous per-user-normalization experiment ran locally. Compare against the existing SERVER `surveyneg-uniform-m1-s2022` baseline only with matching sampling hashes and the same training setup. Do not mix local full-strength results with server half-strength results. If the baseline/environment/model count changed, run `INCLUDE_UNIFORM=1 bash scripts/run_yahoo_survey_negative_mix.sh` for a fresh baseline too. If full-strength arms already completed in the identical server setup, `RUN_FULL_STRENGTH=0` runs just the two half-strength arms.

New messages: `surveyneg-user-{full,mix05}-{survey,shuffled}-m1-s2022`. They preserve earlier files. Optional fresh uniform message: `surveyneg-mix-uniform-m1-s2022`. Return complete logs, matPre/params pickles and negative_audit JSON for every new arm. The corresponding names contain m5 if N_MODELS=5. PYTHON_BIN and YAHOO_SURVEY_PREDICTION overrides still apply.

Question: does half strength reduce the accuracy loss while retaining the concentration improvement? A return toward baseline on all metrics alone is not evidence of added questionnaire value. Assess real score error, candidate ranking, coverage/concentration and both natural-ending and fixed-length simulated reward separately. This is a preselected alpha sensitivity experiment, not a search for the best alpha. No full local training was run for this update.
