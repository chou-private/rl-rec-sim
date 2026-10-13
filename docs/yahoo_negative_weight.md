# Yahoo observed-only and weak sampled-negative controls

Use branch `codex/yahoo-survey-calibrated-reward`. This experiment changes DeepFM training loss, not DQN reward. It does not add survey features or change the existing sampler. Sampled candidates can include observed pairs as well as missing pairs; this is a loss-weight diagnostic, not a strict missing-only sampling experiment.

`--negative_weight` is used by `--loss pointneg` only. The default 1 preserves the existing objective; 0 removes the complete sampled-negative contribution; 0.1 weakens it to one tenth. Both squared-error and negative log-variance terms are weighted together. Accepted weights are [0,1].

After pulling the branch on the server, no patch application is needed:

```bash
cd ~/rl-rec-sim
git fetch origin
git switch codex/yahoo-survey-calibrated-reward
git pull --ff-only
conda activate easyrl4rec

python examples/usermodel/run_DeepFM_ensemble.py --env YahooEnv-v0 --seed 2022 --cuda 0 --epoch 5 --n_models 5 --batch_size 2048 --neg_K 5 --no_userinfo --loss point --message point-observed-only-s2022
python examples/usermodel/run_DeepFM_ensemble.py --env YahooEnv-v0 --seed 2022 --cuda 0 --epoch 5 --n_models 5 --batch_size 2048 --neg_K 5 --no_userinfo --loss pointneg --negative_weight 0.1 --message pointneg-weak01-s2022
```

Keep neg_K=5 even for `point`: the existing data pipeline still constructs sampled pairs, but the point loss ignores them. Setting neg_K=0 is unsupported. Original base results may be used as a preliminary reference; formal comparisons should rerun matched controls when code, sampling or other settings differ.

Inspect logs, params and prediction matrices under `saved_models/YahooEnv-v0/DeepFM`. Compare prediction and greedy diagnostics before training DQN. The default runner evaluates the original Yahoo random-exposure test data; repeated exploratory comparisons on it should not be described as untouched final evaluation.

Test: `python -m unittest discover -s tests -p test_negative_weight.py`.
