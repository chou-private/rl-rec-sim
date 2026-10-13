#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root; quick pilot defaults to one member in all arms.
python_bin="${PYTHON_BIN:-python}"
n_models="${N_MODELS:-1}"
prediction="${YAHOO_SURVEY_PREDICTION:-saved_models/YahooEnv-v0/DeepFM/matsPre/[pointneg-weak01-s2022]_matPre.pickle}"
if [[ ! -f "$prediction" ]]; then
  echo "Missing frozen matrix: $prediction" >&2
  exit 1
fi
for mode in uniform survey shuffled; do
  "$python_bin" examples/usermodel/run_DeepFM_ensemble.py \
    --env YahooEnv-v0 --seed 2022 --sampling_seed 2022 --cuda 0 \
    --epoch 5 --n_models "$n_models" --batch_size 2048 \
    --neg_K 5 --no_userinfo --loss pointneg --negative_weight 0.1 \
    --survey_negative_mode "$mode" --survey_shuffle_seed 2022 \
    --survey_negative_prediction "$prediction" \
    --message "surveyneg-${mode}-m${n_models}-s2022"
done
