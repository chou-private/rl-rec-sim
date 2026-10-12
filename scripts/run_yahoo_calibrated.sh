#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
action="${1:-train}"
mode="${2:-survey}"
seed="${3:-2022}"
penalty="${4:-1.0}"
model_message="${YAHOO_BASE_MESSAGE:-pointneg-main-rerun}"
artifact="${YAHOO_CALIBRATION_ARTIFACT:-saved_models/YahooEnv-v0/calibration/base_s2022.npz}"
common=(--env YahooEnv-v0 --read_message "$model_message" --seed "$seed" --cuda 0
  --simulated_reward_mode yahoo_calibrated --calibration_artifact "$artifact"
  --calibration_mode "$mode" --calibration_lambda "$penalty"
  --remove_recommended_ids --which_tracker avg --reward_handle cat --window_size 3
  --num_leave_compute 3 --leave_threshold 120 --no_userinfo)
case "$action" in
  fit)
    python scripts/fit_yahoo_reward_calibration.py \
      --prediction "saved_models/YahooEnv-v0/DeepFM/matsPre/[$model_message]_matPre.pickle" \
      --output "$artifact" --seed 2022
    ;;
  train)
    python examples/policy/run_DQN.py "${common[@]}" --epoch 10 --is_save \
      --message "cal-$mode-l$penalty-s$seed"
    ;;
  eval)
    checkpoint="${5:?Specify the checkpoint selected on validation users}"
    python examples/policy/run_DQN.py "${common[@]}" --evaluation_only \
      --calibration_eval_split test --policy_checkpoint "$checkpoint" \
      --message "eval-cal-$mode-l$penalty-s$seed" \
      --evaluation_output "visual_results/yahoo_calibration/$mode-l$penalty-s$seed-final.json"
    ;;
  *) echo 'Usage: bash scripts/run_yahoo_calibrated.sh fit|train|eval [mode] [seed] [lambda] [checkpoint]' >&2; exit 2 ;;
esac
