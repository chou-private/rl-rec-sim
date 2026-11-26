#!/usr/bin/env bash
# Run from the repository root in the project's PyTorch environment.
set -euo pipefail
mode=${1:-}
case "$mode" in
  smoke)
    defaults=(--device cuda --smoke-users-per-group 10 --seeds 2022 --budgets 0 1 3 5
              --warmup-epochs 1 --em-rounds 2 --dim 4 --hidden 8 --batch-users 8 --bootstrap 64)
    ;;
  formal)
    defaults=(--device cuda --seeds 2022 2023 2024 --budgets 0 1 3 5
              --warmup-epochs 2 --em-rounds 20 --batch-users 16)
    ;;
  *)
    echo "Usage: bash scripts/run_yahoo_sparse_em.sh {smoke|formal} [runner options]" >&2
    exit 2
    ;;
esac
shift
exec python -u -m experiments.yahoo_sparse_em.run --output "visual_results/yahoo_sparse_em/${mode}_v1" "${defaults[@]}" "$@"
