"""Compare global, own and shuffled star profiles on random-song ratings."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from experiments.yahoo_em.data import PREFIX, estimate_propensities, load_yahoo
from experiments.yahoo_em.run import write_json
from .evaluation import evaluate_rows, paired_comparisons, summarize_runs
from .model import ContextDeepFM, SparseContext, em_round, warmup
from .profiles import ARMS, fit_priors, make_support


def calibration_partition(ratings, original_ids, seed):
    """Reserve half of each user's catalogue for support. The rest is EM supervision.

    Round the number of reserved observed entries randomly to preserve each
    class's marginal inclusion probability. At least five ratings stay in
    each half for the Yahoo cohort. Reuse the same targets at every budget.
    """
    n_items = ratings.shape[1]
    if n_items % 2:
        raise ValueError("Expected an even catalogue size")
    targets = ratings.copy()
    support = np.full_like(ratings, -1)
    for row, uid in enumerate(original_ids):
        rng = np.random.default_rng([seed, int(uid), 1])
        active = np.flatnonzero(ratings[row] >= 0)
        missing = np.flatnonzero(ratings[row] < 0)
        n_active = (len(active) + int(rng.integers(2))) // 2
        keep = rng.permutation(active)[:n_active]
        excluded = rng.permutation(missing)[:n_items // 2 - n_active]
        support[row, keep] = ratings[row, keep]
        targets[row, keep] = -2
        targets[row, excluded] = -2
    return support, targets


def build_context(items, stars, features, device):
    t = torch.as_tensor(items, dtype=torch.long, device=device)
    return SparseContext(t, torch.as_tensor(stars, dtype=torch.long, device=device),
                         torch.ones_like(t, dtype=torch.bool),
                         torch.as_tensor(features, dtype=torch.float32, device=device))


def select_context(context, ids):
    idx = torch.as_tensor(ids, dtype=torch.long, device=context.items.device)
    return SparseContext(context.items[idx], context.stars[idx], context.mask[idx], context.profile[idx])


@torch.no_grad()
def predict_rows(model, context, rows, batch_users):
    model.eval()
    # The random exposure file has 10 songs per user, grouped by user.
    ids = rows[:, 0].reshape(-1, 10)
    if not np.all(ids == ids[:, :1]):
        raise ValueError("Expected ten contiguous random songs per evaluation user")
    device = context.items.device
    users = torch.as_tensor(ids[:, 0].copy(), dtype=torch.long, device=device)
    songs = torch.as_tensor(rows[:, 1].reshape(-1, 10).copy(), dtype=torch.long, device=device)
    probs = [context.logits(model, users[i:i+batch_users], songs[i:i+batch_users]).softmax(-1).cpu().numpy()
             for i in range(0, len(users), batch_users)]
    return np.concatenate(probs).reshape(-1, 5)


def fit_model(args, seed, model, ctx, cal, labels, mu, val_rows, target):
    training = select_context(ctx, cal)
    warm = warmup(model, training, labels, epochs=args.warmup_epochs, batch_users=args.batch_users,
                  lr=args.lr, l2=args.l2, seed=seed)
    write_json(target / "warmup.json", warm)
    best_mae = float("inf")
    history = []
    for step in range(args.em_rounds + 1):
        update = {} if step == 0 else em_round(
            model, training, labels, mu, m_epochs=1, batch_users=args.batch_users,
            lr=args.lr, l2=args.l2, seed=seed + 1009 * step)
        p = predict_rows(model, ctx, val_rows, args.batch_users)
        mae = float(np.abs(p @ np.arange(1, 6) - (val_rows[:, 2] + 1)).mean())
        if not math.isfinite(mae):
            raise FloatingPointError("Nonfinite validation MAE")
        if mae < best_mae:
            best_mae, best_step = mae, step
            best = deepcopy(model.state_dict())
        history.append(dict(round=step, val_mae=mae, **update))
        print(json.dumps(dict(run=str(target), **history[-1])), flush=True)
        write_json(target / "history.json", history)
        if update and not update["accepted"]:
            break
    if not any(r.get("accepted") for r in history):
        raise RuntimeError(f"No accepted EM steps; inspect {target}/history.json")
    model.load_state_dict(best)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, target / "best.pt")
    write_json(target / "selection.json", {"round": best_step, "validation_mae": best_mae,
                                           "accepted_updates": sum(bool(r.get("accepted")) for r in history)})


def run(args):
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu for a local smoke test")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    ratings, rows, groups, original, pools, audit = load_yahoo(
        args.data, args.smoke_users_per_group, args.split_seed)
    if audit["active_random_pair_overlap"]:
        raise ValueError("Random candidates overlap active history; review protocol before training")
    original = original + 1
    q = np.loadtxt(Path(args.data) / (PREFIX + "survey-answers.txt"), dtype=int)[original - 1]
    cal = pools["calibration"]
    # Reuse the B estimate, exclusively from calibration users' labels.
    random_cal = rows[np.isin(rows[:, 0], cal)]
    mu, _, mu_info = estimate_propensities(ratings, random_cal, groups, cal)
    write_json(out / "recording_B.json", mu_info["global"])
    write_json(out / "splits.json", {key: original[ids].tolist() for key, ids in pools.items()})
    support_pool = ratings.copy()
    support_pool[cal], targets = calibration_partition(ratings[cal], original[cal], args.history_seed)
    items, stars = make_support(support_pool, max(args.budgets), args.history_seed, user_ids=original)
    if np.any(np.take_along_axis(targets, items[cal], axis=1) != -2):
        raise RuntimeError("Visible calibration targets leaked into supervision")
    priors = fit_priors(ratings, q, pools, fold_seed=args.fold_seed, shuffle_seed=args.shuffle_seed)
    np.savez_compressed(out / "contexts.npz", original_ids=original, items=items, stars=stars,
                        strength=priors.strength, fold=priors.fold_id, donor_id=original[priors.donor],
                        **priors.priors)
    np.savez_compressed(out / "calibration_targets.npz", original_ids=original[cal], targets=targets)
    write_json(out / "profile_fits.json", priors.fit_audit)
    source = list(Path(__file__).parent.glob("*.py"))
    source += [Path(__file__).parents[1] / "yahoo_em" / (name + ".py") for name in ("data", "model", "run")]
    write_json(out / "config.json", {"args": vars(args), "audit": audit, "numpy": np.__version__,
                                     "torch": str(torch.__version__), "em_targets": int((targets != -2).sum()),
                                     "observed_em_targets": int((targets >= 0).sum()),
                                     "source_sha256": {p.parent.name + "/" + p.name: hashlib.sha256(
                                         p.read_bytes()).hexdigest() for p in source}})
    labels = torch.as_tensor(targets, dtype=torch.int8, device=args.device)
    val = rows[np.isin(rows[:, 0], pools["validation"])]

    def context(k, arm):
        return build_context(items[:, :k], stars[:, :k], priors.features(arm, stars[:, :k]), args.device)

    def new_model(seed):
        torch.manual_seed(seed)
        return ContextDeepFM(ratings.shape[1], args.dim, args.hidden).to(args.device)

    for seed in args.seeds:
        for k in args.budgets:
            for arm in ARMS:
                target = out / f"seed_{seed}" / f"k_{k}" / arm
                target.mkdir(parents=True)
                fit_model(args, seed, new_model(seed), context(k, arm), cal, labels, mu, val, target)

    test = rows[np.isin(rows[:, 0], pools["test"])]
    results, paired = [], []
    for seed in args.seeds:
        for k in args.budgets:
            by_arm = {}
            for arm in ARMS:
                target = out / f"seed_{seed}" / f"k_{k}" / arm
                model = new_model(seed)
                model.load_state_dict(torch.load(target / "best.pt", map_location=args.device, weights_only=True))
                p = predict_rows(model, context(k, arm), test, args.batch_users)
                metrics, by_arm[arm] = evaluate_rows(test, p, original_user_ids=original)
                results.append(dict(seed=seed, k=k, arm=arm, **metrics))
                write_json(target / "test.json", {"metrics": metrics, "users": by_arm[arm]})
            paired.extend(paired_comparisons(by_arm, seed, k, reps=args.bootstrap))
    write_json(out / "results.json", results)
    write_json(out / "paired_bootstrap.json", paired)
    write_json(out / "summary.json", {"status": "completed", "smoke_only": bool(args.smoke_users_per_group),
                                      "by_budget": summarize_runs(results)})
    print(f"Completed. Results: {out}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/YahooR3/data_raw")
    p.add_argument("--output", required=True, help="New run directory")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--seeds", type=int, nargs="+", default=[2022, 2023, 2024])
    p.add_argument("--budgets", type=int, nargs="+", default=[0, 1, 3, 5])
    p.add_argument("--split-seed", type=int, default=20261123)
    p.add_argument("--history-seed", type=int, default=20261127)
    p.add_argument("--fold-seed", type=int, default=20261126)
    p.add_argument("--shuffle-seed", type=int, default=20261124)
    p.add_argument("--smoke-users-per-group", type=int, default=0)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--em-rounds", type=int, default=20)
    p.add_argument("--batch-users", type=int, default=16)
    p.add_argument("--dim", type=int, default=16)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--l2", type=float, default=1e-6)
    args = p.parse_args()
    if args.smoke_users_per_group != 0 and args.smoke_users_per_group < 5:
        p.error("Use at least five smoke users per questionnaire group")
    if min(args.batch_users, args.dim, args.hidden, args.threads, args.em_rounds, args.bootstrap) < 1:
        p.error("Training dimensions, rounds and bootstrap repetitions must be positive")
    if args.warmup_epochs < 0 or not (math.isfinite(args.lr) and args.lr > 0
                                      and math.isfinite(args.l2) and args.l2 >= 0):
        p.error("Invalid optimization settings")
    if min(*args.seeds, args.history_seed, args.fold_seed, args.shuffle_seed, args.split_seed) < 0:
        p.error("Seeds must be nonnegative")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.budgets)) != len(args.budgets):
        p.error("Duplicate seeds or budgets")
    if not set(args.budgets).issubset(range(6)):
        p.error("Supported history budgets are 0..5")
    return args


if __name__ == "__main__":
    run(parse_args())
