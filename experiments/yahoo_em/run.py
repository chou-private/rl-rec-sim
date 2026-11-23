"""Run paired B/C fixed-propensity generalized EM. See docs/yahoo_deepfm_em.md."""
import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from torch.nn import functional as F

from .data import load_yahoo, estimate_propensities, propensity_table, GROUP_NAMES
from .model import CategoricalDeepFM, missing_posterior, observed_nll, soft_cross_entropy


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def batches(n, size, order=None):
    for start in range(0, n, size):
        yield np.arange(start, min(start + size, n)) if order is None else order[start:start+size]


def pair_tensors(ids, n_items, device):
    ids = torch.as_tensor(ids, dtype=torch.long, device=device)
    return ids // n_items, ids % n_items


def regularization(model, coefficient):
    return coefficient * sum(p.square().sum() for p in model.parameters())


@torch.no_grad()
def e_step(model, ratings, groups, table, batch_size, device):
    """Full E-step snapshot on CPU; observed entries are exact one-hot labels."""
    model.eval()
    n, items = ratings.shape
    flat = ratings.reshape(-1)
    targets = np.empty((n * items, 5), dtype=np.float32)
    for ids in batches(n * items, batch_size):
        u, i = pair_tensors(ids, items, device)
        mu = torch.as_tensor(table[groups[ids // items]], dtype=torch.float32, device=device)
        q = missing_posterior(model(u, i), mu)
        y = torch.as_tensor(flat[ids], dtype=torch.long, device=device)
        known = y >= 0
        q[known] = F.one_hot(y[known], 5).float()
        targets[ids] = q.cpu().numpy()
    if not np.isfinite(targets).all() or not np.allclose(targets.sum(1), 1, atol=1e-6):
        raise RuntimeError('Invalid E-step targets')
    return targets


@torch.no_grad()
def objective(model, ratings, groups, table, batch_size, device, l2, targets=None):
    model.eval()
    n, items = ratings.shape
    total = 0.0
    for ids in batches(n * items, batch_size):
        u, i = pair_tensors(ids, items, device)
        logits = model(u, i)
        if targets is None:
            mu = torch.as_tensor(table[groups[ids // items]], dtype=torch.float32, device=device)
            y = torch.as_tensor(ratings.reshape(-1)[ids], dtype=torch.long, device=device)
            losses = observed_nll(logits, y, mu)
        else:
            q = torch.as_tensor(targets[ids], device=device)
            losses = soft_cross_entropy(logits, q)
        total += losses.double().sum().item()
    return total / (n * items) + regularization(model, l2).item()


def train_pass(model, ratings, targets, ids, args, rng, optimizer):
    """Each pair visited once. No arbitrary observed/missing rebalancing."""
    model.train()
    order = rng.permutation(ids)
    for ix in batches(len(order), args.batch_size, order):
        u, i = pair_tensors(ix, ratings.shape[1], args.device)
        logits = model(u, i)
        if targets is None:
            y = torch.as_tensor(ratings.reshape(-1)[ix], dtype=torch.long, device=args.device)
            loss = F.cross_entropy(logits, y)
        else:
            q = torch.as_tensor(targets[ix], device=args.device)
            loss = soft_cross_entropy(logits, q).mean()
        loss = loss + regularization(model, args.l2)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model, rows, groups, historical_counts, args):
    """Random-label evaluation; binary relevance rating>=4 for candidate NDCG."""
    model.eval()
    probs = []
    for ix in batches(len(rows), args.batch_size):
        u = torch.as_tensor(rows[ix, 0], dtype=torch.long, device=args.device)
        i = torch.as_tensor(rows[ix, 1], dtype=torch.long, device=args.device)
        probs.append(model(u, i).softmax(-1).cpu().numpy())
    p = np.concatenate(probs)
    pred = p @ np.arange(1, 6)
    truth = rows[:, 2] + 1
    error = pred - truth
    nll = -np.log(np.maximum(p[np.arange(len(rows)), rows[:, 2]], 1e-30))
    user_metrics = []
    for user in np.unique(rows[:, 0]):
        ix = np.flatnonzero(rows[:, 0] == user)
        relevance = truth[ix] >= 4
        # Deterministic tie breaking by original item index.
        order = np.lexsort((rows[ix, 1], -pred[ix]))[:5]
        discounts = 1 / np.log2(np.arange(2, 2 + len(order)))
        ideal = discounts[:min(int(relevance.sum()), len(order))].sum()
        ndcg = float((relevance[order] * discounts).sum() / ideal) if ideal else None
        user_metrics.append({'user_index': int(user), 'group': GROUP_NAMES[groups[user]],
                             'historical_count': int(historical_counts[user]),
                             'mae': float(abs(error[ix]).mean()),
                             'mse': float((error[ix] ** 2).mean()),
                             'nll': float(nll[ix].mean()), 'ndcg5': ndcg})
    ndcgs = [r['ndcg5'] for r in user_metrics if r['ndcg5'] is not None]
    summary = {'mae': float(abs(error).mean()), 'rmse': float(np.sqrt((error ** 2).mean())),
               'nll': float(nll.mean()), 'ndcg5': float(np.mean(ndcgs)) if ndcgs else None,
               'ranking_users': len(ndcgs), 'n_users': len(user_metrics)}
    return summary, user_metrics


def subgroup_metrics(users):
    output = {}
    for name in GROUP_NAMES + ['historical_10_to_12', 'historical_13_plus']:
        subset = [r for r in users if (r['group'] == name if name in GROUP_NAMES else
                  (r['historical_count'] <= 12 if name == 'historical_10_to_12'
                   else r['historical_count'] > 12))]
        if subset:
            output[name] = {'n_users': len(subset),
                            **{k: float(np.mean([r[k] for r in subset])) for k in ('mae', 'mse', 'nll')}}
    return output


def make_model(ratings, args):
    return CategoricalDeepFM(*ratings.shape, dim=args.dim, hidden=args.hidden).to(args.device)


def initialize(ratings, args, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = make_model(ratings, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(seed)
    observed = np.flatnonzero(ratings.reshape(-1) >= 0)
    for _ in range(args.warmup_epochs):
        train_pass(model, ratings, None, observed, args, rng, optimizer)
    return copy.deepcopy(model.state_dict())


def fit(ratings, rows_val, groups, table, initial, args, seed, label, out):
    model = make_model(ratings, args)
    model.load_state_dict(initial)
    ids = np.arange(ratings.size)
    counts = (ratings >= 0).sum(1)
    best_mae, best_state, best_round = float('inf'), None, None
    history = []
    obs_before = objective(model, ratings, groups, table, args.batch_size, args.device, args.l2)
    for step in range(1, args.em_rounds + 1):
        start = time.monotonic()
        targets = e_step(model, ratings, groups, table, args.batch_size, args.device)
        old = copy.deepcopy(model.state_dict())
        q_before = objective(model, ratings, groups, table, args.batch_size, args.device, args.l2, targets)
        accepted = False
        # Reset Adam each M-step; identical schedule/order for B/C. Backtrack if
        # minibatch optimization did not improve the full fixed-target objective.
        for attempt in range(5):
            model.load_state_dict(old)
            lr = args.lr * (0.5 ** attempt)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            rng = np.random.default_rng(seed + step * 1009)
            for _ in range(args.m_epochs):
                train_pass(model, ratings, targets, ids, args, rng, optimizer)
            q_after = objective(model, ratings, groups, table, args.batch_size, args.device, args.l2, targets)
            obs_after = objective(model, ratings, groups, table, args.batch_size, args.device, args.l2)
            if np.isfinite(q_after) and q_after <= q_before + 1e-7 and obs_after <= obs_before + 1e-7:
                accepted = True
                break
        if not accepted:
            model.load_state_dict(old)
            q_after, obs_after = q_before, obs_before
        val, _ = evaluate(model, rows_val, groups, counts, args)
        row = {'round': step, 'accepted': accepted, 'attempts': attempt + 1, 'lr': lr,
               'q_before': q_before, 'q_after': q_after,
               'observed_nll_before': obs_before, 'observed_nll_after': obs_after,
               'validation': val, 'seconds': time.monotonic() - start}
        history.append(row)
        print(json.dumps({'run': label, 'seed': seed, **row}), flush=True)
        if val['mae'] < best_mae:
            best_mae, best_round = val['mae'], step
            best_state = copy.deepcopy(model.state_dict())
        obs_before = obs_after
        write_json(out / 'history.json', history)
        if not accepted:
            break
    model.load_state_dict(best_state)
    if not any(h['accepted'] for h in history):
        raise RuntimeError(f'{label} accepted no EM updates; inspect history before continuing')
    checkpoint = {'state_dict': {k: v.cpu() for k, v in best_state.items()},
                  'n_users': ratings.shape[0], 'n_items': ratings.shape[1],
                  'dim': args.dim, 'hidden': args.hidden,
                  'seed': seed, 'best_round': best_round, 'mu': table.tolist(),
                  'validation_mae': best_mae, 'accepted_rounds': sum(h['accepted'] for h in history)}
    torch.save(checkpoint, out / 'best.pt')
    write_json(out / 'selection.json', {k: v for k, v in checkpoint.items() if k != 'state_dict'})
    return model, best_mae


@torch.no_grad()
def export_scores(model, ratings, args, out):
    scores = np.empty(ratings.size, dtype=np.float32)
    stars = torch.arange(1, 6, dtype=torch.float32, device=args.device)
    for ids in batches(ratings.size, args.batch_size):
        u, i = pair_tensors(ids, ratings.shape[1], args.device)
        scores[ids] = (model(u, i).softmax(-1) @ stars).cpu().numpy()
    np.save(out / 'expected_ratings.npy', scores.reshape(ratings.shape))


def bootstrap_difference(b, c, seed, reps=2000):
    rng = np.random.default_rng(seed)
    output = {}
    if [r['user_index'] for r in b] != [r['user_index'] for r in c]:
        raise ValueError('Unpaired evaluation users')
    for metric in ('mae', 'mse', 'nll', 'ndcg5'):
        delta = np.array([y[metric] - x[metric] for x, y in zip(b, c)
                          if x[metric] is not None and y[metric] is not None])
        if len(delta):
            means = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(reps)])
            output[metric] = {'C_minus_B': float(delta.mean()),
                              'ci95': np.quantile(means, [.025, .975]).tolist(), 'users': len(delta)}
    return output


def main(args):
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    ratings, random_rows, groups, original_users, splits, audit = load_yahoo(
        args.data, args.smoke_users_per_group, args.split_seed)
    global_mu, group_mu, mu_audit = estimate_propensities(
        ratings, random_rows, groups, splits['calibration'])
    write_json(out / 'data_audit.json', audit)
    write_json(out / 'propensities.json', mu_audit)
    write_json(out / 'splits.json', {k: (original_users[v] + 1).tolist() for k, v in splits.items()})
    np.save(out / 'original_user_ids.npy', original_users + 1)
    np.save(out / 'groups.npy', groups)
    git = {}
    for key, cmd in [('commit', ['git', 'rev-parse', 'HEAD']),
                     ('branch', ['git', 'branch', '--show-current']),
                     ('dirty', ['git', 'status', '--porcelain'])]:
        git[key] = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    source = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    write_json(out / 'config.json', {'args': vars(args), 'git': git, 'source_sha256': source,
                                   'python': platform.python_version(), 'torch': torch.__version__,
                                   'numpy': np.__version__, 'test_status': 'previously explored; internal holdout',
                                   'loss': 'full-grid mean CE + l2 * sum(parameter^2)'})
    rows_val = random_rows[np.isin(random_rows[:, 0], splits['validation'])]
    # Test label values are used for evaluation only AFTER first-seed validation
    # selection. Neither fit() nor propensity calibration receives test labels.
    selected_strength = None
    results, paired = [], []
    for seed_index, seed in enumerate(args.seeds):
        initial = initialize(ratings, args, seed)
        seed_out = out / f'seed_{seed}'
        seed_out.mkdir()
        b_out = seed_out / 'B'
        b_out.mkdir()
        b, b_val = fit(ratings, rows_val, groups, propensity_table(global_mu, group_mu, 0),
                       initial, args, seed, 'B', b_out)
        if seed_index == 0:
            candidates = [(b_val, 0.0, b_out)]
            for strength in args.strengths:
                c_out = seed_out / f'C_lambda_{strength:g}'
                c_out.mkdir()
                c, val = fit(ratings, rows_val, groups, propensity_table(global_mu, group_mu, strength),
                             initial, args, seed, f'C_lambda_{strength:g}', c_out)
                candidates.append((val, strength, c_out))
                del c
            _, selected_strength, selected_path = min(candidates, key=lambda x: (x[0], x[1]))
            write_json(out / 'selected_strength.json', {
                'selection_seed': seed, 'lambda': selected_strength,
                'criterion': 'validation MAE; ties favor smaller lambda; frozen for remaining seeds',
                'candidates': [{'lambda': x[1], 'mae': x[0]} for x in candidates]})
        elif selected_strength == 0:
            selected_path = b_out
        else:
            selected_path = seed_out / f'C_lambda_{selected_strength:g}'
            selected_path.mkdir()
            c, _ = fit(ratings, rows_val, groups, propensity_table(global_mu, group_mu, selected_strength),
                       initial, args, seed, 'C', selected_path)
            del c
        c = make_model(ratings, args)
        checkpoint = torch.load(selected_path / 'best.pt', map_location=args.device, weights_only=True)
        c.load_state_dict(checkpoint['state_dict'])
        write_json(seed_out / 'C_selected.json', {'lambda': selected_strength,
                                                'checkpoint': str(selected_path.relative_to(out) / 'best.pt')})
        rows_test = random_rows[np.isin(random_rows[:, 0], splits['test'])]
        details = []
        for name, model, directory in [('B', b, b_out), ('C', c, selected_path)]:
            summary, per_user = evaluate(model, rows_test, groups, (ratings >= 0).sum(1), args)
            for r in per_user:
                r['original_user_id'] = int(original_users[r['user_index']] + 1)
            write_json(seed_out / f'{name}_test.json', {'metrics': summary, 'users': per_user,
                                                      'subgroups': subgroup_metrics(per_user)})
            with (seed_out / f'{name}_users.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(per_user[0]))
                writer.writeheader()
                writer.writerows(per_user)
            export_scores(model, ratings, args, directory)
            results.append({'seed': seed, 'model': name, **summary})
            details.append(per_user)
        paired.append({'seed': seed, 'differences': bootstrap_difference(*details, args.split_seed)})
        write_json(out / 'test_results.json', results)
        write_json(out / 'paired_user_bootstrap.json', {
            'note': 'Per-seed user uncertainty, not independent training repeats or causal effects.', 'results': paired})
    aggregate = {}
    for metric in ('mae', 'rmse', 'nll', 'ndcg5'):
        pairs = [(results[i][metric], results[i+1][metric]) for i in range(0, len(results), 2)]
        delta = np.array([c - b for b, c in pairs if b is not None and c is not None])
        if len(delta):
            aggregate[metric] = {'mean_C_minus_B': float(delta.mean()),
                                 'seed_deltas': delta.tolist(),
                                 'sd_across_seeds': float(delta.std(ddof=1)) if len(delta) > 1 else None}
    write_json(out / 'summary.json', {'selected_lambda': selected_strength, 'paired_seed_metrics': aggregate,
                                    'smoke_only': bool(args.smoke_users_per_group),
                                    'status': 'completed'})
    print(f'Completed. Results: {out}', flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/YahooR3/data_raw')
    p.add_argument('--output', required=True, help='New directory; existing outputs are never overwritten')
    p.add_argument('--device', default='auto')
    p.add_argument('--seeds', type=int, nargs='+', default=[2022, 2023, 2024])
    p.add_argument('--split-seed', type=int, default=20261123)
    p.add_argument('--strengths', type=float, nargs='+', default=[0.25, 0.5])
    p.add_argument('--smoke-users-per-group', type=int, default=0)
    p.add_argument('--warmup-epochs', type=int, default=2)
    p.add_argument('--em-rounds', type=int, default=5)
    p.add_argument('--m-epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=4096)
    p.add_argument('--dim', type=int, default=16)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--l2', type=float, default=1e-6)
    p.add_argument('--threads', type=int, default=4)
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if min(args.em_rounds, args.m_epochs, args.batch_size, args.dim, args.hidden, args.threads) < 1:
        raise SystemExit('Rounds, epochs, dimensions, threads and batch size must be positive')
    if args.warmup_epochs < 0 or args.lr <= 0 or args.l2 < 0:
        raise SystemExit('Invalid optimization settings')
    if args.smoke_users_per_group and args.smoke_users_per_group < 5:
        raise SystemExit('Smoke subset requires at least five users per group')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.strengths)) != len(args.strengths):
        raise SystemExit('Duplicate seeds or strengths')
    if not all(0 < x <= 1 for x in args.strengths):
        raise SystemExit('C strengths must be in (0,1]; lambda=0 is already the B candidate')
    main(args)
