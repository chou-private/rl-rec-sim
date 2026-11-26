"""Constrained generalized EM: jointly update DeepFM and recording mechanisms."""
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.yahoo_em import run as em
from . import em as frozen
from .core import Recording


def load_recording(root, seed, arm, hidden, device):
    path = root / 'global.pt' if arm == 'global' else root / f'seed_{seed}/{arm}.pt'
    model = Recording(torch.full((5,), .1), arm != 'global', hidden).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model


def soft_counts(ratings, targets):
    """Sufficient statistics for mu(u,star), with missing labels held fixed."""
    q = targets.reshape(*ratings.shape, 5)
    active = np.stack([(ratings == v).sum(1) for v in range(5)], axis=1)
    missing = np.where((ratings < 0)[..., None], q, 0).sum(1, dtype=np.float64)
    if not np.allclose(missing.sum(1), (ratings < 0).sum(1), atol=1e-4):
        raise ValueError('Missing soft counts do not conserve mass')
    return active, missing


def anchor_penalty(mu, anchor, strength):
    """Mean Bernoulli KL(initial || current); fixed throughout EM, all arms."""
    kl = anchor * (anchor.log() - mu.log())
    kl += (1 - anchor) * (torch.log1p(-anchor) - torch.log1p(-mu))
    return strength * kl.mean()


def recording_loss(mu, active, missing, n_pairs):
    return -(active * mu.log() + missing * torch.log1p(-mu)).sum() / n_pairs


@torch.no_grad()
def table_of(recording, features):
    recording.eval()
    return recording(features).cpu().numpy().copy()


def acceptable(q_before, q_after, obs_before, obs_after):
    return bool(np.isfinite([q_after, obs_after]).all()
                and q_after <= q_before + 1e-7 and obs_after <= obs_before + 1e-7)


def fit_joint(ratings, rows_val, groups, features, recording, initial, args, seed, label, out):
    model = em.make_model(ratings, args)
    model.load_state_dict(initial)
    features = torch.as_tensor(features, dtype=torch.float32, device=args.device)
    anchor = recording(features).detach().clone()
    table = table_of(recording, features)
    np.save(out / 'initial_recording_mu.npy', table)
    user_index = np.arange(len(ratings))
    ids = np.arange(ratings.size)
    counts = (ratings >= 0).sum(1)
    initial_val, _ = em.evaluate(model, rows_val, groups, counts, args)
    best_mae, best_round = initial_val['mae'], 0
    best_state = copy.deepcopy(model.state_dict())
    best_recording = copy.deepcopy(recording.state_dict())
    history = []

    def preference(targets=None):
        return em.objective(model, ratings, user_index, table, args.batch_size,
                            args.device, args.l2, targets)

    def penalty():
        with torch.no_grad():
            return anchor_penalty(recording(features), anchor, args.anchor_strength).item()

    obs_before = preference() + penalty()
    for step in range(1, args.em_rounds + 1):
        start = time.monotonic()
        targets = em.e_step(model, ratings, user_index, table, args.batch_size, args.device)
        active, missing = soft_counts(ratings, targets)
        active = torch.as_tensor(active, dtype=torch.float32, device=args.device)
        missing = torch.as_tensor(missing, dtype=torch.float32, device=args.device)
        old = copy.deepcopy(model.state_dict())
        old_recording = copy.deepcopy(recording.state_dict())

        def record_objective():
            mu = recording(features)
            return (recording_loss(mu, active, missing, ratings.size)
                    + anchor_penalty(mu, anchor, args.anchor_strength))

        with torch.no_grad():
            q_before = preference(targets) + record_objective().item()
        accepted = False
        for attempt in range(5):
            # Atomic retry: restore BOTH networks, optimizer state is reset.
            model.load_state_dict(old)
            recording.load_state_dict(old_recording)
            lr = args.lr * .5 ** attempt
            rec_lr = args.recording_lr * .5 ** attempt
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            rng = np.random.default_rng(seed + step * 1009)
            for _ in range(args.m_epochs):
                em.train_pass(model, ratings, targets, ids, args, rng, optimizer)
            optimizer_r = torch.optim.Adam(recording.parameters(), lr=rec_lr)
            recording.train()
            for _ in range(args.recording_steps):
                loss = record_objective()
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite recording objective')
                optimizer_r.zero_grad(set_to_none=True)
                loss.backward()
                optimizer_r.step()
            table = table_of(recording, features)
            with torch.no_grad():
                q_after = preference(targets) + record_objective().item()
            # Both acceptance criteria include the SAME fixed anchor penalty.
            # The raw observed likelihood alone need not be monotone under MAP.
            obs_after = preference() + penalty()
            if acceptable(q_before, q_after, obs_before, obs_after):
                accepted = True
                break
        if not accepted:
            model.load_state_dict(old)
            recording.load_state_dict(old_recording)
            table = table_of(recording, features)
            q_after, obs_after = q_before, obs_before
        val, _ = em.evaluate(model, rows_val, groups, counts, args)
        anchor_value = penalty()
        drift = np.abs(table - anchor.cpu().numpy())
        row = {'round': step, 'accepted': accepted, 'attempts': attempt + 1,
               'lr': lr, 'recording_lr': rec_lr, 'q_before': q_before, 'q_after': q_after,
               'penalized_observed_before': obs_before, 'penalized_observed_after': obs_after,
               'observed_nll_plus_preference_l2': obs_after - anchor_value,
               'anchor_penalty': anchor_value, 'mu_min': float(table.min()),
               'mu_max': float(table.max()), 'mu_mean_abs_drift': float(drift.mean()),
               'mu_max_abs_drift': float(drift.max()), 'validation': val,
               'seconds': time.monotonic() - start}
        history.append(row)
        print(json.dumps({'run': label, 'seed': seed, **row}), flush=True)
        if val['mae'] < best_mae:
            best_mae, best_round = val['mae'], step
            best_state = copy.deepcopy(model.state_dict())
            best_recording = copy.deepcopy(recording.state_dict())
        obs_before = obs_after
        em.write_json(out / 'history.json', history)
        if not accepted:
            break
    model.load_state_dict(best_state)
    recording.load_state_dict(best_recording)
    table = table_of(recording, features)
    np.save(out / 'recording_mu.npy', table)
    selection = {'seed': seed, 'best_round': best_round, 'validation_mae': best_mae,
                 'initial_validation': initial_val, 'n_users': ratings.shape[0],
                 'n_items': ratings.shape[1], 'dim': args.dim, 'hidden': args.hidden,
                 'recording_conditional': recording.conditional,
                 'accepted_rounds': sum(h['accepted'] for h in history),
                 'stop_reason': 'budget' if history[-1]['accepted'] else 'no_acceptable_joint_update',
                 'anchor_strength': args.anchor_strength,
                 'mu_mean_abs_drift': float(np.abs(table-anchor.cpu().numpy()).mean())}
    torch.save({**selection, 'state_dict': {k: v.cpu() for k, v in best_state.items()},
                'recording_state_dict': {k: v.cpu() for k, v in best_recording.items()}}, out / 'best.pt')
    em.write_json(out / 'selection.json', selection)
    if not any(h['accepted'] for h in history):
        raise RuntimeError(f'{label}: no accepted joint EM updates; inspect history')
    return model, best_mae


def parser():
    p = frozen.parser()
    p.description = __doc__
    p.add_argument('--recording-lr', type=float, default=.001)
    p.add_argument('--recording-steps', type=int, default=5)
    p.add_argument('--anchor-strength', type=float, default=1.,
                   help='Mean Bernoulli KL(initial mu || current mu) coefficient; fixed across arms')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    frozen.validate_args(args)
    if (not np.isfinite([args.recording_lr, args.anchor_strength]).all()
            or args.recording_lr <= 0 or args.anchor_strength <= 0 or args.recording_steps < 1):
        raise SystemExit('Joint EM requires positive finite recording LR/anchor and positive steps')
    frozen.main(args, joint=True)
