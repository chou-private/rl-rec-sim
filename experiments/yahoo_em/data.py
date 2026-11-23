"""Yahoo data audits, fixed question groups and calibration-only propensities."""
import hashlib
from pathlib import Path
import numpy as np

GROUP_NAMES = ['flat', 'u_shaped', 'positive', 'negative', 'balanced_other']
PREFIX = 'ydata-ymusic-rating-study-v1_0-'


def questionnaire_groups(q):
    a = q[:, 1:6]
    neg, pos, neutral = a[:, :2].mean(1), a[:, 3:].mean(1), a[:, 2]
    flat = a.max(1) == a.min(1)
    ushape = (neg - neutral >= 1) & (pos - neutral >= 1)
    return np.select([flat, ushape, pos > neg, pos < neg], [0, 1, 2, 3], default=4)


def split_users(groups, seed):
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for g in range(5):
        ids = rng.permutation(np.flatnonzero(groups == g))
        if len(ids) < 5:
            raise ValueError('Need at least five users in each group for stratified splitting')
        ncal, nval = int(0.6 * len(ids)), max(1, int(0.2 * len(ids)))
        for out, chunk in zip(parts, (ids[:ncal], ids[ncal:ncal+nval], ids[ncal+nval:])):
            out.extend(chunk.tolist())
    return {k: np.array(sorted(v), dtype=int)
            for k, v in zip(('calibration', 'validation', 'test'), parts)}


def load_yahoo(raw, smoke_users_per_group=0, split_seed=20261123):
    raw = Path(raw)
    paths = {k: raw / (PREFIX + k + '.txt') for k in ('train', 'test', 'survey-answers')}
    train, test, q = [np.loadtxt(paths[k], dtype=np.int64) for k in paths]
    if q.shape != (5400, 7) or not np.isin(q[:, :6], np.arange(1, 6)).all():
        raise ValueError('Expected original 5400 x 7 Yahoo questionnaire')
    if not np.isin(q[:, 6], [1, 2]).all():
        raise ValueError('Invalid Q7')
    for a in (train, test):
        if a.ndim != 2 or a.shape[1] != 3 or not np.isin(a[:, 2], np.arange(1, 6)).all():
            raise ValueError('Invalid ratings')
        if a[:, 0].min() < 1 or a[:, 1].min() < 1 or a[:, 1].max() > 1000:
            raise ValueError('Invalid user/item ID')
        if len(np.unique(a[:, :2], axis=0)) != len(a):
            raise ValueError('Duplicate user-item pair')
    if train[:, 0].max() != 15400 or test[:, 0].max() != 5400:
        raise ValueError('Expected original Yahoo user universe')
    groups = questionnaire_groups(q)
    if np.bincount(groups, minlength=5).tolist() != [2000, 2201, 934, 184, 81]:
        raise ValueError('Questionnaire groups differ from frozen diagnostic definition')
    users = np.arange(5400)
    if smoke_users_per_group:
        rng = np.random.default_rng(split_seed)
        users = np.sort(np.concatenate([rng.choice(np.flatnonzero(groups == g),
                            min(smoke_users_per_group, (groups == g).sum()), replace=False)
                                       for g in range(5)]))
    groups = groups[users]
    mapper = np.full(15400, -1, dtype=int)
    mapper[users] = np.arange(len(users))
    selected = train[mapper[train[:, 0] - 1] >= 0]
    selected_test = test[mapper[test[:, 0] - 1] >= 0].copy()
    ratings = np.full((len(users), 1000), -1, dtype=np.int8)
    ratings[mapper[selected[:, 0] - 1], selected[:, 1] - 1] = selected[:, 2] - 1
    selected_test[:, 0] = mapper[selected_test[:, 0] - 1]
    selected_test[:, 1:] -= 1
    if not (np.bincount(selected_test[:, 0], minlength=len(users)) == 10).all():
        raise ValueError('Expected ten random ratings per user')
    if not ((ratings >= 0).sum(1) >= 10).all():
        raise ValueError('Expected at least ten historical ratings per user')
    overlap = int((ratings[selected_test[:, 0], selected_test[:, 1]] >= 0).sum())
    audit = {'sha256': {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()},
             'group_counts': np.bincount(groups, minlength=5).tolist(),
             'n_users': len(users), 'n_items': 1000,
             'active_ratings': int((ratings >= 0).sum()), 'random_ratings': len(selected_test),
             'active_random_pair_overlap': overlap,
             'note': 'Historical absence is not exposure; original collection/censoring assumptions remain.'}
    return ratings, selected_test, groups, users, split_users(groups, split_seed), audit


def estimate_propensities(ratings, random_ratings, groups, calibration, smoothing=0.5, eps=1e-5):
    """Smoothed random-rating marginal; no validation/test rating values used.

    active_count / (n_users * n_items * estimated random-rating marginal).
    Symmetric Dirichlet pseudocount on random marginal avoids zero denominators.
    Raw counts, unconstrained estimates and every clipping event are audited.
    """
    def estimate(ids):
        a = ratings[ids]
        ac = np.bincount(a[a >= 0], minlength=5)
        rows = random_ratings[np.isin(random_ratings[:, 0], ids)]
        rc = np.bincount(rows[:, 2], minlength=5)
        marginal = (rc + smoothing) / (rc.sum() + 5 * smoothing)
        raw = ac / (len(ids) * ratings.shape[1] * marginal)
        clipped = np.clip(raw, eps, 1 - eps)
        return clipped, {'n_users': len(ids), 'active_counts': ac.tolist(),
                         'random_counts': rc.tolist(), 'random_marginal': marginal.tolist(),
                         'unclipped_mu': raw.tolist(), 'mu': clipped.tolist(),
                         'clipped_stars': (np.flatnonzero(raw != clipped) + 1).tolist()}
    global_mu, global_audit = estimate(calibration)
    group_results = [estimate(calibration[groups[calibration] == g]) for g in range(5)]
    return global_mu, np.array([x[0] for x in group_results]), {
        'global': global_audit, 'groups': {name: x[1] for name, x in zip(GROUP_NAMES, group_results)},
        'random_marginal_pseudocount': smoothing, 'probability_epsilon': eps}


def propensity_table(global_mu, group_mu, strength):
    return (1 - strength) * np.broadcast_to(global_mu, group_mu.shape) + strength * group_mu
