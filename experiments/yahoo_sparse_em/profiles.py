"""Estimate a user's voluntary star mix, independently of the EM recording rates."""
from dataclasses import dataclass

import numpy as np
import torch


ARMS = ("global", "own", "shuffled")


def make_support(ratings, max_k, seed, *, user_ids=None):
    """Return nested prefixes of observed (item, star) pairs, both zero-based.

    Negative cells are ineligible. Pass the reserved support pool for training
    users. Using the original user IDs keeps the draws stable in a smoke subset.
    """
    ratings = np.asarray(ratings)
    if ratings.ndim != 2 or max_k < 0:
        raise ValueError("Expected a rating matrix and a nonnegative budget")
    if user_ids is None:
        user_ids = np.arange(len(ratings))
    user_ids = np.asarray(user_ids)
    if (user_ids.shape != (len(ratings),) or np.any(user_ids < 0)
            or len(np.unique(user_ids)) != len(ratings)):
        raise ValueError("One distinct nonnegative original user ID is required per row")
    items = np.empty((len(ratings), max_k), dtype=np.int64)
    for i, uid in enumerate(user_ids):
        observed = np.flatnonzero(ratings[i] >= 0)
        if len(observed) < max_k:
            raise ValueError(f"User {uid} has only {len(observed)} eligible support items")
        rng = np.random.default_rng([seed, int(uid)])
        items[i] = rng.permutation(observed)[:max_k]
    stars = np.take_along_axis(ratings, items, axis=1).astype(np.int64)
    if np.any(stars > 4):
        raise ValueError("Support stars must be encoded as 0..4")
    return items, stars


@dataclass
class ProfilePriors:
    priors: dict
    strength: np.ndarray
    fold_id: np.ndarray
    donor: np.ndarray
    fit_audit: list

    def features(self, arm, support_stars):
        """Posterior mean after observing only the supplied history prefix."""
        stars = np.asarray(support_stars)
        if (stars.ndim != 2 or len(stars) != len(self.strength)
                or not np.issubdtype(stars.dtype, np.integer)
                or np.any((stars < 0) | (stars > 4))):
            raise ValueError("Expected a [users, k] matrix of 0..4 support stars")
        counts = np.stack([(stars == v).sum(axis=1) for v in range(5)], axis=1)
        strength = self.strength[:, None]
        return ((counts + strength * self.priors[arm]) / (stars.shape[1] + strength)).astype(np.float32)


def _survey_prior(x, fit_ids, predict_ids, fractions):
    """Fit a five-class linear softmax to user-equal empirical star fractions."""
    train = torch.from_numpy(x[fit_ids])
    target = torch.from_numpy(fractions)
    weights = torch.zeros((6, 5), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([weights], max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        logp = (train @ weights).log_softmax(dim=1)
        loss = -(target * logp).sum(dim=1).mean() + 0.0005 * weights[1:].square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        probabilities = (torch.from_numpy(x[predict_ids]) @ weights).softmax(dim=1).numpy()
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("Questionnaire profile fit produced nonfinite probabilities")
    return probabilities


def fit_priors(ratings, questions, pools, *, fold_seed=20261126, shuffle_seed=20261124, n_folds=5):
    """Use only calibration users' voluntary scores as profile-fitting labels.

    Calibration priors and concentration estimates are out-of-fold. Donors stay
    within calibration folds or held-out pools, so a fit fold cannot borrow a
    held-out questionnaire. Validation and test priors use all calibration users.
    """
    questions = np.asarray(questions)
    n_users = len(ratings)
    if questions.shape != (n_users, 7) or not np.isin(questions[:, 1:6], np.arange(1, 6)).all():
        raise ValueError("Expected seven raw Yahoo answers, with Q2..Q6 in 1..5")
    cal, val, test = (np.asarray(pools[name], dtype=np.int64)
                      for name in ("calibration", "validation", "test"))
    if not np.array_equal(np.sort(np.concatenate((cal, val, test))), np.arange(n_users)):
        raise ValueError("User pools must partition all matrix rows")
    if n_folds < 2 or len(cal) < 2:
        raise ValueError("Cross-fitting requires at least two folds and two calibration users")

    fold_rng = np.random.default_rng(fold_seed)
    folds = np.array_split(fold_rng.permutation(cal), min(n_folds, len(cal)))
    fold_id = np.full(n_users, -1, dtype=np.int64)
    for i, ids in enumerate(folds):
        fold_id[ids] = i
    donor = np.arange(n_users)
    rng = np.random.default_rng(shuffle_seed)
    for ids in [*folds, val, test]:
        donor[ids] = rng.permutation(ids)

    # No validation/test rating values enter this calculation.
    active = np.asarray(ratings)[cal]
    if not np.isin(active, [-1, 0, 1, 2, 3, 4]).all():
        raise ValueError("Expected calibration ratings in -1 (missing), 0..4 (stars)")
    counts = np.zeros((n_users, 5), dtype=np.int64)
    counts[cal] = np.stack([(active == v).sum(axis=1) for v in range(5)], axis=1)
    if np.any(counts[cal].sum(axis=1) < 2):
        raise ValueError("Need at least two voluntary calibration ratings to estimate strength")
    x = np.column_stack((np.ones(n_users), (questions[:, 1:6] - 3) / 2.0))
    shuffled = x[donor]
    priors = {arm: np.empty((n_users, 5)) for arm in ARMS}
    strength = np.empty(n_users)
    audit = []
    splits = [(cal[fold_id[cal] != i], ids) for i, ids in enumerate(folds)]
    splits.append((cal, np.concatenate((val, test))))
    for chunk, (fit_ids, predict_ids) in enumerate(splits):
        c = counts[fit_ids]
        n = c.sum(axis=1)
        fractions = c / n[:, None]
        mean = fractions.mean(axis=0)
        # Dirichlet concentration from within-user agreement of two distinct
        # ratings; tiny smoke folds may have no positive finite moment estimate.
        match = ((c * (c - 1)).sum(axis=1) / (n * (n - 1))).mean()
        between = match - mean @ mean
        fallback = not (between > 1e-12 and match < 1)
        alpha = 4.0 if fallback else (1 - match) / between
        strength[predict_ids] = alpha
        priors["global"][predict_ids] = mean
        priors["own"][predict_ids] = _survey_prior(x, fit_ids, predict_ids, fractions)
        priors["shuffled"][predict_ids] = _survey_prior(shuffled, fit_ids, predict_ids, fractions)
        audit.append({"prediction_pool": f"fold_{chunk}" if chunk < len(folds) else "validation_and_test",
                      "fit_users": len(fit_ids), "predict_users": len(predict_ids),
                      "strength": float(alpha), "strength_used_fallback": bool(fallback)})
    return ProfilePriors(priors, strength, fold_id, donor, audit)
