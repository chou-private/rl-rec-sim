"""Ordinal survey heuristics for sampled-negative penalties, not exposure probabilities."""
import numpy as np


def survey_negative_multipliers(users, items, prediction, survey, mode, shuffle_seed=2022):
    users, items = np.asarray(users, dtype=int), np.asarray(items, dtype=int)
    prediction, survey = np.asarray(prediction), np.asarray(survey)
    if mode not in ('survey', 'shuffled'):
        raise ValueError('mode must be survey or shuffled')
    if survey.ndim != 2 or survey.shape[1] != 7 or prediction.shape[0] != len(survey):
        raise ValueError('Survey rows must match prediction users; require seven answers')
    if not np.isfinite(prediction).all() or not np.isin(survey[:, 1:6], [1, 2, 3, 4, 5]).all():
        raise ValueError('Invalid predictions or ordinal survey answers')
    if users.shape != items.shape or users.ndim != 1 or (users < 0).any():
        raise ValueError('Invalid candidate users/items')
    if (items < 0).any() or (items >= prediction.shape[1]).any():
        raise ValueError('Candidate item outside prediction matrix')
    answers = survey[:, 1:6].astype(float)
    if mode == 'shuffled':
        answers = answers[np.random.default_rng(shuffle_seed).permutation(len(answers))]
    result = np.ones(len(users), dtype=float)
    known = users < len(survey)
    score = np.clip(prediction[users[known], items[known]], 1., 5.) - 1.
    lower = np.floor(score).astype(int)
    upper = np.minimum(lower + 1, 4)
    fraction = score - lower
    q = answers[users[known], lower] * (1 - fraction) + answers[users[known], upper] * fraction
    # More frequent rating of this score category => stronger zero-target penalty.
    # Bounded, ordinal-only sensitivity rule; never interpret as propensity.
    raw = .25 + .75 * (q - 1.) / 4.
    if len(raw):
        result[known] = raw / raw.mean()
    # Nonrespondents stay at the uniform baseline. Respondents' sample mean is 1.
    return result
