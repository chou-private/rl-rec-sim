"""Fit paired survey/no-survey/shuffled conservative rewards with user holdout."""
import argparse
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core.envs.Simulated_Env.yahoo_calibration import design, predict_risk, matrix_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--ridge', type=float, default=10)
    args = parser.parse_args()
    if args.ridge <= 0:
        raise ValueError('ridge must be positive')
    with args.prediction.open('rb') as stream:
        pred = np.asarray(pickle.load(stream), dtype=np.float64)
    if pred.shape != (5400, 1000) or not np.isfinite(pred).all():
        raise ValueError('Expected finite 5400 x 1000 prediction matrix')
    raw = ROOT / 'data/YahooR3/data_raw'
    survey = np.loadtxt(raw / 'ydata-ymusic-rating-study-v1_0-survey-answers.txt')
    if survey.shape != (5400, 7):
        raise ValueError('Expected 5400 survey rows')
    survey[:, :6] = (survey[:, :6] - 1) / 4
    survey[:, 6] -= 1
    train = pd.read_csv(raw / 'ydata-ymusic-rating-study-v1_0-train.txt', sep=r'\s+', header=None, names=['user', 'item', 'rating'])
    train['user'] -= 1
    stats = train.groupby('user').rating.agg(['count', 'mean', 'std']).reindex(np.arange(5400))
    history = np.column_stack([np.log1p(stats['count']) / 10, (stats['mean'] - 1) / 4, stats['std'].fillna(0) / 4])
    test = np.loadtxt(raw / 'ydata-ymusic-rating-study-v1_0-test.txt')
    users, items, ratings = test[:, 0].astype(int) - 1, test[:, 1].astype(int) - 1, test[:, 2]
    assert np.array_equal(np.unique(users), np.arange(5400))
    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(5400)
    fit_users, val_users, final_users = permutation[:3240], permutation[3240:4320], permutation[4320:]
    fit = np.isin(users, fit_users)
    validation = np.isin(users, val_users)
    # Positive residual after consistent rating clipping, normalized to rating range.
    target = np.maximum(np.clip(pred[users, items], 1, 5) - ratings, 0) / 4
    descriptors = {'no_survey': history, 'survey': np.column_stack([history, survey]),
                   'shuffled': np.column_stack([history, survey[rng.permutation(5400)]])}
    arrays = dict(prediction_hash=matrix_hash(pred), fit_users=fit_users, val_users=val_users, test_users=final_users,
                  seed=args.seed, prediction_path=str(args.prediction.resolve()))
    report = dict(seed=args.seed, fit_users=len(fit_users), validation_users=len(val_users), test_users=len(final_users),
                  prediction_path=str(args.prediction.resolve()), prediction_hash=arrays['prediction_hash'],
                  prediction_min=float(pred.min()), prediction_max=float(pred.max()), models={})
    # No final-test metrics: final users remain reserved until experiment settings are fixed.
    for name, features in descriptors.items():
        x = design(pred[users, items], features[users])
        xf = x[fit]
        coefficients = np.linalg.solve(xf.T @ xf + args.ridge * np.eye(x.shape[1]), xf.T @ target[fit])
        estimate = predict_risk(pred[users[validation], items[validation]], features[users[validation]], coefficients)
        report['models'][name] = dict(validation_MAE=float(np.abs(estimate - target[validation]).mean()),
                                    validation_RMSE=float(np.sqrt(np.mean((estimate - target[validation]) ** 2))))
        risk = np.empty(pred.shape, dtype=np.float32)
        for start in range(0, 5400, 50):
            block = pred[start:start + 50]
            f = np.repeat(features[start:start + 50], 1000, axis=0)
            risk[start:start + len(block)] = predict_risk(block.reshape(-1), f, coefficients).reshape(block.shape)
        arrays['risk_' + name] = risk
        arrays['coefficients_' + name] = coefficients
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
