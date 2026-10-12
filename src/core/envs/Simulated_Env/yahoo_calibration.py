"""Small, NumPy-only overestimation calibrator for trusted Yahoo artifacts."""
import hashlib
import numpy as np


def matrix_hash(predictions):
    return hashlib.sha256(np.ascontiguousarray(predictions, dtype=np.float64).tobytes()).hexdigest()


def design(predictions, user_features):
    """Piecewise-linear rating basis, crossed with user descriptors."""
    ratings = np.clip(np.asarray(predictions), 1, 5).reshape(-1)
    basis = np.maximum(0, 1 - np.abs(ratings[:, None] - np.arange(1, 6)))
    descriptors = np.column_stack([np.ones(len(ratings)), user_features])
    return (basis[:, :, None] * descriptors[:, None, :]).reshape(len(ratings), -1)


def predict_risk(predictions, user_features, coefficients):
    return np.clip(design(predictions, user_features) @ coefficients, 0, 1)


def calibrated_reward(prediction, risk, penalty):
    if penalty < 0 or not np.isfinite(penalty):
        raise ValueError('calibration_lambda must be finite and nonnegative')
    return (np.clip(prediction, 1, 5) - 1) / 4 - penalty * np.clip(risk, 0, 1)


def load_artifact(path, predictions, mode):
    with np.load(path, allow_pickle=False) as artifact:
        if str(artifact['prediction_hash'].item()) != matrix_hash(predictions):
            raise ValueError('Calibration artifact belongs to a different predicted matrix')
        risk = np.array(artifact['risk_' + mode], copy=True) if mode != 'normalized' else np.zeros_like(predictions)
        train_users = np.array(artifact['fit_users'], copy=True)
        test_users = np.array(artifact['test_users'], copy=True)
    if risk.shape != predictions.shape or not np.isfinite(risk).all():
        raise ValueError('Invalid calibration risk matrix')
    if len(np.intersect1d(train_users, test_users)):
        raise ValueError('Training/evaluation users overlap')
    return risk, train_users, test_users
