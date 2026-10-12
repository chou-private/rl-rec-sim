import numpy as np
from src.core.envs.Simulated_Env.base import BaseSimulatedEnv
from src.core.envs.Simulated_Env.yahoo_calibration import calibrated_reward
from src.core.envs.YahooR3.YahooEnv import YahooEnv


class UserPoolYahooEnv(YahooEnv):
    """Keep original user IDs while restricting calibration train/test populations."""
    def __init__(self, user_ids, **kwargs):
        self.user_ids = np.asarray(user_ids, dtype=int)
        if len(self.user_ids) == 0:
            raise ValueError('Empty user pool')
        super().__init__(**kwargs)
        if self.user_ids.min() < 0 or self.user_ids.max() >= len(self.mat):
            raise ValueError('User pool outside environment matrix')

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.cum_reward, self.total_turn = 0, 0
        self.cur_user = int(np.random.choice(self.user_ids))
        self.action = np.random.randint(self.mat.shape[1]) if self.random_init else self.mat.shape[1]
        self._reset_history()
        return self.state, {'cum_reward': 0.0}


class CalibratedYahooEnv(BaseSimulatedEnv):
    def __init__(self, risk_mat, calibration_lambda, **kwargs):
        if not np.isfinite(calibration_lambda) or calibration_lambda < 0:
            raise ValueError('Invalid calibration_lambda')
        self.risk_mat = risk_mat
        self.calibration_lambda = calibration_lambda
        super().__init__(**kwargs)
        self.MIN_R, self.MAX_R = -calibration_lambda, 1.0

    def _compute_training_reward(self, action, t):
        item = self._to_action_index(action)
        return float(calibrated_reward(self.predicted_mat[self.cur_user, item],
                                       self.risk_mat[self.cur_user, item], self.calibration_lambda))
