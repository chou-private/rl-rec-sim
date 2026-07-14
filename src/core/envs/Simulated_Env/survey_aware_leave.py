import os

import numpy as np
import pandas as pd

from src.core.envs.Simulated_Env.base import BaseSimulatedEnv


class SurveyAwareLeaveSimulatedEnv(BaseSimulatedEnv):
    """Adds a survey-based leave-risk penalty to the simulated reward.

    The real leave condition is unchanged. Survey answers only control how
    strongly near-leave behavior is penalized during policy learning.
    """

    def __init__(
        self,
        *args,
        lambda_leave_penalty=1.0,
        survey_beta=1.0,
        survey_path=None,
        **kwargs,
    ):
        self.lambda_leave_penalty = lambda_leave_penalty
        self.survey_beta = survey_beta
        self.survey_path = survey_path or os.path.join(
            "data",
            "YahooR3",
            "data_raw",
            "ydata-ymusic-rating-study-v1_0-survey-answers.txt",
        )
        self.user_reactivity = self._load_user_reactivity()
        super().__init__(*args, **kwargs)

    def _load_user_reactivity(self):
        if not os.path.isfile(self.survey_path):
            return None

        survey = pd.read_csv(
            self.survey_path,
            sep=r"\s+",
            header=None,
            names=[
                "rate_frequency",
                "rate_hate",
                "rate_dislike",
                "rate_neutral",
                "rate_like",
                "rate_love",
                "preference_sensitive",
            ],
            dtype=float,
        )
        reaction_cols = ["rate_hate", "rate_dislike", "rate_like", "rate_love"]
        reaction = ((survey[reaction_cols] - 1.0) / 4.0).clip(0.0, 1.0).mean(axis=1)
        return reaction.to_numpy(dtype=float)

    def _get_reactivity(self, user_id):
        if self.user_reactivity is None:
            return 0.0
        if user_id < 0 or user_id >= len(self.user_reactivity):
            return float(np.nanmean(self.user_reactivity))
        return float(self.user_reactivity[user_id])

    def _compute_leave_risk(self, action):
        threshold = float(self.env_task.leave_threshold)
        if threshold <= 0 or self.env_name != "YahooEnv-v0":
            return 0.0

        t = int(self.env_task.total_turn)
        if t == 0:
            return 0.0

        start = max(0, t - int(self.env_task.num_leave_compute))
        window_actions = self.env_task.sequence_action[start:t]
        if not window_actions:
            return 0.0

        action_id = int(np.asarray(action).reshape(-1)[0])
        dist_list = np.array(
            [
                self.env_task.mat_distance[action_id, int(np.asarray(prev).reshape(-1)[0])]
                for prev in window_actions
            ],
            dtype=float,
        )
        min_distance = float(dist_list.min())
        return max(0.0, (threshold - min_distance) / threshold)

    def step(self, action):
        leave_risk = self._compute_leave_risk(action)
        state, pred_reward, terminated, truncated, info = super().step(action)

        reactivity = self._get_reactivity(int(self.cur_user))
        user_lambda = self.lambda_leave_penalty * (1.0 + self.survey_beta * reactivity)
        reward = pred_reward - user_lambda * leave_risk

        self.cum_reward += reward - pred_reward
        info["cum_reward"] = self.cum_reward
        info["pred_reward"] = pred_reward
        info["leave_risk"] = leave_risk
        info["survey_reactivity"] = reactivity
        info["lambda_leave"] = user_lambda
        return state, reward, terminated, truncated, info
