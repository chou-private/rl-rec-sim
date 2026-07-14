import os

import numpy as np
import pandas as pd

from src.core.envs.Simulated_Env.base import BaseSimulatedEnv


class SurveyAwareLeaveSimulatedEnv(BaseSimulatedEnv):
    """Adds survey-based reward sensitivity and a fixed leave-risk penalty.

    The real leave condition is unchanged. Survey answers control how strongly
    positive and negative predicted ratings affect the reward during policy
    learning.
    """

    def __init__(
        self,
        *args,
        lambda_leave_penalty=1.0,
        survey_positive_alpha=1.0,
        survey_negative_beta=1.0,
        positive_rating_threshold=4.0,
        negative_rating_threshold=2.0,
        survey_path=None,
        **kwargs,
    ):
        self.lambda_leave_penalty = lambda_leave_penalty
        self.survey_positive_alpha = survey_positive_alpha
        self.survey_negative_beta = survey_negative_beta
        self.positive_rating_threshold = positive_rating_threshold
        self.negative_rating_threshold = negative_rating_threshold
        self.survey_path = survey_path or os.path.join(
            "data",
            "YahooR3",
            "data_raw",
            "ydata-ymusic-rating-study-v1_0-survey-answers.txt",
        )
        self.user_survey_scores = self._load_user_survey_scores()
        super().__init__(*args, **kwargs)

    def _load_user_survey_scores(self):
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
        norm = ((survey - 1.0) / 4.0).clip(0.0, 1.0)
        pos_score = norm[["rate_like", "rate_love"]].mean(axis=1)
        neg_score = norm[["rate_hate", "rate_dislike"]].mean(axis=1)
        return {
            "positive": pos_score.to_numpy(dtype=float),
            "negative": neg_score.to_numpy(dtype=float),
        }

    def _get_user_score(self, user_id, score_name):
        if self.user_survey_scores is None:
            return 0.0
        scores = self.user_survey_scores[score_name]
        if user_id < 0 or user_id >= len(scores):
            return float(np.nanmean(scores))
        return float(scores[user_id])

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

    def _apply_survey_reward_sensitivity(self, pred_reward):
        raw_rating = pred_reward + self.MIN_R
        user_id = int(self.cur_user)
        pos_score = self._get_user_score(user_id, "positive")
        neg_score = self._get_user_score(user_id, "negative")

        if raw_rating >= self.positive_rating_threshold:
            reward = pred_reward * (1.0 + self.survey_positive_alpha * pos_score)
        elif raw_rating <= self.negative_rating_threshold:
            reward = pred_reward - self.survey_negative_beta * neg_score
        else:
            reward = pred_reward

        return reward, raw_rating, pos_score, neg_score

    def step(self, action):
        leave_risk = self._compute_leave_risk(action)
        state, pred_reward, terminated, truncated, info = super().step(action)

        survey_reward, raw_rating, pos_score, neg_score = self._apply_survey_reward_sensitivity(
            pred_reward
        )
        reward = survey_reward - self.lambda_leave_penalty * leave_risk

        self.cum_reward += reward - pred_reward
        info["cum_reward"] = self.cum_reward
        info["pred_reward"] = pred_reward
        info["raw_rating"] = raw_rating
        info["survey_reward"] = survey_reward
        info["leave_risk"] = leave_risk
        info["survey_positive_score"] = pos_score
        info["survey_negative_score"] = neg_score
        info["lambda_leave"] = self.lambda_leave_penalty
        return state, reward, terminated, truncated, info
