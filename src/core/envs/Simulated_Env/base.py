import gymnasium as gym
import numpy as np
import torch

from torch import FloatTensor


# from virtualTB.model.UserModel import UserModel
# from src.core.envs.VirtualTaobao.virtualTB.utils import *

class BaseSimulatedEnv(gym.Env):
    def __init__(self, ensemble_models,
                 env_task_class, task_env_param: dict, task_name: str,
                 predicted_mat=None,
                 simulated_reward_mode="none",
                 lambda_leave_penalty=1.0,
                 survey_positive_alpha=1.0,
                 survey_negative_beta=1.0,
                 positive_rating_threshold=4.0,
                 negative_rating_threshold=2.0,
                 ):

        self.ensemble_models = ensemble_models.eval()
        self.env_task = env_task_class(**task_env_param)
        self.observation_space = self.env_task.observation_space
        self.action_space = self.env_task.action_space
        self.cum_reward = 0  # total_a in virtualtaobao
        self.total_turn = 0  # total_c in virtualtaobao
        self.env_name = task_name
        self.predicted_mat = predicted_mat
        self.simulated_reward_mode = simulated_reward_mode
        self.lambda_leave_penalty = lambda_leave_penalty
        self.survey_positive_alpha = survey_positive_alpha
        self.survey_negative_beta = survey_negative_beta
        self.positive_rating_threshold = positive_rating_threshold
        self.negative_rating_threshold = negative_rating_threshold

        self._reset_history()
        self.MIN_R = predicted_mat.min()
        self.MAX_R = predicted_mat.max()
        self.survey_reward_scores = self._load_survey_reward_scores()
        if self.simulated_reward_mode == "survey_reward_sensitivity":
            max_positive_score = float(self.survey_reward_scores[:, 0].max())
            self.MAX_R = self.MIN_R + (self.MAX_R - self.MIN_R) * (
                1.0 + self.survey_positive_alpha * max_positive_score
            )

        self.reset()

    @staticmethod
    def _to_action_index(action):
        return int(np.asarray(action).reshape(-1)[0])

    # def compile(self, num_env=1):
    #     self.env_list = DummyVectorEnv([lambda: gym.make(self.env_task) for _ in range(num_env)])

    def _construct_state(self, reward):
        res = self.env_task.state
        return res

    def seed(self, sd=0):
        torch.manual_seed(sd)

    def reset(self):
        self.cum_reward = 0
        self.total_turn = 0
        self.reward = 0
        self.action = None
        self.env_task.action = None
        self.state, self.info = self.env_task.reset()

        self._reset_history()
        if self.env_name == "VirtualTB-v0":
            self.cur_user = self.state[:-3]
        else:  # elif self.env_name == "KuaiEnv-v0":
            self.cur_user = self.state[0]
        # return self.state, {'key': 1, 'env': self}  ## TODO key
        return self.state, {'cum_reward': 0.0}

    def render(self, mode='human', close=False):
        self.env_task.render(mode)

    def _compute_pred_reward(self, action):
        if self.env_name == "VirtualTB-v0":
            feature = np.concatenate((self.cur_user, np.array([self.reward, 0, self.total_turn]), action), axis=-1)
            feature_tensor = torch.unsqueeze(torch.tensor(feature, device=self.user_model.device, dtype=torch.float), 0)
            # pred_reward = self.user_model(feature_tensor).detach().cpu().numpy().squeeze().round()
            pred_reward = self.user_model.forward(feature_tensor).detach().cpu().numpy().squeeze()
            if pred_reward < 0:
                pred_reward = 0
            if pred_reward > 10:
                pred_reward = 10
        else:  # elif self.env_name == "KuaiEnv-v0":
            # get prediction
            pred_reward = self.predicted_mat[int(self.cur_user), self._to_action_index(action)] - self.MIN_R

        return pred_reward

    def _compute_raw_pred_rating(self, action):
        if self.env_name == "VirtualTB-v0":
            return self._compute_pred_reward(action)
        return float(self.predicted_mat[int(self.cur_user), self._to_action_index(action)])

    def _load_survey_reward_scores(self):
        if self.simulated_reward_mode != "survey_reward_sensitivity":
            return None
        if self.env_name != "YahooEnv-v0":
            raise ValueError("survey_reward_sensitivity is currently implemented for YahooEnv-v0 only.")

        from src.core.envs.YahooR3.YahooData import YahooData

        traits = YahooData().load_survey_rating_scores()
        max_user = int(max(self.env_task.mat.shape[0], traits["user_id"].max() + 1))
        scores = np.zeros((max_user, 2), dtype=float)
        scores[:, 0] = float(traits["positive_reward_score"].mean())
        scores[:, 1] = float(traits["negative_reward_score"].mean())
        user_ids = traits["user_id"].to_numpy(dtype=int)
        valid = user_ids < max_user
        scores[user_ids[valid], 0] = traits.loc[valid, "positive_reward_score"].to_numpy(dtype=float)
        scores[user_ids[valid], 1] = traits.loc[valid, "negative_reward_score"].to_numpy(dtype=float)
        return scores

    def _compute_leave_risk(self, t, action):
        if t == 0 or not hasattr(self.env_task, "mat_distance"):
            return 0.0

        action = self._to_action_index(action)
        start = max(0, t - self.env_task.num_leave_compute)
        window_actions = self.history_action[start:t]
        if len(window_actions) == 0:
            return 0.0

        dist_list = np.array([self.env_task.mat_distance[action, int(x)] for x in window_actions], dtype=float)
        min_dist = float(dist_list.min())
        threshold = float(self.env_task.leave_threshold)
        if threshold <= 0:
            return 0.0
        return max(0.0, (threshold - min_dist) / threshold)

    def _apply_survey_reward_sensitivity(self, pred_reward, raw_pred_rating, t, action):
        user_id = int(self.cur_user)
        positive_score = float(self.survey_reward_scores[user_id, 0])
        negative_score = float(self.survey_reward_scores[user_id, 1])

        survey_reward = float(pred_reward)
        if raw_pred_rating >= self.positive_rating_threshold:
            survey_reward *= 1.0 + self.survey_positive_alpha * positive_score
        elif raw_pred_rating <= self.negative_rating_threshold:
            survey_reward -= self.survey_negative_beta * negative_score

        leave_risk = self._compute_leave_risk(t, action)
        return survey_reward - self.lambda_leave_penalty * leave_risk

    def _compute_training_reward(self, action, t):
        pred_reward = self._compute_pred_reward(action)
        if self.simulated_reward_mode in (None, "none"):
            return pred_reward
        if self.simulated_reward_mode == "survey_reward_sensitivity":
            raw_pred_rating = self._compute_raw_pred_rating(action)
            return self._apply_survey_reward_sensitivity(pred_reward, raw_pred_rating, t, action)
        raise ValueError(f"Unknown simulated_reward_mode: {self.simulated_reward_mode}")

    def step(self, action: FloatTensor):
        # 1. Collect ground-truth transition info
        self.action = action
        # real_state, real_reward, real_done, real_info = self.env_task.step(action)
        real_state, real_reward, real_terminated, real_truncated, real_info = self.env_task.step(action)

        t = int(self.total_turn)

        # 2. Predict click score, i.e, reward
        pred_reward = self._compute_training_reward(action, t)

        if t < self.env_task.max_turn:
            self._add_action_to_history(t, action)

        self.cum_reward += pred_reward
        self.total_turn = self.env_task.total_turn

        terminated = real_terminated
        # Rethink commented, do not use new user as new state
        # if terminated:
        #     self.state, self.info = self.env_task.reset()

        self.state = self._construct_state(pred_reward)
        
        # info =  {'CTR': self.cum_reward / self.total_turn / 10}
        info =  {'cum_reward': self.cum_reward}
        truncated = False

        return self.state, pred_reward, terminated, truncated, info

    def _reset_history(self):
        # self.history_action = {}
        if self.env_name == "VirtualTB-v0":
            self.history_action = np.zeros([self.env_task.max_turn, self.env_task.action_space.shape[0]])
        else:  # elif self.env_name == "KuaiEnv-v0":
            self.history_action = np.zeros(self.env_task.max_turn, dtype=int)
        self.max_history = 0

    def _add_action_to_history(self, t, action):
        if self.env_name == "VirtualTB-v0":
            action2 = np.expand_dims(action, 0)
            self.history_action[t] = action2
        else:  # elif self.env_name == "KuaiEnv-v0":
            self.history_action[t] = self._to_action_index(action)

        assert self.max_history == t
        self.max_history += 1
