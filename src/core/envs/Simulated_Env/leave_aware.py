import numpy as np

from src.core.envs.Simulated_Env.base import BaseSimulatedEnv


class LeaveAwareSimulatedEnv(BaseSimulatedEnv):
    """Reward shaping that exposes the environment's leave rule to the agent.

    The original simulated reward is the predicted user-item score. The task
    environment terminates an episode when the current item is too close to a
    recently recommended item, but that signal is only observed indirectly via
    shorter trajectories. This env adds an immediate penalty for actions that
    are likely to trigger the leave condition.
    """

    def __init__(
        self,
        ensemble_models,
        env_task_class,
        task_env_param: dict,
        task_name: str,
        predicted_mat=None,
        lambda_leave_penalty=1.0,
        lambda_distance_bonus=0.0,
    ):
        super().__init__(ensemble_models, env_task_class, task_env_param, task_name, predicted_mat)
        self.lambda_leave_penalty = lambda_leave_penalty
        self.lambda_distance_bonus = lambda_distance_bonus

    def _compute_pred_reward(self, action):
        base_reward = super()._compute_pred_reward(action)
        min_recent_distance = self._min_distance_to_recent_actions(action)
        if min_recent_distance is None:
            return base_reward

        threshold = float(self.env_task.leave_threshold)
        if threshold <= 0:
            return base_reward

        leave_risk = max(0.0, (threshold - min_recent_distance) / threshold)
        safe_distance = min(1.0, min_recent_distance / threshold)
        shaped_reward = (
            base_reward
            - self.lambda_leave_penalty * leave_risk
            + self.lambda_distance_bonus * safe_distance
        )
        return max(0.0, shaped_reward)

    def _min_distance_to_recent_actions(self, action):
        if not hasattr(self.env_task, "mat_distance"):
            return None

        t = int(self.total_turn)
        if t == 0:
            return None

        n_recent = int(self.env_task.num_leave_compute)
        start = max(0, t - n_recent)
        recent_actions = self.history_action[start:t]
        if len(recent_actions) == 0:
            return None

        distances = [self.env_task.mat_distance[int(action), int(prev)] for prev in recent_actions]
        return float(np.min(distances))
