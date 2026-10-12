"""Run in the project's easyrl4rec environment (PyTorch + bundled sources)."""
import sys
import unittest
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT), str(ROOT / 'src'), str(ROOT / 'src/DeepCTR-Torch'), str(ROOT / 'src/tianshou')])
from src.core.envs.Simulated_Env.calibrated_yahoo import UserPoolYahooEnv, CalibratedYahooEnv
from src.core.policy.RecPolicy import RecPolicy
from src.tianshou.tianshou.data import Batch
from types import SimpleNamespace


class IntegrationTests(unittest.TestCase):
    def test_reward_changes_but_evaluation_and_leave_rule_do_not(self):
        mat = np.full((4, 3), 4.0)
        distance = np.full((3, 3), 200.0)
        np.fill_diagonal(distance, 0)
        kwargs = dict(user_ids=[0, 2], mat=mat, mat_distance=distance,
                      num_leave_compute=3, leave_threshold=120, max_turn=30, random_init=False)
        evaluation = UserPoolYahooEnv(**kwargs)
        model = SimpleNamespace(eval=lambda: model)
        training = CalibratedYahooEnv(risk_mat=np.ones((4, 3)) * .75, calibration_lambda=1,
                                     ensemble_models=model, env_task_class=UserPoolYahooEnv,
                                     task_env_param=kwargs, task_name='YahooEnv-v0', predicted_mat=np.full((4, 3), 3.0))
        self.assertAlmostEqual(training.step(1)[1], -.25)
        self.assertEqual(evaluation.step(1)[1], 4.0)
        self.assertTrue(training.step(1)[2])
        self.assertTrue(evaluation.step(1)[2])
        for _ in range(20):
            self.assertIn(training.reset()[0][0], [0, 2])

    def test_exploration_respects_wrapper_mask(self):
        underlying = SimpleNamespace(action_type='discrete', exploration_noise=lambda act, batch: np.zeros_like(act))
        tracker = SimpleNamespace(num_item=3, emb_dim=8, device='cpu')
        wrapper = RecPolicy(SimpleNamespace(remove_recommended_ids=True, device='cpu'), underlying, tracker)
        batch = Batch(mask=torch.tensor([[False, True, False], [False, False, True]]))
        np.testing.assert_array_equal(wrapper.exploration_noise(np.array([1, 2]), batch), [1, 2])


if __name__ == '__main__':
    unittest.main()
