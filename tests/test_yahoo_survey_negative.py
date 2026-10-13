import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.util.yahoo_survey_negative import survey_negative_multipliers
from src.core.util.loss import loss_pointwise_negative


class SurveyNegativeTests(unittest.TestCase):
    def test_interpolation_direction_mean_and_unknown_users(self):
        survey = np.array([[3, 1, 2, 3, 4, 5, 2], [3, 5, 4, 3, 2, 1, 2]])
        pred = np.array([[1., 3., 5.], [1., 3., 5.]])
        users = np.array([0, 0, 0, 1, 1, 1, 2])
        items = np.array([0, 1, 2, 0, 1, 2, 0])
        weights = survey_negative_multipliers(users, items, pred, survey, 'survey')
        self.assertLess(weights[0], weights[1])
        self.assertLess(weights[1], weights[2])
        self.assertAlmostEqual(weights.mean(), 1.)
        self.assertEqual(weights[-1], 1.)
        pred[0, 1] = 2.5
        interpolated = survey_negative_multipliers(users[:3], items[:3], pred, survey, 'survey')
        expected = np.array([.25, .25+.75*1.5/4, 1.])
        np.testing.assert_allclose(interpolated, expected / expected.mean())

    def test_shuffle_deterministic_and_does_not_change_global_rng(self):
        survey = np.array([[3, 1, 1, 1, 1, 1, 2], [3, 5, 5, 5, 5, 5, 2]])
        pred = np.full((2, 1), 3.)
        np.random.seed(42)
        before = np.random.get_state()
        a = survey_negative_multipliers([0, 1], [0, 0], pred, survey, 'shuffled', 3)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        b = survey_negative_multipliers([0, 1], [0, 0], pred, survey, 'shuffled', 3)
        np.testing.assert_array_equal(a, b)
        self.assertAlmostEqual(a.mean(), 1.)

    def test_penalty_weights_both_variance_and_error_without_broadcasting(self):
        y = torch.tensor([[4.], [2.]])
        pos = torch.tensor([[3.], [2.]], requires_grad=True)
        neg = torch.tensor([[2.], [1.]], requires_grad=True)
        var = torch.tensor([[.3], [.2]])
        neg_var = torch.tensor([[.1], [.4]])
        multiplier = torch.tensor([[.5], [1.5]])
        score = torch.cat([torch.zeros_like(y), multiplier], dim=1)
        args = SimpleNamespace(lambda_ab=10, negative_weight=.1, survey_negative_mode='survey')
        actual = loss_pointwise_negative(y, pos, neg, score, args=args, log_var=var, log_var_neg=neg_var)
        expected = ((pos-y).square()*torch.exp(-var)+var).sum()
        expected += .1*((neg.square()*torch.exp(-neg_var)+neg_var)*multiplier).sum()
        torch.testing.assert_close(actual, expected)
        actual.backward()
        torch.testing.assert_close(neg.grad, .2*neg.detach()*torch.exp(-neg_var)*multiplier)
        with self.assertRaises(ValueError):
            loss_pointwise_negative(y, pos, neg, score[:, :1], args=args)


if __name__ == '__main__':
    unittest.main()
