"""Run: python -m unittest discover -s tests -p test_yahoo_em.py -v"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.yahoo_em.data import (questionnaire_groups, split_users,
                                       estimate_propensities, propensity_table)
from experiments.yahoo_em.model import CategoricalDeepFM, missing_posterior, observed_nll
from experiments.yahoo_em.run import e_step, objective, fit


class EMTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_hand_calculated_posterior_and_likelihood(self):
        logits = torch.tensor([[0.6, 0.4]]).log()
        mu = torch.tensor([[0.5, 0.1]])
        q = missing_posterior(logits, mu)
        torch.testing.assert_close(q, torch.tensor([[0.3 / 0.66, 0.36 / 0.66]]))
        torch.testing.assert_close(observed_nll(logits, torch.tensor([-1]), mu),
                                   -torch.tensor([0.66]).log())
        torch.testing.assert_close(observed_nll(logits, torch.tensor([0]), mu),
                                   -torch.tensor([0.3]).log())

    def test_equal_missingness_leaves_probabilities_unchanged(self):
        logits = torch.randn(4, 5)
        torch.testing.assert_close(missing_posterior(logits, torch.full((4, 5), 0.2)), logits.softmax(-1))

    def test_groups_and_disjoint_splits(self):
        q = np.array([[5, 3, 3, 3, 3, 3, 1], [5, 5, 5, 1, 5, 5, 1],
                      [5, 1, 1, 3, 5, 5, 1], [5, 5, 5, 3, 1, 1, 1],
                      [5, 1, 3, 4, 2, 2, 1]])
        np.testing.assert_array_equal(questionnaire_groups(q), np.arange(5))
        groups = np.repeat(np.arange(5), 10)
        splits = split_users(groups, 9)
        joined = np.concatenate(list(splits.values()))
        np.testing.assert_array_equal(np.sort(joined), np.arange(50))

    def test_calibration_does_not_read_heldout_ratings(self):
        ratings = np.tile(np.array([0, 1, 2, 3, 4, -1, -1, -1, -1, -1]), (10, 1))
        groups = np.tile(np.arange(5), 2)
        rows = np.array([[u, 0, u % 5] for u in range(10)])
        cal = np.arange(5)
        g, gm, audit = estimate_propensities(ratings, rows, groups, cal)
        changed = rows.copy()
        changed[5:, 2] = 4
        g2, gm2, _ = estimate_propensities(ratings, changed, groups, cal)
        np.testing.assert_array_equal(g, g2)
        np.testing.assert_array_equal(gm, gm2)
        np.testing.assert_array_equal(propensity_table(g, gm, 0), np.broadcast_to(g, gm.shape))
        self.assertTrue(np.isfinite(gm).all())
        self.assertIn('clipped_stars', audit['global'])

    def test_fm_branch_affects_class_probabilities(self):
        model = CategoricalDeepFM(2, 3, dim=4, hidden=8)
        u, i = torch.tensor([0, 1]), torch.tensor([0, 2])
        loss = -model(u, i).log_softmax(-1)[:, 0].mean()
        loss.backward()
        self.assertGreater(model.fm.weight.grad.abs().sum().item(), 0)

    def test_e_step_frozen_one_hot_and_finite_extremes(self):
        ratings = np.array([[0, -1, 4], [-1, 2, -1]])
        model = CategoricalDeepFM(2, 3, dim=4, hidden=8)
        table = np.tile([1e-5, .1, .2, .5, 1 - 1e-5], (5, 1))
        q = e_step(model, ratings, np.array([0, 1]), table, 2, 'cpu')
        np.testing.assert_array_equal(q[0], [1, 0, 0, 0, 0])
        np.testing.assert_array_equal(q[2], [0, 0, 0, 0, 1])
        saved = q.copy()
        with torch.no_grad():
            model.bias.add_(torch.randn(5))
        np.testing.assert_array_equal(q, saved)
        self.assertTrue(np.isfinite(q).all())

    def test_small_em_and_lambda_zero_equivalence(self):
        ratings = np.array([[0, -1, 4], [-1, 2, -1]])
        groups = np.array([0, 1])
        rows = np.array([[0, 1, 1], [1, 0, 3]])
        table = np.tile([.05, .08, .1, .2, .4], (5, 1))
        args = SimpleNamespace(dim=4, hidden=8, batch_size=6, device='cpu', l2=1e-6,
                               lr=.01, em_rounds=2, m_epochs=1)
        initial = CategoricalDeepFM(2, 3, 4, 8).state_dict()
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'B', Path(tmp) / 'C'
            a.mkdir(); b.mkdir()
            mb, vb = fit(ratings, rows, groups, table, initial, args, 7, 'B', a)
            mc, vc = fit(ratings, rows, groups, propensity_table(table[0], table, 0),
                         initial, args, 7, 'C0', b)
            self.assertEqual(vb, vc)
            for name, p in mb.state_dict().items():
                torch.testing.assert_close(p, mc.state_dict()[name], rtol=0, atol=0)
            old = CategoricalDeepFM(2, 3, 4, 8)
            old.load_state_dict(initial)
            self.assertLess(objective(mb, ratings, groups, table, 6, 'cpu', 1e-6),
                            objective(old, ratings, groups, table, 6, 'cpu', 1e-6))


if __name__ == '__main__':
    unittest.main()
