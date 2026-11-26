import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from experiments.yahoo_em import run as em
from experiments.yahoo_recording.core import Recording
from experiments.yahoo_recording.joint import soft_counts, recording_loss, anchor_penalty, fit_joint


class JointEMTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(8)

    def test_soft_counts_and_record_loss_match_pairwise_formula(self):
        ratings = np.array([[4, -1, -1], [-1, 0, -1]])
        targets = np.full((6, 5), .2, dtype=np.float32)
        targets[0] = [0, 0, 0, 0, 1]
        targets[4] = [1, 0, 0, 0, 0]
        active, missing = soft_counts(ratings, targets)
        np.testing.assert_allclose(missing, .4)
        mu = torch.tensor([[.1, .2, .3, .4, .5], [.2, .3, .4, .5, .6]], requires_grad=True)
        actual = recording_loss(mu, torch.tensor(active), torch.tensor(missing), 6)
        expected = 0
        for u in range(2):
            for i in range(3):
                if ratings[u, i] >= 0:
                    expected -= mu[u, ratings[u, i]].log()
                else:
                    expected -= (torch.tensor(targets[u*3+i]) * torch.log1p(-mu[u])).sum()
        self.assertAlmostEqual(actual.item(), (expected/6).item(), places=6)
        ga = torch.autograd.grad(actual, mu, retain_graph=True)[0]
        ge = torch.autograd.grad(expected/6, mu)[0]
        torch.testing.assert_close(ga, ge)
        # Analytic unconstrained MLE has zero recording-loss gradient.
        a = torch.tensor([[2., 3., 4., 5., 6.]])
        b = torch.tensor([[8., 7., 6., 5., 4.]])
        mle = (a/(a+b)).requires_grad_()
        self.assertLess(torch.autograd.grad(recording_loss(mle, a, b, 50), mle)[0].abs().max(), 1e-6)

    def test_anchor_zero_at_initial_and_penalizes_drift(self):
        anchor = torch.tensor([[.01, .1, .3, .7, .9]])
        mu = anchor.clone().requires_grad_()
        loss = anchor_penalty(mu, anchor, 1.)
        self.assertAlmostEqual(loss.item(), 0.)
        self.assertLess(torch.autograd.grad(loss, mu)[0].abs().max(), 1e-6)
        self.assertGreater(anchor_penalty(anchor*.9, anchor, 1.).item(), 0)

    def fixture(self):
        ratings = np.array([[0, -1, -1, 4], [-1, 2, -1, 4], [1, -1, 3, -1]])
        rows = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 4]])
        args = SimpleNamespace(dim=4, hidden=8, device='cpu', batch_size=12,
                               lr=.001, l2=1e-6, warmup_epochs=1, em_rounds=2,
                               m_epochs=1, recording_lr=.001, recording_steps=5, anchor_strength=1.)
        features = np.array([[-1]*5, [0]*5, [1]*5], dtype=np.float32)
        return ratings, rows, args, features

    def test_joint_descent_and_paired_checkpoint(self):
        ratings, rows, args, features = self.fixture()
        for conditional in [False, True]:
            recording = Recording(torch.tensor([.1, .2, .3, .4, .5]), conditional, 4)
            before = copy.deepcopy(recording.state_dict())
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                model, _ = fit_joint(ratings, rows, np.zeros(3, int), features, recording,
                                     em.initialize(ratings, args, 7), args, 7, 'test', out)
                history = json.loads((out/'history.json').read_text())
                self.assertTrue(any(h['accepted'] for h in history))
                self.assertGreater(max(h['mu_max_abs_drift'] for h in history), 0)
                for h in history:
                    self.assertLessEqual(h['q_after'], h['q_before'] + 1e-7)
                    self.assertLessEqual(h['penalized_observed_after'], h['penalized_observed_before'] + 1e-7)
                ckpt = torch.load(out/'best.pt', weights_only=True)
                for k, v in model.state_dict().items():
                    torch.testing.assert_close(v, ckpt['state_dict'][k])
                for k, v in recording.state_dict().items():
                    torch.testing.assert_close(v, ckpt['recording_state_dict'][k])
                with torch.no_grad():
                    np.testing.assert_array_equal(recording(torch.tensor(features)).numpy(), np.load(out/'recording_mu.npy'))
                if conditional:
                    torch.testing.assert_close(recording.baseline, before['baseline'])

    def test_rejected_update_restores_both_networks(self):
        ratings, rows, args, features = self.fixture()
        recording = Recording(torch.full((5,), .2), True, 4)
        old = copy.deepcopy(recording.state_dict())
        initial = em.initialize(ratings, args, 7)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with patch('experiments.yahoo_recording.joint.acceptable', return_value=False):
                with self.assertRaisesRegex(RuntimeError, 'no accepted'):
                    fit_joint(ratings, rows, np.zeros(3, int), features, recording,
                              initial, args, 7, 'reject', out)
            h = json.loads((out/'history.json').read_text())[0]
            self.assertEqual(h['attempts'], 5)
            self.assertEqual(h['mu_max_abs_drift'], 0.)
            saved = torch.load(out/'best.pt', weights_only=True)
            self.assertEqual(saved['best_round'], 0)
            for k, v in initial.items():
                torch.testing.assert_close(saved['state_dict'][k], v)
            for k, v in old.items():
                torch.testing.assert_close(recording.state_dict()[k], v)
                torch.testing.assert_close(saved['recording_state_dict'][k], v)


if __name__ == '__main__':
    unittest.main()
