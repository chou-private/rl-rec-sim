import unittest

import numpy as np
import torch

from experiments.yahoo_sparse_em.evaluation import evaluate_rows, paired_comparisons
from experiments.yahoo_sparse_em.model import ContextDeepFM, SparseContext, _e_step, em_round
from experiments.yahoo_sparse_em.run import build_context, calibration_partition


class SparseEMTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_calibration_support_and_supervision_are_disjoint(self):
        ratings = np.array([list(range(5)) * 2 + [-1] * 12,
                            list(range(5)) * 2 + [4] + [-1] * 11])
        support, targets = calibration_partition(ratings, [18, 44], 99)
        np.testing.assert_array_equal((targets != -2).sum(1), [11, 11])
        self.assertTrue(np.all(targets[support >= 0] == -2))
        np.testing.assert_array_equal(targets[targets != -2], ratings[targets != -2])
        np.testing.assert_array_equal((support >= 0) | (targets >= 0), ratings >= 0)
        self.assertTrue(np.all((support >= 0).sum(1) >= 5))
        self.assertTrue(np.all((targets >= 0).sum(1) >= 5))

    def test_empty_history_and_fixed_b_em(self):
        model = ContextDeepFM(4, dim=4, hidden=8)
        ids = np.empty((2, 0), dtype=int)
        ctx = build_context(ids, ids, np.full((2, 5), .2), "cpu")
        padded = SparseContext(torch.full((2, 2), -1), torch.full((2, 2), -1),
                               torch.zeros((2, 2), dtype=torch.bool), ctx.profile)
        rows = torch.arange(2)
        songs = torch.arange(4).expand(2, 4)
        logits = ctx.logits(model, rows, songs)
        torch.testing.assert_close(padded.logits(model, rows, songs), logits)
        labels = torch.tensor([[-2, -1, 2, 4], [-1, 0, -1, -2]], dtype=torch.int8)
        mu = torch.tensor([.02, .03, .1, .2, .5])
        targets = _e_step(model, ctx, labels, mu, 1)
        missing = logits[0, 1].softmax(0) * (1 - mu)
        torch.testing.assert_close(targets[0, 1], missing / missing.sum())
        torch.testing.assert_close(targets[0, 2], torch.eye(5)[2])
        self.assertEqual(torch.count_nonzero(targets[labels == -2]).item(), 0)
        result = em_round(model, ctx, labels, mu, m_epochs=1, batch_users=2, lr=.001, l2=0, seed=7)
        self.assertTrue(result["accepted"])
        self.assertLessEqual(result["q_after"], result["q_before"] + 1e-7)
        self.assertLessEqual(result["observed_nll_after"], result["observed_nll_before"] + 1e-7)

    def test_comparison_requires_identical_songs_and_ratings(self):
        rows = np.array([[0, 2, 4], [0, 1, 0], [1, 2, 0], [1, 0, 0]])
        p = np.eye(5)[rows[:, 2]]
        metrics, users = evaluate_rows(rows, p)
        self.assertEqual((metrics["mae"], metrics["ndcg5"], metrics["ranking_users"]), (0, 1, 1))
        arms = dict(global_=users, own=users, shuffled=users)
        arms["global"] = arms.pop("global_")
        diffs = paired_comparisons(arms, seed=7, k=1, reps=4)
        self.assertTrue(all(d["metrics"]["mae"]["difference"] == 0 for d in diffs))
        rows[0, 1] = 3
        _, arms["own"] = evaluate_rows(rows, p)
        with self.assertRaisesRegex(ValueError, "different evaluation"):
            paired_comparisons(arms, seed=7, k=1, reps=4)


if __name__ == "__main__":
    unittest.main()
