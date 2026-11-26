import unittest

import numpy as np
import torch

from experiments.yahoo_sparse_em.profiles import ARMS, fit_priors, make_support


class SparseProfileTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(7)
        self.q = rng.integers(1, 6, size=(24, 7))
        self.ratings = rng.integers(0, 5, size=(24, 20))
        self.ratings[:, 6:] = (self.q[:, 1:2] - 1)
        self.ratings[:, 0] = -1
        self.pools = {"calibration": np.arange(16), "validation": np.arange(16, 20),
                      "test": np.arange(20, 24)}

    def fit(self, ratings):
        return fit_priors(ratings, self.q, self.pools, n_folds=3)

    def test_support_is_nested_excludes_ineligible_and_is_stable_by_user(self):
        items, stars = make_support(self.ratings, 5, 42)
        short, _ = make_support(self.ratings, 2, 42)
        np.testing.assert_array_equal(short, items[:, :2])
        self.assertTrue((items != 0).all())
        self.assertTrue(all(len(set(row)) == 5 for row in items))
        subset, _ = make_support(self.ratings[[9, 2]], 5, 42, user_ids=[9, 2])
        np.testing.assert_array_equal(subset, items[[9, 2]])
        np.testing.assert_array_equal(stars, np.take_along_axis(self.ratings, items, 1))

    def test_heldout_history_and_own_fold_labels_cannot_change_prior(self):
        base = self.fit(self.ratings)
        changed = self.ratings.copy()
        fold = base.fold_id == base.fold_id[0]
        changed[fold] = 0
        changed[16:] = -1
        altered = self.fit(changed)
        np.testing.assert_array_equal(base.fold_id[base.donor], base.fold_id)
        for name in ("validation", "test"):
            ids = self.pools[name]
            np.testing.assert_array_equal(np.sort(base.donor[ids]), ids)
        for arm in ARMS:
            np.testing.assert_allclose(base.priors[arm][fold], altered.priors[arm][fold])
        np.testing.assert_allclose(base.strength[fold], altered.strength[fold])
        only_eval_changed = self.ratings.copy()
        only_eval_changed[16:] = -1
        heldout = self.fit(only_eval_changed)
        for arm in ARMS:
            np.testing.assert_allclose(base.priors[arm], heldout.priors[arm])
        np.testing.assert_allclose(base.strength, heldout.strength)

    def test_small_model_uses_questionnaire_and_only_visible_star_counts(self):
        model = self.fit(self.ratings)
        empty = np.empty((24, 0), dtype=int)
        own = model.features("own", empty)
        self.assertGreater(np.abs(own[16:] - model.priors["global"][16:]).max(), .05)
        for arm in ARMS:
            np.testing.assert_allclose(model.features(arm, empty), model.priors[arm], atol=1e-7)
        one = model.features("own", np.full((24, 1), 4))
        expected = (model.strength[:, None] * own + np.eye(5)[4]) / (model.strength[:, None] + 1)
        np.testing.assert_allclose(one, expected, atol=1e-7)
        np.testing.assert_allclose(one.sum(axis=1), 1.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
