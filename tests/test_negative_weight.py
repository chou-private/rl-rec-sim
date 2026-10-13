import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core.util.loss import loss_pointwise, loss_pointwise_negative


class NegativeWeightTests(unittest.TestCase):
    def test_weight_zero_equals_observed_loss_and_one_preserves_original(self):
        y = torch.tensor([[4.], [2.]])
        pos = torch.tensor([[3.], [2.]])
        neg = torch.tensor([[2.], [1.]])
        score = torch.zeros_like(y)
        var = torch.tensor([[.3], [.2]])
        neg_var = torch.tensor([[.1], [.4]])
        base = SimpleNamespace(lambda_ab=10)
        observed = loss_pointwise(y, pos, neg, score, args=base, log_var=var, log_var_neg=neg_var)
        negative = (neg.square() * torch.exp(-neg_var)).sum() + neg_var.sum()
        for weight in [0., .1, 1.]:
            args = SimpleNamespace(lambda_ab=10, negative_weight=weight)
            actual = loss_pointwise_negative(y, pos, neg, score, args=args, log_var=var, log_var_neg=neg_var)
            torch.testing.assert_close(actual, observed + weight * negative)
        legacy = loss_pointwise_negative(y, pos, neg, score, args=base, log_var=var, log_var_neg=neg_var)
        torch.testing.assert_close(legacy, observed + negative)

    def test_without_variance_and_invalid_weight(self):
        args = SimpleNamespace(lambda_ab=10, negative_weight=.1)
        y, pos, neg, score = [torch.tensor([[v]]) for v in [4., 3., 2., 0.]]
        torch.testing.assert_close(loss_pointwise_negative(y, pos, neg, score, args=args), torch.tensor(1.4))
        args.negative_weight = -1
        with self.assertRaises(ValueError):
            loss_pointwise_negative(y, pos, neg, score, args=args)


if __name__ == '__main__':
    unittest.main()
