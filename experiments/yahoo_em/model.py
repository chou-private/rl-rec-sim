"""Categorical DeepFM and fixed-propensity EM primitives (no legacy losses)."""
import torch
from torch import nn
from torch.nn import functional as F


class CategoricalDeepFM(nn.Module):
    """Shared embeddings; class-specific linear, FM and deep contributions.

    With two fields, the FM interaction vector is user_embedding * item_embedding.
    A learned projection gives each rating its own FM logit, avoiding cancellation
    of a shared scalar under softmax. This is a categorical DeepFM adaptation,
    not a byte-for-byte reproduction of the legacy scalar regressor.
    """

    def __init__(self, n_users, n_items, dim=16, hidden=64):
        super().__init__()
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        self.user_linear = nn.Embedding(n_users, 5)
        self.item_linear = nn.Embedding(n_items, 5)
        self.fm = nn.Linear(dim, 5, bias=False)
        self.deep = nn.Sequential(nn.Linear(2 * dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 5))
        self.bias = nn.Parameter(torch.zeros(5))
        nn.init.normal_(self.user.weight, std=0.05)
        nn.init.normal_(self.item.weight, std=0.05)
        nn.init.zeros_(self.user_linear.weight)
        nn.init.zeros_(self.item_linear.weight)

    def forward(self, users, items):
        u, i = self.user(users), self.item(items)
        return (self.user_linear(users) + self.item_linear(items) + self.bias
                + self.fm(u * i) + self.deep(torch.cat((u, i), dim=-1)))


def missing_posterior(logits, mu):
    """P(X=v | R=0,u,i), evaluated with OLD parameters during E-step."""
    return F.softmax(F.log_softmax(logits, dim=-1) + torch.log1p(-mu), dim=-1)


def observed_nll(logits, ratings, mu):
    """Per-pair -log P(R, observed X); ratings=-1 denotes missing."""
    logp = F.log_softmax(logits, dim=-1)
    missing = -torch.logsumexp(logp + torch.log1p(-mu), dim=-1)
    known = ratings >= 0
    result = missing.clone()
    if known.any():
        ix = ratings[known]
        result[known] = -(logp[known].gather(1, ix[:, None]).squeeze(1)
                          + mu[known].log().gather(1, ix[:, None]).squeeze(1))
    return result


def soft_cross_entropy(logits, targets):
    return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1)
