"""History-conditioned five-class DeepFM and a fixed recording-probability EM."""
from copy import deepcopy
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from experiments.yahoo_em.model import missing_posterior, observed_nll, soft_cross_entropy


class ContextDeepFM(nn.Module):
    """Use the visible (song, star) pairs and a five-star profile instead of a user ID."""

    def __init__(self, n_items, dim=16, hidden=64):
        super().__init__()
        self.item = nn.Embedding(n_items, dim)
        self.star = nn.Embedding(5, dim)
        self.profile = nn.Linear(5, dim, bias=False)
        self.item_linear = nn.Embedding(n_items, 5)
        self.context_linear = nn.Linear(2 * dim, 5)
        self.fm = nn.Linear(dim, 5, bias=False)
        self.deep = nn.Sequential(nn.Linear(3 * dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 5))
        nn.init.normal_(self.item.weight, std=0.05)
        nn.init.normal_(self.star.weight, std=0.05)
        nn.init.zeros_(self.item_linear.weight)

    def forward(self, history_items, history_stars, history_mask, profile, candidate_items):
        """Return logits [users, candidates, 5]. Input item/star IDs are zero-based."""
        safe_items = history_items.masked_fill(~history_mask, 0)
        safe_stars = history_stars.masked_fill(~history_mask, 0)
        # A star-dependent item representation retains which song received which star.
        song = self.item(safe_items)
        rating = self.star(safe_stars)
        events = song + rating + song * rating
        count = history_mask.sum(1, keepdim=True).clamp_min(1)
        history = (events * history_mask.unsqueeze(-1)).sum(1) / count
        profile_emb = self.profile(profile)
        item = self.item(candidate_items)
        h = history[:, None, :].expand_as(item)
        p = profile_emb[:, None, :].expand_as(item)
        linear = self.context_linear(torch.cat((history, profile_emb), -1))[:, None, :]
        interactions = self.fm(h * p + (h + p) * item)
        return (linear + self.item_linear(candidate_items) + interactions
                + self.deep(torch.cat((h, p, item), dim=-1)))


@dataclass
class SparseContext:
    items: torch.Tensor
    stars: torch.Tensor
    mask: torch.Tensor
    profile: torch.Tensor

    def logits(self, model, rows, candidates):
        return model(self.items[rows], self.stars[rows], self.mask[rows],
                     self.profile[rows], candidates)


def _batches(labels, size, generator=None):
    if size < 1:
        raise ValueError("batch_users must be positive")
    n = labels.shape[0]
    order = (torch.arange(n) if generator is None
             else torch.randperm(n, generator=generator))
    for rows in order.split(size):
        yield rows.to(labels.device)


def _all_logits(model, context, rows, n_items):
    items = torch.arange(n_items, device=rows.device).expand(len(rows), -1)
    return context.logits(model, rows, items)


def _regularization(model, coefficient):
    return coefficient * sum(p.square().sum() for p in model.parameters())


def warmup(model, context, labels, *, epochs, batch_users, lr, l2, seed):
    """Train using calibration targets with an observed star. -1 and -2 are ignored."""
    known = int((labels >= 0).sum().item())
    if not known:
        raise ValueError("No observed calibration targets")
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    model.train()
    for _ in range(epochs):
        total = 0.0
        for rows in _batches(labels, batch_users, generator):
            local_rows, items = torch.where(labels[rows] >= 0)
            if not len(items):
                continue
            users = rows[local_rows]
            logits = context.logits(model, users, items[:, None]).squeeze(1)
            ce = F.cross_entropy(logits, labels[users, items].long())
            loss = ce + _regularization(model, l2)
            if not loss.isfinite().item():
                raise FloatingPointError("Nonfinite warmup loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += ce.item() * len(items)
        history.append(total / known)
    return {"known_targets": known, "mean_training_ce": history}


@torch.no_grad()
def _e_step(model, context, labels, mu, batch_users):
    model.eval()
    targets = mu.new_empty((*labels.shape, 5))
    for rows in _batches(labels, batch_users):
        y = labels[rows]
        logits = _all_logits(model, context, rows, labels.shape[1])
        post = missing_posterior(logits, mu)
        known = y >= 0
        post[known] = F.one_hot(y[known].long(), 5).to(post.dtype)
        post[y == -2] = 0
        targets[rows] = post
    return targets


@torch.no_grad()
def _objectives(model, context, labels, mu, targets, batch_users, l2, n_targets):
    model.eval()
    total_q = total_obs = 0.0
    for rows in _batches(labels, batch_users):
        y = labels[rows]
        valid = y != -2
        logits = _all_logits(model, context, rows, labels.shape[1])[valid]
        total_q += soft_cross_entropy(logits, targets[rows][valid]).double().sum().item()
        total_obs += observed_nll(logits, y[valid].long(), mu.expand_as(logits)).double().sum().item()
    reg = _regularization(model, l2).item()
    return total_q / n_targets + reg, total_obs / n_targets + reg


def em_round(model, context, labels, mu, *, m_epochs, batch_users, lr, l2, seed):
    """One generalized EM update. -2 = excluded, -1 = absent, 0..4 = observed.

    labels contain only calibration users. The posterior remains frozen across
    all M-step passes and retries. Rejecting an update restores the incoming model.
    """
    mu = torch.as_tensor(mu, dtype=next(model.parameters()).dtype, device=labels.device)
    if mu.shape != (5,) or not ((mu > 0) & (mu < 1)).all().item():
        raise ValueError("Expected five recording probabilities strictly between 0 and 1")
    if not ((labels >= -2) & (labels <= 4)).all().item():
        raise ValueError("Invalid historical labels")
    n_targets = int((labels != -2).sum().item())
    if not n_targets or m_epochs < 1:
        raise ValueError("EM needs at least one eligible target and one M-step pass")
    targets = _e_step(model, context, labels, mu, batch_users)
    before = _objectives(model, context, labels, mu, targets, batch_users, l2, n_targets)
    if not all(map(math.isfinite, before)):
        raise FloatingPointError("Nonfinite incoming EM objective")
    initial = deepcopy(model.state_dict())
    accepted = False
    for attempt in range(5):
        model.load_state_dict(initial)
        step_lr = lr * (0.5 ** attempt)
        optimizer = torch.optim.Adam(model.parameters(), lr=step_lr)
        generator = torch.Generator().manual_seed(seed)
        model.train()
        for _ in range(m_epochs):
            for rows in _batches(labels, batch_users, generator):
                valid = labels[rows] != -2
                if not valid.any().item():
                    continue
                logits = _all_logits(model, context, rows, labels.shape[1])[valid]
                ce = soft_cross_entropy(logits, targets[rows][valid]).mean()
                loss = ce + _regularization(model, l2)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        after = _objectives(model, context, labels, mu, targets, batch_users, l2, n_targets)
        if all(math.isfinite(a) and a <= b + 1e-7 for a, b in zip(after, before)):
            accepted = True
            break
    if not accepted:
        model.load_state_dict(initial)
        after = before
    return {"accepted": accepted, "attempts": attempt + 1, "lr": step_lr,
            "q_before": before[0], "q_after": after[0],
            "observed_nll_before": before[1], "observed_nll_after": after[1]}
