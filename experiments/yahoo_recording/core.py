import copy
import numpy as np
import torch
from torch import nn


def rating_counts(ratings, random_rows):
    active = np.stack([(ratings == v).sum(1) for v in range(5)], axis=1)
    random = np.zeros((len(ratings),5), dtype=np.int64)
    np.add.at(random,(random_rows[:,0],random_rows[:,2]),1)
    return active, random


class Marginal(nn.Module):
    """Shared nuisance P(star | Q); trained on random ratings only."""
    def __init__(self, prior):
        super().__init__()
        self.linear=nn.Linear(5,5)
        nn.init.zeros_(self.linear.weight)
        with torch.no_grad(): self.linear.bias.copy_(prior.log())

    def forward(self,q):return self.linear(q).softmax(-1)


class Recording(nn.Module):
    def __init__(self, baseline, conditional=False, hidden=8):
        super().__init__()
        logits=torch.logit(baseline)
        self.conditional=conditional
        if conditional:
            self.register_buffer('baseline',logits.clone())
            self.correction=nn.Sequential(nn.Linear(5,hidden),nn.Tanh(),nn.Linear(hidden,5))
            nn.init.zeros_(self.correction[-1].weight)
            nn.init.zeros_(self.correction[-1].bias)
        else:
            self.baseline=nn.Parameter(logits.clone())

    def delta(self,q):
        return self.correction(q) if self.conditional else torch.zeros_like(q)

    def forward(self,q):
        return torch.sigmoid(self.baseline+self.delta(q)).clamp(1e-6,1-1e-6)


def recording_distribution(p,mu):
    """Six outcomes: recorded 1..5 stars, or absent from historical table."""
    observed=p*mu
    return torch.cat([observed,(1-observed.sum(-1,keepdim=True)).clamp_min(1e-12)],-1)


def count_nll(distribution, counts, n_items):
    missing=n_items-counts.sum(-1,keepdim=True)
    if (missing<0).any():raise ValueError('Too many active ratings')
    all_counts=torch.cat([counts,missing],-1)
    # Drop the multinomial coefficient, which is constant across models.
    return -(all_counts*distribution.clamp_min(1e-12).log()).sum(-1)/n_items


def random_nll(p, counts):
    return -(counts*p.clamp_min(1e-12).log()).sum(-1)/counts.sum(-1)


def fit(model, train_loss, validation_loss, epochs=400, patience=30, lr=.01):
    """Select on validation only, including unmodified epoch-zero baseline."""
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    with torch.no_grad():best=float(validation_loss(model))
    best_state=copy.deepcopy(model.state_dict());best_epoch=0;stale=0;history=[]
    for epoch in range(1,epochs+1):
        model.train();optimizer.zero_grad()
        loss=train_loss(model)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training objective')
        loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():val=float(validation_loss(model))
        if not np.isfinite(val):raise RuntimeError('Nonfinite validation objective')
        history.append({'epoch':epoch,'train_objective_before_update':float(loss.detach()),'validation_nll':val})
        if val<best-1e-10:
            best,best_epoch,best_state=val,epoch,copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if stale>=patience:break
    model.load_state_dict(best_state)
    return {'best_epoch':best_epoch,'validation_nll':best,'epochs_run':len(history),'history':history}


@torch.no_grad()
def metrics(p,mu,counts,n_items):
    joint=recording_distribution(p,mu)
    predicted=n_items*joint[:,:5]
    conditional=joint[:,:5]/joint[:,:5].sum(-1,keepdim=True)
    return {'record_nll':count_nll(joint,counts,n_items).cpu().numpy(),
            'total_count_mae':(predicted.sum(-1)-counts.sum(-1)).abs().cpu().numpy(),
            'star_count_mae':(predicted-counts).abs().mean(-1).cpu().numpy(),
            'recorded_star_nll':random_nll(conditional,counts).cpu().numpy()}
