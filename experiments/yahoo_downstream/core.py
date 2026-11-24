import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_artifacts(root, seed):
    root = Path(root).resolve()
    users = np.load(root / 'original_user_ids.npy', allow_pickle=False)
    groups = np.load(root / 'groups.npy', allow_pickle=False)
    if users.ndim != 1 or len(set(users.tolist())) != len(users) or users.min() < 1:
        raise ValueError('Invalid original user IDs')
    lookup = {int(u): i for i, u in enumerate(users)}
    splits = {k: np.array([lookup[u] for u in v], dtype=int)
              for k, v in json.loads((root / 'splits.json').read_text()).items()}
    flat = np.concatenate(list(splits.values()))
    if sorted(flat.tolist()) != list(range(len(users))) or groups.shape != users.shape:
        raise ValueError('Splits must partition users exactly')
    selected = json.loads((root / f'seed_{seed}/C_selected.json').read_text())
    cp = (root / selected['checkpoint']).resolve()
    if not cp.is_relative_to(root):
        raise ValueError('Checkpoint path outside artifact root')
    bcp = root / f'seed_{seed}/B/best.pt'
    paths = {'B': bcp.parent / 'expected_ratings.npy', 'C': cp.parent / 'expected_ratings.npy'}
    matrices = {k: np.load(p, allow_pickle=False) for k, p in paths.items()}
    for a in matrices.values():
        if a.shape != (len(users), 1000) or not np.isfinite(a).all() or a.min() < 1-1e-5 or a.max() > 5+1e-5:
            raise ValueError('Invalid expected-rating matrix')
    checkpoint = torch.load(bcp, map_location='cpu', weights_only=True)
    item_embeddings = checkpoint['state_dict']['item.weight'].numpy().copy()
    user_embeddings = checkpoint['state_dict']['user.weight'].numpy().copy()
    if item_embeddings.shape[0] != 1000 or user_embeddings.shape[0] != len(users) or not np.isfinite(item_embeddings).all() or not np.isfinite(user_embeddings).all():
        raise ValueError('Invalid shared B item embeddings')
    embeddings = {'item': item_embeddings, 'user': user_embeddings}
    audit = {'shared_embedding_checkpoint': str(bcp), 'selected_C': selected,
             'sha256': {str(p.relative_to(root)): digest(p) for p in
                        [bcp, cp, *paths.values(), root/'original_user_ids.npy', root/'splits.json', root/'groups.npy']}}
    return users, groups, splits, matrices, embeddings, audit


def state_dim(embeddings):
    if isinstance(embeddings, dict):
        return embeddings['item'].shape[1] + embeddings['user'].shape[1] + 1
    return embeddings.shape[1] + 1


def state_vector(actions, rewards, embeddings, window=3, user=None):
    """Frozen shared B user vector + mean(last item vectors, normalized feedback).

    The user vector permits personalized first actions. B/C share both embedding
    tables. No model-specific embedding or learned state tracker is used.
    """
    items = embeddings['item'] if isinstance(embeddings, dict) else embeddings
    if not actions:
        history = np.zeros(items.shape[1] + 1, dtype=np.float32)
    else:
        history = np.r_[items[np.asarray(actions[-window:])].mean(0),
                        np.mean(rewards[-window:])].astype(np.float32)
    if isinstance(embeddings, dict):
        if user is None:
            raise ValueError('Missing user index for personalized state')
        return np.r_[embeddings['user'][user], history].astype(np.float32)
    return history


def leave(actions, action, distance, threshold, window):
    # Clamp window boundary, unlike legacy negative-start slicing.
    return bool(actions and np.any(distance[action, actions[-window:]] < threshold))


class QNetwork(nn.Module):
    def __init__(self, state_dim, n_items, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_items))

    def forward(self, x):
        return self.net(x)


def masked_argmax(values, mask):
    if not mask.any(-1).all():
        raise ValueError('No available actions')
    return values.masked_fill(~mask, -torch.inf).argmax(-1)


def double_target(online_next, target_next, mask, rewards, done, gamma):
    # Terminal rows may have no available items: give a dummy action, then zero
    # the bootstrap value. Avoid 0 * (-inf) producing NaN.
    safe = mask.clone()
    safe[done, 0] = True
    actions = masked_argmax(online_next, safe)
    values = target_next.gather(1, actions[:, None]).squeeze(1)
    return rewards + gamma * torch.where(done, torch.zeros_like(values), values)


class Replay:
    def __init__(self, capacity, dim, n_items):
        self.s = np.empty((capacity, dim), np.float32)
        self.ns = np.empty_like(self.s)
        self.a = np.empty(capacity, np.int64)
        self.r = np.empty(capacity, np.float32)
        self.done = np.empty(capacity, bool)
        self.mask = np.empty((capacity, n_items), bool)
        self.capacity, self.size, self.pos = capacity, 0, 0

    def add(self, s, a, r, ns, done, mask):
        j = self.pos
        self.s[j], self.a[j], self.r[j], self.ns[j], self.done[j], self.mask[j] = s, a, r, ns, done, mask
        self.pos = (j + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, rng, batch, device):
        ix = rng.choice(self.size, batch, replace=False)
        return [torch.as_tensor(x[ix], device=device) for x in
                (self.s, self.a, self.r, self.ns, self.done, self.mask)]


@torch.no_grad()
def evaluate(policy, predicted, gt, distance, users, original_ids, groups, embeddings, args, scenario):
    """Each user exactly once; all four methods receive the same pseudo-GT feedback."""
    if policy is not None:
        policy.eval()
    rows, all_actions, first = [], [], []
    limit = args.horizon if scenario == 'fixed' else args.max_turn
    for user in users:
        actions, feedback, raw_rewards = [], [], []
        for _ in range(limit):
            allowed = np.ones(predicted.shape[1], bool)
            allowed[actions] = False
            if policy is None:
                action = int(np.argmax(np.where(allowed, predicted[user], -np.inf)))
            else:
                s = state_vector(actions, feedback, embeddings, args.window, user=user)
                values = policy(torch.as_tensor(s[None], device=args.device))
                action = int(masked_argmax(values, torch.as_tensor(allowed[None], device=args.device)).item())
            terminal = leave(actions, action, distance, args.leave_threshold, args.window)
            raw = float(gt[user, action])
            actions.append(action)
            raw_rewards.append(raw)
            feedback.append((raw - 1) / 4)
            if scenario == 'natural' and terminal:
                break
        rows.append({'user_id': int(original_ids[user]), 'group': int(groups[user]),
                     'total_reward': sum(raw_rewards), 'steps': len(actions),
                     'reward_per_step': float(np.mean(raw_rewards)), 'actions': actions})
        all_actions.extend(actions)
        first.append(actions[0])
    counts = np.bincount(all_actions, minlength=predicted.shape[1])
    summary = {'users': len(rows), 'mean_reward': float(np.mean([r['total_reward'] for r in rows])),
               'mean_steps': float(np.mean([r['steps'] for r in rows])),
               'pooled_reward_per_step': float(sum(r['total_reward'] for r in rows)/sum(r['steps'] for r in rows)),
               'coverage': float(np.count_nonzero(counts)/len(counts)),
               'unique_items': int(np.count_nonzero(counts)),
               'most_recommended_share': float(counts.max()/counts.sum()),
               'first_choice_concentration': float(np.bincount(first).max()/len(first)), 'repeat_rate': 0.0}
    return {'summary': summary, 'users': rows}


def train(predicted, gt, distance, pools, ids, groups, embeddings, args, seed, out, auxiliary=None, alpha=0.0):
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError('Invalid auxiliary strength')
    if auxiliary is not None and (auxiliary.shape != predicted.shape or not np.isfinite(auxiliary).all()):
        raise ValueError('Invalid auxiliary reward matrix')
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    replay_rng = np.random.default_rng(seed + 1)
    policy = QNetwork(state_dim(embeddings), predicted.shape[1], args.hidden).to(args.device)
    initial = copy.deepcopy(policy.state_dict())
    target = copy.deepcopy(policy).eval()
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    replay = Replay(args.capacity, state_dim(embeddings), predicted.shape[1])
    best, best_state, best_epoch = -float('inf'), None, 0
    actions, feedback = [], []
    user = int(rng.choice(pools['calibration']))
    updates, total = 0, 0
    history = []
    for epoch in range(1, args.epochs+1):
        losses = []
        base_sum, auxiliary_sum, auxiliary_abs_sum, auxiliary_steps = 0.0, 0.0, 0.0, 0
        policy.train()
        for _ in range(args.steps_per_epoch):
            s = state_vector(actions, feedback, embeddings, args.window, user=user)
            mask = np.ones(predicted.shape[1], bool); mask[actions] = False
            if rng.random() < args.epsilon or total < args.learning_starts:
                action = int(rng.choice(np.flatnonzero(mask)))
            else:
                with torch.no_grad():
                    action = int(masked_argmax(policy(torch.as_tensor(s[None], device=args.device)),
                                  torch.as_tensor(mask[None], device=args.device)).item())
            done = leave(actions, action, distance, args.leave_threshold, args.window) or len(actions)+1 >= args.max_turn
            base_reward = float((predicted[user, action]-1)/4)
            reward = base_reward + (alpha * float(auxiliary[user, action]) if auxiliary is not None else 0.0)
            contribution = reward - base_reward
            base_sum += base_reward
            auxiliary_sum += contribution
            auxiliary_abs_sum += abs(contribution)
            auxiliary_steps += abs(contribution) > 1e-8
            # Reward shaping affects the TD target only. State feedback remains
            # the unchanged B prediction, matching the unshaped evaluation state.
            actions.append(action); feedback.append(base_reward)
            ns = state_vector(actions, feedback, embeddings, args.window, user=user)
            mask[action] = False
            replay.add(s, action, reward, ns, done, mask)
            total += 1
            if total >= args.learning_starts and replay.size >= args.batch_size:
                ss, aa, rr, nss, dd, mm = replay.sample(replay_rng, args.batch_size, args.device)
                with torch.no_grad():
                    y = double_target(policy(nss), target(nss), mm, rr, dd, args.gamma)
                loss = nn.functional.smooth_l1_loss(policy(ss).gather(1, aa[:, None]).squeeze(1), y)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite DQN loss')
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 10)
                opt.step(); updates += 1; losses.append(loss.item())
                if updates % args.target_update == 0:
                    target.load_state_dict(policy.state_dict())
            if done:
                actions, feedback = [], []
                user = int(rng.choice(pools['calibration']))
        val = evaluate(policy, predicted, gt, distance, pools['validation'], ids, groups, embeddings, args, 'fixed')
        score = val['summary']['mean_reward']
        record = {'epoch': epoch, 'environment_steps': total, 'gradient_updates': updates,
                  'mean_base_training_reward': base_sum / args.steps_per_epoch,
                  'mean_auxiliary_contribution': auxiliary_sum / args.steps_per_epoch,
                  'mean_abs_auxiliary_contribution': auxiliary_abs_sum / args.steps_per_epoch,
                  'auxiliary_nonzero_step_fraction': auxiliary_steps / args.steps_per_epoch,
                  'mean_loss': float(np.mean(losses)) if losses else None, 'validation': val['summary']}
        history.append(record)
        print(json.dumps({'run': str(out), **record}), flush=True)
        (out/'history.json').write_text(json.dumps(history, indent=2))
        if score > best:
            best, best_epoch, best_state = score, epoch, copy.deepcopy(policy.state_dict())
    if updates == 0:
        raise RuntimeError('No gradient updates; increase training budget')
    policy.load_state_dict(best_state)
    torch.save({'state_dict': {k:v.cpu() for k,v in best_state.items()}, 'epoch': best_epoch,
                'validation_reward': best, 'hidden': args.hidden, 'state_dim': state_dim(embeddings),
                'n_items': predicted.shape[1]}, out/'best_dqn.pt')
    (out/'selection.json').write_text(json.dumps({'epoch': best_epoch, 'validation_reward': best,
                   'initial_sha256': hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in initial.values())).hexdigest()}, indent=2))
    # Exercise the exact checkpoint loading path before final evaluation.
    restored = QNetwork(state_dim(embeddings), predicted.shape[1], args.hidden).to(args.device)
    restored.load_state_dict(torch.load(out/'best_dqn.pt', map_location=args.device, weights_only=True)['state_dict'])
    return restored
