"""Paired Greedy and Double DQN downstream comparison of frozen EM B/C."""
import argparse
import json
import os
import pickle
import platform
import subprocess
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from .core import load_artifacts, digest, evaluate, train


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def paired(a, b, seed):
    if [r['user_id'] for r in a['users']] != [r['user_id'] for r in b['users']]:
        raise ValueError('Unpaired users')
    rng = np.random.default_rng(seed)
    report = {}
    for key in ['total_reward', 'steps', 'reward_per_step']:
        d = np.array([y[key]-x[key] for x,y in zip(a['users'], b['users'])])
        bootstrap = [rng.choice(d,len(d),replace=True).mean() for _ in range(2000)]
        report[key] = {'second_minus_first': float(d.mean()),
                       'ci95': np.quantile(bootstrap,[.025,.975]).tolist()}
    return report


def main(args):
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    data = Path(args.yahoo_root)
    gt_path = data/'data_raw/RL4Rec_data/yahoo_pseudoGT_ratingM.ascii'
    dist_path = data/'data_processed/distance_mat.pickle'
    active_path = data/'data_raw/ydata-ymusic-rating-study-v1_0-train.txt'
    full_gt = np.loadtxt(gt_path)
    # Trusted local repository data only; never use an untrusted pickle.
    with dist_path.open('rb') as f:
        distance = np.asarray(pickle.load(f))
    active = np.loadtxt(active_path, dtype=int)
    counts = np.bincount(active[:,0])
    if distance.shape != (1000,1000) or not np.isfinite(distance).all():
        raise ValueError('Invalid distance matrix')
    config = {'args': vars(args), 'torch': torch.__version__, 'numpy': np.__version__,
              'python': platform.python_version(),
              'commit': subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),
              'source_sha256': {p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
              'environment_sha256': {str(p):digest(p) for p in [gt_path,dist_path,active_path]},
              'notes': ['Independent Double DQN implementation; not legacy checkpoint compatible.',
                        'Frozen shared B user and item embeddings; user vector plus mean last-window (item, feedback).',
                        'Common corrected distance window; pseudo-GT simulator, not online evidence.',
                        'All results use validation-selected EM and DQN checkpoints. No test-driven selection.']}
    save(out/'config.json',config)
    summaries, differences = [], []
    for seed in args.seeds:
        ids, groups, pools, matrices, embeddings, audit = load_artifacts(args.em_root,seed)
        gt = full_gt[ids-1]
        if gt.shape != matrices['B'].shape or not np.isfinite(gt).all() or gt.min()<1 or gt.max()>5:
            raise ValueError('Invalid common pseudo-GT ratings')
        if args.smoke_users:
            pools = {k:v[:args.smoke_users] for k,v in pools.items()}
        seedout = out/f'seed_{seed}';seedout.mkdir()
        save(seedout/'artifact_audit.json',audit)
        save(seedout/'user_pools.json',{k:ids[v].tolist() for k,v in pools.items()})
        reports = {}
        for name in ['B','C']:
            directory = seedout/name;directory.mkdir()
            # No test evaluation is used to choose checkpoints.
            policy = None if args.greedy_only else train(matrices[name],gt,distance,pools,ids,groups,
                                                         embeddings,args,seed,directory)
            for method in (['Greedy'] if args.greedy_only else ['Greedy','DQN']):
                for scenario in ['fixed','natural']:
                    report = evaluate(None if method=='Greedy' else policy,matrices[name],gt,distance,
                                      pools['test'],ids,groups,embeddings,args,scenario)
                    for r in report['users']:
                        r['historical_count'] = int(counts[r['user_id']])
                    report['subgroups'] = {}
                    for group in [0,1,2,3,4,'count_10_12','count_13_plus']:
                        rows = [r for r in report['users'] if (r['group']==group if isinstance(group,int)
                           else (r['historical_count']<=12 if group=='count_10_12' else r['historical_count']>12))]
                        if rows:
                            report['subgroups'][str(group)] = {'n':len(rows),**{k:float(np.mean([r[k] for r in rows]))
                                                       for k in ['total_reward','steps','reward_per_step']}}
                    save(directory/f'{method}_{scenario}_test.json',report)
                    summaries.append({'seed':seed,'model':name,'method':method,'scenario':scenario,**report['summary']})
                    reports[(name,method,scenario)] = report
        for scenario in ['fixed','natural']:
            pairs = [(('B','Greedy'),('C','Greedy'))]
            if not args.greedy_only:
                pairs += [(('B','DQN'),('C','DQN')),(('B','Greedy'),('B','DQN')),(('C','Greedy'),('C','DQN'))]
            for a,b in pairs:
                differences.append({'seed':seed,'scenario':scenario,'first':list(a),'second':list(b),
                    'metrics':paired(reports[(*a,scenario)],reports[(*b,scenario)],seed)})
        save(out/'results.json',summaries)
        save(out/'paired_bootstrap.json',{'note':'Per-seed paired users; excludes model training uncertainty.',
                                         'comparisons':differences})
    save(out/'summary.json',{'status':'completed','smoke_only':bool(args.smoke_users),
                            'greedy_only':args.greedy_only,'runs':summaries})
    print(f'Completed. Results: {out}',flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--em-root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--yahoo-root',default='data/YahooR3')
    p.add_argument('--device',default='auto')
    p.add_argument('--seeds',type=int,nargs='+',default=[2022,2023,2024])
    p.add_argument('--greedy-only',action='store_true')
    p.add_argument('--smoke-users',type=int,default=0,help='Cap users in each pool; debug only')
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--steps-per-epoch',type=int,default=10000)
    p.add_argument('--learning-starts',type=int,default=1000)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--capacity',type=int,default=20000)
    p.add_argument('--hidden',type=int,default=128)
    p.add_argument('--gamma',type=float,default=.9)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--epsilon',type=float,default=.1)
    p.add_argument('--target-update',type=int,default=320,help='Gradient updates between target copies')
    p.add_argument('--window',type=int,default=3)
    p.add_argument('--leave-threshold',type=float,default=120)
    p.add_argument('--horizon',type=int,default=10)
    p.add_argument('--max-turn',type=int,default=30)
    p.add_argument('--threads',type=int,default=4)
    return p


if __name__=='__main__':
    args=parser().parse_args()
    for name in ['epochs','steps_per_epoch','batch_size','capacity','hidden','target_update','window','horizon','max_turn','threads']:
        if getattr(args,name)<1: raise SystemExit(f'{name} must be positive')
    if not 0<=args.epsilon<=1 or not 0<=args.gamma<=1 or args.lr<=0:
        raise SystemExit('Invalid optimization settings')
    if args.horizon>1000 or args.max_turn>1000 or args.capacity<args.batch_size or args.smoke_users<0:
        raise SystemExit('Invalid horizon, capacity or user count')
    if len(set(args.seeds))!=len(args.seeds):raise SystemExit('Duplicate seeds')
    main(args)
