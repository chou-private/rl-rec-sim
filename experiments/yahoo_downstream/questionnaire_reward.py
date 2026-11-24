"""Three-way B-based reward experiment: original, own, shuffled questionnaire."""
import json
import platform
import pickle
import subprocess
from pathlib import Path

import numpy as np
import torch

from .run import parser, save, paired
from .core import load_artifacts, digest, train, evaluate
from experiments.yahoo_em.model import CategoricalDeepFM


def centered_scores(answers):
    if answers.ndim != 2 or answers.shape[1] != 5 or not np.isin(answers, [1,2,3,4,5]).all():
        raise ValueError('Expected five Q2-Q6 ordinal answers')
    return ((answers - answers.mean(axis=1, keepdims=True))/4).astype(np.float32)


def permutation_within_pools(n, pools, seed):
    """Shuffle full answer vectors, independently within each disjoint split."""
    mapping = np.arange(n)
    rng = np.random.default_rng(seed)
    for users in pools.values():
        mapping[users] = rng.permutation(users)
    if not np.array_equal(np.sort(mapping), np.arange(n)):
        raise ValueError('Invalid permutation')
    return mapping


def auxiliary_from_probabilities(probabilities, scores, expected, threshold):
    return (probabilities * scores).sum(-1) * (expected >= threshold)


@torch.no_grad()
def build_auxiliary(root, seed, predicted, own_scores, shuffled_scores, args):
    checkpoint = torch.load(Path(root)/f'seed_{seed}/B/best.pt', map_location='cpu', weights_only=True)
    model = CategoricalDeepFM(checkpoint['n_users'], checkpoint['n_items'], checkpoint['dim'], checkpoint['hidden'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to(args.device).eval()
    own, shuffled = np.empty_like(predicted), np.empty_like(predicted)
    nitems = predicted.shape[1]
    max_difference = 0.0
    for start in range(0,predicted.size,4096):
        ix = np.arange(start,min(start+4096,predicted.size))
        u,i = ix//nitems,ix%nitems
        p = model(torch.as_tensor(u,device=args.device),torch.as_tensor(i,device=args.device)).softmax(-1).cpu().numpy()
        expected = p @ np.arange(1,6,dtype=np.float32)
        max_difference = max(max_difference,float(np.max(abs(expected-predicted[u,i]))))
        # Gate uses the same saved expectation across all arms, not survey values.
        own.flat[ix] = auxiliary_from_probabilities(p,own_scores[u],predicted[u,i],args.rating_threshold)
        shuffled.flat[ix] = auxiliary_from_probabilities(p,shuffled_scores[u],predicted[u,i],args.rating_threshold)
    if max_difference > 1e-4:
        raise ValueError('B checkpoint and saved expected-rating matrix disagree')
    return own, shuffled, max_difference


def main(args):
    if args.greedy_only:
        raise ValueError('Reward experiment requires DQN; no greedy-only mode')
    if args.device=='auto':args.device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    data=Path(args.yahoo_root)
    qpath=data/'data_raw/ydata-ymusic-rating-study-v1_0-survey-answers.txt'
    gtpath=data/'data_raw/RL4Rec_data/yahoo_pseudoGT_ratingM.ascii'
    distpath=data/'data_processed/distance_mat.pickle'
    trainpath=data/'data_raw/ydata-ymusic-rating-study-v1_0-train.txt'
    answers=np.loadtxt(qpath,dtype=int)
    if answers.shape!=(5400,7):raise ValueError('Expected original Yahoo questionnaire')
    gt_all=np.loadtxt(gtpath)
    with distpath.open('rb') as f:distance=np.asarray(pickle.load(f))
    if distance.shape!=(1000,1000) or not np.isfinite(distance).all():raise ValueError('Invalid distances')
    active=np.loadtxt(trainpath,dtype=int);counts=np.bincount(active[:,0])
    save(out/'config.json',{'args':vars(args),'torch':torch.__version__,'numpy':np.__version__,
       'python':platform.python_version(),
       'commit':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),
       'source_sha256':{str(p):digest(p) for p in [*Path(__file__).parent.glob('*.py'),Path('experiments/yahoo_em/model.py')]},
       'data_sha256':{str(p):digest(p) for p in [qpath,gtpath,distpath,trainpath]},
       'design':'Fixed alpha and rating threshold; no test tuning. Shaping enters TD reward only; state feedback unchanged.'})
    results,contrasts=[],[]
    for seed in args.seeds:
        ids,groups,full_pools,matrices,embeddings,audit=load_artifacts(args.em_root,seed)
        predicted=matrices['B'];gt=gt_all[ids-1]
        if gt.shape!=predicted.shape or not np.isfinite(gt).all() or gt.min()<1 or gt.max()>5:
            raise ValueError('Invalid pseudo-GT')
        own_scores=centered_scores(answers[ids-1,1:6])
        mapping=permutation_within_pools(len(ids),full_pools,args.shuffle_seed)
        own,shuffled,error=build_auxiliary(args.em_root,seed,predicted,own_scores,own_scores[mapping],args)
        pools={k:v[:args.smoke_users] if args.smoke_users else v for k,v in full_pools.items()}
        seedout=out/f'seed_{seed}';seedout.mkdir()
        save(seedout/'artifact_audit.json',audit)
        save(seedout/'user_pools.json',{k:ids[v].tolist() for k,v in pools.items()})
        save(seedout/'questionnaire_mapping.json',{'user_ids':ids.tolist(),'donor_user_ids':ids[mapping].tolist(),
                                                 'shuffle_seed':args.shuffle_seed})
        train_users=pools['calibration']
        stats={}
        for name,a in [('own',own),('shuffled',shuffled)]:
            x=a[train_users]
            stats[name]={'min':float(x.min()),'max':float(x.max()),'mean':float(x.mean()),
                         'nonzero_fraction':float(np.mean(abs(x)>1e-8)),
                         'scaled_std':float((args.alpha*x).std())}
        print(json.dumps({'seed':seed,'reward_audit':stats,
                          'train_eligible_fraction':float((predicted[train_users]>=args.rating_threshold).mean())}),flush=True)
        if args.alpha and any(v['nonzero_fraction']==0 for v in stats.values()):
            print('WARNING: an auxiliary arm is identically zero on training users; inspect reward_audit.json.',flush=True)
        save(seedout/'reward_audit.json',{'checkpoint_expected_max_error':error,'alpha':args.alpha,
            'threshold':args.rating_threshold,'train_eligible_fraction':float((predicted[train_users]>=args.rating_threshold).mean()),
            'train_flat_questionnaire_fraction':float((np.ptp(own_scores[train_users],axis=1)==0).mean()),
            'changed_donor_fraction':float((mapping!=np.arange(len(ids))).mean()),'auxiliary':stats})
        reports={}
        for arm,aux in [('original',None),('own',own),('shuffled',shuffled)]:
            directory=seedout/arm;directory.mkdir()
            policy=train(predicted,gt,distance,pools,ids,groups,embeddings,args,seed,directory,auxiliary=aux,alpha=args.alpha)
            for scenario in ['fixed','natural']:
                report=evaluate(policy,predicted,gt,distance,pools['test'],ids,groups,embeddings,args,scenario)
                for row in report['users']:row['historical_count']=int(counts[row['user_id']])
                report['subgroups']={}
                for group in [0,1,2,3,4,'count_10_12','count_13_plus']:
                    rows=[r for r in report['users'] if (r['group']==group if isinstance(group,int) else
                            (r['historical_count']<=12 if group=='count_10_12' else r['historical_count']>12))]
                    if rows:report['subgroups'][str(group)]={'n':len(rows),**{k:float(np.mean([r[k] for r in rows]))
                                                             for k in ['total_reward','steps','reward_per_step']}}
                save(directory/f'{scenario}_test.json',report);reports[arm,scenario]=report
                results.append({'seed':seed,'arm':arm,'scenario':scenario,**report['summary']})
        for scenario in ['fixed','natural']:
            for a,b in [('original','own'),('original','shuffled'),('shuffled','own')]:
                contrasts.append({'seed':seed,'scenario':scenario,'first':a,'second':b,
                                  'metrics':paired(reports[a,scenario],reports[b,scenario],seed)})
        save(out/'results.json',results)
        save(out/'paired_bootstrap.json',{'note':'Paired user intervals exclude training and shuffle uncertainty.',
                                         'comparisons':contrasts})
    save(out/'summary.json',{'status':'completed','smoke_only':bool(args.smoke_users),'runs':results})
    print(f'Completed. Results: {out}',flush=True)


if __name__=='__main__':
    p=parser()
    p.description=__doc__
    p.add_argument('--alpha',type=float,default=.1)
    p.add_argument('--rating-threshold',type=float,default=3.0)
    p.add_argument('--shuffle-seed',type=int,default=20261124)
    args=p.parse_args()
    if not np.isfinite(args.alpha) or not 0<=args.alpha<=1 or not 1<=args.rating_threshold<=5:
        raise SystemExit('Require finite alpha in [0,1] and threshold in [1,5]')
    for k in ['epochs','steps_per_epoch','batch_size','capacity','hidden','target_update','window','horizon','max_turn','threads']:
        if getattr(args,k)<1:raise SystemExit(f'{k} must be positive')
    if args.capacity<args.batch_size or args.smoke_users<0 or args.max_turn>1000 or args.horizon>1000:
        raise SystemExit('Invalid capacity, smoke users or horizon')
    if not 0<=args.epsilon<=1 or not 0<=args.gamma<=1 or args.lr<=0 or args.learning_starts<0:
        raise SystemExit('Invalid optimization settings')
    if len(set(args.seeds))!=len(args.seeds):raise SystemExit('Duplicate seeds')
    main(args)
