"""Q2-Q6 as fixed DQN state inputs: zero / own / shuffled, unmodified B reward."""
import json
import pickle
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

from .run import parser, save, paired
from .core import load_artifacts, digest, train, evaluate, state_dim
from .questionnaire_reward import permutation_within_pools


def questionnaire_features(answers):
    if answers.ndim != 2 or answers.shape[1] != 5 or not np.isin(answers,[1,2,3,4,5]).all():
        raise ValueError('Expected Q2-Q6 in original 1-5 coding')
    # Preserve absolute willingness: do NOT center within users.
    return ((answers.astype(np.float32)-3)/2).astype(np.float32)


def with_questionnaire(embeddings, features):
    if features.shape != (len(embeddings['user']),5) or not np.isfinite(features).all() or np.max(abs(features))>1:
        raise ValueError('Expected one bounded five-dimensional feature vector per mapped user')
    return {**embeddings, 'questionnaire': features.copy()}


def main(args):
    if args.greedy_only:raise ValueError('State experiment requires DQN')
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
    if answers.shape!=(5400,7):raise ValueError('Expected original questionnaire')
    gt_all=np.loadtxt(gtpath)
    with distpath.open('rb') as f:distance=np.asarray(pickle.load(f))
    if distance.shape!=(1000,1000) or not np.isfinite(distance).all():raise ValueError('Invalid distance matrix')
    counts=np.bincount(np.loadtxt(trainpath,dtype=int)[:,0])
    save(out/'config.json',{'args':vars(args),'torch':torch.__version__,'numpy':np.__version__,
      'python':platform.python_version(),
      'commit':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),
      'source_sha256':{str(p):digest(p) for p in Path(__file__).parent.glob('*.py')},
      'data_sha256':{str(p):digest(p) for p in [qpath,gtpath,distpath,trainpath]},
      'design':'Append (Q2-Q6 - 3)/2 to state; all arms same input size; original B reward only; no auxiliary reward.'})
    results,contrasts=[],[]
    for seed in args.seeds:
        ids,groups,full_pools,matrices,shared,audit=load_artifacts(args.em_root,seed)
        predicted=matrices['B'];gt=gt_all[ids-1]
        if gt.shape!=predicted.shape or not np.isfinite(gt).all() or gt.min()<1 or gt.max()>5:
            raise ValueError('Invalid pseudo-GT')
        own=questionnaire_features(answers[ids-1,1:6])
        mapping=permutation_within_pools(len(ids),full_pools,args.shuffle_seed)
        pools={k:v[:args.smoke_users] if args.smoke_users else v for k,v in full_pools.items()}
        seedout=out/f'seed_{seed}';seedout.mkdir()
        save(seedout/'artifact_audit.json',audit)
        save(seedout/'user_pools.json',{k:ids[v].tolist() for k,v in pools.items()})
        save(seedout/'questionnaire_mapping.json',{'user_ids':ids.tolist(),'donor_user_ids':ids[mapping].tolist(),
                                                 'shuffle_seed':args.shuffle_seed})
        reports={}
        for arm,features in [('zero',np.zeros_like(own)),('own',own),('shuffled',own[mapping])]:
            directory=seedout/arm;directory.mkdir()
            embeddings=with_questionnaire(shared,features)
            np.save(directory/'state_features.npy',features)
            save(directory/'feature_audit.json',{'state_dim':state_dim(embeddings),
                 'feature_sha256':digest(directory/'state_features.npy'),
                 'feature_min':float(features.min()),'feature_max':float(features.max()),
                 'nonzero_user_fraction':float(np.any(features!=0,axis=1).mean()),
                 'unchanged_from_own_fraction':float(np.all(features==own,axis=1).mean())})
            # Identical features are used for training, validation and final test;
            # they are observed questionnaire covariates, never reward labels.
            policy=train(predicted,gt,distance,pools,ids,groups,embeddings,args,seed,directory)
            for scenario in ['fixed','natural']:
                report=evaluate(policy,predicted,gt,distance,pools['test'],ids,groups,embeddings,args,scenario)
                for r in report['users']:r['historical_count']=int(counts[r['user_id']])
                report['subgroups']={}
                for g in [0,1,2,3,4,'count_10_12','count_13_plus']:
                    rows=[r for r in report['users'] if (r['group']==g if isinstance(g,int) else
                       (r['historical_count']<=12 if g=='count_10_12' else r['historical_count']>12))]
                    if rows:report['subgroups'][str(g)]={'n':len(rows),**{k:float(np.mean([r[k] for r in rows]))
                                                       for k in ['total_reward','steps','reward_per_step']}}
                save(directory/f'{scenario}_test.json',report);reports[arm,scenario]=report
                results.append({'seed':seed,'arm':arm,'scenario':scenario,**report['summary']})
        for scenario in ['fixed','natural']:
            for a,b in [('zero','own'),('zero','shuffled'),('shuffled','own')]:
                contrasts.append({'seed':seed,'scenario':scenario,'first':a,'second':b,
                                  'metrics':paired(reports[a,scenario],reports[b,scenario],seed)})
        save(out/'results.json',results)
        save(out/'paired_bootstrap.json',{'note':'Paired user intervals exclude training and shuffle uncertainty.',
                                         'comparisons':contrasts})
    save(out/'summary.json',{'status':'completed','smoke_only':bool(args.smoke_users),'runs':results})
    print(f'Completed. Results: {out}',flush=True)


if __name__=='__main__':
    p=parser();p.description=__doc__
    p.add_argument('--shuffle-seed',type=int,default=20261124)
    args=p.parse_args()
    for k in ['epochs','steps_per_epoch','batch_size','capacity','hidden','target_update','window','horizon','max_turn','threads']:
        if getattr(args,k)<1:raise SystemExit(f'{k} must be positive')
    if args.capacity<args.batch_size or args.smoke_users<0 or args.max_turn>1000 or args.horizon>1000:
        raise SystemExit('Invalid capacity, smoke users or horizon')
    if not 0<=args.epsilon<=1 or not 0<=args.gamma<=1 or not np.isfinite(args.lr) or args.lr<=0 or args.learning_starts<0:
        raise SystemExit('Invalid optimization settings')
    if len(set(args.seeds))!=len(args.seeds):raise SystemExit('Duplicate seeds')
    main(args)
