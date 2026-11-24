"""Frozen global/own/shuffled recording networks as missingness models for DeepFM EM."""
import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

from experiments.yahoo_em import run as em
from experiments.yahoo_em.data import load_yahoo, PREFIX
from experiments.yahoo_downstream.core import digest
from .core import Recording


def read(path):return json.loads(Path(path).read_text())


def validate_mapping(source, splits):
    users=np.asarray(source['user_ids'],dtype=int)
    donors=np.asarray(source['donor_ids'],dtype=int)
    if users.ndim!=1 or donors.shape!=users.shape or len(set(users))!=len(users) or not np.array_equal(np.sort(users),np.sort(donors)):
        raise ValueError('Invalid recording questionnaire mapping')
    if sorted(sum(splits.values(),[]))!=sorted(users.tolist()):
        raise ValueError('Recording splits must partition source users')
    lookup=dict(zip(users.tolist(),donors.tolist()))
    for pool in splits.values():
        if set(lookup[u] for u in pool)!=set(pool):raise ValueError('Donor mapping crosses data splits')
    return dict(zip(users.tolist(),donors.tolist()))


def prepare_data(args,root):
    config=read(root/'config.json')['args']
    if read(root/'summary.json')['status']!='completed':raise ValueError('Recording experiment incomplete')
    ratings,rows,groups,ids,splits,audit=load_yahoo(args.data,config['smoke_users_per_group'],config['split_seed'])
    if audit['sha256']!=read(root/'data_audit.json')['sha256']:
        raise ValueError('Raw data hashes differ from recording experiment')
    saved_splits=read(root/'splits.json')
    if {k:(ids[v]+1).tolist() for k,v in splits.items()}!=saved_splits:
        raise ValueError('Recording and EM user splits differ')
    donors=validate_mapping(read(root/'questionnaire_mapping.json'),saved_splits)
    # Subset each ORIGINAL split only for smoke. The full donor map is retained.
    selected=np.arange(len(ids))
    if args.smoke_users_per_split:
        selected=np.sort(np.concatenate([v[:args.smoke_users_per_split] for v in splits.values()]))
        reverse=np.full(len(ids),-1,dtype=int);reverse[selected]=np.arange(len(selected))
        rows=rows[np.isin(rows[:,0],selected)].copy();rows[:,0]=reverse[rows[:,0]]
        splits={k:reverse[v[np.isin(v,selected)]] for k,v in splits.items()}
        ratings,groups,ids=ratings[selected],groups[selected],ids[selected]
    answers=np.loadtxt(Path(args.data)/(PREFIX+'survey-answers.txt'),dtype=int)
    own=((answers[ids,1:6]-3)/2).astype(np.float32)
    donor_ids=np.array([donors[int(u+1)] for u in ids])
    shuffled=((answers[donor_ids-1,1:6]-3)/2).astype(np.float32)
    return ratings,rows,groups,ids,splits,audit,own,shuffled,config


@torch.no_grad()
def load_probabilities(root,seed,hidden,own,shuffled):
    tables={};audit={}
    for arm,features in [('global',own),('own',own),('shuffled',shuffled)]:
        path=root/'global.pt' if arm=='global' else root/f'seed_{seed}/{arm}.pt'
        selection=root/'global_selection.json' if arm=='global' else root/f'seed_{seed}/{arm}_selection.json'
        model=Recording(torch.full((5,),.1),arm!='global',hidden)
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        model.eval().requires_grad_(False)
        mu=model(torch.from_numpy(features)).numpy().copy()
        if mu.shape!=features.shape or not np.isfinite(mu).all() or mu.min()<=0 or mu.max()>=1:
            raise ValueError('Invalid frozen recording probabilities')
        tables[arm]=mu
        audit[arm]={'checkpoint':str(path),'sha256':digest(path),'selection_sha256':digest(selection),
                    'recording_best_epoch':read(selection)['best_epoch'],'mu_min':float(mu.min()),'mu_max':float(mu.max()),
                    'mu_sha256':hashlib.sha256(mu.tobytes()).hexdigest()}
    audit['shuffled_exactly_global']=bool(np.array_equal(tables['shuffled'],tables['global']))
    return tables,audit


def main(args):
    if args.device=='auto':args.device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    root=Path(args.recording_root)
    ratings,rows,groups,ids,splits,audit,own,shuffled,source_args=prepare_data(args,root)
    if any(seed not in source_args['seeds'] for seed in args.seeds):raise ValueError('Requested recording seed unavailable')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    em.write_json(out/'config.json',{'args':vars(args),'source_recording_args':source_args,
        'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
        'commit':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),
        'source_sha256':{str(p):digest(p) for p in [Path(__file__),Path('experiments/yahoo_em/run.py'),
           Path('experiments/yahoo_em/model.py'),Path('experiments/yahoo_em/data.py'),Path('experiments/yahoo_recording/core.py')]},
        'recording_metadata_sha256':{n:digest(root/n) for n in ['config.json','data_audit.json','splits.json','questionnaire_mapping.json']},
        'note':'Frozen recording mechanisms; common preference initialization; shared previously explored internal holdout.'})
    em.write_json(out/'data_audit.json',{'source_universe':audit,'effective_users':len(ids),
          'effective_active_ratings':int((ratings>=0).sum()),'effective_random_ratings':len(rows),
          'effective_split_counts':{k:len(v) for k,v in splits.items()}})
    em.write_json(out/'splits.json',{k:(ids[v]+1).tolist() for k,v in splits.items()})
    np.save(out/'original_user_ids.npy',ids+1);np.save(out/'groups.npy',groups)
    rows_val=rows[np.isin(rows[:,0],splits['validation'])]
    rows_test=rows[np.isin(rows[:,0],splits['test'])]
    results,comparisons=[],[]
    for seed in args.seeds:
        seedout=out/f'seed_{seed}';seedout.mkdir()
        tables,mechanism_audit=load_probabilities(root,seed,source_args['hidden'],own,shuffled)
        em.write_json(seedout/'recording_audit.json',mechanism_audit)
        initial=em.initialize(ratings,args,seed)
        initial_hash=hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in initial.values())).hexdigest()
        em.write_json(seedout/'initialization.json',{'sha256':initial_hash,'shared_across_arms':True})
        # Train/select ALL arms without looking at their test results.
        for arm in ['global','own','shuffled']:
            directory=seedout/arm;directory.mkdir()
            np.save(directory/'recording_mu.npy',tables[arm])
            model,_=em.fit(ratings,rows_val,groups,tables[arm],initial,args,seed,arm,directory,
                           recording_groups=np.arange(len(ids)))
            del model
        per_arm={}
        for arm in ['global','own','shuffled']:
            directory=seedout/arm
            model=em.make_model(ratings,args)
            model.load_state_dict(torch.load(directory/'best.pt',map_location=args.device,weights_only=True)['state_dict'])
            summary,users=em.evaluate(model,rows_test,groups,(ratings>=0).sum(1),args)
            for u in users:u['original_user_id']=int(ids[u['user_index']]+1)
            em.write_json(directory/'test.json',{'metrics':summary,'users':users,'subgroups':em.subgroup_metrics(users)})
            em.export_scores(model,ratings,args,directory)
            per_arm[arm]=users;results.append({'seed':seed,'arm':arm,**summary})
        for first,second in [('global','own'),('global','shuffled'),('shuffled','own')]:
            delta=em.bootstrap_difference(per_arm[first],per_arm[second],source_args['split_seed'])
            delta={k:{'second_minus_first':v['C_minus_B'],'ci95':v['ci95'],'users':v['users']} for k,v in delta.items()}
            comparisons.append({'seed':seed,'first':first,'second':second,'metrics':delta})
        em.write_json(out/'results.json',results)
        em.write_json(out/'paired_bootstrap.json',{'comparisons':comparisons,
          'note':'User intervals conditional on fixed models; excludes recording and training estimation uncertainty.'})
    em.write_json(out/'summary.json',{'status':'completed',
       'smoke_only':bool(args.smoke_users_per_split or source_args['smoke_users_per_group']),
       'runs':results,'note':'No DQN. Prediction metrics on random collected ratings; all mechanisms frozen.'})
    print(f'Completed. Results: {out}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--data',default='data/YahooR3/data_raw');p.add_argument('--device',default='auto')
    p.add_argument('--seeds',type=int,nargs='+',default=[2022,2023,2024])
    p.add_argument('--smoke-users-per-split',type=int,default=0)
    p.add_argument('--warmup-epochs',type=int,default=2);p.add_argument('--em-rounds',type=int,default=50)
    p.add_argument('--m-epochs',type=int,default=1);p.add_argument('--batch-size',type=int,default=4096)
    p.add_argument('--dim',type=int,default=16);p.add_argument('--hidden',type=int,default=64)
    p.add_argument('--lr',type=float,default=.001);p.add_argument('--l2',type=float,default=1e-6)
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    if min(args.em_rounds,args.m_epochs,args.batch_size,args.dim,args.hidden,args.threads)<1:
        raise SystemExit('Invalid training budget or dimensions')
    if args.warmup_epochs<0 or args.smoke_users_per_split<0 or not np.isfinite(args.lr) or not np.isfinite(args.l2) or args.lr<=0 or args.l2<0:
        raise SystemExit('Invalid optimization settings')
    if len(set(args.seeds))!=len(args.seeds):raise SystemExit('Duplicate seeds')
    main(args)
