"""Global vs own/shuffled questionnaire recording networks on held-out users."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch

from experiments.yahoo_em.data import load_yahoo, PREFIX, GROUP_NAMES
from experiments.yahoo_downstream.core import digest
from experiments.yahoo_downstream.questionnaire_reward import permutation_within_pools
from .core import Marginal, Recording, rating_counts, recording_distribution, count_nll, random_nll, fit, metrics


def save(path,x):Path(path).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def main(args):
    if args.device=='auto':args.device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    ratings,rows,groups,ids,splits,audit=load_yahoo(args.data,args.smoke_users_per_group,args.split_seed)
    active,random=rating_counts(ratings,rows);n_items=ratings.shape[1]
    answers=np.loadtxt(Path(args.data)/(PREFIX+'survey-answers.txt'),dtype=int)[ids,1:6]
    q=torch.tensor((answers-3)/2,dtype=torch.float32,device=args.device)
    a=torch.tensor(active,dtype=torch.float32,device=args.device)
    r=torch.tensor(random,dtype=torch.float32,device=args.device)
    cal,val,test=[torch.as_tensor(splits[k],device=args.device) for k in ['calibration','validation','test']]
    mapping=permutation_within_pools(len(ids),splits,args.shuffle_seed)
    shuffled=q[torch.as_tensor(mapping,device=args.device)]
    save(out/'config.json',{'args':vars(args),'python':platform.python_version(),'torch':torch.__version__,
      'numpy':np.__version__,'commit':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),
      'source_sha256':{str(p):digest(p) for p in [*Path(__file__).parent.glob('*.py'),
         Path('experiments/yahoo_em/data.py'),Path('experiments/yahoo_downstream/questionnaire_reward.py')]},
      'interpretation':'Composite count-model score, not ground-truth propensity error. Previously explored internal holdout.'})
    save(out/'data_audit.json',audit)
    save(out/'splits.json',{k:(ids[v]+1).tolist() for k,v in splits.items()})
    save(out/'questionnaire_mapping.json',{'user_ids':(ids+1).tolist(),'donor_ids':(ids[mapping]+1).tolist()})
    # Nuisance uses ONLY calibration random labels; validation random labels
    # select its checkpoint. It is frozen and identical in all three arms.
    prior=(r[cal].sum(0)+.5)/(r[cal].sum()+2.5)
    nuisance=Marginal(prior).to(args.device)
    nuisance_info=fit(nuisance,
        lambda m:random_nll(m(q[cal]),r[cal]).mean()+args.marginal_l2*m.linear.weight.square().mean(),
        lambda m:random_nll(m(q[val]),r[val]).mean(),args.epochs,args.patience,args.lr)
    with torch.no_grad():p=nuisance(q).detach()
    save(out/'marginal_selection.json',nuisance_info)
    torch.save(nuisance.state_dict(),out/'marginal.pt')
    print(json.dumps({'stage':'shared_star_marginal','best_epoch':nuisance_info['best_epoch'],
                      'validation_nll':nuisance_info['validation_nll']}),flush=True)
    raw_mu=a[cal].sum(0)/(len(cal)*n_items*prior)
    global_model=Recording(raw_mu.clamp(1e-5,1-1e-5)).to(args.device)
    def likelihood(m,x,index):
        return count_nll(recording_distribution(p[index],m(x[index])),a[index],n_items).mean()
    global_info=fit(global_model,lambda m:likelihood(m,q,cal),lambda m:likelihood(m,q,val),
                    args.epochs,args.patience,args.lr)
    with torch.no_grad():baseline=global_model(q[:1])[0].detach()
    save(out/'global_selection.json',global_info)
    save(out/'global_probability.json',{'initial_unclipped_mu':raw_mu.cpu().tolist(),
         'selected_mu':baseline.cpu().tolist(),'initial_clipped_stars':(torch.nonzero((raw_mu<1e-5)|(raw_mu>1-1e-5)).flatten()+1).cpu().tolist()})
    torch.save(global_model.state_dict(),out/'global.pt')
    print(json.dumps({'stage':'global_recording','best_epoch':global_info['best_epoch'],
                      'validation_nll':global_info['validation_nll']}),flush=True)
    results,contrasts=[],[]
    for seed in args.seeds:
        directory=out/f'seed_{seed}';directory.mkdir()
        models={'global':(global_model,q,global_info)}
        # Same initial weights, only the questionnaire-to-user matching differs.
        for arm,x in [('own',q),('shuffled',shuffled)]:
            torch.manual_seed(seed)
            model=Recording(baseline,True,args.hidden).to(args.device)
            initial_hash=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
            info=fit(model,lambda m:likelihood(m,x,cal)+args.shrinkage*m.delta(x[cal]).square().mean(),
                     lambda m:likelihood(m,x,val),args.epochs,args.patience,args.lr)
            info['initial_sha256']=initial_hash
            save(directory/f'{arm}_selection.json',info)
            torch.save(model.state_dict(),directory/f'{arm}.pt')
            models[arm]=(model,x,info)
            print(json.dumps({'seed':seed,'arm':arm,'best_epoch':info['best_epoch'],
                              'validation_nll':info['validation_nll']}),flush=True)
        # Evaluate test labels only after every arm has been selected.
        reports={}
        for arm,(model,x,info) in models.items():
            with torch.no_grad():mu=model(x[test]);per=metrics(p[test],mu,a[test],n_items)
            reports[arm]=per
            users=[{'user_id':int(ids[u]+1),'group':GROUP_NAMES[groups[u]],
                    **{k:float(v[i]) for k,v in per.items()}} for i,u in enumerate(splits['test'])]
            summary={k:float(v.mean()) for k,v in per.items()}
            subgroups={g:{'n':sum(z['group']==g for z in users),
                         **{k:float(np.mean([z[k] for z in users if z['group']==g])) for k in per}}
                       for g in GROUP_NAMES if any(z['group']==g for z in users)}
            save(directory/f'{arm}_test.json',{'metrics':summary,'users':users,'subgroups':subgroups,
                'mu_min':float(mu.min()),'mu_max':float(mu.max()),
                'boundary_fraction':float(((mu<1e-4)|(mu>.95)).float().mean())})
            results.append({'seed':seed,'arm':arm,**summary})
        for first,second in [('global','own'),('global','shuffled'),('shuffled','own')]:
            rng=np.random.default_rng(args.split_seed);diff={}
            for k in reports[first]:
                delta=reports[second][k]-reports[first][k]
                means=[rng.choice(delta,len(delta),replace=True).mean() for _ in range(2000)]
                diff[k]={'second_minus_first':float(delta.mean()),'ci95':np.quantile(means,[.025,.975]).tolist()}
            contrasts.append({'seed':seed,'first':first,'second':second,'metrics':diff})
        save(out/'results.json',results);save(out/'paired_bootstrap.json',{'comparisons':contrasts,
          'note':'Conditional on fitted models/marginal/single permutation; excludes their estimation uncertainty.'})
    with torch.no_grad():
        nuisance_test={'global_random_nll':float(random_nll(prior.expand(len(test),-1),r[test]).mean()),
                       'conditional_random_nll':float(random_nll(p[test],r[test]).mean())}
    save(out/'marginal_test.json',nuisance_test)
    save(out/'summary.json',{'status':'completed','smoke_only':bool(args.smoke_users_per_group),
         'runs':results,'note':'Global and nuisance models shared across seeds; not three independent global fits.'})
    print(f'Completed. Results: {out}',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',default='data/YahooR3/data_raw');ap.add_argument('--output',required=True)
    ap.add_argument('--device',default='cpu');ap.add_argument('--seeds',type=int,nargs='+',default=[2022,2023,2024])
    ap.add_argument('--epochs',type=int,default=400);ap.add_argument('--patience',type=int,default=30)
    ap.add_argument('--hidden',type=int,default=8);ap.add_argument('--lr',type=float,default=.01)
    ap.add_argument('--shrinkage',type=float,default=.001);ap.add_argument('--marginal-l2',type=float,default=.01)
    ap.add_argument('--split-seed',type=int,default=20261123);ap.add_argument('--shuffle-seed',type=int,default=20261124)
    ap.add_argument('--smoke-users-per-group',type=int,default=0);ap.add_argument('--threads',type=int,default=2)
    args=ap.parse_args()
    if min(args.epochs,args.patience,args.hidden,args.threads)<1 or (args.smoke_users_per_group and args.smoke_users_per_group<5):
        raise SystemExit('Invalid dimensions, budgets or smoke users')
    if not all(np.isfinite(x) for x in [args.lr,args.shrinkage,args.marginal_l2]) or args.lr<=0 or min(args.shrinkage,args.marginal_l2)<0:
        raise SystemExit('Invalid optimization settings')
    if len(set(args.seeds))!=len(args.seeds):raise SystemExit('Duplicate seeds')
    main(args)
