"""Local endpoint analysis; never opens a transcript or decodes a token."""
import argparse
import hashlib
import json
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
from scipy.stats import binomtest, spearmanr, t as student_t
from sklearn.metrics import roc_auc_score


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, x): p.write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')
def auc(a,b): return float(roc_auc_score(np.r_[np.ones(len(a)),np.zeros(len(b))],np.r_[a,b]))

def stats(a,b,cfg):
    d=a-b;n=len(d);rng=np.random.default_rng(cfg['seed']);idx=rng.integers(0,n,(cfg['bootstrap_draws'],n))
    boots=d[idx].mean(1);c=(d>0)+.5*(d==0)
    # Recompute pooled AUROC on paired resamples, preserving each task's two observations.
    ab=np.array([auc(a[i],b[i]) for i in idx])
    swaps=rng.choice([-1,1],(cfg['shuffle_draws'],n));null=(swaps*d).mean(1)
    swapped_aucs=np.array([auc(np.where(s>0,a,b),np.where(s>0,b,a)) for s in swaps])
    p=lambda obs, vals: float((1+np.count_nonzero(np.abs(vals)>=abs(obs)))/(1+len(vals)))
    return {'n_tasks':n,'mean_difference':float(d.mean()),'difference_ci95':np.percentile(boots,[2.5,97.5]).tolist(),
       'positive_pairs':int((d>0).sum()),'ties':int((d==0).sum()),'paired_concordance':float(c.mean()),
       'sign_p_two_sided':float(binomtest(int((d>0).sum()),int((d!=0).sum())).pvalue) if (d!=0).any() else 1.,
       'pooled_auroc':auc(a,b),'auroc_ci95':np.percentile(ab,[2.5,97.5]).tolist(),
       'task_swap_p_mean_two_sided':p(d.mean(),null),'task_swap_p_auroc_two_sided':p(auc(a,b)-.5,swapped_aucs-.5),
       'task_swap_null_mean_ci95':np.percentile(null,[2.5,97.5]).tolist(),
       'task_swap_null_auroc_ci95':np.percentile(swapped_aucs,[2.5,97.5]).tolist(),
       'mean_left':float(a.mean()),'mean_right':float(b.mean())}

def adjustment(left,right):
    y=np.array([a['score']-b['score'] for a,b in zip(left,right)])
    cov=np.array([[np.log1p(a[k])-np.log1p(b[k]) for k in ('epoch_tokens_sum','n_commands')] for a,b in zip(left,right)])
    x=np.c_[np.ones(len(y)),cov];beta=np.linalg.lstsq(x,y,rcond=None)[0];res=y-x@beta
    # HC3 heteroskedasticity-robust covariance; one independent row per main task.
    inv=np.linalg.pinv(x.T@x);h=np.einsum('ij,jk,ik->i',x,inv,x);v=inv@(x.T@((res/(1-h))[:,None]**2*x))@inv
    se=np.sqrt(np.diag(v));q=student_t.ppf(.975,len(y)-x.shape[1])
    return {'predictors':['intercept_at_equal_lengths','delta_log1p_epoch_tokens','delta_log1p_command_count'],
      'coef':beta.tolist(),'hc3_se':se.tolist(),'intercept_ci95':[float(beta[0]-q*se[0]),float(beta[0]+q*se[0])],
      'note':'Associational sensitivity only. Epoch token sum includes repeated context across epochs; not unique emitted tokens.'}

def main(cfgpath,out):
    cfg=json.loads(cfgpath.read_text());root=Path(cfg['input_root']);out.mkdir(parents=True,exist_ok=False)
    roster={r['key']:r['main'] for r in json.loads(Path(cfg['roster']).read_text())['cells']}
    assert len(roster)==len(set(roster.values()))==196
    p=root/'probes_block_output.npz';assert sha(p)==cfg['probe_sha256'];probe=np.load(p)
    w,mu,sd=[probe[k][cfg['layer']].astype(np.float64) for k in ('coef','scaler_mean','scaler_scale')]
    assert np.isfinite(np.r_[w,mu,sd]).all() and (sd>0).all()
    bychunk=defaultdict(list);runmeta={};excluded=Counter();manifest=[]
    limits={'rerun_strict':(1,20),'scaleup_canary':(21,25),'scaleup_l2':(26,185),'scaleup_topup':(186,196),'second_attack':(1,196)}
    for batch in sorted(root.glob('2026*')):
        c=json.loads((batch/'config.json').read_text());assert c['model']==cfg['model'] and c['revision']==cfg['revision']
        bounds=next(v for k,v in limits.items() if k in batch.name)
        manifest.append({'path':str(batch/'config.json'),'sha256':sha(batch/'config.json')})
        for f in sorted((batch/'meta').rglob('*.json')):
            m=json.loads(f.read_text());arm=m['arm'];num=int(m['task'].split('_')[0][1:])
            if arm not in ('attack','honest_a','honest_b') or not bounds[0]<=num<=bounds[1]:continue
            if m.get('redrawn_because') or m.get('attempt_kind')=='rerun':excluded['redrawn_or_rerun_epochs']+=1;continue
            if arm=='attack':arm='attack2' if 'second_attack' in batch.name else 'attack1'
            runkey=(arm,m['task']);rd=runmeta.setdefault(runkey,{'arm':arm,'task':m['task'],'main_task':roster[m['task']], 'run':m['run'],'epochs':[],'input_hashes':[]})
            assert rd['run']==m['run'],'Several counted runs per arm/task'
            rd['epochs'].append({'epoch':m['epoch'],'tokens':m['n_tokens']});rd['input_hashes'].append(m['inputs_sha256'])
            for call in m['calls']:
                if call.get('kind')!='command':continue
                if call.get('readout') is None or call.get('row') is None:excluded[arm+'_missing_command_rows']+=1;continue
                assert call['call_index'] is not None
                record={'arm':arm,'task':m['task'],'run':m['run'],'irid':call['irid'],'call_index':call['call_index'],
                        'epoch':m['epoch'],'readout':call['readout'],'row':m['row_start']+call['row'],'turn':call['turn']}
                bychunk[(batch,m['chunk'])].append(record)
    scores={};duplicates=[]
    for (batch,chunk),recs in sorted(bychunk.items()):
        matches=list((batch/'L31').rglob(chunk+'.npy'));assert len(matches)==1
        matrix=np.load(matches[0],mmap_mode='r');x=matrix[[r['row'] for r in recs]].astype(np.float64)
        assert x.shape[1]==len(w) and np.isfinite(x).all()
        vals=((x-mu)/sd)@w
        for r,v in zip(recs,vals):
            k=(r['arm'],r['run'],r['irid'],r['call_index']);r=r|{'score':float(v)}
            if k in scores:
                duplicates.append({'key':k,'score_gap':abs(v-scores[k]['score'])})
                if r['epoch']>=scores[k]['epoch']:continue
            scores[k]=r
    previous=json.loads(Path(cfg['parity_scores']).read_text());parity=[]
    for r in previous['keys']:
        got=scores[('attack1',r['run'],r['irid'],r['call'])];parity.append(abs(got['score']-r['score']))
    assert len(parity)==60 and max(parity)<1e-8,'Frozen-score parity failed'
    byrun=defaultdict(list)
    for r in scores.values():byrun[(r['arm'],r['task'])].append(r)
    runs=[]
    for k,m in sorted(runmeta.items()):
        rs=byrun.get(k,[])
        if not rs:excluded[m['arm']+'_no_endpoint_run']+=1;continue
        runs.append(m|{'score':float(np.mean([r['score'] for r in rs])),'n_commands':len(rs),
          'epoch_tokens_sum':sum(e['tokens'] for e in m['epochs']),'max_context_tokens':max(e['tokens'] for e in m['epochs'])})
    ix={(r['arm'],r['task']):r for r in runs};arms=['attack1','attack2','honest_a','honest_b']
    common=set.intersection(*[{r['task'] for r in runs if r['arm']==a} for a in arms]);comparisons={}
    for a,b in cfg['comparisons']:
        keys=sorted({r['task'] for r in runs if r['arm']==a}&{r['task'] for r in runs if r['arm']==b})
        left=[ix[(a,k)] for k in keys];right=[ix[(b,k)] for k in keys]
        result=stats(np.array([r['score'] for r in left]),np.array([r['score'] for r in right]),cfg)
        result['length_adjustment']=adjustment(left,right);result['task_keys']=keys
        ka=sorted(common);result['all_four_intersection']=stats(np.array([ix[(a,k)]['score'] for k in ka]),np.array([ix[(b,k)]['score'] for k in ka]),cfg)
        comparisons[a+'_vs_'+b]=result
    correlations={}
    for a in arms:
        rr=[r for r in runs if r['arm']==a];correlations[a]={}
        for col in ['epoch_tokens_sum','max_context_tokens','n_commands']:
            rho,pv=spearmanr([r['score'] for r in rr],[r[col] for r in rr]);correlations[a][col]={'spearman_rho':float(rho),'p':float(pv)}
    result={'created_pt':datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),'config':cfg,'script_sha256':sha(Path(__file__)),
      'config_sha256':sha(cfgpath),'input_configs':manifest,'parity_60_max_abs_gap':max(parity),'duplicate_calls':duplicates,
      'coverage':{a:{'n_runs':sum(r['arm']==a for r in runs),'missing_tasks':sorted(set(roster)-{r['task'] for r in runs if r['arm']==a})} for a in arms},
      'excluded':dict(excluded),'comparisons':comparisons,'length_correlations':correlations,
      'threshold':{'status':'unavailable_pending_compatible_separate_honest_calibration','reason':'Legacy last-parameter call_end is not generally the command endpoint. No threshold has been substituted.'},'paid_calls':0}
    save(out/'metrics.json',result)
    (out/'runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in runs));(out/'commands.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scores.values()))
    print(json.dumps({'coverage':result['coverage'],'parity':max(parity),'duplicates':len(duplicates),'comparisons':{k:{j:v[j] for j in ['n_tasks','mean_difference','difference_ci95','paired_concordance','pooled_auroc','auroc_ci95','task_swap_p_mean_two_sided','length_adjustment']} for k,v in comparisons.items()},'correlations':correlations},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();main(a.config,a.output)
