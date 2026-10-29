"""Approved numeric-only Modal pilot. Fixed input hashes, checkpoints, bounded calls, no retries."""
import hashlib,json,time
from pathlib import Path
import modal

APP='probe-shift_aswin-surya_sub-evalaware_start_kill-2026-11-09'
VOLUME='probe-shift_aswin-surya_sub-evalaware_acts_kill-2026-11-09'
app=modal.App(APP);vol=modal.Volume.from_name(VOLUME,version=2);hf=modal.Volume.from_name('probe-shift_aswin-surya_hf-cache').read_only()
ACTS=Path('/acts')
IMPLEMENTATION_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PIP=['torch==2.8.0','transformers==5.19.0','accelerate>=1.1.0','numpy','huggingface_hub','flash-linear-attention==0.5.2']
gpu_image=modal.Image.debian_slim(python_version='3.12').uv_pip_install(*PIP).env({'HF_HOME':'/hf','HF_HUB_OFFLINE':'1'})
cpu_image=modal.Image.debian_slim(python_version='3.12').uv_pip_install('numpy','scikit-learn==1.7.2').uv_pip_install('torch==2.8.0',index_url='https://download.pytorch.org/whl/cpu').add_local_python_source('probe_shift')

def inputs(run,name):
    d=ACTS/run;manifest=json.loads((d/'input_manifest.json').read_text());p=d/(name+'.jsonl');raw=p.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==manifest['files'][p.name]
    if name=='real_inputs':
        assert hashlib.sha256((d/'baseline.json').read_bytes()).hexdigest()==manifest['files']['baseline.json'],'Frozen baseline hash mismatch'
    return [json.loads(l) for l in raw.decode().splitlines()],manifest['config']

def bind_checkpoint(run, stage, existing, **extra):
    """Refuse legacy or changed checkpoints rather than silently combine executions."""
    d=ACTS/run
    expected={'input_manifest_sha256':hashlib.sha256((d/'input_manifest.json').read_bytes()).hexdigest(),
              'implementation_sha256':IMPLEMENTATION_SHA, **extra}
    p=d/'checkpoint_provenance'/(stage+'.json')
    if p.exists():
        if json.loads(p.read_text())!=expected:raise ValueError('Checkpoint provenance changed: '+stage)
    else:
        if existing:raise ValueError('Unbound legacy checkpoint: use a fresh run ID: '+stage)
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(expected));vol.commit()
    return expected

def model(cfg):
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(4)
    return AutoModelForCausalLM.from_pretrained(cfg['model'],revision=cfg['revision'],dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval()

@app.function(image=gpu_image,gpu='H100',cpu=(4,4),memory=(65536,65536),volumes={'/hf':hf,'/acts':vol},timeout=900,startup_timeout=300,retries=0,max_containers=1,scaledown_window=2)
def extract(run):
    import numpy as np
    import torch
    items,cfg=inputs(run,'synthetic_inputs');d=ACTS/run/'extract';d.mkdir(parents=True,exist_ok=True)
    bind_checkpoint(run,'extract',list(d.glob('block*.npz')))
    lm=model(cfg);start=time.monotonic();forward=0
    for bi in range((len(items)+199)//200):
        path=d/f'block{bi:03d}.npz'
        if path.exists():continue
        rows={l:[] for l in cfg['layers']}
        for it in items[bi*200:(bi+1)*200]:
            acts={};position=it['position']
            def hook(layer):
                def save(module,args,output):
                    h=output[0] if isinstance(output,tuple) else output;acts[layer]=h[0,position].to(torch.float16).cpu().numpy()
                return save
            handles=[lm.model.layers[l].register_forward_hook(hook(l)) for l in cfg['layers']]
            ids=it['ids'];ids=ids+[cfg['pad_token_id']]*(-len(ids)%1024);t=time.monotonic()
            try:
                with torch.inference_mode():lm.model(input_ids=torch.tensor([ids],device='cuda'),use_cache=False)
            finally:
                for h in handles:h.remove()
            forward+=time.monotonic()-t
            for l in cfg['layers']:rows[l].append(acts[l])
        np.savez(path,**{f'L{l}':np.stack(v) for l,v in rows.items()});vol.commit();print(f'Synthetic block {bi+1}: saved, forward={forward:.1f}s',flush=True)
    result={'dialogues':len(items),'layers':cfg['layers'],'forward_s':forward,'runtime_after_model_load_s':time.monotonic()-start}
    (d/'complete.json').write_text(json.dumps(result));vol.commit();return result

@app.function(image=cpu_image,cpu=(16,16),memory=(16384,16384),volumes={'/acts':vol},timeout=600,startup_timeout=300,retries=0,max_containers=1,scaledown_window=2)
def fit(run):
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from probe_shift import torch_probe
    torch.set_num_threads(16);items,cfg=inputs(run,'synthetic_inputs');d=ACTS/run;assert (d/'extract/complete.json').exists()
    bind_checkpoint(run,'extract',list((d/'extract').glob('block*.npz')))
    bind_checkpoint(run,'fit',list(d.glob('fit_tight_layer*.json')),
        solver_sha256=hashlib.sha256(Path(torch_probe.__file__).read_bytes()).hexdigest(),
        activation_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((d/'extract').glob('block*.npz'))})
    y=np.array([r['label'] for r in items]);split=np.array([r['split'] for r in items]);pair=np.array([r['pair'] for r in items]);rng=np.random.default_rng(cfg['seed']);labels=[y]
    for _ in range(cfg['null_draws']):
        flips={p:rng.random()<.5 for p in sorted(set(pair[split!='heldout']))};labels.append(np.array([1-v if s!='heldout' and flips[p] else v for v,s,p in zip(y,split,pair)]))
    Y=np.stack(labels,1);train=np.flatnonzero(split=='train');dev=np.flatnonzero(split=='dev');held=np.flatnonzero(split=='heldout');results={};frozen={};scores={}
    for layer in cfg['layers']:
        checkpoint=d/f'fit_tight_layer{layer}.json'
        if checkpoint.exists():
            q=json.loads(checkpoint.read_text());results[layer]=q['result'];frozen[layer]=q['probe'];scores[layer]=q['heldout_scores'];continue
        X=np.concatenate([np.load(p)[f'L{layer}'] for p in sorted((d/'extract').glob('block*.npz'))]).astype(np.float64);assert len(X)==len(items) and np.isfinite(X).all()
        mu=X[train].mean(0);sd=X[train].std(0)+1e-6;Z=torch.from_numpy((X-mu)/sd);zt=Z[train];yt=torch.from_numpy(Y[train].astype(np.float64));t=time.monotonic()
        w,_,n=torch_probe.fit(zt,yt,C=cfg['C'],max_iter=5000,tol=1e-8,intercept=False,tolerance_change=1e-14)
        with torch.no_grad():
            grad=(cfg['C']*zt.T@(torch.sigmoid(zt@w)-yt)+w)/len(zt);logit=(Z@w).numpy()
        print(f'L{layer} numerical check: max gradient={grad.abs().max().item():.3g}, true-label gradient={grad[:,0].abs().max().item():.3g}',flush=True)
        assert np.isfinite(logit).all() and grad.abs().max().item()<=1e-7,'Unconverged fit'
        result={'dev':[float(roc_auc_score(Y[dev,k],logit[dev,k])) for k in range(Y.shape[1])],
           'heldout':[float(roc_auc_score(y[held],logit[held,k])) for k in range(Y.shape[1])], 'evals':n,'max_grad':grad.abs().max().item(),'seconds':time.monotonic()-t}
        if layer==32:
            sk=LogisticRegression(C=cfg['C'],fit_intercept=False,max_iter=10000,tol=1e-6).fit(Z[train].numpy(),y[train]);difference=float(np.max(np.abs(sk.decision_function(Z.numpy())-logit[:,0])))
            result['sklearn_max_logit_difference']=difference
            result['sklearn_train_max_logit_difference']=float(np.max(np.abs(sk.decision_function(Z[train].numpy())-logit[train,0])))
            result['sklearn_logit_correlation']=float(np.corrcoef(sk.decision_function(Z.numpy()),logit[:,0])[0,1])
            (d/f'solver_check_layer{layer}.json').write_text(json.dumps(result));vol.commit()
            print(f'Independent solver check: max logit difference={difference:.8f}',flush=True)
            assert difference<.02
        probe={'layer':layer,'w':w[:,0].numpy().tolist(),'mu':mu.tolist(),'sd':sd.tolist()};heldout=[{'id':items[i]['id'],'pair':items[i]['pair'],'label':int(y[i]),'score':float(logit[i,0])} for i in held]
        checkpoint.write_text(json.dumps({'result':result,'probe':probe,'heldout_scores':heldout}));vol.commit();results[layer]=result;frozen[layer]=probe;scores[layer]=heldout
        print(f'L{layer}: dev={result["dev"][0]:.3f}, heldout={result["heldout"][0]:.3f}, {result["seconds"]:.1f}s',flush=True)
    pick=lambda k:min(l for l in cfg['layers'] if results[l]['dev'][k]>=max(results[q]['dev'][k] for q in cfg['layers'])-cfg['selection_tolerance'])
    layer=pick(0);null=[results[pick(k)]['heldout'][k] for k in range(1,cfg['null_draws']+1)]
    result={'layer':layer,'dev_auroc':results[layer]['dev'][0],'heldout_auroc':results[layer]['heldout'][0],'selected_null_heldout':null,
            'null_p95':float(np.percentile(null,95)),'null_p':float((1+sum(v>=results[layer]['heldout'][0] for v in null))/(len(null)+1)),
            'by_layer':results,'heldout_scores':scores[layer]}
    (d/'probe.json').write_text(json.dumps(frozen[layer]));(d/'fit.json').write_text(json.dumps(result));vol.commit();return result

@app.function(image=gpu_image,gpu='H200',cpu=(4,4),memory=(65536,65536),volumes={'/hf':hf,'/acts':vol},timeout=900,startup_timeout=300,retries=0,max_containers=2,scaledown_window=2)
def score(run,phase,shard,expected_probe_sha256):
    import numpy as np
    import torch
    items,cfg=inputs(run,'real_inputs');d=ACTS/run;P=json.loads((d/'probe.json').read_text());base=json.loads((d/'baseline.json').read_text())['probes']['per_token'];layers=sorted({P['layer'],base['layer']})
    probe_sha256=hashlib.sha256((d/'probe.json').read_bytes()).hexdigest()
    if not expected_probe_sha256 or probe_sha256!=expected_probe_sha256:raise ValueError('Frozen probe hash mismatch')
    provenance=bind_checkpoint(run,f'score_{phase}_{shard}',list((d/'scores'/phase/f'shard{shard}').glob('*.json')),
        probe_sha256=probe_sha256,baseline_sha256=hashlib.sha256((d/'baseline.json').read_bytes()).hexdigest())
    chosen=[r for r in items if (r['group']=='confirmation')==(phase=='confirmation')];chosen.sort(key=lambda r:(-len(r['ids']),r['key'],r['version']));mine=chosen[shard::2]
    dest=d/'scores'/phase/f'shard{shard}';dest.mkdir(parents=True,exist_ok=True);lm=model(cfg);P={k:torch.tensor(P[k],device='cuda',dtype=torch.float32) if k in ('w','mu','sd') else P[k] for k in P};base={k:torch.tensor(base[k],device='cuda',dtype=torch.float32) if k in ('w','mu','sd') else base[k] for k in base}
    class StopForward(Exception):pass
    fwd=0
    for count,it in enumerate(mine):
        name=hashlib.sha256((it['key']+'|'+it['version']).encode()).hexdigest()[:20];path=dest/(name+'.json')
        if path.exists():continue
        acts={}
        def hook(layer):
            def capture(module,args,output):
                acts[layer]=(output[0] if isinstance(output,tuple) else output)[0]
                if layer==layers[-1]:raise StopForward
            return capture
        handles=[lm.model.layers[l].register_forward_hook(hook(l)) for l in layers];ids=it['ids'];ids=ids+[cfg['pad_token_id']]*(-len(ids)%4096);t=time.monotonic()
        try:
            with torch.inference_mode():lm.model(input_ids=torch.tensor([ids],device='cuda'),use_cache=False)
        except StopForward:pass
        finally:
            for h in handles:h.remove()
        torch.cuda.synchronize();fwd+=time.monotonic()-t
        with torch.inference_mode():
            x=acts[P['layer']][[p['position'] for p in it['points']]].to(torch.float16).float();z=((x-P['mu'])/P['sd'])@P['w'];assert len(z)>0
            calls=[]
            for c in it['baseline_calls']:
                a,b=c['span'];x=acts[base['layer']][a:b].to(torch.float16).float();value=float((((x-base['mu'])/base['sd'])@base['w']).mean());calls.append({'step':c['step'],'action':c['action'],'score':value})
        result={k:it[k] for k in ['key','main_task','group','version','messages_sha256']}|{'start_score':float(z.mean()),'start_rows':[p|{'score':float(v)} for p,v in zip(it['points'],z.tolist())],
             'baseline_score':float(np.mean([c['score'] for c in calls])) if calls else None,'baseline_calls':calls,'n_tokens':len(it['ids']),'provenance':provenance}
        path.write_text(json.dumps(result));vol.commit();acts.clear()
        if count%10==0:print(f'{phase} shard{shard}: {count+1}/{len(mine)} versions; forward={fwd:.1f}s',flush=True)
    return {'phase':phase,'shard':shard,'versions':len(mine),'forward_s':fwd}

@app.local_entrypoint()
def main(run:str,step:str,probe_sha256:str=""):
    if step=='extract':result=extract.remote(run)
    elif step=='fit':result=fit.remote(run)
    elif step in ('calibration','confirmation'):result=list(score.starmap([(run,step,0,probe_sha256),(run,step,1,probe_sha256)]))
    else:raise ValueError(step)
    out=Path(__file__).parent/'results'/f'{run}_{step}.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result if step!='fit' else {k:result[k] for k in ['layer','dev_auroc','heldout_auroc','null_p95','null_p']},indent=2))
