"""Local v2 preparation, blind matching, and ID-only reattachment; no text logging."""
import argparse
import csv
import hashlib
import json
import os
import random
import uuid
import zipfile
from collections import Counter,defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from probe_shift.target_touch import label

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rows(path):
    with Path(path).open() as fh:
        return list(csv.DictReader(fh))

def write_json(path,obj):
    with Path(path).open('x') as fh:
        json.dump(obj,fh,indent=2);fh.write('\n')

def prepare(a):
    os.umask(0o077)
    a.cache.mkdir(parents=True,exist_ok=False)
    manifest=rows(a.manifest)
    assert len(manifest)==962 and len({r['run_id'] for r in manifest})==962
    source_index=json.loads(a.index.read_text())
    assert set(source_index)=={r['run_id'] for r in manifest}
    targets=defaultdict(list)
    for r in rows(a.targets):
        if r['type']!='dynamic':
            targets[r['side_task_id']].append({'type':r['type'],'aliases':json.loads(r['aliases'])})
    join,blind,provenance,coverage=[],[],[],Counter()
    seen=set()
    rng=random.Random(a.seed)
    rng.shuffle(manifest)
    for n,row in enumerate(manifest,1):
        sources=[Path(p) for p in source_index[row['run_id']]]
        hashes={sha(p) for p in sources}
        assert len(hashes)==1,'Divergent archive copies: resolve before using any labels'
        with zipfile.ZipFile(sources[0]) as archive:
            members=[m for m in archive.namelist() if m.startswith('samples/') and m.endswith('.json')]
            assert len(members)==1
            sample=json.loads(archive.read(members[0]))
        att=sample.get('attachments') or {}
        def resolve(x):
            if isinstance(x,str) and x.startswith('attachment://'):
                key=x.removeprefix('attachment://'); assert key in att
                return resolve(att[key])
            if isinstance(x,dict): return {k:resolve(v) for k,v in x.items()}
            if isinstance(x,list): return [resolve(v) for v in x]
            return x
        turn=0;calls=0
        for event in sample.get('events') or []:
            if event.get('event')!='model' or event.get('model')!=a.acting_model: continue
            irid=(((event.get('call') or {}).get('request') or {}).get('extra_headers') or {}).get('x-irid')
            assert irid is not None,'Missing acting-model x-irid'
            choices=(event.get('output') or {}).get('choices') or []
            assert len(choices)==1
            for ci,call in enumerate(choices[0]['message'].get('tool_calls') or []):
                key=(row['run_id'],int(sample.get('epoch',1)),turn,ci)
                assert key not in seen;seen.add(key)
                opaque=uuid.UUID(int=rng.getrandbits(128)).hex
                args=resolve(call.get('arguments') or {})
                assert isinstance(args,dict),'Arguments are not parsed JSON'
                name=call['function']
                offered={t.get('name') for t in event.get('tools') or []}
                excluded=int(name=='submit' or bool(offered and name not in offered) or (name=='text_editor' and args.get('command') is None))
                blind.append({'id':opaque,'tool':name,'args':args,'targets':targets[row['env']+'/'+row['side_task']]})
                join.append({'id':opaque,'run_id':key[0],'epoch':key[1],'turn':key[2],'call_index':key[3],'arm':row['arm'],'q2_excluded':excluded,'irid':irid,'sample_id':sample['id']})
                calls+=1
            turn+=1
        assert turn>0,'No acting model event'
        provenance.append({'run_id':row['run_id'],'archive_sha256':next(iter(hashes)),'events':turn,'calls':calls,'sample_error':int(bool(sample.get('error')))})
        coverage['runs']+=1;coverage['calls']+=calls;coverage['errored_runs']+=bool(sample.get('error'))
        if n%100==0: print(json.dumps({'prepared_runs':n}),flush=True)
    rng.shuffle(blind);rng.shuffle(join)
    order={r['id']:i for i,r in enumerate(blind)}
    for r in join: r['blind_order']=order[r['id']]
    assert len({(r['irid'],r['call_index']) for r in join})==len(join), 'Duplicate replay join key'
    with (a.cache/'blind_calls.jsonl').open('x') as fh:
        for item in blind: fh.write(json.dumps(item)+'\n')
    write_json(a.cache/'join.json',join)
    write_json(a.cache/'source_hashes.json',provenance)
    write_json(a.cache/'coverage.json',dict(coverage))
    print(json.dumps(dict(coverage)))

def match(a):
    a.out.mkdir(parents=True,exist_ok=False)
    counts=Counter()
    with (a.cache/'blind_calls.jsonl').open() as src,(a.out/'blind_labels.csv').open('x',newline='') as dst:
        w=csv.writer(dst,lineterminator='\n');w.writerow(['id','touch_any','touch_modify','unclear'])
        for line in src:
            call=json.loads(line)
            result=label(call['tool'],call['args'],call['targets'])
            w.writerow([call['id'],*result]);counts['calls']+=1
    write_json(a.out/'blind_freeze.json',{'sha256':sha(a.out/'blind_labels.csv'),'created_at_pt':datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),'calls':counts['calls']})
    print(json.dumps(counts))

def reattach(a):
    labels={r.pop('id'):r for r in rows(a.out/'blind_labels.csv')}
    frozen=json.loads((a.out/'blind_freeze.json').read_text())
    assert sha(a.out/'blind_labels.csv')==frozen['sha256']
    join=json.loads((a.cache/'join.json').read_text())
    assert set(labels)=={j['id'] for j in join} and len(labels)==len(join)
    counts=defaultdict(Counter); output=[];per_run=defaultdict(Counter)
    for item in join:
        y={k:int(v) for k,v in labels[item['id']].items()}
        assert set(y.values())<={0,1} and y['touch_modify']<=y['touch_any']
        output.append({k:item[k] for k in ('run_id','epoch','turn','call_index')}|y)
        counters={'calls':1,**y,'q2_excluded':item['q2_excluded'],'positive_q2_excluded':item['q2_excluded']*y['touch_any']}
        counts[item['arm']].update(counters);per_run[item['run_id']].update(counters)
    output.sort(key=lambda r:(r['run_id'],r['epoch'],r['turn'],r['call_index']))
    with (a.out/'target_labels_v2.csv').open('x',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(output[0]),lineterminator='\n');w.writeheader();w.writerows(output)
    write_json(a.out/'counts_v2.json',{'by_arm':counts,'per_run':per_run,'labels_sha256':sha(a.out/'target_labels_v2.csv')})
    write_json(a.out/'source_hashes_v2.json',json.loads((a.cache/'source_hashes.json').read_text()))
    write_json(a.out/'coverage_v2.json',json.loads((a.cache/'coverage.json').read_text()))
    mapping=[]
    for item in join:
        mapping.append({'run_id':item['run_id'],'inspect_epoch':item['epoch'],
                        **{k:item[k] for k in ('turn','call_index','irid','sample_id','q2_excluded','blind_order')},
                        'audit_opaque_id':item['id']})
    mapping.sort(key=lambda r:(r['run_id'],r['inspect_epoch'],r['turn'],r['call_index']))
    with (a.out/'replay_join_v2.csv').open('x',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(mapping[0]),lineterminator='\n');w.writeheader();w.writerows(mapping)
    write_json(a.out/'summary_v2.json',{'by_arm':counts,'labels_sha256':sha(a.out/'target_labels_v2.csv'),
                                      'join_sha256':sha(a.out/'replay_join_v2.csv')})
    print(json.dumps({'by_arm':counts},indent=2))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['prepare','match','reattach'])
    p.add_argument('--manifest',type=Path)
    p.add_argument('--targets',type=Path)
    p.add_argument('--index',type=Path)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path)
    p.add_argument('--acting-model',default='vllm/Qwen/Qwen3.5-27B')
    p.add_argument('--seed',type=int,default=20261011)
    p.add_argument('--freeze',type=Path,required=True)
    a=p.parse_args()
    freeze=json.loads(a.freeze.read_text())
    for name,expected in freeze['files'].items():
        assert sha(Path(__file__).parent/name)==expected, 'Producer differs from pre-scan freeze'
    if a.stage=='prepare':
        assert sha(a.manifest)==freeze['runs_sha256'] and sha(a.targets)==freeze['targets_sha256']
    globals()[a.stage](a)

if __name__=='__main__': main()
