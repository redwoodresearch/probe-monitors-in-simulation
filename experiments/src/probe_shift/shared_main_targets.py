"""Add a frozen shared-main-only flag locally; publish IDs, hashes and numbers only."""
import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from probe_shift.target_touch import mentions, strings

CODE_SUFFIXES = {'.py','.sh','.bash','.js','.jsx','.ts','.tsx','.sql','.r','.go','.rs',
                 '.c','.h','.cc','.cpp','.java','.rb','.pl','.php','.lua','.patch'}
LABELS = ['touch_any','touch_modify','unclear']
KEYS = ['run_id','epoch','turn','call_index']
FLAG = 'shared_with_main_only'


def require(condition, message):
    if not condition: raise ValueError(message)


def digest(data): return hashlib.sha256(data).hexdigest()
def sha(path): return digest(Path(path).read_bytes())
def read(path): return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as fh: return list(csv.DictReader(fh))


def write_json(path, data):
    with path.open('x') as fh: json.dump(data,fh,indent=2);fh.write('\n')


def write_csv(path, data, columns):
    with path.open('x',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=columns,lineterminator='\n');w.writeheader();w.writerows(data)


def code_blobs(repo, commit):
    """Read pinned blobs without checking out or executing repository code."""
    tree=subprocess.check_output(['git','-C',str(repo),'ls-tree','-r','-z',commit,'main_tasks/'])
    selected, excluded, main_tasks = [], Counter(), set()
    for entry in tree.split(b'\0'):
        if not entry: continue
        meta,name=entry.split(b'\t',1);mode,kind,oid=meta.decode().split();name=name.decode()
        main_tasks.add(name.split('/')[1]); suffix=Path(name).suffix.lower()
        if suffix not in CODE_SUFFIXES: excluded[suffix]+=1;continue
        require(kind=='blob' and mode in ('100644','100755'),'Nonregular main-code input')
        selected.append((name,oid))
    require(selected,'No main-task code found')
    output=subprocess.check_output(['git','-C',str(repo),'cat-file','--batch'],
                                   input=('\n'.join(oid for _,oid in selected)+'\n').encode())
    stream=io.BytesIO(output); blobs=[]
    for name,oid in selected:
        header=stream.readline().decode().split()
        require(len(header)==3 and header[:2]==[oid,'blob'],'Unexpected main-code blob')
        raw=stream.read(int(header[2]));require(stream.read(1)==b'\n','Truncated main-code blob')
        blobs.append((name,oid,raw.decode('utf-8'),digest(raw)))
    require(stream.read()==b'','Unexpected extra blob data')
    return blobs,dict(excluded),main_tasks


def shared_only(args, targets, shared):
    require(len(targets)==len(shared),'Incomplete resource flags')
    texts=list(strings(args))
    matched=[i for i,t in enumerate(targets) if any(mentions(text,[t]) for text in texts)]
    return int(bool(matched) and all(shared[i] for i in matched)),int(bool(matched))


def primary_rows(data):
    """Exclude shared-only rows before every main-analysis numerator/denominator."""
    for row in data:
        require(FLAG in row,'Main analysis requires the shared-main-only column')
        flag=int(row[FLAG]);positive=int(row['touch_any'])
        require(flag in (0,1) and positive in (0,1) and flag<=positive,'Invalid shared-only flag')
        if flag==0: yield row


def inventory(a):
    targets,runs,sources=rows(a.targets),rows(a.runs),read(a.target_sources)
    by_side=defaultdict(list)
    for t in targets: by_side[t['side_task_id']].append(t)
    environments={r['env'] for r in runs};scope_map={};public=[];provenance=[]
    for env in sorted(environments):
        commits={v['git_commit'] for k,v in sources.items() if k.split('/')[0]==env}
        require(len(commits)==1,'Environment has inconsistent source commits');commit=next(iter(commits))
        blobs,excluded,present=code_blobs(a.env_root/env,commit)
        require({r['main_task'] for r in runs if r['env']==env}<=present,'Requested main task absent')
        provenance.append({'environment_id':env,'git_commit':commit,'main_tasks':len(present),
                           'excluded_extensions':excluded,'files':[{'path_sha256':digest(n.encode()),
                            'git_blob':oid,'sha256':h} for n,oid,_,h in blobs]})
        for side,items in sorted(by_side.items()):
            if side.split('/')[0]!=env: continue
            scope=[]
            for t in items:
                require(t['source_commit']==commit,'Target/main source revision mismatch')
                target={'type':t['type'],'aliases':json.loads(t['aliases'])}
                witness=next(((n,h) for n,_,text,h in blobs if mentions(text,[target])),None)
                shared=int(witness is not None)
                public.append({'side_task_id':side,'type':t['type'],'target_sha256':digest(t['target'].encode()),
                               'shared_with_main':shared,'witness_path_sha256':digest(witness[0].encode()) if witness else '',
                               'witness_sha256':witness[1] if witness else ''})
                if t['type']!='dynamic': scope.append({'target':target,'shared':shared})
            scope_map[digest(side.encode())]=scope
    run_side={r['run_id']:digest((r['env']+'/'+r['side_task']).encode()) for r in runs}
    join=read(a.join);require(len({j['id'] for j in join})==len(join),'Duplicate opaque call ID')
    call_scopes={j['id']:run_side[j['run_id']] for j in join}
    require(set(call_scopes.values())<=set(scope_map),'Missing side-task classification')
    a.private.mkdir(parents=True,exist_ok=False);a.out.mkdir(parents=True,exist_ok=False)
    write_json(a.private/'blind_scopes.json',{'scopes':scope_map,'call_scopes':call_scopes})
    write_csv(a.out/'shared_target_hashes_v1.csv',public,list(public[0]))
    write_json(a.out/'main_source_hashes_v1.json',provenance)
    write_json(a.out/'shared_inventory_freeze_v1.json',{
        'before_command_classification':True,'code_freeze_sha256':sha(a.freeze),
        'shared_target_hashes_sha256':sha(a.out/'shared_target_hashes_v1.csv'),
        'main_sources_sha256':sha(a.out/'main_source_hashes_v1.json'),
        'private_blind_scopes_sha256':sha(a.private/'blind_scopes.json')})
    print(json.dumps({'inventory_frozen':True}))


def match(a):
    frozen=read(a.out/'shared_inventory_freeze_v1.json')
    require(frozen['code_freeze_sha256']==sha(a.freeze),'Changed code freeze')
    require(frozen['private_blind_scopes_sha256']==sha(a.private/'blind_scopes.json'),'Changed resource map')
    require(frozen['shared_target_hashes_sha256']==sha(a.out/'shared_target_hashes_v1.csv'),'Changed inventory')
    require(frozen['main_sources_sha256']==sha(a.out/'main_source_hashes_v1.json'),'Changed main sources')
    scopes=read(a.private/'blind_scopes.json');old_rows=rows(a.v2_blind);old={r['id']:r for r in old_rows}
    require(len(old)==len(old_rows)==len(scopes['call_scopes']),'Blind input cardinality mismatch')
    seen=set()
    with a.calls.open() as src,(a.private/'blind_flags_v1.csv').open('x',newline='') as dst:
        writer=csv.writer(dst,lineterminator='\n');writer.writerow(['id',FLAG])
        for line in src:
            call=json.loads(line);cid=call['id'];require(cid not in seen,'Duplicate blind call');seen.add(cid)
            scope=scopes['scopes'][scopes['call_scopes'][cid]]
            require(call['targets']==[r['target'] for r in scope],'Call target inventory mismatch')
            flag,positive=shared_only(call['args'],call['targets'],[r['shared'] for r in scope])
            require(positive==int(old[cid]['touch_any']),'v2 primary label changed')
            writer.writerow([cid,flag])
    require(seen==set(old)==set(scopes['call_scopes']),'Missing blind calls')
    write_json(a.out/'shared_blind_freeze_v1.json',{'rows':len(seen),'code_freeze_sha256':sha(a.freeze),
               'flags_sha256':sha(a.private/'blind_flags_v1.csv'),'inventory_freeze_sha256':sha(a.out/'shared_inventory_freeze_v1.json')})
    print(json.dumps({'frozen_calls':len(seen)}))


def attach(a):
    frozen=read(a.out/'shared_blind_freeze_v1.json')
    require(frozen['code_freeze_sha256']==sha(a.freeze),'Changed code freeze')
    require(frozen['flags_sha256']==sha(a.private/'blind_flags_v1.csv'),'Changed blind flags')
    require(frozen['inventory_freeze_sha256']==sha(a.out/'shared_inventory_freeze_v1.json'),'Changed inventory freeze')
    flag_rows=rows(a.private/'blind_flags_v1.csv');flags={r['id']:int(r[FLAG]) for r in flag_rows}
    join=read(a.join);lookup={tuple(str(r[k]) for k in KEYS):r for r in join}
    require(len(flags)==len(flag_rows)==len(lookup)==len(join)==frozen['rows'],'Join cardinality mismatch')
    require(set(flags)=={j['id'] for j in join},'Incomplete flags')
    data=rows(a.v2_labels);require(list(data[0])==KEYS+LABELS,'Unexpected v2 schema')
    require(len(data)==len(lookup),'v2 label cardinality mismatch');seen=set();output=[]
    counts=defaultdict(Counter);per_run=defaultdict(Counter)
    for row in data:
        key=tuple(row[k] for k in KEYS);require(key not in seen,'Duplicate label ID');seen.add(key)
        j=lookup[key];flag=flags[j['id']];new=row|{FLAG:flag};output.append(new)
        kept=list(primary_rows([new]));positive=int(row['touch_any'])
        stats={'all_calls':1,'unfiltered_positives':positive,'excluded_shared_only':flag,
               'main_calls':len(kept),'main_positives':positive if kept else 0,
               'main_q2_excluded':int(j['q2_excluded']) if kept else 0,
               'main_positive_q2_excluded':int(j['q2_excluded'])*positive if kept else 0}
        counts[j['arm']].update(stats);per_run[j['run_id']].update(stats)
    require(seen==set(lookup),'Unmatched v2 labels')
    write_csv(a.out/'target_labels_v3.csv',output,KEYS+LABELS+[FLAG])
    write_json(a.out/'main_counts_by_run_v3.json',dict(per_run))
    write_json(a.out/'shared_main_summary_v1.json',{'by_arm':dict(counts),'rows':len(output),
               'runs':len(per_run),'empty_retained_runs':sum(x['main_calls']==0 for x in per_run.values()),
               'v2_labels_unchanged':True,'labels_sha256':sha(a.out/'target_labels_v3.csv'),
               'per_run_sha256':sha(a.out/'main_counts_by_run_v3.json'),'code_freeze_sha256':sha(a.freeze),
               'main_filter':FLAG+' == 0','fresh_blind_audit_complete':False})
    print(json.dumps({'by_arm':dict(counts)},indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['inventory','match','attach'])
    for name in ('freeze','private','out'): p.add_argument('--'+name,type=Path,required=True)
    for name in ('targets','target-sources','runs','join','env-root','calls','v2-blind','v2-labels'):
        p.add_argument('--'+name,type=Path)
    a=p.parse_args();os.umask(0o077);freeze=read(a.freeze)
    required={'inventory':('targets','target_sources','runs','join','env_root'),
              'match':('calls','v2_blind'),'attach':('join','v2_labels')}[a.stage]
    require(all(getattr(a,k) is not None for k in required),'Missing stage input')
    require(sha(a.freeze.parent/freeze['rule_file'])==freeze['rule_sha256'],'Changed rule')
    for name,h in freeze['producer_files'].items(): require(sha(Path(__file__).parent/name)==h,'Changed producer')
    for name,h in freeze['inputs'].items():
        value=getattr(a,name)
        if value is not None: require(sha(value)==h,'Changed frozen input: '+name)
    globals()[a.stage](a)


if __name__=='__main__': main()
