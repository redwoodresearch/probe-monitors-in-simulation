"""Rebuild numeric validation/audit reports. Raw calls can only be exported privately."""
import argparse
import csv
import hashlib
import html
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open() as fh: return list(csv.DictReader(fh))


def key(row):
    return row['run_id'], int(row.get('epoch', row.get('inspect_epoch'))), int(row['turn']), int(row['call_index'])


def joined(labels_path, mapping_path, runs_path):
    labels, mapping, runs = rows(labels_path), rows(mapping_path), rows(runs_path)
    assert list(labels[0]) == ['run_id','epoch','turn','call_index','touch_any','touch_modify','unclear']
    assert len({key(r) for r in labels}) == len(labels)
    lookup = {key(r): r for r in mapping}
    assert len(lookup) == len(mapping) == len(labels) and set(lookup) == {key(r) for r in labels}
    assert len({(r['irid'],r['call_index']) for r in mapping}) == len(mapping)
    arms = {r['run_id']: r['arm'] for r in runs}
    assert len(arms) == len(runs) and set(arms) == {r['run_id'] for r in labels}
    result = []
    for row in labels:
        y = {k:int(row[k]) for k in ('touch_any','touch_modify','unclear')}
        assert set(y.values()) <= {0,1} and y['touch_modify'] <= y['touch_any']
        result.append({**lookup[key(row)], **y, 'arm':arms[row['run_id']]})
    assert len({r['audit_opaque_id'] for r in result}) == len(result)
    assert {int(r['blind_order']) for r in result} == set(range(len(result)))
    return sorted(result, key=lambda r:int(r['blind_order']))


def validate(data, old):
    counts, changes = defaultdict(Counter), defaultdict(Counter)
    previous = {key(r):r for r in rows(old)} if old else None
    if previous is not None: assert set(previous) == {key(r) for r in data}
    for r in data:
        counts[r['arm']].update({'calls':1, **{k:r[k] for k in ('touch_any','touch_modify','unclear')}})
        if previous is not None:
            for name in ('touch_any','touch_modify','unclear'):
                before, after = int(previous[key(r)][name]), r[name]
                changes[r['arm']][name+'_added'] += before == 0 and after == 1
                changes[r['arm']][name+'_removed'] += before == 1 and after == 0
    return {'rows':len(data),'runs':len({r['run_id'] for r in data}),
            'unique_replay_keys':len(data),'exact_id_coverage':True,'binary_labels':True,
            'by_arm':counts,'v1_to_v2_changes':changes,'v2_blind_audit_complete':False}


def audit(data, decisions):
    lookup = {r['audit_opaque_id']:r for r in data}
    assert len({d['id'] for d in decisions}) == len(decisions)
    strata, arms = defaultdict(Counter), defaultdict(Counter)
    for d in decisions:
        r = lookup[d['id']]
        any_y, mod_y = int(d['expected_any']), int(d['expected_modify'])
        assert any_y in (0,1) and mod_y in (0,1) and mod_y <= any_y
        assert int(d['predicted_any']) == r['touch_any'] and int(d['predicted_modify']) == r['touch_modify']
        e, m = int(any_y != r['touch_any']), int(mod_y != r['touch_modify'])
        assert int(d['any_right']) == 1-e and int(d['modify_right']) == 1-m
        stats = {'reviewed':1,'any_errors':e,'modify_errors':m}
        strata[str(r['touch_any'])].update(stats)
        arms[r['arm']].update({**stats,'positive_reviewed':r['touch_any'],
                              'negative_reviewed':1-r['touch_any'],
                              'false_positives':int(r['touch_any']==1 and any_y==0),
                              'false_negatives':int(r['touch_any']==0 and any_y==1)})
    return {'by_stratum':strata,'by_arm':arms,
            'revision_triggered':any(x['any_errors']/x['reviewed'] > .1 for x in strata.values())}


def select(data, seed):
    rng, selected = random.Random(seed), []
    for y in (1,0):
        group = [r for r in data if r['touch_any']==y]
        selected.extend(rng.sample(group, min(50,len(group))))
    rng.shuffle(selected)
    return [{'id':r['audit_opaque_id'], **{k:r[k] for k in ('touch_any','touch_modify','unclear')}} for r in selected]


def export_private(selection, calls_path, page):
    # No stdout display mode. A human opens this private file outside agent chats.
    if '.run_cache' not in page.resolve().parts:
        raise AssertionError('Private audit pages must stay in .run_cache')
    wanted = {r['id']:r for r in selection}; calls = {}
    with calls_path.open() as fh:
        for line in fh:
            call=json.loads(line)
            if call['id'] in wanted: calls[call['id']]=call
    assert set(calls)==set(wanted)
    page.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(page,os.O_WRONLY | os.O_CREAT | os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as fh:
        fh.write('<!doctype html><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src \'none\'"><title>Private blind audit</title><h1>Private blind audit</h1><p>Human review only. Do not upload or paste this page into any agent session. Record decisions by opaque ID in a separate numeric CSV.</p>')
        for r in selection:
            call = calls[r['id']]
            assert set(call)=={'id','tool','args','targets'}
            fh.write('<article><h2>'+r['id']+'</h2><pre>'+html.escape(json.dumps(r))+'</pre><pre>'+html.escape(json.dumps(call,indent=2))+'</pre></article>')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['validate','audit','select','export'])
    for name in ('labels','mapping','runs','out'): p.add_argument('--'+name,type=Path,required=True)
    for name in ('old-labels','decisions','calls'): p.add_argument('--'+name,type=Path)
    p.add_argument('--seed',type=int,default=941318)
    a=p.parse_args(); data=joined(a.labels,a.mapping,a.runs)
    if a.mode=='export':
        export_private(select(data,a.seed),a.calls,a.out)
        print(json.dumps({'exported_calls':min(50,sum(r['touch_any'] for r in data))+min(50,sum(1-r['touch_any'] for r in data))}))
        return
    if a.mode=='validate': result=validate(data,a.old_labels)
    elif a.mode=='audit': result=audit(data,rows(a.decisions))
    else: result={'seed':a.seed,'selected':select(data,a.seed),'decisions_complete':False}
    result['inputs_sha256']={n:sha(getattr(a,n)) for n in ('labels','mapping','runs')}
    if a.decisions: result['inputs_sha256']['decisions']=sha(a.decisions)
    if a.old_labels: result['inputs_sha256']['old_labels']=sha(a.old_labels)
    result['producer_sha256']=sha(__file__)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as fh: json.dump(result,fh,indent=2);fh.write('\n')
    print(json.dumps({'report_written':True,'rows':len(data)}))


if __name__=='__main__': main()
