"""Prepare and export panel labels for an exact item-level calibration manifest."""
import argparse
import csv
import json
import os
import random
import uuid
import zipfile
from collections import Counter
from pathlib import Path
from probe_shift.credential_stage1 import checked_reader_messages, sha
from probe_shift.transcripts import reader_messages
from probe_shift.label_panel_results import aggregate, apply_adjudication, write_csv


def key(row):
    return row['run_id'], int(row['step']), int(row['action'])


def prepare(a):
    os.umask(0o077)
    assert sha(a.items) == a.items_sha256
    with a.items.open() as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ['run_id','step','action']
    wanted={key(r) for r in rows};runs=sorted({r['run_id'] for r in rows})
    assert len(wanted)==len(rows)==1788 and len(runs)==59
    sources=json.loads(a.index.read_text())
    outputs=[];provenance=[]
    for run in runs:
        paths=[Path(p) for p in sources[run] if Path(p).is_file()]
        assert paths
        path=paths[0];_,filename,sid,epoch=run.split('|')
        assert path.name==filename
        with zipfile.ZipFile(path) as archive:
            sample=json.loads(archive.read(f'samples/{sid}_epoch_{epoch}.json'))
        assert str(sample['id'])==sid and int(sample['epoch'])==int(epoch) and not sample.get('error')
        messages,mode=checked_reader_messages(sample)
        original=[m for m in reader_messages(sample,'system') if m['role'] in ('assistant','tool')]
        assert original==messages, 'Original screening and panel reader outputs differ; stop for an explicit audit'
        step=-1;action=0;seen=[]
        for m in messages:
            if m['role']=='assistant':step+=1;action=0
            elif m['role']=='tool':
                item=(run,step,action);action+=1
                if item in wanted:
                    assert 'attachment://' not in m['content']
                    outputs.append({'run_id':run,'step':step,'action':item[2],'text':m['content']});seen.append(item)
        assert set(seen)=={k for k in wanted if k[0]==run}
        provenance.append({'run_id':run,'archive_sha256':sha(path),'source_mode':mode,'listed_outputs':len(seen)})
    assert {key(r) for r in outputs}==wanted and len(outputs)==len(wanted)
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'inputs').mkdir()
    rng=random.Random(a.seed);rng.shuffle(outputs)
    packets=[];join=[];batch=[];nbytes=0
    def flush():
        nonlocal batch,nbytes
        if not batch:return
        pid=uuid.UUID(int=rng.getrandbits(128)).hex
        payload={'outputs':[]}
        for i,r in enumerate(batch):
            cid=f'o{i:03d}';payload['outputs'].append({'id':cid,'text':r['text']})
            join.append({k:r[k] for k in ('run_id','step','action')}|{'packet':pid,'id':cid})
        path=a.out/'inputs'/f'{pid}.json';path.write_text(json.dumps(payload,ensure_ascii=False))
        packets.append({'packet':pid,'ids':[r['id'] for r in payload['outputs']],'sha256':sha(path),'bytes':path.stat().st_size})
        batch=[];nbytes=0
    for row in outputs:
        size=len(row['text'].encode())
        if batch and (len(batch)>=20 or nbytes+size>120000):flush()
        batch.append(row);nbytes+=size
    flush()
    summary={'items_sha256':sha(a.items),'source_index_sha256':sha(a.index),'outputs':len(outputs),'runs':len(runs),
             'packets':len(packets),'input_bytes':sum(p['bytes'] for p in packets),'largest_input_bytes':max(p['bytes'] for p in packets),
             'original_reader_outputs_match':True,'source_modes':dict(Counter(r['source_mode'] for r in provenance)),'seed':a.seed}
    for name,data in [('packets.json',packets),('join.json',join),('provenance.json',provenance),('preparation.json',summary)]:
        (a.out/name).write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(summary))


def export(a):
    preparation = json.loads((a.prepared / 'preparation.json').read_text())
    assert sha(a.items) == preparation['items_sha256'], 'Calibration manifest changed after preparation'
    models=json.loads(a.config.read_text())['models']
    blind,failures,valid,expected=aggregate(a.prepared,a.ledger,'credential_calibration',models)
    a.out.mkdir(parents=True,exist_ok=False)
    freeze=a.out/'blind_consensus.json';freeze.write_text(json.dumps(blind,indent=2)+'\n')
    summary={'valid_judge_packets':valid,'expected_judge_packets':expected,'failed_attempts':failures,
             'blind_consensus_sha256':sha(freeze),'ledger_sha256':sha(a.ledger),'items_sha256':sha(a.items)}
    if a.adjudication:
        decisions=json.loads(a.adjudication.read_text())
        assert decisions['primary_consensus_sha256']==sha(freeze) and decisions['ledger_sha256']==sha(a.ledger)
        blind=apply_adjudication(blind,decisions['decisions'])
        summary.update(adjudicated_outputs=len(decisions['decisions']),adjudication_sha256=sha(a.adjudication))
    labels={(r['packet'],r['id']):r['label'] for r in blind}
    join=json.loads((a.prepared/'join.json').read_text());mapping={key(r):labels[r['packet'],r['id']] for r in join}
    with a.items.open() as f:
        original=list(csv.DictReader(f))
    assert len(original)==len(mapping)==1788 and {key(r) for r in original}==set(mapping)
    unresolved=[{'packet':r['packet'],'id':r['id']} for r in blind if r['label'] is None]
    (a.out/'unresolved_ids.json').write_text(json.dumps(unresolved,indent=2)+'\n')
    summary['unresolved_items']=len(unresolved)
    if not unresolved:
        final=[r|{'credential':mapping[key(r)],'category':'','placeholder_flag':''} for r in original]
        assert all(type(r['credential']) is int and r['credential'] in (0,1) for r in final)
        path=a.out/'screening_labels_calibration_panel.csv'
        write_csv(path,final,['run_id','step','action','credential','category','placeholder_flag'])
        summary.update(rows=len(final),complete_runs=len({r['run_id'] for r in final}),positives=sum(r['credential'] for r in final),labels_sha256=sha(path))
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='failed_attempts'}))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--items',type=Path,required=True);prep.add_argument('--items-sha256',required=True);prep.add_argument('--index',type=Path,required=True);prep.add_argument('--out',type=Path,required=True);prep.add_argument('--seed',type=int,default=20261011)
    ex=sub.add_parser('export');ex.add_argument('--items',type=Path,required=True);ex.add_argument('--prepared',type=Path,required=True);ex.add_argument('--ledger',type=Path,required=True);ex.add_argument('--config',type=Path,required=True);ex.add_argument('--out',type=Path,required=True);ex.add_argument('--adjudication',type=Path)
    a=p.parse_args();prepare(a) if a.command=='prepare' else export(a)


if __name__=='__main__':main()
