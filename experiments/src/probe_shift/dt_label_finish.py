"""Prepare independent extra reviews and export only complete, ID-only labels."""
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from probe_shift.dt_label_panel import AXES, digest, jobs_for, save
from probe_shift.dt_label_prepare import sha


def votes(config, records):
    prepared=Path(config['prepared'])
    packets=json.loads((prepared/'packets.json').read_text())
    joins=json.loads((prepared/'join.json').read_text())
    mapping={(r['packet'],r['id']):r['blind_id'] for r in joins}
    wanted={(p['packet'],cid) for p in packets for cid in p['ids']}
    if set(mapping)!=wanted or len(mapping)!=len(joins) or len(set(mapping.values()))!=len(mapping):
        raise ValueError('Invalid or non-unique private blind-ID join')
    expected={j['request_id'] for j in jobs_for(config,prepared,
        json.loads(Path(config['catalog']).read_text()),False)}
    result={sid:{} for sid in mapping.values()}
    for r in records:
        if r['request_id'] not in expected and r.get('retry_of') not in expected:
            continue
        if r['status']!='ok':continue
        for label in r['labels']:
            sid=mapping[(r['packet'],label['id'])]
            if r['model'] in result[sid]:raise ValueError('Multiple successful votes for one configured judge')
            result[sid][r['model']]=label
    return result


def winner(labels,axis,minimum):
    counts=Counter(r[axis] for r in labels if r[axis+'_confidence']=='sure')
    if not counts:return None
    value,n=counts.most_common(1)[0]
    return value if n>=minimum and n>counts[1-value] else None


def combine(primary,review=None,final=None):
    results={}
    for sid,labels in primary.items():
        row={'blind_id':sid,'primary_valid_judges':len(labels)}
        for axis in AXES:
            value=winner(labels.values(),axis,4);method='primary'
            if value is None and review is not None:
                value=winner(review.get(sid,{}).values(),axis,2);method='two_high_reviews'
            if value is None and final is not None:
                extra=list((review or {}).get(sid,{}).values())+list(final.get(sid,{}).values())
                value=winner(extra,axis,2);method='high_review_majority'
            row[axis]=value
            row[axis+'_method']=method if value is not None else 'unresolved'
        results[sid]=row
    return results


def prepare_review(base,results,out,kind,*,prefix=None,models=None,reasoning=None,max_tokens=None,batch_size=None):
    chosen={sid for sid,row in results.items() if any(row[a] is None for a in AXES)}
    if not chosen:raise ValueError('No unresolved calls need review')
    prepared=Path(base['prepared'])
    joins=json.loads((prepared/'join.json').read_text())
    mapping={(r['packet'],r['id']):r['blind_id'] for r in joins}
    commands=[]
    for p in json.loads((prepared/'packets.json').read_text()):
        if not any(mapping[(p['packet'],cid)] in chosen for cid in p['ids']):continue
        packet=json.loads((prepared/'inputs'/(p['packet']+'.json')).read_text())
        for row in packet['calls']:
            sid=mapping[(p['packet'],row['id'])]
            if sid in chosen:
                commands.append((sid,row,packet['tool_definitions'][row['definition']]))
    if len(commands)!=len(chosen):raise ValueError('Review evidence coverage mismatch')
    random.SystemRandom().shuffle(commands)
    out.mkdir(parents=True,exist_ok=False)
    packets,join=[],[]
    prefix=prefix or {'adjudication':'a','final_review':'f','audit':'u'}[kind]
    batch_size=batch_size or (5 if kind=='final_review' else 10)
    # Small packets give high-reasoning reviewers room for careful individual judgments.
    for start in range(0,len(commands),batch_size):
        pid=f'{prefix}{len(packets):05d}'; payload={'packet_id':pid,'calls':[],'tool_definitions':{}}
        for i,(sid,row,definition) in enumerate(commands[start:start+batch_size]):
            cid=f'c{i:03d}'; payload['calls'].append(row|{'id':cid})
            payload['tool_definitions'][row['definition']]=definition
            join.append({'packet':pid,'id':cid,'blind_id':sid})
        path=out/'inputs'/(pid+'.json');save(path,payload)
        packets.append({'packet':pid,'ids':[r['id'] for r in payload['calls']],
                        'bytes':path.stat().st_size,'sha256':sha(path)})
    save(out/'join.json',join);save(out/'packets.json',packets)
    metadata=json.loads((prepared/'preparation.json').read_text())
    metadata.update(rows=len(commands),packets=len(packets),review_stage=kind,
                    previous_votes_shown=False,source_packets_sha256=sha(prepared/'packets.json'),
                    per_tool=dict(Counter(row['tool'] for _,row,_ in commands)),
                    packet_bytes=sum(p['bytes'] for p in packets),batch_size=batch_size,
                    script_sha256=sha(__file__))
    save(out/'preparation.json',metadata)
    cfg=dict(base)
    models=models or (base['adjudication_reviewers'] if kind in ('adjudication','audit') else [base['final_adjudicator']])
    cfg.update(prepared=str(out.resolve()),models=models,reasoning_effort=reasoning or {m:'high' for m in models})
    if max_tokens:cfg['max_tokens']=max_tokens
    elif kind=='final_review':cfg['max_tokens']=16384
    save(out/'config.json',cfg)
    return {'rows':len(commands),'packets':len(packets),'requests':len(packets)*len(models)}


def repaired_primary(config, records, repair_configs, source_config_sha256=None):
    primary=votes(config,records)
    for path in repair_configs:
        repair=json.loads(Path(path).read_text())
        if repair.get('repair_of_config_sha256') != source_config_sha256 or repair['prompt']!=config['prompt']:
            raise ValueError('Repair must reference the same rubric and original configuration')
        for sid,extra in votes(repair,records).items():
            if sid not in primary or set(primary[sid])&set(extra):
                raise ValueError('Repair duplicates an already successful judge or invents an ID')
            primary[sid].update(extra)
    return primary


def attempts_complete(config, records):
    expected={j['request_id'] for j in jobs_for(config,Path(config['prepared']),
        json.loads(Path(config['catalog']).read_text()),False)}
    finished={r['request_id'] for r in records if r['status']!='inflight'}
    return expected<=finished


def repair_record(request_id, records):
    """Select the latest failed attempt, without repairing an already recovered vote."""
    original = records.get(request_id)
    retry = records.get(digest([request_id, 'retry-1']))
    if original is None or any(r and r['status'] == 'inflight' for r in (original, retry)):
        raise ValueError('Requests are not all complete')
    if retry and retry.get('retry_of') != request_id:
        raise ValueError('Retry provenance mismatch')
    attempts = [r for r in (original, retry) if r]
    if any(r.get('refused') for r in attempts):
        raise ValueError('Refusal cannot be retried')
    successes = sum(r['status'] == 'ok' for r in attempts)
    if successes > 1:
        raise ValueError('Multiple successful votes for one request')
    return None if successes else retry or original


def export(config,results,out,rules):
    if any(row[a] is None for row in results.values() for a in AXES):
        raise ValueError('Unresolved labels remain; refusing a completed sheet')
    sheet=Path(config['sheet'])
    if sha(sheet)!=config['blind_sheet_sha256']:raise ValueError('Source sheet changed')
    with sheet.open() as f:
        r=csv.DictReader(f);fields=r.fieldnames;rows=list(r)
    if len(rows)!=len(results) or {r['blind_id'] for r in rows}!=set(results):
        raise ValueError('Final label coverage mismatch')
    out.mkdir(parents=True,exist_ok=False)
    output=out/(sheet.stem+'_labelled.csv')
    with output.open('x') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow(r|{a:results[r['blind_id']][a] for a in AXES})
    with output.open() as f:reread=list(csv.DictReader(f))
    if len(reread)!=len(rows):raise ValueError('Export lost rows')
    for before,after in zip(rows,reread):
        if any(before[k]!=after[k] for k in fields if k not in AXES):
            raise ValueError('Export changed non-label data')
        if any(after[a] not in ('0','1') for a in AXES):raise ValueError('Export contains invalid labels')
    if rules:
        with rules.open() as f:frozen=list(csv.DictReader(f))
        lookup={r['blind_id']:r for r in frozen}
        if len(lookup)!=len(frozen) or set(lookup)!=set(results):raise ValueError('Frozen-rule ID mismatch')
        with (out/'frozen_q2_comparison.csv').open('x') as f:
            w=csv.DictWriter(f,fieldnames=['blind_id','rule_destructive','rule_transfer']);w.writeheader()
            for r in rows:w.writerow(lookup[r['blind_id']])
    save(out/'label_audit.json',list(results.values()))
    report={'rows':len(rows),'source_sheet_sha256':sha(sheet),'output_sha256':sha(output),
            'primary_valid_judges':dict(Counter(r['primary_valid_judges'] for r in results.values())),
            'counts':{a:dict(Counter(r[a] for r in results.values())) for a in AXES},
            'methods':{a:dict(Counter(r[a+'_method'] for r in results.values())) for a in AXES},
            'counts_by_existing_label':{g:{a:sum(results[r['blind_id']][a] for r in rows if r['existing_label']==g)
                for a in AXES} for g in sorted({r['existing_label'] for r in rows})},
            'world_identity_key_read':False,'monitor_fitting_started':False}
    save(out/'manifest.json',report)
    return report


def export_supplement(config, results, out, *, audit_caveat=None, decision_rule_sha256=None):
    """Preserve accepted semantic votes and explicit blanks without claiming completion."""
    sheet = Path(config['sheet'])
    if sha(sheet) != config['blind_sheet_sha256']:
        raise ValueError('Source sheet changed')
    with sheet.open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(results) or {r['blind_id'] for r in rows} != set(results):
        raise ValueError('Supplement coverage mismatch')
    if any(r[a] not in (0, 1, None) for r in results.values() for a in AXES):
        raise ValueError('Invalid semantic labels')
    out.mkdir(parents=True, exist_ok=False)
    path = out / 'semantic_labels_provisional.csv'
    names = {'destructive': 'destructive_broad', 'transfer': 'transfer_semantic'}
    fields = ['blind_id', *names.values(), 'destructive_unresolved', 'transfer_unresolved']
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for source in rows:
            row = results[source['blind_id']]
            writer.writerow({'blind_id': row['blind_id'],
                **{names[a]: row[a] for a in AXES},
                **{a + '_unresolved': int(row[a] is None) for a in AXES}})
    with path.open(newline='') as f:
        reread = list(csv.DictReader(f))
    if [r['blind_id'] for r in reread] != [r['blind_id'] for r in rows]:
        raise ValueError('Supplement changed row order')
    for row in reread:
        for axis, name in names.items():
            expected = results[row['blind_id']][axis]
            if row[name] != ('' if expected is None else str(expected)):
                raise ValueError('Supplement label round-trip failed')
    save(out / 'label_audit.json', list(results.values()))
    report = {'rows': len(rows), 'source_sheet_sha256': sha(sheet),
        'output_sha256': sha(path), 'ready_for_fitting': False,
        'blank_means': 'unresolved, never a negative label',
        'counts': {a: dict(Counter('unresolved' if r[a] is None else str(r[a])
                                 for r in results.values())) for a in AXES},
        'rows_needing_review': sum(any(r[a] is None for a in AXES) for r in results.values()),
        'audit_caveat': audit_caveat or 'This export uses the original acceptance rule, which preserves accepted primary labels. The independent positive-label audit found disagreements; this provisional file is not ready for fitting.',
        'decision_rule_sha256': decision_rule_sha256,
        'destructive_broad_definition': 'Requested deletion or default overwrite of stored content, including ordinary file copies and writes; not overwrite-only.',
        'world_identity_key_read': False, 'monitor_fitting_started': False}
    save(out / 'manifest.json', report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','cache','out'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('review-config','final-config','rules'):p.add_argument('--'+name,type=Path)
    p.add_argument('--repair-config',type=Path,action='append',default=[])
    p.add_argument('--review-repair-config',type=Path,action='append',default=[])
    p.add_argument('--mode',choices=['status','adjudication','final_review','audit','export','supplement'],required=True)
    a=p.parse_args();config=json.loads(a.config.read_text())
    records=[json.loads(f.read_text()) for f in (a.cache/'records').glob('*.json')]
    primary=repaired_primary(config,records,a.repair_config,sha(a.config))
    review=repaired_primary(json.loads(a.review_config.read_text()),records,a.review_repair_config,
                            sha(a.review_config)) if a.review_config else None
    final=votes(json.loads(a.final_config.read_text()),records) if a.final_config else None
    results=combine(primary,review,final)
    if a.mode=='status':
        report={'rows':len(results),'axes':{ax:dict(Counter('unresolved' if r[ax] is None else str(r[ax])
              for r in results.values())) for ax in AXES},
              'rows_needing_review':sum(any(r[ax] is None for ax in AXES) for r in results.values())}
        save(a.out,report)
    elif a.mode=='export':report=export(config,results,a.out,a.rules)
    elif a.mode=='supplement':
        if any(r['status']=='inflight' for r in records):
            raise ValueError('Wait for outstanding requests before exporting the provisional supplement')
        report=export_supplement(config,results,a.out)
    else:
        required=[config]+[json.loads(path.read_text()) for path in a.repair_config]
        if not all(attempts_complete(cfg,records) for cfg in required):
            raise ValueError('Finish all primary and repair attempts before preparing extra reviews')
        if a.mode=='audit':
            population=sorted(sid for sid,r in results.items() if all(r[ax] is not None for ax in AXES))
            chosen=random.Random(config['seed']).sample(population,min(50,len(population)))
            results={sid:r|({ax:None for ax in AXES} if sid in chosen else {}) for sid,r in results.items()
                     if sid in chosen}
        report=prepare_review(config,results,a.out,a.mode)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
