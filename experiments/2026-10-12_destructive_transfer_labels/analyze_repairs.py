"""One technical repair of failed primary batches, with exact evidence and model preserved."""
import argparse
import json
from pathlib import Path
from probe_shift.dt_label_finish import prepare_review,repair_record
from probe_shift.dt_label_panel import jobs_for,provider_error_codes,validate,save
from probe_shift.dt_label_prepare import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','cache','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=10)
    p.add_argument('--skip-unclassified',action='store_true',
                   help='Leave unclassified responses missing; never resubmit them.')
    a=p.parse_args();cfg=json.loads(a.config.read_text());prepared=Path(cfg['prepared'])
    jobs=jobs_for(cfg,prepared,json.loads(Path(cfg['catalog']).read_text()),False)
    records={f.stem:json.loads(f.read_text()) for f in (a.cache/'records').glob('*.json')}
    join=json.loads((prepared/'join.json').read_text())
    mapping={(r['packet'],r['id']):r['blind_id'] for r in join}
    failed={model:{} for model in cfg['models']};reasons={};skipped={}
    for job in jobs:
        r=repair_record(job['request_id'],records)
        if r is None:continue
        eligible=r.get('retryable') or r.get('http_status')==413
        if r['status']=='invalid_response':
            response=json.loads((a.cache/'responses'/(r['request_id']+'.raw')).read_text())
            eligible=eligible or any(c in (408,429,500,502,503,504) for c in provider_error_codes(response))
            try:validate(response,job['ids'])
            except (ValueError,KeyError,TypeError,IndexError) as error:
                eligible=eligible or str(error) in ('missing_duplicate_or_invented_ids',
                    'label_must_be_binary_integer','missing_or_invalid_confidence')
        if not eligible:
            if not a.skip_unclassified:
                raise ValueError('Unclassified failure requires inspection before any retry')
            skipped[r['request_id']]={k:r.get(k) for k in ('model','status','finish_reason','http_status')}
            continue
        for cid in job['ids']:
            sid=mapping[(job['packet'],cid)]
            failed[job['model']][sid]={'blind_id':sid,'destructive':None,'transfer':None}
        reasons[r['request_id']]={k:r.get(k) for k in ('model','status','finish_reason','http_status')}
    a.out.mkdir(parents=True,exist_ok=False)
    configs=[];reports={}
    for i,(model,rows) in enumerate(failed.items()):
        if not rows:continue
        directory=a.out/f'm{i}'
        reports[model]=prepare_review(cfg,rows,directory,'repair',prefix=f'r{i}_',models=[model],
            reasoning={model:cfg['reasoning_effort'][model]},max_tokens=16384,batch_size=a.batch_size)
        path=directory/'config.json';new=json.loads(path.read_text())
        new.update(repair_of_config_sha256=sha(a.config),technical_repair_number=1,concurrency=256)
        save(path,new);configs.append(str(path.resolve()))
    save(a.out/'repair_manifest.json',{'configs':configs,'per_model':reports,'failed_original_requests':reasons,
        'unclassified_left_missing_without_retry':skipped})
    print(json.dumps({'failed_original_requests':len(reasons),'repairs':reports,
        'unclassified_left_missing_without_retry':len(skipped)},indent=2))


if __name__=='__main__':main()
