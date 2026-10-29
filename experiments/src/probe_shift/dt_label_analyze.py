"""Metadata-only judge diagnostics; never print command evidence or responses."""
import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from probe_shift.dt_label_panel import AXES, aggregate, jobs_for, save, stamp


def analyze(config, cache, pilot=False):
    prepared = Path(config['prepared'])
    catalog = json.loads(Path(config['catalog']).read_text())
    jobs = jobs_for(config, prepared, catalog, pilot)
    records = [json.loads(p.read_text()) for p in (cache/'records').glob('*.json')]
    expected = {j['request_id']:j for j in jobs}
    selected = [r for r in records if r['request_id'] in expected or r.get('retry_of') in expected]
    active_packets = {j['packet'] for j in jobs}
    rows_per_packet = {p['packet']:len(p['ids']) for p in json.loads((prepared/'packets.json').read_text())}
    successful = [r for r in selected if r['status']=='ok']
    result = [r for r in aggregate(prepared, successful, config['models']) if r['packet'] in active_packets]
    largest = max(json.loads((prepared/'packets.json').read_text()), key=lambda p:p['bytes'])['packet']
    models = {}
    for m in config['models']:
        rr = [r for r in selected if r['model']==m]
        ok = [r for r in rr if r['status']=='ok']
        completed = [r for r in rr if r['status']!='inflight']
        random_costs = [r['reported_usd'] for r in ok if r['packet']!=largest and 'reported_usd' in r]
        models[m] = {'selected_requests':sum(j['model']==m for j in jobs),
                     'statuses':dict(Counter(r['status'] for r in rr)),
                     'finish_reasons':dict(Counter(str(r.get('finish_reason')) for r in completed)),
                     'reported_usd':sum(r.get('reported_usd',0) for r in rr),
                     'median_seconds':median([r['elapsed_seconds'] for r in completed]) if completed else None,
                     'projected_panel_usd':mean(random_costs)*len(rows_per_packet) if random_costs else None,
                     'uncertain_labels':{a:sum(l[a+'_confidence']=='unsure' for r in ok for l in r['labels']) for a in AXES}}
    axes = {a:dict(Counter('unresolved' if r[a] is None else str(r[a]) for r in result)) for a in AXES}
    full = [r for r in result if r['valid_judges']==len(config['models'])]
    succeeded = {rid for r in successful for rid in (r['request_id'],r.get('retry_of')) if rid}
    report = {'at_pt':stamp(),'pilot':pilot,'selected_rows':len(result),'selected_packets':len(active_packets),
              'rows_with_all_judges':len(full),'rows_with_some_judges':sum(r['valid_judges']>0 for r in result),
              'completed_all_requested':set(expected)<=succeeded,
              'axes':axes,'full_panel_rows_needing_review':sum(any(r[a] is None for a in AXES) for r in full),
              'models':models,'total_reported_usd':sum(r.get('reported_usd',0) for r in records),
              'total_charge_or_reserve_usd':sum(r['accounted_usd'] for r in records)}
    projections = [v['projected_panel_usd'] for v in models.values()]
    report['projected_primary_panel_usd'] = sum(projections) if all(v is not None for v in projections) else None
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config','cache','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--pilot',action='store_true')
    a = p.parse_args()
    report = analyze(json.loads(a.config.read_text()),a.cache,a.pilot)
    save(a.out,report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
