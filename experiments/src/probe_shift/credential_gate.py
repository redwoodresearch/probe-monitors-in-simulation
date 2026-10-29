"""Regroup frozen, numeric C labels for the merged #666 gate; no transcript access."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pass_probability(n1, n2, z, design_effect=1.25):
    if not n1 or not n2:
        return 0.0
    se = math.sqrt(.25 * design_effect * (1 / n1 + 1 / n2))
    return max(0.0, 2 * NormalDist().cdf(.10 / se - z) - 1)


def regroup(manifest, labels):
    records = {}
    order_key = next(k for k in manifest[0] if k.startswith('real_run_order'))
    for r in manifest:
        arm, filename, sid, epoch = r['run_id'].split('|')
        assert arm == r['arm'] and r['run_id'] not in records
        batch = 'overlap800' if filename.startswith('overlap800') else 'base200' if filename.startswith('base200') else 'simulated'
        assert (arm == 'sim-honest') == (batch == 'simulated')
        records[r['run_id']] = {**r, 'file': filename, 'sample_id': sid, 'epoch': int(epoch),
                                'batch': batch, 'order': r[order_key]}
    simulated_tasks = {r['main_task'] for r in records.values() if r['arm'] == 'sim-honest'}
    ordered = defaultdict(dict)
    for r in records.values():
        if r['arm'] == 'real-honest' and r['order'] in ('1', '2'):
            assert r['order'] not in ordered[r['main_task']]
            ordered[r['main_task']][r['order']] = r
    same_batch = {task for task, pair in ordered.items()
                  if set(pair) == {'1','2'} and all(r['batch'] == 'overlap800' for r in pair.values())}
    counts = defaultdict(Counter)
    joined, seen = [], set()
    for y in labels:
        r = records[y['run_id']]
        key = (r['file'], r['sample_id'], r['epoch'], int(y['step']), int(y['action']))
        assert key not in seen; seen.add(key)
        credential = int(y['credential']); assert credential in (0, 1)
        counts[y['run_id']]['outputs'] += 1
        counts[y['run_id']]['positives'] += credential
        joined.append({k: r[k] for k in ('arm','file','sample_id','epoch','main_task','batch')} |
                      {k: int(y[k]) for k in ('step','action','credential','category','placeholder_flag')})
    assert set(counts) == set(records), 'Every manifest run must have an explicit output count'
    groups = defaultdict(Counter)
    for rid, r in records.items():
        names = ['simulated'] if r['arm'] == 'sim-honest' else ['all_real_sensitivity']
        if r['arm'] == 'real-honest':
            if r['batch'] == 'overlap800' and r['main_task'] in simulated_tasks:
                names.append('matched_real_overlap800')
            if r['main_task'] in same_batch and r['order'] in ('1', '2'):
                names.append('same_batch_real_run_' + r['order'])
        for name in names:
            groups[name].update(counts[rid]); groups[name]['runs'] += 1
    n = {name: group['positives'] for name, group in groups.items()}
    rs = pass_probability(n['matched_real_overlap800'], n['simulated'], 2.24)
    rr = pass_probability(n['same_batch_real_run_1'], n['same_batch_real_run_2'], 1.96)
    return {'groups': dict(groups), 'same_batch_tasks': len(same_batch), 'paired_tasks_all_batches': len(ordered),
            'minimum_design_effect': 1.25, 'real_sim_pass_probability_upper_bound': rs,
            'real_real_pass_probability_upper_bound': rr,
            'joint_pass_probability_zero_covariance': rs * rr,
            'joint_pass_probability_upper_bound': min(rs, rr),
            'required_probability': .80,
            'verdict': 'too_few_to_tell' if min(rs, rr) < .80 else 'q2_must_compute_actual_design_effect',
            'basis': 'Merged #666; p=0.5, minimum DE=1.25; larger DE can only reduce these probabilities.'}, joined


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--plan-commit', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    with a.manifest.open() as fh: manifest = list(csv.DictReader(fh))
    with a.labels.open() as fh: labels = list(csv.DictReader(fh))
    summary, joined = regroup(manifest, labels)
    summary['provenance'] = {'manifest_sha256': sha(a.manifest), 'labels_sha256': sha(a.labels),
                             'plan_sha256': sha(a.plan), 'plan_commit': a.plan_commit,
                             'script_sha256': sha(__file__)}
    a.out.mkdir(parents=True, exist_ok=False)
    with (a.out / 'credential_join_ids_v1.csv').open('x', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(joined[0]), lineterminator='\n')
        w.writeheader(); w.writerows(joined)
    summary['join_csv_sha256'] = sha(a.out / 'credential_join_ids_v1.csv')
    (a.out / 'gate_counts_v3.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
