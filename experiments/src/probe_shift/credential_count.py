"""Screen a fixed local manifest; persist only IDs, hashes and numeric labels/counts."""

import argparse
import csv
import hashlib
import json
import random
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from probe_shift.credential_screen import detect
from probe_shift.transcripts import reader_messages


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20261011)
    args = p.parse_args()
    manifest = list(csv.DictReader(args.manifest.open()))
    order_key = next(k for k in manifest[0] if k.startswith('real_run_order'))
    assert len(manifest) == 1437
    assert Counter(r['arm'] for r in manifest) == {'real-honest': 1000, 'sim-honest': 437}
    pair1 = {r['main_task'] for r in manifest if r[order_key] == '1'}
    pair2 = {r['main_task'] for r in manifest if r[order_key] == '2'}
    assert pair1 == pair2 and len(pair1) == 258
    index = json.loads(args.index.read_text())
    assert set(index) == {r['run_id'] for r in manifest}
    archive_runs = defaultdict(dict)
    for row in manifest:
        _, filename, sample_id, epoch = row['run_id'].split('|')
        source = Path(index[row['run_id']][0])
        assert source.name == filename and source.is_file()
        key = (sample_id, int(epoch))
        assert key not in archive_runs[source]
        archive_runs[source][key] = row
    args.out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)
    sources = list(archive_runs)
    rng.shuffle(sources)
    counts = {g: Counter() for g in ('simulated', 'real_run_1_paired', 'real_run_2_paired', 'all_real')}
    seen_runs, seen_ids, per_run, provenance = set(), set(), [], []
    label_columns = ['run_id','step','action','credential','category','placeholder_flag']
    with (args.out/'screening_labels_v1.csv').open('x',newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=label_columns)
        writer.writeheader()
        for n, source in enumerate(sources, 1):
            wanted = archive_runs[source]
            provenance.append({'archive_sha256':digest(source), 'run_ids':sorted(r['run_id'] for r in wanted.values())})
            with zipfile.ZipFile(source) as archive:
                names = [s for s in archive.namelist() if s.startswith('samples/') and s.endswith('.json')]
                rng.shuffle(names)
                for name in names:
                    # Only exact requested sample+epoch members are opened when the standard Inspect naming is used.
                    expected = {f'samples/{sid}_epoch_{ep}.json': (sid,ep) for sid,ep in wanted}
                    if name not in expected:
                        continue
                    sample = json.loads(archive.read(name))
                    key = (str(sample['id']),int(sample.get('epoch',1)))
                    assert key == expected[name]
                    row = wanted[key]
                    assert row['run_id'] not in seen_runs
                    seen_runs.add(row['run_id'])
                    outputs = []
                    step, action = -1, 0
                    for msg in reader_messages(sample, 'system'):
                        if msg['role'] == 'assistant':
                            step += 1
                            action = 0
                        elif msg['role'] == 'tool':
                            assert step >= 0
                            outputs.append((step,action,msg['content']))
                            action += 1
                    rng.shuffle(outputs)
                    run_counts = Counter(outputs=len(outputs))
                    for step,action,content in outputs:
                        # detect is pure and receives no source identity or world.
                        credential,category,placeholder = detect(content)
                        output_id = (row['run_id'],step,action)
                        assert output_id not in seen_ids
                        seen_ids.add(output_id)
                        writer.writerow(dict(zip(label_columns,(*output_id,credential,category,placeholder))))
                        run_counts['positives'] += credential
                        run_counts['placeholder_positives'] += credential and placeholder
                    groups = ['simulated'] if row['arm'] == 'sim-honest' else ['all_real']
                    if row['arm'] == 'real-honest' and row[order_key] in ('1','2'):
                        groups.append(f"real_run_{row[order_key]}_paired")
                    for group in groups:
                        counts[group].update(run_counts)
                        counts[group]['runs'] += 1
                    per_run.append({'run_id':row['run_id'],**dict(run_counts)})
            if n % 20 == 0:
                print(json.dumps({'archives_completed':n,'runs_completed':len(seen_runs)}),flush=True)
    assert seen_runs == {r['run_id'] for r in manifest}, 'Manifest coverage failure; do not use partial counts'
    summary = {'status':'rough_screening_not_audited','created_at_pt':datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
               'manifest_sha256':digest(args.manifest),'seed':args.seed,'groups':counts,'runs':per_run,
               'failed_runs':0,'missing_runs':0,'paired_tasks':258,'output_rows':len(seen_ids),
               'labels_sha256':digest(args.out/'screening_labels_v1.csv')}
    (args.out/'counts_v1.json').write_text(json.dumps(summary,indent=2)+'\n')
    (args.out/'source_hashes_v1.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({'groups':counts,'runs':len(seen_runs),'outputs':len(seen_ids)},indent=2))


if __name__ == '__main__':
    main()
