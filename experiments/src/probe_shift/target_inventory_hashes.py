"""Publish a hash-only view of an outside-session target inventory."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--targets', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    with a.targets.open() as fh: original = list(csv.DictReader(fh))
    rows, counts = [], Counter()
    for r in original:
        aliases = json.loads(r['aliases'])
        assert isinstance(aliases, list) and all(isinstance(x,str) for x in aliases)
        rows.append({'side_task_id': r['side_task_id'], 'target_sha256': digest(r['target']),
                     'type': r['type'], 'alias_hashes': json.dumps(sorted(digest(x) for x in aliases)),
                     'alias_count': len(aliases), 'source_commit': r['source_commit']})
        counts[r['type']] += 1
    a.out.mkdir(parents=True, exist_ok=False)
    with (a.out/'target_hashes_v1.csv').open('x',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    summary={'rows':len(rows),'side_tasks':len({r['side_task_id'] for r in rows}), 'counts_by_type':dict(counts),
             'raw_inventory_sha256':hashlib.sha256(a.targets.read_bytes()).hexdigest(),
             'producer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'hash_csv_sha256':hashlib.sha256((a.out/'target_hashes_v1.csv').read_bytes()).hexdigest()}
    (a.out/'target_inventory_summary_v1.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'rows':len(rows),'side_tasks':summary['side_tasks']}))


if __name__=='__main__': main()
