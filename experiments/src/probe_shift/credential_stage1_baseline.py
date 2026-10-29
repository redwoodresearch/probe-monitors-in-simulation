"""Apply the unchanged screening detector to the same outputs used by the judges."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from probe_shift.credential_screen import detect


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    blind = {}
    for packet in json.loads((a.prepared / 'packets.json').read_text()):
        path = a.prepared / 'inputs' / (packet['packet'] + '.json')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == packet['sha256']
        for row in json.loads(path.read_text())['outputs']:
            blind[packet['packet'], row['id']] = detect(row['text'])
    join = json.loads((a.prepared / 'join.json').read_text())
    assert len(blind) == len(join)
    rows, counts = [], defaultdict(Counter)
    for row in join:
        label, category, placeholder = blind[row['packet'], row['id']]
        rows.append({k: row[k] for k in ('run_id', 'step', 'action')} |
                    {'credential': label, 'category': category, 'placeholder_flag': placeholder})
        counts[row['run_id']].update(outputs=1, positives=label)
    assert len(counts) == 100
    a.out.mkdir(parents=True, exist_ok=False)
    with (a.out / 'screening_baseline_v1.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['run_id', 'step', 'action', 'credential', 'category', 'placeholder_flag'])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r['run_id'], r['step'], r['action'])))
    summary = {'runs': len(counts), 'outputs': len(rows), 'positives': sum(r['credential'] for r in rows),
               'positives_per_run': sum(r['credential'] for r in rows) / len(counts), 'per_run': counts}
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_run'}))


if __name__ == '__main__':
    main()
