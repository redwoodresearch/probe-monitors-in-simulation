"""Export q2's exact frozen labels in the original v3 sheet schema and order."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from probe_shift.dt_label_prepare import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'rules', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    sheet = Path(config['sheet'])
    if sha(sheet) != config['blind_sheet_sha256']:
        raise ValueError('Source sheet hash changed')
    summary_path = a.rules.parent / 'summary.json'
    summary = json.loads(summary_path.read_text())
    if sha(a.rules) != summary['output_sha256']:
        raise ValueError('Frozen labels changed')
    with sheet.open(newline='') as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    if fields != ['blind_id', 'tool', 'existing_label', 'destructive', 'transfer']:
        raise ValueError('Expected the v3 five-column sheet')
    with a.rules.open(newline='') as f:
        rules = list(csv.DictReader(f))
    lookup = {r['blind_id']: r for r in rules}
    if (len(rows) != 74873 or len(lookup) != len(rules) or len(rows) != len(rules)
            or len({r['blind_id'] for r in rows}) != len(rows)
            or set(lookup) != {r['blind_id'] for r in rows}):
        raise ValueError('Frozen labels must cover each sheet row exactly once')
    completed = []
    for row in rows:
        labels = {axis: lookup[row['blind_id']]['rule_' + axis]
                  for axis in ('destructive', 'transfer')}
        if any(v not in ('0', '1') for v in labels.values()):
            raise ValueError('Invalid frozen label')
        completed.append(row | labels)
    a.out.mkdir(parents=True, exist_ok=False)
    output = a.out / (sheet.stem + '_q2_labelled.csv')
    with output.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(completed)
    with output.open(newline='') as f:
        reader = csv.DictReader(f)
        reread = list(reader)
        if reader.fieldnames != fields or reread != completed:
            raise ValueError('CSV round-trip failed')
    if any(any(before[k] != after[k] for k in fields[:3])
           for before, after in zip(rows, reread)):
        raise ValueError('Changed IDs, order, tools, or existing labels')
    report = {
        'rows': len(rows), 'columns': fields, 'label_method': 'exact_frozen_q2_rules',
        'output_file': str(output.resolve()), 'output_sha256': sha(output),
        'source_sheet_sha256': sha(sheet), 'rule_labels_sha256': sha(a.rules),
        'commands_sha256': summary['commands_sha256'], 'rule_sources': summary['sources'],
        'planted_transfer_checks_passed': summary['planted_passed'],
        'counts': {axis: sum(int(r[axis]) for r in completed)
                   for axis in ('destructive', 'transfer')},
        'counts_by_existing_label': {
            flag: {'rows': n, **{axis: sum(int(r[axis]) for r in completed
                                          if r['existing_label'] == flag)
                                for axis in ('destructive', 'transfer')}}
            for flag, n in Counter(r['existing_label'] for r in completed).items()},
        'source_sheet_unchanged': True, 'non_label_columns_and_order_preserved': True,
        'held_identity_key_read': False, 'world_breakdown_computed': False,
        'monitor_fitting_started': False,
        'semantic_judge_labels_included': False,
    }
    with (a.out / 'manifest.json').open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
