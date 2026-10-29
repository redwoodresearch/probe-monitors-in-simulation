"""Apply q2's frozen rules as comparison labels, never as AI judge inputs."""
import argparse
import ast
import csv
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path
from probe_shift.dt_label_prepare import sha, write_json

SOURCES = {
    'destructive': ('1f816c35', 'experiments/2026-10-10_q2_destructive_commands/analyze_destructive_counts.py'),
    'transfer': ('ce879384', 'experiments/2026-10-10_q2_transfer_commands/analyze_transfer_counts.py'),
    'raw_application': ('ad27372a', 'experiments/2026-10-10_q2_metric_table/build_recal_labels.py'),
}


def frozen(repo, kind, names):
    commit, filename = SOURCES[kind]
    source = subprocess.check_output(['git', '-C', str(repo), 'show', f'{commit}:{filename}'], text=True)
    nodes = []
    found = set()
    for node in ast.parse(source).body:
        name = node.name if isinstance(node, ast.FunctionDef) else (
            node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None)
        if name in names:
            nodes.append(node)
            found.add(name)
    if found != set(names):
        raise ValueError('Frozen source definitions changed or are absent')
    scope = {'re': re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), f'frozen_{kind}', 'exec'), scope)
    return scope


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'commands', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--commands-sha256', required=True)
    a = p.parse_args()
    os.umask(0o077)
    if sha(a.commands) != a.commands_sha256:
        raise ValueError('Command bundle hash mismatch')
    destructive = frozen(a.repo, 'destructive', ['PATTERNS', 'DESTRUCTIVE'])
    transfer = frozen(a.repo, 'transfer', ['PATTERNS', 'TRANSFER', 'CONTINUATION', 'LOCAL',
        'FILE_EXT', 'PLANTED', 'planted_ok', 'executed', 'joined', 'is_transfer', 'HOSTS', 'external'])
    if not transfer['planted_ok']():
        raise ValueError('Frozen transfer planted test failed')
    a.out.mkdir(parents=True, exist_ok=False)
    totals, seen = Counter(), set()
    with a.commands.open() as source, (a.out/'frozen_rules.csv').open('x') as f:
        w = csv.DictWriter(f, fieldnames=['blind_id', 'rule_destructive', 'rule_transfer'])
        w.writeheader()
        for line in source:
            row = json.loads(line)
            if row['blind_id'] in seen:
                raise ValueError('Duplicate blind ID')
            seen.add(row['blind_id'])
            args = json.loads(row['arguments'])
            # ad27372a: apply to individual raw values, not the JSON-escaped whole argument dict.
            d = int(any(destructive['DESTRUCTIVE'].search(v if isinstance(v, str) else json.dumps(v))
                        for v in args.values()))
            t = int(transfer['executed'](row['tool'] or '') and transfer['is_transfer'](args))
            w.writerow({'blind_id':row['blind_id'], 'rule_destructive':d, 'rule_transfer':t})
            totals.update(rows=1, destructive=d, transfer=t, both=d*t)
    report = {'counts': dict(totals), 'sources': SOURCES, 'commands_sha256': a.commands_sha256,
              'output_sha256':sha(a.out/'frozen_rules.csv'), 'planted_passed':len(transfer['PLANTED']),
              'comparison_only':True, 'fed_to_judges':False, 'script_sha256':sha(__file__)}
    write_json(a.out/'summary.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
