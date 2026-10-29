"""KEY HOLDER ONLY: export blind command evidence without releasing the identity key.

This module has no network or inference capability. Do not run it from the labeling
session: the user explicitly requires that session to leave held_key_v2 unopened.
It prints only a numeric receipt and file hashes, never key rows or command text.
"""
import argparse
import csv
import json
import os
from pathlib import Path
from probe_shift.dt_label_prepare import sha, write_json


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def identity(row):
    return tuple(row[k] for k in ('arm', 'run_key', 'irid', 'call', 'call_index'))


def export(a):
    os.umask(0o077)
    if sha(a.blind_sheet) != a.blind_sheet_sha256 or sha(a.original_sheet) != a.original_sheet_sha256:
        raise ValueError('Sheet hash mismatch')
    if sha(a.private_key) != a.private_key_sha256:
        raise ValueError('Private mapping hash mismatch')
    blind = rows(a.blind_sheet)
    original = rows(a.original_sheet)
    key_rows = rows(a.private_key)
    if len(blind) != a.expected_rows or len(original) != a.expected_rows or len(key_rows) != a.expected_rows:
        raise ValueError('Unexpected row count')
    original_by_identity = {identity(r): i for i, r in enumerate(original)}
    private_map = {}
    for row in key_rows:
        source_row = original_by_identity.get(identity(row))
        if source_row is None:
            raise ValueError('Private mapping contains an unknown source identity')
        private_map[row['blind_id']] = source_row
    blind_ids = {r['blind_id'] for r in blind}
    if (len(original_by_identity) != a.expected_rows or len(private_map) != a.expected_rows
            or set(private_map) != blind_ids or len(set(private_map.values())) != a.expected_rows):
        raise ValueError('Private map is not a one-to-one correspondence')
    if any(r['destructive'] or r['transfer'] for r in blind):
        raise ValueError('Blinded input is already labeled')
    prepared = json.loads((a.source_prepared / 'preparation.json').read_text())
    if prepared['sheet_sha256'] != a.original_sheet_sha256:
        raise ValueError('Prepared source belongs to another original sheet')
    source_join = json.loads((a.source_prepared / 'join.json').read_text())
    by_packet_id = {(j['packet'], j['id']): j['sheet_row'] for j in source_join}
    if len(by_packet_id) != a.expected_rows or {j['sheet_row'] for j in source_join} != set(range(a.expected_rows)):
        raise ValueError('Source evidence join is incomplete')
    evidence = {}
    for packet in json.loads((a.source_prepared / 'packets.json').read_text()):
        p = a.source_prepared / 'inputs' / (packet['packet'] + '.json')
        if sha(p) != packet['sha256']:
            raise ValueError('Prepared source packet changed')
        data = json.loads(p.read_text())
        for item in data['calls']:
            i = by_packet_id[(packet['packet'], item['id'])]
            if i in evidence:
                raise ValueError('Duplicate source evidence')
            evidence[i] = {'tool': item['tool'], 'args': item['args'],
                           'tool_definition': data['tool_definitions'][item['definition']]}
    if set(evidence) != set(range(a.expected_rows)):
        raise ValueError('Missing source evidence')
    a.out.mkdir(parents=True, exist_ok=False)
    bundle = a.out / 'blind_commands_v2.jsonl'
    with bundle.open('x') as f:
        # Preserve the v2 sheet's already-secret shuffle; do not reconstruct or record its seed.
        for row in blind:
            i = private_map[row['blind_id']]
            if any(row[k] != original[i][k] for k in ('task', 'tool', 'existing_label')):
                raise ValueError('Blinded row does not match its source metadata')
            if evidence[i]['tool'] != row['tool']:
                raise ValueError('Evidence tool differs from sheet tool')
            f.write(json.dumps({'blind_id': row['blind_id'], **evidence[i]}, ensure_ascii=False) + '\n')
    receipt = {'blinding_version': 2, 'rows': len(blind), 'blind_sheet_sha256': sha(a.blind_sheet),
               'blind_commands_sha256': sha(bundle), 'private_mapping_sha256': a.private_key_sha256,
               'original_sheet_sha256': a.original_sheet_sha256, 'script_sha256': sha(__file__),
               'fields': ['blind_id', 'tool', 'args', 'tool_definition'],
               'origin_identity_or_task_columns_exported': False, 'network_calls': 0, 'inference_calls': 0}
    write_json(a.out / 'receipt.json', receipt)
    print(json.dumps(receipt, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('blind-sheet', 'original-sheet', 'private-key', 'source-prepared', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--blind-sheet-sha256', default='707188aedd55fc7091c090e6a555bb8aa61b4e83f10897d06fc52eb94dda8385')
    p.add_argument('--original-sheet-sha256', default='9d43c790e5dc7415d2e6e23f67673dba5e4946de11e7b61c7f1d253d3d9bf87b')
    p.add_argument('--private-key-sha256', required=True, help='Expected hash from the key holder’s trusted manifest')
    p.add_argument('--expected-rows', type=int, default=74873)
    export(p.parse_args())


if __name__ == '__main__':
    main()
