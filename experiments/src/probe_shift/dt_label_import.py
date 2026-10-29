"""Import a v2 blind command bundle without reading its held identity key."""
import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from probe_shift.dt_label_prepare import REF, sha, write_json


def unresolved_reference(value):
    if isinstance(value, str):
        return bool(REF.fullmatch(value))
    if isinstance(value, dict):
        return any(unresolved_reference(v) for v in value.values())
    if isinstance(value, list):
        return any(unresolved_reference(v) for v in value)
    return False


def prepare(a):
    os.umask(0o077)
    for path, expected in ((a.sheet, a.sheet_sha256), (a.commands, a.commands_sha256),
                           (a.definitions, a.definitions_sha256)):
        if sha(path) != expected:
            raise ValueError('Input hash mismatch')
    with a.sheet.open() as f:
        reader = csv.DictReader(f)
        schemas = [ ['blind_id', 'task', 'tool', 'existing_label', 'destructive', 'transfer'],
                    ['blind_id', 'tool', 'existing_label', 'destructive', 'transfer'] ]
        if reader.fieldnames not in schemas:
            raise ValueError('Unexpected blind sheet columns')
        sheet_version = 2 if 'task' in reader.fieldnames else 3
        sheet = list(reader)
    expected = {r['blind_id']: r for r in sheet}
    if len(expected) != len(sheet) or len(sheet) != a.expected_rows:
        raise ValueError('Wrong sheet size or duplicate IDs')
    if any(r['destructive'] or r['transfer'] for r in sheet):
        raise ValueError('Sheet labels are not empty')
    definition_tools = {}
    if a.definitions.suffix == '.jsonl':
        definitions = {}
        for line in a.definitions.read_text().splitlines():
            item = json.loads(line)
            if set(item) != {'tool_def_id', 'tool', 'definition'}:
                raise ValueError('Unexpected definition metadata fields')
            did = item['tool_def_id']
            if did in definitions:
                raise ValueError('Duplicate tool definition ID')
            definitions[did] = item['definition']
            definition_tools[did] = item['tool']
    else:
        definitions = json.loads(a.definitions.read_text())
    if not isinstance(definitions, dict) or any(not isinstance(d, dict) for d in definitions.values()):
        raise ValueError('Definitions must be a JSON object mapping IDs to definition objects')
    wanted_fields = {'blind_id', 'tool', a.arguments_field, a.definition_id_field}
    commands, definition_hashes = [], defaultdict(set)
    with a.commands.open() as f:
        for line in f:
            row = json.loads(line)
            if set(row) != wanted_fields:
                raise ValueError('Command bundle contains unexpected metadata fields')
            sid = row['blind_id']
            if sid not in expected or row['tool'] != expected[sid]['tool']:
                raise ValueError('Command ID or tool does not match blinded sheet')
            args = row[a.arguments_field]
            if isinstance(args, str):
                parsed = json.loads(args)
            else:
                parsed = args
            if not isinstance(parsed, dict) or unresolved_reference(parsed):
                raise ValueError('Arguments are not an object or contain unresolved attachment references')
            definition = definitions.get(row[a.definition_id_field])
            if definition is None:
                raise ValueError('Unknown tool definition ID')
            if definition_tools and definition_tools[row[a.definition_id_field]] != row['tool']:
                raise ValueError('Tool definition belongs to another tool')
            if unresolved_reference(definition):
                raise ValueError('Unresolved attachment in tool definition')
            did = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
            definition_hashes[row['tool']].add(did)
            commands.append({'blind_id': sid, 'tool': row['tool'], 'args': args,
                             'definition': did, 'tool_definition': definition})
    if [c['blind_id'] for c in commands] != [r['blind_id'] for r in sheet]:
        raise ValueError('Command bundle must exactly cover the v2 sheet in its supplied shuffled order')
    if any(len(values) != 1 for values in definition_hashes.values()):
        raise ValueError('Multiple definition variants for one tool; require the key holder’s normalization')
    # Fresh private ordering; no reproducible public seed can recover sheet positions.
    random.SystemRandom().shuffle(commands)
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'inputs').mkdir()
    packets, join, batch = [], [], []
    nbytes = 0
    def flush():
        nonlocal batch, nbytes
        if not batch:
            return
        pid = f'p{len(packets):05d}'
        payload = {'calls': [], 'tool_definitions': {}}
        for i, command in enumerate(batch):
            cid = f'c{i:03d}'
            payload['calls'].append({'id': cid, 'tool': command['tool'], 'args': command['args'],
                                     'definition': command['definition']})
            payload['tool_definitions'][command['definition']] = command['tool_definition']
            join.append({'packet': pid, 'id': cid, 'blind_id': command['blind_id']})
        path = a.out / 'inputs' / (pid + '.json')
        write_json(path, payload)
        packets.append({'packet': pid, 'ids': [r['id'] for r in payload['calls']],
                        'bytes': path.stat().st_size, 'sha256': sha(path)})
        batch, nbytes = [], 0
    for command in commands:
        size = len(json.dumps(command, ensure_ascii=False).encode())
        if batch and (len(batch) >= a.batch_size or nbytes + size > a.batch_bytes):
            flush()
        batch.append(command)
        nbytes += size
    flush()
    write_json(a.out / 'packets.json', packets)
    write_json(a.out / 'join.json', join)
    summary = {'blinding_version': 2, 'sheet_version': sheet_version, 'blind_sheet_sha256': a.sheet_sha256,
               'commands_sha256': a.commands_sha256, 'definitions_sha256': a.definitions_sha256,
               'definitions_normalized_by_exporter': a.definitions_normalized == 'true',
               'definition_variants_per_tool': {k:len(v) for k,v in definition_hashes.items()},
               'rows': len(commands), 'per_tool': dict(Counter(c['tool'] for c in commands)),
               'packets': len(packets), 'packet_bytes': sum(p['bytes'] for p in packets),
               'batch_size': a.batch_size, 'batch_bytes': a.batch_bytes,
               'script_sha256': sha(__file__), 'shuffle': 'SystemRandom; seed not recorded',
               'held_identity_key_read': False, 'raw_arguments_preserved': True,
               'judges_see_task_or_existing_labels': False, 'labels_created': False}
    write_json(a.out / 'preparation.json', summary)
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('sheet', 'commands', 'definitions', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--sheet-sha256', default='707188aedd55fc7091c090e6a555bb8aa61b4e83f10897d06fc52eb94dda8385')
    p.add_argument('--commands-sha256', required=True)
    p.add_argument('--definitions-sha256', required=True)
    p.add_argument('--definitions-normalized', choices=['true', 'false'], required=True)
    p.add_argument('--arguments-field', default='arguments')
    p.add_argument('--definition-id-field', default='tool_definition_id')
    p.add_argument('--expected-rows', type=int, default=74873)
    p.add_argument('--batch-size', type=int, default=40)
    p.add_argument('--batch-bytes', type=int, default=70000)
    prepare(p.parse_args())


if __name__ == '__main__':
    main()
