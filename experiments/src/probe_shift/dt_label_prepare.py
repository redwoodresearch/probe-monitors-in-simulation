"""Resolve an ID-only sheet to blind command packets; never execute or print commands."""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


FIELDS = ['arm', 'task', 'run_key', 'irid', 'call', 'call_index', 'tool',
          'existing_label', 'destructive', 'transfer']
REF = re.compile(r'attachment://[0-9a-f]{32}')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with path.open('x') as fh:
        json.dump(value, fh, ensure_ascii=False)
        fh.write('\n')


def resolve(value, attachments, chain=()):
    if isinstance(value, str) and REF.fullmatch(value):
        key = value.removeprefix('attachment://')
        if key in chain or key not in attachments:
            raise ValueError('Missing or cyclic attachment')
        return resolve(attachments[key], attachments, (*chain, key))
    if isinstance(value, dict):
        return {k: resolve(v, attachments, chain) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, attachments, chain) for v in value]
    return value


def extract(path, wanted, acting_model):
    """The sheet's #e is a context epoch, NOT Inspect's sample epoch.

    Join to the unique actor request irid and its original tool-call index.
    Preserve #e and the readout ordinal unchanged in the separate row join.
    """
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.startswith('samples/') and n.endswith('.json')]
        if len(names) != 1:
            return None
        sample = json.loads(archive.read(names[0]))
    attachments = sample.get('attachments') or {}
    events = {}
    for event in sample.get('events') or []:
        if event.get('event') != 'model' or event.get('model') != acting_model:
            continue
        irid = (((event.get('call') or {}).get('request') or {}).get('extra_headers') or {}).get('x-irid')
        if irid is None:
            raise ValueError('Actor event without irid')
        if irid in events:
            raise ValueError('Duplicate actor request irid')
        events[irid] = event
    result = {}
    for row_number, row in wanted:
        event = events[row['irid']]
        choices = (event.get('output') or {}).get('choices') or []
        if len(choices) != 1:
            raise ValueError('Actor event without exactly one completion')
        call = choices[0]['message']['tool_calls'][int(row['call_index'])]
        if call['function'] != row['tool']:
            raise ValueError('Tool name mismatch')
        args = resolve(call.get('arguments') or {}, attachments)
        if not isinstance(args, dict):
            raise ValueError('Tool arguments are not a dictionary')
        definitions = [t for t in event.get('tools') or [] if t.get('name') == row['tool']]
        if len(definitions) > 1 or not definitions:
            raise ValueError('Missing or duplicate tool definition')
        result[row_number] = {'tool': row['tool'], 'args': args, 'tool_definition': definitions[0]}
    return result


def prepare(a):
    os.umask(0o077)
    if sha(a.sheet) != a.sheet_sha256:
        raise ValueError('Sheet hash mismatch')
    with a.sheet.open() as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != FIELDS:
            raise ValueError('Sheet columns changed')
        rows = list(reader)
    keys = [(r['arm'], r['run_key'], r['irid'], r['call']) for r in rows]
    actor_keys = [(r['run_key'].rsplit('#e', 1)[0], r['irid'], r['call_index']) for r in rows]
    if len(rows) != a.expected_rows or len(set(keys)) != len(rows) or len(set(actor_keys)) != len(rows):
        raise ValueError('Sheet row count or uniqueness check failed')
    if any(r['destructive'] or r['transfer'] for r in rows):
        raise ValueError('Input labels are not empty')
    if any(r['existing_label'] not in ('impactful', 'read_only') for r in rows):
        raise ValueError('Unknown existing label')
    # The source staging code emits call_end then command for each eligible action.
    if any(int(r['call']) != 2 * int(r['call_index']) + 1 for r in rows):
        raise ValueError('Unexpected command readout ordinal; inspect source staging code')
    wanted = defaultdict(list)
    for i, row in enumerate(rows):
        wanted[row['run_key'].rsplit('#e', 1)[0]].append((i, row))
    index = json.loads(a.index.read_text())
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'inputs').mkdir()
    outputs, provenance, seen_irids = {}, [], {}
    for n, (run, listed) in enumerate(wanted.items(), 1):
        candidates = [Path(p) for p in index[run] if Path(p).is_file()]
        # Compare every distinct archive copy by the exact evidence sent to judges.
        # Incomplete copies are recorded; conflicting complete evidence stops preparation.
        hashes = defaultdict(list)
        for path in candidates:
            if '/'.join(path.parts[-3:]) != run:
                raise ValueError('Source path does not match archive ID')
            hashes[sha(path)].append(path)
        evidence, selected = None, None
        rejected = []
        for digest, paths in hashes.items():
            candidate = extract(paths[0], listed, a.acting_model)
            if candidate is None:
                rejected.append(digest)
                continue
            if evidence is not None and evidence != candidate:
                raise ValueError('Divergent complete archive copies; resolve before judging')
            evidence, selected = candidate, str(paths[0])
        if evidence is None:
            raise ValueError('No complete archive found')
        for _, row in listed:
            if row['irid'] in seen_irids and seen_irids[row['irid']] != run:
                raise ValueError('Actor irid occurs in multiple archives')
            seen_irids[row['irid']] = run
        outputs.update(evidence)
        provenance.append({'archive_id': run, 'selected_path': selected,
                           'candidate_hashes': list(hashes), 'incomplete_hashes': rejected,
                           'evidence_sha256': hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest(),
                           'matched_rows': len(listed)})
        if n % 100 == 0:
            print(json.dumps({'archives_matched': n, 'rows_matched': len(outputs)}), flush=True)
    if set(outputs) != set(range(len(rows))):
        raise ValueError('Incomplete row coverage')
    rng = random.Random(a.seed)
    order = list(outputs)
    rng.shuffle(order)
    packets, join, batch = [], [], []
    nbytes = 0
    def flush():
        nonlocal batch, nbytes
        if not batch:
            return
        pid = f'p{len(packets):05d}'
        payload = {'calls': [], 'tool_definitions': {}}
        for j, row_number in enumerate(batch):
            item = outputs[row_number]
            definition = item['tool_definition']
            did = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()[:16]
            payload['tool_definitions'][did] = definition
            cid = f'c{j:03d}'
            payload['calls'].append({'id': cid, 'tool': item['tool'], 'args': item['args'], 'definition': did})
            join.append({'packet': pid, 'id': cid, 'sheet_row': row_number})
        path = a.out / 'inputs' / f'{pid}.json'
        write_json(path, payload)
        packets.append({'packet': pid, 'ids': [c['id'] for c in payload['calls']],
                        'sha256': sha(path), 'bytes': path.stat().st_size})
        batch, nbytes = [], 0
    for row_number in order:
        size = len(json.dumps(outputs[row_number], ensure_ascii=False).encode())
        if batch and (len(batch) >= a.batch_size or nbytes + size > a.batch_bytes):
            flush()
        batch.append(row_number)
        nbytes += size
    flush()
    write_json(a.out / 'packets.json', packets)
    write_json(a.out / 'join.json', join)
    write_json(a.out / 'sources.json', provenance)
    summary = {'stamp_pt': datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
               'sheet_sha256': sha(a.sheet), 'source_index_sha256': sha(a.index),
               'script_sha256': sha(__file__), 'rows': len(rows), 'archive_ids': len(wanted),
               'context_epochs': len({r['run_key'] for r in rows}), 'packets': len(packets),
               'packet_bytes': sum(p['bytes'] for p in packets),
               'largest_packet_bytes': max(p['bytes'] for p in packets),
               'incomplete_archive_copies': sum(len(p['incomplete_hashes']) for p in provenance),
               'labels_by_existing_type': dict(Counter(r['existing_label'] for r in rows)),
               'seed': a.seed, 'batch_size': a.batch_size, 'batch_bytes': a.batch_bytes,
               'acting_model': a.acting_model, 'raw_content_printed': False,
               'judge_metadata': ['opaque call id', 'tool', 'arguments', 'tool definition'],
               'labels_created': False}
    write_json(a.out / 'preparation.json', summary)
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sheet', type=Path, required=True)
    p.add_argument('--sheet-sha256', required=True)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--expected-rows', type=int, default=74873)
    p.add_argument('--acting-model', default='vllm/Qwen/Qwen3.5-27B')
    p.add_argument('--seed', type=int, default=20261012)
    p.add_argument('--batch-size', type=int, default=40)
    p.add_argument('--batch-bytes', type=int, default=70000)
    prepare(p.parse_args())


if __name__ == '__main__':
    main()
