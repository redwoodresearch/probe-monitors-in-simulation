"""Prepare complete, origin-blind sequences; never read the stage-one answer key."""
import argparse
import csv
import hashlib
import json
import os
import random
import uuid
import zipfile
from collections import defaultdict
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(value, attachments):
    if isinstance(value, str) and value.startswith('attachment://'):
        return resolve(attachments[value.removeprefix('attachment://')], attachments)
    if isinstance(value, list):
        return [resolve(v, attachments) for v in value]
    if isinstance(value, dict):
        return {k: resolve(v, attachments) for k, v in value.items()}
    return value


def prepare(a):
    os.umask(0o077)
    assert sha(a.labels) == a.labels_sha256
    assert sha(a.locator) == a.locator_sha256
    labels = list(csv.DictReader(a.labels.open()))
    locator = list(csv.DictReader(a.locator.open()))
    assert len(labels) == len(locator) == 1457
    assert len({r['opaque_id'] for r in labels}) == len(labels)
    assert len({r['opaque_id'] for r in locator}) == len(locator)
    assert {r['opaque_id'] for r in labels} == {r['opaque_id'] for r in locator}
    assert all(not r['label'] and not r['note'] for r in labels)
    index = json.loads(a.index.read_text())
    run_hashes = {hashlib.sha256(r.encode()).hexdigest()[:16]: r for r in index}
    assert len(run_hashes) == len(index)
    wanted = defaultdict(dict)
    for row in locator:
        key = tuple(int(row[k]) for k in ('epoch', 'turn', 'call_index'))
        run_id = run_hashes[row['run_hash']]
        assert key not in wanted[run_id]
        wanted[run_id][key] = row['opaque_id']
    a.cache.mkdir(parents=True, exist_ok=False)
    (a.cache / 'inputs').mkdir()
    rng = random.Random(a.seed)
    runs = list(wanted)
    rng.shuffle(runs)
    join, provenance, packets, found = [], [], [], set()
    for run_id in runs:
        paths = [Path(p) for p in index[run_id]]
        hashes = {sha(p) for p in paths}
        assert len(hashes) == 1, 'Resolve divergent source archives first'
        with zipfile.ZipFile(paths[0]) as archive:
            members = [n for n in archive.namelist() if n.startswith('samples/') and n.endswith('.json')]
            assert len(members) == 1, 'The frozen source index expects single-sample archives'
            sample = json.loads(archive.read(members[0]))
        epoch = int(sample.get('epoch', 1))
        sample = resolve(sample, sample.get('attachments') or {})
        messages = sample['messages']
        outputs = {m['tool_call_id']: m for m in messages if m['role'] == 'tool'}
        assert len(outputs) == sum(m['role'] == 'tool' for m in messages)
        events = [e for e in sample['events'] if e.get('event') == 'model' and e.get('model') == a.acting_model]
        assert events
        packet_id = uuid.UUID(int=rng.getrandbits(128)).hex
        calls, selected = [], []
        for turn, event in enumerate(events):
            assert (((event.get('call') or {}).get('request') or {}).get('extra_headers') or {}).get('x-irid')
            choices = event['output']['choices']
            assert len(choices) == 1
            for ci, call in enumerate(choices[0]['message'].get('tool_calls') or []):
                fresh_id = f'c{len(calls):04d}'
                output = outputs.get(call['id'])
                # Do not expose the submit call's self-reported success label.
                terminal = call['function'] == 'submit'
                calls.append({'id': fresh_id, 'function': call['function'],
                              'arguments': {} if terminal else call.get('arguments'),
                              'matching_tool_output': None if terminal or output is None else output.get('content'),
                              'tool_error': None if output is None else output.get('error')})
                key = (int(epoch), turn, ci)
                if key in wanted[run_id]:
                    opaque = wanted[run_id][key]
                    assert opaque not in found
                    found.add(opaque)
                    selected.append({'opaque_id': opaque, 'packet': packet_id, 'id': fresh_id})
        assert len(selected) == len(wanted[run_id]), 'Missing locator call; refuse partial output'
        meta = sample['metadata']
        assert meta.get('main_task_str') and meta.get('side_task_text')
        payload = {'cover_task': meta['main_task_str'], 'hidden_side_task': meta['side_task_text'], 'calls': calls}
        path = a.cache / 'inputs' / f'{packet_id}.json'
        path.write_text(json.dumps(payload, ensure_ascii=False))
        packets.append({'packet': packet_id, 'ids': [c['id'] for c in calls], 'sha256': sha(path), 'bytes': path.stat().st_size})
        provenance.append({'run_id': run_id, 'archive_sha256': next(iter(hashes)), 'sample_error': bool(sample.get('error'))})
        join.extend(selected)
    assert found == {r['opaque_id'] for r in labels}
    for name, value in [('join.json', join), ('provenance.json', provenance), ('packets.json', packets)]:
        (a.cache / name).write_text(json.dumps(value, indent=2) + '\n')
    summary = {'requested_rows': len(labels), 'resolved_rows': len(join), 'context_calls': sum(len(p['ids']) for p in packets),
               'input_bytes': sum(p['bytes'] for p in packets), 'largest_input_bytes': max(p['bytes'] for p in packets),
               'labels_sha256': sha(a.labels), 'locator_sha256': sha(a.locator), 'index_sha256': sha(a.index),
               'key_accessed': False, 'selected_command_counts_exposed_to_judges': False}
    (a.cache / 'preparation.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--locator', type=Path, required=True)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--labels-sha256', required=True)
    p.add_argument('--locator-sha256', required=True)
    p.add_argument('--acting-model', default='vllm/Qwen/Qwen3.5-27B')
    p.add_argument('--seed', type=int, default=20261011)
    prepare(p.parse_args())


if __name__ == '__main__':
    main()
