"""Download the exact new cohort and prepare every reader-visible output for judges."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import random
import subprocess
import uuid
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from probe_shift.transcripts import reader_messages, resolve_attachments


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_reader_messages(sample):
    """Cross-check events against saved messages; recover only an entirely absent conversation."""
    models = {e['model'] for e in sample['events'] if e['event'] == 'model'}
    assert len(models) == 1, 'Multiple event models need explicit actor selection'
    messages = []
    for event in sample['events']:
        if event['event'] == 'model':
            assert len(event['output']['choices']) == 1
            messages.append(event['output']['choices'][0]['message'])
        elif event['event'] == 'tool':
            messages.append({'role': 'tool', 'function': event['function'], 'tool_call_id': event['id'],
                             'content': event.get('result', ''), 'error': event.get('error')})
    def read(s):
        return [m for m in reader_messages(resolve_attachments(s), 'system') if m['role'] in ('assistant', 'tool')]
    saved, rebuilt = read(sample), read(sample | {'messages': messages})
    if saved == rebuilt:
        return saved, 'saved_messages_event_verified'
    assert not saved, 'Partial or inconsistent saved conversation: stop instead of silently replacing it'
    assert any(m['role'] == 'tool' for m in rebuilt)
    return rebuilt, 'event_recovery_saved_conversation_absent'


def prepare(a):
    os.umask(0o077)
    manifest = list(csv.DictReader(a.manifest.open()))
    assert len(manifest) == len({r['run_id'] for r in manifest}) == 100
    order = next(k for k in manifest[0] if k.startswith('real_run_order'))
    pairs = defaultdict(list)
    for row in manifest:
        pairs[row['main_task']].append(row[order])
    assert len(pairs) == 50 and all(sorted(v) == ['1', '2'] for v in pairs.values())
    a.cache.mkdir(parents=True, exist_ok=True)
    (a.cache / 'archives').mkdir(exist_ok=True)
    def fetch(row):
        _, filename, sid, epoch = row['run_id'].split('|')
        assert len(filename.split('/')) == 2 and '..' not in filename
        path = a.cache / 'archives' / (hashlib.sha256(row['run_id'].encode()).hexdigest() + '.eval')
        if not path.exists():
            tmp = path.with_suffix('.partial')
            subprocess.run(['aws', 's3', 'cp', '--profile', a.profile, '--region', a.region,
                            a.s3_prefix.rstrip('/') + '/' + filename, str(tmp), '--only-show-errors'], check=True)
            tmp.rename(path)
        with zipfile.ZipFile(path) as archive:
            sample = json.loads(archive.read(f'samples/{sid}_epoch_{epoch}.json'))
        assert str(sample['id']) == sid and int(sample['epoch']) == int(epoch)
        assert not sample.get('error'), 'Errored run: stop rather than treating it as negative'
        messages, source_mode = checked_reader_messages(sample)
        outputs, step, action = [], -1, 0
        for msg in messages:
            if msg['role'] == 'assistant':
                step, action = step + 1, 0
            elif msg['role'] == 'tool':
                assert step >= 0 and 'attachment://' not in msg['content']
                outputs.append({'run_id': row['run_id'], 'step': step, 'action': action, 'text': msg['content']})
                action += 1
        assert outputs, 'No reader outputs: refuse an empty run'
        return outputs, {'run_id': row['run_id'], 'archive_sha256': sha(path), 'outputs': len(outputs), 'source_mode': source_mode}
    outputs, provenance = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        for n, (items, source) in enumerate(pool.map(fetch, manifest), 1):
            outputs.extend(items)
            provenance.append(source)
            if n % 20 == 0:
                print(json.dumps({'downloaded_and_extracted_runs': n}), flush=True)
    assert len({(r['run_id'], r['step'], r['action']) for r in outputs}) == len(outputs)
    rng = random.Random(a.seed)
    rng.shuffle(outputs)
    inputs = a.cache / 'inputs'
    inputs.mkdir(exist_ok=False)
    packets, join, batch, nbytes = [], [], [], 0
    def flush():
        nonlocal batch, nbytes
        if not batch:
            return
        packet_id = uuid.UUID(int=rng.getrandbits(128)).hex
        payload = {'outputs': []}
        for i, row in enumerate(batch):
            fresh_id = f'o{i:03d}'
            payload['outputs'].append({'id': fresh_id, 'text': row['text']})
            join.append({k: row[k] for k in ('run_id', 'step', 'action')} | {'packet': packet_id, 'id': fresh_id})
        path = inputs / f'{packet_id}.json'
        path.write_text(json.dumps(payload, ensure_ascii=False))
        packets.append({'packet': packet_id, 'ids': [r['id'] for r in payload['outputs']], 'sha256': sha(path), 'bytes': path.stat().st_size})
        batch, nbytes = [], 0
    for row in outputs:
        size = len(row['text'].encode())
        if batch and (len(batch) >= a.batch_size or nbytes + size > a.batch_bytes):
            flush()
        batch.append(row)
        nbytes += size
    flush()
    for name, value in [('join.json', join), ('provenance.json', provenance), ('packets.json', packets)]:
        (a.cache / name).write_text(json.dumps(value, indent=2) + '\n')
    summary = {'runs': len(manifest), 'paired_tasks': len(pairs), 'outputs': len(outputs),
               'input_bytes': sum(p['bytes'] for p in packets), 'largest_input_bytes': max(p['bytes'] for p in packets),
               'packets': len(packets), 'manifest_sha256': sha(a.manifest), 'missing_runs': 0, 'failed_runs': 0,
               'source_modes': dict(Counter(p['source_mode'] for p in provenance))}
    (a.cache / 'preparation.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--s3-prefix', required=True)
    p.add_argument('--profile', default='redwood-sandbox')
    p.add_argument('--region', default='us-west-2')
    p.add_argument('--concurrency', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=20)
    p.add_argument('--batch-bytes', type=int, default=120000)
    p.add_argument('--seed', type=int, default=20261011)
    prepare(p.parse_args())


if __name__ == '__main__':
    main()
