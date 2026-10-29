"""Save full private judge records as compressed JSONL without displaying evidence."""
import argparse
import base64
import fcntl
import gzip
import json
import os
from collections import Counter
from pathlib import Path

from probe_shift.dt_label_prepare import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    os.umask(0o077)
    with (a.cache / 'budget.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = sorted((a.cache / 'records').glob('*.json'))
        if not records:
            raise ValueError('No judge records')
        a.out.mkdir(parents=True, exist_ok=False)
        output = a.out / 'judge_records.jsonl.gz'
        statuses, providers = Counter(), Counter()
        reported = accounted = 0.0
        with gzip.open(output, 'xt', encoding='utf-8', compresslevel=3) as stream:
            for path in records:
                record = json.loads(path.read_text())
                if record['status'] == 'inflight':
                    raise ValueError('Cannot archive an unfinished run')
                rid = record['request_id']
                if rid != path.stem:
                    raise ValueError('Record filename mismatch')
                request = a.cache / 'requests' / (rid + '.json')
                responses = [a.cache / 'responses' / (rid + suffix)
                             for suffix in ('.raw', '.http')]
                response_files = [f for f in responses if f.is_file()]
                if len(response_files) > 1:
                    raise ValueError('Ambiguous response files')
                if record['status'] in ('ok', 'invalid_response', 'http_error') and not response_files:
                    raise ValueError('Missing saved response')
                row = {'record': record, 'request': json.loads(request.read_text()),
                       'request_file_sha256': sha(request), 'response': None}
                if response_files:
                    response = response_files[0]
                    row['response'] = {'encoding': 'base64', 'suffix': response.suffix,
                        'sha256': sha(response),
                        'body': base64.b64encode(response.read_bytes()).decode('ascii')}
                stream.write(json.dumps(row, ensure_ascii=True) + '\n')
                statuses[record['status']] += 1
                providers[record.get('provider') or 'not_reported'] += 1
                reported += record.get('reported_usd', 0)
                accounted += record['accounted_usd']
        count = 0
        with gzip.open(output, 'rt', encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                if row['response'] is not None:
                    import hashlib
                    if hashlib.sha256(base64.b64decode(row['response']['body'])).hexdigest() != row['response']['sha256']:
                        raise ValueError('Response archive round-trip failed')
                count += 1
        if count != len(records):
            raise ValueError('Archive row count mismatch')
        report = {'records': count, 'statuses': dict(statuses), 'providers': dict(providers),
                  'reported_usd': reported, 'accounted_usd': accounted,
                  'archive_sha256': sha(output), 'bytes': output.stat().st_size,
                  'response_encoding': 'base64 preserves exact provider bytes',
                  'raw_evidence_displayed': False}
        with (a.out / 'manifest.json').open('x') as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
