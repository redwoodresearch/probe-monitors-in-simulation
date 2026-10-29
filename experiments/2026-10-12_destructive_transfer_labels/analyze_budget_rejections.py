"""Reconcile only documented pre-provider budget rejections; preserve originals."""
import argparse
import fcntl
import json
import os
from pathlib import Path

from probe_shift.dt_label_panel import budget_rejection, save, stamp
from probe_shift.dt_label_prepare import sha


def main():
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--apply', action='store_true')
    a = p.parse_args()
    with (a.cache / 'budget.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selected = []
        before = reported = 0.0
        for path in sorted((a.cache / 'records').glob('*.json')):
            record = json.loads(path.read_text())
            if record['status'] == 'inflight':
                raise ValueError('Cannot reconcile while requests are unfinished')
            before += record['accounted_usd']
            reported += record.get('reported_usd', 0)
            if record.get('http_status') != 403 or record['accounted_usd'] == 0:
                continue
            response = a.cache / 'responses' / (path.stem + '.http')
            budget = budget_rejection(json.loads(response.read_bytes()))
            if budget is None:
                continue
            if record.get('reported_usd') or record.get('labels'):
                raise ValueError('Budget rejection unexpectedly has charges or labels')
            selected.append((path, record, budget, sha(response)))
        released = sum(r['accounted_usd'] for _, r, _, _ in selected)
        report = {'stamp_pt': stamp(), 'applied': a.apply, 'requests': len(selected),
            'reported_usd': reported, 'accounted_before_usd': before,
            'documented_unbilled_reservations_usd': released,
            'accounted_after_usd': before - released,
            'source': 'https://openrouter.ai/docs/api_reference/errors-and-debugging',
            'basis': 'Structured HTTP 403 budget metadata identifies rejection before provider routing.',
            'unknown_interrupted_and_other_error_reservations': 'unchanged',
            'retry_permission': 'unchanged; no retries authorized by accounting reconciliation'}
        if a.apply:
            a.out.mkdir(parents=True, exist_ok=False)
            originals = a.out / 'originals'
            originals.mkdir()
            changes = []
            for path, record, budget, response_sha in selected:
                # Copy exact original bytes before atomically replacing accounting metadata.
                saved = originals / path.name
                saved.write_bytes(path.read_bytes())
                saved.chmod(0o600)
                updated = record | {'accounted_usd': 0, 'budget_rejection': budget,
                    'billing_basis': 'documented_pre_provider_budget_rejection',
                    'original_record_sha256': sha(saved)}
                save(path, updated)
                changes.append({'request_id': path.stem, 'original_record_sha256': sha(saved),
                                'updated_record_sha256': sha(path), 'response_sha256': response_sha})
            save(a.out / 'changes.json', changes)
            save(a.out / 'summary.json', report)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
