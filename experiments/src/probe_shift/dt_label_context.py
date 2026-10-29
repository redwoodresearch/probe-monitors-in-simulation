"""Add approved blinded historical context without exposing evidence to the session."""
import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

from probe_shift.dt_label_finish import prepare_review
from probe_shift.dt_label_panel import AXES, digest, jobs_for, save
from probe_shift.dt_label_prepare import sha
from probe_shift.dt_label_verify import load_state, verified_results


def read_context(path, expected_sha):
    if sha(path) != expected_sha:
        raise ValueError('Context supplement hash mismatch')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    result = {}
    for row in rows:
        if set(row) != {'blind_id', 'context_status', 'files', 'missing_context_reason'}:
            raise ValueError('Unexpected context metadata')
        if row['blind_id'] in result or row['context_status'] not in ('available', 'partial', 'unavailable'):
            raise ValueError('Invalid context identity or status')
        if not isinstance(row['files'], list) or not isinstance(row['missing_context_reason'], (str, type(None))):
            raise ValueError('Invalid context fields')
        for f in row['files']:
            if set(f) != {'path', 'contents', 'complete', 'display_normalized', 'other_programs_between', 'missing_reason'}:
                raise ValueError('Unexpected file-context metadata')
            if not isinstance(f['path'], (str, type(None))) or not isinstance(f['contents'], (str, type(None))):
                raise ValueError('Invalid file context')
            if f['contents'] is not None and f['path'] is None:
                raise ValueError('Recovered contents require a resolved path')
            if any(type(f[k]) is not bool for k in ('complete', 'display_normalized', 'other_programs_between')):
                raise ValueError('Invalid context quality flags')
            if not isinstance(f['missing_reason'], (str, type(None))) or (f['complete'] and f['contents'] is None):
                raise ValueError('Invalid context completeness')
        result[row['blind_id']] = row
    return result


def merge_context_high(pool, extra, superseded, records):
    """Replace only a recorded, doubly-unsure prior vote; failures leave it untouched."""
    records = {r['request_id']: r for r in records}
    for sid, labels in extra.items():
        existing = pool.setdefault(sid, {})
        for model, label in labels.items():
            if model in existing:
                old = existing[model]
                if any(old[a + '_confidence'] != 'unsure' for a in AXES):
                    raise ValueError('Context update cannot supersede a confident vote')
                provenance = superseded.get(sid, {}).get(model)
                if not provenance or digest(old) != provenance['label_sha256']:
                    raise ValueError('Missing or mismatched superseded vote')
                record = records.get(provenance['request_id'])
                if not record or record['status'] != 'ok' or record['model'] != model:
                    raise ValueError('Invalid superseded request provenance')
                if sum(digest(r) == provenance['label_sha256'] for r in record['labels']) != 1:
                    raise ValueError('Superseded label absent from original response')
            existing[model] = label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'cache', 'supplement', 'selection', 'prompt', 'catalog', 'out', 'report'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--supplement-sha256', required=True)
    parser.add_argument('--model', action='append', required=True)
    parser.add_argument('--incremental-cap', type=float, required=True)
    parser.add_argument('--concurrency', type=int, default=32)
    parser.add_argument('--exclude-ids', type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    plan = json.loads(args.plan.read_text())
    if sha(plan['decision_rule']) != plan['decision_rule_sha256'] or plan.get('allow_uncertain_majority'):
        raise ValueError('Context review requires the unchanged confident-vote rule')
    contexts = read_context(args.supplement, args.supplement_sha256)
    selection = list(csv.DictReader(args.selection.open()))
    ids = {r['blind_id'] for r in selection}
    if len(ids) != len(selection) or not ids <= contexts.keys():
        raise ValueError('Context request coverage mismatch')
    excluded = set()
    if args.exclude_ids:
        excluded = {r['blind_id'] for r in json.loads(args.exclude_ids.read_text())}
    records = [json.loads(p.read_text()) for p in (args.cache / 'records').glob('*.json')]
    if any(r['status'] == 'inflight' for r in records):
        raise ValueError('Wait for outstanding inference')
    base, primary, high, _ = load_state(plan, records)
    results = verified_results(primary, high)
    chosen = {sid: results[sid] for sid in ids - excluded
              if any(isinstance(f['contents'], str) for f in contexts[sid]['files'])
              and any(results[sid][a] is None for a in AXES)}
    superseded = {}
    by_request = {r['request_id']: r for r in records}
    for stage in plan['high_stages']:
        cfg = json.loads(Path(stage['config']).read_text())
        if not set(cfg['models']) & set(args.model):
            continue
        prepared = Path(cfg['prepared'])
        mapping = {(r['packet'], r['id']): r['blind_id'] for r in json.loads((prepared / 'join.json').read_text())}
        for job in jobs_for(cfg, prepared, json.loads(Path(cfg['catalog']).read_text()), False):
            if job['model'] not in args.model:
                continue
            for rid in (job['request_id'], digest([job['request_id'], 'retry-1'])):
                record = by_request.get(rid)
                if not record or record['status'] != 'ok':
                    continue
                for label in record['labels']:
                    sid = mapping[(record['packet'], label['id'])]
                    if sid not in chosen:
                        continue
                    model = record['model']
                    if label != high[sid][model] or any(label[a + '_confidence'] != 'unsure' for a in AXES):
                        raise ValueError('Only the current doubly-unsure vote may be superseded')
                    if model in superseded.setdefault(sid, {}):
                        raise ValueError('Duplicate superseded model provenance')
                    superseded[sid][model] = {'request_id': rid, 'label_sha256': digest(label)}
    for sid in chosen:
        if set(high.get(sid, {})) & set(args.model) != set(superseded.get(sid, {})):
            raise ValueError('Missing prior review provenance')
    report = prepare_review(base, chosen, args.out, 'adjudication', prefix='ctx_',
                            models=args.model, max_tokens=16384, batch_size=5)
    mapping = {(r['packet'], r['id']): r['blind_id'] for r in json.loads((args.out / 'join.json').read_text())}
    packets = json.loads((args.out / 'packets.json').read_text())
    for packet in packets:
        path = args.out / 'inputs' / (packet['packet'] + '.json')
        payload = json.loads(path.read_text())
        for call in payload['calls']:
            context = contexts[mapping[(packet['packet'], call['id'])]]
            call['context'] = {k: v for k, v in context.items() if k != 'blind_id'}
        save(path, payload)
        packet.update(bytes=path.stat().st_size, sha256=sha(path))
    save(args.out / 'packets.json', packets)
    save(args.out / 'superseded_votes.json', superseded)
    metadata = json.loads((args.out / 'preparation.json').read_text())
    metadata.update(context_supplement_sha256=sha(args.supplement),
                    packet_bytes=sum(p['bytes'] for p in packets),
                    context_status_counts=dict(Counter(contexts[s]['context_status'] for s in chosen)))
    save(args.out / 'preparation.json', metadata)
    cfg = json.loads((args.out / 'config.json').read_text())
    baseline = sum(r['accounted_usd'] for r in records)
    if args.incremental_cap <= 0 or baseline + args.incremental_cap > 1000:
        raise ValueError('Requested context allowance exceeds the active job cap')
    cfg.update(prompt=str(args.prompt.resolve()), catalog=str(args.catalog.resolve()),
               concurrency=args.concurrency, omit_temperature_for=args.model,
               provider_order={m: ['openai'] for m in args.model}, provider_only={m: ['openai'] for m in args.model},
               budget_usd=baseline + args.incremental_cap, incremental_budget_baseline_usd=baseline,
               context_supplement_sha256=sha(args.supplement), context_authorization='Have 2e provide blinded context (Recommended)',
               new_models_approved_by_user='yes but feel free to add more as needed')
    save(args.out / 'config.json', cfg)
    jobs = jobs_for(cfg, args.out, json.loads(args.catalog.read_text()), False)
    report.update(context_supplement_sha256=sha(args.supplement), selected_rows=len(ids),
                  rows_without_recovered_contents=sum(not any(isinstance(f['contents'], str) for f in contexts[s]['files']) for s in ids),
                  superseded_votes=sum(len(v) for v in superseded.values()), baseline_accounted_usd=baseline,
                  incremental_cap_usd=args.incremental_cap, maximum_reservation_usd=sum(j['reserve'] for j in jobs),
                  config_sha256=sha(args.out / 'config.json'), models=args.model)
    save(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
