"""Verify semantic positives with independent high-reasoning votes and explicit uncertainty."""
import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from probe_shift.dt_label_finish import AXES, attempts_complete, combine, export, export_supplement, prepare_review, repaired_primary, winner
from probe_shift.dt_label_panel import save
from probe_shift.dt_label_prepare import sha

APPROVED_MODELS = {'google/gemini-3.1-pro-preview', 'openai/gpt-5.4',
                   'deepseek/deepseek-v4-pro', 'moonshotai/kimi-k3', 'x-ai/grok-4.7'}

def merge_high(pool, extra):
    for sid, labels in extra.items():
        existing = pool.setdefault(sid, {})
        if set(existing) & set(labels):
            raise ValueError('A high-review model cannot vote twice on the same call')
        existing.update(labels)


def verified_results(primary, high, *, allow_uncertain_majority=False):
    result = {}
    for sid, labels in primary.items():
        reviews = list(high.get(sid, {}).values())
        row = {'blind_id': sid, 'primary_valid_judges': len(labels), 'high_valid_judges': len(reviews)}
        for axis in AXES:
            value = winner(reviews, axis, 2)
            method = 'verified_high_majority'
            original = winner(labels.values(), axis, 4)
            if value is None:
                # Only original negatives without a confident high-review contradiction
                # may survive without a high quorum. Every positive requires verification.
                contradicts_negative = any(r[axis] == 1 and r[axis + '_confidence'] == 'sure'
                                           for r in reviews)
                if original == 0 and not contradicts_negative:
                    value, method = 0, 'primary_negative'
            if value is None and allow_uncertain_majority and set(high.get(sid, {})) == APPROVED_MODELS:
                votes = Counter(r[axis] for r in reviews)
                value, n = votes.most_common(1)[0]
                if n < 3:
                    raise ValueError('Five valid binary votes must have a strict majority')
                method = 'five_high_majority_uncertain'
            row[axis] = value
            row[axis + '_method'] = method if value is not None else 'unresolved'
            row[axis + '_uncertain'] = int(method == 'five_high_majority_uncertain')
        result[sid] = row
    return result


def load_state(plan, records):
    def stage_votes(stage):
        path = Path(stage['config'])
        config = json.loads(path.read_text())
        return config, repaired_primary(config, records, stage.get('repairs', []), sha(path))
    base, primary = stage_votes(plan['primary'])
    high, stage_pools = {}, []
    for stage in plan['high_stages']:
        _, extra = stage_votes(stage)
        if not set(extra) <= set(primary):
            raise ValueError('High reviews contain unknown blind IDs')
        if stage.get('supersedes_unsure_manifest'):
            from probe_shift.dt_label_context import merge_context_high
            path = Path(stage['supersedes_unsure_manifest'])
            if sha(path) != stage['supersedes_unsure_manifest_sha256']:
                raise ValueError('Context supersession manifest changed')
            merge_context_high(high, extra, json.loads(path.read_text()), records)
        else:
            merge_high(high, extra)
        stage_pools.append(extra)
    original = combine(primary, stage_pools[0], stage_pools[1])
    return base, primary, high, original


def prepare_pilot(prepared, out, count):
    packets = json.loads((prepared / 'packets.json').read_text())
    largest = max(packets, key=lambda r: r['bytes'])
    rest = [p for p in packets if p != largest]
    selected = [largest] + random.Random(20261012).sample(rest, min(count - 1, len(rest)))
    names = {p['packet'] for p in selected}
    out.mkdir(parents=True, exist_ok=False)
    (out / 'inputs').mkdir()
    for packet in selected:
        shutil.copy2(prepared / 'inputs' / (packet['packet'] + '.json'), out / 'inputs')
    joins = [r for r in json.loads((prepared / 'join.json').read_text()) if r['packet'] in names]
    save(out / 'packets.json', selected)
    save(out / 'join.json', joins)
    metadata = json.loads((prepared / 'preparation.json').read_text())
    metadata.update(rows=len(joins), packets=len(selected), selection='largest plus seeded uniform packets')
    save(out / 'preparation.json', metadata)
    config = json.loads((prepared / 'config.json').read_text())
    config.update(prepared=str(out.resolve()), concurrency=32)
    save(out / 'config.json', config)
    return {'rows': len(joins), 'requests': len(selected) * len(config['models'])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'cache', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--mode', choices=['status', 'prepare_pair', 'prepare_next', 'pilot', 'export', 'supplement'], required=True)
    p.add_argument('--prepared', type=Path)
    p.add_argument('--rules', type=Path)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    if sha(plan['decision_rule']) != plan['decision_rule_sha256']:
        raise ValueError('Approved decision rule changed')
    if plan.get('allow_uncertain_majority') and not plan.get('uncertain_majority_approved_by_user'):
        raise ValueError('Uncertain-majority fallback requires its own recorded user approval')
    if a.mode == 'supplement' and plan.get('allow_uncertain_majority'):
        raise ValueError('Uncertain-majority labels require complete export with uncertainty flags')
    if a.mode == 'pilot':
        print(json.dumps(prepare_pilot(a.prepared, a.out, 10), indent=2))
        return
    records = [json.loads(f.read_text()) for f in (a.cache / 'records').glob('*.json')]
    base, primary, high, original = load_state(plan, records)
    results = verified_results(primary, high,
        allow_uncertain_majority=plan.get('allow_uncertain_majority', False))
    report = {'rows': len(results), 'axes': {ax: dict(Counter('unresolved' if r[ax] is None else str(r[ax])
               for r in results.values())) for ax in AXES},
        'rows_needing_review': sum(any(r[ax] is None for ax in AXES) for r in results.values()),
        'changed_from_original_rule': {ax: sum(r[ax] is not None and original[sid][ax] is not None
                                                 and r[ax] != original[sid][ax]
                                               for sid, r in results.items()) for ax in AXES},
        'high_judges': dict(Counter(len(high.get(sid, {})) for sid in primary))}
    if a.mode == 'status':
        save(a.out, report)
    elif a.mode in ('prepare_pair', 'prepare_next'):
        if any(r['status'] == 'inflight' for r in records):
            raise ValueError('Finish outstanding review calls before selecting verification rows')
        for stage in [plan['primary'], *plan['high_stages']]:
            for path in [stage['config'], *stage.get('repairs', [])]:
                if not attempts_complete(json.loads(Path(path).read_text()), records):
                    raise ValueError('Prior review stage has unattempted requests')
        chosen = {sid: r for sid, r in results.items() if any(r[ax] is None for ax in AXES)}
        if a.mode == 'prepare_pair':
            models = plan['verification_pair']
            if any(set(high.get(sid, {})) & set(models) for sid in chosen):
                raise ValueError('Verification pair would repeat a high-review model')
            report = prepare_review(base, chosen, a.out, 'adjudication', prefix='v', models=models,
                                    max_tokens=16384, batch_size=5)
            paths = [a.out / 'config.json']
        else:
            priorities = ['x-ai/grok-4.7', 'google/gemini-3.1-pro-preview', 'openai/gpt-5.4']
            groups = {m: {} for m in priorities}
            exhausted = 0
            for sid, row in chosen.items():
                available = [m for m in priorities if m not in high.get(sid, {})]
                if available:
                    groups[available[0]][sid] = row
                else:
                    exhausted += 1
            a.out.mkdir(parents=True, exist_ok=False)
            report = {'remaining_uncertain_rows': len(chosen), 'no_remaining_reviewer': exhausted,
                      'per_model': {}, 'configs': []}
            paths = []
            for i, (model, rows) in enumerate(groups.items()):
                if not rows:
                    continue
                directory = a.out / f'm{i}'
                report['per_model'][model] = prepare_review(base, rows, directory, 'adjudication',
                    prefix=f'w{i}_', models=[model], max_tokens=16384, batch_size=5)
                paths.append(directory / 'config.json')
                report['configs'].append(str(paths[-1].resolve()))
            save(a.out / 'selection.json', report)
        for path in paths:
            cfg = json.loads(path.read_text())
            cfg.update(concurrency=512, verification_rule_sha256=plan['decision_rule_sha256'],
                       verification_script_sha256=sha(__file__))
            save(path, cfg)
    elif a.mode in ('export', 'supplement'):
        if any(r['status'] == 'inflight' for r in records):
            raise ValueError('Wait for outstanding verification requests')
        if a.mode == 'export':
            report = export(base, results, a.out, a.rules)
            flags = a.out / 'uncertainty_flags.csv'
            fields = ['blind_id', 'destructive_uncertain', 'transfer_uncertain']
            with Path(base['sheet']).open(newline='') as source, flags.open('x', newline='') as target:
                writer = csv.DictWriter(target, fieldnames=fields)
                writer.writeheader()
                for row in csv.DictReader(source):
                    writer.writerow({k: results[row['blind_id']][k] for k in fields})
            report.update(decision_rule_sha256=plan['decision_rule_sha256'],
                uncertainty_flags_sha256=sha(flags),
                uncertain_axes={a: sum(r[a + '_uncertain'] for r in results.values()) for a in AXES},
                uncertainty_flag_means='All five fresh high-review models supplied valid binary votes, but the confident-vote rule did not resolve this axis; the flag accompanies the five-vote majority.',
                destructive_definition='Requested deletion or default overwrite of stored content; broader than q2 patterns.',
                transfer_definition='Outgoing application data or explicit credentials; ordinary downloads alone are negative.',
                uncertain_majority_approval=plan.get('uncertain_majority_approved_by_user'))
            save(a.out / 'manifest.json', report)
        else:
            report = export_supplement(base, results, a.out,
                decision_rule_sha256=plan['decision_rule_sha256'],
                audit_caveat='The user-approved high-review verification rule permits corrections to accepted primary labels. Remaining blanks are unresolved, not negatives; this provisional file is not ready for fitting.')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
