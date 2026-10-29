"""Report fixed uncertainty codes and resolution gains without exposing command text."""
import argparse
import json
from collections import Counter
from pathlib import Path

from probe_shift.dt_label_finish import votes
from probe_shift.dt_label_panel import AXES, jobs_for, save
from probe_shift.dt_label_prepare import sha
from probe_shift.dt_label_verify import load_state, merge_high, verified_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'config', 'cache', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--supersedes-unsure-manifest', type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    config = json.loads(args.config.read_text())
    records = [json.loads(p.read_text()) for p in (args.cache / 'records').glob('*.json')]
    _, primary, high, _ = load_state(plan, records)
    before = verified_results(primary, high)
    extra = votes(config, records)
    if args.supersedes_unsure_manifest:
        from probe_shift.dt_label_context import merge_context_high
        merge_context_high(high, extra, json.loads(args.supersedes_unsure_manifest.read_text()), records)
    else:
        merge_high(high, extra)
    after = verified_results(primary, high)
    expected = {j['request_id'] for j in jobs_for(config, Path(config['prepared']),
                json.loads(Path(config['catalog']).read_text()), False)}
    selected = [r for r in records if r['request_id'] in expected or r.get('retry_of') in expected]
    if not expected <= {r['request_id'] for r in selected if r['status'] != 'inflight'}:
        raise ValueError('Diagnostic requests are not finished')
    allowed = {'none', 'missing_referenced_code', 'unresolved_runtime_value',
               'missing_environment_state', 'ambiguous_command_semantics',
               'complex_visible_code', 'other', 'potentially_stale_context',
               'partial_context', 'display_normalization'}
    reasons = {axis: Counter() for axis in AXES}
    confidence = {axis: Counter() for axis in AXES}
    for labels in extra.values():
        for label in labels.values():
            for axis in AXES:
                reason = label.get(axis + '_uncertainty_reason')
                consistent = ((reason == 'none') == (label[axis + '_confidence'] == 'sure'))
                valid = isinstance(reason, str) and reason in allowed and consistent
                reasons[axis][reason if valid else 'invalid_reason_code'] += 1
                confidence[axis][label[axis + '_confidence']] += 1
    report = {
        'plan_sha256': sha(args.plan), 'config_sha256': sha(args.config),
        'rows': len(extra), 'rows_with_valid_vote': sum(bool(v) for v in extra.values()),
        'statuses': dict(Counter(r['status'] for r in selected)),
        'providers': dict(Counter(r.get('provider', 'not_reported') for r in selected)),
        'reported_usd': sum(r.get('reported_usd', 0) for r in selected),
        'charge_or_reserve_usd': sum(r['accounted_usd'] for r in selected),
        'newly_resolved_rows': sum(any(before[sid][a] is None for a in AXES)
                                   and all(after[sid][a] is not None for a in AXES) for sid in extra),
        'rows_needing_review_after': sum(any(r[a] is None for a in AXES) for r in after.values()),
        'newly_resolved_axes': {a: sum(before[sid][a] is None and after[sid][a] is not None
                                     for sid in extra) for a in AXES},
        'confidence': {a: dict(c) for a, c in confidence.items()},
        'uncertainty_reasons': {a: dict(c) for a, c in reasons.items()},
        'failures_are_missing_votes': True,
        'supersedes_unsure_manifest_sha256': sha(args.supersedes_unsure_manifest) if args.supersedes_unsure_manifest else None,
    }
    save(args.out, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
