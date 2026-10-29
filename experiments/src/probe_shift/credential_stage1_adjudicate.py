"""Freeze matching independent reviews of unresolved credential outputs."""
import argparse
import hashlib
import json
from pathlib import Path
from probe_shift.label_panel_results import aggregate, apply_adjudication, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=('credential', 'credential_calibration'), default='credential')
    p.add_argument('--primary-prepared', type=Path, required=True)
    p.add_argument('--adjudication-prepared', type=Path, required=True)
    p.add_argument('--primary-config', type=Path, required=True)
    p.add_argument('--adjudication-config', type=Path, required=True)
    p.add_argument('--final-prepared', type=Path)
    p.add_argument('--final-config', type=Path)
    p.add_argument('--ledger', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    primary, *_ = aggregate(a.primary_prepared, a.ledger, a.kind, json.loads(a.primary_config.read_text())['models'])
    models = json.loads(a.adjudication_config.read_text())['models']
    assert len(set(models)) == len(models) == 2
    ledger = json.loads(a.ledger.read_text())
    votes = {}
    for entry in ledger.values():
        if entry['dataset'] != a.kind + '_adjudication' or entry['status'] != 'ok' or entry['model'] not in models:
            continue
        for row in entry['labels']:
            key = entry['packet'], row['id']
            assert entry['model'] not in votes.setdefault(key, {}), 'Duplicate successful review'
            assert type(row['label']) is int and row['label'] in (0, 1)
            assert row.get('reason') and row['confidence'] in ('sure', 'unsure')
            votes[key][entry['model']] = row
    join = json.loads((a.adjudication_prepared / 'join.json').read_text())
    packets = json.loads((a.adjudication_prepared / 'packets.json').read_text())
    by_id = {cid: packet['packet'] for packet in packets for cid in packet['ids']}
    assert len(by_id) == sum(len(packet['ids']) for packet in packets)
    final_reviews = {}
    if a.final_prepared:
        final_models = json.loads(a.final_config.read_text())['models']
        assert len(final_models) == 1
        final_join = json.loads((a.final_prepared / 'join.json').read_text())
        final_packets = json.loads((a.final_prepared / 'packets.json').read_text())
        final_ids = {cid: packet['packet'] for packet in final_packets for cid in packet['ids']}
        final_votes = {}
        for entry in ledger.values():
            if entry['dataset'] == a.kind + '_final_adjudication' and entry['status'] == 'ok' and entry['model'] in final_models:
                for row in entry['labels']:
                    key = entry['packet'], row['id']
                    assert key not in final_votes
                    assert type(row['label']) is int and row['label'] in (0, 1) and row.get('reason')
                    final_votes[key] = row | {'model': entry['model']}
        final_reviews = {r['adjudication_id']: final_votes[final_ids[r['id']], r['id']] for r in final_join}
    decisions = []
    disagreements = set()
    for row in join:
        reviews = votes.get((by_id[row['id']], row['id']), {})
        assert set(reviews) == set(models), 'Missing adjudication reviewer'
        labels = {r['label'] for r in reviews.values()}
        decision = {'packet': row['original_packet'], 'id': row['original_id'], 'reviews': reviews}
        if len(labels) == 1:
            decision['label'] = labels.pop()
        else:
            disagreements.add(row['id'])
            assert row['id'] in final_reviews, 'Reviewers disagree; explicit final adjudication required'
            decision.update(label=final_reviews[row['id']]['label'], final_review=final_reviews[row['id']])
        decisions.append(decision)
    assert set(final_reviews) == disagreements, 'Final adjudication must cover exactly the disagreements'
    apply_adjudication(primary, decisions)
    result = {'primary_consensus_sha256': hashlib.sha256((json.dumps(primary, indent=2) + '\n').encode()).hexdigest(),
              'ledger_sha256': sha(a.ledger), 'adjudication_config_sha256': sha(a.adjudication_config),
              'decisions': decisions}
    if a.final_config:
        result['final_adjudication_config_sha256'] = sha(a.final_config)
    with a.out.open('x') as f:
        f.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'adjudicated_outputs': len(decisions), 'final_adjudications': len(disagreements)}))


if __name__ == '__main__':
    main()
