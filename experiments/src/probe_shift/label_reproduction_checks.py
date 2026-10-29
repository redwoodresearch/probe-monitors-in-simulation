"""Verify archived provenance and regenerated reports without displaying raw inputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from probe_shift.label_reports import joined, select


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for name in ('archive','old-cache','new-cache','new-results','reports','credential-checks','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); b=Path('experiments/2026-10-11_target_touch_labels')
    c=Path('experiments/2026-10-11_credential_count')
    frozen=read(a.archive/b/'config/freeze_v1.json')['files']
    for name,digest in frozen.items(): assert sha(a.archive/name)==digest
    old_blind_sha=sha(a.old_cache/'blind_calls.jsonl')
    assert old_blind_sha==sha(a.new_cache/'blind_calls.jsonl')
    old_sources=read(a.archive/b/'results/v1/source_hashes_v1.json')
    assert old_sources==read(a.new_results/'source_hashes_v2.json')
    v1=read(a.archive/b/'results/v1/audit_results_v1.json')
    reproduced=read(a.reports/'reproduced_v1_audit.json')
    for name in ('by_stratum','by_arm','revision_triggered'): assert v1[name]==reproduced[name]
    data=joined(a.archive/b/'results/v1/target_labels_v1.csv',a.new_results/'replay_join_v2.csv',a.archive/b/'config/runs_phase2_196.csv')
    with (a.archive/b/'results/v1/audit_selection_v1.csv').open() as fh: old_selection=list(csv.DictReader(fh))
    selected=select(data,941317)
    assert [r['id'] for r in selected]==[r['id'] for r in old_selection]
    for name in ('extraction_checks_v1.json','terminal_output_check_v1.json'):
        assert read(a.archive/c/'results/screen_v1'/name)==read(a.credential_checks/name)
    # Check changes are confined to calls assigned an endpoint target, without
    # emitting target aliases or arguments. No individual label is changed here.
    with (a.archive/b/'results/v1/blind_labels.csv').open() as fh: old={r['id']:r for r in csv.DictReader(fh)}
    with (a.new_results/'blind_labels.csv').open() as fh: new={r['id']:r for r in csv.DictReader(fh)}
    changed={i for i in old if any(old[i][k]!=new[i][k] for k in ('touch_any','touch_modify','unclear'))}
    seen=set()
    with (a.new_cache/'blind_calls.jsonl').open() as fh:
        for line in fh:
            call=json.loads(line)
            if call['id'] in changed:
                assert any(t['type']=='endpoint' for t in call['targets'])
                seen.add(call['id'])
    assert seen==changed
    result={'frozen_v1_files_verified':len(frozen),'same_blind_inputs_and_order':True,
            'same_source_archives':True,'v1_audit_selection_reproduced':True,'v1_audit_counts_reproduced':True,
            'credential_extraction_and_terminal_checks_reproduced':True,
            'changed_calls':len(changed),'all_changed_calls_have_endpoint_targets':True,
            'private_blind_input_sha256':old_blind_sha,'producer_sha256':sha(__file__)}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as fh: json.dump(result,fh,indent=2);fh.write('\n')
    print(json.dumps(result))


if __name__=='__main__': main()
