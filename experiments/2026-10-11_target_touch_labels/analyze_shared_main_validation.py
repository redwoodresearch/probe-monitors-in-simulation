"""Verify the new column using ID-only artifacts, independently of raw-call processing."""
import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path


def require(ok, message):
    if not ok: raise ValueError(message)


def rows(path):
    with path.open() as fh: return list(csv.DictReader(fh))


def main():
    p=argparse.ArgumentParser()
    for name in ('v2','v3','runs','replay','summary','out'):
        p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();before,after=rows(a.v2),rows(a.v3)
    columns=list(before[0]);flag='shared_with_main_only'
    require(list(after[0])==columns+[flag],'Unexpected schema')
    require(len(before)==len(after),'Changed row count')
    reconstructed=io.StringIO(newline='')
    writer=csv.DictWriter(reconstructed,fieldnames=columns,lineterminator='\n')
    writer.writeheader()
    for old,new in zip(before,after):
        original={k:new[k] for k in columns}
        require(old==original,'Changed existing ID or label')
        require(new[flag] in ('0','1'),'Nonbinary flag')
        require(int(new[flag])<=int(new['touch_any']),'Negative call excluded')
        writer.writerow(original)
    require(reconstructed.getvalue().encode()==a.v2.read_bytes(),'v2 bytes did not reconstruct')
    keys=['run_id','epoch','turn','call_index']
    replay={tuple(r[k] for k in ['run_id','inspect_epoch','turn','call_index']):r for r in rows(a.replay)}
    manifest=rows(a.runs);arms={r['run_id']:r['arm'] for r in manifest}
    counts=defaultdict(Counter);seen=set()
    for row in after:
        key=tuple(row[k] for k in keys);require(key not in seen,'Duplicate call ID');seen.add(key)
        excluded=int(row[flag]);positive=int(row['touch_any']);kept=1-excluded
        replay_excluded=int(replay[key]['q2_excluded'])
        counts[arms[row['run_id']]].update({
            'all_calls':1,'unfiltered_positives':positive,'excluded_shared_only':excluded,
            'main_calls':kept,'main_positives':positive*kept,
            'main_q2_excluded':replay_excluded*kept,
            'main_positive_q2_excluded':replay_excluded*kept*positive})
    require(seen==set(replay),'Incomplete replay join')
    summary=json.loads(a.summary.read_text())
    require(dict(counts)==summary['by_arm'],'Arm counts disagree')
    require(set(arms)=={r['run_id'] for r in after},'Missing run')
    result={'rows':len(after),'runs':len(arms),'v2_byte_reconstruction':True,
            'all_flags_binary':True,'negative_calls_excluded':0,'replay_join_coverage':len(seen),
            'independent_arm_counts_agree':True,
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'input_hashes':{name:hashlib.sha256(getattr(a,name).read_bytes()).hexdigest()
                            for name in ('v2','v3','runs','replay','summary')}}
    with a.out.open('x') as fh: json.dump(result,fh,indent=2);fh.write('\n')
    print(json.dumps({'validated_rows':len(after),'v2_bytes_unchanged':True}))


if __name__=='__main__': main()
