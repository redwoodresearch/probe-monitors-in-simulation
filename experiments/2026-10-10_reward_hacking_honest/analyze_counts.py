"""Publish ID-only counts after the seeded candidate audit has been reviewed."""
import argparse,json
from collections import defaultdict
from pathlib import Path

def main(root,out):
    audit=json.loads((out.parent/'audit_v4.json').read_text());assert audit['syntax_review_passed']
    calls=[json.loads(l) for l in (root/'labels.jsonl').read_text().splitlines()];runs=[json.loads(l) for l in (root/'runs.jsonl').read_text().splitlines()]
    cohorts=defaultdict(list)
    for r in runs:cohorts[r['cohort']].append(r)
    def summarize(rs,cs):
        result={'runs':len(rs),'tasks':len({r['main_task'] for r in rs}),'calls':len(cs),'runs_with_sample_error':sum(r['sample_error'] for r in rs)}
        for kind in ['test_edit','test_disable','forced_success','test_cleanup','strong_union']:
            hits=[c for c in cs if (bool(set(c['labels'])&{'test_disable','forced_success'}) if kind=='strong_union' else kind in c['labels'])]
            result[kind]={'calls':len(hits),'runs':len({r['run'] for r in hits}),'tasks':len({r['main_task'] for r in hits}),'per_1000_calls':1000*len(hits)/len(cs) if cs else None}
        return result
    result={'cohorts':{k:summarize(rs,[c for c in calls if c['cohort']==k]) for k,rs in cohorts.items()},'common_tasks':{}}
    comparisons=[('qwen_honest_a','qwen_sim_honest_september'),('qwen_honest_a','qwen_sim_honest_today'),('sol_real','sol_sim')]
    for a,b in comparisons:
        common={r['main_task'] for r in cohorts[a]}&{r['main_task'] for r in cohorts[b]}
        result['common_tasks'][a+'_vs_'+b]={'tasks':len(common),**{arm:summarize([r for r in cohorts[arm] if r['main_task'] in common],[r for r in calls if r['cohort']==arm and r['main_task'] in common]) for arm in [a,b]}}
    result['missing']=json.loads((root/'inventory.json').read_text())['missing'];result['provenance']=json.loads((root/'provenance.json').read_text());result['audit']=audit
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:{'runs':v['runs'],'tasks':v['tasks'],'calls':v['calls'],**{j:v[j] for j in ['test_edit','test_disable','forced_success','strong_union']}} for k,v in result['cohorts'].items()},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.input,a.output)
