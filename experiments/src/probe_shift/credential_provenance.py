"""Reproduce extraction/terminal checks locally; never emit output text or values."""
import argparse
import ast
import csv
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from probe_shift import transcripts
from probe_shift.credential_screen import detect


def function_hashes(path):
    tree=ast.parse(Path(path).read_text())
    return {n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
            for n in tree.body if isinstance(n,ast.FunctionDef)
            and n.name in ('visible_text','tool_result_text','reader_messages')}


def strings(value):
    if isinstance(value,str): yield value
    elif isinstance(value,list):
        for x in value: yield from strings(x)
    elif isinstance(value,dict):
        for x in value.values(): yield from strings(x)


def main():
    p=argparse.ArgumentParser()
    for name in ('manifest','index','plan-reader','archived-reader','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    hashes=function_hashes(a.archived_reader)
    assert hashes==function_hashes(a.plan_reader)==function_hashes(transcripts.__file__)
    with a.manifest.open() as fh: manifest=list(csv.DictReader(fh))
    index=json.loads(a.index.read_text()); wanted=defaultdict(dict)
    assert set(index)=={r['run_id'] for r in manifest}
    for r in manifest:
        _,filename,sid,epoch=r['run_id'].split('|')
        source=Path(index[r['run_id']][0]); assert source.name==filename
        wanted[source][(sid,int(epoch))]=r
    coverage=Counter(samples=0,sample_errors=0,tool_messages_before_submit_filter=0,
                     unresolved_tool_attachments=0,part_attachments=0)
    terminal=defaultdict(lambda:Counter(outputs=0,credential_positives=0)); seen=set()
    for source,requested in wanted.items():
        with zipfile.ZipFile(source) as z:
            for (sid,epoch),r in requested.items():
                sample=json.loads(z.read(f'samples/{sid}_epoch_{epoch}.json'))
                assert str(sample['id'])==sid and int(sample.get('epoch',1))==epoch
                assert r['run_id'] not in seen; seen.add(r['run_id'])
                coverage['samples']+=1; coverage['sample_errors']+=bool(sample.get('error'))
                submit_ids={tc.get('id') for m in sample['messages'] if m['role']=='assistant'
                            for tc in m.get('tool_calls') or [] if tc['function']=='submit'}
                for m in sample['messages']:
                    if m['role']!='tool': continue
                    coverage['tool_messages_before_submit_filter']+=1
                    content=m.get('content')
                    coverage['unresolved_tool_attachments']+=sum(s.startswith('attachment://') for s in strings(content))
                    if isinstance(content,list):
                        coverage['part_attachments']+=sum(x.get('type') not in ('text',) for x in content)
                    if m.get('tool_call_id') in submit_ids or m.get('function')=='submit':
                        terminal[r['arm']]['outputs']+=1
                        terminal[r['arm']]['credential_positives']+=detect(transcripts.tool_result_text(m))[0]
    assert seen==set(index)
    assert coverage['sample_errors']==coverage['unresolved_tool_attachments']==coverage['part_attachments']==0
    a.out.mkdir(parents=True,exist_ok=False)
    reports={'extraction_checks_v1.json':{'functions_identical_to_plan_base_b108df86':list(hashes),
                                          'function_ast_sha256':hashes,'coverage':dict(coverage)},
             'terminal_output_check_v1.json':{'excluded_by_q2_reader':dict(terminal),
                    'positive_counts_for_all_tool_outputs_unchanged':all(x['credential_positives']==0 for x in terminal.values())}}
    for name,obj in reports.items(): (a.out/name).write_text(json.dumps(obj,indent=2)+'\n')
    print(json.dumps({'samples':len(seen),'tool_outputs':coverage['tool_messages_before_submit_filter'],
                      'excluded_positives':sum(x['credential_positives'] for x in terminal.values())}))


if __name__=='__main__': main()
