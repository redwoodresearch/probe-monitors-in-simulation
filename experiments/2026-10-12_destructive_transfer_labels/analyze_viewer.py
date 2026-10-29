"""Build a private human review viewer without printing command contents."""
import argparse
import json
import os
import random
from pathlib import Path
from probe_shift.dt_label_finish import repaired_primary, combine
from probe_shift.dt_label_panel import save
from probe_shift.dt_label_prepare import sha

HTML = '''<!doctype html><meta charset="utf-8"><title>Private blind command review</title>
<style>
body{font:15px system-ui;margin:24px;color:#18232f;background:#fafafa}h1{font-size:24px}
.layout{display:grid;grid-template-columns:330px 1fr;gap:24px}select{width:100%;height:70vh;font:13px monospace}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;border:1px solid #ccd5df;padding:16px}
table{border-collapse:collapse}td,th{border:1px solid #ccd5df;padding:6px 12px;text-align:left}
.note{max-width:1000px;line-height:1.5;color:#435266}button{padding:8px;margin:8px 8px 8px 0}
</style><h1>Private blind command review</h1>
<p class="note">For human inspection only. This file contains command text from the blind bundle.
Nothing is executed and no network requests are made. The agent session has not opened this viewer.
The sample contains up to 50 accepted calls and 200 calls needing extra review, selected with a fixed audit seed.
These are broad semantic labels; agreement is not ground truth. No world, task or run identities are available.</p>
<div class="layout"><select id="rows" size="30"></select><main><h2 id="heading"></h2>
<p id="status"></p><table id="votes"></table><h3>Raw argument string</h3><pre id="args"></pre>
<details><summary>Standardized tool definition</summary><pre id="definition"></pre></details></main></div>
<script id="data" type="application/json">DATA_PLACEHOLDER</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent), select=document.getElementById('rows');
for(const [i,row] of data.entries()){const option=document.createElement('option');option.value=i;
 option.textContent=row.category+' | '+row.tool+' | '+row.blind_id;select.append(option);}
function show(){const row=data[Number(select.value)];
 document.getElementById('heading').textContent=row.blind_id+' — '+row.tool;
 document.getElementById('status').textContent='Primary consensus: destructive='+row.destructive+', transfer='+row.transfer;
 document.getElementById('args').textContent=row.arguments;
 document.getElementById('definition').textContent=JSON.stringify(row.definition,null,2);
 const table=document.getElementById('votes');table.replaceChildren();
 const values=[['Judge','Destructive','Confidence','Transfer','Confidence'],...Object.entries(row.votes).map(([m,v])=>
 [m,v.destructive,v.destructive_confidence,v.transfer,v.transfer_confidence])];
 values.forEach((v,i)=>{const tr=document.createElement('tr');for(const x of v){const cell=document.createElement(i?'td':'th');
 cell.textContent=x;tr.append(cell);}table.append(tr);});}
select.addEventListener('change',show);select.value='0';if(data.length)show();
</script>'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','cache','commands','definitions','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--repair-config',type=Path,action='append',default=[])
    a=p.parse_args();os.umask(0o077)
    cfg=json.loads(a.config.read_text());metadata=json.loads((Path(cfg['prepared'])/'preparation.json').read_text())
    if sha(a.commands)!=metadata['commands_sha256'] or sha(a.definitions)!=metadata['definitions_sha256']:
        raise ValueError('Bundle hash mismatch')
    records=[json.loads(f.read_text()) for f in (a.cache/'records').glob('*.json')]
    primary=repaired_primary(cfg,records,a.repair_config,sha(a.config));complete=combine(primary)
    groups={name:sorted(sid for sid,r in complete.items() if any(r[ax] is None for ax in ('destructive','transfer'))==flag)
            for name,flag in [('accepted',False),('review',True)]}
    chosen={sid:name for name,ids in groups.items() for sid in random.Random(cfg['seed']).sample(ids,min(len(ids),50 if name=='accepted' else 200))}
    definitions={r['tool_def_id']:r['definition'] for r in map(json.loads,a.definitions.read_text().splitlines())}
    rows=[]
    with a.commands.open() as f:
        for line in f:
            r=json.loads(line);sid=r['blind_id']
            if sid in chosen:
                rows.append({'blind_id':sid,'category':chosen[sid],'tool':r['tool'],'arguments':r['arguments'],
                    'definition':definitions[r['tool_def_id']],'votes':primary[sid],
                    'destructive':complete[sid]['destructive'],'transfer':complete[sid]['transfer']})
    rows.sort(key=lambda r:(r['category'],r['blind_id']))
    # Escape script terminators; displayed data uses textContent, never HTML injection.
    payload=json.dumps(rows,ensure_ascii=True).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    a.out.mkdir(parents=True,exist_ok=False);viewer=a.out/'index.html'
    viewer.write_text(HTML.replace('DATA_PLACEHOLDER',payload));viewer.chmod(0o600)
    report={'rows':len(rows),'accepted':sum(r['category']=='accepted' for r in rows),
            'review':sum(r['category']=='review' for r in rows),'viewer_sha256':sha(viewer),
            'human_review_performed':False,'agent_session_read_command_text':False}
    save(a.out/'summary.json',report);print(json.dumps(report))


if __name__=='__main__':main()
