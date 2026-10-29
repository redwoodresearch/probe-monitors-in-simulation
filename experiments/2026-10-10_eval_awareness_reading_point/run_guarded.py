"""Launch one approved stage and fail closed on billing failures or aggregate budget exhaustion."""
import argparse,datetime,fcntl,json,os,re,selectors,subprocess,time
from pathlib import Path

BASE=['uv','run','--offline','--no-sync','modal']
HERE=Path(__file__).resolve().parent

def cli(args):
    r=subprocess.run(BASE+args,capture_output=True,text=True,timeout=45)
    if r.returncode:raise RuntimeError('Modal command failed: '+args[0]+' '+args[1])
    return r.stdout

def stop_app(app_id):
    try:cli(['app','stop',app_id,'--yes'])
    except RuntimeError:
        # Stopping an already terminated failed app can return an error; verify its state.
        states=json.loads(cli(['app','list','--json']))
        if not any(a['app_id']==app_id and a['state']=='stopped' and str(a['tasks'])=='0' for a in states):raise

def billing_window(history, today):
    earliest=min([datetime.datetime.fromisoformat(r['start_utc']).date() for r in history['stages']]+[today])
    return (earliest-datetime.timedelta(days=1)).isoformat(),(today+datetime.timedelta(days=2)).isoformat()

def main(step,ledger,project,run):
    BASE[2:2]=['--project',str(project)]
    if step in ('calibration','confirmation') and not (HERE/'results/early_probe_freeze.json').exists():
        raise RuntimeError('Scoring requires the local frozen-probe record before deployment')
    history=json.loads(ledger.read_text()) if ledger.exists() else {'cap_usd':8,'stop_at_usd':7.2,'stages':[]}
    start=datetime.datetime.now(datetime.timezone.utc).date()
    begin,end=billing_window(history,start)
    prior=[r['app_id'] for r in history['stages'] if r.get('app_id')]
    def cost(ids):
        records=json.loads(cli(['billing','report','--start',begin,'--end',end,'--json']));return sum(float(r['cost']) for r in records if r['object_id'] in ids)
    previous=cost(prior);reserve={'extract':1.50,'fit':.35,'calibration':2.85,'confirmation':2.85}[step]
    if previous+reserve>history['stop_at_usd']:raise RuntimeError('Insufficient remaining approved budget for bounded stage')
    stage={'step':step,'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'app_id':None,'status':'launching','prior_cost':previous,'reserved_upper_estimate':reserve};history['stages'].append(stage);ledger.write_text(json.dumps(history,indent=2)+'\n')
    env=os.environ.copy();env['PYTHONPATH']=str(HERE.parents[1]/'src');env.setdefault('UV_CACHE_DIR',str(Path.home()/'.cache/uv'))
    command=BASE+['run','--detach',str(HERE/'run_pilot.py'),'--run',run,'--step',step]
    if step in ('calibration','confirmation'):
        freeze=json.loads((HERE/'results/early_probe_freeze.json').read_text())
        if freeze.get('run',run)!=run:raise ValueError('Frozen probe belongs to a different run')
        command+=['--probe-sha256',freeze['probe_sha256']]
    p=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env,bufsize=0);sel=selectors.DefaultSelector();sel.register(p.stdout,selectors.EVENT_READ);last=0;errors=0;seen='';launch=time.monotonic();log=(HERE/'results'/f'guard_{step}_{int(time.time())}.log').open('wb')
    def stop(reason):
        stage['status']='stopped';stage['reason']=reason;ledger.write_text(json.dumps(history,indent=2)+'\n')
        if stage['app_id']:
            stop_app(stage['app_id'])
            for _ in range(12):
                containers=json.loads(cli(['container','list','--json']))
                if not any(c.get('app_id')==stage['app_id'] for c in containers):break
                time.sleep(5)
            else:raise RuntimeError('Stop requested but containers remain')
        p.terminate();raise RuntimeError(reason)
    while True:
        for key,_ in sel.select(timeout=1):
            data=os.read(key.fd,65536)
            if data:
                log.write(data);log.flush();text=data.decode(errors='replace');print(text,end='',flush=True);seen=(seen+text)[-10000:]
                ids=re.findall(r'ap-[A-Za-z0-9]+',seen)
                if ids and stage['app_id'] is None:
                    stage['app_id']=ids[0];stage['status']='running';ledger.write_text(json.dumps(history,indent=2)+'\n')
            else:sel.unregister(key.fileobj)
        if stage['app_id'] and time.monotonic()-last>=15:
            try:total=cost(prior+[stage['app_id']])
            except (RuntimeError,subprocess.TimeoutExpired,json.JSONDecodeError):
                errors+=1
                if errors>=3:stop('Billing unreadable on three consecutive polls')
            else:
                errors=0;stage['aggregate_cost_latest']=total;ledger.write_text(json.dumps(history,indent=2)+'\n');print(f'BUDGET: aggregate ${total:.4f} / $8; stop at $7.20',flush=True)
                if total>=history['stop_at_usd']:stop('Aggregate spend reached conservative stop threshold')
            last=time.monotonic()
        if p.poll() is not None and not sel.get_map():break
        if time.monotonic()-launch>1800:stop('Stage wall time exceeded 30 minutes')
    if stage['app_id']:stop_app(stage['app_id'])
    stage['exit_code']=p.returncode;stage['status']='complete' if p.returncode==0 else 'failed';stage['final_aggregate_cost']=cost(prior+[stage['app_id']]);stage['end_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();ledger.write_text(json.dumps(history,indent=2)+'\n');log.close()
    print(json.dumps(stage,indent=2));raise SystemExit(p.returncode)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--step',choices=['extract','fit','calibration','confirmation'],required=True)
    ap.add_argument('--project',type=Path,default=HERE.parents[1]);ap.add_argument('--run',default='start_2026-10-10_codex_v1')
    a=ap.parse_args();ledger=HERE/'results/budget_scoring_ledger.json'
    # One ledger and OS lock cover all stages and concurrent launch attempts for this pilot.
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main(a.step,ledger,a.project,a.run)
