import json,sys
d=json.load(open(sys.argv[1]))['result']
r=d['scenarios']['real']['rows']
f=lambda x:f"{x:+.1f}"
ci=lambda c:f"[{c[0]:+.1f},{c[1]:+.1f}]"
for row,rv in r.items():
    p=rv['variants']['primary_reader_pool_membership']
    print(row,'eligible',rv['eligible_frozen_v6_reader_mechanical'])
    for V in ('A0','A','B'):
        L=p[V]['L']['pooled']; Lr=p[V]['L_rerun']['pooled']
        flag=[k for k in p[V] if k not in ('L','reading','L_rerun')]
        print(f"  {V:2} pooled L {f(L['point'])} 90%{ci(L['ci90'])} 95%{ci(L['ci95'])} | {p[V]['reading']} | L_rerun {f(Lr['point'])} 95%{ci(Lr['ci95'])} | other keys {flag}")
        for pr,v in p[V]['L']['per_probe'].items():
            print(f"      {pr:12} {f(v['point'])} 90%{ci(v['ci90'])} 95%{ci(v['ci95'])} nz={v['nonzero_task_contrasts']}")
