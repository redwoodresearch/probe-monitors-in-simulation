import json,sys
r=json.load(open(sys.argv[1]))['result']; ph=r['posthoc_exclude']
print('tasks_removed',ph['tasks_removed']); print('task_counts',json.dumps(ph['task_counts'])[:400]); print('calibration_unchanged',json.dumps(ph['calibration_unchanged'])[:200])
c=lambda x:f"[{x[0]:+.1f},{x[1]:+.1f}]"
fv=ph['full_vs_reduced']['real']
for row,rv in fv.items():
    cf=rv['configurations']['primary_reader_pool_membership']
    for V in ('A0','A','B'):
        a,b=cf[V]['full'],cf[V]['reduced']
        print(f"{row:32} {V:2} pooled L full {a['L_pooled']['point']:+.1f}{c(a['L_pooled']['ci95'])} [{a['reading'][:22]}] | reduced {b['L_pooled']['point']:+.1f}{c(b['L_pooled']['ci95'])} [{b['reading'][:22]}]")
        for p in a['L_per_probe']:
            pa,pb=a['L_per_probe'][p],b['L_per_probe'][p]
            print(f"      {p:12} full {pa['point']:+.1f}{c(pa['ci95'])}  reduced {pb['point']:+.1f}{c(pb['ci95'])}")
