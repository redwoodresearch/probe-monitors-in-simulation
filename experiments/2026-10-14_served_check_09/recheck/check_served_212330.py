import json,sys,re,ast
rec=json.load(open(sys.argv[1])); claims=open(sys.argv[2]).read().splitlines()
res=rec['result']; rows=res['scenarios']['real']['rows']
mrows=res['sensitivity_endpoint_token_matched_units']['result']['rows']
f1=lambda x: f"{x:+.1f}"
NOM="no material change in the family's average signed gap"; MAT="material change in the family's average signed gap"; INC="inconclusive at this n"
def rule(pooled, sat=False):
    if sat: return "A0 catch saturation (descriptive)"
    l90,u90=pooled['ci90']; l95,u95=pooled['ci95']
    if l90>-5 and u90<5: return NOM
    if l95>5 or u95<-5: return MAT
    return INC
nchk=0; mism=[]
def cmp(label, claimed, computed):
    global nchk; nchk+=1
    if claimed!=computed: mism.append((label,claimed,computed))
# --- parse primary block
cur=None; var=None
for ln in claims:
    m=re.match(r"^(\S+\|alpha=\S+) eligible (\[.*\])$", ln)
    if m:
        cur=m.group(1); cmp(f"{cur} eligible", ast.literal_eval(m.group(2)), rows[cur]['eligible_frozen_v6_reader_mechanical']); continue
    m=re.match(r"^  (A0|A|B)\s+pooled L (\S+) 90%\[(\S+),(\S+)\] 95%\[(\S+),(\S+)\] \| (.*?) \| L_rerun (\S+) 95%\[(\S+),(\S+)\] \| other keys (\[.*\])$", ln)
    if m and cur:
        var=m.group(1); V=rows[cur]['variants']['primary_reader_pool_membership'][var]; P=V['L']['pooled']; Rr=V['L_rerun']['pooled']
        base=f"rows/{cur}/variants/primary_reader_pool_membership/{var}"
        comp=[f1(P['point']),f1(P['ci90'][0]),f1(P['ci90'][1]),f1(P['ci95'][0]),f1(P['ci95'][1])]
        for i,(c,k) in enumerate(zip(m.groups()[1:6],comp)): cmp(f"{base}/L/pooled[{i}]",c,k)
        cmp(f"{base}/reading", m.group(7), V['reading'])
        rc=[f1(Rr['point']),f1(Rr['ci95'][0]),f1(Rr['ci95'][1])]
        for i,(c,k) in enumerate(zip(m.groups()[7:10],rc)): cmp(f"{base}/L_rerun/pooled[{i}]",c,k)
        cmp(f"{base} other keys", ast.literal_eval(m.group(11)), [k for k in V if k not in ('L','reading','L_rerun')])
        continue
    m=re.match(r"^      (\S+)\s+(\S+) 90%\[(\S+),(\S+)\] 95%\[(\S+),(\S+)\] nz=(\d+)$", ln)
    if m and cur and var:
        pr=m.group(1); PP=rows[cur]['variants']['primary_reader_pool_membership'][var]['L']['per_probe'][pr]
        base=f"rows/{cur}/primary/{var}/L/per_probe/{pr}"
        comp=[f1(PP['point']),f1(PP['ci90'][0]),f1(PP['ci90'][1]),f1(PP['ci95'][0]),f1(PP['ci95'][1]),str(PP['nonzero_task_contrasts'])]
        for i,(c,k) in enumerate(zip(m.groups()[1:],comp)): cmp(f"{base}[{i}]",c,k)
        continue
    if ln.startswith("SERVED X_err"): sec='x'; cur=None; continue
    if ln.startswith("MATCHED-UNITS"): sec='m'; continue
    m=re.match(r"^(\S+\|alpha=\S+) (\{.*\})$", ln)
    if m:
        key=m.group(1); dct=ast.literal_eval(m.group(2))
        if sec=='x':
            X=res['scenarios']['real']['x_err'][key]
            for lay,(xe,u) in dct.items():
                kx=X[lay]['X_err']; ku=X[lay].get('U_approx_upper_95')
                cmp(f"x_err/{key}/{lay}/X_err", xe, None if kx is None else round(kx,2))
                cmp(f"x_err/{key}/{lay}/U", u, None if ku is None else round(ku,2))
        else:
            for lay,(rd,pt,ci) in dct.items():
                V=mrows[key]['variants']['primary_reader_pool_membership'][lay]; P=V['L']['pooled']
                b=f"matched/{key}/{lay}"
                cmp(b+"/reading", rd, V['reading']); cmp(b+"/point", pt, round(P['point'],2))
                cmp(b+"/ci90", ci, [round(P['ci90'][0],1),round(P['ci90'][1],1)])
print("numbers/fields compared:", nchk, "mismatches:", len(mism))
for x in mism: print("MISMATCH", x)
# --- check 4/5: rule application + rerun flag
print("\n== rule application ==")
for name,R in (("primary",rows),("matched",mrows)):
    for key in R:
        for lay in ("A0","A","B"):
            V=R[key]['variants']['primary_reader_pool_membership'][lay]; P=V['L']['pooled']
            sat=V.get('saturation',{}).get('saturated',False) if lay=="A0" else False
            mine=rule(P,sat); cd=V['L']['complete_draws']
            rr=V['L_rerun']['pooled']['ci95']; incl0 = rr[0]<=0<=rr[1]
            ok = "AGREE" if mine==V['reading'] else "DISAGREE"
            print(f"{name:8s} {key:30s} {lay:2s} ci90=[{P['ci90'][0]:+.3f},{P['ci90'][1]:+.3f}] ci95=[{P['ci95'][0]:+.3f},{P['ci95'][1]:+.3f}] sat={sat} draws={cd} mine='{mine[:14]}' rec='{V['reading'][:14]}' {ok} | rerun95=[{rr[0]:+.3f},{rr[1]:+.3f}] incl0={incl0} flag='{V['rerun_flag']}'")
