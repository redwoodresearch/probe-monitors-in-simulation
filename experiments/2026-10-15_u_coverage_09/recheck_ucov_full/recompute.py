import json, sys
import numpy as np
from scipy.stats import beta
f=json.load(open(sys.argv[1]))
r=f['result']
def cp(k,n):
    lo=0.0 if k==0 else beta.ppf(0.025,k,n-k+1)
    hi=1.0 if k==n else beta.ppf(0.975,k+1,n-k)
    return lo,hi
tot=0; undef=0; nonnull=[]; maxdiff=0
for key,ds in r['datasets'].items():
    tot+=len(ds)
    X=np.array([d['X_err'] if d['X_err'] is not None else np.nan for d in ds],float)
    U=np.array([d['U'] if d['U'] is not None else np.nan for d in ds],float)
    th=np.array([d['theta_That'] for d in ds],float); ts=np.array([d['theta_Tstar'] for d in ds],float)
    nu=int(np.isnan(X).sum()+np.isnan(U).sum()); undef+=nu
    ids=[d['i'] for d in ds]; assert sorted(ids)==list(range(len(ds))),key
    k1=int((U>=th).sum()); n=len(ds); k2=int((U>=ts).sum())
    lo,hi=cp(k1,n); lo2,hi2=cp(k2,n)
    s=r['summary'][key]
    p=s['coverage_primary_theta_That']; q=s['coverage_secondary_theta_Tstar']
    match = (p['k']==k1 and q['k']==k2 and abs(p['cp95'][0]-lo)<1e-9 and abs(p['cp95'][1]-hi)<1e-9 and abs(q['cp95'][0]-lo2)<1e-9)
    xs=dict(Xmean=X.mean(), Umean=U.mean(), Up95=np.percentile(U,95), pos=(X>0).mean(), thHat=th.mean(), thStar=ts[0], tsconst=bool(np.all(ts==ts[0])))
    # compare against summary
    d1=max(abs(xs['Xmean']-s['X_err']['mean']),abs(xs['Umean']-s['U']['mean']),abs(xs['pos']-s['share_X_err_positive']),abs(xs['thHat']-s['theta_That']['mean']))
    maxdiff=max(maxdiff,d1)
    tag='NULL' if 'a_null' in key else ''
    if not tag: nonnull.append((k1/n,key,lo,hi))
    print(f"{key:55s} n={n} undef={nu} prim {k1}/{n}={k1/n:.3f} [{lo:.3f},{hi:.3f}] sec {k2}/{n}={k2/n:.3f} [{lo2:.3f},{hi2:.3f}] match={match} | th*={xs['thStar']:.2f} thHat={xs['thHat']:.2f} Xmean={xs['Xmean']:.2f} pos={xs['pos']:.3f} Umean={xs['Umean']:.2f} Up95={xs['Up95']:.2f} {tag}")
print('total',tot,'undefined',undef,'summary max diff',maxdiff)
nonnull.sort()
print('non-null cells',len(nonnull),'at 1.000:',sum(1 for c in nonnull if c[0]==1.0),'lowest',nonnull[0])
