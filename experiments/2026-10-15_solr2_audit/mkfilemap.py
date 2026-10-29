import json,sys,glob,os,re
out={}
for f in sorted(glob.glob(sys.argv[1]+'/land/scaleup507_su311_q[23]_*.json')):
    d=json.load(open(f)); o=d.get('outputs')
    if not o: continue
    m=re.search(r'/scaleup/([^/]+)/(sol(?:_r2)?)/',o['path']); arm,sub=m.groups()
    out[o['manifest_sha256']]=f'{arm}/{sub} MANIFEST'
    for fn,s in o['files'].items():
        if fn.endswith('.npz'): out[s]=f'{arm}/{sub}/{fn}'
json.dump(out,open(sys.argv[1]+'/filemap.json','w'),indent=1)
print(len(out))
for k,v in out.items():
    if 'q2_main_v1__attack' in v: print(k[:12],v)
