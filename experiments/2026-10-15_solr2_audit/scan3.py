import json,sys,os,re,collections
M=json.load(open(sys.argv[1]))
pref={}
for s,lab in M.items():
    for n in (64,16,12,10,8): pref.setdefault(s[:n],[]).append((lab,n))
for f in sys.argv[2:]:
    t=open(f).read()
    toks=set(re.findall(r'[0-9a-f]{8,64}',t))
    hits=collections.defaultdict(set)
    for tok in toks:
        for n in (64,16,12,10,8):
            if len(tok)>=n and tok[:n] in pref and (len(tok)==n or n==64):
                for lab,nn in pref[tok[:n]]:
                    if nn==n: hits[lab].add(n)
    sub=collections.defaultdict(list)
    for lab,ns in hits.items():
        arm,rest=lab.split('/',1); sd=rest.split('/')[0].split(' ')[0]
        sub[arm+'/'+sd].append((rest.split('/')[-1] if '/' in rest else 'MANIFEST', min(ns)))
    pc=collections.Counter(m.group(0) for m in re.finditer(r'su311_q[23][A-Za-z0-9_]*__[a-z_]+/(sol_r2|sol)(?=[/"\s])', t))
    print('##',os.path.basename(f), 'su311_q2/q3 mentions', len(re.findall(r'su311_q[23]',t)))
    print('   sha hits:', {k:len(v) for k,v in sorted(sub.items())} or 'NONE')
    print('   path strings:', dict(pc) or 'NONE')
