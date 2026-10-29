import json,sys,re
d=json.load(open(sys.argv[1])); pat=re.compile(sys.argv[2]); n=int(sys.argv[3]) if len(sys.argv)>3 else 12
out=[]
def walk(o,p):
    if len(out)>=n: return
    if isinstance(o,dict):
        for k,v in o.items():
            if pat.search(str(k)): out.append(('KEY',p+'.'+str(k)))
            walk(v,p+'.'+str(k))
    elif isinstance(o,list):
        for i,v in enumerate(o): walk(v,p+'[%d]'%i)
    elif isinstance(o,str) and pat.search(o): out.append((p,o[:220]))
walk(d,'')
for x in out: print(x)
