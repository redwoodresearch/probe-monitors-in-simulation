# ON THE BOX: image-level variant of compose_race_check.py: per (project env prefix, kind, compose service), the distinct
# images used across projects; prints only groups with more than one image (counts per image).
import collections, glob, json, re
ev = {}
for f in set(glob.glob("/home/aswin/phase2/runs*/*.docker_events.jsonl") + glob.glob("/home/aswin/phase2/nm/runs/*.docker_events.jsonl")):
    for l in open(f):
        try: e = json.loads(l)
        except ValueError: continue
        if e.get("Type") == "container" and e.get("Action") == "create": ev[e["Actor"]["ID"]] = e
proj = collections.defaultdict(dict)
for e in ev.values():
    a = e["Actor"]["Attributes"]; p = a.get("com.docker.compose.project")
    if p: proj[p][a.get("com.docker.compose.service")] = a.get("image", "")
g = collections.defaultdict(collections.Counter)
for p, svc in proj.items():
    imgs = set(svc.values()); kind = "sim" if any(i.startswith("control-tower-sim-scratch") for i in imgs) else "real" if any(i.startswith("linuxarena/env-") for i in imgs) else "other"
    pre = re.sub(r"^inspect-la-", "", p).rsplit("-", 1)[0]
    for s, i in svc.items(): g[(pre, kind, s)][i] += 1
multi = {f"{k[0]}/{k[1]}/{k[2]}": dict(c) for k, c in g.items() if len(c) > 1}
print(json.dumps({"projects": len(proj), "groups": len(g), "groups_with_more_than_one_image": multi}))
