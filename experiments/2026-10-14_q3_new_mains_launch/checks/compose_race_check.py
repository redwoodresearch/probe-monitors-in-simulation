"""Read-only compose-race check (d9, 10-15 ~00:05 PT): concurrent launchers on one box share ~/phase2/settings, and real
and simulated runs both write an environment's generated compose file (run_pilot.py:20/450), while the 90 s stagger only
orders launches inside one launcher. Did any run start containers from a compose file other than its own kind's?
ON THE BOX, python3 only. Evidence: every Docker container-create event in every <run>.docker_events.jsonl on the box
(box-wide logs; deduplicated by container id). Per compose project (one per sample sandbox):
  kind = "sim" if it created control-tower-sim-scratch, else "real" if it created a linuxarena/env-* image;
  per (environment prefix of the project name, kind, compose service): the compose config-hash label must be the same
  for every project; a project whose hash differs from the others' is a MISMATCH; a project with both an env image and
  the scratch image is MIXED; a real project's env image tag must equal config/run.yaml's env_commits for that env.
Then each real attempt of the given launch(es) is mapped to its project (the first not-yet-matched env-image create of its
environment at or after its start, as compose_latency.py) and labelled clean / mismatch / mixed / bad_tag / unmatched.
Keys, labels, hashes and counts only.
  python3 compose_race_check.py <harness dir> <run dir> [<run dir> ...]"""
import collections, glob, json, re, sys
from datetime import datetime
from pathlib import Path

harness, runs = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
y = (harness / "config" / "run.yaml").read_text()
commits = dict(re.findall(r"^\s+([a-z_]+): ([0-9a-f]{40})\b", y, re.M))
events = {}
for f in sorted(set(glob.glob("/home/aswin/phase2/runs/*.docker_events.jsonl") + glob.glob("/home/aswin/phase2/nm/runs/*.docker_events.jsonl")
                    + glob.glob("/home/aswin/phase2/runs*/*.docker_events.jsonl"))):
    for line in open(f):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("Type") == "container" and e.get("Action") == "create":
            events[e["Actor"]["ID"]] = e
projects = collections.defaultdict(lambda: {"svc": {}, "images": set(), "t0": None})
for e in events.values():
    a = e["Actor"]["Attributes"]
    p = a.get("com.docker.compose.project")
    if not p:
        continue
    d = projects[p]
    d["svc"][a.get("com.docker.compose.service")] = a.get("com.docker.compose.config-hash")
    d["images"].add(a.get("image", ""))
    t = e["timeNano"] / 1e9
    d["t0"] = t if d["t0"] is None else min(d["t0"], t)
def env_of(images):
    envs = {i.split("/env-")[1].split(":")[0] for i in images if i.startswith("linuxarena/env-")}
    return envs.pop() if len(envs) == 1 else None
info = {}
for p, d in projects.items():
    scratch = any(i.startswith("control-tower-sim-scratch") for i in d["images"])
    env = env_of(d["images"])
    kind = "sim" if scratch else "real" if env else "other"
    prefix = re.sub(r"^inspect-la-", "", p).rsplit("-", 1)[0]
    bad_tag = bool(env) and not scratch and not any(i == f"linuxarena/env-{env}:{commits.get(env)}" for i in d["images"])
    info[p] = {"kind": kind, "prefix": prefix, "env": env, "mixed": scratch and env is not None, "bad_tag": bad_tag, "t0": d["t0"]}
groups = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
for p, d in projects.items():
    for svc, h in d["svc"].items():
        groups[(info[p]["prefix"], info[p]["kind"])][svc][h] += 1
for p, d in projects.items():
    g = groups[(info[p]["prefix"], info[p]["kind"])]
    info[p]["mismatch"] = any(len(g[s]) > 1 and h != g[s].most_common(1)[0][0] for s, h in d["svc"].items())
summary = {"events": len(events), "projects": dict(collections.Counter(v["kind"] for v in info.values())),
           "groups_with_more_than_one_hash": sorted(f"{k[0]}/{k[1]}/{s}: {dict(c)}" for k, g in groups.items() for s, c in g.items() if len(c) > 1),
           "mixed": sorted(p for p, v in info.items() if v["mixed"]), "bad_tag": sorted(p for p, v in info.items() if v["bad_tag"]),
           "mismatch": sorted(p for p, v in info.items() if v["mismatch"]), "env_commits": len(commits)}
real_by_env = collections.defaultdict(list)
for p, v in info.items():
    if v["kind"] == "real":
        real_by_env[v["env"]].append((v["t0"], p))
for v in real_by_env.values():
    v.sort()
per_run = {}
for run in runs:
    d = json.loads((run / "launch.json").read_text())
    starts = []
    for r in d["runs"]:
        if not r["real"]:
            continue
        env = r["task"].split("_", 1)[1].removesuffix("_r2")
        for i, a in enumerate(r["attempts"]):
            if a.get("start"):
                starts.append((datetime.fromisoformat(a["start"]).timestamp(), env, f"{r['key']}#{i}"))
    starts.sort()
    used, labels = collections.defaultdict(int), {}
    for t, env, key in starts:
        xs, i = real_by_env.get(env, []), used[env]
        while i < len(xs) and xs[i][0] < t - 1:
            i += 1
        if i < len(xs) and xs[i][0] - t < 900:
            v = info[xs[i][1]]
            labels[key] = "mixed" if v["mixed"] else "bad_tag" if v["bad_tag"] else "mismatch" if v["mismatch"] else "clean"
            used[env] = i + 1
        else:
            labels[key] = "unmatched"
    c = collections.Counter(labels.values())
    per_run[run.name] = {"attempts": len(labels), "labels": dict(c), "not_clean": sorted(k for k, l in labels.items() if l not in ("clean", "unmatched")),
                         "unmatched": sorted(k for k, l in labels.items() if l == "unmatched")[:20], "labels_by_key": labels}  # addendum: every label
print(json.dumps({"box_summary": summary, "per_run": per_run}))
