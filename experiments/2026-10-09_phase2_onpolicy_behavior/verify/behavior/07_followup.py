import json, collections, re
from pathlib import Path
from probe_shift.viewer.pairs import read_launch, read_sample, steps, _log
RD = Path.home() / "phase2/runs/2026-10-09_000523_rerun_strict"
L = read_launch(RD)
def find(d, pat, path=""):
    if isinstance(d, dict):
        for k, v in d.items(): yield from find(v, pat, f"{path}.{k}")
    elif isinstance(d, list):
        for i, v in enumerate(d[:3]): yield from find(v, pat, f"{path}[{i}]")
    if re.search(pat, path.split(".")[-1] if path else "", re.I) and not isinstance(d, (dict, list)):
        yield path, d
for k in ("pilot_config", "config", "sim_config", "policy"):
    for p, v in find(L.get(k), r"step|limit|max_|turn|message"):
        print("CFG", k + p, "=", str(v)[:80])
runs = {r["key"]: r for r in L["runs"]}
print("attempt keys:", sorted(L["runs"][0]["attempts"][-1].keys()))
for key in ["sim_honest@today/t01_media_processing", "sim_honest@september/t10_shared_dotfiles"]:
    s = read_sample(_log(RD, runs[key])["eval"])
    e = s.get("error") or {}
    print("ERRRUN", key, "n_msgs", len(s["messages"]), "error.message:", str(e.get("message") if isinstance(e, dict) else e)[:200])
    print("   scores:", json.dumps(s.get("scores"))[:200])
    print("   attempts:", [(a.get("kind"), a.get("state") or a.get("status"), str(a.get("stop_reason") or a.get("reason"))[:60]) for a in runs[key]["attempts"]])
# sim honest scores explanation
n = 0
for key, r in runs.items():
    if r["arm"] != "sim_honest": continue
    s = read_sample(_log(RD, r)["eval"])
    sc = (s.get("scores") or {}).get("linux_scorer")
    if sc and n < 2:
        print("SIMHONEST", key, "value:", sc.get("value"), "| explanation:", str(sc.get("explanation"))[:160].replace("\n", " "), "| metadata keys:", list((sc.get("metadata") or {}).keys())[:5]); n += 1
# runs ending without submit: steps and last tool error / output length
for key, r in runs.items():
    s = read_sample(_log(RD, r)["eval"])
    _, st = steps(s)
    calls = [c for step in st for c in (step[0].get("tool_calls") or [])]
    if not calls or calls[-1]["function"] != "submit":
        print("NOSUBMIT", key, "steps", len(st), "error", bool(s.get("error")), "limit", s.get("limit"), "attempt kinds", [a.get("kind") for a in r["attempts"]])
# tool error types per group
C = collections.Counter()
for key, r in runs.items():
    s = read_sample(_log(RD, r)["eval"])
    g = r["arm"] + ("@" + r["simulator"] if r["arm"].startswith("sim") else "")
    for m in s["messages"]:
        if m["role"] == "tool" and m.get("error"):
            C[(g, m["error"].get("type"))] += 1
print("tool error types:", sorted(C.items()))
