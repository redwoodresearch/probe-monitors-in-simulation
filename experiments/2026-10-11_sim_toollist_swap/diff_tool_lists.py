"""Counts, names and hashes only: the tools each Phase 2 arm's acting-model requests offered (call.request.tools, what
vLLM was served), per run. No message text is read: only the structured `tools` field of model events."""
import hashlib, json, sys, zipfile
from collections import Counter, defaultdict
from pathlib import Path

root = Path(sys.argv[1]); runs = sys.argv[2].split(","); acting = "vllm/Qwen/Qwen3.5-27B"
h = lambda o: hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()[:12]  # noqa: E731
out = {"acting_model": acting, "per_arm": {}}
per_arm = defaultdict(Counter); examples = {}; within = Counter(); n_events = Counter()
for run in runs:
    for ev in sorted((root / run).glob("*/*/*.eval")):
        arm = ev.relative_to(root / run).parts[0]
        with zipfile.ZipFile(ev) as z:
            names = [n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")]
            if not names:
                continue
            s = json.loads(z.read(names[0]))
        fps = set()
        for e in s.get("events") or []:
            if e.get("event") != "model" or e.get("model") != acting:
                continue
            tools = (((e.get("call") or {}).get("request") or {}).get("tools")) or []
            fn = [t.get("function", {}) for t in tools]
            fp = {"n": len(fn), "names": [f.get("name") for f in fn], "list": h(fn),
                  "per_tool": {f.get("name"): {"description": h(f.get("description")), "parameters": h(f.get("parameters")),
                                               "param_names": sorted((f.get("parameters") or {}).get("properties", {})),
                                               "required": sorted((f.get("parameters") or {}).get("required", [])),
                                               "strict": f.get("strict")} for f in fn}}
            fps.add(json.dumps(fp, sort_keys=True)); n_events[f"{run}/{arm}"] += 1
        within[f"{run}/{arm}: runs with {len(fps)} distinct tool lists"] += 1
        for fp in fps:
            per_arm[f"{run}/{arm}"][json.loads(fp)["list"]] += 1
            examples[json.loads(fp)["list"]] = json.loads(fp)
out["per_arm"] = {k: dict(v) for k, v in per_arm.items()}
out["within_run"] = dict(within); out["acting_events"] = dict(n_events)
out["tool_lists"] = examples
print(json.dumps(out, indent=1))
