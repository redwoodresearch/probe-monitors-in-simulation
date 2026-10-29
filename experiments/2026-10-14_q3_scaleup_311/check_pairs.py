"""The interleaved second attack's checks (experiments/2026-10-14_q3_scaleup_311/RULE.md), ON THE DEVBOX, read-only.
For every cell whose two real attacks (attack/<key> and attack/<key>_r2) are both done in this launch:
  - same configuration: their first attempts' env vars are equal except VLLM_BASE_URL (the server each task drew), and
    their commands are equal except --log-dir;
  - not a replay: sha256 of the first assistant reply and of the full message list (model_dump JSON), per run, compared;
    hashes only, no text is printed or written.
Tool-call ids (vLLM draws them at random) are replaced by their order of first appearance before hashing, in assistant
tool_calls and in tool messages' tool_call_id, so identical generated content is caught. Exit 2 if any pair is
byte-identical (pause the launch and tell d9), 3 on a configuration mismatch, 0 if every pair checked passes; any other
exit (a crash reading a pair) is an error the caller counts toward a fail-closed stop.
  PYTHONPATH=<tree>/src <sept venv python> check_pairs.py <run dir> <out json> [--first N]
"""

import argparse
import hashlib
import json
from pathlib import Path

from inspect_ai.log import read_eval_log

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--first", type=int, default=0, help="only the first N pairs to finish (0 = all)")
a = ap.parse_args()
rec = json.loads((a.run / "launch.json").read_text())
runs = {r["key"]: r for r in rec["runs"]}
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()  # noqa: E731


def hashes(r: dict) -> dict:
    att = r["attempts"][-1]
    evals = sorted(Path(att["log_dir"]).glob("*.eval"))
    assert len(evals) == 1, f"{r['key']}: {len(evals)} .eval files"
    s = read_eval_log(str(evals[0]), resolve_attachments=True).samples[0]
    msgs = [m.model_dump(mode="json", exclude={"id"}) for m in s.messages]
    ids = {}
    canon = lambda i: f"call_{ids.setdefault(i, len(ids))}"  # noqa: E731
    for m in msgs:
        for c in m.get("tool_calls") or []:
            c["id"] = canon(c.get("id"))
        if m.get("role") == "tool" and "tool_call_id" in m:
            m["tool_call_id"] = canon(m["tool_call_id"])
    first = next((m for m in msgs if m["role"] == "assistant"), None)
    return {"first_reply": sha(json.dumps(first, sort_keys=True)) if first else None,
            "messages": sha(json.dumps(msgs, sort_keys=True)), "n_messages": len(msgs)}


def strip(att: dict) -> tuple:
    env = {k: v for k, v in att["env_vars"].items() if k != "VLLM_BASE_URL"}
    cmd, i = list(att["cmd"]), att["cmd"].index("--log-dir")
    del cmd[i:i + 2]
    return env, cmd


pairs = []
done = sorted((r for k, r in runs.items() if k.startswith("attack/") and k.endswith("_r2") and r["state"] == "done"
               and runs.get(k[:-3], {}).get("state") == "done"),
              key=lambda r: max(r["attempts"][-1].get("end", ""), runs[r["key"][:-3]]["attempts"][-1].get("end", "")))
for r2 in done[:a.first or None]:
    r1 = runs[r2["key"][:-3]]
    e1, c1 = strip(r1["attempts"][0])
    e2, c2 = strip(r2["attempts"][0])
    h1, h2 = hashes(r1), hashes(r2)
    pairs.append({"cell": r1["task"], "same_config": e1 == e2 and c1 == c2,
                  "config_diff": sorted(k for k in set(e1) | set(e2) if e1.get(k) != e2.get(k)) + (["cmd"] if c1 != c2 else []),
                  "identical_first_reply": h1["first_reply"] == h2["first_reply"],
                  "identical_messages": h1["messages"] == h2["messages"], "r1": h1, "r2": h2})
summary = {"run": str(a.run), "pairs": len(pairs), "config_mismatch": sum(not p["same_config"] for p in pairs),
           "identical_first_reply": sum(p["identical_first_reply"] for p in pairs),
           "identical_messages": sum(p["identical_messages"] for p in pairs)}
a.out.write_text(json.dumps({"summary": summary, "pairs": pairs}, indent=1))
print(json.dumps(summary))
raise SystemExit(2 if summary["identical_messages"] or summary["identical_first_reply"] else 3 if summary["config_mismatch"] else 0)
