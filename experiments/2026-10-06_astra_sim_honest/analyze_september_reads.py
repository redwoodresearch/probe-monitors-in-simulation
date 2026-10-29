"""Do the env files the September simulator read (its `read_file` results) equal our v2-tag checkouts byte for byte?

Every simulated sample keeps the simulator's own conversation in `events_data.messages`, including each `read_file`
call and the file contents it returned from the September env folder. Compares each against the same path in the
local checkout (`~/simulated-control-evals-sept/.settings/linuxarena/<env>/codebase`). Paths are resolved with the
simulator's own `locate` and the sample's `container_path_mappings`, so container paths (`/app/...`) are compared too.
Strict: anything but the same text (up to trailing newlines) counts as different; a path September answered with
`No such file` must also be missing here. Reads a local copy of the attack cohort's logs.

Run inside the September snapshot, so path resolution is September's:
  cd ~/simulated-control-evals-sept && uv run python \\
    ~/probe-shift-sims/experiments/2026-10-06_astra_sim_honest/analyze_september_reads.py --logs <dir of .eval files>
"""

import argparse
import json
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from sim.affordances.codebase_read import locate

HERE = Path(__file__).parent
ENVS = Path.home() / "simulated-control-evals-sept" / ".settings" / "linuxarena"


def text(content, attachments):
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if isinstance(content, str) and content.startswith("attachment://"):
        content = attachments[content.removeprefix("attachment://")]
    return content


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", type=Path, required=True)
    args = p.parse_args()
    counts = defaultdict(Counter)
    differ = defaultdict(set)
    for log in sorted(args.logs.glob("*.eval")):
        cohort = log.name.split("_")[0]
        with zipfile.ZipFile(log) as z:
            for name in (n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")):
                s = json.loads(z.read(name))
                env, att = s["metadata"]["env"], s.get("attachments") or {}
                mappings = s["metadata"].get("container_path_mappings") or {}
                root = ENVS / env / "codebase"
                pool = (s.get("events_data") or {}).get("messages") or []
                calls = {tc["id"]: tc for m in pool for tc in m.get("tool_calls") or []}
                for m in pool:
                    if m.get("role") != "tool" or m.get("function") != "read_file" or m.get("error"):
                        continue
                    path = (calls.get(m.get("tool_call_id")) or {}).get("arguments", {}).get("path")
                    got = text(m.get("content"), att)
                    relative = locate(root, str(path or ""), "", None, mappings) if path else None
                    if relative is None or not (root / relative).is_file():
                        both_missing = got.startswith("No such file")
                        counts[env, cohort]["missing in both" if both_missing else "differ"] += 1
                        if not both_missing:
                            differ[env, cohort].add(str(path))
                        continue
                    want = (root / relative).read_text(errors="replace")
                    if got == want or got.rstrip("\n") == want.rstrip("\n"):
                        counts[env, cohort]["identical"] += 1
                    else:
                        counts[env, cohort]["differ"] += 1
                        differ[env, cohort].add(path)
    rows = [{"environment": e, "cohort": c, **counts[e, c], "differing_paths": sorted(differ[e, c])} for e, c in sorted(counts)]
    for r in rows:
        print(json.dumps(r)[:400])
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")
    with open(HERE / "results" / f"september_reads_{stamp}.json", "x") as f:  # results/ is append-only
        f.write(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
