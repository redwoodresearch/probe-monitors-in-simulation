"""How often the level-2 ID rule mangled IDs joined by `_`, per arm and text level, before the 2026-10-07 fix.

An output counts when its scrubbed text differs between the rule before the fix (src/probe_shift/scrub.py at
`--before`, read from git) and the fixed rule (the working tree), at each level 2-7. Only outputs with a hex run of 6+
characters next to `_` can differ, so the rest are skipped. Also: per environment, and the most common contexts of the
underscore-joined IDs (the 12 characters before each, digits and hex shown as placeholders).

  uv run python experiments/2026-10-07_hex_id_scrub_fix/analyze_mangling_counts.py
Writes results/mangling_counts_<stamp>.json and .md.
"""

import argparse
import gzip
import importlib.util
import json
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from probe_shift.provenance import capture
from probe_shift.scrub import scrub

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
ARMS = ("real-attack-e1", "real-attack-e2", "sim-attack")
CANDIDATE = re.compile(r"_[0-9a-fA-F]{6}|[0-9a-fA-F]{6}_")  # a hex run next to an underscore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config/hexfix.yaml"))
    ap.add_argument("--before", default="c47f0b0", help="a commit with the rule before the fix")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    old_path = HERE / ".run_cache" / f"scrub_{args.before}.py"
    old_path.parent.mkdir(exist_ok=True)
    old_path.write_bytes(subprocess.run(["git", "show", f"{args.before}:src/probe_shift/scrub.py"], cwd=ROOT,
                                        capture_output=True, check=True).stdout)
    spec = importlib.util.spec_from_file_location("scrub_before", old_path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    assert old.scrub("web_46a7d3b90e5c", 2) != scrub("web_46a7d3b90e5c", 2), "--before already has the fix"

    rows = [r for r in map(json.loads, gzip.open(Path(cfg["staging_dir"]).expanduser() / "outputs.jsonl.gz", "rt")) if r["kept"]]
    total = Counter(r["arm"] for r in rows)
    cand = [r for r in rows if CANDIDATE.search(r["readout_text"])]
    levels = range(2, 8)
    changed = {lv: Counter() for lv in levels}
    by_env = {lv: Counter() for lv in levels}
    runs = {lv: {a: set() for a in ARMS} for lv in levels}
    rows_changed = {lv: [] for lv in levels}
    contexts = {a: Counter() for a in ARMS}
    for r in cand:
        t = r["readout_text"]
        for lv in levels:
            if old.scrub(t, lv) != scrub(t, lv):
                changed[lv][r["arm"]] += 1
                by_env[lv][(r["environment"], r["arm"])] += 1
                runs[lv][r["arm"]].add(r["key"])
                rows_changed[lv].append(f"{r['key']}|{r['step']}|{r['action']}")
        for m in re.finditer(r"(?<![^\W_])(?=[0-9a-fA-F]*[a-fA-F])(?=[0-9a-fA-F]*\d)[0-9a-fA-F]{6,}(?![^\W_])", t):
            before, after = t[max(0, m.start() - 12):m.start()], t[m.end():m.end() + 1]
            if before.endswith("_") or after == "_":
                contexts[r["arm"]][re.sub(r"\d+", "<n>", before) + "<hex>" + after] += 1
    out = {"stamp": stamp, "provenance": capture(ROOT, [Path(__file__), Path(args.config), ROOT / "src/probe_shift/scrub.py"]), "args": vars(args),
           "outputs_per_arm": dict(total), "candidate_outputs": len(cand),
           "changed_outputs": {f"L{lv}": {a: changed[lv][a] for a in ARMS} for lv in levels},
           "changed_share": {f"L{lv}": {a: round(changed[lv][a] / total[a], 5) for a in ARMS} for lv in levels},
           "runs_with_a_changed_output": {f"L{lv}": {a: len(runs[lv][a]) for a in ARMS} for lv in levels},
           "by_environment_L7": {f"{e}|{a}": n for (e, a), n in sorted(by_env[7].items())},
           "top_contexts": {a: contexts[a].most_common(15) for a in ARMS},
           "changed_rows": {f"L{lv}": rows_changed[lv] for lv in levels}}  # key|step|action of every changed output
    md = [f"# Outputs whose scrubbed text changes with the underscore fix ({stamp})", "",
          "| level | " + " | ".join(f"{a}: outputs (share; runs)" for a in ARMS) + " |", "|---|---|---|---|"]
    for lv in levels:
        md.append(f"| {lv} | " + " | ".join(f"{changed[lv][a]} ({changed[lv][a] / total[a]:.4f}; {len(runs[lv][a])})" for a in ARMS) + " |")
    md += ["", "Level 7, per environment (e1 / e2 / sim):", "", "| environment | e1 | e2 | sim |", "|---|---|---|---|"]
    for e in sorted({e for e, _ in by_env[7]}):
        md.append(f"| {e} | " + " | ".join(str(by_env[7][(e, a)]) for a in ARMS) + " |")
    md += ["", "Most common contexts of underscore-joined hex IDs (12 characters before, digits as <n>):", ""]
    for a in ARMS:
        md.append(f"- {a}: " + "; ".join(f"`{c}` ×{n}" for c, n in contexts[a].most_common(8)))
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"mangling_counts_{stamp}.json").write_text(json.dumps(out, indent=1))
    (HERE / "results" / f"mangling_counts_{stamp}.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
