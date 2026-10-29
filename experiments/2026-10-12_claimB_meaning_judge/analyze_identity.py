"""Meaning judge, POST HOC descriptive ($0, counts only, never text; b7's request): at the 91 strict sites, by same- vs
mixed-batch e1/e2 (the key's batch prefix) and natural vs new simulated outputs (new = sim_new.jsonl reader_output, the
mapping run_meaning_judge.py uses):
1. e1 == e2 byte-identical, and with whitespace collapsed;
2. sim == e1 byte-identical, and collapsed; at mixed-batch sites also sim == the overlap800-side real run, collapsed;
3. per-site (sim score, e2 score) cross-tables under analyze_meaning.answers (each score = share of the two orders judged
   same_meaning), for Sol and, when complete, Gemini (otherwise its partial counts, labelled);
4. each n437 run's reference source (metadata fork run prefix) beside the e1 and e2 batch prefixes.

  uv run --no-project --python 3.14 --with numpy python experiments/2026-10-12_claimB_meaning_judge/analyze_identity.py
"""

import hashlib
import json
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import analyze_meaning as A  # noqa: E402  (answers, unchanged)

DATA = HERE / "data"
SIMDIR = Path.home() / ".cache/probe-shift/s3"
collapse = lambda t: " ".join((t or "").split())  # noqa: E731


def fork_prefix(key, rows):
    r = rows[key]
    with zipfile.ZipFile(SIMDIR / r["cohort"].removeprefix("s3://") / "logs" / r["file"]) as z:
        for n in z.namelist():
            if n.startswith("samples/") and n.endswith(".json"):
                s = json.loads(z.read(n))
                if s["id"] == r["sample_id"]:
                    return (s["metadata"].get("fork") or {}).get("run", "").rsplit("-", 1)[0]
    return None


def main():
    rows = {r["key"]: r for r in json.loads((HERE.parents[1] / "splits/exclusions_v4.json").read_text())["samples"]}
    pm = json.loads((HERE / "results/prefix_matched_2026-10-12_145915.json").read_text())["strict_sites"]
    sites = [s for s in map(json.loads, (DATA / "sites.jsonl").read_text().splitlines()) if s["set"] == "strict"]
    new = {r["pair"]: r["result"]["reader_output"] for r in map(json.loads, (DATA / "sim_new.jsonl").read_text().splitlines()) if r["status"] == "ok"}
    judged = {}
    for j in ("sol", "gemini"):
        p = DATA / f"meaning_{j}_calls.jsonl"
        if p.exists():
            judged[j] = A.answers(p)
    c, xt, refs, tasks = defaultdict(Counter), defaultdict(Counter), Counter(), defaultdict(set)
    for s in sites:
        e1k, e2k = pm[s["site"]]["e1"], pm[s["site"]]["e2"]
        b1, b2 = e1k.split("_")[0], e2k.split("_")[0]
        batch = "same_batch" if b1 == b2 else "mixed_batch"
        sim = s["sim"] if s["sim"] is not None else new.get(s["site"])
        for g in (batch, f"{batch}|{s['sim_source']}"):
            c[g]["sites"] += 1
            tasks[g].add(s["task"])
            c[g]["e1_eq_e2_bytes"] += s["e1"] == s["e2"]
            c[g]["e1_eq_e2_collapsed"] += collapse(s["e1"]) == collapse(s["e2"])
            c[g]["sim_missing"] += sim is None
            c[g]["sim_eq_e1_bytes"] += sim is not None and sim == s["e1"]
            c[g]["sim_eq_e1_collapsed"] += sim is not None and collapse(sim) == collapse(s["e1"])
            if batch == "mixed_batch":
                ov = s["e1"] if b1.startswith("overlap") else s["e2"]
                c[g]["sim_eq_overlap800_side_collapsed"] += sim is not None and collapse(sim) == collapse(ov)
        refs[(b1, b2, fork_prefix(pm[s["site"]]["sim"], rows))] += 1
        for j, a in judged.items():
            ks = [(str(s["site"]), pair, "main", o, "0") for pair in ("e1-sim", "e1-e2") for o in ("AB", "BA")]
            if all(k in a for k in ks):
                xt[f"{j}|{batch}"][str(((a[ks[0]] + a[ks[1]]) / 2, (a[ks[2]] + a[ks[3]]) / 2))] += 1
            else:
                xt[f"{j}|{batch}"]["incomplete"] += 1
    out = {"label": "POST HOC descriptive: identity counts and score cross-tables at the strict sites (counts only)",
           "counts": {g: dict(v) | {"tasks": len(tasks[g])} for g, v in sorted(c.items())}, "cross_tables_(sim_score,e2_score)": {g: dict(v) for g, v in sorted(xt.items())},
           "reference_source_by_e1_e2_batch": [{"e1_batch": k[0], "e2_batch": k[1], "n437_fork_run_prefix": k[2], "sites": n} for k, n in sorted(refs.items(), key=str)],
           "judges_present": sorted(judged), "inputs": {f: hashlib.sha256((DATA / f).read_bytes()).hexdigest() for f in ("sites.jsonl", "sim_new.jsonl", "meaning_sol_calls.jsonl", "meaning_gemini_calls.jsonl") if (DATA / f).exists()},
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(HERE / "results" / f"identity_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("inputs",)}, indent=1))


if __name__ == "__main__":
    main()
