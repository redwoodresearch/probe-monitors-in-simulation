"""09's provenance scope for an interleaved batch (09 @ 65ef3a9ff section 4; via d9 10-15 ~07:10; not a relaxation): its approach provenance is evaluated over
the rows of the STAGED sets its attestations are keyed to. A row whose run staging excluded is listed separately as "excluded at
staging: not evaluated", with the staging record's reason, and the staged-out keys must equal the staging records' exclusion
lists exactly. Writes NEW files approach_provenance_<...>_v2.json and attestations_<pool>_<batch>_v2.json (v1 kept). Any failure
on a staged row is a hard stop: exit 2 and nothing written. Keys and hashes only.

  python3 make_prov_attest_v2.py <chain state dir> <batch id>
"""
import copy
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
HYG = HERE / "results" / "hygiene"
REL = "experiments/2026-10-14_q3_overnight_chain"
SIM_ARM = "sim_attack@september"
SCOPE = ("rows of the staged sets the attestations are keyed to (09 @ 65ef3a9ff section 4): exactly the staged set's runs, each evaluated once; "
         "rows whose run staging excluded are listed in excluded_at_staging, not evaluated, and equal the staging exclusions; the exclusion "
         "list is never changed here; a failure on a staged row stays a hard stop")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    st, bid = Path(sys.argv[1]), sys.argv[2]
    b = json.loads((st / "state.json").read_text())["batches"][bid]
    assert b["kind"] == "il" and "attest" in b["done"], (b["kind"], b["done"])
    pv_p = HYG / b["provenance"]["file"]
    at_p = HYG / f"attestations_{b['pool']}_{b['id']}.json"
    pv, at = json.loads(pv_p.read_text()), json.loads(at_p.read_text())
    excl, staged, staged_runs = {}, {}, {}  # run path -> (world, reason); world -> (target, staged sha); world -> the staged set's runs
    for t in b["targets"]:
        rec = json.loads((HERE / "results" / b["stage"][t]["record"]).read_text())
        k = next(x for x in rec if x not in ("git", "stamp"))
        w = "run3" if t.endswith("_run3") else "sim2"
        arm = "attack" if w == "run3" else SIM_ARM
        staged[w] = (t, b["stage"][t]["staged_inputs_sha256"].get(f"{arm}.jsonl"))
        gate = rec[k]["c_gate"]
        for key, why in gate["excluded"].items():
            excl[key.split("#", 1)[0]] = (w, why)
        # the staged set's runs: the target's manifest rows of the arm minus the staging exclusions, tied to the record's counts
        man = json.loads((HERE / "config" / f"manifest_{t}{b.get('rev', '')}.json").read_text())
        rows = [f["path"] for f in man["files"] if f["arm_dir"] == arm]
        out_w = {p for p, (w_, _) in excl.items() if w_ == w}
        assert len(rows) == len(set(rows)) == gate["manifest_keys"] and out_w <= set(rows), f"{t}: manifest rows do not tie to the staging record"
        staged_runs[w] = set(rows) - out_w
        assert len(staged_runs[w]) == gate["staged"], f"{t}: {len(staged_runs[w])} staged runs != the record's {gate['staged']}"
    evaluated, out_rows = [], []
    for r in pv["rows"]:
        path = r["worlds"][r["subject"]]["eval"]
        if path in excl:
            assert excl[path][0] == r["subject"], (path, r["key"])
            out_rows.append({"key": r["key"], "cell": r["cell"], "subject": r["subject"], "excluded_at_staging": "not evaluated",
                             "staging_reason": excl[path][1], "run": path, "row_v1": r})
        else:
            assert r["staged_shas"].get(r["subject"]) == staged[r["subject"]][1], f"{r['key']}: row's staged sha is not the set's"
            evaluated.append(r)
    got = {x["run"] for x in out_rows}
    if got != set(excl):  # 09: the staged-out keys must equal the staging records' exclusion lists exactly
        sys.exit(f"staged-out rows {sorted(got)} != staging exclusions {sorted(excl)}")
    for w, runs in staged_runs.items():  # 09 @ 65ef3a9ff section 4: the evaluated rows are exactly the staged set's runs, each once
        ev = [r["worlds"][r["subject"]]["eval"] for r in evaluated if r["subject"] == w]
        if len(ev) != len(set(ev)) or set(ev) != runs:
            sys.exit(f"{w}: evaluated rows ({len(ev)}, {len(set(ev))} distinct) != the staged set's {len(runs)} runs "
                     f"(missing {sorted(runs - set(ev))[:3]}, extra {sorted(set(ev) - runs)[:3]})")
    pairs = lambda rows: sum(len(r["pairs"]) for r in rows)
    pv2 = {k: v for k, v in pv.items() if k not in ("rows", "counts", "pass")} | {
        "covers": {**pv.get("covers", {}), "keys": sorted(r["key"] for r in evaluated), "cells": sorted({r["cell"] for r in evaluated})},  # evaluated rows only
        "rows": evaluated, "excluded_at_staging": out_rows, "scope": SCOPE, "supersedes": pv_p.name, "v1_sha256": sha(pv_p),
        "counts": {"rows": len(evaluated), "pass": sum(r["pass"] for r in evaluated), "pairs": pairs(evaluated),
                   "pairs_equal": sum(v["equal_approach"] and v["equal_system_prompt"] for r in evaluated for v in r["pairs"].values()), "excluded_at_staging": len(out_rows)},
        "pass": all(r["pass"] for r in evaluated)}
    bad = [r["key"] for r in evaluated if not r["pass"]]
    if bad:
        print(f"HARD STOP: provenance fails on staged rows {bad}", file=sys.stderr)
        sys.exit(2)
    pv2_p = HYG / pv_p.name.replace(".json", "_v2.json")
    pv2_text = json.dumps(pv2, indent=1) + "\n"
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    at2 = copy.deepcopy(at)
    for ssha, es in at2["attestations"].items():
        for e in es:
            if e["check"] != "approach_provenance":
                continue
            w = e["covers"]["world"]
            assert staged[w][1] == ssha, (w, ssha)
            rows = [r for r in evaluated if r["subject"] == w]
            has = ssha in pv2_text
            e["pass"] = bool(rows) and all(r["pass"] for r in rows) and has
            e["covers"]["cells"] = sorted(r["cell"] for r in rows)
            e["evidence"] = {"path": f"{REL}/results/hygiene/{pv2_p.name}", "sha256": hashlib.sha256(pv2_text.encode()).hexdigest()}
            e["checker"], e["computed_utc"] = f"{REL}/make_prov_attest_v2.py@{head}", utc
            e["detail"] = {"rows_pass": sum(r["pass"] for r in rows), "rows": len(rows), "pairs": pairs(rows), "scope": SCOPE,
                           "excluded_at_staging": sorted(x["key"] for x in out_rows if x["subject"] == w), "evidence_contains_staged_sha": has}
    failing = {s: [e["check"] for e in es if not e["pass"]] for s, es in at2["attestations"].items()}
    failing = {s: c for s, c in failing.items() if c}
    if failing:
        print(f"HARD STOP: attestations still fail {failing}", file=sys.stderr)
        sys.exit(2)
    at2 |= {"supersedes": at_p.name, "v1_sha256": sha(at_p), "scope": SCOPE, "failing": failing}
    at2_p = HYG / at_p.name.replace(".json", "_v2.json")
    with pv2_p.open("x") as f:
        f.write(pv2_text)
    with at2_p.open("x") as f:
        f.write(json.dumps(at2, indent=1) + "\n")
    print(json.dumps({"provenance_v2": {"path": f"{REL}/results/hygiene/{pv2_p.name}", "sha256": sha(pv2_p), "counts": pv2["counts"], "pass": pv2["pass"]},
                      "attestations_v2": {"path": f"{REL}/results/hygiene/{at2_p.name}", "sha256": sha(at2_p), "failing": failing},
                      "excluded_at_staging": [(x["key"], x["staging_reason"]) for x in out_rows]}, indent=1))


if __name__ == "__main__":
    main()
