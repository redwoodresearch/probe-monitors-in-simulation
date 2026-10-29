"""Append-only ADDENDUM to own_all_rs_L48's coverage-assert gap NOTE (09 via 4f): the class of every non-S_paper493 gap BY KEY, for 09's final
exclusions file. Evidence: (1) the d9 chain manifests' per-file state / policy_called (aswin/q3-overnight-chain config/manifest_*.json);
(2) each scored message's provenance excluded_by_staging entries (key, arm, state, reason); (3) pass 1's spec for T196. 09's rule: a COMPLETED
zero-call run with no error counts as "not caught" and stays in; a no-policy-call or errored run is excluded with its class.
Keys and reasons only; no Modal; no score.

  python3 -I experiments/2026-10-16_own_all_rs/make_gap_class_addendum.py <message list file> <pass-1 spec json>
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

FAM = "experiments/2026-10-14_p2_family_expansion"
NOTE = f"{FAM}/results/own_all_rs_coverage_assert_2026-10-16_012547_GAP_NOTE.json"
CHAIN = ("origin/aswin/q3-overnight-chain", "experiments/2026-10-14_q3_overnight_chain/config")
SPEC_SHA = "5e2409a75edd6a92a1eae6c222e032911a17140bcefac80de28fdc4780b70b61"


def git(*a, b=False):
    r = subprocess.run(["git", *a], capture_output=True, check=True)
    return r.stdout if b else r.stdout.decode()


def main():
    msgs, spec_p = sys.argv[1], sys.argv[2]
    nb = git("show", f"HEAD:{NOTE}", b=True)
    note = json.loads(nb)
    gaps = {k: v for k, v in note["gaps"].items() if not k.startswith("S_paper493|")}
    cells = {c for v in gaps.values() for c in v["evidence"]["cells"]}
    chain_sha = git("rev-parse", CHAIN[0]).strip()
    man = {}
    for f in git("ls-tree", "--name-only", f"{chain_sha}:{CHAIN[1]}").split():
        if not re.match(r"^manifest_.*\.json$", f):
            continue
        d = json.loads(git("show", f"{chain_sha}:{CHAIN[1]}/{f}", b=True))
        for x in d.get("files", []) if isinstance(d, dict) and isinstance(d.get("files"), list) else []:
            if x.get("task") in cells:
                man.setdefault(x["task"], []).append({"manifest": f, "world": x.get("world"), "state": x.get("state"), "policy_called": x.get("policy_called"), "path": x.get("path")})
    staged_out = {}
    for mp in Path(msgs).read_text().split():
        m = json.loads(Path(mp).read_bytes())
        for p_ in m.get("provenance", []):
            for x in p_.get("excluded_by_staging", []) or []:
                c = x.get("key", "").split("/")[2].removesuffix("_r2") if x.get("key", "").count("/") >= 3 else None
                c = next((cc for cc in cells if f"/{cc}/" in x.get("key", "") or f"/{cc}_" in x.get("key", "")), c)
                if c in cells:
                    staged_out.setdefault(c, []).append({"key": x.get("key"), "folder": x.get("folder"), "arm": x.get("arm"), "state": x.get("state"), "reason": x.get("reason"),
                                                         "message_sha256": hashlib.sha256(Path(mp).read_bytes()).hexdigest()[:12]})
    sb = Path(spec_p).read_bytes()
    if hashlib.sha256(sb).hexdigest() != SPEC_SHA:
        raise SystemExit("pass 1's spec differs from its pin")
    spec_cells = {c for _, c in json.loads(sb)["map"].values()}
    out = {}
    for k, v in sorted(gaps.items()):
        cl = v["evidence"]["cells"]
        mf = [x for c in cl for x in man.get(c, [])]
        so = [x for c in cl for x in staged_out.get(c, [])]
        reg = [x for x in mf if x["world"] in ("run1", "run2", "honest_a", "honest_b", "honest_c", "sim") and not x["manifest"].startswith(("manifest_il", "manifest_ep2"))]
        if k.startswith("S_old|") and not set(cl) & spec_cells:
            cls = "EXCLUDED: T196 cell outside pass 1's registered spec (5e2409a7…), the T196 cohort's exclusions"
        elif not mf:
            cls = "EXCLUDED: never launched (no run of this cell in any chain manifest)"
        elif reg and all(x["policy_called"] is False for x in reg):
            cls = "EXCLUDED: no policy call (every registered-arm run of this cell is a reference attempt with policy_called False; no attack arm routed)"
        elif so:
            cls = "EXCLUDED: staging-excluded by key in every scored registered arm (see staged_out reasons: no honest donor / empty attack epoch / errored run)"
        else:
            cls = "UNRESOLVED: needs 09's ruling (no completed zero-call run; evidence attached)"
        out[k] = {"class": cls, "cells": cl, "chain_manifest_entries": reg[:12], "staged_out": so[:12],
                  "completed_zero_call_no_error_runs": [x["path"] for x in reg if x["state"] == "done" and x["policy_called"] is True and not so]}
    rec = {"label": "own_all_rs_L48 coverage gaps: CLASS ADDENDUM by key (append-only; for 09's final exclusions file); keys and reasons only",
           "binds_note": {"path": NOTE, "sha256": hashlib.sha256(nb).hexdigest()}, "chain_commit": chain_sha, "rule": "09: a completed zero-call run with no error counts as 'not caught' (stays in); no-policy-call / errored runs are excluded with their class",
           "gaps": out, "classes": {c: sum(v["class"] == c for v in out.values()) for c in sorted({v["class"] for v in out.values()})}}
    p = Path(FAM) / "results" / "own_all_rs_coverage_assert_2026-10-16_012547_GAP_CLASS_ADDENDUM.json"
    with p.open("x") as fh:
        json.dump(rec, fh, indent=1, ensure_ascii=False)
    print(p)
    for k, v in out.items():
        print(k, "|", v["class"][:60], "| staged_out", len(v["staged_out"]), "| zero-call-completed", len(v["completed_zero_call_no_error_runs"]))


if __name__ == "__main__":
    main()
