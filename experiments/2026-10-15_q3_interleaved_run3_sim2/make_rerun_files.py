"""Build the IL paired rate-limit reruns (RULE.md "Paired reruns"; 09 Append 35 option B): per IL host, one registered IL
launch (run3 + sim2 per cell) of exactly the cells whose sim2 failed PURELY on provider rate limits, plus ilrr_box.sh.

Input: the failure census by key (results/sim_ratelimit_failures_2026-10-15.json, read with git show at a pinned commit).
The new-mains fork's schema (sim_ratelimit_list.py @ 36a2047fa): {"runs": [{host, launch_run_dir, launch, cell, run_key,
state, pure_429, attempts: [{terminal_errors: [...], eval_path, eval_sha256, ...}]}]}. Eligible (Append 35) = pure_429 true
AND every attempt's terminal_errors non-empty and all RateLimitError; the failed .eval is the LAST attempt's. Per host it writes, never overwriting:
  config/tasks_ilrr_<h>.json      the eligible cells' ORIGINAL entries, seeded order (sha256('<seed>:<h>_rlr'));
  config/pilot_ilrr_<h>.yaml      the host's IL config with only tasks and the spend lines changed;
  config/ilrr_rerun_of_<h>.json   host, rerun_reason, and per cell rerun_of {launch_run_dir, run_key, eval_path,
                                  eval_sha256, pure_rate_limit};
and config/tasks_ilrr_manifest.json (shas, counts, inputs) and ilrr_box.sh (from ilrr_box.sh.tmpl, its case table).

  python3 make_rerun_files.py <failures git rev>:<path> [--place <host>=<box>[,<box>]]

A placement override (d9: il_q4's rerun on box 1 or box 2, since box 4 runs 4r) adds case keys <host>b<k> that run the
SAME files on box k: the target box's own harness manifest, the Q_k main and the original il_<host> launch relayed to the
same absolute paths (hash-verified at relay; the main's launch.json sha256 is pinned here and checked on the box; each
rerun cell's failed .eval is bound to the census sha read on the original box), the original IL task file staged, and the target box's own launches (il_q<k>, its rerun) added to the must-have-finished list.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
C = HERE / "config"
RATE = 20.5
SEED = "q3-interleaved-run3-sim2-2026-10-14"
R311, RNM = "/home/aswin/phase2/runs", "/home/aswin/phase2/nm/runs"
HOSTS = {  # host -> kind, references, earlier launches on the host (original first), manifest, IL task file and config
    "q1": ("311", f"{R311}/2026-10-14_154401_scaleup311_q1", [f"{R311}/2026-10-14_il_q1"], "harness_manifest_fu.tsv", "q1_all"),
    "q2": ("311", f"{R311}/2026-10-14_154401_scaleup311_q2", [f"{R311}/2026-10-14_il_q2"], "harness_manifest_fu_q2r.tsv", "q2_all"),
    "q3": ("311", f"{R311}/2026-10-14_155838_scaleup311_q3", [f"{R311}/2026-10-14_il_q3"], "harness_manifest_fu.tsv", "q3_all"),
    # d9 02:55: on box 3 q3's reruns run BEFORE s3 (they fill the primary estimate); s3 then needs no other launcher
    "q4": ("311", f"{R311}/2026-10-14_160246_scaleup311_q4", [f"{R311}/2026-10-14_il_q4"], "harness_manifest_fu.tsv", "q4_all"),
    "old": ("311", "/home/aswin/phase2/il_refindex_old", [f"{R311}/2026-10-14_il_old"], "harness_manifest_fu.tsv", "old_all"),
    "s1": ("nm", f"{RNM}/2026-10-14_newmains_s1", [f"{RNM}/2026-10-14_il_s1", f"{RNM}/2026-10-14_il_s4d"],
           "harness_manifest_fu.tsv", "s1"),
    "s2": ("nm", f"{RNM}/2026-10-14_newmains_s2", [f"{RNM}/2026-10-14_il_s2"], "harness_manifest_fu.tsv", "s2"),
}
RSHA = {"old": "79acbfc1ef4523f9a57cc19dc876252159c6c1d62521001f1ac71e2644036940"}
# a placed rerun's relayed main: its launch.json sha256 read on the original box (the main finished long ago)
MAIN_SHA = {"q4": "f12ed4cd86ed516036bcc47857c7a4d90ba95e21595a8c8f484653b9055a9ad6"}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731


def main() -> None:
    src = sys.argv[1]
    place = {}
    if len(sys.argv) > 3 and sys.argv[2] == "--place":
        h0, boxes = sys.argv[3].split("=")
        place[h0] = [b.removeprefix("box") for b in boxes.split(",")]
    if Path(src).exists():  # a plain file (tests only); real builds pin <rev>:<path>
        commit, path, raw = None, src, Path(src).read_bytes()
    else:
        rev, path = src.split(":", 1)
        commit = subprocess.check_output(["git", "rev-parse", f"{rev}^{{commit}}"], text=True).strip()
        raw = subprocess.check_output(["git", "show", f"{commit}:{path}"])
    recs = json.loads(raw)
    recs = recs["runs"]
    by_host: dict[str, list[dict]] = {}
    skipped = []
    for r in recs:
        h = {f"2026-10-14_il_{x}": x for x in HOSTS}.get(Path(r["launch_run_dir"]).name)
        if h is None:
            skipped.append((r["launch_run_dir"], r["cell"], "not an IL host prepared here"))
            continue
        assert r["launch_run_dir"] == HOSTS[h][2][0], r
        assert r["run_key"] == f"sim_attack@september/{r['cell']}", r
        pure = r.get("pure_429") is True and bool(r["attempts"]) and all(
            a.get("terminal_errors") and set(a["terminal_errors"]) == {"RateLimitError"} for a in r["attempts"])
        if not pure:
            skipped.append((r["launch_run_dir"], r["cell"], "not a pure rate-limit failure"))
            continue
        by_host.setdefault(h, []).append(r)
    manifest = {"failures_source": {"commit": commit, "path": path, "sha256": hashlib.sha256(raw).hexdigest()},
                "rate_usd_per_cell": RATE, "skipped": skipped, "hosts": {}}
    cases = []
    for h, rs in sorted(by_host.items()):
        kind, main_dir, prevs, man, il = HOSTS[h]
        keys = {r["cell"] for r in rs}
        assert len(keys) == len(rs), f"{h}: a cell listed twice"
        doc = json.loads((C / f"tasks_il_{il}.json").read_text())
        cells = [c for c in doc["cells"] if c["key"] in keys]
        random.Random(int(hashlib.sha256(f"{SEED}:{h}_rlr".encode()).hexdigest(), 16)).shuffle(cells)
        assert {c["key"] for c in cells} == keys, f"{h}: failure keys not in the IL task file"
        out = {k: v for k, v in doc.items() if k not in ("cells", "stages", "stage")} | {
            "rerun_of_launch": prevs[0], "rerun_reason": "infra: RateLimitError", "cells": cells}
        tp, cp, mp = C / f"tasks_ilrr_{h}.json", C / f"pilot_ilrr_{h}.yaml", C / f"ilrr_rerun_of_{h}.json"
        for p in (tp, cp, mp):
            assert not p.exists(), f"{p} exists"
        tp.write_text(json.dumps(out, indent=1) + "\n")
        cap = math.floor(len(cells) * RATE)
        text = (C / f"pilot_il_{il}.yaml").read_text()
        n = 0
        for pat, rep in ((r"(?m)^tasks: .*$", f"tasks: config/tasks_ilrr_{h}.json   # rate-limit rerun: {len(cells)} original cells"),
                         (r"(?m)^  cap_usd: .*$", f"  cap_usd: {cap}   # {len(cells)} cells x ${RATE}"),
                         (r"(?m)^  worst_case_limit_usd: .*$", f"  worst_case_limit_usd: {cap}   # = cap")):
            text, k = re.subn(pat, rep, text)
            n += k
        assert n == 3, (h, n)
        cp.write_text(f"# PAIRED RATE-LIMIT RERUN {h} (Append 35 B): from pilot_il_{il}.yaml with only tasks and caps changed.\n" + text)
        mp.write_text(json.dumps({"host": h, "rerun_reason": "infra: RateLimitError", "original_launch": prevs[0],
                                  "failures_source": manifest["failures_source"],
                                  "rerun_of": [{"cell": r["cell"], "launch_run_dir": r["launch_run_dir"], "run_key": r["run_key"],
                                                "eval_path": r["attempts"][-1]["eval_path"],
                                                "eval_sha256": r["attempts"][-1]["eval_sha256"],
                                                "pure_rate_limit": True} for r in sorted(rs, key=lambda r: r["cell"])]},
                                 indent=1) + "\n")
        manifest["hosts"][h] = {"cells": len(cells), "cap_usd": cap, "tasks_sha256": sha(tp), "config_sha256": sha(cp),
                                "rerun_of_sha256": sha(mp), "original_launch": prevs[0], "references_from": main_dir}
        common = (f'CAP={cap}; TSHA={sha(tp)}; CSHA={sha(cp)}; MSHA={sha(mp)}; ILTASKS=config/tasks_il_{il}.json; '
                  f'ILTSHA={sha(C / f"tasks_il_{il}.json")}; RSHA={RSHA.get(h, "")}')
        base = "attack_check_311" if kind == "311" else "nm/attack_check_nm"
        if h not in place:
            cases.append(f'  {h}) KIND={kind}; MAIN={main_dir}; PREVS="{" ".join(prevs)}"; MANIFEST=/home/aswin/phase2/{base}'
                         f'/config/{man}; {common}; ILTASKS_STAGE=- ;;')
        for k in place.get(h, []):
            assert kind == "311", "placement overrides are for +311 hosts"
            tman = "harness_manifest_fu_q2r.tsv" if k == "2" else "harness_manifest_fu.tsv"
            tprev = [f"{R311}/2026-10-14_il_q{k}", f"{R311}/2026-10-15_il_q{k}_rlr"]
            cases.append(f'  {h}b{k}) HF={h}; KIND={kind}; MAIN={main_dir}; PREVS="{" ".join(prevs + tprev)}"; '
                         f'MANIFEST=/home/aswin/phase2/{base}/config/{tman}; {common.replace("RSHA=", "RSHA_UNUSED=")}; '
                         f'RSHA={MAIN_SHA[h]}; ILTASKS_STAGE=staged ;;  # {h} placed on box {k}: its main launch.json pinned')
    mf = C / "tasks_ilrr_manifest.json"
    assert not mf.exists()
    mf.write_text(json.dumps(manifest, indent=1) + "\n")
    (HERE / "ilrr_box.sh").write_text((HERE / "ilrr_box.sh.tmpl").read_text().replace("__CASES__", "\n".join(cases)))
    print(json.dumps({h: (v["cells"], v["cap_usd"]) for h, v in manifest["hosts"].items()}), "skipped", len(skipped))


if __name__ == "__main__":
    main()
