"""Adapter (metadata only; no activation or score is read) for the sel97 IL descriptive add-on (09's Append 32 note @ 20c2837f8): build a
pass-2-schema message for ONE extraction root (one arm: run3 attack or sim2 sim_attack) from d9's committed records, gate it, and write the
message plus an adapter record that score_step3_il.py accepts. Gates as build_ep2_message.py (attestation, staging, extraction H200, repro-c,
format check, manifest one file per cell, staged counts), with the run table read world-keyed: every staged cell has a "ran" row of its world.

  python3 -I build_il_message.py --root <extraction root> --staged-set <set> --volume <vol> --arm attack=run3 (or sim_attack@september=sim2)
      --rec manifest=... --rec staging=... --rec device=... --rec repro=... --rec attest=... --rec run_table=... --rec format_<arm>=... --out-dir <dir>
"""

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
SEL_RE = re.compile(r"^ilold_oldsel_(?:run3_v1(?:_try[0-9]+)?__attack|sim2_v1(?:_try[0-9]+)?__sim_attack)$")  # 09's regex @ 20c2837f8, verbatim
S4RGAP_RE = re.compile(r"^ilnm_s4rgap_sim2(_s[0-9]+)?_v1(_try[0-9]+)?__sim_attack$")  # 09's sim2 gap-fill s4r regex (section 3), verbatim
SUFFIX = {"run3": "attack", "sim2": "sim_attack"}


SUPERSEDED = {"ilnm_s4rgap_sim2_v1_try2__sim_attack"}  # 09 10-16 ~00:4x (nm145): refused by name


def world_ok(f, w):
    """The folder is in 09's sel97 or s4rgap grammar and names its world (s4rgap: always sim2, whatever its optional _s<n>)."""
    if f in SUPERSEDED:
        return False
    if S4RGAP_RE.fullmatch(f):
        return w == "sim2"
    return bool(SEL_RE.fullmatch(f)) and f.split("_v1")[0].split("_")[-1] == w


def main():
    ap = argparse.ArgumentParser()
    for x in ("root", "staged_set", "volume", "out_dir"):
        ap.add_argument("--" + x.replace("_", "-"), required=True)
    ap.add_argument("--arm", action="append", required=True, help="<meta arm>=<world>, e.g. e2a=honest_a")
    ap.add_argument("--rec", action="append", required=True)
    a = ap.parse_args()
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, check=True).stdout  # noqa: E731
    arms = dict(x.split("=", 1) for x in a.arm)
    pins, recs = {}, {}
    for r in a.rec:
        name, cp = r.split("=", 1)
        c, p = cp.split(":", 1)
        if not any(b.strip().startswith("origin/") for b in git("branch", "-r", "--contains", c).decode().splitlines()):
            raise SystemExit(f"{name}: commit {c} is not on origin")
        b = git("show", f"{c}:{p}")
        pins[name] = {"commit": git("rev-parse", c).decode().strip(), "path": p, "sha256": hashlib.sha256(b).hexdigest()}
        recs[name] = json.loads(b)
    need = {"manifest", "staging", "device", "repro", "attest", "run_table"} | {f"format_{m}" for m in arms}
    if set(recs) != need:
        raise SystemExit(f"records needed: {sorted(need)}")
    st = recs["staging"]["main"]
    sha = {m: st["assembled"]["staged_inputs_sha256"].get(f"{m}.jsonl") for m in arms}
    folder = {m: f"{a.staged_set}__{SUFFIX[w]}" for m, w in arms.items()}
    if len(arms) != 1 or not all(world_ok(f, w) for f, w in zip(folder.values(), arms.values())):
        raise SystemExit(f"one arm per root, inside 09's sel97 / s4rgap grammar, with its world in the name: {folder}")
    stratum = "S_sim2gap" if S4RGAP_RE.fullmatch(next(iter(folder.values()))) else "S_old"
    gate = {}
    att = recs["attest"]["attestations"]
    gate["attestations_all_pass"] = {"pass": all(att.get(sha[m]) and all(str(x["pass"]) == "True" for x in att[sha[m]]) for m in arms)}
    excluded = st["c_gate"]["excluded"]
    gate["staging"] = {"pass": st["out"] == a.staged_set and all(sha.values()) and set(st["assembled"]["epochs"]) >= set(arms)
                       and st["c_gate"]["manifest_keys"] == st["c_gate"]["staged"] + len(excluded)}
    dv = recs["device"]
    gate["extraction_h200"] = {"shards": len(dv["shards"]), "pass": dv["config"]["run"] == a.root and dv["config"]["inputs"] == a.staged_set and dv["volume"] == a.volume
                               and all(dv["config"]["inputs_sha256"].get(f"{m}.jsonl") == sha[m] for m in arms) and dv["config"]["model"] == "Qwen/Qwen3.5-27B"
                               and len(dv["shards"]) == dv["config"]["shards"] and all(s.get("gpu_name") == "NVIDIA H200" for s in dv["shards"])}
    rp = recs["repro"]
    gate["repro_c"] = {"pass": rp["run"] == a.root and all(rp["staged_sha"].get(f"{m}.jsonl") == sha[m] for m in arms) and not rp["failing_shards"]
                       and rp["passing"] == rp["epochs"] and rp["coverage"]["every_shard_one_of_each"] is True and rp["min_cosine"] >= rp["bar"]["cosine"]}
    gate["format_checks"] = {"pass": all(str(recs[f"format_{m}"]["passes"]) == "True" and recs[f"format_{m}"]["staged_sha"] == sha[m] and recs[f"format_{m}"]["arm"] == m for m in arms)}
    files = [f for f in recs["manifest"]["files"] if f["arm_dir"] in arms]
    keys = {f"{f['launch']}/{f['rel']}#e0": f for f in files}
    per_arm = {m: [f for f in files if f["arm_dir"] == m] for m in arms}
    # Append 38 section 3: pre-start reference redraws are part of the epoch-2 protocol; a cell's single counted file may be a redraw (listed)
    gate["manifest"] = {"files": {m: len(v) for m, v in per_arm.items()}, "redraws": sorted(f"{arms[f['arm_dir']]}|{f['task']}" for f in files if f["attempt_kind"] == "redraw"),
                        "pass": len(keys) == len(files) and all(len({f["task"] for f in v}) == len(v) for v in per_arm.values()) and set(excluded) <= set(keys)
                        and all(f["attempt_kind"] in ("first", "redraw") for f in files) and sum(map(len, per_arm.values())) == st["c_gate"]["manifest_keys"]
                        and all(sum(1 for f in per_arm[m] if f"{f['launch']}/{f['rel']}#e0" not in excluded) == st["assembled"]["epochs"].get(m) for m in arms)}
    staged_keys = {f"{arms[f['arm_dir']]}|{f['task']}" for k, f in keys.items() if k not in excluded}
    rt = recs["run_table"]
    rows = rt.get("cells") if "cells" in rt else rt["rows"]
    ran = {r["key"] for r in rows if r.get("status") == "ran" and r.get("world") in arms.values()}
    gate["run_table"] = {"rows": len(rows), "ran_this_world": len(ran), "status_rows_not_scored": sorted(r["key"] for r in rows if r.get("world") in arms.values() and r.get("status") != "ran"),
                         "pass": staged_keys <= ran}
    if not all(g["pass"] for g in gate.values()):
        raise SystemExit(f"adapter gates failed: {json.dumps({k: g for k, g in gate.items() if not g['pass']})}")
    epochs = [{"key": k, "cell": f["task"], "world": arms[f["arm_dir"]], "stratum": stratum, "run": a.root, "arm": f["arm_dir"],
               "staged": f"{a.staged_set}/inputs/{f['arm_dir']}.jsonl", "staged_sha256": sha[f["arm_dir"]], "folder": folder[f["arm_dir"]],
               "launch_task": f.get("launch_task")} for k, f in sorted(keys.items()) if k not in excluded]
    exc = [{"key": k, "world": arms[keys[k]["arm_dir"]], "cell": keys[k]["task"], "arm": keys[k]["arm_dir"], "folder": folder[keys[k]["arm_dir"]], "reason": excluded[k]}
           for k in sorted(excluded)]
    msg = {"d9_arm": a.staged_set, "volume": a.volume, "provenance": [{"kind": "il_sel97_addon" if stratum == "S_old" else "il_s4rgap_descriptive", "run": a.root, "staged_set": a.staged_set,
           "device_record": {"commit": pins["device"]["commit"], "path": pins["device"]["path"]}, "staging_record": {"commit": pins["staging"]["commit"], "path": pins["staging"]["path"]},
           "excluded_by_staging": exc, "rows": len(epochs)}], "epochs": epochs}
    mb = json.dumps(msg, indent=1, sort_keys=True).encode()
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    mp = od / f"msg_{a.staged_set}.json"
    mp.open("xb").write(mb)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"label": "adapter record: d9's committed records -> pass-2 message for one sel97 IL add-on root (metadata only); descriptive add-on (09 @ 20c2837f8)",
           "stamp (PT)": stamp, "git": git("rev-parse", "HEAD").decode().strip(), "staged_set": a.staged_set, "folders": folder, "arms": arms,
           "message_sha256": hashlib.sha256(mb).hexdigest(), "n_epochs": len(epochs), "per_folder": {fo: sum(e["folder"] == fo for e in epochs) for fo in folder.values()},
           "excluded_by_staging": exc, "pins": pins, "gate": gate}
    rp_ = od / f"adapter_{a.staged_set}_{stamp}.json"
    rp_.open("x").write(json.dumps(rec, indent=1))
    print(rp_, rec["message_sha256"], rec["per_folder"], len(exc), {k: g["pass"] for k, g in gate.items()}, gate["run_table"]["status_rows_not_scored"])


if __name__ == "__main__":
    main()
