"""RULE.md "Pass 2": convert d9's scale-up routing (aswin/q3-overnight-chain layout) into run_famexp_pass2.py's message schema.
A REAL message covers two extraction runs (su311_<q>_main_extract_v1: honest_a, honest_b, attack -> run1; su311_<q>_run2_extract_v1:
attack -> run2); a SIM message covers one (su311_<q>_sim_extract_v1: sim). Metadata only: no activation or score is read here.

  python3 make_pass2_message.py --d9-arm su311_q1_real --volume probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14 \\
      --part main:<commit>:<manifest path>:<extraction run>:<staging record path>:<set key>:<device record path> \\
      --part run2:... --out <message.json>
The staging record (d9's results/chain_stage_*.json, `<set key>` its sub-record) gives the staged set (`out`), the staged
sha256 per <arm>.jsonl (`assembled.staged_inputs_sha256`) and the runs staging excluded (`c_gate.excluded`, with reasons): those
are dropped and LISTED (7b item 5), never scored as zero. Output folders follow 09: `<staged set>[_try<n>]__<meta arm>`.
Per part, rows of the manifest's `files` with the part's world(s) are kept (main: honest_a, honest_b, run1; run2: run2; sim: sim;
the run2/sim manifests' honest donor rows are not extracted there and are dropped). Staged epoch key = "<launch>/<rel>#e0".
The device record (results/extract_chain_<run>_a<n>_<stamp>.json) must show every shard on an NVIDIA H200 (Append 21 §5).
"""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROW_META = ("stage", "il_order", "first_start", "pause_gate_at_start", "interleaved_launch", "snapshot_id", "main_launch", "sim_attempt_timeout_s")  # IL rows (Append 32)
ENDED = ("done", "no policy call")
EP2_ARM = {"e2a": "honest_a", "e2b": "honest_b"}  # 4f / 09: T196's epoch-2 arms e2a -> honest_a, e2b -> honest_b  # ended states in d9's manifests ("no policy call": the run ended before any policy call); "stopped" is not ended
WORLDS = {"main": ("honest_a", "honest_b", "run1"), "run2": ("run2",), "sim": ("sim",), "run3": ("run3",), "sim2": ("sim2",)}  # run3 / sim2: IL (Append 32)


def show(commit: str, path: str) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{path}"], capture_output=True, check=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d9-arm", required=True)
    ap.add_argument("--volume", required=True)
    ap.add_argument("--part", action="append", required=True, help="kind:commit:manifest:run:staging_record:set_key:device_record")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stopped-verdict", default="", help="commit:path:sha256 of d9's header-verdict record: a staged 'stopped' run is admitted only if it says success")
    ap.add_argument("--cite", action="append", default=[], help="label=commit:path of a record to cite per part (sha256 recorded), e.g. run_table=..., il_donor_shas=...")
    ap.add_argument("--unusable", default="", help="commit:path:sha256 of d9's unusable-honest record (submit-only runs): cited per part like --setup-failures")
    ap.add_argument("--setup-failures", default="", help="commit:path:sha256 of d9's setup-failure record (nm360): cited per part for its runs")
    ap.add_argument("--attest", default="", help="commit:path:sha256 of an H200 attestation (resumed extractions whose early shards logged no gpu_name)")
    ap.add_argument("--test-only-keys", default="", help="TEST only: a JSON list of the extraction's staged keys (a limited test set)")
    ap.add_argument("--test-no-c-gate", action="store_true", help="TEST only: accept a staging record without c_gate (test sets)")
    ap.add_argument("--simh-world", default="", help="the world name d9's manifest gives a scale-up sim-honest arm (09/d9's naming): enables part kind 'simh'")
    ap.add_argument("--ep2", action="store_true", help="part kind 'ep2' (09 Append 38: epoch-2 real honest; worlds honest_a / honest_b, T196's e2a / e2b; stratum = the cell's origin)")
    ap.add_argument("--ug-origin", default="", help="commit:path:sha256 of d9's run_table_ug.json (09 Q2/Q4 @ 48363962a): enables part kind 'ug' (world sim_ug; stratum = the cell's origin)")
    a = ap.parse_args()
    epochs, provenance = [], []
    sf, sv = None, None
    if a.stopped_verdict:  # 09: a staged 'stopped' run whose .eval header says success (d9's v2 record) ended; nothing else is admitted
        vc_, vp_, vsha = a.stopped_verdict.split(":", 2)
        vraw = show(vc_, vp_)
        if hashlib.sha256(vraw).hexdigest() != vsha:
            raise SystemExit("the header-verdict record differs from its sha256")
        sv = {"ref": {"commit": vc_, "path": vp_, "sha256": vsha},
              "ok": {x["record"]: x for x in json.loads(vraw)["records"] if x.get("eval_header_status") == "success" and x.get("excluded") is False and x.get("staged") is True}}
    if a.setup_failures:  # 09: runs whose environment setup failed before the first policy call count as staging exclusions
        sc_, sp_, ssha = a.setup_failures.split(":", 2)
        sraw = show(sc_, sp_)
        if hashlib.sha256(sraw).hexdigest() != ssha:
            raise SystemExit("the setup-failure record differs from its sha256")
        sd = json.loads(sraw)
        sf = {"ref": {"commit": sc_, "path": sp_, "sha256": ssha, "reason": sd.get("reason")}, "records": {x["record"] for x in sd["records"]}}
    cites = {}
    for c_ in a.cite:  # IL: d9's run tables and donor-sha records, cited by path @ commit with their sha256
        lab, rest = c_.split("=", 1)
        cc_, cp_ = rest.split(":", 1)
        cites[lab] = {"commit": cc_, "path": cp_, "sha256": hashlib.sha256(show(cc_, cp_)).hexdigest()}
    un = None
    if a.unusable:  # d9's submit-only (policy) honest runs: unusable, excluded from calibration pools
        uc_, up_, ush = a.unusable.split(":", 2)
        uraw = show(uc_, up_)
        if hashlib.sha256(uraw).hexdigest() != ush:
            raise SystemExit("the unusable-honest record differs from its sha256")
        un = {"ref": {"commit": uc_, "path": up_, "sha256": ush}, "records": {x["record"] for x in json.loads(uraw)["records"]}}
    if a.simh_world:  # a scale-up sim-honest arm (run_famexp_simh_scaleup.py): one world, named by d9 / 09
        WORLDS["simh"] = (a.simh_world,)
    origin = None
    if a.ug_origin:  # ug (Append 34): d9's registry label sim_ug; each cell's origin (S_old / S_new) from the registered 149-cell table
        oc_, op_, osh = a.ug_origin.split(":", 2)
        oraw = show(oc_, op_)
        if hashlib.sha256(oraw).hexdigest() != osh:
            raise SystemExit("run_table_ug differs from its sha256")
        origin = {c["cell"]: c["origin"] for c in json.loads(oraw)["cells"]}
        if len(origin) != 149 or set(origin.values()) != {"S_old", "S_new"}:
            raise SystemExit("run_table_ug: not 149 cells with origin S_old / S_new")
        WORLDS["ug"] = ("sim_ug",)
    if a.ep2:  # epoch-2 real honest (Append 38): registry stratum 'ep2'; the row's stratum is its origin by the pinned cell lists
        origin = {}
        for (c_, p_, s_), og in ((("4d1eec572", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json", "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb"), "S_old"),
                                 (("0caadbfb5", "experiments/2026-10-14_q2_scaleup_cohort/results/scaleup_tasks_plus311_2026-10-14_125135.json", "69f1f75313c9610f7d16b18225c1166268c0b6e084859c641d0213f293896a8b"), "S_new"),
                                 (("ef171c813", "experiments/2026-10-14_q3_new_mains_mode/config/new_mains_cells_v1.json", "98bfb97e26e7ec580b08643a3a38a5e66a0f6b63333f4013aa57559383402e8e"), "S_newmains")):
            b_ = show(c_, p_)
            if hashlib.sha256(b_).hexdigest() != s_:
                raise SystemExit(f"{p_}: sha256 differs from its pin")
            origin |= {c["key"]: og for c in json.loads(b_)["cells"]}
        WORLDS["ep2"] = ("honest_a", "honest_b", "e2a", "e2b")
    for spec in a.part:
        kind, commit, mpath, run, srec_path, set_key, dev = spec.split(":", 6)
        if kind not in WORLDS:
            raise SystemExit(f"part kind {kind}: not one of {sorted(WORLDS)}")
        raw = show(commit, mpath)
        man = json.loads(raw)
        devs = dev.split("+")  # one device record, or several attempts' records ("a1+a7")
        recs = [json.loads(show(commit, x)) for x in devs]
        gpus = [s_.get("gpu_name") for d in recs for s_ in d.get("shards", [])]
        att = None
        if not gpus or any(g not in ("NVIDIA H200", None) for g in gpus):
            raise SystemExit(f"{run}: device records show {sorted(set(map(str, gpus)))}, not all NVIDIA H200")
        if None in gpus:  # a shard logged no gpu_name: only an attestation for THIS root can admit it (4f/09, Append 21g)
            if not a.attest:
                raise SystemExit(f"{run}: a shard has no gpu_name and no --attest was given: refuse")
            ac, ap_, ash = a.attest.split(":", 2)
            araw = show(ac, ap_)
            at = json.loads(araw)
            root = [r for r in at.get("roots", {}).values() if (r.get("v1") or {}).get("root") == run]
            if hashlib.sha256(araw).hexdigest() != ash or at.get("pass") is not True or len(root) != 1 or \
                    not (root[0]["v1_vs_v2"].get("all_identical") is True and root[0]["v2"].get("every_shard_h200") is True
                         and root[0]["a1_without_gpu_name"].get("all_bit_exact_vs_v2") is True):
                raise SystemExit(f"{run}: the attestation does not pass for this root (sha, pass, all_identical, every_shard_h200, all_bit_exact_vs_v2)")
            att = {"commit": ac, "path": ap_, "sha256": ash, "verdict": at.get("verdict"), "v1_vs_v2": {k: root[0]["v1_vs_v2"].get(k) for k in ("path", "sha256", "all_identical")}}
        srec = json.loads(show(commit, srec_path))[set_key]
        staged, ssha = srec["out"], srec["assembled"]["staged_inputs_sha256"]
        if ("c_gate" not in srec or "excluded" not in srec["c_gate"]) and not a.test_no_c_gate:
            raise SystemExit(f"{srec_path}[{set_key}]: no c_gate.excluded (staging coverage unknown); refuse")
        excl = (srec.get("c_gate") or {}).get("excluded", {})
        try_ = re.search(r"_try\d+$", run)
        tsuf = try_.group(0) if try_ and not staged.endswith(try_.group(0)) else ""  # a staged set already named _try<n> keeps it once (nm360_s3_run2_v1_try2)
        folder = lambda arm: f"{staged}{tsuf}__{EP2_ARM.get(str(arm), str(arm)).split('@', 1)[0]}" + ("_ug" if kind == "ug" else "")  # noqa: E731  (09: one folder per (set, meta arm); "@..." dropped, as the DT writer; ug: __sim_attack_ug)
        rows = [r for r in man["files"] if r.get("world") in WORLDS[kind]]
        NOT_ENDED = "not ended (state {}): never counted (4f: gate Ctrl+C, selected on in-flight status, which correlates with run duration and possibly outcome)"
        ek = lambda r: f"{r['launch']}/{r['rel']}#e0"  # noqa: E731

        def why(r):
            st = r.get("state", "done")
            return excl[ek(r)] if st in ENDED else f"{NOT_ENDED.format(st)}; staging reason: {excl[ek(r)]}"

        dropped = [{"key": ek(r), "world": r["world"], "cell": r.get("task"), "arm": r.get("arm_dir"), "folder": folder(r.get("arm_dir")),
                    "state": r.get("state"), "reason": why(r)} for r in rows if ek(r) in excl]
        keep = [r for r in rows if f"{r['launch']}/{r['rel']}#e0" not in excl]
        ne = []
        for r in keep:
            if r.get("state", "done") in ENDED:
                continue
            v = (sv or {"ok": {}})["ok"].get(f"{r['launch']}/{r['rel']}")
            if r.get("state") != "stopped" or v is None or v.get("sha256") != r.get("sha256"):  # ENDED only, or exactly the runs d9's record says ended in success
                raise SystemExit(f"{mpath}: a run that did not end (state not in ENDED, no success verdict) is staged: refuse")
            ne.append({"key": f"{r['launch']}/{r['rel']}#e0", "world": r["world"], "cell": r.get("task"), "state": r.get("state"), "folder": folder(r.get("arm_dir")),
                       "eval_sha256": r.get("sha256"), "eval_header_status": v["eval_header_status"], "relabel_rule_outcome": v.get("relabel_rule_outcome"),
                       "verdict_record": sv["ref"]})
        if a.test_only_keys:  # TEST: only the runs the limited test extraction staged
            only = set(json.loads(Path(a.test_only_keys).read_text()))
            keep = [r for r in keep if f"{r['launch']}/{r['rel']}#e0" in only]
        bad = [r.get("rel") for r in keep if not {"launch", "rel", "task", "arm_dir", "world", "stratum"} <= set(r)]
        if bad or not keep:
            raise SystemExit(f"{mpath}: {len(bad)} rows without launch/rel/task/arm_dir/world/stratum, or no {WORLDS[kind]} rows")
        if kind == "ep2" and any(r["stratum"] != "ep2" or r["task"] not in origin for r in keep):
            raise SystemExit(f"{mpath}: an epoch-2 row whose registry stratum is not ep2 or whose cell is in no pinned list")
        if kind == "ug" and any(r["stratum"] != "sim_ug" or r["task"] not in origin for r in keep):
            raise SystemExit(f"{mpath}: a ug row whose registry stratum is not sim_ug or whose cell is not in run_table_ug")
        epochs += [{"key": f"{r['launch']}/{r['rel']}#e0", "cell": r["task"], "world": EP2_ARM.get(r["world"], r["world"]), "stratum": origin[r["task"]] if kind in ("ug", "ep2") else r["stratum"], "run": run,
                    "arm": r["arm_dir"], "staged": f"{staged}/inputs/{r['arm_dir']}.jsonl", "staged_sha256": ssha[f"{r['arm_dir']}.jsonl"],
                    "folder": folder(r["arm_dir"]), "launch_task": r.get("launch_task"), "eval_sha256": r.get("sha256"),
                    "row_meta": {k: r.get(k) for k in ROW_META if k in r or k == "pause_gate_at_start"} if r.get("interleaved_launch") else None}
                   | ({"registry_stratum": r["stratum"]} if kind in ("ug", "ep2") else {}) for r in keep]
        sf_keys = sorted(f"{r['launch']}/{r['rel']}#e0" for r in rows if sf and f"{r['launch']}/{r['rel']}" in sf["records"])
        un_keys = sorted(f"{r['launch']}/{r['rel']}#e0" for r in rows if un and f"{r['launch']}/{r['rel']}" in un["records"])
        provenance.append({"citations": cites, "unusable_honest": None if un is None else {**un["ref"], "keys": un_keys}, "ended_by_header_verdict": ne, "setup_failures": None if sf is None else {**sf["ref"], "keys": sf_keys,
                                                                          "staged": [k for k in sf_keys if k in {f"{r['launch']}/{r['rel']}#e0" for r in keep}]},
                           "kind": kind, "manifest": {"commit": commit, "path": mpath, "sha256": hashlib.sha256(raw).hexdigest()},
                           "run": run, "staged_set": staged, "staging_record": {"commit": commit, "path": srec_path, "set": set_key},
                           "excluded_by_staging": dropped, "device_record": {"commit": commit, "path": dev, "gpus": sorted({str(g) if g else "unrecorded (attested)" for g in gpus}), "shards": len(gpus), "attestation": att},
                           "rows": len(keep), "by_world": {w: sum(r["world"] == w for r in keep) for w in WORLDS[kind]}}
                          | ({"ug_origin": {"commit": oc_, "path": op_, "sha256": osh}} if kind == "ug" else {}))
    dup = len(epochs) - len({(e["world"], e["cell"]) for e in epochs})
    if dup:
        raise SystemExit(f"{dup} duplicate world|cell rows across parts")
    msg = {"d9_arm": a.d9_arm, "volume": a.volume, "provenance": provenance, "epochs": epochs} | ({"test": True} if a.test_only_keys or a.test_no_c_gate else {})
    out = json.dumps(msg, indent=1).encode()
    Path(a.out).write_bytes(out)
    print(json.dumps({"out": a.out, "sha256": hashlib.sha256(out).hexdigest(), "epochs": len(epochs), "parts": [p["by_world"] for p in provenance]}))


if __name__ == "__main__":
    main()
