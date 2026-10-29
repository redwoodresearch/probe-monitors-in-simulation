"""Stage B pooled-RMS record: a KEYS-ONLY row index for 09 (2e, 10-15 22:4x). On Modal: re-hash the record on the Volume (stop on a mismatch), then emit one
row per probe x row kind x action level x readout x scenario x set with its status (computed | withheld + verbatim label | absent), complete draws, and
every label field present. No value is read out: numeric fields other than counts are never copied. POST HOC, PRELIMINARY.

  PYTHONPATH=src modal run experiments/2026-10-15_il_analysis_09/index_stageB_pooled.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REC = ("xerr/il/records/il_posthoc_stageB_pooled_2026-10-15_222027_ap-9mhXzkrOinj65yzNcOqb64.json",
       "d41a3647772ef0ec8fa2e4b4a71225dff6a31c418df5d8980532d0255cf02a55")
app = modal.App("probe-shift_aswin-surya_sub-ilz_kill-2026-11-14")
vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14")
BODY = {"sol_a": "Sol (a)", "Sol (a)": "Sol (a)", "fibs": "FIBS", "FIBS": "FIBS", "harmfulness": "harmfulness", "refusal": "refusal",
        "highstakes": "high-stakes", "high-stakes": "high-stakes", "own_frac100_L48": "own_frac100_L48", "c_asis": "c_asis"}
BODY6 = ["Sol (a)", "FIBS", "harmfulness", "refusal", "high-stakes", "own_frac100_L48"]
OWN_LABEL = "real run 1 shares the probe's training launch"  # RULE @ 95dd0a8bc (own_frac: S_old cells only)
COUNTS = ("complete_draws", "n_tasks", "cells_used", "mains", "N")


@app.function(cpu=1, memory=4096, timeout=900, retries=0, volumes={"/w": vol.read_only()},
              image=modal.Image.debian_slim(python_version="3.12"))
def build() -> dict:
    b = (Path("/w") / REC[0]).read_bytes()
    got = hashlib.sha256(b).hexdigest()
    if got != REC[1]:
        raise SystemExit(f"record sha256 {got} differs from the pin; stop")
    d = json.loads(b)
    ph = d["result"]["il_posthoc_POST_HOC"]
    rows = []

    def labels_of(o):  # every string field whose key names a label, note, scenario, role or source; never a number
        return {k: v for k, v in o.items() if isinstance(v, str) and any(t in k for t in ("label", "note", "scenario", "role", "source"))} if isinstance(o, dict) else {}

    def emit(probe, kind, level, readout, scenario, set_, block, extra=None):
        st = "computed" if isinstance(block, dict) and "bias_squared" in block else "withheld" if block is not None else "absent"
        rows.append({"probe": BODY.get(probe.rsplit(" ", 1)[0] if probe.endswith((" R", " C")) else probe, probe), "probe_key": probe, "kind": kind,
                     "action_level": level, "readout": readout, "scenario": scenario, "set": set_, "status": st,
                     "labels": labels_of(block)} | (extra or {}))

    # run-level rows (score level, run-level catch) per set
    for scope, sets in (("primary", ("S_old", "S_new_on_M")), ("extended", ("S_newmains",))):
        for rk, r in ph["rms_bias"][scope].items():
            if not isinstance(r, dict) or "scope" not in r:
                continue
            rd, a = rk.split("|")
            for s_ in sets:
                for kind in ("score_level", "catch_level_real_line_run_level"):
                    blk = r[s_].get(kind) or {}
                    mem = blk.get("members") or []
                    for p in mem:
                        for sc, key in (("A", "A"), ("B", "B"), ("run3 reference", "reference_run3")):
                            emit(p, kind, a, rd, sc, s_, (blk.get("per_probe") or {}).get(p, {}).get(key),
                                 {"complete_draws": blk.get("complete_draws"), "row_labels": labels_of(blk), "members": mem})
                    for p, why in (r.get("score_level_withheld_probes") or {}).items() if kind == "score_level" else []:
                        rows.append({"probe": BODY.get(p, p), "probe_key": p, "kind": kind, "action_level": a, "readout": rd, "scenario": "-", "set": s_,
                                     "status": "withheld", "labels": {"withheld": why if isinstance(why, str) else json.dumps(why)}})
    # per-action rows per set (members-mean per probe) and the extension note
    for s_, blk in ph["rms_bias_per_action"].items():
        for rk, r in blk["rows"].items():
            ln, pt, rd = rk.split("|")
            mm = r.get("members_mean") or {}
            for p, role in (r.get("roles") or {}).items():
                for sc, key in (("A", "A"), ("B", "B"), ("run3 reference", "reference_run3")):
                    emit(p, f"per_action:{ln}", pt, rd, sc, s_, (mm.get("per_probe") or {}).get(p, {}).get(key) if role == "member" else None,
                         {"role": role, "complete_draws": mm.get("complete_draws"), "export_labels": blk.get("export_labels") or {}})
            for p, note in (r.get("pooled_extension_note") or {}).items():
                rows.append({"probe": BODY.get(p.rsplit(" ", 1)[0], p), "probe_key": p, "kind": f"per_action:{ln}", "action_level": pt, "readout": rd,
                             "scenario": "-", "set": "extension", "status": "stashed" if note.startswith("extension row stashed") else "not used",
                             "labels": {"pooled_extension_note": note}})
    # pooled rows
    pb = ph["rms_bias_pooled_across_sets"]
    for rk, r in pb["rows"].items():
        kind, rest = rk.split("|", 1)
        if kind == "per_action":
            ln, pt, rd = rest.split("|")
            kind, level = f"per_action:{ln}", pt
        else:
            rd, level = rest.split("|")
        for p in r.get("members") or []:
            for sc, key in (("A", "A"), ("B", "B"), ("run3 reference", "reference_run3")):
                emit(p, kind, level, rd, sc, "pooled", (r.get("per_probe") or {}).get(p, {}).get(key),
                     {"complete_draws": r.get("complete_draws"), "S_newmains_source": (r.get("S_newmains_source") or {}).get(p),
                      "extension_rows": (r.get("extension_rows") or {}).get(p), "row_label": r.get("label")})
        for p, why in (r.get("withheld_probes") or {}).items():
            rows.append({"probe": BODY.get(p.rsplit(" ", 1)[0] if p.endswith((" R", " C")) else p, p), "probe_key": p, "kind": kind, "action_level": level,
                         "readout": rd, "scenario": "A/B/run3 reference", "set": "pooled", "status": "withheld", "labels": {"withheld": why},
                         "withheld_detail": (r.get("withheld_detail") or {}).get(p)})
        if r.get("note") and not r.get("members"):
            rows.append({"probe": "-", "kind": kind, "action_level": level, "readout": rd, "set": "pooled", "status": "no pooled members",
                         "labels": {"note": r["note"]}})
    return {"rows": rows, "record_sha256": got, "record_git": d["git"], "pooled_label": pb.get("label"), "pooled_rule": pb.get("rule"),
            "all_row": pb.get("all_row"), "numerics_note": d["spec"].get("stage_b_numerics_note")}


@app.local_entrypoint()
def main():
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("uncommitted tracked changes")
    res = build.remote()
    rows = res["rows"]
    present = {r["probe"] for r in rows if r["status"] != "absent"}
    flags = {}
    one = [r for r in rows if r["set"] == "pooled" and r["readout"] == "R" and r["action_level"] in ("0.01",) and r["status"] in ("computed", "withheld")]
    flags["a_pooled_or_withheld_at_1pct_reply"] = {
        p: sorted({f"{r['kind']}: {r['status']}" for r in one if r["probe"] == p}) or ["ABSENT from every pooled 1% reply row"] for p in BODY6}
    flags["b_sol_a_extension_label"] = sorted({json.dumps(r.get("extension_rows"), ensure_ascii=False) for r in rows
                                               if r["probe"] == "Sol (a)" and r["set"] == "pooled" and r.get("extension_rows")})
    own = [r for r in rows if r["probe"] == "own_frac100_L48"]
    flags["c_own_frac100"] = {"rows_in_record": len(own), "label_present": any(OWN_LABEL in json.dumps(r) for r in own),
                              "S_newmains_rows": len([r for r in own if r["set"] == "S_newmains"]),
                              "reading": "ABSENT: own_frac100_L48 is in no RMS block of this record (not in the stage B exports' probe lists, not a score-level member)"
                              if not own else "present"}
    ref = [r for r in rows if r.get("scenario") == "run3 reference" and r["set"] == "pooled"]
    flags["d_run3_reference_pooled"] = {"pooled_reference_rows": len(ref), "computed": sum(r["status"] == "computed" for r in ref),
                                        "same_rule": "pooled_rms pools b2_ref with the same per-draw numerator/weight sums as A and B (rule text in the record's pooled block)"}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = {"label": "KEYS-ONLY row index of the stage B pooled-RMS record (no values); POST HOC, PRELIMINARY",
           "record": {"volume": "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", "path": REC[0], "sha256": res["record_sha256"], "git": res["record_git"]},
           "pooled_label": res["pooled_label"], "pooled_rule": res["pooled_rule"], "all_row": res["all_row"], "numerics_note": res["numerics_note"],
           "body_probes": BODY6, "body_probes_present": sorted(present & set(BODY6)), "body_probes_absent": sorted(set(BODY6) - present),
           "flags": flags, "n_rows": len(rows), "rows": rows, "meta": {"git": git("rev-parse", "HEAD").decode().strip(), "stamp_pt": stamp,
                                                                       "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    b = json.dumps(out, indent=1, ensure_ascii=False).encode()
    fn = HERE / "results" / f"stageB_pooled_index_{stamp}_{app.app_id}.json"
    fn.open("xb").write(b)
    print(fn.name, hashlib.sha256(b).hexdigest(), json.dumps(flags, ensure_ascii=False)[:1500])
