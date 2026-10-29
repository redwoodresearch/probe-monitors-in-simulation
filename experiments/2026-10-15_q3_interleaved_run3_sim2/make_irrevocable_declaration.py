"""Draft the "rollout_decisions_irrevocably_fixed" declaration row for the IL rollout log (d9 11:2x), plus a manifest
listing every rollout-decision row by sha. Read-only on everything; prints the row and writes the manifest (never appends
to the log: d9 and 0d sign first). Keys and run states only.

  python make_irrevocable_declaration.py --reads <dir of sim_spend_read.py <host>.jsonl> [--reads <older dir> ...] \
      --extra-finished <json> --manifest-out <json> --decl-out <json> --index <json> [--final-index <path@commit>]

Conditions recorded in the row:
- every registered IL launch has ended: its launch.json "finished" value, its launch.json sha256 and where it was read
  (the newest read wins; a launch read on several boxes lists them all; copies must agree on "finished");
- launches registered but never created, and the aborted xbox3 q4c attempt (no launch.json), listed with their log rows;
- the only disclosed IL exposure (rows @ 8f769dac9, bfd18daf5, f1281ba0c);
- every rollout-decision row before this one, by line number, sha256 of the line and the commit that added it (git blame).
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
LOG = "results/il_rollout_decisions_2026-10-15.jsonl"
PT = ZoneInfo("America/Los_Angeles")
IL_LAUNCHES = [  # every IL launch that has a launch.json (registered: RULE.md; rlr, rlr2, rlr3, completions; placements)
    "2026-10-14_il_q1", "2026-10-14_il_q2", "2026-10-14_il_q3", "2026-10-14_il_q4", "2026-10-14_il_old",
    "2026-10-14_il_s1", "2026-10-14_il_s2", "2026-10-14_il_s3", "2026-10-14_il_s5", "2026-10-14_il_s4d",
    "2026-10-15_il_q1_rlr", "2026-10-15_il_q2_rlr", "2026-10-15_il_q3_rlr", "2026-10-15_il_old_rlr", "2026-10-15_il_s2_rlr",
    "2026-10-15_il_q4_rlr2", "2026-10-15_il_s1_rlr2", "2026-10-15_il_s2_rlr2",
    "2026-10-15_il_old_rlr3", "2026-10-15_il_q1_rlr3", "2026-10-15_il_q2_rlr3", "2026-10-15_il_q3_rlr3",
    "2026-10-15_il_q1c", "2026-10-15_il_q2c", "2026-10-15_il_q3c", "2026-10-15_il_oldc", "2026-10-15_il_q4c",
]
NEVER_CREATED = {  # registered designations that never got a launch.json (retired by later pinned records)
    "2026-10-15_il_q4_rlr": "never created: starter held 04:08 before any start (cap $225 would strand cells); retired by q4_rlr2 (il_rlr2_q4_supersession @ 6caac8427)",
    "2026-10-15_il_s1_rlr": "never started: retired by s1_rlr2 (il_rlr2_supersession @ 7a034803d)",
}
ABORTED = {"2026-10-15_il_q4c (xbox3 attempt)": "aborted 08:56:09 before any task, no launch.json; dir kept audit-only as xbox3:~/phase2/runs/2026-10-15_il_q4c.aborted_0855 (results/il_q4c_xbox3_abort_evidence_2026-10-15.json)"}
EXPOSURE = [{"commit": "8f769dac9", "action": "exposure_disclosed", "time_pt": "2026-10-15T08:40:43-07:00"},
            {"commit": "bfd18daf5", "action": "exposure_accepted", "time_pt": "2026-10-15T09:48:27-07:00"},
            {"commit": "f1281ba0c", "action": "name_correction", "time_pt": "2026-10-15T09:49:03-07:00"}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reads", action="append", required=True)
    ap.add_argument("--extra-finished", required=True)
    ap.add_argument("--manifest-out", required=True)
    ap.add_argument("--decl-out", required=True)
    ap.add_argument("--index", required=True, help="the resolution index JSON used for pending_rerun_designations")
    ap.add_argument("--final-index", default=None, help="path@commit of the FINAL index (pending must be 0)")
    ap.add_argument("--od-fields", default=None, help="0d's fields JSON (read only): declared_by, statement, concealment, released, read_holds, not_in_scope")
    ap.add_argument("--time-pt", default=None, help="the declaration's time_pt, from date at commit")
    ap.add_argument("--final-index-note", default=None)
    a = ap.parse_args()
    errors, seen = [], {}
    for i, d in enumerate(a.reads):  # the first --reads dir is the newest; older dirs fill launches on unreachable boxes
        for f in sorted(Path(d).glob("*.jsonl")):
            for line in f.read_text().splitlines():
                r = json.loads(line)
                name = Path(r.get("run_dir", "")).name
                if "error" in r or name not in IL_LAUNCHES:
                    continue
                e = seen.setdefault(name, {"reads": []})
                e["reads"].append({"box": r["host"], "finished": r["finished"], "launch_json_sha256": r["launch_json_sha256"],
                                   "read_time": r["read_time"], "reads_dir_rank": i, "source": f"{Path(d).name}/{f.name}"})
    for x in json.loads(Path(a.extra_finished).read_text()):  # e.g. q2c, read at its end (box2 stopped since)
        seen.setdefault(x["launch_id"], {"reads": []})["reads"].append(x | {"reads_dir_rank": -1})
    launches = []
    for lid in IL_LAUNCHES:
        rs = seen.get(lid, {}).get("reads", [])
        if not rs:
            errors.append(f"{lid}: no read")
            continue
        top = max(datetime.fromisoformat(r["read_time"]) for r in rs)  # Codex r37b: the newest read by time wins, whatever its source
        newest = [r for r in rs if datetime.fromisoformat(r["read_time"]) == top]
        done = [r for r in newest if r.get("finished")]
        if len(done) != len(newest):
            errors.append(f"{lid}: the newest read shows it unfinished on {[r['box'] for r in newest if not r.get('finished')]}")
            continue
        if len({r["finished"] for r in done}) != 1:
            errors.append(f"{lid}: copies disagree on finished: {sorted({r['finished'] for r in done})}")
        best = min(done, key=lambda r: r["reads_dir_rank"])
        launches.append({"launch_id": lid, "finished": best["finished"], "launch_json_sha256": best["launch_json_sha256"],
                         "read_on": sorted({r["box"].split()[0] for r in done}), "source": best["source"], "read_time": best.get("read_time")})
    # every rollout-decision row by sha and commit
    log = HERE / LOG
    lines = log.read_bytes().splitlines(keepends=True)
    blame = subprocess.run(["git", "blame", "-l", "--line-porcelain", LOG], capture_output=True, text=True, check=True, cwd=HERE).stdout
    commits = [ln.split()[0] for ln in blame.splitlines() if len(ln.split()) >= 3 and len(ln.split()[0]) == 40 and ln.split()[0].isalnum()]
    if any(set(c) == {"0"} for c in commits):
        errors.append("the rollout log has uncommitted lines (git blame all-zero commit); commit it before the manifest")
    if len(commits) != len(lines):
        errors.append(f"git blame gave {len(commits)} commits for {len(lines)} lines (is the log committed and clean?)")
    rows = []
    for i, ln in enumerate(lines, 1):
        r = json.loads(ln)
        rows.append({"line": i, "time_pt": r.get("time_pt"), "action": r.get("action"), "line_sha256": hashlib.sha256(ln).hexdigest(),
                     "commit": commits[i - 1][:12] if i - 1 < len(commits) else None})
    file_sha = hashlib.sha256(log.read_bytes()).hexdigest()
    exposures = [r for r in rows if r["action"] in ("exposure_disclosed", "exposure_accepted", "name_correction")]
    if len(exposures) != 3:
        errors.append(f"expected the 3 exposure rows, found {len(exposures)}")
    manifest = {"what": "every IL rollout-decision row before the declaration, by line sha256 and adding commit", "path": f"experiments/2026-10-15_q3_interleaved_run3_sim2/{LOG}",
                "file_sha256": file_sha, "n_rows": len(rows), "rows": rows}
    Path(a.manifest_out).write_text(json.dumps(manifest, indent=1) + "\n")
    mani_sha = hashlib.sha256(Path(a.manifest_out).read_bytes()).hexdigest()
    stops = {}  # the original 311 launches stopped on 0d's GO at 02:43-02:44 (graceful Ctrl+C): terminally stopped
    for r in rows:
        if r["action"] == "stop":
            full = json.loads(lines[r["line"] - 1])
            for lid in IL_LAUNCHES:
                if str(full.get("launch", "")).startswith(lid + " "):
                    stops[lid] = r
    for e in launches:
        st = stops.get(e["launch_id"])
        e["end_status"] = "terminally stopped" if st else "completed"
        e["evidence"] = {"launch_json_sha256": e.pop("launch_json_sha256"), "read_on": e.pop("read_on"), "read_source": e.pop("source"), "read_time": e.pop("read_time")}
        if st:
            e["evidence"]["stop_row"] = {"line": st["line"], "time_pt": st["time_pt"], "commit": st["commit"], "line_sha256": st["line_sha256"]}
    for lid, why in NEVER_CREATED.items():
        launches.append({"launch_id": lid, "end_status": "never created", "evidence": why})
    for lid, why in ABORTED.items():
        launches.append({"launch_id": lid, "end_status": "aborted before a launch.json (no task dispatched)", "evidence": why})
    # pending rerun designations under 09's rules (Append 35 section 2; rlr3 only when both arms had 0 attempts; at most one
    # extra designation per cell): a cell is pending if its authoritative designation is a replacement that never started and the
    # cell has not used its extra designation, or if an authoritative run failed with RateLimitError (infra) and has no designation
    fin_raw = None
    if a.final_index:
        path, commit = a.final_index.split("@")
        full = subprocess.run(["git", "rev-parse", "--verify", f"{commit}^{{commit}}"], capture_output=True, text=True, cwd=HERE).stdout.strip()
        if not full:
            errors.append(f"final index commit {commit} does not resolve")
        else:
            fin_raw = subprocess.run(["git", "show", f"{full}:{path}"], capture_output=True, check=True, cwd=HERE).stdout
    if a.final_index and not fin_raw:
        errors.append(f"final index {a.final_index} could not be read; no fallback")  # Codex r37b
    idx = json.loads(fin_raw) if fin_raw else json.loads(Path(a.index).read_text())
    idx_cite = (f"{a.final_index.split('@')[0]} @ {full} (sha256 {hashlib.sha256(fin_raw).hexdigest()[:16]}...)" if fin_raw
                else f"{Path(a.index).name} (sha256 {hashlib.sha256(Path(a.index).read_bytes()).hexdigest()[:16]}...)")
    if idx.get("pass") is not True:
        errors.append("the index used for pending_rerun_designations does not pass")
    pend = []
    for r in idx["rows"]:
        if r["replacement_kind"] in ("rlr", "rlr2", "rlr3", "completion") and r["status"] == "never started":
            pend.append({"key": r["key"], "launch_id": r["launch_id"], "why": "authoritative replacement never started"})
        if r["status"].startswith("run_failed") and "awaiting routing" in r["status"]:
            pend.append({"key": r["key"], "launch_id": r["launch_id"], "why": "failure not yet classified (awaiting routing): a rate-limit class would make it rerun-eligible"})
        if "RateLimit" in r["status"] or "RateLimit" in (r.get("staging_reason") or ""):
            pend.append({"key": r["key"], "launch_id": r["launch_id"], "why": "rate-limit failure"})
    fin = None
    if fin_raw:
        path, commit = a.final_index.split("@")
        commit = full
        raw = fin_raw
        fi = json.loads(raw)
        npend = sum(1 for r in fi["rows"] if r["status"].startswith("pending"))
        if npend or not fi.get("pass"):
            errors.append(f"final index {a.final_index}: pending {npend}, pass {fi.get('pass')}")
        fin = {"path": path, "commit": commit, "sha256": hashlib.sha256(raw).hexdigest(), "pending": npend, "pass": fi.get("pass"),
               "ended_awaiting_routing": sum(1 for r in fi["rows"] if r["status"] == "ended; awaiting routing"), "note": a.final_index_note}
    decl = {"declaration": "irrevocably_fixed",
            "rule": {"source": "09, aswin/convergence-audit-09: experiments/2026-10-14_priority1_spec_09/RULE.md, Append 35 section 2 'Release'",
                     "text": "IL outcomes are released only after every registered IL launch, replacements included, has completed or been terminally stopped, AND the timestamped \"irrevocably fixed\" declaration explicitly closes all replacement, resumption and additional-launch decisions. Later reruns cannot amend that registered result. The concealment audit continues through replacement decisions."},
            "declared_by": None, "assembled_by": "d9 (facts assembled by the design fork, by script: make_irrevocable_declaration.py)",
            "time_pt": None, "statement": None, "concealment": None, "released": None, "read_holds": None,
            "launches": launches,
            "pending_rerun_designations": {"count": len(pend), "by_key": pend,
                                           "evidence": f"resolution index {idx_cite}: no authoritative replacement is 'never started' and no authoritative run failed with RateLimitError; the cells 09 ruled not rerun-eligible are cited per row there"},
            "final_resolution_index": fin,
            "disclosed_exposure_rows": EXPOSURE,
            "rollout_rows_before": {"path": manifest["path"], "n_rows": len(rows), "file_sha256": file_sha,
                                    "manifest": f"experiments/2026-10-15_q3_interleaved_run3_sim2/results/{Path(a.manifest_out).name}", "manifest_sha256": mani_sha}}
    if a.od_fields:
        od = json.loads(Path(a.od_fields).read_text())
        for k in ("declared_by", "statement", "concealment", "released", "read_holds", "not_in_scope"):
            if k in od:
                decl[k] = od[k]
        decl["od_fields_sha256"] = hashlib.sha256(Path(a.od_fields).read_bytes()).hexdigest()
    if a.time_pt:
        decl["time_pt"] = a.time_pt
    if errors:  # Codex r37: never write an unqualified declaration
        decl = {"declaration": "INVALID: not a declaration (errors below)", "errors": errors, "draft": decl}
    Path(a.decl_out).write_text(json.dumps(decl, indent=1) + "\n")
    print(json.dumps({"launches": len(launches), "il_launch_jsons": len(IL_LAUNCHES), "pending_rerun": len(pend), "final_index": fin, "rollout_rows": len(rows), "file_sha256": file_sha[:16], "errors": errors}))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
