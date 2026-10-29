"""Reference and honest-log audit of one simulated-only follow-up launch, for q2/df's S_new eligibility replay (d9,
2026-10-15). ON THE BOX, read-only, keys and paths only: per cell its reference record (source, reused_from, copy,
sha256) and whether the source is its own main launch's honest_a; per simulated attempt the BA_HONEST_LOG path and
whether it is the cell's own pinned references/<task>.eval; the honest-log settings in force; the not-run reasons
split into "no pinned reference" and "never started before a pause or cut"; the launch.json sha256, input_sha256 and
the sims tree's src_sha256.  python3 reference_audit.py <run dir>   (prints one JSON line)"""
import collections, hashlib, json, re, sys
from pathlib import Path

run = Path(sys.argv[1])
raw = (run / "launch.json").read_bytes()
d = json.loads(raw)
meta = json.loads(Path(f"{run}.meta.json").read_text())
main = Path(meta["references_from"]).name
refs = d["references"]
cells, attempts, settings = [], [], collections.Counter()
counts = collections.Counter()
for r in d["runs"]:
    task = r["task"] if isinstance(r["task"], str) else r["task"]["key"]
    ref = refs.get(task)
    src = (ref or {}).get("source") or ""
    base = f"/home/aswin/phase2/runs/{main}/honest_a/"
    sub = src[len(base):].split("/", 1)[0] if src.startswith(base) else None
    kind = (None if ref is None else "own_main_honest_a" if sub == task else
            "own_main_honest_a_rerun_or_redraw" if sub and re.fullmatch(re.escape(task) + r"_(rerun|redraw)\d+", sub) else "other")
    own = kind in ("own_main_honest_a", "own_main_honest_a_rerun_or_redraw") if ref else None
    cells.append({"cell": task, "run_key": r["key"], "state": r["state"], "why": r.get("why"),
                  "reference": None if ref is None else {k: ref.get(k) for k in ("source", "reused_from", "copy", "sha256", "try")},
                  "reference_source_is_own_main_honest_a": own, "reference_source_kind": kind})
    if r["state"] == "not run":
        w = (r.get("why") or "").lower()
        counts["not_run_no_pinned_reference" if "no pinned reference" in w else
               "not_run_never_started_before_pause_or_cut" if ("stopped" in w or "launcher" in w or "pause" in w) else
               "not_run_other"] += 1
    elif r["state"] == "stopped":
        counts["stopped_in_flight_at_cut"] += 1
    else:
        counts[r["state"]] += 1
    pinned = str(run / "references" / f"{task}.eval")
    for n, a in enumerate(r["attempts"], 1):
        ev = a.get("env_vars") or {}
        log = ev.get("BA_HONEST_LOG")
        s = {k: ev.get(k) for k in ("BA_HONEST_LOG_MODE", "BA_HONEST_LOG_OUT_CAP", "BA_CODEBASE_EXEC", "BA_CODEBASE_READ")}
        settings[json.dumps(s, sort_keys=True)] += 1
        attempts.append({"run_key": r["key"], "attempt": n, "kind": a.get("kind"), "BA_HONEST_LOG": log,
                         "equals_own_pinned_reference": log == pinned, "equals_reference_copy": bool(ref) and log == ref.get("copy")})
not_why = collections.Counter((r.get("why") or "")[:80] for r in d["runs"] if r["state"] == "not run")
print(json.dumps({
    "launch": run.name, "references_from_main": main, "launch_json_sha256": hashlib.sha256(raw).hexdigest(),
    "input_sha256": d.get("input_sha256"), "sims_tree_src_sha256": {k: v.get("src_sha256") for k, v in (d.get("trees") or {}).items()},
    "finished": d.get("finished"), "counts": dict(counts), "not_run_reasons": dict(not_why),
    "honest_log_settings_in_force": {k: v for k, v in settings.items()},
    "cells_with_reference": sum(c["reference"] is not None for c in cells),
    "reference_source_own_main_honest_a": sum(c["reference_source_is_own_main_honest_a"] is True for c in cells),
    "reference_source_kinds": dict(collections.Counter(c["reference_source_kind"] for c in cells if c["reference"])),
    "reference_source_rerun_or_redraw": sorted(c["cell"] for c in cells if c["reference_source_kind"] == "own_main_honest_a_rerun_or_redraw"),
    "reference_source_other": sorted(c["cell"] for c in cells if c["reference_source_kind"] == "other"),
    "attempts": len(attempts), "attempts_log_equals_own_pinned": sum(a["equals_own_pinned_reference"] for a in attempts),
    "attempts_log_not_own_pinned": [a for a in attempts if not a["equals_own_pinned_reference"]],
    "cells": cells, "attempt_rows": attempts}))
