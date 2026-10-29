"""Figure 4 gate and count (keys, labels and counts only; no score, gap or catch value leaves the container). Run BEFORE the Figure 4 render on
the record it will draw: it applies post_fig4_sweep.gate(), the ONE predicate the renderer also STOPs on, to the plan's Figure 4 inputs and
reports every failure (keys and labels) plus counts (not-readable flags per pool type, degenerate intervals, dropped resamples, task counts,
whether the B-estimator keys are present: information only). all_pass = the record's sha256 matches the plan's and gate() lists no failure.
The record, its sha256, the rows, origin, average group, record type and per-probe gray switch come from the render plan's Figure 4 entry (one
source for both runs); the plan's comparator must be run_mean and its estimator "A" (Fig 1's, 507-prep 10-16), else STOP before any Modal
call. Exits non-zero (after writing the JSON) unless all_pass. Also reports the container's peak memory after loading the record (render_all
gives each figure container 4 GiB).

  uv run modal run --detach fig4_gate_count.py::gate_main --stamp _fig4_sweep15 --which sweep15_perset
  uv run modal run --detach fig4_gate_count.py::gate_main --stamp _fig4_pooledB --which pooled_world_B
  "$M" run --detach fig4_gate_count.py::gate_main --stamp _fig4_routecB_<stamp> --which pooled_world_B_routec   (route C: one plan for gate
    and render, fill tokens STOP, --label-sidecar / --label-sha / --record-meta required, B's meta re-asserted in the container, the
    fixed-interval predicate applied when the plan draws fixed+joint)
--which picks one plan from the fixed PLANS table (no free path), and the run STOPs before any function call when this script,
post_fig4_sweep.py or that plan differ from HEAD (an uncommitted or edited input is never run; the wrappers pin their reviewed sha256s too).
Writes results/fig4_gate_count<stamp>.json (exclusive create) when the client is still attached; the same JSON is in the app log.
"""

import hashlib
import json
import resource
import subprocess
import sys
from pathlib import Path

import modal
from post_fig4_sweep import GROUP, ORIGIN, gate, gate_mode

HERE = Path(__file__).resolve().parent
PLANS = {"sweep15_perset": "config/render_plan_fig4_sweep15.json", "pooled_world_B": "config/render_plan_fig4_pooledB.json",
         "pooled_world_B_routec": "config/render_plan_fig4_pooledB_routec.json"}  # route C (10-17): the ONE plan both the gate and the render read
RECORD_TYPE = {"pooled_world_B_routec": "pooled_world_B"}  # --which -> the plan's --record-type, where they differ
ROUTEC = ("pooled_world_B_routec",)  # plans that must carry --label-sidecar, --label-sha and --record-meta (09 / 2e, 10-18)

app = modal.App("probe-shift_aswin-surya_sub-routec-figs_f4gate_kill-2026-11-14")  # route C's own description (was render_fig4-gate_kill-2026-11-13)
image = (modal.Image.debian_slim(python_version="3.12").add_local_file(HERE / "post_fig4_sweep.py", "/root/post_fig4_sweep.py")
         .add_local_file(HERE / "postfig.py", "/root/postfig.py").add_local_file(HERE / "figio.py", "/root/figio.py"))  # check_record_meta (stdlib)
SOL = modal.Volume.from_name("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", version=2)  # Stage A's records (read only here)


@app.function(image=image, cpu=1, memory=8192, timeout=600, retries=0, volumes={"/sol": SOL})  # the record is 276 MB of JSON
def gate_count(path: str, sha: str, rows: list, origin: str, group: str, est: str, record_type: str, probe_gray: str,
               intervals: str = "joint", record_meta: list = None) -> dict:
    b = Path("/sol", path).read_bytes()
    got = hashlib.sha256(b).hexdigest()
    res = {"record": {"path": f"sol:{path}", "sha256_expected": sha, "sha256_ok": got == sha}, "fails": None}
    if got == sha:
        rec = json.loads(b)
        del b
        meta_fail, meta_ok = [], None
        if record_meta is not None:  # route C (09 S2 item 10 via 2e, 10-18): the record's meta by key FIRST, the render's own check (postfig)
            from postfig import check_record_meta

            try:
                meta_ok = check_record_meta(rec.get("meta"), *record_meta)
            except SystemExit as e:  # keys only in the message; listed as a failure so all_pass is false
                meta_fail = [f"record meta: {e}"]
        fails, info = gate(rec, rows, origin, group, est, record_type, probe_gray)
        fails = meta_fail + fails
        if meta_ok is not None:
            info["record_meta"] = meta_ok
        if intervals == "fixed+joint":  # route C: the renderer's own fixed-interval predicate too (post_fig4_sweep main), so the gate is the render's
            f_fx, info["fixed"] = gate_mode(rec, rows, origin, group, est, record_type, probe_gray, "fixed", info["levels"] or ())
            fails += f_fx
        res |= {"fails": fails, "n_fails": len(fails), "info": info}
    res["all_pass"] = got == sha and res["n_fails"] == 0
    res["peak_rss_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)  # Linux reports KiB
    print(json.dumps(res, sort_keys=True))  # keys, labels and counts only: the app log keeps a copy
    return res


@app.local_entrypoint()
def gate_main(stamp: str, which: str):
    if which not in PLANS:
        sys.exit(f"STOP: --which {which!r} is not one of {sorted(PLANS)}")
    plan = PLANS[which]
    out = HERE / "results" / f"fig4_gate_count{stamp}.json"
    if out.exists():
        sys.exit(f"{out} exists (results are append-only)")
    pb = (HERE / plan).read_bytes()
    if b"<FILL" in pb or b"TO_FILL" in pb:  # route C: the plan's A / B pins are fill tokens until 09's S2 (fail closed before any call)
        sys.exit(f"STOP: {plan} still holds fill tokens")
    pl = json.loads(pb)
    if pl.get("comparator") != "run_mean" or pl.get("estimator") != "A":
        sys.exit(f"STOP: the plan's comparator/estimator are {pl.get('comparator')!r}/{pl.get('estimator')!r}; Figure 4 needs run_mean/A (Fig 1's)")
    figs = [f for f in pl["figures"] if f["script"] == "post_fig4_sweep.py"]
    if len(figs) != 1:
        sys.exit(f"STOP: {len(figs)} post_fig4_sweep.py entries in the plan (one expected)")
    fa = figs[0]["args"]

    def arg(flag, default=None, n=1):
        if flag not in fa:
            return default
        i = fa.index(flag)
        return fa[i + 1:i + 1 + n] if n > 1 else fa[i + 1]
    base, sha, _ = arg("--record", n=3)
    src = pl["records"][base]
    if not src.startswith("sol:"):
        sys.exit(f"STOP: the record source {src!r} is not on the solavg Volume")
    i = fa.index("--rows") + 1
    rows = fa[i:i + 6]
    if len(rows) != 6 or any(r.startswith("--") for r in rows) or (i + 6 < len(fa) and not fa[i + 6].startswith("--")):
        sys.exit(f"STOP: six --rows expected in the plan, got {fa[i:]}")
    origin, group = arg("--origin", ORIGIN), arg("--average-group", GROUP)
    record_type, probe_gray = arg("--record-type"), arg("--probe-gray", "none")
    intervals = arg("--intervals", "joint")
    if record_type != RECORD_TYPE.get(which, which):
        sys.exit(f"STOP: the plan's --record-type {record_type!r} is not {RECORD_TYPE.get(which, which)!r}")
    rmeta = None
    if which in ROUTEC:  # 09's label rule (B's own sidecar, a git: plan record, its label sha pinned) and the B meta re-assert; no typed status
        sc_, ls_, rmeta = arg("--label-sidecar", n=3), arg("--label-sha"), arg("--record-meta", n=3)
        hx = lambda x, n: isinstance(x, str) and len(x) == n and all(c in "0123456789abcdef" for c in x)  # noqa: E731
        if not (sc_ and hx(ls_, 64) and rmeta and hx(rmeta[0], 40) and hx(rmeta[1], 64) and hx(rmeta[2], 64)) or "--text-note" in fa \
                or not str(pl["records"].get(sc_[0], "")).startswith("git:"):
            sys.exit("STOP: a route C plan needs --label-sidecar (a git: plan record), --label-sha and --record-meta, and no --text-note")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", __file__, str(HERE / "post_fig4_sweep.py"), str(HERE / "postfig.py"),
                            str(HERE / "figio.py"), str(HERE / plan)], capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        sys.exit(f"STOP: uncommitted changes in the gate, the renderer or the plan:\n{dirty}")
    res = gate_count.remote(src[4:].lstrip("/"), sha, rows, origin, group, pl["estimator"], record_type, probe_gray, intervals, rmeta)
    res["plan"] = {"file": plan, "sha256": hashlib.sha256(pb).hexdigest(), "comparator": pl["comparator"], "estimator": pl["estimator"],
                   "record_type": record_type, "probe_gray": probe_gray, "rows": rows, "origin": origin, "average_group": group,
                   "intervals": intervals, "record_meta": rmeta}
    res["script"] = {"file": Path(__file__).name, "gate_from": "post_fig4_sweep.py", "git": head, "dirty": bool(dirty),
                     "app_id": getattr(app, "app_id", None)}
    with open(out, "x") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print(out, "all_pass =", res["all_pass"], "| n_fails =", res.get("n_fails"), "| not readable:",
          json.dumps((res.get("info") or {}).get("not_readable")), "| peak_rss_mib =", res["peak_rss_mib"])
    if not res["all_pass"]:
        sys.exit(1)
