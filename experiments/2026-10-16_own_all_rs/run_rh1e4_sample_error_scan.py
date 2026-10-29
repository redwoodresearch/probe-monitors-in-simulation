"""09's rh1e4 gate "sample_error null" (4f, 10-16): ONE shared scan, citable by every scorer, of the SAMPLE-level error of all 963 staged rh1e4
relaunch epochs (the keys of d9's sold__rh1e4_stage_cutfull_r2.json @ 703ccbe0f3fb, equal by key to inventory_b_fix_v1's relaunch rows), read from each
.eval on Modal CPU, not the log status. Every file's sha256 is checked against the registered upload manifest (manifest_rh1e4_eval_upload_v1.json @
7bbc0e3df) before it is read. Positive controls: the 22 counted runs the honest_b_relaunch members record (@ 051899844) excludes as unusable (17 "error
after start"). Per run it keeps ONLY the log status, the number of samples and, per sample, the error's CLASS (or null) and the limit type; no message,
tool or transcript text. Counts and keys only. A copy of run_honest_eval_scan.py's reader (@ fecf1c5f1) without its error-message field.
  modal run --detach experiments/2026-10-16_own_all_rs/run_rh1e4_sample_error_scan.py"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-p2probes_rh1e4scan_kill-2026-11-14"
VOL = "probe-shift_aswin-surya_sub-q3realhonest_data_kill-2026-11-14"  # the manifest's source: <volume>:/evals_v1/<launch>/...
SOLD = ("703ccbe0f3fb", "experiments/2026-10-14_q3_overnight_chain/results/cutfull/routing/sold__rh1e4_stage_cutfull_r2.json", "b37c94f7")
INV = ("09b070617", "experiments/2026-10-15_cut_sensitivity/results/inventory_b_fix_v1_2026-10-16_025058.json", "4a5d0fb3cad32740f849df5def3c7a03a40a18aa4c42bc360709f0c740973d03")
MAN = ("7bbc0e3df", "experiments/2026-10-14_q3_q35_realhonest_1e4/config/manifest_rh1e4_eval_upload_v1.json")
MEMBERS = ("051899844", "experiments/2026-10-15_fpr_sweep_507_prep/results/honest_b_relaunch_members_2026-10-16_032731.json")
image = modal.Image.debian_slim(python_version="3.12").pip_install("inspect_ai==0.3.251", "zstandard==0.25.0")
app = modal.App(APP, image=image)
vol = modal.Volume.from_name(VOL, version=2)


@app.function(volumes={"/rh": vol.read_only()}, cpu=2, memory=4096, timeout=1800, retries=0)
def scan(chunk: list) -> list:
    import os
    import re

    from inspect_ai.log import read_eval_log

    cls = re.compile(r"\b([A-Z][A-Za-z]*(?:Error|Exception|Timeout|Exceeded))\b")

    def klass(err):
        if err is None:
            return None
        m = cls.search(err.message or "") or cls.search(err.traceback or "")
        return m.group(1) if m else "unclassified"

    out = []
    for r in chunk:
        p = f"/rh/evals_v1/{r['path']}"
        rec = {"key": r["key"], "role": r["role"]}
        if not os.path.isfile(p):
            out.append(rec | {"missing": True})
            continue
        if hashlib.sha256(open(p, "rb").read()).hexdigest() != r["sha256"]:
            out.append(rec | {"sha_mismatch": True})
            continue
        log = read_eval_log(p)
        ss = [{"error_class": klass(s.error), "limit": getattr(s.limit, "type", None) if getattr(s, "limit", None) else None} for s in log.samples or []]
        out.append(rec | {"sha_ok": True, "log_status": log.status, "n_samples": len(ss), "samples": ss})
    return out


@app.local_entrypoint()
def main():
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no").strip() or git("rev-parse", "HEAD") != git("rev-parse", "origin/aswin/own-all-loo"):
        raise SystemExit("uncommitted tracked changes, or HEAD is not on origin")
    sb, ib, mb, eb = (git("show", f"{c}:{p}") for c, p, *_ in (SOLD, INV, MAN, MEMBERS))
    if not hashlib.sha256(sb).hexdigest().startswith(SOLD[2]) or hashlib.sha256(ib).hexdigest() != INV[2]:
        raise SystemExit("the sold__ record or inventory_b_fix_v1 differs from its pin")
    (arm,) = json.loads(sb)["arms"]
    keys = sorted(arm["keys"])
    inv = {r["key"]: r for r in json.loads(ib)["runs"] if r["world"] == "relaunch"}
    man = {f["path"]: f for f in json.loads(mb)["files"]}
    by_sha = {f["sha256"]: f for f in man.values()}
    if len(keys) != 963 or set(keys) != set(inv):
        raise SystemExit(f"the sold__ keys ({len(keys)}) differ from inventory_b_fix_v1's relaunch rows ({len(inv)})")
    rows, bad = [], []
    for k in keys:
        path = k.removesuffix("#e0")
        f = man.get(path)
        if not f or f["sha256"] != inv[k]["eval_sha256"]:
            bad.append(k)
            continue
        rows.append({"key": k, "path": path, "sha256": f["sha256"], "role": "staged"})
    if bad:
        raise SystemExit(f"STOP: {len(bad)} staged keys without a registered upload-manifest file of the inventory's .eval sha, e.g. {bad[:2]}")
    mem = json.loads(eb)
    ctl_keys = {}
    for x in mem["excluded"]:  # positive controls: counted but unusable; the staged ones (staged outside membership) are scanned once, as staged
        f = by_sha.get(x["eval_sha256"])
        if not f:
            raise SystemExit(f"control {x['key']}: no manifest file for its sha")
        ctl_keys[f"{f['path']}#e0"] = x["cause"]
        if f"{f['path']}#e0" not in set(keys):
            rows.append({"key": f"{f['path']}#e0", "path": f["path"], "sha256": f["sha256"], "role": "control_unusable"})
    chunks = [rows[i:i + 40] for i in range(0, len(rows), 40)]
    res = [x for part in scan.map(chunks) for x in part]
    if len(res) != len(rows) or any(not x.get("sha_ok") for x in res):
        raise SystemExit(f"STOP: {sum(bool(x.get('missing')) for x in res)} missing, {sum(bool(x.get('sha_mismatch')) for x in res)} sha mismatches")
    st = [x for x in res if x["role"] == "staged"]
    ctl = [x for x in res if x["key"] in ctl_keys]
    se = lambda x: next((s["error_class"] for s in x["samples"] if s["error_class"]), None)  # noqa: E731
    nonnull = sorted(x["key"] for x in st if se(x) is not None)
    cause = ctl_keys
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"label": "rh1e4 relaunch sample-error scan (09's rh1e4 gate 'sample_error null'; 4f, 10-16): sample-level error CLASS per run from its .eval; keys and counts only, no text",
           "stamp (PT)": stamp, "writer": {"commit": git("rev-parse", "HEAD").decode().strip(), "app": APP, "volume": VOL},
           "sources": {"sold_record": {"commit": SOLD[0], "path": SOLD[1], "sha256": hashlib.sha256(sb).hexdigest()}, "inventory_b_fix_v1": {"commit": INV[0], "path": INV[1], "sha256": INV[2]},
                       "upload_manifest": {"commit": MAN[0], "path": MAN[1], "sha256": hashlib.sha256(mb).hexdigest()},
                       "members_record": {"commit": MEMBERS[0], "path": MEMBERS[1], "sha256": hashlib.sha256(eb).hexdigest()}},
           "counts": {"staged": len(st), "staged_sha_checked_vs_manifest": len(st), "staged_sample_error_null": len(st) - len(nonnull), "staged_sample_error_nonnull": len(nonnull),
                      "staged_by_log_status": {s: sum(x["log_status"] == s for x in st) for s in sorted({x["log_status"] for x in st})},
                      "staged_by_n_samples": {str(n): sum(x["n_samples"] == n for x in st) for n in sorted({x["n_samples"] for x in st})},
                      "controls": len(ctl), "controls_with_sample_error": sum(se(x) is not None for x in ctl)},
           "staged_nonnull_keys": [{"key": k, "sample_error": se(next(x for x in st if x["key"] == k))} for k in nonnull],
           "controls": [{"key": x["key"], "cause": cause[x["key"]], "staged": x["role"] == "staged", "log_status": x["log_status"], "sample_error": se(x)} for x in ctl],
           "by_key": {x["key"]: {"sample_error": se(x), "log_status": x["log_status"], "n_samples": x["n_samples"],
                                 "limits": sorted({s["limit"] for s in x["samples"] if s["limit"]})} for x in st},
           "pass": not nonnull}
    out = REPO / "experiments" / "2026-10-14_p2_family_expansion" / "results" / f"rh1e4_sample_error_scan_{stamp}.json"
    out.open("x").write(json.dumps(rec, indent=1))
    print(json.dumps({"counts": rec["counts"], "staged_nonnull": rec["staged_nonnull_keys"][:10], "controls": [(c["cause"], c["log_status"], c["sample_error"]) for c in rec["controls"]],
                      "path": str(out.relative_to(REPO))}))
