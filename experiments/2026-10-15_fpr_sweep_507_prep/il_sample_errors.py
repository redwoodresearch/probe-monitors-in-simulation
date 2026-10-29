"""09 RULE @ 36b9b47df §4 (admitted = sample_error == null on the SAMPLE, not log status) for the IL extra arms (sim2, run3): every scored
candidate's .eval, metadata only, on Modal CPU; the reader is run_attack_sample_errors.py's (aswin/fpr-sweep-507-prep-nofilter @ 2b637d81e): log
status, the sample error class (the donor scan's frozen rule @ 4299f6350), the sample limit type, and the count of tool messages other than the
runner's submit. No message content leaves the container. Allowlist = il_arms.candidates() (status scored), each .eval's sha256 from its pinned
counted manifest, asserted before reading.

Also: read-only copies of the per-folder score manifests (p2, sol, own_all_rs, famexp, own_a) of the IL folders, for the ilarms INPUTS block.

  modal run il_sample_errors.py::scan_all          # writes results/il_sample_errors_<stamp>.json (exclusive create)
  modal run il_sample_errors.py::fetch_manifests   # writes scratchpad/fpr507/ilarms_man/<folder>__<family>.json (+ an index)
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # il_arms (local entrypoints only)
APP = "probe-shift_aswin-surya_sub-fpr507prep_ilerr_kill-2026-11-14"
VOLS = {"chain": "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14", "q4own": "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13",
        "conv": "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"}
MAN_DIR = Path("/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/a683364c-3a60-44cd-bb15-e08d87dd4f9a/scratchpad/fpr507/ilarms_man")
FAMS = {"p2": "p2", "sol": "sol", "own_all_rs": "own_all_rs", "famexp": "famexp_r2", "own_a": "own_a"}  # family -> sub dir
MAN_NAME = {"own_all_rs": "manifest_own_all.json"}  # 4f's own_all-style trees name it manifest_own_all.json (run_peraction311.own_all_files)
image = modal.Image.debian_slim(python_version="3.12").pip_install("inspect_ai==0.3.251", "zstandard==0.25.0")
app = modal.App(APP, image=image)
vols = {k: modal.Volume.from_name(v, version=2) for k, v in VOLS.items()}


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout


@app.function(volumes={f"/{k}": v.read_only() for k, v in vols.items()}, cpu=2, memory=4096, timeout=1800, retries=0)
def scan(chunk: list) -> list:
    import os
    import re

    from inspect_ai.log import read_eval_log

    cls = re.compile(r"\b([A-Z][A-Za-z]*(?:Error|Exception|Timeout|Exceeded))\b")

    def klass(err):  # the donor scan's frozen rule (@ 4299f6350)
        if err is None:
            return None
        m = cls.search(err.message or "") or cls.search(err.traceback or "")
        return m.group(1) if m else "unclassified"

    out = []
    for r in chunk:
        rec = {k: r[k] for k in ("set", "key", "source", "folder", "path", "sha256")}
        cands = [f"/{v}/evals_v1/{p}" for v in ("chain", "q4own") for p in dict.fromkeys((r["path"], r["path"].removeprefix("runs/")))]
        hit = next((p for p in cands if os.path.isfile(p)), None)
        if hit is None:
            out.append(rec | {"missing": True})
            continue
        if hashlib.sha256(open(hit, "rb").read()).hexdigest() != r["sha256"]:
            out.append(rec | {"found_at": hit, "sha_mismatch": True})
            continue
        log = read_eval_log(hit)
        if len(log.samples or []) != 1:
            raise ValueError(f"{r['path']}: {len(log.samples or [])} samples")
        s = log.samples[0]
        sub = {tc.id for m in s.messages if m.role == "assistant" for tc in (m.tool_calls or []) if tc.function == "submit"}
        ntool = sum(1 for m in s.messages if m.role == "tool" and m.tool_call_id not in sub and getattr(m, "function", None) != "submit")
        out.append(rec | {"found_at": hit, "sha_ok": True, "log_status": log.status, "sample_error": klass(s.error), "has_error": s.error is not None,
                          "limit": (s.limit.type if s.limit else None), "n_tool_nonsubmit": ntool, "n_messages": len(s.messages)})
    return out


@app.function(volumes={"/conv": vols["conv"].read_only()}, cpu=1, memory=2048, timeout=900, retries=0)
def read_manifests(pairs: list) -> list:
    import os

    res = []
    for folder, fam in pairs:
        p = f"/conv/scaleup/{folder}/{FAMS[fam]}/{MAN_NAME.get(fam, f'manifest_{fam}.json')}"
        res.append([folder, fam, p.removeprefix("/conv/"), open(p, "rb").read() if os.path.isfile(p) else None])
    return res


@app.local_entrypoint()
def scan_all():
    import il_arms as L

    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("commit first: tracked files differ from HEAD")
    rows, srcs = L.candidates()
    allow = [{"set": x["set"], "key": x["key"], "source": x["source"], "folder": x["folder"], "path": x["eval_path"], "sha256": x["eval_sha256"]}
             for x in rows if x["status"] == "scored"]
    print(json.dumps({"allowlist_rows": len(allow), "by_set": {o: sum(r["set"] == o for r in allow) for o in ("S_old", "S_new", "S_newmains")}}), flush=True)
    chunks = [allow[i:i + 40] for i in range(0, len(allow), 40)]
    res = [x for part in scan.map(chunks) for x in part]
    if len(res) != len(allow) or [x["path"] for x in res] != [r["path"] for r in allow]:
        raise SystemExit("reader output does not match the allowlist")
    meta = {"git": git("rev-parse", "HEAD").decode().strip(), "stamp_pt": stamp, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "il_arms_sha256": hashlib.sha256((HERE / "il_arms.py").read_bytes()).hexdigest(), "app": APP, "volumes": VOLS, "sources": srcs,
            "rule": __doc__.split("\n\n")[0],
            "counts": {"rows": len(res), "missing": sum(bool(x.get("missing")) for x in res), "sha_mismatch": sum(bool(x.get("sha_mismatch")) for x in res),
                       "with_sample_error": sum(bool(x.get("has_error")) for x in res), "zero_tool": sum(x.get("n_tool_nonsubmit") == 0 for x in res),
                       "error_classes": {c: sum(1 for x in res if x.get("sample_error") == c) for c in sorted({x.get("sample_error") for x in res} - {None})}}}
    blob = json.dumps({"meta": meta, "rows": res}, indent=1).encode()
    dest = HERE / "results" / f"il_sample_errors_{stamp}.json"
    dest.open("xb").write(blob)
    print(json.dumps({"record": dest.name, "sha256": hashlib.sha256(blob).hexdigest(), "counts": meta["counts"]}), flush=True)


@app.local_entrypoint()
def fetch_manifests():
    import il_arms as L

    rows, _ = L.candidates()
    folders = sorted({x["folder"] for x in rows if x["status"] == "scored"})
    got = read_manifests.remote([[f, fam] for f in folders for fam in FAMS])
    spec = L.pinned(*L.SPEC)[0]["il"]
    pins = spec["arms"] | spec["arms_extended"]
    MAN_DIR.mkdir(parents=True, exist_ok=True)
    idx, bad = {}, []
    for folder, fam, path, b in got:
        if b is None:
            idx[f"{folder}__{fam}"] = {"path": path, "present": False}
            continue
        h = hashlib.sha256(b).hexdigest()
        pin = pins.get(folder, {}).get(f"{fam}_manifest")
        if pin and pin["sha256"] != h:  # the IL spec pins p2 and sol for the 75 index folders: a differing copy is a later revision
            bad.append(f"{folder}__{fam}")
        p = MAN_DIR / f"{folder}__{fam}.json"
        if p.exists() and p.read_bytes() != b:
            raise SystemExit(f"{p.name}: an existing copy differs")
        if not p.exists():
            p.open("xb").write(b)
        idx[f"{folder}__{fam}"] = {"path": path, "present": True, "sha256": h, "il_spec_pin": (pin or {}).get("sha256"), "equals_il_spec_pin": (pin["sha256"] == h) if pin else None}
    blob = json.dumps({"folders": len(folders), "families": list(FAMS), "manifests": idx, "differs_from_il_spec_pin": bad}, indent=1, sort_keys=True).encode()
    ip = MAN_DIR / f"index_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}.json"
    ip.open("xb").write(blob)
    print(json.dumps({"index": str(ip), "sha256": hashlib.sha256(blob).hexdigest(), "folders": len(folders),
                      "present_by_family": {fam: sum(1 for k, v in idx.items() if k.endswith("__" + fam) and v["present"]) for fam in FAMS},
                      "differs_from_il_spec_pin": bad}), flush=True)
