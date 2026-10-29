"""Verify the full-length (cutfull) step-3 packs by key against their messages, and write the local run records the hung client did not
(RULE F.8). Reads only manifests and the non-score arrays of each pack (key, run_key, unit, call_index, irid, token, n_tokens, fold);
`score` and `score_by_model` are never loaded. Per folder: the manifest exists, line "cutfull", its batch, folder, original folder,
message sha256 and keys equal the message's, one original folder, complete with no runs without rows and exactly the six recipe x readout
packs; every file's sha256 equals the manifest's; every npz has equal-length arrays and run_keys equal to this folder's epochs,
key == the run's world|cell, token < n_tokens == the epoch's tokens_before, no repeated (run_key, readout position) row; and no epoch
key appears in two folders. Writes results/cutfull_<target>_r<k>_<stamp>.json per message.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run <this file> --runs-file <json> --app-id <ap-...>
"""

import hashlib
import io
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE  # in the container the file sits at /root
APP = "probe-shift_aswin-surya_sub-step3_cutfullverify_kill-2026-10-22"
FORMAT_LABEL = "format_check not run (stage format_gate only)"  # 09's wording, relayed by 4f 10-15 ~23:5x
READ_STATUS = ("UNREAD: (a'') packs stay unread until d9 re-attests at a new revision after the format check; if the staged shas are unchanged "
               "the packs are confirmed by sha (09, relayed by 4f)")
PACKS = {f"step3_{r}_L48__{x}.npz" for r in ("r1only", "r2only", "r12") for x in ("R", "C")}
NONSCORE = ("key", "run_key", "unit", "call_index", "irid", "token", "n_tokens", "fold")
app = modal.App(APP)
OUTV = modal.Volume.from_name("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", version=2).read_only()
CFV = modal.Volume.from_name("probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14", version=2).read_only()  # metas only: rowless epochs


def verify(msg: dict, msg_sha: str) -> dict:
    import numpy as np

    out, problems = {}, []
    ep_of = {e["key"]: e for e in msg["epochs"]}
    folders = sorted({e["folder"] for e in msg["epochs"]})
    for fo in folders:
        od = Path("/out/cutfull") / msg["batch"] / "step3" / fo
        mb = (od / "manifest_step3.json").read_bytes()
        man = json.loads(mb)
        mine = {e["key"] for e in msg["epochs"] if e["folder"] == fo}
        keys = {ep_of[k]["world"] + "|" + ep_of[k]["cell"] for k in mine}
        wc_key = {ep_of[k]["world"] + "|" + ep_of[k]["cell"]: k for k in mine}
        # a run may have no unit at all (its extraction meta has zero calls): the scorer lists it in runs_without_rows for every pack and
        # marks the folder incomplete; that is structural and accepted only when the meta shows zero calls, never for a missing readout
        rwr = man.get("runs_without_rows") or {}
        listed = {x for v in rwr.values() for x in v}
        rowless = {wc_key[x] for x in listed if x in wc_key and not json.loads((Path("/b") / ep_of[wc_key[x]]["run"] / "meta" / ep_of[wc_key[x]]["arm"]
                                                                                  / f"{hashlib.sha1(wc_key[x].encode()).hexdigest()[:20]}.json").read_text())["calls"]}
        struct_ok = (set(rwr) == {f"{f[:-4].replace('__', '|')}" for f in PACKS} and all(set(v) == listed for v in rwr.values())
                     and {wc_key.get(x) for x in listed} == rowless) if rwr else True
        if (man.get("line"), man.get("batch"), man.get("folder"), man.get("message_sha256")) != ("cutfull", msg["batch"], fo, msg_sha) or set(man["keys"]) != keys \
                or {ep_of[k]["original_folder"] for k in mine} != {man["original_folder"]} or man.get("complete") is not (not rwr) or set(man["files"]) != PACKS \
                or not struct_ok:
            problems.append(f"{fo}: manifest line/batch/folder/message/keys/original/complete")
        files = {}
        for fn, h in sorted(man["files"].items()):
            b = (od / fn).read_bytes()
            if hashlib.sha256(b).hexdigest() != h:
                problems.append(f"{fo}/{fn}: sha256 differs from the manifest")
            z = np.load(io.BytesIO(b))
            a = {k: z[k] for k in NONSCORE}  # score and score_by_model are not loaded
            n = {len(v) for v in a.values()}
            rk = a["run_key"].astype(str).tolist()
            bad = len(n) != 1 or set(rk) != mine - rowless or any(a["key"][i] != ep_of[k]["world"] + "|" + ep_of[k]["cell"] for i, k in enumerate(rk)) \
                or any(not 0 <= a["token"][i] < a["n_tokens"][i] or a["n_tokens"][i] != ep_of[k]["tokens_before"] for i, k in enumerate(rk)) \
                or len(set(zip(rk, a["token"].tolist(), a["call_index"].tolist()))) != len(rk)
            if bad:
                problems.append(f"{fo}/{fn}: per-row arrays fail a by-key check")
            files[fn] = {"rows": int(next(iter(n))), "runs": len(set(rk)), "units_past_32768": int((a["token"] >= 32768).sum()), "fold_minus1": int((a["fold"] < 0).sum())}
        out[fo] = {"rowless_epochs": sorted(rowless),
                   "manifest": str(od / "manifest_step3.json"), "manifest_sha256": hashlib.sha256(mb).hexdigest(), "counts": man["counts"], "complete": man["complete"],
                   "verified_files": files, "head": {k: man[k] for k in man if k not in ("folder", "original_folder", "excluded_by_staging", "files", "counts",
                                                                                         "runs_without_rows", "keys", "routing", "complete")}}
    return {"folders": out, "problems": problems}


@app.function(image=modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3"), cpu=(2, 2), memory=(8192, 8192), timeout=1800, retries=0,
              volumes={"/out": OUTV, "/b": CFV})
def verify_all(msgs: list, shas: list) -> list:
    """One container, one call (no map: the scorer's map delivered inputs twice)."""
    return [verify(m, h) for m, h in zip(msgs, shas)]


@app.local_entrypoint()
def main(runs_file: str, app_id: str, why: str = ""):
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout  # noqa: E731
    jobs = json.loads(Path(runs_file).read_text())
    msgs = [json.loads(Path(j["message"]).read_bytes()) for j in jobs]
    shas = [hashlib.sha256(Path(j["message"]).read_bytes()).hexdigest() for j in jobs]
    allk = [e["key"] for m in msgs for e in m["epochs"]]
    if len(allk) != len(set(allk)):
        raise SystemExit("an epoch key appears in two messages")
    res = verify_all.remote(msgs, shas)
    probs = [p for r in res for p in r["problems"]]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    for m, j, r in zip(msgs, jobs, res):
        h = next(iter(r["folders"].values()))["head"]
        rec = h | {"result": {fo: {k: v for k, v in x.items() if k != "head"} for fo, x in r["folders"].items()}, "app": app_id, "format_check": h.get("format_check") or FORMAT_LABEL,
                   "read_status": "SEALED; (b) descriptive" if h.get("scope") == "b" else READ_STATUS,
                   "recovered": {"by": "verify_cutfull_packs.py", "git": git("rev-parse", "HEAD").decode().strip(), "stamp (PT)": stamp,
                   "why": why or ("double delivery: in the one map call fc-01M51PET8QNPC3WAA8DWJV9KRR, Modal ran a second copy of 44 of the 45 inputs concurrently "
                                  "(23:31:35-23:31:49 PT); every second copy was refused by the key reservation or the tree scan and wrote nothing; the client then "
                                  "hung and was stopped, so these records are written from the Volume manifests (RULE F.8)"), "problems": r["problems"]}}
        p = HERE / "results" / f"cutfull_{m['d9_arm']}_r{h['revision']}_{h['stamp (PT)']}.json"
        if p.exists():  # a re-verification never rewrites an earlier record (results are append-only)
            p = p.with_name(f"{p.stem}_reverify_{stamp}.json")
        p.open("x").write(json.dumps(rec, indent=1))
    print(json.dumps({"messages": len(res), "folders": sum(len(r["folders"]) for r in res), "rows": sum(f["rows"] for r in res for x in r["folders"].values()
                                                                                                        for f in x["verified_files"].values()), "problems": probs}, indent=1))
    if probs:
        raise SystemExit(f"{len(probs)} verification problems")
