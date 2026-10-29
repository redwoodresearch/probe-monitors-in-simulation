"""Append-only manifest revision for the 53 (a'') cutfull step-3 packs (RULE F.11): each folder gets manifest_step3_r2.json beside its
manifest_step3.json (never overwritten), = the old manifest + "staged_sha256" + a "supersedes" pointer (path, sha256), for 2e's
INPUTS_FULL verifier. The staged sha is asserted equal BY KEY across: every epoch of the committed message for that folder, the routing
record revision the pack was scored from (arms[].staged_sha) and d9's re-attested revision (confirm note @ 7c5e329d3). Reads only
manifests (no npz). One remote call. Writes results/cutfull_manifest_r2_<stamp>.json (paths + sha256 per folder).

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run <this file> --routing-commit <sha>
"""

import glob
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-step3_cutfullmanr2_kill-2026-10-22"
app = modal.App(APP)
OUTV = modal.Volume.from_name("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", version=2)


@app.function(image=modal.Image.debian_slim(python_version="3.12"), cpu=(1, 1), memory=(2048, 2048), timeout=600, retries=0, volumes={"/out": OUTV})
def write_r2(rows: list) -> list:
    out = []
    for r in rows:
        od = Path("/out/cutfull") / r["batch"] / "step3" / r["folder"]
        old = (od / "manifest_step3.json").read_bytes()
        m = json.loads(old)
        if hashlib.sha256(old).hexdigest() != r["manifest_sha256"] or m.get("line") != "cutfull" or m["folder"] != r["folder"] or "staged_sha256" in m:
            raise RuntimeError(f"{r['folder']}: the manifest differs from the committed record's, or already carries a staged sha")
        new = m | {"staged_sha256": r["staged_sha256"], "manifest_revision": 2, "change": "adds staged_sha256 (RULE F.11); nothing else changed",
                   "supersedes": {"path": str(od / "manifest_step3.json"), "sha256": r["manifest_sha256"]}}
        nb = json.dumps(new, indent=1, default=str).encode()
        (od / "manifest_step3_r2.json").open("xb").write(nb)  # create-only
        out.append({"folder": r["folder"], "manifest_r2": str(od / "manifest_step3_r2.json"), "sha256": hashlib.sha256(nb).hexdigest(),
                    "supersedes_sha256": r["manifest_sha256"], "staged_sha256": r["staged_sha256"]})
    OUTV.commit()
    return out


@app.local_entrypoint()
def main(routing_commit: str):
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no").strip() or git("rev-parse", "HEAD") != git("rev-parse", "origin/aswin/step3-retrain"):
        raise SystemExit("uncommitted changes, or HEAD not pushed")
    note = json.loads((HERE / "results" / "cutfull_reattest_note_2026-10-16_002556.json").read_text())
    if note["confirmed"] != 45 or not note["routing_commit"].startswith(routing_commit[:9]):
        raise SystemExit("the re-attestation note does not confirm all 45 records at this routing commit")
    newrev = {Path(r["record"]).name: r["new_revision"] for r in note["rows"]}
    rows = []
    for p in sorted(glob.glob(str(HERE / "results" / "cutfull_s*_r[0-9]_2026-10-15_23*.json"))):
        rec = json.loads(Path(p).read_text())
        src = json.loads(git("show", f"{rec['source_record']['commit']}:{rec['source_record']['path']}"))
        msg = json.loads((REPO / src["message_path"]).read_bytes())
        old = json.loads(git("show", f"{rec['routing_record']['commit']}:{rec['routing_record']['path']}"))
        np_, nc = (x.strip() for x in newrev[Path(p).name].split(" @ "))
        new = json.loads(git("show", f"{nc}:{np_}"))
        for fo, x in rec["result"].items():
            ep = {e["staged_sha256"] for e in msg["epochs"] if e["folder"] == fo}
            arm = {e["arm"] for e in msg["epochs"] if e["folder"] == fo}
            (a_,) = arm
            o_sha = {a["staged_sha"] for a in old["arms"] if a["meta_arm"] == a_}
            n_sha = {a["staged_sha"] for a in new["arms"] if a["meta_arm"] == a_}
            if len(ep) != 1 or ep != o_sha or ep != n_sha:
                raise SystemExit(f"{fo}: staged sha differs across the message, the scored revision and the re-attested revision")
            rows.append({"batch": rec["batch"], "folder": fo, "manifest_sha256": x["manifest_sha256"], "staged_sha256": next(iter(ep))})
    if len(rows) != 53:
        raise SystemExit(f"expected 53 (a'') folders, found {len(rows)}")
    out = write_r2.remote(rows)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    q = HERE / "results" / f"cutfull_manifest_r2_{stamp}.json"
    q.open("x").write(json.dumps({"what": "append-only manifest_step3_r2.json for the 53 (a'') cutfull packs (RULE F.11)", "git": git("rev-parse", "HEAD").decode().strip(),
                                  "stamp (PT)": stamp, "routing_commit": routing_commit, "folders": out}, indent=1))
    print(q.name, len(out), re.sub(r"\s+", " ", json.dumps(out[0]))[:300])
