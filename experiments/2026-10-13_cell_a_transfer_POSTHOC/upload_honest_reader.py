"""Cell (a) transfer, Amendment 1: upload the staged honest reader-style runs to the la-activations Volume (new paths only;
follow-up (c)'s upload_staged.py with this folder), verify every file against its manifest, and the staging gate: the 3
gate transcripts' messages (sha256 of json.dumps(messages)) must equal the existing staging's real-attack-r1 rows of the
same key, at level 0 (2026-10-10_la_september-onpolicy-l0_v1) and level 1 (..._l1_v1). Hashes only; no text leaves Modal.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' modal run \\
      experiments/2026-10-13_cell_a_transfer_POSTHOC/upload_honest_reader.py --stage-record results/stage_honest_<stamp>.json
"""

import hashlib
import json
from pathlib import Path

import modal

HERE = Path(__file__).parent
app = modal.App("probe-shift_aswin-surya_sub-onpol-reply_xferupload_kill-2026-11-13")
vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
EXISTING = {"l0": "2026-10-10_la_september-onpolicy-l0_v1", "l1": "2026-10-10_la_september-onpolicy-l1_v1"}


@app.function(image=modal.Image.debian_slim(python_version="3.12"), cpu=2, memory=8192, volumes={"/acts": vol}, timeout=900, retries=0)
def verify(run: str, gate: dict, level: str) -> dict:
    vol.reload()
    man = json.loads((Path("/acts") / run / "manifest.json").read_text())
    bad = [n for n, h in man["files"].items() if hashlib.sha256((Path("/acts") / run / n).read_bytes()).hexdigest() != h]
    found = {}
    for f in sorted((Path("/acts") / EXISTING[level] / "inputs").glob("shard_*.jsonl")):
        for line in f.open():
            r = json.loads(line)
            if r["arm"] == "real-attack-r1" and r["key"] in gate:
                found[r["key"]] = hashlib.sha256(json.dumps(r["messages"]).encode()).hexdigest()
    return {"run": run, "files": len(man["files"]), "sha256_mismatch": bad,
            "gate": {k: {"found": k in found, "equal": found.get(k) == v[level]} for k, v in gate.items()}}


@app.local_entrypoint()
def main(stage_record: str):
    rec = json.loads((HERE / stage_record).read_text())
    for run in rec["runs"]:
        local = HERE / "data" / run
        man = json.loads((local / "manifest.json").read_text())
        if hashlib.sha256((local / "manifest.json").read_bytes()).hexdigest() != rec["runs"][run]["manifest_sha256"]:
            raise SystemExit(f"{run}: manifest differs from the staging record")
        for n, h in man["files"].items():
            if hashlib.sha256((local / n).read_bytes()).hexdigest() != h:
                raise SystemExit(f"{run}/{n}: local file differs from its manifest")
        try:
            next(iter(vol.listdir(run)))
            raise SystemExit(f"{run} already exists on the Volume; staging runs are written once")
        except (FileNotFoundError, modal.exception.NotFoundError, StopIteration):
            pass
        with vol.batch_upload() as batch:
            for f in sorted(p for p in local.rglob("*") if p.is_file()):
                batch.put_file(f, f"{run}/{f.relative_to(local)}")
        res = verify.remote(run, rec["gate_message_sha256"], "l0" if "-l0_" in run else "l1")
        print(json.dumps(res), flush=True)
        if res["sha256_mismatch"] or not all(g["found"] and g["equal"] for g in res["gate"].values()):
            raise SystemExit(f"{run}: upload or staging gate failed")
