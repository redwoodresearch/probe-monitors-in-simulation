"""Local copies of an own_all-style slot's per-folder manifest_own_all.json (own_all_rs; the runner's own_all_files() reads them), fetched from
the Volumes read-only and each checked against its score record's manifest_sha256 (a missing or differing one fails; nothing is overwritten).
Trees as 4f's scorer writes them (sub = own_all_rs): conv:/scaleup/<folder>/<sub>, conv:/simh/<batch>/<sub>/<folder>,
p2:/famexp/2026-10-14_163027/<sub>_<fit_stamp>/<folder> (t196), conv:/simh/primary_v3/<sub> (a25simh).

  modal run fetch_slot_manifests.py --record <commit>:<path>:<sha256> --sub own_all_rs --dest <dir under scratchpad/fpr507>
"""

import hashlib
import json
import subprocess
from pathlib import Path

import modal

app = modal.App("probe-shift_aswin-surya_sub-fpr507prep_rsman_kill-2026-11-14")
VC = modal.Volume.from_name("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14")
VP = modal.Volume.from_name("probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13")
COPIES = Path("/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/a683364c-3a60-44cd-bb15-e08d87dd4f9a/scratchpad/fpr507")


@app.function(image=modal.Image.debian_slim(python_version="3.12"), volumes={"/conv": VC, "/p2": VP}, cpu=1, memory=1024, timeout=600, retries=0)
def read(folders: dict, sub: str, fit_stamp: str) -> dict:
    out = {}
    for fo, kind in folders.items():
        if kind == "scaleup":
            c = [Path("/conv/scaleup") / fo / sub / "manifest_own_all.json"]
        elif kind == "simh":
            c = sorted(Path("/conv/simh").glob(f"*/{sub}/{fo}/manifest_own_all.json"))
        elif kind == "t196":
            c = [Path("/p2/famexp/2026-10-14_163027") / f"{sub}_{fit_stamp}" / fo / "manifest_own_all.json"]
        elif kind == "a25simh":
            c = [Path("/conv/simh/primary_v3") / sub / "manifest_own_all.json"]
        else:
            raise SystemExit(f"{fo}: unknown kind {kind}")
        c = [p for p in c if p.exists()]
        if len(c) != 1:
            raise SystemExit(f"{fo} ({kind}): {len(c)} manifest paths")
        out[fo] = {"path": str(c[0]), "bytes": c[0].read_bytes()}
    return out


@app.local_entrypoint()
def main(record: str, sub: str, dest: str):
    c_, p_, h_ = record.split(":")
    b = subprocess.run(["git", "show", f"{c_}:{p_}"], capture_output=True, check=True).stdout
    if hashlib.sha256(b).hexdigest() != h_:
        raise SystemExit("score record sha256")
    rec = json.loads(b)
    want = {fo: fi for v in rec["results"].values() for fo, fi in v["folders"].items()}
    got = read.remote({fo: fi["kind"] for fo, fi in want.items()}, sub, rec["fit_stamp"])
    d = COPIES / dest
    d.mkdir(parents=True, exist_ok=True)
    bad = [fo for fo, fi in want.items() if hashlib.sha256(got[fo]["bytes"]).hexdigest() != fi["manifest_sha256"]]
    if bad:
        raise SystemExit(f"{len(bad)} manifests differ from the score record, e.g. {bad[:3]}")
    for fo in sorted(want):
        p = d / f"{fo}__own_all.json"
        if p.exists():
            if p.read_bytes() != got[fo]["bytes"]:
                raise SystemExit(f"{p.name}: an existing copy differs")
            continue
        p.open("xb").write(got[fo]["bytes"])
    print(json.dumps({"record": record, "folders": len(want), "by_kind": {k: sum(fi["kind"] == k for fi in want.values()) for k in ("scaleup", "simh", "t196", "a25simh")},
                      "dest": str(d), "all_sha_match": True}))


@app.function(image=modal.Image.debian_slim(python_version="3.12"), volumes={"/conv": VC, "/p2": VP}, cpu=1, memory=1024, timeout=600, retries=0)
def read_paths(paths: list) -> list:
    return [(Path(f"/{v}") / p.lstrip("/")).read_bytes() for v, p in paths]


@app.local_entrypoint()
def fetch_list(spec: str, dest: str):
    """spec: a JSON list of {vol: conv | p2, path, sha256, copy} (e.g. famexp's manifest_famexp.json per S_paper493 folder, sha from its record);
    every fetched file must match its sha256; copies are written once (an existing differing copy STOPs)."""
    items = json.loads(Path(spec).read_text())
    got = read_paths.remote([(x["vol"], x["path"]) for x in items])
    bad = [x["copy"] for x, b in zip(items, got) if hashlib.sha256(b).hexdigest() != x["sha256"]]
    if bad:
        raise SystemExit(f"{len(bad)} files differ from their sha256, e.g. {bad[:3]}")
    d = COPIES / dest
    d.mkdir(parents=True, exist_ok=True)
    for x, b in zip(items, got):
        p = d / x["copy"]
        if p.exists():
            if p.read_bytes() != b:
                raise SystemExit(f"{p.name}: an existing copy differs")
            continue
        p.open("xb").write(b)
    print(json.dumps({"files": len(items), "dest": str(d), "all_sha_match": True}))
