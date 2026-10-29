"""Clean-room pooled reproduction, step 0 (a1, 10-16): LAYOUT only, no score values. One Modal CPU container, Volumes read-only.

(1) For each score file in spec.json (paths and sha256 from INPUTS_FULL's S_new overrides): sha256 vs the pin, and per npz array its
    name, dtype and shape; string arrays also give up to 3 entries (keys). Float arrays give dtype and shape only. The per-file
    scorer manifest JSON (path in the override) gives its key tree (types and lengths; string leaves cut to 80 chars).
(2) The pooled catalogue and manifest (input contracts, 0d 10-16): sha256 vs the pin and their key trees, the same way. Neither
    holds score values (keys, roles, counts, paths).

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run layout_probe.py --spec spec.json
"""

import hashlib
import json
from pathlib import Path

import modal

SCORES = "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"
POOLED = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"
REC = "xfer/fpr507prep/records/results/alltasks_CRD_FULL_r2_allorigins_pooledworld_catalogue_e12_2026-10-16_193738"
CONTRACTS = {"catalogue": (REC + "_catalogue.json", "15ee837f"), "manifest": (REC + "_manifest.json",
             "6f4282810cd8526badf193827b1e4c7cc989b2302d44c2c896ced543bdda72fc")}
app = modal.App("probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")


def tree(x, depth=0, maxd=5):
    """Key tree: dict -> {key: tree} (first 12 keys + count), list -> [len, tree of item 0], scalars -> type (strings cut)."""
    if isinstance(x, dict):
        ks = list(x)
        t = {k: tree(x[k], depth + 1, maxd) for k in ks[:12]} if depth < maxd else "dict"
        return {"_n": len(ks), **t} if isinstance(t, dict) else f"dict[{len(ks)}]"
    if isinstance(x, list):
        return [len(x), tree(x[0], depth + 1, maxd) if x and depth < maxd else None]
    if isinstance(x, str):
        return "str:" + x[:80]
    return type(x).__name__


@app.function(image=image, cpu=1, memory=4096, timeout=900, retries=0,
              volumes={"/s": modal.Volume.from_name(SCORES, version=2).read_only(),
                       "/p": modal.Volume.from_name(POOLED, version=2).read_only()})
def layout(entries: list) -> dict:
    import numpy as np

    files = []
    for e in entries:
        p = Path("/s") / e["path"]
        r = {"label": e["label"], "exists": p.exists()}
        if p.exists():
            b = p.read_bytes()
            r["sha256_ok"] = hashlib.sha256(b).hexdigest() == e["sha256"]
            z = np.load(p, allow_pickle=False)
            r["arrays"] = []
            for k in z.files:
                a = z[k]
                d = {"name": k, "dtype": str(a.dtype), "shape": list(a.shape)}
                if a.dtype.kind in ("U", "S"):  # key-like arrays only (never free text); others give max length
                    if any(t in k.lower() for t in ("key", "run", "world", "cell", "id", "fold", "main", "folder")):
                        d["examples"] = [str(x)[:120] for x in a.reshape(-1)[:3]]
                    d["max_len"] = int(max((len(str(x)) for x in a.reshape(-1)), default=0))
                r["arrays"].append(d)
            mp = Path("/s") / e["manifest_path"]
            if mp.exists():
                r["manifest_sha_ok"] = hashlib.sha256(mp.read_bytes()).hexdigest() == e["manifest_sha256"]
                r["manifest_tree"] = tree(json.loads(mp.read_text()))
        files.append(r)
    contracts = {}
    for name, (rel, pin) in CONTRACTS.items():
        p = Path("/p") / rel
        b = p.read_bytes()
        h = hashlib.sha256(b).hexdigest()
        contracts[name] = {"bytes": len(b), "sha256": h, "pin_ok": h.startswith(pin), "tree": tree(json.loads(b))}
    return {"files": files, "contracts": contracts}


@app.local_entrypoint()
def main(spec: str):
    res = layout.remote(json.loads(Path(spec).read_text()))
    Path("layout_out.json").open("x").write(json.dumps(res, indent=1))
    for r in res["files"]:
        print(r["label"], "exists", r["exists"], "sha_ok", r.get("sha256_ok"), "manifest_sha_ok", r.get("manifest_sha_ok"),
              [(a["name"], a["dtype"], a["shape"]) for a in r.get("arrays", [])])
    for k, c in res["contracts"].items():
        print(k, c["bytes"], c["sha256"], "pin_ok", c["pin_ok"])
