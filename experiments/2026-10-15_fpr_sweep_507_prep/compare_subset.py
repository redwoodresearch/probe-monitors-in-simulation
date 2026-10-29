"""Equality gate for a TEST all-tasks record on a label subset vs a reference record (the resumable-rows fix, 10-16): for each --labels label,
the TEST's per-draw npz keys under that label must equal the reference's key set and every array must be EQUAL (NaN == NaN, same shape and dtype);
its rows (every origin, all fields but "resources") and its pooled block must equal the reference's as JSON. KEYS AND COUNTS ONLY: no value is
printed. Runs on Modal (records and npz never leave the Volume); writes xfer/fpr507prep/checks/<out>.json (exclusive create) and returns it.

  modal run --detach compare_subset.py --ref <record on the Volume>@<sha> --ref-npz <npz>@<sha> --new <record>@<sha> --new-npz <npz>@<sha> \
      --labels "Sol (a) C,FIBS R" --out <name>
"""

import hashlib
import json
from pathlib import Path

import modal

app = modal.App("probe-shift_aswin-surya_sub-fpr507prep_cmpsubset_kill-2026-11-14")
vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", version=2)  # run_common455.VOLS["solavg"]
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")


def npzlabel(lb):
    return lb.replace("@", "_at_").replace("|", "_").replace(" ", "_").replace("(", "").replace(")", "")  # run_peraction311.npzkey's part rule


@app.function(image=image, volumes={"/sol": vol}, cpu=2, memory=16384, timeout=1800)
def check(ref, ref_npz, new, new_npz, labels, out):
    import io

    import numpy as np

    def load(spec, what):
        p_, h_ = spec.rsplit("@", 1)
        b = (Path("/sol") / p_).read_bytes()
        if hashlib.sha256(b).hexdigest() != h_:
            raise SystemExit(f"{what}: sha256 differs")
        return b

    R, N = (json.loads(load(x, w)) for x, w in ((ref, "ref record"), (new, "new record")))
    zr, zn = (np.load(io.BytesIO(load(x, w))) for x, w in ((ref_npz, "ref npz"), (new_npz, "new npz")))
    links = {w: [d_["npz"]["path"], d_["npz"]["sha256"]] == z_.rsplit("@", 1) for w, d_, z_ in (("ref", R, ref_npz), ("new", N, new_npz))}
    meta_ok = {"new_test": N["meta"].get("test") is True, "new_row_store": "row_store" in N["meta"], "npz_is_the_records": links}
    res, ok = {}, all(links.values()) and meta_ok["new_test"] and meta_ok["new_row_store"]
    for lb in labels:
        pre = npzlabel(lb) + "__"
        kr, kn = {k for k in zr.files if k.startswith(pre)}, {k for k in zn.files if k.startswith(pre)}
        bad = [k for k in sorted(kr & kn) if zr[k].shape != zn[k].shape or zr[k].dtype != zn[k].dtype or not np.array_equal(zr[k], zn[k], equal_nan=True)]
        strip = lambda d: {o: {k: v for k, v in r.items() if k != "resources"} for o, r in d.items()}  # noqa: E731
        rr, rn = strip(R["rows"].get(lb, {})), strip(N["rows"].get(lb, {}))
        rows_bad = sorted(o for o in set(rr) | set(rn) if json.dumps(rr.get(o), sort_keys=True) != json.dumps(rn.get(o), sort_keys=True))
        fields_bad = {o: sorted(k for k in set(rr.get(o) or {}) | set(rn.get(o) or {})
                                if json.dumps((rr.get(o) or {}).get(k), sort_keys=True) != json.dumps((rn.get(o) or {}).get(k), sort_keys=True)) for o in rows_bad}
        pooled_eq = json.dumps(R["pooled"].get(lb), sort_keys=True) == json.dumps(N["pooled"].get(lb), sort_keys=True)
        r_ = {"npz_keys_ref": len(kr), "npz_keys_new": len(kn), "npz_only_ref": sorted(kr - kn)[:10], "npz_only_new": sorted(kn - kr)[:10],
              "npz_equal": len(kr & kn) - len(bad), "npz_mismatch": bad[:20], "n_npz_mismatch": len(bad), "row_origins_ref": sorted(rr),
              "row_origins_new": sorted(rn), "rows_differing_origins": rows_bad, "rows_differing_fields": fields_bad, "pooled_equal": pooled_eq}
        r_["pass"] = bool(kr) and kr == kn and not bad and bool(rr) and not rows_bad and pooled_eq and lb in R["pooled"]
        ok &= r_["pass"]
        res[lb] = r_
    rec = {"what": "TEST label subset vs reference, keys and equality only (no value read out)", "ref": ref, "ref_npz": ref_npz, "new": new,
           "new_npz": new_npz, "labels": labels, "meta_checks": meta_ok, "per_label": res, "rows_compare_rule": "every field but 'resources' (container timings)", "pass": ok}
    blob = json.dumps(rec, indent=1).encode()
    dest = Path("/sol/xfer/fpr507prep/checks") / f"{out}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("xb") as fh:
        fh.write(blob)
    vol.commit()
    return {"file": str(dest).removeprefix("/sol/"), "sha256": hashlib.sha256(blob).hexdigest(), "pass": ok}


@app.local_entrypoint()
def main(ref: str, ref_npz: str, new: str, new_npz: str, labels: str, out: str):
    fc = check.spawn(ref, ref_npz, new, new_npz, [x.strip() for x in labels.split(",")], out)
    print(json.dumps({"call": fc.object_id, "out": f"xfer/fpr507prep/checks/{out}.json"}), flush=True)
