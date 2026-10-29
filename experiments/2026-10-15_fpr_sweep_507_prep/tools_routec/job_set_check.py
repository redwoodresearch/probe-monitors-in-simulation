"""Job sets by key (09, 10-18, S1a item 3): two drive specs as spec_diff takes them, each job keyed "<label> | <origin tag>" (the label its
row leaves with: cfg full_as, else the job label, rp L6914; and _row_impl's origin tag, rp L1353 and L1359-1362, both at 71633875), compared
as multisets, and each side's jobs factored as label base x readout x origin tag (a label's last word is its readout). Keys and counts only
(labels, origin tags, counts, shas). The specs are unpickled only in a Modal CPU container, by spec_diff's loader and image (spec_diff.py
itself is untouched: its blob is pinned elsewhere).

  modal run -n <description> tools_routec/job_set_check.py --a <spec> --b <spec> --out results/spec_diff/<name>.json
    <spec> = a local .pkl (sent as bytes) or vol:<path> on the solavg Volume; --out is relative to the 507-prep folder, written exclusively
"""

import hashlib
import json
import pickle
import subprocess
from pathlib import Path

import modal

VOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"  # = spec_diff.VOL (= run_common455.VOLS["solavg"])
AO = " (all-origins draws)"  # = run_peraction311.AO (rp L3218)
app = modal.App("probe-shift_aswin-surya_sub-fpr507prep_jobset_kill-2026-11-14")
vol = modal.Volume.from_name(VOL, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")  # = spec_diff's image


def _key(k):
    return k if isinstance(k, str) else repr(k)


def job_parts(job):
    """(the label a job's row leaves with, its origin tag)."""
    c = job[1] if isinstance(job[1], dict) else {}
    o = c.get("origin", "S_new")
    tag = (o + (" without S4" if c.get("no_s4") else "") + (" (all-units deviation)" if c.get("allunits_dev") else "")
           + (" (registered cohort)" if c.get("cohort") == "registered" else "") + (f" [{c['sp_block']}]" if o == "S_paper493" and c.get("sp_block") else "")
           + (AO if c.get("merge") else "") + (f" [pooled: {c['pw_combo']}]" if c.get("pw_combo") else ""))
    return _key(c.get("full_as", job[0])), tag


def job_set(a, b):
    """The two specs' job keys: counts, the keys on one side only, and equal (as multisets, so a repeated key counts)."""
    ka, kb = ([" | ".join(job_parts(j)) for j in s.get("jobs") or []] for s in (a, b))
    oa, ob = sorted(set(ka) - set(kb)), sorted(set(kb) - set(ka))
    return {"n_a": len(ka), "n_b": len(kb), "n_distinct_a": len(set(ka)), "n_distinct_b": len(set(kb)), "n_only_a": len(oa), "n_only_b": len(ob),
            "only_a": oa, "only_b": ob, "equal": sorted(ka) == sorted(kb)}


def job_factors(spec):
    """The jobs as label base x readout x origin tag: the counts, the bases, readouts and tags, and whether the jobs are exactly that full
    product (each cell once)."""
    cells = [(lb.rsplit(" ", 1)[0], lb.rsplit(" ", 1)[-1] if " " in lb else "", t) for lb, t in map(job_parts, spec.get("jobs") or [])]
    bs, rs, ts = (sorted({c[i] for c in cells}) for i in range(3))
    return {"n_jobs": len(cells), "n_label_bases": len(bs), "n_readouts": len(rs), "n_origin_tags": len(ts), "product": len(bs) * len(rs) * len(ts),
            "full_product": len(cells) == len(set(cells)) == len(bs) * len(rs) * len(ts), "label_bases": bs, "readouts": rs, "origin_tags": ts}


@app.function(image=image, cpu=(2, 4), memory=(4096, 32768), timeout=1800, retries=0, volumes={"/sol": vol.read_only()})
def check_remote(a_src, b_src) -> dict:
    def load(s):  # = spec_diff.diff_remote's loader
        return pickle.loads((Path("/sol") / s[1]).read_bytes() if s[0] == "vol" else s[1])

    a, b = load(a_src), load(b_src)
    return {"job_set": job_set(a, b), "a_job_factors": job_factors(a), "b_job_factors": job_factors(b)}


@app.local_entrypoint()
def main(a: str, b: str, out: str):
    here = Path(__file__).resolve().parents[1]  # the 507-prep folder

    def src(s):  # = spec_diff.main's src
        if s.startswith("vol:"):
            return ("vol", s[4:]), {"volume_path": s[4:]}
        blob = Path(s).read_bytes()
        return ("bytes", blob), {"local": s, "sha256": hashlib.sha256(blob).hexdigest()}

    (sa, ia), (sb, ib) = src(a), src(b)
    res = check_remote.remote(sa, sb)
    rec = {"what": "job sets by key (09 S1a item 3)", "a": ia, "b": ib,
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, check=True, cwd=here).stdout.decode().strip()} | res
    blob = json.dumps(rec, indent=1).encode()
    dest = here / out
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("xb") as fh:
        fh.write(blob)
    js, fb = res["job_set"], res["b_job_factors"]
    print(json.dumps({"out": out, "sha256": hashlib.sha256(blob).hexdigest(), "job_set": {k: js[k] for k in ("n_a", "n_b", "n_only_a", "n_only_b", "equal")},
                      "b_job_factors": {k: fb[k] for k in ("n_label_bases", "n_origin_tags", "n_readouts", "product", "n_jobs", "full_product")}}))
