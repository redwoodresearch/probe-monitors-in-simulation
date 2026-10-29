"""Extract, unchanged, record A's stored per-draw real_line arrays for the six headline probes x C/R at pooled_all_origins / joint /
0.01 (draw 0 = point, draws 1..10,000) into one small npz, so the post's reproducibility folder can rebuild the [^realline] six-probe
means and intervals from git. No computation: the arrays are copied byte-for-value (float64) and each is checked against record A's own
stored summary point (draw 0) before it is written.

  M=/Users/aswinsurya/.cache/probe-shift-modal161/bin/modal
  "$M" run experiments/2026-10-25_realline_final_read/run_extract_realline_draws.py --out <path>/realline_draws.npz
"""

import hashlib
import json
from pathlib import Path

import modal

APP = "probe-shift_aswin-surya_sub-realline_drawsx_kill-2026-11-25"
SOLAVG_VOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"
REC_A = ("experiments/2026-10-15_fpr_sweep_507_prep/results/routec_records/"
         "routec_A_alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_2026-10-19_180327.json.gz")
REC_A_SHA = "191c467ab095b0f9ddfe2271822655bc5b49ee02a1a2c5a5b0e11bc855a00865"
REC_A_DSHA = "54afad04422afd288db9b77043713cfdb708adf5c0eaa74499ea3e5dd600082a"
SIX = ["sol_a_uncut", "FIBS|uncut", "harmfulness|uncut", "refusal|uncut", "high-stakes|uncut", "own_all_rs_uncut_L48"]
LEAD, MODE, LVL = "pooled_all_origins", "joint", "0.01"

HERE = Path(__file__).resolve()
REPO = HERE.parents[2] if len(HERE.parents) > 2 else HERE.parent  # /root in the container; the local file is read only at launch

app = modal.App(APP)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
         .add_local_file(str(REPO / REC_A), "/in/routec_A.json.gz"))
vol = modal.Volume.from_name(SOLAVG_VOL, version=2)


def npzkey(*parts):  # run_peraction311.npzkey @ 20c6fbd20, verbatim rule (as run_group_catch.py)
    out = []
    for p_ in parts:
        q = str(p_).replace("@", "_at_").replace("|", "_").replace(" ", "_").replace("(", "").replace(")", "")
        if "__" in q or not q:
            raise SystemExit(f"npz key part {p_!r} would be ambiguous")
        out.append(q)
    return "__".join(out)


@app.function(image=image, volumes={"/sol": vol.read_only()}, cpu=(1.0, 1.0), memory=(4096, 4096), timeout=600,
              startup_timeout=300, scaledown_window=10, retries=0)
def extract() -> bytes:
    import gzip
    import io

    import numpy as np

    gz = Path("/in/routec_A.json.gz").read_bytes()
    assert hashlib.sha256(gz).hexdigest() == REC_A_SHA, "STOP: record A sha256"
    dec = gzip.decompress(gz)
    assert hashlib.sha256(dec).hexdigest() == REC_A_DSHA, "STOP: record A decompressed sha256"
    rec = json.loads(dec)
    npz = rec["npz"]
    assert npz["volume"] == SOLAVG_VOL and not npz["path"].startswith("/") and ".." not in Path(npz["path"]).parts, "STOP: npz pointer"
    h = hashlib.sha256()
    with open(Path("/sol", npz["path"]), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    assert h.hexdigest() == npz["sha256"], "STOP: npz sha256"
    Z = np.load(Path("/sol", npz["path"]))
    out = {}
    for name in SIX:
        for ro in ("C", "R"):
            lab = f"{name} {ro}"
            node = rec["pooled"][lab][LEAD][MODE]
            lk = [k for k in node if k == LVL or (k.replace(".", "", 1).isdigit() and float(k) == float(LVL))]
            assert len(lk) == 1, f"STOP: level key for {lab}"
            q = node[lk[0]]["gap"]["real_line"]
            a = np.asarray(Z[npzkey(lab, LEAD, MODE, lk[0], "pooled", "gap_real_line")], dtype=np.float64)
            assert a.shape == (q["draws"] + 1,), f"STOP: {lab} shape {a.shape}"
            assert float(a[0]) == q["point"], f"STOP: {lab} draw 0 differs from record A's stored point"
            out[f"{name}|{ro}"] = a
    buf = io.BytesIO()
    np.savez(buf, **{k.replace("|", "__").replace(" ", "_"): v for k, v in out.items()})
    return buf.getvalue()


@app.local_entrypoint()
def main(out: str):
    b = extract.remote()
    p = Path(out)
    with p.open("xb") as fh:
        fh.write(b)
    print("REALLINE_DRAWS_OK", p, hashlib.sha256(b).hexdigest(), len(b))
