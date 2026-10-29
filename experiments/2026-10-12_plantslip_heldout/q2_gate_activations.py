"""Row 2, step 2(a)'s activation gate (RULE.md here, Amendment 1, item 1). One Modal CPU container; saved activations only,
real runs only. Per layer, the 20 controls' re-extracted kept output_end rows (q2_run_extract_heldout.py's controls run) are
compared with the same rows of the same 20 real e2 runs in 2026-10-06_la_segment-end_qwen3-32b_alllayers_v1. A layer passes if
max over elements of |a - b| / max(|b|, 1) <= 1e-3 (b, the dump's value). Also reported: max |a - b| and the share of exact
matches. Rows are matched by (step, action); the two metas must list the same kept output_end positions, or it stops.
Failures re-raise by type name only.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run \\
      experiments/2026-10-12_plantslip_heldout/q2_gate_activations.py --stage <q2_reply_stage_2026-10-12_135318.json>
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-heldout_actgate_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
RUN, DUMP = "2026-10-12_q2_heldout_controls-e2_qwen3-32b_l20-60_v1", "2026-10-06_la_segment-end_qwen3-32b_alllayers_v1"
ARM_RUN, ARM_DUMP, LAYERS, TOL = "real-attack-e2-control", "real-attack-e2", (20, 28, 36, 44, 52, 60), 1e-3
STAGE_SHA = "f1a68c90a5d2"  # the staging record's sha256 prefix (q2_reply_s3_manifest_v2)

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2).read_only()
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")


def filename(key: str) -> str:  # the extractor's file stems
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


@app.function(image=image, volumes={"/acts": acts}, cpu=4, memory=32768, timeout=3600, retries=0)
def gate(keys: list[str]) -> dict:
    try:
        return _gate(keys)
    except Exception as e:  # noqa: BLE001 — sanitized and re-raised, never swallowed
        raise RuntimeError(f"gate failed: {type(e).__name__}") from None


def _gate(keys):
    import numpy as np
    def rows(run, arm, key):
        m = json.loads((Path("/acts") / run / "meta" / arm / f"{filename(key)}.json").read_text())
        pos = sorted((q["step"], q["action"], m["row_start"] + q["row"]) for q in m["positions"] if q["kind"] == "output_end" and q["kept"])
        return m["chunk"], pos
    pairs = []
    for k in keys:  # keys arrive sorted; pairs keep that order
        (ca, pa), (cb, pb) = rows(RUN, ARM_RUN, k), rows(DUMP, ARM_DUMP, k)
        if [p[:2] for p in pa] != [p[:2] for p in pb] or not pa:
            raise ValueError("kept output_end positions differ between the controls run and the dump")
        pairs.append((ca, [p[2] for p in pa], cb, [p[2] for p in pb]))
    # The dump's chunk files pack transcripts of every arm, sim-attack included, so its rows are read by exact byte range,
    # never whole files, after the chunk index confirms each row lies in that control's own real-attack-e2 block.
    for k, (ca, ra, cb, rb) in zip(keys, pairs):
        idx = json.loads((Path("/acts") / DUMP / "chunks" / f"{cb}.json").read_text())
        blk = [t for t in idx["transcripts"] if t["key"] == k and t["arm"] == ARM_DUMP]
        if len(blk) != 1 or not all(blk[0]["row_start"] <= r < blk[0]["row_start"] + blk[0]["n_rows"] for r in rb):
            raise ValueError("a dump row lies outside its control's real-attack-e2 block")

    def read_rows(path, rows):
        with open(path, "rb", buffering=0) as fh:  # unbuffered: only the requested bytes are read
            v = np.lib.format.read_magic(fh)
            shape, fortran, dtype = (np.lib.format.read_array_header_1_0 if v == (1, 0) else np.lib.format.read_array_header_2_0)(fh)
            if fortran or len(shape) != 2:
                raise ValueError("unexpected array layout")
            off, rb_ = fh.tell(), shape[1] * dtype.itemsize
            out_ = np.empty((len(rows), shape[1]), dtype)
            for i, r in enumerate(rows):
                fh.seek(off + r * rb_); out_[i] = np.frombuffer(fh.read(rb_), dtype)
        return out_

    out = {}
    for layer in LAYERS:
        mx_rel, mx_abs, exact, n = 0.0, 0.0, 0, 0
        for ca, ra, cb, rb in pairs:
            a = read_rows(Path("/acts") / RUN / "by_layer" / f"L{layer}" / f"{ca}.npy", ra).astype(np.float64)
            b = read_rows(Path("/acts") / DUMP / "by_layer" / f"L{layer}" / f"{cb}.npy", rb).astype(np.float64)
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                raise ValueError("a non-finite activation")
            d = np.abs(a - b)
            mx_rel = max(mx_rel, float((d / np.maximum(np.abs(b), 1.0)).max())); mx_abs = max(mx_abs, float(d.max()))
            exact += int((d == 0).sum()); n += d.size
        out[str(layer)] = {"max rel diff": mx_rel, "max abs diff": mx_abs, "share exact": exact / n, "elements": n, "rows": sum(len(p[1]) for p in pairs), "passes": mx_rel <= TOL}
    return out


@app.local_entrypoint()
def main(stage: str):
    for f in (Path(__file__).resolve(), HERE / "RULE.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    sp = Path(stage)
    if not hashlib.sha256(sp.read_bytes()).hexdigest().startswith(STAGE_SHA):
        raise SystemExit("not the pinned staging record")
    ctl = json.loads(sp.read_text())["controls"]
    keys = sorted(c["key"] for c in ctl)
    if len(keys) != 20 or any(c["orig_arm"] != ARM_DUMP or c["arm"] != ARM_RUN for c in ctl):
        raise SystemExit("not 20 real e2 controls")
    res = gate.remote(keys)
    rec = {"rule": "RULE.md Amendment 1, item 1 (this folder)", "git": head, "run": RUN, "dump": DUMP, "tolerance (relative)": TOL,
           "controls": len(keys), "by layer": res, "layers passing": [l for l in res if res[l]["passes"]], "layers failing": [l for l in res if not res[l]["passes"]]}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    with open(dest := HERE / "results" / f"q2_gate_activations_{stamp}.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1)); print(dest)
