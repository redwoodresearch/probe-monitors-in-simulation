"""sample_error per .eval (09 / 4f via d9 10-16 ~00:00): the class of each run's sample-level error, read from the .eval's own sample metadata, never
from the log status (errored sim-honest runs carry log status "success"). The SAME rule as the B fork's frozen-pool scan (run_hsim_eval_scan.py @
d1c147767, klass): per sample, the first match of CLASS in error.message, else in error.traceback, else "unclassified"; a run's sample_error is the
first sample with a class, else null. A RetryError around a RateLimitError stays RetryError (not unwrapped). Each .eval is read from the chain Volume
and its sha256 asserted against the manifest row. Keys and classes only.

  modal run sample_errors_chain.py --spec <json: [{"path": <evals_v1-relative>, "sha256": <hex>}, ...]> --out <json>
"""
import hashlib
import json
import re
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
VOL = "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14"
CLASS = re.compile(r"\b([A-Z][A-Za-z]*(?:Error|Exception|Timeout|Exceeded))\b")  # the B fork's rule, verbatim
image = modal.Image.debian_slim(python_version="3.13").uv_pip_install("zstandard").add_local_file(HERE / "evalzip.py", "/root/evalzip.py")
app = modal.App("probe-shift_aswin-surya_sub-q3chain_sampleerr_kill-2026-11-14")


def klass(err) -> str | None:
    """A sample's error class (None when the sample has no error)."""
    if not err:
        return None
    for field in ("message", "traceback"):
        m = CLASS.search(str((err or {}).get(field) or ""))
        if m:
            return m.group(1)
    return "unclassified"


def run_class(samples: list) -> str | None:
    return next((c for c in (klass(s.get("error")) for s in samples) if c is not None), None)


@app.function(image=image, cpu=2, memory=4096, timeout=1800, retries=0, volumes={"/ev": modal.Volume.from_name(VOL, version=2).read_only()})
def scan(rows: list) -> dict:
    import io
    import sys
    import zipfile

    sys.path.insert(0, "/root")
    import evalzip  # noqa: F401  (zstd .eval members, as stage_q4b reads them)

    out = {}
    for r in rows:
        raw = (Path("/ev/evals_v1") / r["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != r["sha256"]:
            raise ValueError(f"{r['path']}: sha256 differs from the manifest row")
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            samples = [json.loads(z.read(n)) for n in sorted(z.namelist()) if n.startswith("samples/") and n.endswith(".json")]
        if not samples:
            raise ValueError(f"{r['path']}: no sample in the .eval")
        out[r["path"]] = {"sha256": r["sha256"], "sample_error": run_class(samples)}  # the sha256 this scan asserted
    return out


@app.local_entrypoint()
def main(spec: str, out: str):
    rows = json.loads(Path(spec).read_text())
    res = {}
    for i in range(0, len(rows), 50):  # inputs are paths + shas only (no blobs); 50 runs per call
        res |= scan.remote(rows[i:i + 50])
    Path(out).open("x").write(json.dumps({"rule": "B fork's klass @ d1c147767 (first CLASS match in error.message, else error.traceback, else unclassified)",
                                          "rows": res}, indent=1) + "\n")
    print(json.dumps({"file": out, "runs": len(res), "errors": sum(v["sample_error"] is not None for v in res.values())}))
