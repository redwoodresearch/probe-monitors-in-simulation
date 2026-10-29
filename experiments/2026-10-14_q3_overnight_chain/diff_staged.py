"""By-key byte diff of two staged sets on the chain Volume (09 via d9 10-16 ~01:05 / ~01:45: a re-stage may change ONLY the recovered rows): per
arm file inputs/<arm>.jsonl, each line (its exact bytes, newline included) keyed by its "key". Reports per arm: rows, identical, only_a, only_b,
differ (sha256 of the bytes), whether the common keys come in the same order, and the file's trailing-newline state. Both sets' inputs dirs
must exist and hold the same arm files (else it stops). Keys and hashes only (no ids, no text).

  modal run diff_staged.py --a <staged set> --b <staged set> --out <json>
"""
import hashlib
import json
from pathlib import Path

import modal

VOL = "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14"
app = modal.App("probe-shift_aswin-surya_sub-q3chain_diffstaged_kill-2026-11-14")


@app.function(image=modal.Image.debian_slim(python_version="3.13"), cpu=2, memory=8192, timeout=1800, retries=0,
              volumes={"/ev": modal.Volume.from_name(VOL, version=2).read_only()})
def diff(a: str, b: str) -> dict:
    dirs = {s: Path("/ev") / s / "inputs" for s in (a, b)}
    for s, p in dirs.items():
        if not p.is_dir():
            raise FileNotFoundError(f"{s}: no inputs dir on the Volume")
    arms = {s: sorted(x.name for x in p.glob("*.jsonl")) for s, p in dirs.items()}
    if arms[a] != arms[b] or not arms[a]:
        raise ValueError(f"arm files differ or are empty: {arms}")
    out = {}
    for arm in arms[a]:
        side, order, tail = {}, {}, {}
        for s in (a, b):
            raw = (dirs[s] / arm).read_bytes()
            tail[s] = raw.endswith(b"\n")
            rows, keys = {}, []
            for line in raw.splitlines(keepends=True):  # each line's exact bytes, its own ending included (Codex r109)
                k = json.loads(line)["key"]
                assert k not in rows, (s, arm, k)
                rows[k] = hashlib.sha256(line).hexdigest()
                keys.append(k)
            side[s], order[s] = rows, keys
        A, B = side[a], side[b]
        common = A.keys() & B.keys()
        out[arm] = {"rows": [len(A), len(B)], "identical": sum(A[k] == B[k] for k in common),
                    "only_a": sorted(A.keys() - B.keys()), "only_b": sorted(B.keys() - A.keys()),
                    "differ": sorted(k for k in common if A[k] != B[k]),
                    "common_order_equal": [k for k in order[a] if k in common] == [k for k in order[b] if k in common],
                    "trailing_newline": [tail[a], tail[b]]}
    return {"a": a, "b": b, "arms": out}


@app.local_entrypoint()
def main(a: str, b: str, out: str):
    r = diff.remote(a, b)
    Path(out).open("x").write(json.dumps(r, indent=1) + "\n")
    print(json.dumps({arm: {k: (v if not isinstance(v, list) or k in ("rows", "trailing_newline") else len(v)) for k, v in x.items()} for arm, x in r["arms"].items()}))
