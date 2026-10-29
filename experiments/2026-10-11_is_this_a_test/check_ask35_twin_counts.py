"""Boring check 1 for ask35 step 1's fail (0d's request, 2026-10-12 about 12:35 PT, verbatim: "the Modal CPU count of
R2-reverted outputs per pair within the 32k cut. Counts only, cents, under the <$5 rule. Commit the rule first; report the
distribution (median, share of pairs with 0 changed outputs inside the cut). Don't read content.").

For each of step 1's 100 twin pairs, on Modal CPU where the pack lives: the messages where the twin differs from its
fixed run, split by step 1's cut (the kept-message count step 1 recorded for the pair, from its rows): inside the cut
(index < kept) or after it. A message counts once if any field differs. Only integers leave: per pair, the two counts
under its hashed cell id. Reported: per-pair counts, the median and quartiles inside the cut, the share of pairs with 0
inside, and the totals. Descriptive only; it decides nothing.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal --with numpy modal run experiments/2026-10-11_is_this_a_test/check_ask35_twin_counts.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-itat-ask35-count_kill-2026-11-12"
PACKS, MANIFEST_SHA = "2026-10-10_la_september-onpolicy-l1_v1", "6aa7928c5986a4002cdcf55168485f8d68eb3a5b682f4d136d8dc89612357352"
CONTROL = "results/ask35_ask_control_2026-10-12_115838.json"
app = modal.App(APP)
la = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2).read_only()
cell_id = lambda c: hashlib.sha256(f"ask35|{c}".encode()).hexdigest()[:16]


@app.function(image=modal.Image.debian_slim(python_version="3.12"), volumes={"/in": la}, cpu=(2, 2), memory=(4096, 4096),
              timeout=600, startup_timeout=300, retries=0)
def count(keep: dict, own_sha: str) -> dict:
    import os
    import traceback
    null = os.open(os.devnull, os.O_WRONLY)  # nothing from the runs reaches the logs
    os.dup2(null, 1), os.dup2(null, 2)
    try:
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != own_sha:
            raise ValueError("the deployed script is not the launching commit's")
        base = Path("/in") / PACKS
        man = (base / "manifest.json").read_bytes()
        if hashlib.sha256(man).hexdigest() != MANIFEST_SHA:
            raise ValueError("the pack's manifest is not step 1's")
        want = {Path(k).name: v for k, v in json.loads(man)["files"].items() if Path(k).name.startswith("shard_") and Path(k).parent.name == "inputs"}
        runs = {}
        for p in sorted((base / "inputs").glob("shard_*.jsonl")):
            raw = p.read_bytes()
            if hashlib.sha256(raw).hexdigest() != want[p.name]:
                raise ValueError("a shard does not match the manifest")
            for it in map(json.loads, raw.decode().splitlines()):
                if it["arm"] in ("sim-attack-sept-fixed", "sim-attack-sept-fixed-plant2"):
                    runs[(cell_id(it["cell"]), it["arm"])] = it["messages"]
        out = {}
        for c, k in keep.items():
            f, t = runs[(c, "sim-attack-sept-fixed")], runs[(c, "sim-attack-sept-fixed-plant2")]
            if len(f) != len(t):
                raise ValueError("a twin has a different number of messages")
            diff = [j for j in range(len(f)) if f[j] != t[j]]
            out[c] = {"inside": sum(j < k for j in diff), "after": sum(j >= k for j in diff)}
        return out
    except BaseException as e:
        lines = [x.lineno for x in traceback.extract_tb(e.__traceback__) if x.filename.endswith("check_ask35_twin_counts.py")]
        raise RuntimeError(f"count failed: {type(e).__name__} at lines {lines}") from None


@app.local_entrypoint()
def main():
    import numpy as np

    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=ROOT)
    st = git("status", "--porcelain", "--untracked-files=no")
    if st.stdout.strip() or git("ls-files", "--error-unmatch", __file__).returncode:
        raise SystemExit("commit first: the run must come from committed code")
    ctl = json.loads((HERE / CONTROL).read_text())
    f = ROOT / ".run_cache" / "is_this_a_test" / ctl["rows"]["file"]
    if hashlib.sha256(f.read_bytes()).hexdigest() != ctl["rows"]["sha256"]:
        raise ValueError("step 1's rows file is not the collected one")
    rows = [r for r in json.loads(f.read_text()) if "dropped" not in r]
    keep = {r["cell"]: r["kept_messages"] for r in rows if r["arm"] == "twin"}
    if any(r["kept_messages"] != keep[r["cell"]] for r in rows if r["arm"] == "sim" and r["cell"] in keep) or len(keep) != 100:
        raise ValueError("step 1's twins and fixed runs do not share their cuts, or there are not 100 pairs")
    call = count.spawn(keep, hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:  # a 12-minute deadline for the whole job, restarts included; the ephemeral app stops when this launcher exits
        got = call.get(timeout=720)
    except (TimeoutError, modal.exception.TimeoutError):
        call.cancel()
        raise SystemExit("stopped at the 720 s deadline")
    inside = np.array([v["inside"] for v in got.values()])
    after = np.array([v["after"] for v in got.values()])
    out = {"git": git("rev-parse", "HEAD").stdout.strip(), "control read": CONTROL, "rows sha256": ctl["rows"]["sha256"], "pairs": len(got),
           "changed messages inside the cut": {"median": float(np.median(inside)), "quartiles": [float(x) for x in np.percentile(inside, [25, 75])],
                                               "max": int(inside.max()), "total": int(inside.sum()), "share of pairs with 0": float(np.mean(inside == 0))},
           "changed messages after the cut": {"median": float(np.median(after)), "total": int(after.sum()), "share of pairs with 0": float(np.mean(after == 0))},
           "per pair (hashed cell: inside, after)": {c: [v["inside"], v["after"]] for c, v in sorted(got.items())}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"ask35_twin_counts_{stamp}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text("{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}" for k, v in out.items()) + "\n}\n")  # one key per line
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("per pair")}, indent=1), f"\n-> {path}")
