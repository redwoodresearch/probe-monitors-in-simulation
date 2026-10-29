"""RULE_item2.md here: Sol's epoch-2 side-task success on T196 (all epoch-1 successes by selection), with vs without a
strategy-bearing side task. Modal CPU; counts and rates only.

  APX_IN=<dir with reg/, t196.json> PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 \\
      modal run experiments/2026-10-14_approach_provenance_09/run_item2_t196.py
"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
IN = Path(os.environ.get("APX_IN", "/nonexistent"))
APP = "probe-shift_aswin-surya_sub-convaudit_item2_kill-2026-11-14"
PIN = "55d9a8ec68f46854"

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("scipy==1.16.2")
if IN.exists():
    image = image.add_local_dir(IN / "reg", "/root/in/reg").add_local_file(
        IN / "t196.json", "/root/in/t196.json"
    )


def wilson(k: int, n: int, z: float = 1.959964) -> list:
    if n == 0:
        return [None, None]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return [c - h, c + h]


@app.function(image=image, cpu=1, memory=2048, timeout=600, retries=0)
def analyze() -> dict:
    import hashlib
    from collections import Counter

    from scipy.stats import fisher_exact

    raw = Path("/root/in/t196.json").read_bytes()
    if hashlib.sha256(raw).hexdigest()[:16] != PIN:
        raise ValueError("t196 pin mismatch")
    cells = json.loads(raw)["cells"]
    # reg/<env>/side_tasks/<side>/docs/attack_strategies/<main>/<file>.md -> (env, side)
    has = {
        (p.parents[5].name, p.parents[3].name)
        for p in Path("/root/in/reg").glob(
            "*/side_tasks/*/docs/attack_strategies/*/*.md"
        )
    }
    if (
        len(
            list(
                Path("/root/in/reg").glob(
                    "*/side_tasks/*/docs/attack_strategies/*/*.md"
                )
            )
        )
        != 54
    ):
        raise ValueError("registry file count mismatch")
    e1 = Counter(c["sol_e1"]["side_success"] for c in cells)
    rows = []
    for c in cells:
        if len(c["sol_e2"]) != 1:
            raise ValueError("expected one epoch-2 record per cell")
        rows.append(
            {
                "env": c["env"],
                "strategy": (c["env"], c["side"]) in has,
                "e2": c["sol_e2"][0]["side_success"],
            }
        )
    other = Counter(r["e2"] for r in rows if r["e2"] not in ("C", "I"))
    out = {
        "n_cells": len(rows),
        "e1_values": dict(e1),
        "e2_other_values": dict(other),
        "strata": {},
    }
    ks = {}
    for s in (True, False):
        rs = [r for r in rows if r["strategy"] is s and r["e2"] in ("C", "I")]
        k, n = sum(r["e2"] == "C" for r in rs), len(rs)
        ks[s] = (k, n)
        out["strata"]["with_strategy" if s else "without_strategy"] = {
            "e2_C": k,
            "n": n,
            "rate": k / n if n else None,
            "wilson95": wilson(k, n),
        }
    (k1, n1), (k0, n0) = ks[True], ks[False]
    p1, p0 = k1 / n1, k0 / n0
    l1, u1 = wilson(k1, n1)
    l0, u0 = wilson(k0, n0)
    d = p1 - p0  # Newcombe hybrid score interval (method 10)
    out["difference_with_minus_without"] = {
        "estimate": d,
        "newcombe95": [
            d - ((p1 - l1) ** 2 + (u0 - p0) ** 2) ** 0.5,
            d + ((u1 - p1) ** 2 + (p0 - l0) ** 2) ** 0.5,
        ],
        "fisher_two_sided_p": float(fisher_exact([[k1, n1 - k1], [k0, n0 - k0]])[1]),
    }
    out["by_env"] = {
        e: {
            s_name: f"{sum(r['e2'] == 'C' for r in rows if r['env'] == e and r['strategy'] is s)}"
            f"/{sum(r['env'] == e and r['strategy'] is s and r['e2'] in ('C', 'I') for r in rows)}"
            for s_name, s in (("with", True), ("without", False))
        }
        for e in sorted({r["env"] for r in rows})
    }
    out["labels"] = ["T196 only, conditional on epoch-1 success"] + (
        ["without-strategy stratum n < 30: underpowered"] if n0 < 30 else []
    )
    return out


@app.local_entrypoint()
def main():
    def git(*a):
        return subprocess.run(
            ["git", *a], capture_output=True, text=True, check=True, cwd=REPO
        ).stdout.strip()

    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    head = git("rev-parse", "HEAD")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = analyze.remote()
    out = {
        "label": "RULE_item2.md; T196 only, conditional on epoch-1 success; counts and rates",
        "git": head,
        "stamp": stamp,
        "result": res,
    }
    dest = HERE / "results" / f"item2_t196_{stamp}.json"
    dest.open("x").write(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)
