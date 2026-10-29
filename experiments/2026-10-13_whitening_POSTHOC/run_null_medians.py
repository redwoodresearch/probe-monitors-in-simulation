"""PR review (#1492, #1493): the null medians as the mean of the two middle values, for every cell, null design, threshold and
metric of the two stored null results (Amendments 4 and 6). One detached Modal CPU function; the client only checks the
inputs' sha256 against their manifests and writes the returned JSON. No fit, no resampling.

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach \\
      experiments/2026-10-13_whitening_POSTHOC/run_null_medians.py --scratch <dir with the result files> --out <results json>
"""

import hashlib
import json
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-whitening_medians_kill-2026-11-13"
INPUTS = {  # manifests null v1 and null_leak v1, relative to --scratch
    "within_task": ("null/null_POSTHOC_2026-10-14_002519.json", "afecc3e837ac69e5e04ad66cedd04131fa4b9b8c703675a5419c03161397b051"),
    "leak": ("leak/null_leak_POSTHOC_2026-10-14_004344.json", "c39573b53f123df9ae15e99daf1b6b950cbf96a78c272f3bb8fe103d5251c13e"),
}
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=image, cpu=(1, 1), memory=(1024, 1024), timeout=600, retries=0)
def run(results: dict) -> list:
    rows = []
    for design, r in results.items():
        for cell, x in r["cells"].items():
            for alpha, q in x.items():
                for metric in ("e2", "fake"):
                    v = sorted(q[f"null_{metric}"])
                    k = len(v)
                    true = v[k // 2] if k % 2 else (v[k // 2 - 1] + v[k // 2]) / 2
                    rows.append({"design": design, "cell": cell, "alpha": alpha, "metric": metric, "n": k, "median": true, "upper_middle": v[k // 2],
                                 "stored_median": q.get("null_median", {}).get(metric), "min": v[0], "max": v[-1], "observed": q["observed"][metric]})
    return rows


@app.local_entrypoint()
def main(scratch: str = "", out: str = ""):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO, check=True).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=all"):
        raise SystemExit("refusing to run from a tree with changes or untracked files")
    results = {}
    for k, (rel, h) in INPUTS.items():
        b = (Path(scratch) / rel).read_bytes()
        if hashlib.sha256(b).hexdigest() != h:
            raise SystemExit(f"{rel}: sha256 differs from its manifest")
        results[k] = json.loads(b)
    rows = run.remote(results)
    Path(out).open("x").write(json.dumps({"git": git("rev-parse", "HEAD"), "app": APP, "inputs": INPUTS, "rows": rows}, indent=1))
    lines = ["design | cell | alpha | metric: observed | median (mean of the two middle) | upper middle | stored median | range (percent)"]
    for r in rows:
        sm = "-" if r["stored_median"] is None else f"{100 * r['stored_median']:.2f}"
        lines.append(f"{r['design']} | {r['cell']} | {r['alpha']} | {r['metric']}: {100 * r['observed']:.1f} | {100 * r['median']:.2f} | {100 * r['upper_middle']:.2f} | {sm} | "
                     f"{100 * r['min']:.1f}-{100 * r['max']:.1f}")
    txt = "\n".join(lines)
    print(txt)
    Path(out[:-5] + ".printout.txt").open("x").write(txt + "\n")
