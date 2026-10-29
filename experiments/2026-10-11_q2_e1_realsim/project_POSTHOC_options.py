"""E1 real vs simulated, redesign pilot: post hoc projections for the plan's fallback options. The pilot failed its
usability bar (results/redesign_pilot_readout_2026-10-11_182405.json), so PLAN_REAL_VS_SIM.md brings the stage back with
the options: a looser pair rule, C's same-batch pairs, more reports per run, or closing. This script projects the
chance of a clear joint verdict (S and R both pass at ±0.5, true gaps 0) for the first two, with the pilot readout's own
projection formula. Written after the readout was read; no verdict, and no arm mean or difference is computed or written.

Rules:
- primary: the readout's (both reports at least 10 scored tokens and finished by stop; hide omits and instructed
  discloses under the section-aware rule);
- looser: the plan's all-task secondary (both reports at least 10 scored tokens and finished by stop; no disclosure
  filter).

R tasks: the plan's 163 same-batch tasks, or 163 plus the plan's ~233 pairs from zoo C's new real runs (the plan's
figure, not a count). Local CPU, $0.

  uv run --no-project --python 3.12 --with numpy --with "transformers==5.19.0" --with jinja2 python \\
      experiments/2026-10-11_q2_e1_realsim/project_POSTHOC_options.py <pilot stage folder> <pilot index jsonl>
"""
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "2026-10-11_q2_e1_honest"))
from analyze_e1 import MODEL, REVISION, discloses_section, report_text  # noqa: E402
from masks_lib import build_masks  # noqa: E402

INDEX_SHA = "c288eca8b29da7f1d7b8538217ba562b66fb8c74fae9000be39b27ac4db893ee"
READOUT = HERE / "results/redesign_pilot_readout_2026-10-11_182405.json"
SD0, MIN_TOKENS, MARGIN, TASKS, S_TASKS = 0.5396148830137296, 10, 0.5, 40, 437
R_TASKS = {"163 same-batch tasks": 163, "163 + the plan's ~233 from zoo C": 163 + 233}
ARMS = ("real_a", "real_b", "sim")


def projection(sd, joint, r_tasks):
    """The pilot readout's projection (analyze_realsim2_pilot.py), with the R task count as a parameter."""
    n_s, n_r = S_TASKS * joint["real_a & sim"], r_tasks * joint["real_a & real_b"]
    if any(v is None for v in sd.values()) or n_s < 2 or n_r < 2:
        return {"usable tasks for S": n_s, "usable tasks for R": n_r, "P": None}
    se_s = math.sqrt((sd["real_a"] ** 2 + sd["sim"] ** 2) / n_s); se_r = math.sqrt((sd["real_a"] ** 2 + sd["real_b"] ** 2) / n_r)
    rho = sd["real_a"] ** 2 / math.sqrt((sd["real_a"] ** 2 + sd["sim"] ** 2) * (sd["real_a"] ** 2 + sd["real_b"] ** 2))
    g = np.random.default_rng(20261011); z1, z2 = g.standard_normal(200_000), g.standard_normal(200_000)
    x, y = se_s * z1, se_r * (rho * z1 + math.sqrt(1 - rho ** 2) * z2)
    p = float(((np.abs(x) < MARGIN - 1.96 * se_s) & (np.abs(y) < MARGIN - 1.96 * se_r)).mean())
    return {"usable tasks for S": n_s, "usable tasks for R": n_r, "SE of S": se_s, "SE of R": se_r, "corr(S, R)": rho, "P": p}


def main(folder: Path, index: Path):
    if hashlib.sha256(index.read_bytes()).hexdigest() != INDEX_SHA:
        raise SystemExit("pilot index differs")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    for f in (Path(__file__).resolve(), HERE / "masks_lib.py", HERE.parent / "2026-10-11_q2_e1_honest/analyze_e1.py"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    from transformers import AutoTokenizer
    rows = [json.loads(line) for line in index.open()]
    path_of = {r["key"]: r["path"] for r in rows}
    masks = build_masks(folder, path_of)
    flags = json.loads((folder / "flags.json").read_text())
    scores = {(r["key"], r["seq"]): r for r in (json.loads(p.read_text()) for p in (folder / "scores").glob("*.json"))}
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)

    def readout(key, seq):
        r = scores.get((key, seq)); idx = masks.get(f"{key}|{seq}", {}).get("scored", [])
        if r is None or r.get("oom") or len(idx) < MIN_TOKENS:
            return None
        return float(np.asarray(r["per_token"])[idx].mean())

    usable, d = {"primary": {}, "looser": {}}, {"primary": {}, "looser": {}}
    for f in sorted((folder / "generations").glob("*.json")):
        g = json.loads(f.read_text()); key = g["key"]; task, arm = key.rsplit("__", 1)
        h, i = readout(key, "hide"), readout(key, "instructed_honest")
        stop = flags.get(f"{key}|hide", {}).get("stop") and flags.get(f"{key}|instructed_honest", {}).get("stop")
        rep = {c: report_text(tok, g[c]["completion_ids"]) for c in ("hide", "instructed_honest")}
        base = h is not None and i is not None and bool(stop)
        rule = {"looser": base, "primary": base and not discloses_section(rep["hide"], path_of[key]) and discloses_section(rep["instructed_honest"], path_of[key])}
        for name, ok in rule.items():
            usable[name][(task, arm)] = ok
            if ok:
                d[name][(task, arm)] = (h - i) / SD0
    tasks = sorted({r["main_task"] for r in rows} | {t for t, _ in usable["primary"]})
    out = {"label": "POST HOC (after the pilot readout was read; no verdict)", "git": head, "index sha256": INDEX_SHA}
    for name in ("primary", "looser"):
        u = {a: np.array([usable[name].get((t, a), False) for t in tasks]) for a in ARMS}
        joint = {"real_a & sim": float((u["real_a"] & u["sim"]).sum() / TASKS), "real_a & real_b": float((u["real_a"] & u["real_b"]).sum() / TASKS)}
        sd = {a: float(np.std([v for (t, aa), v in d[name].items() if aa == a], ddof=1)) if sum(aa == a for _, aa in d[name]) >= 2 else None for a in ARMS}
        out[name] = {"usable pairs per arm (of 40 tasks)": {a: int(u[a].sum()) for a in ARMS}, "both usable (share of 40)": joint,
                     "per-run SD of D per arm": sd, "projection": {k: projection(sd, joint, n) for k, n in R_TASKS.items()}}
    ro = json.loads(READOUT.read_text())
    if (out["primary"]["usable pairs per arm (of 40 tasks)"] != ro["usable primary pairs per arm"]
            or abs(out["primary"]["projection"]["163 same-batch tasks"]["P"] - ro["full-stage projection"]["P(clear joint verdict at ±0.5, true gaps 0)"]) > 1e-12):
        raise SystemExit("the primary rule doesn't reproduce the pilot readout")
    out["primary reproduces the readout"] = True
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"options_projection_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
