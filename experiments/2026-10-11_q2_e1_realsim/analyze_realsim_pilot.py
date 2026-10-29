"""E1 real-vs-simulated pilot readout (PLAN_REAL_VS_SIM.md @ e74b53c5; committed before any pilot score is read). It
reports only what the pilot is for: usability per arm, usability's correlation across worlds, the spread of D, time, and
the projected chance of a clear joint verdict for the full stage at ±0.5. No arm mean or difference of means is
computed or written. Masks come from the frozen sentence rule (masks_lib, checked against stage 1's committed masks).

  uv run --no-project --python 3.12 --with numpy --with "transformers==5.19.0" --with jinja2 python \\
      experiments/2026-10-11_q2_e1_realsim/analyze_realsim_pilot.py <pilot stage folder> <pilot index jsonl>
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
SD0, MIN_TOKENS, MARGIN, TASKS, FULL = 0.5396148830137296, 10, 0.5, 40, {"S tasks": 437, "R tasks": 163}
ARMS = ("real_a", "real_b", "sim")


def main(folder: Path, index: Path):
    if hashlib.sha256(index.read_bytes()).hexdigest() != INDEX_SHA:
        raise SystemExit("pilot index differs")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    me = Path(__file__).resolve()
    if (subprocess.run(["git", "ls-files", "--error-unmatch", str(me)], cwd=HERE, capture_output=True).returncode
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(me)], cwd=HERE).returncode):
        raise SystemExit("the script differs from HEAD: commit it before running")
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

    usable, d, oom = {}, {}, sum(1 for r in scores.values() if r.get("oom"))
    for f in sorted((folder / "generations").glob("*.json")):
        g = json.loads(f.read_text()); key = g["key"]
        task, arm = key.rsplit("__", 1)
        h, i = readout(key, "hide"), readout(key, "instructed_honest")
        fl = lambda c: flags.get(f"{key}|{c}", {})
        rep = {c: report_text(tok, g[c]["completion_ids"]) for c in ("hide", "instructed_honest")}
        ok = (h is not None and i is not None and fl("hide").get("stop") and fl("instructed_honest").get("stop")
              and not discloses_section(rep["hide"], path_of[key]) and discloses_section(rep["instructed_honest"], path_of[key]))
        usable[(task, arm)] = bool(ok)
        if ok:
            d[(task, arm)] = (h - i) / SD0
    tasks = sorted({r["main_task"] for r in rows} | {t for t, _ in usable})
    u = {a: np.array([usable.get((t, a), False) for t in tasks]) for a in ARMS}
    pad = TASKS - len(tasks)  # tasks with no eligible run in any arm count as unusable everywhere
    rate = {a: float(u[a].sum() / TASKS) for a in ARMS}
    joint = {"real_a & sim": float((u["real_a"] & u["sim"]).sum() / TASKS), "real_a & real_b": float((u["real_a"] & u["real_b"]).sum() / TASKS)}

    def phi(x, y):
        x = np.r_[x, np.zeros(pad, bool)]; y = np.r_[y, np.zeros(pad, bool)]
        return float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else None
    sd = {a: float(np.std([v for (t, aa), v in d.items() if aa == a], ddof=1)) if sum(aa == a for _, aa in d) >= 2 else None for a in ARMS}
    out = {"plan": "PLAN_REAL_VS_SIM.md @ e74b53c5", "git": head, "index sha256": INDEX_SHA, "tasks": TASKS, "eligible runs": len(rows),
           "scored sequences out of memory": oom, "usable primary pairs per arm": {a: int(u[a].sum()) for a in ARMS},
           "usable share of the 40 tasks": rate, "both usable (share of 40)": joint,
           "usability correlation (phi over 40 tasks)": {"real_a vs sim": phi(u["real_a"], u["sim"]), "real_a vs real_b": phi(u["real_a"], u["real_b"])},
           "per-run SD of D (frozen stage-1 honest SD units), per arm": sd}
    enough = all(u[a].sum() >= 10 for a in ARMS)
    if all(v is not None for v in sd.values()):
        n_s, n_r = FULL["S tasks"] * joint["real_a & sim"], FULL["R tasks"] * joint["real_a & real_b"]
        if n_s >= 2 and n_r >= 2:
            se_s = math.sqrt((sd["real_a"] ** 2 + sd["sim"] ** 2) / n_s); se_r = math.sqrt((sd["real_a"] ** 2 + sd["real_b"] ** 2) / n_r)
            rho = sd["real_a"] ** 2 / math.sqrt((sd["real_a"] ** 2 + sd["sim"] ** 2) * (sd["real_a"] ** 2 + sd["real_b"] ** 2))
            g = np.random.default_rng(20261011); z1, z2 = g.standard_normal(200_000), g.standard_normal(200_000)
            x, y = se_s * z1, se_r * (rho * z1 + math.sqrt(1 - rho ** 2) * z2)
            p = float(((np.abs(x) < MARGIN - 1.96 * se_s) & (np.abs(y) < MARGIN - 1.96 * se_r)).mean())
            out["full-stage projection"] = {"usable tasks for S": n_s, "usable tasks for R": n_r, "SE of S": se_s, "SE of R": se_r, "corr(S, R)": rho,
                                            "P(clear joint verdict at ±0.5, true gaps 0)": p}
    proj = out.get("full-stage projection", {}).get("P(clear joint verdict at ±0.5, true gaps 0)")
    out["decision"] = ("extend once to 80 tasks (fewer than 10 usable pairs in an arm)" if not enough else
                       "full stage launches (projected P >= 0.80)" if proj is not None and proj >= 0.80 else
                       "comes back with numbers (projected P under 0.80)")
    pj = folder / "pipeline.json"
    out["pipeline"] = json.loads(pj.read_text()) if pj.exists() else None
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"pilot_readout_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "pipeline"}, indent=1)); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
