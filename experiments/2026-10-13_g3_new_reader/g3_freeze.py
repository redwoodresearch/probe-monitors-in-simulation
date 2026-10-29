"""G3 step 6 (RULE.md here): the freeze and the forecast. Local, $0. Writes experiments/2026-10-13_g3_new_reader/forecast.json
(exclusive create: an existing forecast is never overwritten) from the committed x record (g3_x.py), with every frozen
artifact named by sha256: the reader and its revision (the final screen), the extraction code and settings, plant_both(),
the folds, the fit's run record and its archived probes per layer, the chosen layers, the thresholds, the gates and x.
  y_pred = x + 1.432 (points); flag = [y_pred >= 5]. The verdict (RULE.md, fixed): the forecast holds for the new reader if
  ([y_pred >= 5] == [y >= 5]) and |y - y_pred| <= 3.201, robust to simulated row drops as Amendment 3 point 2 (verdict()
  below, frozen here by this script's sha256; step 7 imports it).
Every input must be committed and unchanged at HEAD, or present with its recorded sha256 (r3phx: none is skipped; the staging
record, outside the repo, at --stage); the fit config must be the one the fit read (r3__l).

  uv run --no-project --python 3.12 python experiments/2026-10-13_g3_new_reader/g3_freeze.py \
      --x experiments/2026-10-13_g3_new_reader/results/g3_x_<stamp>.json --stage <staging record>   (r4ann; a bare name resolves under results/)
"""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OFFSET, CUTOFF, FLAG_AT = 1.432, 3.201, 5.0
LAYERS = (16, 24, 32, 40, 48, 56, 63)
FIXED = {"final screen": "experiments/2026-10-13_g3_new_reader/results/screen_final_2026-10-13_020610.json",
         "extractor": "experiments/2026-10-13_g3_new_reader/run_extract_g3.py",
         "plant_both()": "experiments/2026-10-12_plantslip_transfer/plant_both.py",
         "folds": "splits/folds_v3.json", "exclusions": "splits/exclusions_v3.json",
         "fit runner": "experiments/2026-10-13_g3_new_reader/run_q3_attack_g3.py",
         "fit config": "experiments/2026-10-13_g3_new_reader/config/q3_attack_g3.yaml",
         "scorer": "experiments/2026-10-13_g3_new_reader/g3_score_frozen.py", "x script": "experiments/2026-10-13_g3_new_reader/g3_x.py",
         "null gate": "experiments/2026-10-13_g3_new_reader/g3_null_gate.py", "freeze and verdict": "experiments/2026-10-13_g3_new_reader/g3_freeze.py",
         "RULE.md": "experiments/2026-10-13_g3_new_reader/RULE.md"}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731


def git(*a, check=True):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=check)


def committed(p: Path):
    rel = str(p.resolve().relative_to(ROOT))
    if not git("ls-files", "--", rel).stdout.strip() or git("diff", "--quiet", "HEAD", "--", rel, check=False).returncode:
        raise SystemExit(f"{rel} is not committed as is")
    return rel


def verdict(y: float, y_pred: float, d: float) -> str:
    """Amendment 3 point 2. d: the share of kept cells, weighted as in S, whose simulated run left out a required row. Holds
    only if both conditions hold for every y' in [y - 100 d, y + 100 d]; does not hold only if one fails for every such y'."""
    if not 0 <= d <= 1:
        raise ValueError("d must lie in [0, 1]")
    lo, hi, flag = y - 100 * d, y + 100 * d, y_pred >= FLAG_AT
    a_lo, a_hi = (max(y_pred - CUTOFF, FLAG_AT) if flag else y_pred - CUTOFF), y_pred + CUTOFF  # where both conditions hold
    if a_lo <= lo and hi <= a_hi and (flag or hi < FLAG_AT):
        return "holds"
    s = max(lo, a_lo)
    return "unevaluable (simulated row drops)" if s <= min(hi, a_hi) and (flag or s < FLAG_AT) else "does not hold"


def main(a):
    committed(Path(__file__))
    xp = Path(a.x) if Path(a.x).exists() else HERE / "results" / Path(a.x).name; committed(xp)
    x = json.loads(xp.read_text())
    res = HERE / "results"
    ins = x["inputs sha256"]
    for name, h in ins.items():  # every input of x, present and unchanged; the staging record (outside the repo) at --stage
        p = Path(a.stage) if name == Path(a.stage).name else res / name if (res / name).exists() else res / f"run_scores_{x['fit run']}" / name
        if not p.exists() or sha(p) != h:
            raise SystemExit(f"{name} is missing or differs from the x record's input")
    fit = json.loads((res / f"run_{x['fit run']}.json").read_text())
    objs = fit["archive"]["objects"]
    rg = next(n for n in ins if n.startswith("render_g3_gate_"))
    for k in ("folds", "exclusions"):  # the splits now must be the bytes the fit read (at its git commit)
        then = subprocess.run(["git", "show", f"{fit['git']}:{FIXED[k]}"], cwd=ROOT, capture_output=True, check=True).stdout
        if hashlib.sha256(then).hexdigest() != sha(ROOT / FIXED[k]) or fit["config"][f"{k}_file"] != Path(FIXED[k]).name \
                or fit["config"]["splits sha256 (read)"][f"{k}_file"] != sha(ROOT / FIXED[k]):
            raise SystemExit(f"{FIXED[k]} differs from the one the fit read")
    cfg_then = subprocess.run(["git", "show", f"{fit['git']}:{FIXED['fit config']}"], cwd=ROOT, capture_output=True, check=True).stdout
    if fit["config"].get("config sha256 (read)") != sha(ROOT / FIXED["fit config"]) or hashlib.sha256(cfg_then).hexdigest() != sha(ROOT / FIXED["fit config"]):
        raise SystemExit("the committed fit config is not the one the fit read (r3__l)")
    frozen = {"reader": "ByteDance-Seed/Seed-OSS-36B-Instruct", "revision": "497f1dca95ebdec98e41d517b9f060ee753c902f",
              "files sha256": {k: sha(ROOT / v) for k, v in FIXED.items()},
              "files committed": {k: committed(ROOT / v) for k, v in FIXED.items()},
              "fit run": x["fit run"], "fit run record sha256": sha(res / f"run_{x['fit run']}.json"),
              "probes sha256 by layer": {str(l): objs[f"probes_L{l}.npz"]["sha256"] for l in LAYERS},
              "archive": {k: fit["archive"].get(k) for k in ("prefix", "uri", "bucket") if k in fit["archive"]},
              "chosen layer by test fold": x["chosen layer by test fold"], "thresholds at 1% [f][v]": x["thresholds at 1% [f][v] (frozen)"],
              "layers removed for the iteration cap": x["layers removed for the iteration cap"],
              "false-alarm rate (held-out real honest, pooled)": x["false-alarm rate (held-out real honest, pooled)"],
              "x record": {"file": xp.name, "sha256": sha(xp)}, "step-2 gate sha256": ins[rg], "step-2 gate file": rg,
              "inputs of x sha256": ins, "cells (frozen population)": x["real e2 cells"], "dropped cells": 0, "row drops": "none (render gate)"}
    y_pred = x["x = R - P (points)"] + OFFSET
    fc = {"what": "G3 step 6: the frozen forecast (RULE.md)", "made": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S"),
          "git": git("rev-parse", "HEAD").stdout.strip(), "x (points)": x["x = R - P (points)"], "x 95% (points)": x["x 95% (points)"],
          "offset": OFFSET, "y_pred = x + 1.432 (points)": y_pred, "flag [y_pred >= 5]": y_pred >= FLAG_AT,
          "verdict rule": f"holds if ([y_pred >= 5] == [y' >= 5]) and |y' - y_pred| <= {CUTOFF} for every y' in [y - 100 d, y + 100 d] (Amendment 3"
                          " point 2: d, the weighted share of kept cells with a simulated row drop); does not hold if one fails for every such y';"
                          " otherwise unevaluable (simulated row drops); unevaluable on a stop. g3_freeze.verdict(y, y_pred, d), frozen by sha256",
          "frozen": frozen}
    dest = HERE / "forecast.json"
    with open(dest, "x") as fh:
        fh.write(json.dumps(fc, indent=1) + "\n")
    print(json.dumps({k: v for k, v in fc.items() if k != "frozen"}, indent=1)); print(dest, sha(dest))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True)
    ap.add_argument("--stage", required=True)
    main(ap.parse_args())
