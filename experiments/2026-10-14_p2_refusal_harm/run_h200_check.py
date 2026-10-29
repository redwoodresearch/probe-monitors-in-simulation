"""RULE.md Amendment 2: the H200 cross-hardware check of the stage-1 directions. The exact stage-1 read (run_read_fit.read_fit's raw
function) runs on ONE H200; then on CPU both difference-in-means directions are recomputed at the SAME layers (harmfulness L34 at
inst, refusal L47 at post) and compared with the H100 directions (cosine); the local entrypoint writes a record the shared scorer
reads. `compare` then diffs the H100 and H200 scoring records (catch at 1%/5%, every probe x readout x arm).

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run experiments/2026-10-14_p2_refusal_harm/run_h200_check.py
  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run experiments/2026-10-14_p2_refusal_harm/run_h200_check.py::compare \\
      --h100 <scores_...130313.json> --h200 <scores_...json> --check <h200_check_...json>
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_read_fit as rrf  # the stage-1 module, unchanged

APP = "probe-shift_aswin-surya_sub-p2probes_refh200_kill-2026-11-13"
FIXED = {"harmfulness": ("inst", 34), "refusal": ("post", 47)}
H100 = {"harmfulness": "harmfulness_direction_2026-10-14_125514.safetensors", "refusal": "refusal_direction_2026-10-14_125514.safetensors"}
COS_BAR, CATCH_BAR = 0.999, 0.005

app = modal.App(APP)
gpu_image = rrf.image.add_local_file(HERE / "run_read_fit.py", "/root/run_read_fit.py")
cpu_image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "safetensors==0.8.0")
             .add_local_file(HERE / "run_read_fit.py", "/root/run_read_fit.py"))  # this module imports it at load time (Codex)


@app.function(image=gpu_image, gpu=["H200"], cpu=(8, 8), memory=(98304, 98304), timeout=1200, retries=0,
              volumes={"/hf": modal.Volume.from_name("probe-shift_aswin-surya_hf-cache").read_only(), "/reads": rrf.reads})
def read_h200(stamp: str) -> dict:
    import torch

    r = rrf.read_fit.local(stamp)  # the stage-1 read and fit, byte-identical code, run inside this H200 container
    return {k: v for k, v in r.items() if k != "files"} | {"gpu": torch.cuda.get_device_name(0)}


@app.function(image=cpu_image, cpu=(4, 4), memory=(16384, 16384), timeout=900, retries=0, volumes={"/reads": rrf.reads})
def fixed_dirs(stamp: str, h100: dict) -> dict:
    import numpy as np
    import safetensors.numpy as stn

    rrf.reads.reload()
    d = Path("/reads") / "refusal" / stamp
    X = np.load(d / "rows.npy", mmap_mode="r")
    items = json.loads((d / "items.json").read_text())
    split, lab = np.array([x["split"] for x in items]), np.array([x["label"] for x in items])
    out, files = {}, {}
    for name, (pos, L) in FIXED.items():
        pi = rrf.POS.index(pos)
        tr = np.asarray(X[split == "train", pi, L]).astype(np.float32)
        y = lab[split == "train"]
        r = tr[y == 1].mean(0) - tr[y == 0].mean(0)
        u = (r / np.linalg.norm(r)).astype(np.float32)
        old = stn.load(h100[name])["direction"].astype(np.float64)
        m = {"cosine_h100_h200": float(u.astype(np.float64) @ old / (np.linalg.norm(old) * np.linalg.norm(u.astype(np.float64))))}
        for s_ in ("val", "test"):
            sc, ys = np.asarray(X[split == s_, pi, L]).astype(np.float32) @ u, lab[split == s_]
            m[f"{s_}_auroc"] = rrf._auroc(sc[ys == 1], sc[ys == 0])
        out[name] = {"position": pos, "layer": L, "passes": m["val_auroc"] >= rrf.GATE, "chosen": m}
        files[name] = stn.save({"direction": u}, metadata={"layer": str(L), "position": pos, "gpu": "H200", "stamp": stamp})
    return {"directions": out, "files": files}


@app.local_entrypoint()
def main():
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if git("status", "--porcelain", "--", ".") or git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("uncommitted changes")
    head = git("rev-parse", "HEAD")
    if not git("branch", "-r", "--contains", head):
        raise SystemExit("HEAD is not pushed")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rd = read_h200.remote(stamp)
    fx = fixed_dirs.remote(stamp, {n: (HERE / "results" / f).read_bytes() for n, f in H100.items()})
    res = HERE / "results"
    rec = {"label": "RULE.md Amendment 2 (H200 cross-hardware check)", "git": head, "app": APP, "stamp (PT)": stamp, "gpu": rd["gpu"],
           "h200_run_own_layer_choice (descriptive)": {n: d["layer"] for n, d in rd["directions"].items()}, "load_s": rd["load_s"], "forward_s": rd["forward_s"],
           "reads": f"{rrf.READS}:/refusal/{stamp}", "directions": fx["directions"]}
    for name, b in fx["files"].items():
        f = res / f"{name}_direction_h200_{stamp}.safetensors"
        f.open("xb").write(b)
        rec["directions"][name]["file"] = {"path": f.name, "sha256": hashlib.sha256(b).hexdigest()}
    dest = res / f"h200_check_{stamp}.json"
    dest.open("x").write(json.dumps(rec, indent=1))
    lines = [f"gpu {rd['gpu']}; H200 run's own layer choice {rec['h200_run_own_layer_choice (descriptive)']} (descriptive)"]
    lines += [f"{n}: L{d['layer']} cosine H100 vs H200 {d['chosen']['cosine_h100_h200']:.6f}; val AUROC {d['chosen']['val_auroc']:.4f}; test {d['chosen']['test_auroc']:.4f}"
              for n, d in rec["directions"].items()]
    dest.with_suffix(".printout.txt").open("x").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(dest.name)


@app.function(image=cpu_image, cpu=(2, 2), memory=(4096, 4096), timeout=600, retries=0)
def diff(a: dict, b: dict) -> dict:
    """Every probe x readout x catch (gate run1, run2, mean_real at 1% and 5%; pair run1, run2, sim at 1% and 5%): 96 finite pairs
    required, else the comparison fails (Codex)."""
    import math

    rows, bad = [], []
    probes = sorted(set(a["probes"]) | set(b["probes"]))
    for name in probes:
        ra, rb = a["probes"].get(name, {}).get("stats", {}), b["probes"].get(name, {}).get("stats", {})
        for r in sorted(set(ra) | set(rb)):
            sa, sb = ra.get(r, {}), rb.get(r, {})
            if "gate" not in sa or "gate" not in sb or "pair" not in sa or "pair" not in sb:
                bad.append(f"{name}/{r}: skipped or missing")
                continue
            pairs = [(f"gate {k}", sa["gate"]["catch"].get(k), sb["gate"]["catch"].get(k)) for k in sorted(set(sa["gate"]["catch"]) | set(sb["gate"]["catch"]))]
            pairs += [(f"pair {arm}@{al}", sa["pair"].get(al, {}).get(f"catch_{arm}"), sb["pair"].get(al, {}).get(f"catch_{arm}"))
                      for al in ("0.01", "0.05") for arm in ("run1", "run2", "sim")]
            for what, va, vb in pairs:
                if not (isinstance(va, float) and isinstance(vb, float) and math.isfinite(va) and math.isfinite(vb)):
                    bad.append(f"{name}/{r}/{what}: non-finite or missing")
                    continue
                rows.append({"probe": name, "readout": r, "what": what, "h100": va, "h200": vb, "diff": vb - va})
            rows.append({"probe": name, "readout": r, "what": "pool", "h100": sa["pool"], "h200": sb["pool"]})
    d = [abs(x["diff"]) for x in rows if "diff" in x]
    return {"rows": rows, "problems": bad, "n_catches": len(d), "max_abs_catch_diff": max(d) if d else None}


@app.local_entrypoint()
def compare(h100: str, h200: str, check: str):
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    A, B, ck = (json.loads(Path(p).read_text()) for p in (h100, h200, check))
    fit = json.loads((HERE / "results" / "read_fit_2026-10-14_125514.json").read_text())  # the H100 stage-1 record (committed)
    for n in FIXED:  # Codex: tie each scoring record to its directions
        if A["directions"][n]["sha256"] != fit["directions"][n]["file"]["sha256"] or A["directions"][n]["layer"] != FIXED[n][1]:
            raise SystemExit(f"{n}: the H100 scoring record is not the stage-1 H100 direction at L{FIXED[n][1]}")
        if B["directions"][n]["sha256"] != ck["directions"][n]["file"]["sha256"] or B["directions"][n]["layer"] != FIXED[n][1]:
            raise SystemExit(f"{n}: the H200 scoring record does not use this check's H200 direction")
        if sha(HERE / "results" / H100[n]) != fit["directions"][n]["file"]["sha256"]:
            raise SystemExit(f"{n}: the committed H100 direction file differs from the stage-1 record")
    if A.get("inputs") != B.get("inputs"):
        raise SystemExit("the two scoring records used different input pins")
    r = diff.remote(A, B)
    cos = {n: d["chosen"]["cosine_h100_h200"] for n, d in ck["directions"].items()}
    complete = r["n_catches"] == 96 and not r["problems"] and len(cos) == 2
    ok = complete and min(cos.values()) >= COS_BAR and r["max_abs_catch_diff"] <= CATCH_BAR
    rec = {"label": "RULE.md Amendment 2 comparison", "h100_scores": {"path": Path(h100).name, "sha256": sha(h100)},
           "h200_scores": {"path": Path(h200).name, "sha256": sha(h200)}, "check": {"path": Path(check).name, "sha256": sha(check)}, "cosines": cos,
           "complete (96 finite catches)": complete,
           "reading": "no material effect" if ok else ("INCOMPLETE comparison" if not complete else "MATERIAL: report to 0d before anything else"),
           "bars": {"cosine": COS_BAR, "catch": CATCH_BAR}} | r
    dest = HERE / "results" / f"h200_compare_{Path(h200).stem.split('_', 1)[1]}.json"
    dest.open("x").write(json.dumps(rec, indent=1))
    lines = [f"cosines {cos}; max |catch diff| {r['max_abs_catch_diff']} over {r['n_catches']} catches; problems {r['problems']}; reading: {rec['reading']}"]
    lines += [f"  {x['probe']} {x['readout']} pool {x['h100']} vs {x['h200']}" for x in r["rows"] if x.get("what") == "pool" and x["h100"] != x["h200"]]
    dest.with_suffix(".printout.txt").open("x").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(dest.name)
