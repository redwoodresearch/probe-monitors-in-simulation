"""Cell (a) transfer, item 4 (RULE.md; POST HOC): q2's frozen Sol-trained probes at every layer q2 stored (16, 24, 32, 40,
48, 56, 63) on Qwen3.5's own runs, per variant. Own output_end rows are read from the replay dumps (all 64 layers) with the
caches' rule (counted epochs: no redrawn or rerun attempts; output_end calls with a row), one Modal CPU container per
variant. Run score = mean over its outputs, each scored by q2's outer model of its task's q2 fold.
Gate: at L63, the rebuilt run scores must equal cell (a)'s stored run scores for the variant (checked locally by the
analysis, 1e-6).

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run \\
      experiments/2026-10-13_cell_a_transfer_POSTHOC/run_xfer_layers.py --probes-dir <dir with probes_L*.npz> --fold-of-task <cell_a json>
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-onpol-reply_xferlayers_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"
LAYERS = (16, 24, 32, 40, 48, 56, 63)
SHA_L63 = "6cf38abe14c5f3badc1d"
D0 = ["2026-10-09_rerun_strict_segends_v1", "2026-10-10_scaleup_canary_segends_v1", "2026-10-10_scaleup_l2_segends_v1", "2026-10-10_scaleup_topup_segends_v1"]
R2 = "2026-10-10_second_attack_segends_v1"
SRC = {"unswapped_L0": {"run1": [(d, "attack") for d in D0], "run2": [(R2, "attack")], "sim": [(d, "sim_attack@september") for d in D0],
                        "honest_a": [(d, "honest_a") for d in D0], "honest_b": [(d, "honest_b") for d in D0]},
       "unswapped_L1": {"run1": [(d + "_scrub1", "attack") for d in D0], "run2": [(R2 + "_scrub1", "attack")], "sim": [(d + "_scrub1", "sim_attack@september") for d in D0],
                        "honest_a": [(d + "_scrub1", "honest_a") for d in D0], "honest_b": [(d + "_scrub1", "honest_b") for d in D0]},
       "swapped_L1": {"run1": [(d + "_scrub1_swapsys", "attack") for d in D0], "run2": [(R2 + "_scrub1_swapsys", "attack")],
                      "sim": [(d + "_scrub1_swapsys2", "sim_attack@september") for d in D0],
                      "honest_a": [(d + "_scrub1", "honest_a") for d in D0], "honest_b": [(d + "_scrub1", "honest_b") for d in D0]}}
app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2).read_only()
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")


@app.function(image=image, cpu=8, memory=65536, timeout=3600, retries=0, volumes={"/acts": acts})
def score(variant: str, probes: dict, fold_of_task: dict) -> dict:
    import numpy as np

    rows = {}  # (dump, chunk) -> list of (key, row)
    outputs = {}
    for world, sources in SRC[variant].items():
        for dump, arm in sources:
            for fm in sorted((Path("/acts") / dump / "meta" / arm).glob("*.json")):
                m = json.loads(fm.read_text())
                if m.get("redrawn_because") or m.get("attempt_kind") == "rerun":
                    continue
                cs = [c for c in m["calls"] if c.get("kind") == "output_end" and c.get("row") is not None]
                k = f"{world}|{m['task']}"
                for c in cs:
                    rows.setdefault((dump, m["chunk"]), []).append((k, m["row_start"] + c["row"]))
                outputs[k] = outputs.get(k, 0) + len(cs)
    out = {"variant": variant, "outputs": outputs, "run_scores": {}}
    for L in LAYERS:
        P = probes[L]
        tot = {}
        for (dump, chunk), items in rows.items():
            A = np.load(Path("/acts") / dump / "by_layer" / f"L{L}" / f"{chunk}.npy", mmap_mode="r")
            for f in range(5):
                sel = [(k, r) for k, r in items if fold_of_task[k.split("|", 1)[1]] == f]
                if not sel:
                    continue
                X = np.asarray(A[[r for _, r in sel]], np.float64)
                s = ((X - P[f"output_end__task_outer_{f}__mean"]) / P[f"output_end__task_outer_{f}__std"]) @ P[f"output_end__task_outer_{f}__w"] + float(P[f"output_end__task_outer_{f}__b"])
                for (k, _), v in zip(sel, s):
                    tot[k] = tot.get(k, 0.0) + float(v)
        out["run_scores"][str(L)] = {k: tot[k] / outputs[k] for k in sorted(tot)}
    return out


@app.local_entrypoint()
def main(probes_dir: str, fold_of_task: str):
    import numpy as np

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=HERE).stdout:
        raise SystemExit("uncommitted tracked changes")
    paths = {L: Path(probes_dir) / f"probes_L{L}.npz" for L in LAYERS}
    shas = {L: hashlib.sha256(p.read_bytes()).hexdigest() for L, p in paths.items()}
    if not shas[63].startswith(SHA_L63):
        raise SystemExit("probes_L63 is not cell (a)'s pinned file")
    probes = {L: {k: np.asarray(v, np.float64) for k, v in np.load(p).items() if k.startswith("output_end__task_outer_")} for L, p in paths.items()}
    fot = json.loads(Path(fold_of_task).read_text())["fold_of_task"]
    res = list(score.starmap([(v, probes, fot) for v in SRC]))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "results" / f"xfer_layers_POSTHOC_{stamp}.json"
    dest.open("x").write(json.dumps({"label": "POST HOC (RULE.md item 4); run scores only", "git": git, "probes_sha256": shas, "q2_run": "2026-10-12_011944",
                                     "variants": {r["variant"]: r for r in res}}))
    print({r["variant"]: {L: len(s) for L, s in r["run_scores"].items()} for r in res})
    print(dest.name)
