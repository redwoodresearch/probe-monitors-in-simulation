"""Row 2, step 2 supplement (RULE.md here): Llama-3.3-70B's retained-cell masks, now that row 3's planted e2 dump exists. It
takes the committed step-2(d) record (q2_step2d.py's output, pinned by manifest part 2), lists the planted metas of row 3's
planted run (one Modal CPU container: kept output_end counts only, no activations), and finalizes the five Llama
configurations exactly as q2_step2d.py does: a retained cell's real e2 run has four finite own-fold nested scores (already
in the record), and, if the plant changed it, its planted run has a meta with at least one kept output_end. Thresholds,
false-alarm and convergence outcomes are carried over unchanged. Everything else in the record is copied, so q2_x.py and
q2_manifest_part2.py read the output as they read the original. Reads no score and no simulated-arm file.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run experiments/2026-10-12_plantslip_heldout/q2_step2d_llama.py \\
      --step2d <results/q2_step2d_2026-10-12_200953.json> --stage <staging record> --real-only <real_only dir>
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-heldout_llamaavail_kill-2026-11-12"
ACTS, RUN, ARM = "probe-shift_aswin-surya_sub-q2-llama_acts_kill-2026-11-12", "2026-10-12_q2_plantslip_reply_planted-e2_llama33-70b_v1", "real-attack-e2-planted"
STAGE_SHA = "f1a68c90a5d2"

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12")
acts = modal.Volume.from_name(ACTS, version=2).read_only()


def filename(key: str) -> str:
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


@app.function(image=image, volumes={"/acts": acts}, cpu=2, memory=4096, timeout=1800, retries=0)
def planted_kept(keys: list[str]) -> dict:
    try:
        d = Path("/acts") / RUN / "meta" / ARM
        out = {}
        for k in keys:
            p = d / f"{filename(k)}.json"
            out[k] = sum(q["kind"] == "output_end" and q["kept"] for q in json.loads(p.read_text())["positions"]) if p.exists() else -1
        return {"kept": out, "summary exists": (Path("/acts") / RUN / "stats" / "summary.json").exists()}
    except Exception as e:  # noqa: BLE001 — sanitized and re-raised, never swallowed
        raise RuntimeError(f"listing failed: {type(e).__name__}") from None


@app.local_entrypoint()
def main(step2d: str, stage: str, real_only: str):
    for f in (Path(__file__).resolve(), HERE / "RULE.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    man = sorted((HERE / "results").glob("manifest_part2_*.json"))
    pinned = {json.loads(m.read_text())["frozen thresholds and masks"]["sha256"] for m in man}
    if sha(step2d) not in pinned or not sha(stage).startswith(STAGE_SHA):
        raise SystemExit("not the step-2(d) record manifest part 2 pins, or not the pinned staging record")
    s2 = json.loads(Path(step2d).read_text())
    ro = Path(real_only); rrec = json.loads((ro / "real_only_record.json").read_text())["readers"]["llama"]
    if s2["inputs sha256"]["real_only_record.json"] != sha(ro / "real_only_record.json") or sha(ro / "llama_runs.json") != rrec["runs file"]["sha256"]:
        raise SystemExit("not the real-only files step 2 used")
    runs = json.loads((ro / "llama_runs.json").read_text())
    if any(r["arm"] == "sim-attack" for r in runs):
        raise SystemExit("a sim-attack run in the real-only runs")
    key_of = {r["cell"]: r["key"] for r in runs if r["arm"] == "real-attack-e2"}
    changed = {x["key"] for x in json.loads(Path(stage).read_text())["planted"] if x["arm"] == ARM}
    res = planted_kept.remote(sorted(changed))
    if not res["summary exists"]:
        raise SystemExit("row 3's planted extraction has no stats/summary.json yet")
    kept = res["kept"]
    out = dict(s2) | {"git": head, "supplements": {"file": Path(step2d).name, "sha256": sha(step2d)}, "configurations": dict(s2["configurations"])}
    out["inputs sha256"] = dict(s2["inputs sha256"]) | {Path(step2d).name: sha(step2d)}
    for key, c in s2["configurations"].items():
        if not key.startswith("llama"):
            continue
        if c["masks final"] or c["removed, by reason"].get("pending", 0) != len(changed):
            raise SystemExit(f"{key}: not the pending configuration this supplement finalizes")
        retained, why = [], {"real e2 scores": c["removed, by reason"]["real e2 scores"], "planted extraction": 0, "pending": 0}
        for cell in c["retained cell ids"]:
            if key_of[cell] in changed and kept.get(key_of[cell], -1) < 1:
                why["planted extraction"] += 1; continue
            retained.append(cell)
        out["configurations"][key] = dict(c) | {"retained cells": len(retained), "removed, by reason": why, "retained cell ids": retained, "masks final": True,
                                                "eligible": bool(c["false-alarm gate passes"]) and bool(c["convergence gate passes"])}
    out["planted availability (kept output_end per changed e2 run)"] = dict(s2["planted availability (kept output_end per changed e2 run)"]) | {
        "llama": {"runs": len(kept), "no meta": sum(x < 0 for x in kept.values()), "no kept row": sum(x == 0 for x in kept.values()), "dump": f"{ACTS}:{RUN}"}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"q2_step2d_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    for k, v in out["configurations"].items():
        if k.startswith("llama"):
            print(k, "| retained", v["retained cells"], v["removed, by reason"], "| masks final", v["masks final"], "| eligible", v["eligible"])
    print(dest)
