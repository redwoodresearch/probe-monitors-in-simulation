"""Honest main-task success of the on-policy Qwen agent in a phase-2 run directory (local, read-only), for figure 1:
every .eval under honest_a/ and honest_b/ (one per task), each sample's linux_scorer main_task_success (C, I, or N =
unscored), pooled over both arms with N excluded. Writes results/honest_success_<model tag>_<stamp>.json in
plot_figure1.py's input format. .eval files are zstd zips: run with Python 3.14.

  uv run --no-project --python 3.14 python experiments/2026-10-08_q2_figure_pack/analyze_honest_rerun.py <run dir> <model name>
"""
import json
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent


def main(root: Path, model: str):
    launch = json.loads((root / "launch.json").read_text())
    if not launch["policy"].endswith(model):
        raise ValueError(f"launch.json's policy is {launch['policy']}, not {model}")
    rows = []
    for arm in ("honest_a", "honest_b"):
        for f in sorted(root.glob(f"{arm}/*/*.eval")):
            with zipfile.ZipFile(f) as z:
                samples = [n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")]
                if len(samples) != 1:
                    raise ValueError(f"{f}: {len(samples)} samples")
                s = json.loads(z.read(samples[0]))
                sc = s["scores"]["linux_scorer"]
                v = sc["value"]["main_task_success"]
                peak = max((e["output"]["usage"]["input_tokens"] + e["output"]["usage"]["output_tokens"] for e in s["events"]
                            if e.get("event") == "model" and (e.get("output") or {}).get("usage")), default=None)  # largest context of one call
                rows.append({"arm": arm, "task_dir": f.parent.name, "sample_id": s["id"], "file": f.name, "main_task_success": v,
                             "peak_context_tokens": peak}
                            | ({} if v in ("C", "I") else {"explanation": str(sc.get("explanation"))[:200]}))  # why it went unscored
    per_arm = {a: dict(Counter(r["main_task_success"] for r in rows if r["arm"] == a)) for a in ("honest_a", "honest_b")}
    scored = [r for r in rows if r["main_task_success"] in ("C", "I")]
    k = sum(r["main_task_success"] == "C" for r in scored)
    unscored = [r for r in rows if r not in scored]
    note = (", ".join(f"{a} {c.get('C', 0)}/{c.get('C', 0) + c.get('I', 0)}" for a, c in per_arm.items())
            + f"; {len(unscored)} unscored (N) excluded")
    over = {w: sum((r["peak_context_tokens"] or 0) > 32768 for r in rows if w == "all" or r["main_task_success"] == "C") for w in ("all", "C")}
    out = {model.split("/")[-1]: {"successes": k, "runs": len(scored), "note": note, "per_arm": per_arm, "unscored": unscored,
                                  "runs over 32,768 tokens (all, successful)": [over["all"], over["C"]],
                   "source": f"{root.name} honest_a+honest_b (policy {launch['policy']}, probe_shift {launch['probe_shift_commit']})",
                   "rows": rows}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"honest_success_{model.split('/')[-1].lower()}_{stamp}.json"
    path.write_text(json.dumps(out) + "\n")
    print(json.dumps({m: {x: v[x] for x in ("successes", "runs", "note", "per_arm", "runs over 32,768 tokens (all, successful)", "source")} for m, v in out.items()}, indent=1))
    for r in unscored:
        print("unscored:", r["arm"], r["task_dir"], r["sample_id"], repr(r["explanation"]))
    print(path)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
