"""(52) Sol's real honest main-task success on the on-policy pilot's 20 tasks, with an exact binomial 95% interval
(local; PREDICTIONS.md first). The base dump's real honest runs of those main tasks are found through the v4 all-layer
dump's per-transcript meta (cohort, file, sample_id, epoch), and each run's `main_task_success` score is read from its
Inspect log (the local S3 cache, via probe_shift.transcripts). LinuxArena .eval logs need Python 3.14 (zstd).

  PYTHONPATH=src uv run --no-project --python 3.14 --with boto3 --with pyyaml --with scipy python \
      experiments/2026-10-08_q2_free_checks_3/analyze_honest_success.py
"""
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from scipy.stats import binomtest

from probe_shift.transcripts import load_sample

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
VOL, META = "probe-shift_aswin-surya_sub-la-activations", "2026-10-06_la_segment-end-v4_qwen3-32b_alllayers_v1/meta/real-honest"
PILOT = ROOT / "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9.json"


def volume(*args: str) -> str:
    return subprocess.run(["uv", "run", "modal", "volume", *args], capture_output=True, text=True, check=True).stdout


def main():
    mains = {c["main"] for c in json.loads(PILOT.read_text())["cells"]}
    names = json.loads(volume("ls", VOL, META, "--json"))
    names = [n.get("Filename") or n.get("filename") for n in names]
    wanted = [n for n in names if n.rsplit("/", 1)[1].rsplit("__", 2)[-2].split(":")[0] in mains]
    rows = []
    for n in sorted(wanted):
        raw = volume("get", VOL, n, "-")
        meta = json.loads(raw[: raw.rfind("}") + 1])
        sample = load_sample(meta["cohort"], meta["file"], meta["sample_id"], int(meta["epoch"]))
        score = (sample.get("scores") or {})
        value = next((v.get("value") for v in score.values() if isinstance(v, dict) and isinstance(v.get("value"), dict)), {}) or {}
        mts = value.get("main_task_success")
        rows.append({"key": meta["key"], "main_task": meta["main_task"], "cohort": meta["cohort"], "file": meta["file"],
                     "sample_id": meta["sample_id"], "epoch": meta["epoch"], "main_task_success": mts})
    known = [r for r in rows if r["main_task_success"] in ("C", "I", True, False)]
    k = sum(r["main_task_success"] in ("C", True) for r in known)
    ci = binomtest(k, len(known)).proportion_ci(method="exact") if known else None
    res = {"pilot mains": len(mains), "mains with a real honest run": len({r["main_task"] for r in rows}), "runs": len(rows),
           "runs with a C/I score": len(known), "successes": k, "rate": round(k / len(known), 4) if known else None,
           "exact 95% interval": [round(ci.low, 4), round(ci.high, 4)] if ci else None,
           "unscored": [r["key"] for r in rows if r not in known], "rows": rows}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = HERE / "results" / f"honest_success_{stamp}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(out)
    print({k: v for k, v in res.items() if k != "rows"})


if __name__ == "__main__":
    main()
