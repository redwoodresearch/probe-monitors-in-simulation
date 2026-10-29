"""N2, post hoc batch checks (0d's ruling on the #666 batch threads; not in the plan; free). From the run of record's
stored per-run counts, no new count run:
- (a) real minus simulated on every task with both worlds, using each task's first overlap800 real run (by file and
  sample key), so no base200 run enters; beside it, the registered secondary's key-ordered run 1 (base200 on 95 tasks);
- (b) on the primary's 163 tasks, real run 2 minus simulated, beside run 1 minus simulated.
Task bootstrap 10,000 draws (seed 20261011). Also records the sha256 of the as-run log-path pattern and the script blob.

  uv run --no-project --python 3.14 --with numpy --with boto3 --with pyyaml python experiments/2026-10-11_q2_n2_honest/analyze_n2_batch_POSTHOC.py data/n2_honest/per_run_2026-10-11_112022.jsonl
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from analyze_n2_honest import LOGPATH, MARGIN, SEED, summary  # noqa: E402

AS_RUN = "fb25be88"


def main(per_run: Path):
    runs = [json.loads(line) for line in per_run.open()]
    real, sim = {}, {}
    for r in sorted((r for r in runs if r["n_calls"]), key=lambda r: f"{r['file']}|{r['sample_id']}|{r['epoch']}"):
        real.setdefault(r["task"], []).append(r) if r["world"] == "real" else sim.setdefault(r["task"], r)
    with_sim = sorted(t for t in sim if t in real)
    first_o800 = {t: next((r for r in real[t] if r["batch"] == "overlap800"), None) for t in with_sim}
    clean = [t for t in with_sim if first_o800[t]]
    common = [t for t in with_sim if len(real[t]) >= 2 and real[t][0]["batch"] == real[t][1]["batch"] == "overlap800"]
    blob = subprocess.run(["git", "rev-parse", f"{AS_RUN}:experiments/2026-10-11_q2_n2_honest/analyze_n2_honest.py"],
                          capture_output=True, text=True, check=True).stdout.strip()
    out = {"per-run file sha256": hashlib.sha256(per_run.read_bytes()).hexdigest(), "as-run script": f"{AS_RUN} (blob {blob})",
           "log-path pattern sha256": hashlib.sha256(LOGPATH.pattern.encode()).hexdigest(), "log-path pattern": LOGPATH.pattern,
           "tasks": {"every task with both": len(with_sim), "with an overlap800 real run": len(clean), "primary": len(common),
                     "key-ordered run 1 is base200": sum(real[t][0]["batch"] == "base200" for t in with_sim)}}
    for k, m in enumerate(MARGIN):
        d = lambda tasks, pick: np.array([pick(t)[m] - sim[t][m] for t in tasks], float)
        out[m] = {"(a) first overlap800 real run minus sim, every task with one": summary(d(clean, first_o800.get), SEED + 30 + k),
                  "(a) key-ordered run 1 minus sim, every task (registered secondary)": summary(d(with_sim, lambda t: real[t][0]), SEED + 10 + k),
                  "(b) run 1 minus sim, primary tasks": summary(d(common, lambda t: real[t][0]), SEED),
                  "(b) run 2 minus sim, primary tasks": summary(d(common, lambda t: real[t][1]), SEED + 40 + k)}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "results" / f"n2_batch_POSTHOC_{stamp}.json"
    with open(dest, "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    for m in MARGIN:
        for name, v in out[m].items():
            print(f"{m} {name}: {v['point']:+.4f} [{v['95%'][0]:+.4f}, {v['95%'][1]:+.4f}] ({v['tasks']} tasks)")
    print(json.dumps({k: out[k] for k in ("tasks", "log-path pattern sha256", "as-run script")}))
    print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
