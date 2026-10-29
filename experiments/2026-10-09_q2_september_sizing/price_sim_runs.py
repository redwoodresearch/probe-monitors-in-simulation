"""Price each simulated run of the strict rerun from its eval log (local, free): every model event that is not the acting
model (Qwen3.5-27B), priced with simulated-control-evals' sim.prices table (input, output, cache read and write). The
four simulated arms' total must equal the launch's recorded simulator spend ($1,222.79, launch.json "spend"), or the
script stops. Writes results/sim_run_costs_<stamp>.json: per arm, per task, the run's cost.

  PYTHONPATH=<simulated-control-evals>/src uv run --no-project --python 3.14 --with attrs python \
      experiments/2026-10-09_q2_september_sizing/price_sim_runs.py <rerun folder>
"""
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from sim.prices import load_prices

HERE = Path(__file__).parent
ACTING = "vllm/Qwen/Qwen3.5-27B"
ARMS = ("sim_honest@september", "sim_attack@september", "sim_honest@today", "sim_attack@today")


def main(run_dir: Path):
    prices, out = load_prices(), {}
    for arm in ARMS:
        for f in sorted((run_dir / arm).glob("*/*.eval")):
            with zipfile.ZipFile(f) as z:
                (name,) = [x for x in z.namelist() if x.startswith("samples/") and x.endswith(".json")]
                s = json.loads(z.read(name))
            usd = 0.0
            for e in s.get("events") or []:
                if e.get("event") != "model" or e.get("model") == ACTING:
                    continue
                u = (e.get("output") or {}).get("usage") or {}
                usd += prices[e["model"]].usd(input_tokens=u.get("input_tokens", 0), output_tokens=u.get("output_tokens", 0),
                                              cache_read_tokens=u.get("input_tokens_cache_read") or 0,
                                              cache_write_tokens=u.get("input_tokens_cache_write") or 0)
            task = f.parent.name
            if task in out.setdefault(arm, {}):
                raise ValueError(f"{arm} {task}: two runs")
            out[arm][task] = round(usd, 4)
    total = sum(sum(v.values()) for v in out.values())
    recorded = json.loads((run_dir / "launch.json").read_text())["spend"][-1]["actual_usd"]
    if abs(total - recorded) > 0.01:
        raise ValueError(f"priced ${total:.2f}, launch recorded ${recorded:.2f}")
    res = {"run": run_dir.name, "recorded simulator spend": recorded, "priced total": round(total, 2), "per run": out,
           "per arm": {a: {"runs": len(v), "total": round(sum(v.values()), 2), "mean": round(sum(v.values()) / len(v), 2)} for a, v in out.items()}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"sim_run_costs_{stamp}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "per run"}, indent=1))
    print(path)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
