"""Meaning judge: the 58 new unrepaired-simulator replies, through the Oct-6 pipeline unchanged (`run_simulate_pairs.py`
@ cba62df4, read from git into data/, sha256 printed) on the September snapshot. v3 (Codex): this launcher, not the
pipeline, holds the stop. It runs waves of WAVE never-attempted pairs (`--only`); before each wave it persists a
reservation of WAVE x $2.86 (twice the Oct-6 maximum of $1.43 a pair) in data/sim_ledger.json and refuses if spent +
reservation would pass $60, the simulator's share of the $175 job stop. A pending reservation left by a crash counts as
spent. Spent = each written row's cost, an attempt with no usage or a model entry without token counts charged $2.86.
Attempted pairs are never retried. Residual risk, stated: a failed inner model call that leaves no usage at all inside
an otherwise complete attempt is not counted. OPENAI_API_KEY comes from ~/probe-distribution-shift-from-sims/.env into
this process only (never printed); --check-only needs no key and runs everything at once.

  cd <scratch dir with no .env above> && uv run --project ~/simulated-control-evals-sept --no-sync python \\
      <repo>/experiments/2026-10-12_claimB_meaning_judge/run_sim_sites.py [--check-only]
"""

import asyncio
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = "cba62df4:experiments/2026-10-06_matched_step_pairs/run_simulate_pairs.py"
ENV = Path.home() / "probe-distribution-shift-from-sims/.env"
WORST_PAIR, STOP, WAVE = 2.86, 60.0, 4


def put(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


if __name__ == "__main__":
    code = subprocess.run(["git", "show", SRC], capture_output=True, text=True, check=True, cwd=HERE).stdout
    script = HERE / "data" / "run_simulate_pairs_cba62df4.py"
    if not script.exists():
        script.write_text(code)
    assert script.read_text() == code, "the materialized pipeline differs from git"
    print("pipeline", SRC, hashlib.sha256(code.encode()).hexdigest(), flush=True)
    check = "--check-only" in sys.argv
    spec = importlib.util.spec_from_file_location("run_simulate_pairs", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # defines only; its __main__ block does not run
    priced = mod.cost

    def cost(usage):  # fail closed: no usage, or a model entry without token counts, is charged the worst case
        partial = not usage or any(not isinstance(u.get("input_tokens"), int) or not isinstance(u.get("output_tokens"), int) for u in usage.values())
        return priced(usage) if check else max(priced(usage) if usage else 0.0, WORST_PAIR if partial else 0.0)
    mod.cost = cost
    first_log = mod.log_judges

    def log_once():  # the pipeline wraps the judge models each time main() runs; wrap them only once across waves
        first_log()
        mod.log_judges = lambda: None
    mod.log_judges = log_once
    pairs = [json.loads(x)["pair"] for x in (HERE / "data/sim_recs.jsonl").read_text().splitlines()]
    base = [str(script), "--pairs", str(HERE / "data/sim_recs.jsonl")]
    if check:
        sys.argv = base + ["--out", str(HERE / "data/sim_check"), "--check-only", "--concurrency", "8"]
        asyncio.run(mod.main())
        raise SystemExit(0)
    os.environ["OPENAI_API_KEY"] = next(line.split("=", 1)[1].strip().strip('"').strip("'") for line in ENV.read_text().splitlines()
                                        if line.split("=", 1)[0].strip() == "OPENAI_API_KEY")
    out, ledger = HERE / "data/sim_new.jsonl", HERE / "data/sim_ledger.json"
    assert ledger.exists() or not out.exists(), "simulator rows exist but no ledger: reconcile before any new wave"
    led = json.loads(ledger.read_text()) if ledger.exists() else {"pending": 0.0}
    while True:
        rows = [json.loads(x) for x in out.read_text().splitlines()] if out.exists() else []
        spent = sum(r.get("cost_usd", WORST_PAIR) for r in rows) + led["pending"]  # a crash's reservation stays spent
        attempted = {r["pair"] for r in rows}
        todo = [p for p in pairs if p not in attempted][:WAVE]
        if not todo:
            print(f"done: {len(rows)} attempts, spent ${spent:.2f}", flush=True)
            break
        reserve = len(todo) * WORST_PAIR
        if spent + reserve > STOP:
            print(f"stop: spent ${spent:.2f} + reservation ${reserve:.2f} would pass ${STOP}", flush=True)
            break
        led = {"spent_before_wave": spent, "pending": led["pending"] + reserve}
        put(ledger, led)
        sys.argv = base + ["--out", str(HERE / "data/sim_new"), "--only", ",".join(map(str, todo)), "--concurrency", str(WAVE), "--budget-usd", str(STOP)]
        try:
            asyncio.run(mod.main())
        except SystemExit:  # the pipeline exits 1 when a pair failed; failures are recorded and never retried
            pass
        led = {"spent_before_wave": spent, "pending": led["pending"] - reserve}  # the wave's rows now carry its cost
        put(ledger, led)
