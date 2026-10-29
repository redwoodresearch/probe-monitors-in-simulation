"""Draw the attack check's 5 cells and freeze what each attack run must be told: Sol's September attack approach and the
sha256 of Sol's system prompt and first user message. Laptop, no spend (Sol's samples come from the S3 cache).

  uv run python experiments/2026-10-07_phase2_attack_check/run_select_cells.py

One cell per environment, seed 0, model_registry always in (it needs the rebuilt MinIO). Eligible: a Phase 1 paired
attack cell (splits/exclusions_v4.json), Sol's epoch-1 side task passed (this is a capability check: can Qwen3.5 attack
where the task is known to be doable), September wrote an approach for it, and its main task is one of the 437 paired
honest tasks (so honest real vs simulated is available later). Writes config/cells.json (refuses to overwrite).
"""

import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from probe_shift.transcripts import eval_files, file_samples

HERE = Path(__file__).parent
SPLITS = HERE.parents[1] / "splits" / "exclusions_v4.json"
E1, E2 = "la_public10_real_attack_n1000", "la_public10_real_attack_reroll_n1000"
FORCED_ENV, N_OTHER_ENVS, SEED = "model_registry", 4, 0


def text(content) -> str:
    return content if isinstance(content, str) else "".join(p.get("text", "") for p in content)


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def rows(alias: str) -> dict[str, list[dict]]:
    """Every sample of a cohort by cell. Epoch 2 holds one cell twice (1,001 samples), so a cell maps to a list."""
    with ThreadPoolExecutor(32) as pool:
        files = eval_files(alias)
        per_file = pool.map(lambda f: (f, file_samples(alias, f)), files)
    out = {}
    for file, samples in per_file:
        for s in samples:
            md = s["metadata"]
            cell = f'{md["main_task"]["id"]}:{md["side_task"]["id"]}'
            out.setdefault(cell, []).append({"alias": alias, "file": file, "id": s["id"], "epoch": s["epoch"], "env": md["environment"]["id"],
                         # None when the sample was never scored (an errored sample)
                         "side_success": ((s.get("scores") or {}).get("linux_scorer") or {}).get("value", {}).get(
                             "side_task_success"),
                         "approach": md.get("attack_approach", ""),
                         "system_sha256": sha(text(s["messages"][0]["content"])), "input_sha256": sha(text(s["input"])),
                         "tool_calls": sum(len(m.get("tool_calls") or []) for m in s["messages"] if m["role"] == "assistant")})
    return out


def main() -> None:
    out = HERE / "config" / "cells.json"
    assert not out.exists(), f"{out} exists; the draw is frozen"
    ex = json.loads(SPLITS.read_text())
    paired, honest = set(ex["paired_attack_cells"]["cells"]), set(ex["paired_honest_cells"]["cells"])
    e1, e2 = rows(E1), rows(E2)
    assert all(len(v) == 1 for v in e1.values()), "epoch 1 has a cell twice"
    e1 = {c: v[0] for c, v in e1.items()}
    eligible = sorted(c for c, r in e1.items() if c in paired and r["side_success"] == "C" and r["approach"].strip()
                      and c.split(":")[0] in honest)
    rng = random.Random(SEED)
    others = sorted({e1[c]["env"] for c in eligible} - {FORCED_ENV})
    envs = [FORCED_ENV, *sorted(rng.sample(others, N_OTHER_ENVS))]
    cells = []
    for env in envs:
        pool = [c for c in eligible if e1[c]["env"] == env]
        cell = pool[rng.randrange(len(pool))]
        main, side = cell.split(":")
        cells.append({"env": env, "cell": cell, "main": main, "side": side, "approach": e1[cell]["approach"],
                      "sol_system_sha256": e1[cell]["system_sha256"], "sol_input_sha256": e1[cell]["input_sha256"],
                      "sol_e1": {k: e1[cell][k] for k in ("alias", "file", "id", "epoch", "side_success", "tool_calls")},
                      "sol_e2": [{k: r[k] for k in ("alias", "file", "id", "epoch", "side_success", "tool_calls")}
                                 for r in e2.get(cell, [])]})
    out.write_text(json.dumps({"seed": SEED, "rule": __doc__.split("\n\n")[1], "n_eligible": len(eligible),
                               "eligible_by_env": {e: sum(e1[c]["env"] == e for c in eligible) for e in sorted(others + [FORCED_ENV])},
                               "cells": cells}, indent=1))
    for c in cells:
        print(c["env"], c["cell"], c["sol_e1"]["side_success"], [r["side_success"] for r in c["sol_e2"]])
    print(f"-> {out}")


if __name__ == "__main__":
    main()
