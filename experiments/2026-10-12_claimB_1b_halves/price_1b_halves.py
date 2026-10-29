"""Price, $0: Llama-3.2-1B (the small-model baseline) through the halves test (Aswin about 12:00 PT 10-12, via 0d).
Reads the billed costs it scales from committed SPEND rows by git show, prints the inputs and the price table, and
writes results/price_<stamp>.json. Nothing launches.

  uv run --no-project --python 3.12 python experiments/2026-10-12_claimB_1b_halves/price_1b_halves.py
"""
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent


def spend(ref, key, pattern):
    rows = [ln for ln in subprocess.run(["git", "show", f"{ref}:SPEND.md"], capture_output=True, text=True, check=True, cwd=HERE).stdout.splitlines()
            if ln.startswith("|") and key in ln]
    if len(rows) != 1:
        raise SystemExit(f"{ref}: expected one SPEND row containing {key!r}, found {len(rows)}")
    m = re.search(pattern, rows[0])
    if not m:
        raise SystemExit(f"{ref}: no match for {pattern!r} in the {key!r} row")
    return float(m.group(1))


# billed costs scaled from (Modal billing by app ID, as recorded in each SPEND row)
llama_extract = spend("origin/main", "Small-model baseline, the full run (Llama-3.2-1B-Instruct", r"extraction ap-Aj4lTEi3sh3lkvBlO4BZEc \$([0-9.]+)")
llama_pack = spend("origin/main", "Small-model baseline, the full run (Llama-3.2-1B-Instruct", r"pack \$([0-9.]+)")
halves_refit = spend("origin/aswin/claimb-halves", "Item (A), probes after all four fixes", r"\*\*\$([0-9.]+)\*\* billed")
halves_e2 = spend("origin/aswin/claimb-halves", "Item (A) POSTHOC, the direct yardstick", r"\*\*\$([0-9.]+)\*\* billed")

LLAMA_TRANSCRIPTS = 3117  # the full run's staged transcripts (ACTIVATIONS.md, the llama32-1b dump row)
# evened (changed) runs to re-read, from the Qwen3.5 halves PLAN.md @ 4c232252 lines 16-19 (cells.json of H3 and B-dev):
CHANGED = {"test (sim 365, e1 166, e2 181, plant-2 119)": 831, "dev (sim 373, e1 229, e2 250)": 852, "dev plant-2 copies": 100}
GATE = 4  # unchanged runs re-read and required to equal the stored Llama rows bitwise
FIXED = (0.3, 0.8)  # start-up, separate gate container, Volume loads and the merge pack not captured by per-read scaling (Codex); a range, not a measurement
LENGTH = (1.0, 1.3)  # changed runs may be longer than the average transcript (more outputs to change); a range, not a measurement

reads = sum(CHANGED.values()) + GATE
per_read = llama_extract / LLAMA_TRANSCRIPTS
extract = [reads * per_read * f for f in LENGTH]
refit = [halves_refit, halves_refit + halves_e2]  # low: e2 and plant copies scored in the refit's own pass; high: as separately billed for Qwen
raw = [extract[0] + llama_pack + refit[0] + FIXED[0], extract[1] + llama_pack + refit[1] + FIXED[1]]
out = {"inputs": {"fixed_overhead": FIXED, "llama_extraction_billed": llama_extract, "llama_pack_billed": llama_pack, "halves_refit_billed": halves_refit,
                  "halves_direct_e2_billed": halves_e2, "llama_transcripts": LLAMA_TRANSCRIPTS, "changed_runs": CHANGED, "gate_runs": GATE,
                  "length_factor": LENGTH},
       "reads": reads, "per_read": per_read, "extraction": extract, "pack": llama_pack, "refit_e2_plants": refit, "staging": 0.0,
       "raw": raw, "with_30pct": [1.3 * x for x in raw],
       "fixed_overhead": FIXED,
       "variant_unevened": {"note": "Llama on its stored unevened level-1 rows, plus a read of the 100 dev plant-2 copies (stored Llama plants cover test only)",
                            "raw": [100 * per_read * LENGTH[0] + refit[0] + FIXED[0], 100 * per_read * LENGTH[1] + refit[1] + FIXED[1]],
                            "with_30pct": [1.3 * (100 * per_read * LENGTH[0] + refit[0] + FIXED[0]), 1.3 * (100 * per_read * LENGTH[1] + refit[1] + FIXED[1])]},
       "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
       "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
with open(HERE / "results" / f"price_{stamp}.json", "x") as fh:
    fh.write(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: v for k, v in out.items() if k not in ("git", "script_sha256")}, indent=1))
