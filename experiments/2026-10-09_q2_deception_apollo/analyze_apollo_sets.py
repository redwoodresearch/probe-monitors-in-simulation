"""Step 0 of the deception tests (local, free): build Apollo's insider-trading (report, double-down), sandbagging and
Alpaca-control sets with src/probe_shift/apollo.py and check them against Apollo's shipped results for its Llama-3.3-70B
instructed-pairs probe (example_results/instructed_pairs: scores.json, control_scores.json). What it shows: the label
sequence after Apollo's seed-42 shuffle equals the shipped labels item for item (Alpaca's shipped file has no labels,
so there only the count), which a different cohort or shuffle would almost surely break; swaps among dialogues with
the same label would still pass. What it does not: dialogue text identity and the
token-level detect mask (Apollo also reads its end-of-turn token, and four header tokens on untrimmed messages); a
rough span check is the correlation of our detected characters with Apollo's detected token counts, and exact mask
parity is step 1's job (Llama tokenizer, Apollo's padding, their shipped scores). Also the token budget for step 2
(characters / 4). Exits non-zero if a label sequence or count differs.

  uv run --no-project --with numpy python experiments/2026-10-09_q2_deception_apollo/analyze_apollo_sets.py <deception-detection repo>
"""
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from probe_shift import apollo

repo = Path(sys.argv[1])
shipped = json.loads((repo / "example_results" / "instructed_pairs" / "scores.json").read_text())
control = json.loads((repo / "example_results" / "instructed_pairs" / "control_scores.json").read_text())["alpaca__plain"]
sets = {"insider_trading__upscale": apollo.insider_trading(repo),
        "insider_trading_doubledown__upscale": apollo.insider_trading(repo, double_down=True),
        "sandbagging_v2__wmdp_mmlu": apollo.from_rollouts(repo, "sandbagging_v2__wmdp_mmlu__llama-70b-3.3.json", "sandbagging", "<step4>"),
        "alpaca__plain": apollo.from_rollouts(repo, "alpaca__plain__llama-70b-3.3.json", "alpaca", None)}
out = {"repo commit": subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(),
       "scores.json sha256": hashlib.sha256((repo / "example_results" / "instructed_pairs" / "scores.json").read_bytes()).hexdigest(), "sets": {}}
for name, ds in sets.items():
    ref = control if name == "alpaca__plain" else shipped[name]
    ours = [d["apollo_label"] for d in ds]
    theirs = ref.get("labels", ["HONEST"] * len(ref["scores"]))
    chars = np.array([d["detect"][1] - d["detect"][0] for d in ds])
    toks = np.array([len(s) for s in ref["scores"]])
    ok = chars > 0
    out["sets"][name] = {"dialogues": len(ds), "shipped": len(theirs), "labels": dict(Counter(ours)),
                         "label sequence equals the shipped one": ours == theirs,
                         "first mismatch": next((i for i, (a, b) in enumerate(zip(ours, theirs)) if a != b), None),
                         "skipped for a missing separator": int(sum(d["apollo_label"] == "SKIP" and d["detect"][0] == 0 for d in ds)) if name != "alpaca__plain" else 0,
                         "detected characters vs shipped token count, correlation": round(float(np.corrcoef(chars[ok], toks[ok])[0, 1]), 4) if len(toks) == len(chars) else None,
                         "shipped tokens per detected character (median)": round(float(np.median(toks[ok] / chars[ok])), 3) if len(toks) == len(chars) else None,
                         "dialogues with merged same-role messages": int(sum(d["merged_messages"] > 0 for d in ds)),
                         "tokens for a full forward pass, characters / 4 (millions)": round(sum(len(m["content"]) for d in ds for m in d["messages"]) / 4e6, 2)}
print(json.dumps(out, indent=1))
bad = [n for n, v in out["sets"].items() if not v["label sequence equals the shipped one"] or v["dialogues"] != v["shipped"]]
if bad:
    raise SystemExit(f"label sequence or count differs from Apollo's: {bad}")
