"""POST HOC (added after step 2's results, at review; not in the pre-results analysis): is step 2's sandbagging AUROC
only the question domain? Apollo's sandbagging set puts most deceptive dialogues on WMDP questions (bio, chem, cyber)
and most honest ones on MMLU questions, so a probe could score high by telling the domains apart. Free: CPU on the saved
per-dialogue scores, no SPEND row.

At the frozen step-2 probe (block_output:31), per domain: AUROC honest vs deceptive with a 95% bootstrap interval
(2,000 resamples, seed 0) and a within-domain label-shuffle null (1,000 permutations, seed 0; p = (1 + count >=
observed) / 1,001), and the probe score's AUROC with the domain itself as the label, for comparison. A dialogue's
domain is its rollout's metadata topic in Apollo's file.

  uv run --no-project --with numpy --with scikit-learn python experiments/2026-10-09_q2_deception_apollo/analyze_sandbagging_domains.py \
      <step2 results json> <deception-detection repo>
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

res_p, repo = Path(sys.argv[1]), Path(sys.argv[2])
roll_p = repo / "data" / "rollouts" / "sandbagging_v2__wmdp_mmlu__llama-70b-3.3.json"
d = json.loads(res_p.read_text())
topic = [r["metadata"]["topic"] for r in json.loads(roll_p.read_text())["rollouts"] for _ in r["output_strs"]]
es = d["eval_sets"]["sandbagging"]
dom = np.array(["WMDP" if topic[int(i.split("_")[1])] in ("bio", "chem", "cyber") else "MMLU" for i in es["ids"]])  # id = the rollout's index
y = np.array(es["labels"], float)
s = np.array(d["scores"]["sandbagging"]["block_output:31"], float)
rng = np.random.default_rng(0)
out = {"post hoc": True, "inputs sha256": {res_p.name: hashlib.sha256(res_p.read_bytes()).hexdigest(), roll_p.name: hashlib.sha256(roll_p.read_bytes()).hexdigest()},
       "probe": "block_output:31", "per domain": {}}
for name, m in (("all", np.ones(len(y), bool)), ("WMDP only", dom == "WMDP"), ("MMLU only", dom == "MMLU")):
    yy, ss = y[m].astype(int), s[m]
    a = float(roc_auc_score(yy, ss))
    null = [roc_auc_score(rng.permutation(yy), ss) for _ in range(1000)]
    boot = [roc_auc_score(yy[i], ss[i]) for i in (rng.integers(0, len(yy), len(yy)) for _ in range(2000)) if len(set(yy[i])) == 2]
    out["per domain"][name] = {"deceptive / honest": [int(yy.sum()), int((yy == 0).sum())], "AUROC": round(a, 4),
                               "95% bootstrap": [round(float(q), 4) for q in np.percentile(boot, [2.5, 97.5])],
                               "label-shuffle null: p, 95th percentile": [round((1 + sum(n >= a for n in null)) / 1001, 4), round(float(np.percentile(null, 95)), 4)]}
out["AUROC with the domain as the label (WMDP = 1)"] = round(float(roc_auc_score((dom == "WMDP").astype(int), s)), 4)
print(json.dumps(out, indent=1))
