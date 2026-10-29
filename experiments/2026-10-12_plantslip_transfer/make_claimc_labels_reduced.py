"""claim C's attempt labels, reduced to (world|task) -> label, for the image of run_plantslip_onpolicy.py ($0, no text).

The same logic as the inline build that made data/claimc_labels_reduced.json at about 14:04 PT 10-12 (sha256
d00e8877...): read the five label files from claim C's pinned commit 73c9f978 (as analyze_claim_c.main does): `attack` is run 2 in
the second-attack file and run 1 otherwise, `sim_attack@september` is sim; keep rows with a label; refuse duplicates.
Writes {"ref", "files_sha256", "labels"}; keys and labels only.

  python make_claimc_labels_reduced.py <out.json>     (run from inside the repo)
"""

import hashlib
import json
import subprocess
import sys

REF = "73c9f9789d9dbc23e7f8350bfdbc285ca5368e6b"
files = ["phase2_attempt_labels_2026-10-09_000523_rerun_strict_selfreport_2026-10-10_111809.jsonl",
         "phase2_attempt_labels_2026-10-09_173815_scaleup_canary_selfreport_2026-10-10_111817.jsonl",
         "phase2_attempt_labels_2026-10-09_182505_scaleup_selfreport_2026-10-10_111838.jsonl",
         "phase2_attempt_labels_2026-10-10_001314_scaleup_topup_selfreport_2026-10-10_111821.jsonl",
         "phase2_attempt_labels_2026-10-10_001314_second_attack_2026-10-10_111840.jsonl"]
lab, shas = {}, {}
for f in files:
    raw = subprocess.run(["git", "show", f"{REF}:experiments/2026-10-08_q3_onpolicy_figures/results/{f}"], capture_output=True, check=True).stdout
    shas[f] = hashlib.sha256(raw).hexdigest()
    for line in raw.decode().splitlines():
        r = json.loads(line)
        w = {"attack": "run2" if "second_attack" in f else "run1", "sim_attack@september": "sim"}.get(r["arm"])
        if w and r.get("label"):
            k = f"{w}|{r['task']}"
            if k in lab:
                raise SystemExit("two labels")
            lab[k] = r["label"]
out = {"ref": REF, "files_sha256": shas, "labels": lab}
open(sys.argv[1], "x").write(json.dumps(out))
print(len(lab), hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())
