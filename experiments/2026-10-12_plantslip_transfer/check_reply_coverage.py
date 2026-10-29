"""Review check for #1085 (Codex thread rrTQZ): did the as-run P4, P5 and P6 extractions cover every staged run (1,687
planted + 20 controls = 1,707)? q2_reply_extract_olmo.py's staged() accepts whatever shards it finds, so this checks the
records, not the code. Per pair, from the committed records at the as-run commit ee60bf09 (reachable from reply-7's merge)
and the gitignored score npz files (sha256 checked against their records; on S3 via q2_reply_s3_manifest_v1.json):
- the staging record (sha256 = every score record's "stage sha256"): planted keys per arm and the 20 controls;
- the controls and planted score records: "runs" equals the staged count, "runs with no meta" is 0;
- the score npz: the run index j maps, in order, to exactly the staged (original arm, key) list through THAT PAIR'S OWN
  reader runs.json (P4 #36's, P5 Olmo's, P6 Qwen3.5's; each checked against the score record's "stored scores sha256"),
  with folds from the same file (Codex ru0MV on #1097: an earlier version reused #36's mapping for all pairs), and every run has a finite nested score at every non-own-fold model and every scored layer (no missing meta, no run
  without a kept readout);
- corroborating, not the coverage test: the response record's "runs planted (scored)" (runs whose planted score differs
  from the unplanted one) equals the staged count per arm, and its "runs excluded" is empty.
The coverage test is the exact run index, the zero missing metas and the finite scores. Exits nonzero on any gap.
  uv run --no-project --with numpy python experiments/2026-10-12_plantslip_transfer/check_reply_coverage.py <dir with the six score npz> \
      <P4 runs.json> <P5 runs.json> <P6 runs.json>
"""
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

C, R = "ee60bf09", "experiments/2026-10-12_plantslip_transfer/results/"
STAGE, STAGE_SHA = "q2_reply_stage_2026-10-12_135318.json", "f1a68c90a5d26799925bda1c0d574728a547cde6c2ec3b76bfa18c3691c75157"
RECORDS = {"P4": ("q2_reply_scores_P4_controls_2026-10-12_143802.json", "q2_reply_scores_P4_planted_2026-10-12_151508.json", "q2_reply_delta_P4_2026-10-12_151742.json"),
           "P5": ("q2_reply_scores_P5_controls_2026-10-12_143825.json", "q2_reply_scores_P5_planted_2026-10-12_150508.json", "q2_reply_delta_P5_2026-10-12_150711.json"),
           "P6": ("q2_reply_scores_P6_controls_2026-10-12_143839.json", "q2_reply_scores_P6_planted_2026-10-12_150239.json", "q2_reply_delta_P6_2026-10-12_150434.json")}
show = lambda f: subprocess.run(["git", "show", f"{C}:{R}{f}"], capture_output=True, check=True).stdout  # noqa: E731


def main(npz_dir, *runs_ps):
    sb = show(STAGE); assert hashlib.sha256(sb).hexdigest() == STAGE_SHA, "staging record"
    st = json.loads(sb)
    runs_by = dict(zip(RECORDS, runs_ps))
    staged = {"controls": [(x["orig_arm"], x["key"]) for x in st["controls"]], "planted": [(x["orig_arm"], x["key"]) for x in st["planted"]]}
    per_arm = Counter(a for a, _ in staged["planted"])
    assert len(set(staged["planted"])) == len(staged["planted"]) == 1687 and len(set(staged["controls"])) == 20, "staged counts"
    print(f"staging record {STAGE} (sha256 {STAGE_SHA[:12]}…): planted {len(staged['planted'])} {dict(per_arm)}, controls {len(staged['controls'])}")
    for pair, (cf, pf, df) in RECORDS.items():
        rb = Path(runs_by[pair]).read_bytes(); runs_sha = hashlib.sha256(rb).hexdigest()
        runs = json.loads(rb); j_of = {(r["arm"], r["key"]): j for j, r in enumerate(runs)}
        line = [pair, f"own runs.json sha256 {runs_sha[:12]}…"]
        for which, f in (("controls", cf), ("planted", pf)):
            rec = json.loads(show(f))
            assert rec["pair"] == pair and rec["stage sha256"] == STAGE_SHA, (pair, which, "stage")
            assert rec["stored scores sha256"]["runs.json"] == runs_sha, (pair, which, "runs.json differs from the record's reader runs.json")
            n = len(staged[which]); assert rec["runs"] == n and rec["runs with no meta"] == 0, (pair, which, rec["runs"], rec["runs with no meta"])
            if which == "controls":
                assert rec["gate"] == "pass" and rec["gate: distinct controls"] == 20, (pair, "gate")
            z_p = Path(npz_dir) / rec["scores file"]; assert hashlib.sha256(z_p.read_bytes()).hexdigest() == rec["scores sha256"], (pair, which, "npz sha256")
            z = np.load(z_p)
            want = [j_of[k] for k in staged[which]]
            assert z["j"].tolist() == want, (pair, which, "run index differs from the staged list")
            layers = [k for k in z.files if k.startswith("L")]
            bad = 0
            for i, j in enumerate(want):
                f_ = int(runs[j]["fold"])
                bad += any(not np.isfinite(np.delete(z[l][i], f_)).all() for l in layers)
            assert bad == 0, (pair, which, "runs without a finite score", bad)
            line.append(f"{which}: record runs {rec['runs']}, no meta {rec['runs with no meta']}, npz runs {len(want)} in staged order, finite at every non-own-fold model on {layers}: {len(want) - bad}/{len(want)}")
        d = json.loads(show(df))
        for arm, orig in (("real-attack-e2", "real-attack-e2"), ("real-attack-e1", "real-attack-e1")):
            sc, ex = d[f"{arm}: runs planted (scored)"], d[f"{arm}: runs excluded (planted extraction missing or no kept readout)"]
            assert sc == per_arm[orig] and ex == [], (pair, arm, sc, ex)
        line.append(f"response (corroborating): runs whose planted score differs from unplanted, e2 {d['real-attack-e2: runs planted (scored)']}/{per_arm['real-attack-e2']}, e1 {d['real-attack-e1: runs planted (scored)']}/{per_arm['real-attack-e1']}, excluded none")
        print(" | ".join(line))
    print("all three pairs cover all 1,707 staged runs (1,687 planted + 20 controls)")


if __name__ == "__main__":
    assert len(sys.argv) == 5, "usage: <npz dir> <P4 runs.json> <P5 runs.json> <P6 runs.json>"
    main(*sys.argv[1:5])
