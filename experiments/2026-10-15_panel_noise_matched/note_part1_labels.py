"""09's part-1 NOTEs (Final-build checklist items 2 and 4, 10-16 01:25 PDT): (a) the group S field's LOO term used sim - run2 (non-registered comparator),
superseded by the run-mean Delta_full in the final; (b) the corrected LOO label and the pooled_455 coverage_note. Records are not edited: this script lists
the affected fields BY PATH (no values) and writes results/NOTE_part1_labels_<stamp>.json.

  python3 experiments/2026-10-15_panel_noise_matched/note_part1_labels.py
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
RECS = ["results/superseded_part1_r2/panel_part1_LOO3_2026-10-16_011309.json", "results/panel_part1_LOO3_2026-10-16_012433.json"]
LOO = "launch-disjoint: trained on the other task sets (LOO3); no real arm, honest pool or sim arm of the scored set shares a training launch (eligibility @ 71fc9f650)"
out = {"label": "NOTE (09 Final-build checklist items 2 and 4 via 2e; records unchanged): field paths only, no values", "records": [], "items": {}}
for rp in RECS:
    b = (HERE / rp).read_bytes()
    r = json.loads(b)
    hg = r["headline_group"]
    s_fields, cov, inh = [], [], []
    for s, by in r["results"].items():
        for pt, res in by.items():
            for k, g in (res.get("bias", {}).get("groups") or {}).items():
                if k.split("|")[0] == hg:
                    s_fields.append(f"results/{s}/{pt}/bias/groups/{k}/signed_mean_sim_minus_real")
            for k, g in res["groups"].items():
                if g.get("coverage_note") and s.startswith("pooled_455"):
                    cov.append({"path": f"results/{s}/{pt}/groups/{k}/coverage_note", "as_written": g["coverage_note"]})
    def walk(x, p_):  # inherited launch phrases in LOO contexts only (a string or path naming the LOO probe)
        if isinstance(x, dict):
            for k_, v_ in x.items():
                walk(v_, f"{p_}/{k_}")
        elif isinstance(x, list):
            for i_, v_ in enumerate(x):
                walk(v_, f"{p_}[{i_}]")
        elif isinstance(x, str) and ("LOO" in x or "LOO" in p_):
            inh.extend(f"{p_}: {ph}" for ph in  # noqa: B023 (called within the iteration)
                        ("real run 1 shares the probe's training launch", "run1 shares the training launch", "yardstick includes the training arm") if ph in x)

    walk(r, "")
    out["records"].append({"path": rp, "sha256": hashlib.sha256(b).hexdigest(), "headline_group": hg, "group_S_fields_with_LOO_term_sim_minus_run2": s_fields,
                           "pooled_455_coverage_notes": cov[:6], "n_pooled_455_coverage_notes": len(cov), "inherited_launch_phrases_present": inh})
out["items"]["a"] = ("group S (bias section signed_mean_sim_minus_real of the off-policy 5 + LOO group): the LOO term's comparator was sim - run2 (non-registered comparator; "
                     "the on-policy convention of tonight's bias section); superseded by the run-mean Delta_full = sim - (run1 + run2)/2 in the final (09 checklist item 2). "
                     "The per-row signed_delta_full (record 012433) and the co-headline S use the run-mean.")
out["items"]["b"] = {"LOO label": LOO + " (09's sidecar label; record 011309 carried the inherited 'real run 1 shares the probe's training launch' text in the off-policy 5 + LOO group "
                     "name and on-policy captions; record 012433 carries 09's label)",
                     "pooled_455 coverage_note, corrected": ("group mean mixes coverage only as follows: in pooled_455 the LOO probe covers S_old + S_new (its launch-disjoint sets "
                                                             "among pooled_455's members), the off-policy 5 cover S_old + S_new too; the as-written note lists S_newmains without S4, "
                                                             "which is not a pooled_455 member")}
p = HERE / "results" / f"NOTE_part1_labels_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}.json"
p.open("x").write(json.dumps(out, indent=1))
print(json.dumps({"file": p.name, "records": [{k: (v if not isinstance(v, list) else len(v)) for k, v in x.items()} for x in out["records"]]}, indent=1))
