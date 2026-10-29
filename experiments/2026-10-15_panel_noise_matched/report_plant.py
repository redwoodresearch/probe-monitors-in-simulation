"""Values table of a PANEL record's planted-shift check (JSON reads only): per level, set (pooled; per-set sensitivity), readout group and arm: the achieved
shift, the paired change (planted - unplanted, same draws) of the excess, signed mean and b with 95% joint intervals, the share of draws above 0, detection,
the prediction and mismatch flags; unattainable arms as labelled; the delta-0 baselines; the captions. Also checks that the record's results equal a reference
PANEL record's by key (the planted check must not move the base statistics). Writes results/table_<record stem>.json (exclusive create).

  python3 experiments/2026-10-15_panel_noise_matched/report_plant.py <record.json> <reference record.json>
"""

import hashlib
import json
import sys
from pathlib import Path

from probe_shift.fmtpts import p2, raw_and_printed

P, Q = Path(sys.argv[1]), Path(sys.argv[2])
b = P.read_bytes()
r, ref = json.loads(b), json.loads(Q.read_bytes())
pc_ = r["plant_check"]

f = lambda z: None if not z else raw_and_printed(z) | ({"share_of_draws_above_0": z["share_of_draws_above_0"]} if z.get("share_of_draws_above_0") is not None else {})
out = {"record": P.name, "sha256": hashlib.sha256(b).hexdigest(), "git": r["meta"]["git"], "label": pc_["label"], "headline_group": r["headline_group"],
       "base_results_equal_reference": json.dumps(r["results"], sort_keys=True) == json.dumps(ref["results"], sort_keys=True),
       "base_results_equal_reference_outside_bias": json.dumps({s_: {p_: {k: v for k, v in x.items() if k != "bias"} for p_, x in by.items()} for s_, by in r["results"].items()}, sort_keys=True)
       == json.dumps({s_: {p_: {k: v for k, v in x.items() if k != "bias"} for p_, x in by.items()} for s_, by in ref["results"].items()}, sort_keys=True),
       "base_results_note": ("with signed_runmean_all the bias section's on-policy signed terms are the run-mean (and a signed Y field is added), so the whole-results equality "
                             "to an older reference is not the invariant; the results outside the bias section must equal it") if r["meta"]["config"].get("signed_runmean_all") else None,
       "detection_rule": (pc_.get("detection_rule") or next((a.get("detection_rule") for by in pc_.values() if isinstance(by, dict) for bs in by.values() if isinstance(bs, dict)
                                                              for arms in bs.values() if isinstance(arms, dict) for a in arms.values() if isinstance(a, dict) and a.get("detection_rule")), None)),
       "reference": {"record": Q.name, "sha256": hashlib.sha256(Q.read_bytes()).hexdigest()}, "gates": pc_["gates"], "captions": {k: pc_[k] for k in ("caption", "caption_signed_mean") if k in pc_} | {"by_readout": pc_.get("captions_by_readout"), "lead": pc_.get("caption_readout"),
                                                                                                         "calibration_label": pc_.get("calibration_label")},
       "unattainable_at_lead": pc_.get("unattainable_at_lead"), "arms": [],
       "paired_change_excludes_0_definition": "paired planted - unplanted on the same draws excludes 0; diagnostic only, NOT Aswin's detection rule",
       "conservative_note": pc_.get("conservative_note")}
for pt, by in pc_.items():
    if not (isinstance(by, dict) and "pooled" in by):
        continue
    for sname, bys in by.items():
        for sel, arms in bys.items():
            for tag, a in arms.items():
                row = {"level": pt, "set": sname, "readout": sel, "arm": tag}
                if a.get("unattainable"):
                    row |= {"unattainable": a["unattainable"], "label": a["label"]}
                elif tag == "0":
                    row |= {"baseline": {k: f(a[k]) for k in ("excess", "signed", "b")}, "members": a["members"]}
                else:
                    row |= {"achieved_points": round(a["achieved_points"]["group_mean"], 3), "change": {k: f(v) for k, v in a["paired_change"].items()},
                            "detected_aswin_rule": a["detected"], "paired_change_excludes_0": a.get("paired_change_excludes_0"),
                            "aswin_rule": {k: v | {"margin_points_printed": p2(v.get("margin_points")), "move_ratio_printed": p2(v.get("move_ratio"), sign=False)}
                                           for k, v in (a.get("aswin_rule") or {}).items()},
                            "planted": {k: f(v) for k, v in a["planted"].items()},
                            "prediction": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in a["prediction"].items()},
                            "mismatch": a["mechanism_mismatch_flag"]}
                out["arms"].append(row)
q_ = P.with_name(f"table_{P.stem}.json")
q_.open("x").write(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
