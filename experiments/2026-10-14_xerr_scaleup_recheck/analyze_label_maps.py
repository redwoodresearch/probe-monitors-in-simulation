"""Blinded check of the key map's label maps against a full record. Prints leaf counts and all/not-all per check only
(never a label, a value, or a partial count that could reveal a decision category). PRELIMINARY.

  python3 -I analyze_label_maps.py <full record json> <config/full_override.json> <target record json>

Per mapped or composed leaf it checks:
  (a) my category is defined (my labels agree with my own booleans);
  (b) the category from the TARGET's booleans equals mine;
  (c) the target's label is the vocabulary string of the target's own category (09's label agrees with 09's booleans);
  (d) map(mine) / compose(my category) equals the target's label.
Decision-free maps (U_label, the fewer-than-two note) check (d) and their condition, as before.
"""

import json
import math
import sys

UNDER_MINE = "underpowered (n < 30 mains)"
POOLED_OK, DIFFER = "strata pooled (TOST passed)", "strata differ (legacy label: equivalence not established); stratum results lead"
NDRAW = 10000


def get(x, path):
    for k in [p for p in path.split("/") if p]:
        if not isinstance(x, dict) or k not in x:
            return None
        x = x[k]
    return x


def cat_from(pp: dict, mains: list):
    names = list(pp)
    return {"pass": bool(names) and all(pp[n]["within_5"] is True for n in names), "under": min(mains) < 30,
            "beyond": [n for n in names if pp[n]["difference_beyond_5_established"] is True]}


def mine_cat(t):
    try:
        c = cat_from(t["per_member"], t["n_mains_with_without"])
    except (KeyError, TypeError, ValueError):
        return None
    u = f"; {UNDER_MINE}" if c["under"] else ""
    ok = ((t["pooled_label"] == POOLED_OK) == c["pass"] and (c["pass"] or t["pooled_label"] == DIFFER + u)
          and t["difference_reading"].startswith("difference established") == bool(c["beyond"])
          and (bool(c["beyond"]) or t["difference_reading"] == "inconclusive" + u))
    return c if ok else None


def vocab_label(kind, c, v):
    u = v["underpowered_suffix"] if c["under"] else ""
    if kind == "tost_pooled_label":
        return (v["pooled"] if c["pass"] else v["differ"]) + u
    if kind == "tost_difference_reading":
        return (v["beyond_prefix"] + ",".join(c["beyond"]) if c["beyond"] else v["inconclusive"]) + u
    return v["combined_pooled"] if c["pass"] else v["differ"] + u


def a30_cat(ci, finite, point, final: bool, rule: str):
    """Each side's OWN rule, in its own order, on its own fields. Mine (contrast_block): withhold on < 95% jointly finite
    draws or no interval FIRST, then descriptive outside final mode, then the three labels. 09's (LABEL_VOCAB.md):
    outside final mode the primary cell carries the non-final string; in final mode it withholds on < 95% finite draws
    or an undefined point, then the three labels."""
    no_ci = ci is None or ci[0] is None or ci[1] is None or finite is None or finite < 0.95 * NDRAW
    if rule == "mine":
        if no_ci:
            return "withheld"
        if not final:
            return "nonfinal"
    else:
        if not final:
            return "nonfinal"
        if no_ci or not (isinstance(point, (int, float)) and math.isfinite(point)):
            return "withheld"
    return "replicated" if ci[1] < 0 else ("not replicated" if ci[0] > -2.5 else "inconclusive")


rec, ov, tgt = (json.load(open(p)) for p in sys.argv[1:4])
mine, v = rec["result"]["mine"], ov["label_vocab"]
tmap = {e["mine"]: e.get("target") for e in ov["key_map"] if "target" in e}
out = {}


def tally(key, checks):
    o = out.setdefault(key, {"leaves": 0, **{k: 0 for k in checks}})
    o["leaves"] += 1
    for k, ok in checks.items():
        o[k] += bool(ok)


for e in ov["key_map"]:
    specs = []
    if "compose" in e:
        specs.append((e["compose"], e["mine"], e["target"]))
    for f, sp in (e.get("field_compose") or {}).items():
        if get(tgt, f"{e['target']}/{f}") is not None or get(mine, f"{e['mine']}/{f}") is not None:
            specs.append((sp, f"{e['mine']}/{f}", f"{e['target']}/{f}"))
    for sp, mpath, tpath in specs:
        t_mine = get(mine, sp["test"])
        own = get(mine, mpath) if "field" in sp else None  # a family block's own label must agree with its test
        own_ok = "field" not in sp or (own is None and sp.get("absent_ok", False)) or (
            isinstance(t_mine, dict) and own == t_mine.get("pooled_label"))
        T = tmap[f"{sp['test']}/pooled_label"].rsplit("/", 1)[0]  # the target's test block for that test
        tb = get(tgt, T) or {}
        mc = mine_cat(t_mine) if isinstance(t_mine, dict) else None
        try:
            tc = cat_from(tb["per_probe"], tb["mains"])
        except (KeyError, TypeError, ValueError):
            tc = None
        lab = get(tgt, tpath)
        tally(sp["kind"], {"a_mine_category_defined": mc is not None and own_ok,
                           "b_target_booleans_same_category": mc is not None and mc == tc,
                           "c_target_label_matches_its_booleans": tc is not None and lab == vocab_label(sp["kind"], tc, v),
                           "d_mapped_mine_equals_target": mc is not None and vocab_label(sp["kind"], mc, v) == lab})
    if e.get("mine", "").endswith("/primary/reading") and "value_map" in e:
        c = e["mine"].split("/")[1]
        P, A = e["mine"].rsplit("/", 1)[0], e["target"].rsplit("/", 1)[0]
        lab_m, lab_t = get(mine, e["mine"]), get(tgt, e["target"])
        if c == "R|0.05":
            final = (tgt.get("spec") or {}).get("mode") == "final"
            mc = a30_cat(get(mine, f"{P}/C_ci95"), get(mine, f"{P}/jointly_finite_draws"), get(mine, f"{P}/C"), final, "mine")
            tc = a30_cat(get(tgt, f"{A}/delta/C/ci95"), get(tgt, f"{A}/delta/C/finite_draws"), get(tgt, f"{A}/delta/C/point"),
                         final, "09")
            want = {"withheld": "label withheld (< 95% jointly finite draws)", "replicated": "replicated",
                    "not replicated": "not replicated: a gap as large as half the prediction is excluded",
                    "inconclusive": "inconclusive at this n", "nonfinal": "descriptive (not the primary cell, or not final)"}
            tally("append30_primary_cell", {"a_mine_category_defined": lab_m == want[mc],
                                            "b_target_booleans_same_category": mc == tc,
                                            "c_target_label_matches_its_booleans": lab_t == e["value_map"].get(want[tc]),
                                            "d_mapped_mine_equals_target": e["value_map"].get(lab_m) == lab_t})
        else:
            d = "descriptive (not the primary cell, or not final)"
            tally("append30_other_cells", {"a_mine_category_defined": lab_m == d, "b_target_booleans_same_category": True,
                                           "c_target_label_matches_its_booleans": lab_t == e["value_map"][d],
                                           "d_mapped_mine_equals_target": e["value_map"].get(lab_m) == lab_t})
    for f, fm in (e.get("field_value_map") or {}).items():
        a, b = get(get(mine, e["mine"]) or {}, f), get(get(tgt, e["target"]) or {}, f)
        if not (isinstance(a, str) and a in fm):
            continue
        if f == "note":
            pm, pt = get(mine, e["mine"]).get("per_probe") or {}, get(tgt, e["target"]).get("per_probe") or {}
            cond = len(pm) < 2 and len(pt) < 2 and set(pm) == set(pt)
        else:
            cond = b == fm[a]
        tally(f, {"d_mapped_mine_equals_target": fm[a] == b, "condition_holds_both_sides": cond})
# all-or-nothing per check: a partial count could reveal how many leaves fall in one decision category
shown = {k: {"leaves": o["leaves"], **{c: ("all" if n == o["leaves"] else "NOT ALL") for c, n in o.items() if c != "leaves"}}
         for k, o in out.items()}
print(json.dumps({"label": "PRELIMINARY; blinded label-map check (leaf counts and all/not-all only)", "by_kind": shown},
                 indent=1))
ok = bool(out) and all(all(n == o["leaves"] for n in o.values()) for o in out.values())
print("ALL MAPPED PAIRS AGREE" if ok else "CHECK FAILED")
sys.exit(0 if ok else 1)  # Codex round 31 (nonblocking): a failed check exits nonzero
