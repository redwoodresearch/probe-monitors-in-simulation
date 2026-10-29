"""Post-lock ratification of CUT's corrected C17 overrides against the LOCKED INPUTS_FULL's pins (keys, paths and sha256 values only; no score).

Asserts, per set and readout, fail closed: the locked override's bytes match its pin; both sides have entries and no duplicate entry identity
(arm folder x file, H_sim file x root); the same entries; every entry has a scored path and sha256, unchanged; every new entry's gate citation names
the fixed gate record (full path and sha256) and keeps its "file"; every other field that differs (absent and null distinguished) is runs_without_rows
provenance on an own_all_rs / own_all_rs_uncut file; the fixed gate record has n_failing_entries 0 and covers n_entries = the override's entries.
Prints the counts and writes nothing.

  python ratify_c17_overrides.py config/INPUTS_FULL_2026-10-16_110447.json
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = "experiments/2026-10-15_fpr_sweep_507_prep/"
NEW = {  # CUT's corrected overrides @ 6c55efa9a and their fixed gate records (507-prep relay 11:12)
    "S_paper493|R": ("6c55efa9a", "config/cutfull/override_S_paper493_R_2026-10-16_110306_gated.json", "results/cutfull/gates_S_paper493_R_2026-10-16_110518.json"),
    "S_paper493|C": ("6c55efa9a", "config/cutfull/override_S_paper493_C_2026-10-16_110237_gated.json", "results/cutfull/gates_S_paper493_C_2026-10-16_110515.json"),
    "pools_ep2|R": ("6c55efa9a", "config/cutfull/override_pools_ep2_R_2026-10-16_105917_gated.json", "results/cutfull/gates_pools_ep2_R_2026-10-16_110516.json"),
    "pools_ep2|C": ("6c55efa9a", "config/cutfull/override_pools_ep2_C_2026-10-16_105907_gated.json", "results/cutfull/gates_pools_ep2_C_2026-10-16_110511.json"),
    "S_old|R": ("6c55efa9a", "config/cutfull/override_S_old_R_2026-10-16_110647_gated.json", "results/cutfull/gates_S_old_R_2026-10-16_110811.json"),
    "S_old|C": ("6c55efa9a", "config/cutfull/override_S_old_C_2026-10-16_110600_gated.json", "results/cutfull/gates_S_old_C_2026-10-16_110814.json"),
}
PROV = {"runs_without_rows", "runs_without_rows_source"}
MISSING = object()


def show(commit, path):
    return subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True, cwd=HERE).stdout


def entries(ov, where, bad):
    out = {}
    for k, e in [(("arm", f, fn), e) for f, x in (ov.get("arms") or {}).items() for fn, e in x.items()] + \
                [(("simh", e.get("root"), fn), e) for fn, ents in (ov.get("simh") or {}).items() for e in ents]:
        if k in out:
            bad.append(f"{where} {k}: duplicate entry identity")
        out[k] = e
    if not out:
        bad.append(f"{where}: no entries")
    return out


def main():
    m = json.loads((HERE / sys.argv[1]).read_text())
    out, bad = {}, []
    for key, (c, p, gp) in NEW.items():
        pin_ = m["overrides"][key]
        ob_ = show(pin_["commit"], pin_["path"])
        if hashlib.sha256(ob_).hexdigest() != pin_["sha256"]:
            bad.append(f"{key}: the locked override's bytes differ from its pin")
        old, new = entries(json.loads(ob_), f"{key} locked", bad), entries(json.loads(show(c, D + p)), f"{key} new", bad)
        if set(old) != set(new):
            bad.append(f"{key}: entries added {len(set(new) - set(old))}, removed {len(set(old) - set(new))}")
        gb_ = show(c, D + gp)
        gsha, g = hashlib.sha256(gb_).hexdigest(), json.loads(gb_)
        changed, rwr_old, rwr_new, restamp = {}, 0, 0, 0
        for k in sorted(set(old) & set(new), key=str):
            eo, en = old[k], new[k]
            if not all(isinstance(x.get(f), str) and x.get(f) for x in (eo, en) for f in ("path", "sha256")) or \
                    (eo["path"], eo["sha256"]) != (en["path"], en["sha256"]):
                bad.append(f"{key} {k}: scored path or sha256 missing or changed")
            go_, gn_ = eo.get("gate"), en.get("gate")
            if not (isinstance(gn_, dict) and gn_.get("path") == D + gp and gn_.get("sha256") == gsha and isinstance(go_, dict)
                    and gn_.get("file") == go_.get("file") and set(gn_) == set(go_)):
                bad.append(f"{key} {k}: the gate citation does not name the fixed gate record (full path, sha256) with the same file and fields")
            restamp += go_ != gn_
            diff = {f for f in set(eo) | set(en) if eo.get(f, MISSING) != en.get(f, MISSING)} - {"gate"}
            if (diff - PROV) or (diff and k[2].split("/")[0] not in ("own_all_rs", "own_all_rs_uncut")):
                bad.append(f"{key} {k}: fields changed beyond the gate citation and own_all_rs provenance: {sorted(diff)}")
            if diff:
                changed[k[2].split("/")[0]] = changed.get(k[2].split("/")[0], 0) + 1
            rwr_old += len(eo.get("runs_without_rows") or [])
            rwr_new += len(en.get("runs_without_rows") or [])
        if g.get("n_failing_entries") != 0 or g.get("n_entries") != len(new):
            bad.append(f"{key}: gate record n_failing_entries {g.get('n_failing_entries')}, n_entries {g.get('n_entries')} vs {len(new)} entries")
        lead_ = sorted(f"{k[1]} | {r}" for k, e in new.items() if k[2].startswith("own_all_rs_uncut/own_all_rs_uncut_L48") and k[0] == "arm"
                       and k[1].endswith("attack") for r in (e.get("runs_without_rows") or []))
        out[key] = {"entries": len(new), "added": len(set(new) - set(old)), "removed": len(set(old) - set(new)), "gate_citations_restamped": restamp,
                    "provenance_changed_by_family": changed, "runs_without_rows_total_locked_vs_new": [rwr_old, rwr_new],
                    "gate_record": {"path": D + gp, "sha256": gsha, "n_entries": g.get("n_entries"), "n_failing_entries": g.get("n_failing_entries")},
                    "lead_attack_listings": lead_}
    print(json.dumps({"locked": sys.argv[1], "sets": out, "problems": bad, "pass": not bad}, indent=1))
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
