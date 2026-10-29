"""Per-set pool membership BY RUN KEY of the nm172 / nm305 runs, from the locked inputs (2e 19:0x: "counts by script"; a CUT glance at leisure).
KEYS AND PATHS ONLY, laptop, from git with every file sha256-checked: the locked INPUTS_FULL_2026-10-16_110447 (d6923a75...), its base
INPUTS_FINAL_V2_FROZEN (0e065958...), its E12 block (683d4de1...), the S_newmains H_B file the base pins (membership710, 6516d15f...) and the
S_newmains inventory (inventory_b_fix_v1, 4a5d0fb3...).
1. Contains-match: every key or string value holding nm172_shared_dotfiles or nm305_user_tracking in INPUTS_FULL, the base's pools.stage_a,
   membership710 and the E12 block's pools, with its path (list indices []). main_of / env_of / strategy_of values (task names) are never printed.
2. Admission of the five keys: H_B (membership710 pool_honest_b), H_A (membership710 held_out_honest_a; E12 H_A.e2 key_e2), H_B / H_B_coverage
   epoch 2 (E12 key_e2), and H_sim by the base's ROOT rule (the run's original folder, from the inventory, in S_newmains H_sim_roots or
   replacement_roots, and its <world batch>|<cell prefix> not in excluded_cells). S_old / S_new pool keys live in a results file (the xerr
   scale-up record), which this script does not read; the two cells are S_newmains (new-mains) cells only.
Usage: python check_membership_by_key.py --out <new json>"""
import argparse
import hashlib
import json
import re
import subprocess

KEYS = ["simh_nm360|nm172_shared_dotfiles", "honest_a_e2|nm172_shared_dotfiles", "honest_a_e2|nm305_user_tracking", "honest_b_e2|nm305_user_tracking",
        "honest_b|nm305_user_tracking"]
NEEDLE = re.compile(r"nm172_shared_dotfiles|nm305_user_tracking")
QUIET = ("main_of", "env_of", "strategy_of")  # cell -> task-name maps: paths reported, values never
PREP = "experiments/2026-10-15_fpr_sweep_507_prep/"
INPUTS_FULL = ("927fabb0d", PREP + "config/INPUTS_FULL_2026-10-16_110447.json", "d6923a75aa8f914cd3b2e92d8403f125dd095219997aebea8b1d6e4e2b67f77c")
MEMBERSHIP710_SHA = "6516d15fab77d7f344497e72db10b60c5f709b35c3a1859a2c9e016d2fc6e4ad"


def git_json(commit, path, want):
    b = subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True).stdout
    got = hashlib.sha256(b).hexdigest()
    assert got == want, f"STOP: {path} @ {commit} is {got[:16]}, not {want[:16]}"
    return json.loads(b)


def forms(x, path, out):  # path -> sorted forms holding a needle (as a key or a value); values under QUIET maps are replaced by "<value>"
    if isinstance(x, dict):
        for k, v in x.items():
            p_ = f"{path}.{k}" if path else str(k)
            if NEEDLE.search(str(k)):
                out.setdefault(f"{path}.{{key}}" if path else "{key}", set()).add(str(k))
                if path.split(".")[-1] in QUIET or str(k) in QUIET:
                    continue
            forms(v, p_, out)
    elif isinstance(x, list):
        for v in x:
            forms(v, f"{path}[]", out)
    elif isinstance(x, str) and NEEDLE.search(x):
        out.setdefault(path, set()).add("<value>" if any(q in path.split(".") for q in QUIET) else x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    inf = git_json(*INPUTS_FULL)
    base = git_json(inf["base"]["commit"], inf["base"]["path"], inf["base"]["sha256"])
    e12 = git_json(inf["e12"]["commit"], inf["e12"]["path"], inf["e12"]["sha256"])
    nm = base["pools"]["stage_a"]["S_newmains"]
    assert nm["H_B"]["sha256"] == MEMBERSHIP710_SHA, "STOP: the base pins another membership file"
    m710 = git_json(inf["base"]["commit"], PREP + nm["H_B"]["path"], nm["H_B"]["sha256"])
    ip = inf["inventories"]["S_newmains"]
    inv = git_json(ip["commit"], ip["path"], ip["sha256"])["runs"]
    src = {"INPUTS_FULL": inf, "base.pools.stage_a": base["pools"]["stage_a"], "membership710": m710, "e12.pools": e12["pools"]}
    found = {name: {p_: sorted(v) for p_, v in forms(doc, "", {}).items()} for name, doc in src.items()}
    e2 = sorted({(e["key_e2"], f"{s}.{pool}") for s, p in e12["pools"].items() for pool, g in p.items() for e in (g.get("e2") or [])})
    adm = {}
    for k in KEYS:
        w, c = k.split("|", 1)
        r = {"H_B (membership710 pool_honest_b)": k in m710["pool_honest_b"], "H_A (membership710 held_out_honest_a)": k in m710["held_out_honest_a"],
             "E12 epoch-2 pools": [sp for kk, sp in e2 if kk == k]}
        if w.startswith("simh_"):  # the root rule: the run's original folder (inventory) among the roots, its cell not excluded
            runs = [x for x in inv if x.get("world") == "sim_honest" and x.get("task") == c]
            folders = sorted({x.get("original_folder") for x in runs})
            batch = [re.match(r"(simh_nm360_s\w+?)_sim", f_).group(1) for f_ in folders if re.match(r"(simh_nm360_s\w+?)_sim", f_)]
            excl = [e for e in nm["excluded_cells"] if any(e == f"{b_}|{c.split('_', 1)[0]}" for b_ in batch)]
            r |= {"H_sim root rule": {"inventory_runs": len(runs), "original_folders": folders,
                                       "in_H_sim_roots": [f_ in nm["H_sim_roots"] for f_ in folders],
                                       "in_replacement_roots": [f_ in nm["replacement_roots"] for f_ in folders], "excluded_cells_matching": excl,
                                       "admitted": len(runs) == 1 and folders[0] in nm["H_sim_roots"] + nm["replacement_roots"] and not excl}}
        adm[k] = r
    res = {"sources": {"INPUTS_FULL": INPUTS_FULL, "base": inf["base"], "e12": inf["e12"], "membership710": [inf["base"]["commit"], nm["H_B"]["path"],
                                                                                                             MEMBERSHIP710_SHA],
                       "inventory_S_newmains": ip}, "forms": found, "admission": adm}
    with open(a.out, "x") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({"admission": adm, "forms_per_source": {n: len(v) for n, v in found.items()}}))


if __name__ == "__main__":
    main()
