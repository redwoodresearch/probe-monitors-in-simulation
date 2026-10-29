"""The pairs file for check_eval_pairs.py --pairs-file (09 via 2e 19:0x; prepare only, runs on 2e's go after refresh2's count of byte-identical
declared E12 pairs lands). KEYS AND DIGESTS ONLY, laptop (keys-only reads of the pinned configs). For each pair in refresh2's list, the epoch-2 side
is the key_e2 of an E12 block entry (683d4de1..., git 22d7089f5) and the other side must be that entry's declared epoch-1 key; the epoch-2 .eval is
the entry's <launch_id>/<eval> with its eval_sha256; the epoch-1 .eval is the one run of that world and task in the set's frozen inventory (the
locked INPUTS_FULL's inventories, d6923a75..., git 927fabb0d), narrowed to the set's pool flag when the world and task name more than one run.
A pair that resolves otherwise is listed in "unresolved" with the reason; the check runs only when that list is empty.
Usage: python make_e12_pairs_file.py --identical <refresh2's pairs json> --out <new json>"""
import argparse
import hashlib
import json
import subprocess

E12 = ("22d7089f5", "experiments/2026-10-15_fpr_sweep_507_prep/config/E12_BLOCK_2026-10-16_092322.json",
       "683d4de1040f489a768a44fe5ad53f787d35d8ca22117b897a42a75880e1f4dc")
INPUTS_FULL = ("927fabb0d", "experiments/2026-10-15_fpr_sweep_507_prep/config/INPUTS_FULL_2026-10-16_110447.json",
               "d6923a75aa8f914cd3b2e92d8403f125dd095219997aebea8b1d6e4e2b67f77c")
POOL_FLAG = {"S_new": "in_registered_S_new_pool (FINAL pool_keys_new)", "S_old": "in_S_old_pool (FINAL pool_keys_old)",
             "S_newmains": "in_counted_membership710"}


def git_json(commit, path, want):
    b = subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True).stdout
    assert hashlib.sha256(b).hexdigest() == want, f"STOP: {path} @ {commit} differs from {want[:16]}"
    return json.loads(b)


def pairs_of(doc):  # refresh2's list: [[k1, k2]], {"pairs": [...]}, or items {"a", "b"} / {"keys": [k1, k2]}
    items = doc["pairs"] if isinstance(doc, dict) else doc
    out = []
    for it in items:
        ks = it if isinstance(it, list) else it.get("keys") or [it["a"], it["b"]]
        assert len(ks) == 2, ("a pair of two keys", ks)
        out.append(tuple(ks))
    return out


def resolve(pairs, e12, inv):
    """-> (resolved pairs, unresolved). e12: key_e2 -> {set, key, launch_id, eval, eval_sha256}; inv: set -> runs."""
    ok, bad = [], []
    for x, y in pairs:
        e2 = x if x in e12 else y if y in e12 else None
        e1 = y if e2 == x else x
        if e2 is None or (x in e12 and y in e12):
            bad.append({"pair": [x, y], "why": "not exactly one epoch-2 key of the E12 block"})
            continue
        e = e12[e2]
        if e["key"] != e1:
            bad.append({"pair": [x, y], "why": "not the declared E12 pair"})
            continue
        w, c = e1.split("|", 1)
        cand = [r for r in inv[e["set"]] if r.get("world") == w and r.get("task") == c]
        if len(cand) > 1:
            cand = [r for r in cand if r.get(POOL_FLAG[e["set"]]) is True]
        if len(cand) != 1:
            bad.append({"pair": [x, y], "why": f"{len(cand)} epoch-1 runs of that world and task in the {e['set']} inventory"})
            continue
        ok.append({"a": e2, "a_eval": f"{e['launch_id']}/{e['eval']}", "a_sha256": e["eval_sha256"],
                   "b": e1, "b_eval": f"{cand[0]['launch']}/{cand[0]['eval_rel']}", "b_sha256": cand[0]["eval_sha256"], "set": e["set"]})
    return ok, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--identical", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ib = open(a.identical, "rb").read()
    e12d, inf = git_json(*E12), git_json(*INPUTS_FULL)
    e12 = {e["key_e2"]: e | {"set": s} for s, p in e12d["pools"].items() for g in p.values() for e in (g.get("e2") or [])}
    inv = {s: git_json(inf["inventories"][s]["commit"], inf["inventories"][s]["path"], inf["inventories"][s]["sha256"])["runs"] for s in POOL_FLAG}
    ok, bad = resolve(pairs_of(json.loads(ib)), e12, inv)
    res = {"identical_file": a.identical, "identical_sha256": hashlib.sha256(ib).hexdigest(), "e12": E12, "inputs_full": INPUTS_FULL,
           "pairs": ok, "unresolved": bad}
    with open(a.out, "x") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({"pairs": len(ok), "unresolved": len(bad), "unresolved_reasons": sorted({b["why"] for b in bad})}))


if __name__ == "__main__":
    main()
