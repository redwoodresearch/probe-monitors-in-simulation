"""Inputs of check_token_prefix.py (09 / 0d via 2e 19:1x; laptop, KEYS AND PATHS ONLY): per run, its eval key, its raw score-file key, its staged file
in each line (32k on the q3su311 acts Volume, uncut on q3cutfull) and the score files that hold it; the pairs to compare; and, per run, the
canonical main its cell maps to (S_newmains: membership710 main_of, reported as a sha256 prefix, never the name).
Score files come from check 1's result (check_nm172_provenance_2026-10-16_185935.json, sha256 b6132b04..., the nm172 hits by label and path); a
run of another cell reads the files of the nm172 run of the same world and epoch (one folder holds every cell of its batch). Eval keys come from the
frozen inventory (inventory_b_fix_v1, 4a5d0fb3...) for epoch 1 and the sim, and from the E12 block (683d4de1...) for epoch 2.
Usage: python make_token_prefix_inputs.py --check1 <result json> --out <new json>"""
import argparse
import hashlib
import json
import subprocess

CHECK1_SHA = "b6132b04ddb3b038261c518166df7ecca45273a0e071c40d70e3c0ad4ba73f87"
INV = ("09b070617", "experiments/2026-10-15_cut_sensitivity/results/inventory_b_fix_v1_2026-10-16_025058.json",
       "4a5d0fb3cad32740f849df5def3c7a03a40a18aa4c42bc360709f0c740973d03")
E12 = ("22d7089f5", "experiments/2026-10-15_fpr_sweep_507_prep/config/E12_BLOCK_2026-10-16_092322.json",
       "683d4de1040f489a768a44fe5ad53f787d35d8ca22117b897a42a75880e1f4dc")
M710 = ("a6bde80a2", "experiments/2026-10-15_fpr_sweep_507_prep/config/membership710.json", "6516d15fab77d7f344497e72db10b60c5f709b35c3a1859a2c9e016d2fc6e4ad")
STAGED = {  # (world, epoch) -> line -> (mount, staged file)
    ("honest_a", 2): {"32k": ("/a32", "ep2_nm_v1/inputs/honest_a.jsonl"), "uncut": ("/aU", "ep2_nm_cutfull_v1/inputs/honest_a.jsonl")},
    ("honest_b", 2): {"32k": ("/a32", "ep2_nm_v1/inputs/honest_b.jsonl"), "uncut": ("/aU", "ep2_nm_cutfull_v1/inputs/honest_b.jsonl")},
    ("honest_a", 1): {"32k": ("/a32", "nm360_s1_main_s1_v1/inputs/honest_a.jsonl"), "uncut": ("/aU", "nm360_s1_main_s1_cutfull_v1/inputs/honest_a.jsonl")},
    ("honest_b", 1): {"32k": ("/a32", "nm360_s1_main_s1_v1/inputs/honest_b.jsonl"), "uncut": ("/aU", "nm360_s1_main_s1_cutfull_v1/inputs/honest_b.jsonl")},
    ("simh_nm360", 1): {"32k": ("/a32", "simh_nm360_s1_sim_s1_v1/inputs/sim_honest@september.jsonl"),
                        "uncut": ("/aU", "simh_nm360_s1_sim_s1_cutfull_v1/inputs/sim_honest@september.jsonl")},
}
C172, C305, C204 = "nm172_shared_dotfiles", "nm305_user_tracking", "nm204_slack_clone"
PAIRS = [  # (x, y, kind)
    (f"honest_a_e2|{C172}", f"simh_nm360|{C172}", "nm172 real/sim"),
    (f"honest_b|{C305}", f"honest_a_e2|{C305}", "nm305"), (f"honest_b|{C305}", f"honest_b_e2|{C305}", "nm305"),
    (f"honest_a_e2|{C305}", f"honest_b_e2|{C305}", "nm305"),
    (f"honest_a_e2|{C172}", f"honest_a|{C172}", "famexp e2/e1"), (f"honest_b_e2|{C172}", f"honest_b|{C172}", "famexp e2/e1"),
    (f"honest_a_e2|{C204}", f"honest_a|{C204}", "CONTROL: a long e2/e1 pair (expected to diverge before a readout)"),
]


def git_json(commit, path, want):
    b = subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True).stdout
    assert hashlib.sha256(b).hexdigest() == want, f"STOP: {path} @ {commit}"
    return json.loads(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check1", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cb = open(a.check1, "rb").read()
    assert hashlib.sha256(cb).hexdigest() == CHECK1_SHA, "STOP: check 1's result differs"
    hits = json.loads(cb)["score_hits"]
    inv, e12, m710 = git_json(*INV)["runs"], git_json(*E12), git_json(*M710)
    e2_key = {e["key_e2"]: e["inventory_key"] for p in e12["pools"].values() for g in p.values() for e in (g.get("e2") or [])}
    runs = {}
    for lab in sorted({x for p in PAIRS for x in p[:2]}):
        w, c = lab.split("|", 1)
        ep = 2 if w.endswith("_e2") else 1
        w0 = w.removesuffix("_e2")
        if ep == 2:
            ek = e2_key[lab]
        else:
            cand = [r["key"] for r in inv if r.get("task") == c and r.get("world") == ("sim_honest" if w0.startswith("simh_") else w0)]
            assert len(cand) == 1, (lab, len(cand))
            ek = cand[0]
        src = f"{w}|{C172}"  # the nm172 run of the same world and epoch: its files hold every cell of the batch
        files = sorted({(h["path"].split("/", 2)[1], h["path"]) for h in hits if h["label"] == src})
        main_ = m710["main_of"].get(c)
        runs[lab] = {"eval_ref": ek, "npz_row": f"{w0}|{c}", "staged": STAGED[(w0, ep)], "files": [["/" + v, p] for v, p in files],
                     "main_sha256_12": hashlib.sha256(main_.encode()).hexdigest()[:12] if main_ else None}
    by_cell = {}
    for lab, r in runs.items():
        by_cell.setdefault(lab.split("|", 1)[1], set()).add(r["main_sha256_12"])
    res = {"check1": [a.check1, CHECK1_SHA], "inventory": INV, "e12": E12, "membership710": M710, "pairs": PAIRS, "runs": runs,
           "one_main_per_cell": {c: len(v) == 1 and None not in v for c, v in by_cell.items()}}
    with open(a.out, "x") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({"runs": {k: {"files": len(v["files"]), "main": v["main_sha256_12"]} for k, v in runs.items()},
                      "one_main_per_cell": res["one_main_per_cell"]}))


if __name__ == "__main__":
    main()
