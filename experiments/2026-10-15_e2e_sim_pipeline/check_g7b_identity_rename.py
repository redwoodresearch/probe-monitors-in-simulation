"""G7b VALUE: keys-only reconciliation of the identity_parts rename (2e 10-17 00:2x). Reads four input listings (no score, npz or record value):
the two gitleaks-blocked copies (scratchpad, not committable: the hook flags their digests under credential-like part names) and the two committed
recaptures, plus the PASSED Stage 1 listing @ c8887f6e7 from git. Prints and writes counts, part names and file sha256s only.
Run from the repo root: uv run python experiments/2026-10-15_e2e_sim_pipeline/check_g7b_identity_rename.py"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
SCR = Path("/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/a683364c-3a60-44cd-bb15-e08d87dd4f9a/scratchpad/g7bfold/dup_listings/gitleaks_blocked")
R = "check_stagea_inputs_alltasks_CRD_FULL_r2_allorigins_"
FILES = {"stage1_blocked": SCR / f"{R}e12_2026-10-16_113538_pwcatalogue_foldmodel_2026-10-16_234849.json",
         "stage1_new": HERE / "results" / f"{R}e12_2026-10-16_113538_pwcatalogue_foldmodel_2026-10-17_000133.json",
         "value_blocked": SCR / f"{R}pooledworld_scored_e12_2026-10-16_211215_foldmodel_2026-10-16_235302.json",
         "value_new": HERE / "results" / f"{R}pooledworld_scored_e12_2026-10-16_211215_foldmodel_2026-10-17_000358.json"}
PASSED = ("c8887f6e7", f"experiments/2026-10-15_e2e_sim_pipeline/results/{R}e12_2026-10-16_113538_pwcatalogue_foldmodel_2026-10-16_211643.json")
SUF = "|sha256"


def parts(d):  # blocked copies: {name: digest}; new ones: sorted [name, digest] pairs (names end in "|sha256")
    p_ = d["identity_parts"]
    if isinstance(p_, dict):
        return p_, "dict"
    out = dict(p_)
    assert len(out) == len(p_), "a repeated part name"
    return out, "pairs"


def main():
    L, out = {}, {"files": {}, "rename_pairs": {}, "value_vs_stage1": {}, "passed_stage1": {}}
    for k_, p_ in FILES.items():
        b_ = p_.read_bytes()
        d_ = json.loads(b_)
        L[k_], form_ = parts(d_)
        out["files"][k_] = {"path": str(p_.relative_to(HERE.parents[1])) if HERE.parents[1] in p_.parents else str(p_), "file_sha256": hashlib.sha256(b_).hexdigest(),
                            "inputs_sha256": d_["inputs_sha256"], "form": form_, "n_parts": len(L[k_])}
    for old_, new_ in (("stage1_blocked", "stage1_new"), ("value_blocked", "value_new")):  # the rename map: name -> name + "|sha256"
        o_, n_ = L[old_], L[new_]
        mapped_ = {k_ + SUF: v_ for k_, v_ in o_.items()}
        out["rename_pairs"][f"{old_} -> {new_}"] = {
            "n_old": len(o_), "n_new": len(n_), "n_equal_under_map": sum(1 for k_, v_ in mapped_.items() if n_.get(k_) == v_),
            "old_names_without_a_new_part": sorted(k_ for k_ in o_ if k_ + SUF not in n_), "new_names_without_an_old_part": sorted(k_ for k_ in n_ if k_[:-len(SUF)] not in o_),
            "names_with_a_different_digest": sorted(k_ for k_ in mapped_ if k_ in n_ and n_[k_] != mapped_[k_])}
    s_, v_ = L["stage1_new"], L["value_new"]  # the tie as VALUE main() computes it (identity_parts of the value inputs vs the Stage 1 record)
    both_ = set(s_) & set(v_)
    out["value_vs_stage1"] = {"n_stage1": len(s_), "n_value": len(v_), "n_union": len(set(s_) | set(v_)), "n_in_both": len(both_),
                              "n_equal": sum(1 for k_ in both_ if s_[k_] == v_[k_]), "in_both_differing": sorted(k_ for k_ in both_ if s_[k_] != v_[k_]),
                              "stage1_only": sorted(set(s_) - set(v_)), "value_only": sorted(set(v_) - set(s_))}
    out["value_vs_stage1"]["allowed_to_differ"] = sorted(out["value_vs_stage1"]["in_both_differing"] + out["value_vs_stage1"]["stage1_only"] + out["value_vs_stage1"]["value_only"])
    pb_ = subprocess.run(["git", "show", f"{PASSED[0]}:{PASSED[1]}"], capture_output=True, check=True).stdout
    pd_ = json.loads(pb_)
    hj_ = lambda x: hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()  # noqa: E731
    n1_ = json.loads(FILES["stage1_new"].read_bytes())
    out["passed_stage1"] = {"listing": f"{PASSED[1]} @ {PASSED[0]}", "file_sha256": hashlib.sha256(pb_).hexdigest(), "inputs_sha256": pd_["inputs_sha256"],
                            "has_identity_parts": "identity_parts" in pd_,
                            "inputs_sha256_equal": {k_: out["files"][k_]["inputs_sha256"] == pd_["inputs_sha256"] for k_ in ("stage1_blocked", "stage1_new")},
                            "stage1_new_fields_differing_from_passed": sorted(k_ for k_ in set(pd_) & set(n1_) if hj_(pd_[k_]) != hj_(n1_[k_])),
                            "stage1_new_fields_only_in_one": sorted(set(pd_) ^ set(n1_))}
    out["run"] = {"read_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S %Z"),
                  "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, check=True, text=True).stdout.strip(),
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    dest = HERE / "results" / f"g7b_identity_rename_check_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}.json"
    dest.open("x").write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    print("written", dest)


if __name__ == "__main__":
    main()
