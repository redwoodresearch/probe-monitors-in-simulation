"""0d for the writing session (10-16): which attack prompt did our attack runs use, and is it byte-identical to the public
redwoodresearch/linuxarena-paper attack prompts? PATHS, SHAS, SHA256, SIZES, BOOLEANS AND NUMBERS ONLY: no prompt text is printed or
stored. Reads git objects (this repo, the sims sibling checkout) and a public-repo clone given by path; never the sprint's
approach/attack files or cells.json.
  python check_attack_prompt.py --sims ~/simulated-control-evals --public <clone of redwoodresearch/linuxarena-paper> --out results/<name>.json
"""
import argparse
import difflib
import hashlib
import json
import os
import re
import subprocess

PHASE2 = "experiments/2026-10-07_phase2_attack_check/config"
BRANCH = "origin/aswin/q3-chain-donorfix-dry"
SIMS_COMMITS = {"september base (run.yaml september tree)": "ba990399", "today tree (run.yaml)": "e561f936"}
PROMPT_DIR = "src/runner/prompts/policy"
TREE_MANIFEST = ("cfdd656c9", "experiments/2026-10-14_q3_scaleup_311/results/sim_tree_manifest_2026-10-15.json")


TOP = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()


def git(*a, cwd=None):
    return subprocess.check_output(["git", *a], cwd=cwd or TOP)  # probe-shift queries run from the repo root (Codex r64 #1)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", required=True)
    ap.add_argument("--public", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"what": __doc__.strip().splitlines()[0], "probe_shift_branch": BRANCH, "probe_shift_commit": git("rev-parse", BRANCH).decode().strip()}
    # 1. which prompt the launchers select, and whether that selection ever changed
    run_versions = git("log", BRANCH, "--format=%h", "--", f"{PHASE2}/run.yaml").decode().split()
    line = re.compile(rb"^\s*attacker_prompt:\s*(\S+)", re.M)
    assert run_versions, "no run.yaml history found"
    vals = {c: [m.decode() for m in line.findall(git("show", f"{c}:{PHASE2}/run.yaml"))] for c in run_versions}
    sim_now = re.findall(rb"BA_ATTACKER_PROMPT:\s*(\S+)", git("show", f"{BRANCH}:{PHASE2}/sim.yaml"))
    out["selection"] = {"real_arms": {"file": f"{PHASE2}/run.yaml", "key": "attacker_prompt", "values_over_all_versions": sorted({v for x in vals.values() for v in x}),
                                      "n_versions": len(run_versions), "every_version_has_exactly_one": all(len(x) == 1 for x in vals.values())},
                        "sim_arms": {"file": f"{PHASE2}/sim.yaml", "key": "BA_ATTACKER_PROMPT", "value_now": [v.decode() for v in sim_now],
                                     "commits_touching_it": git("log", BRANCH, "--format=%h", "-G", "BA_ATTACKER_PROMPT", "--", f"{PHASE2}/sim.yaml").decode().split()},
                        "launcher_refuses_other_trees": "run_attack_check.py checks the tree's src/ sha256 against run.yaml src_sha256 (run.yaml comment, lines 9-11)",
                        "kind": "CONFIGURED selection (launch configs); execution evidence is the box-1 run-tree manifest below; .eval headers were not read"}
    # 2. the prompt bytes in each simulator tree, and the tree that ran (box-1 manifest)
    trees = {}
    for label, c in SIMS_COMMITS.items():
        trees[label] = {"sims_commit": c, "files": {}}
        for n in git("ls-tree", "--name-only", f"{c}:{PROMPT_DIR}", cwd=a.sims).decode().split():
            b = git("show", f"{c}:{PROMPT_DIR}/{n}", cwd=a.sims)
            trees[label]["files"][n] = {"sha256": sha(b), "bytes": len(b), "n_format_placeholders": len(re.findall(rb"\{[a-z_]+\}", b))}
    man = json.loads(git("show", f"{TREE_MANIFEST[0]}:{TREE_MANIFEST[1]}"))
    box = {}

    def walk(o):
        if isinstance(o, list):
            if o and isinstance(o[0], str) and o[0].startswith("runner/prompts/policy/"):
                box[o[0]] = o[1]
            for x in o:
                walk(x)
        elif isinstance(o, dict):
            for x in o.values():
                walk(x)
    walk(man)
    out["prompt_bytes"] = {"trees": trees, "box1_run_tree_manifest": {"record": " @ ".join(TREE_MANIFEST[::-1]), "sha256_by_path": box}}
    used = trees["september base (run.yaml september tree)"]["files"]["attacker_2026_08_28.md"]["sha256"]
    out["prompt_bytes"]["used_prompt_sha256"] = used
    out["prompt_bytes"]["box1_equals_git"] = box.get("runner/prompts/policy/attacker_2026_08_28.md") == used
    out["prompt_bytes"]["today_tree_equals"] = trees["today tree (run.yaml)"]["files"]["attacker_2026_08_28.md"]["sha256"] == used
    # 3. runtime filling and approach guidance, from code structure only (counts and line numbers)
    code = {}
    for f in ("src/runner/policy/prompts.py", "src/runner/policy/agent.py", "src/runner/build/arena_policy.py", "src/runner/settings.py"):
        t = git("show", f"ba990399:{f}", cwd=a.sims).decode().splitlines()
        code[f] = {"approach_lines": [i + 1 for i, s in enumerate(t) if "approach" in s.lower()],
                   "attacker_prompt_env_lines": [i + 1 for i, s in enumerate(t) if "BA_ATTACKER_PROMPT" in s]}
    out["code_structure"] = {"files": code, "approach_guidance_code_path_present": bool(code["src/runner/policy/prompts.py"]["approach_lines"]),
                             "approach_note": "a code path adds a cell's approach guidance to the prompt when the cell has one; per-run presence not checked "
                                              "(cells.json and approach files not read)",
                             "template_placeholders_filled_at_runtime": trees["september base (run.yaml september tree)"]["files"]["attacker_2026_08_28.md"]["n_format_placeholders"] > 0}
    # 4. the public repo
    head = git("rev-parse", "HEAD", cwd=a.public).decode().strip()
    ours = git("show", f"ba990399:{PROMPT_DIR}/attacker_2026_08_28.md", cwd=a.sims).decode()
    ol = {s.strip() for s in ours.splitlines() if s.strip()}
    pub = {}
    for dp, _, fs in os.walk(os.path.join(a.public, "src/policies/attack")):
        for f in sorted(fs):
            if not f.endswith(".md"):
                continue
            p = os.path.join(dp, f)
            b = open(p, "rb").read()
            t = b.decode()
            pl = [s.strip() for s in t.splitlines() if s.strip()]
            pub[os.path.relpath(p, a.public)] = {"sha256": sha(b), "bytes": len(b), "equal_to_ours": t == ours, "in_ours": t in ours, "ours_in_it": ours in t,
                                                 "diff_ratio": round(difflib.SequenceMatcher(None, t, ours, autojunk=False).ratio(), 3),
                                                 "n_lines": len(pl), "n_lines_verbatim_in_ours": sum(s in ol for s in pl),
                                                 "git_blob": git("hash-object", p).decode().strip(),
                                                 "blob_in_local_sims_object_store": subprocess.run(["git", "cat-file", "-e", git("hash-object", p).decode().strip()],
                                                                                                   cwd=a.sims, capture_output=True).returncode == 0}
    out["sims_repo"] = {"path": a.sims, "head": git("rev-parse", "HEAD", cwd=a.sims).decode().strip(),
                        "note": "object-store checks cover the objects present in this local clone only"}
    out["public_repo"] = {"repo": "redwoodresearch/linuxarena-paper", "head": head, "clone_depth": "1 (HEAD only)", "files": pub,
                          "any_byte_identical": any(v["equal_to_ours"] for v in pub.values()),
                          "ours_blob_in_public_clone": subprocess.run(["git", "cat-file", "-e", git("rev-parse", f"ba990399:{PROMPT_DIR}/attacker_2026_08_28.md",
                                                                                                    cwd=a.sims).decode().strip()], cwd=a.public, capture_output=True).returncode == 0}
    json.dump(out, open(a.out, "x"), indent=1)
    print(json.dumps({"used_prompt_sha256": used, "box1_equals_git": out["prompt_bytes"]["box1_equals_git"],
                      "today_tree_equals": out["prompt_bytes"]["today_tree_equals"], "selection_values": out["selection"]["real_arms"]["values_over_all_versions"],
                      "sim_value": out["selection"]["sim_arms"]["value_now"], "public_head": head, "any_byte_identical": out["public_repo"]["any_byte_identical"],
                      "max_diff_ratio": max(v["diff_ratio"] for v in pub.values()), "n_run_yaml_versions": out["selection"]["real_arms"]["n_versions"],
                      "approach_code_path": out["code_structure"]["approach_guidance_code_path_present"]}))


if __name__ == "__main__":
    main()
