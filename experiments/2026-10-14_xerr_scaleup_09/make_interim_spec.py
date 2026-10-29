"""Assemble the Append 21d INTERIM spec for run_xerr_scaleup.py on every released input (09, 2026-10-14 ~23:25 and ~23:55 PT).

S_new: Q1-Q4 run1 / run2 and honest calibration; sims Q1 s1+s2, Q2 (first follow-up, its 19 cut runs dropped by d9's exclusion
record), Q3 s1, Q4 s1+s2. S_newmains (secondary, interim only): nm360 S3 main, run2 try2, sim s1 and sim. Sources: Sol from 2e
(sol_r2/ where it exists, else sol/), p2 and famexp(_r2) from 4f, hygiene / labels / registry / staging records from d9's chain
branch, run tables pinned per file, d9's setup-failure record. Git inputs are pinned by sha256; Volume manifests by the sha256
of the copies in --dl (fetched with `modal volume get`). Keys and shas only; no score is read.

  python3 experiments/2026-10-14_xerr_scaleup_09/make_interim_spec.py --d9 <chain commit> --dl <dir> --out <spec.json> \\
      --runs36 <runs36.json>
"""

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

CHAIN = "experiments/2026-10-14_q3_overnight_chain"
RES, HYG = f"{CHAIN}/results", f"{CHAIN}/results/hygiene"
R36 = "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96"
RUN_TABLES = [  # (commit, path): the combined su311 final (09 ~00:2x) and nm360 S3; None = the --d9 commit
    ("806b7a63a6", f"{HYG}/run_table_311.json"),
    (None, f"{HYG}/run_table_nm360_s3.json"),
]
RT_CROSSCHECK = [  # per-quarter tables, which must agree with the combined final by key (in-job; any disagreement stops)
    ("65dcfec2f", f"{HYG}/run_table_su311_q1.json"),
    ("1407f6d66d", f"{HYG}/run_table_su311_q2.json"),
    ("bab8e5ba1", f"{HYG}/run_table_su311_q3.json"),
    ("007fb0482", f"{HYG}/run_table_su311_q4.json"),
]
LABELS_311 = (
    "806b7a63a6",
    f"{HYG}/attempt_labels_311.json",
)  # combined su311 final labels (Q2 rerun in; 19 cut keys out)
SETUP_FAILURES = ("b0176a82c", f"{HYG}/setup_failures_nm360_2026-10-14.json")
# folder -> (tool-error file stem, approach-provenance stems or None for honest folders)
Q_APPROACH = {
    1: ["su311_q1_s1", "su311_q1_s2"],
    2: ["su311_q2", "su311_q2r"],
    3: ["su311_q3_s1", "su311_q3_s2"],
    4: ["su311_q4_s1", "su311_q4_s2"],
}
FOLDERS = {}
for q in (1, 2, 3, 4):
    for arm, w in (("attack", "run1"), ("honest_a", "honest_a"), ("honest_b", "honest_b")):
        FOLDERS[f"su311_q{q}_main_v1__{arm}"] = (f"su311_q{q}_{w}", Q_APPROACH[q] if w == "run1" else None)
    FOLDERS[f"su311_q{q}_run2_v1__attack"] = (f"su311_q{q}_run2", Q_APPROACH[q])
for f, te, ap in (
    ("su311_q1_sim_s1_v1__sim_attack", "su311_q1fu_s1_sim", ["su311_q1_s1"]),
    ("su311_q1_sim_s2_v1__sim_attack", "su311_q1fu_s2_sim", ["su311_q1_s2"]),
    ("su311_q2_sim_v1__sim_attack", "su311_q2fu_sim", ["su311_q2"]),
    ("su311_q3_sim_s1_v1__sim_attack", "su311_q3fu_s1_sim", ["su311_q3_s1"]),
    ("su311_q4_sim_s1_v1__sim_attack", "su311_q4fu_s1_sim", ["su311_q4_s1"]),
    ("su311_q4_sim_s2_v1__sim_attack", "su311_q4fu_s2_sim", ["su311_q4_s2"]),
    ("su311_q3_sim_s2_v1__sim_attack", "su311_q3fu_s2_sim", ["su311_q3_s2"]),
    ("su311_q2r_sim_v1__sim_attack", "su311_q2rfu_sim", ["su311_q2r"]),
    ("nm360_s3_main_v1__attack", "nm360_nm360-s3_run1", ["nm360_s3_s1", "nm360_s3"]),
    ("nm360_s3_main_v1__honest_a", "nm360_nm360-s3_honest_a", None),
    ("nm360_s3_main_v1__honest_b", "nm360_nm360-s3_honest_b", None),
    ("nm360_s3_run2_v1_try2__attack", "nm360_nm360-s3_run2", ["nm360_s3_s1", "nm360_s3"]),
    ("nm360_s3_sim_s1_v1__sim_attack", "nm360_nm360-s3fu_s1_sim", ["nm360_s3_s1"]),
    ("nm360_s3_sim_v1__sim_attack", "nm360_nm360-s3fu_sim", ["nm360_s3"]),
):
    FOLDERS[f] = (te, ap)
SIM_MANIFESTS = [  # counted by d9; the unscored ones (Q2 rerun, Q3 s2) are listed as missing
    "su311_q1_sim_s1",
    "su311_q1_sim_s2",
    "su311_q2_sim",
    "su311_q2r_sim",
    "su311_q3_sim_s1",
    "su311_q3_sim_s2",
    "su311_q4_sim_s1",
    "su311_q4_sim_s2",
    "nm360_s3_sim_s1",
    "nm360_s3_sim",
]
LABELS_XC = [  # per-quarter su311 label files, cross-checked by key against the combined final (in-job)
    "su311_q1",
    "su311_q2",
    "su311_q3",
    "su311_q4",
    "su311_q1fu_s1",
    "su311_q1fu_s2",
    "su311_q2fu",
    "su311_q2rfu",
    "su311_q3fu_s1",
    "su311_q3fu_s2",
    "su311_q4fu_s1",
    "su311_q4fu_s2",
]
LABELS_NM = ["nm360_nm360-s3", "nm360_nm360-s3fu", "nm360_nm360-s3fu_s1"]


def git(*a, raw=False):
    out = subprocess.run(["git", *a], capture_output=True, check=True)
    return out.stdout if raw else out.stdout.decode().strip()


def show(commit, path):
    return git("show", f"{commit}:{path}", raw=True)


def pin(commit, path):
    return {"commit": commit, "path": path, "sha256": hashlib.sha256(show(commit, path)).hexdigest()}


def one(commit, folder, prefix):
    names = [
        n
        for n in git("ls-tree", "--name-only", f"{commit}:{folder}").split("\n")
        if n.startswith(prefix) and n.endswith(".json")
    ]
    if len(names) != 1:
        raise SystemExit(f"{folder}/{prefix}*: {len(names)} matches, expected 1: {names}")
    return f"{folder}/{names[0]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d9", required=True)
    ap.add_argument("--dl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs36", required=True)
    ap.add_argument("--final", action="store_true", help="FINAL 7b primary: S_old + S_new only (no nm360)")
    ap.add_argument("--parse-only", action="store_true", help="add parse_only (a TEST or final-mode parse)")
    a = ap.parse_args()
    keep = lambda name: not (a.final and ("nm360" in name or "newmains" in name))  # noqa: E731
    c9, dl = git("rev-parse", "--short=12", a.d9), Path(a.dl)
    regs = {st: pin(c9, f"{HYG}/registry_{st}.json") for st in ("S_new", "S_newmains") if keep(st.lower())}
    honest = {}  # target -> the registry's honest (main-target) manifest version
    for st, rp in regs.items():
        for e in json.loads(show(c9, rp["path"]))["entries"].values():
            for hm in e["honest_manifests"]:
                path, commit = [x.strip() for x in hm["manifest"].split("@")]
                p_ = {"commit": commit, "path": path, "sha256": hm["sha256"]}
                if pin(commit, path)["sha256"] != hm["sha256"] or honest.setdefault(path, p_) != p_:
                    raise SystemExit(f"registry {st}: honest manifest {path} does not reproduce, or two versions")
    d9f = lambda p_: p_ | {"keys_path": ["files"], "format": "d9_files"}  # noqa: E731
    mains = [f"{CHAIN}/config/manifest_su311_q{q}_main.json" for q in (1, 2, 3, 4)] + [
        f"{CHAIN}/config/manifest_nm360_s3_main.json"
    ]
    mains = [m for m in mains if keep(m)]
    counted = {w: [d9f(honest[m]) for m in mains] for w in ("run1", "honest_a", "honest_b")}
    counted["run2"] = [
        d9f(pin(c9, f"{CHAIN}/config/manifest_{t}.json"))
        for t in ["su311_q1_run2", "su311_q2_run2", "su311_q3_run2", "su311_q4_run2", "nm360_s3_run2"]
        if keep(t)
    ]
    counted["sim"] = []
    for t in filter(keep, SIM_MANIFESTS):
        e = d9f(pin(c9, f"{CHAIN}/config/manifest_{t}.json"))
        if (
            t == "su311_q2_sim"
        ):  # d9's record of the 19 cut first-follow-up sims (rerun in q2r_fu): dropped before counting
            e["exclusions"] = pin(c9, f"{HYG}/exclusions_su311_q2fu.json")
        counted["sim"].append(e)
    rows = {}  # eval key -> world|cell over every counted manifest's rows
    for ents in counted.values():
        for e in ents:
            for r in json.loads(show(e["commit"], e["path"]))["files"]:
                k, v = f"{r['launch']}/{r['rel']}#e0", f"{r['world']}|{r['task']}"
                if rows.setdefault(k, v) != v:
                    raise SystemExit(f"{k}: two world|cell values")
    arms, exp_arms, targets = {}, {}, []
    not_published = {}
    for f, (te, approach) in FOLDERS.items():
        sh = lambda n: hashlib.sha256((dl / n).read_bytes()).hexdigest()  # noqa: E731
        if not keep(f):
            continue
        miss_ = [
            x
            for x, ok in (
                ("p2 (4f)", (dl / f"{f}.p2.json").exists()),
                ("Sol (2e)", any((dl / f"{f}.{d}.json").exists() for d in ("sol", "sol_r2"))),
                ("famexp_r2 (4f)", any((dl / f"{f}.{d}.json").exists() for d in ("famexp", "famexp_r2"))),
            )
            if not ok
        ]
        if miss_:
            not_published[f] = miss_
        if "p2 (4f)" in miss_ or "Sol (2e)" in miss_:  # its counted runs stay counted and are listed as missing scores
            continue
        p2 = json.loads((dl / f"{f}.p2.json").read_text())
        (st_,) = [x for x in p2["dump"]["staged"] if x["folder"] == p2["folder"]]
        root = st_["run"]  # the extraction root this folder's p2 read
        target = re.sub(r"_extract_v1(_try[0-9]+)?$", "", root)
        if target not in targets:
            targets.append(target)
        arm = f.split("__", 1)[1]
        sd = "sol_r2" if (dl / f"{f}.sol_r2.json").exists() else "sol"
        fd = "famexp_r2" if (dl / f"{f}.famexp_r2.json").exists() else "famexp"
        h = {
            "longest_epoch_h200_gate": {
                "extract": [
                    pin(c9, f"{RES}/{n}")
                    for n in sorted(
                        n
                        for n in git("ls-tree", "--name-only", f"{c9}:{RES}").split("\n")
                        if n.startswith(f"extract_chain_{root}_a")
                        and re.fullmatch(rf"extract_chain_{root}_a[0-9]+_.*\.json", n)
                    )
                ],
                "compare": pin(c9, one(c9, RES, f"compare_repro_{root}_repro-c_")),
                "staging": pin(c9, one(c9, RES, f"staged_stats_{target}_2026")),
            },
            "corrected_render": pin(
                c9, one(c9, RES, f"format_check_{target}_{'sim_attack@september' if arm == 'sim_attack' else arm}_2026")
            ),
            "tool_error_table": pin(c9, f"{HYG}/toolerr_{te}.json"),
        }
        if approach:
            h["approach_system_prompt_check"] = [pin(c9, f"{HYG}/approach_provenance_{x}.json") for x in approach]
        arms[f] = {
            "p2_manifest": {"path": f"/work/scaleup/{f}/p2/manifest_p2.json", "sha256": sh(f"{f}.p2.json")},
            "sol_manifest": {"path": f"/work/scaleup/{f}/{sd}/manifest_sol.json", "sha256": sh(f"{f}.{sd}.json")},
            "hygiene": h,
        }
        if (dl / f"{f}.{fd}.json").exists():
            exp_arms[f] = {
                "manifest": {"path": f"/work/scaleup/{f}/{fd}/manifest_famexp.json", "sha256": sh(f"{f}.{fd}.json")}
            }
    targets += [t for t in filter(keep, SIM_MANIFESTS) if t not in targets]  # counted sims without a scored folder
    stex = []  # staging exclusions: d9's c_gate.excluded eval keys -> world|cell through the counted manifests' rows
    for t in targets:
        rec = pin(c9, one(c9, RES, f"chain_stage_{t}_2026"))
        ex = {}
        for set_name, v in json.loads(show(c9, rec["path"])).items():
            if set_name == "git":
                continue
            for k, why in ((v.get("c_gate") or {}).get("excluded") or {}).items():
                if k not in rows:
                    raise SystemExit(f"{t}: excluded key {k} is in no counted manifest")
                ex[rows[k]] = why
        stex.append({"record": rec, "exclusions": ex, "for_manifest": f"{CHAIN}/config/manifest_{t}.json"})
    vm = {f"own_frac{x}_L{y}": "own_a_adapted" for x in (10, 25, 100) for y in (16, 32, 48)} | {
        f"ad_k{k}": "own_a_adapted" for k in (5, 10, 20)
    }
    r36 = {"local_path": a.runs36, "sha256": R36}
    spec = {
        "mode": "final" if a.final else "interim",
        **({} if a.final else {"snapshot_index": 1}),
        "cutoff_time": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S PT"),
        "note": "Append 21d interim snapshot 1 on every released input (09 ~23:25, ~23:50, ~23:55): S_new Q1-Q4 run1/run2/honest, "
        "sims Q1 s1+s2, Q2 (cut runs dropped; rerun extraction pending), Q3 s1, Q4 s1+s2; S_newmains nm360 S3 (secondary)",
        "counted": counted,
        "labels": [pin(*LABELS_311)] + [pin(c9, f"{HYG}/labels_{x}.json") for x in LABELS_NM if keep(x)],
        "labels_crosscheck": [pin(c9, f"{HYG}/labels_{x}.json") for x in LABELS_XC],
        "run_table": [pin(c or c9, p) for c, p in RUN_TABLES if keep(p)],
        "run_table_crosscheck": [pin(c, p) for c, p in RT_CROSSCHECK],
        "registry": regs,
        "staging_exclusions": stex,
        "setup_failure_exclusions": [pin(*SETUP_FAILURES)],
        "q2_rerun_audit": pin(
            "e0dec1714", "experiments/2026-10-14_q3_scaleup_311/results/q2_rerun_audit_2026-10-14.json"
        ),
        "q2_rerun_run_keys": pin(
            "e0dec1714", "experiments/2026-10-14_q3_scaleup_311/results/q2r_fu_run_keys_2026-10-14.json"
        ),
        "fold_maps": {
            "sol": r36,
            "own_a_adapted": dict(r36),
            "b_asis": [
                dict(r36, row_filter={"arm": "sim-attack"}),
                pin(
                    "573054a4d",
                    "experiments/2026-10-06_q3_attack_detector/results/sim_honest_scores_2026-10-07_124717/runs.json",
                ),
            ],
        },
        "expansion": {
            "variant_fold_map": vm | {"b_asis": "b_asis"},
            "nofold_exclusion_record": pin(
                "548525c3c",
                "experiments/2026-10-14_p2_family_expansion/results/nofold_exclusion_2026-10-14_204632.json",
            ),
            "v7_phase1": pin(
                "6514cf401", "experiments/2026-10-14_xerr_prelim_09/results/xerr_v7_phase1_2026-10-14_180820.json"
            ),
            "t196_open": {
                "manifest": {
                    "path": "/p2/famexp/2026-10-14_163027/pack_open_2026-10-14_180501/manifest.json",
                    "sha256": "a0765577c34182c48f7827492f9b745c76201da57c03088686d7f2f434c01e9f",
                }
            },
            "t196_sealed": {
                "manifest": {
                    "path": "/p2/famexp/2026-10-14_163027/pack_sim_sealed_2026-10-14_181123/manifest.json",
                    "sha256": "6795cba0598256e92cd93435216edf45c31b865919d553a8282e9bcbafe79ea6",
                }
            },
            "arms": exp_arms,
        },
        "q4_h200": {
            "attestation": pin("3172e9c88", f"{RES}/q4_attest/q4_h200_attestation_2026-10-14.json"),
            "coverage_check": pin("96c716806", "experiments/2026-10-14_priority1_spec_09/q4_coverage_check/CHECK.md"),
        },
        "arms": arms,
        "d9_commit": c9,
        "not_published_at_build": not_published,
        **({"parse_only": True} if a.parse_only else {}),
    }
    if a.final:
        spec["note"] = "FINAL 7b primary (S_old + S_new; S_newmains out): every released S_new input, d9 @ " + c9
    Path(a.out).write_text(json.dumps(spec, indent=1))
    print(
        f"{a.out}: {len(arms)} folders, {sum(len(v) for v in counted.values())} counted manifests, "
        f"{sum(len(x['exclusions']) for x in stex)} staging exclusions over {len(targets)} targets, d9 @ {c9}"
    )


if __name__ == "__main__":
    main()
