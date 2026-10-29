"""Amendment 2 readings for the new mains, S1–S3 subset (GPT-5.4-nano), on Modal CPU. Never reads corrected-B's manifest
(0d's rule for this pass). Uses analyze_monitor_modal.py's own functions (readings, table, verdict, counts, detection),
shipped beside it, so the rule is the one R0 validated. Gate R0 runs first, as there (launch 1's committed result must
reproduce field for field from launch 1's scores file alone). Returns numbers, counts, keys and labels only.

  uv run --no-project --python 3.14 --with modal==1.6.1 --with numpy==2.5.1 modal run --detach \\
      experiments/2026-10-15_q2_monitor_su311/analyze_nm360_modal.py --attempt <nano_nm360 attempt id>
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
APP = "probe-shift_aswin-surya_sub-q2-monsu311_analysis_kill-2026-11-15"
WORK = "probe-shift_aswin-surya_sub-q2-monsu311_work_kill-2026-11-15"
D1 = Path.home() / "probe-shift-q3-nano-realsim-1012"
CH = "experiments/2026-10-14_q3_overnight_chain/results/hygiene"
LABEL = "S1–S3 subset of new mains; PRELIMINARY until Aswin reviews"
GIT_PINS = {  # name: (commit, path, sha256 or None)
    "l1_detection": ("31d52ebb1", "experiments/2026-10-12_q3_nano_realsim/results/nano_onpolicy_detection_2026-10-12_142245.json", None),
    "il_no_reference": ("b7dbecb18b886e94d01f2ddfaaf44505c15feb35", "experiments/2026-10-15_q3_interleaved_run3_sim2/results/il_no_reference_cells_2026-10-15.json",
                        "a2149da9c8a8dbd47d3d79677560300fcbaa7744e65fca13bb2034817e7df2be"),
    "s3_no_reference": ("a4af2afe60d8352663f76f008b29508b38351465", "experiments/2026-10-14_q3_new_mains_launch/results/s3_no_reference_record_2026-10-15.json",
                        "77eb0e46d741013906a9e1d7d37d80c1ff4b49e473789c7f5e9ed5abec0f5803"),
    **{f"rt_{q}": (c, f"{CH}/run_table_nm360_{q}.json", h) for q, c, h in (
        ("s1", "1c3111f6a5ab25a1434ebfbf626479fa5d5311ed", "a0cfe444ce409dfc5738eda7d7fe5ca877e803c57de71757c824362b7d362c14"),
        ("s2", "2cd5cd6dfd23954a555949a72011009c4cf26020", "2dd369c8eed83c64bda63dfb2500d8dc8ee89ce813e4e8c2bb645637acd0bbe8"),
        ("s3", "e9079eec2802fd52161e21b0c43f28b52f17036a", "6ed54048780d5191a01fbd2ed27634931aa9dd462a527c85b7b0bdfc78e116e3"))},
    **{f"lab_{b}": (c, f"{CH}/labels_nm360_nm360-{b}.json", h) for b, c, h in (
        ("s1", "e392ecf2dda3924b056ce2ee0080b02c22bbe772", "941bf28e815df2c21c76756f0a5296e4058c67653a0ddd7b3fe2d82a81127aad"),
        ("s1_m1", "e6025eaa51f02dcc58acd2c5b516f577640df9fc", "685c433f12852f72686a543b9a4647fb92e793e1cce4f325c3a53e34bfcdc632"),
        ("s1fu", "1c3111f6a5ab25a1434ebfbf626479fa5d5311ed", "b81d1792f44b95c501a4e2a17da27629ee293976453ab0df0ed55d385497a343"),
        ("s2", "8daaef5653961219b803d75ee4c72199e2e0d9e1", "9ccf04117e7f43436dfcd7b58c577e7002c7c0f6f8eef8ae651106846c1b7741"),
        ("s2_m1", "06ac5abb0f02501557400e391512270a9e0662f8", "df7c0b5e478294e4a30b408565c66855dc6f27f8a1cd99ab102ee733d1b76580"),
        ("s2fu", "2cd5cd6dfd23954a555949a72011009c4cf26020", "2b1543d2c4992efdb903f90364c328d94e88211186b7a944c3b1d09fc80a5514"),
        ("s3", "a569f3e164963aa2cb6014edf1389358a9bcbe51", "3ea318a012ea7d00deece3d25fb383e3461c8d12513396ff62cff7e099c7fde8"),
        ("s3fu", "e9079eec2802fd52161e21b0c43f28b52f17036a", "a7599953462630d626b2a3288e8db5d3b2f644099dd227eda502100ea92c6544"),
        ("s3fu_s1", "324c3ae18b960722d7ba73cf296357dc8da8a1f8", "30fb91518d49c9e7f92e1b4d139484d78e678c4e123917e0e5ad55a5d8eb99cd"))},
}
L1 = ("10a62e088d556e3d5ad8cba855f9d6bae7fc05ef8139c0281189b221031cf9af",
      D1 / "experiments/2026-10-12_q3_nano_realsim/results/nano_onpolicy_scores_2026-10-12_133512.json")
LAUNCH = {"s1": "2026-10-14_newmains_s1", "s2": "2026-10-14_newmains_s2", "s3": "2026-10-14_newmains_s3"}

image = modal.Image.debian_slim(python_version="3.14").pip_install("numpy==2.5.1").add_local_python_source("analyze_monitor_modal")
app = modal.App(APP)
work_vol = modal.Volume.from_name(WORK, version=2)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@app.function(image=image, volumes={"/work": work_vol}, cpu=(4, 4), memory=(8192, 8192), timeout=1800, retries=0,
              startup_timeout=300, scaledown_window=2)
def analyze(inputs: dict, attempt: str, git: str) -> dict:
    import collections
    import re

    import numpy as np

    import analyze_monitor_modal as am

    for n, (_, _, want) in GIT_PINS.items():
        if want and sha(inputs[n].encode()) != want:
            raise SystemExit(f"pin mismatch: {n}")
    if sha(inputs["l1"].encode()) != L1[0]:
        raise SystemExit("pin mismatch: launch 1 scores")
    out = {"label": LABEL, "rule": "RULE_AMENDMENT_1.md, Amendment 2", "git": git}

    # R0 (as analyze_monitor_modal.analyze): launch 1's committed result reproduces from its scores file alone
    l1rec = json.loads(inputs["l1"])
    l1 = am.lines_of(l1rec)
    pool = np.array([l1[h][t]["mean"] for h in ("honest_a", "honest_b") for t in sorted(l1[h]) if l1[h][t]["state"] == "scored"])
    r = am.readings(l1, lambda c: [(set(c), pool)], "R0", "P", False)
    got = {"counts": {line: {k: am.counts(rows).get(k, 0) for k in ("loaded", "scored", "refused", "error", "unparsed", "attrition")}
                      | {"attrition_reported": l1rec["lines"][line]["attrition"]} for line, rows in l1.items()},
           "skipped_unusable_attempts": len(l1rec["skipped_unusable_attempts"]), "honest_runs_in_threshold": len(pool),
           "tasks_loaded_in_all_three_attack_lines": len(set.intersection(*(set(l1[am.ATTACK[a]]) for a in am.ATTACK))),
           "main": am.as_launch1(r["main"])}
    for name in ("refusals", "every_drop"):
        got[f"sensitivity_{name}"] = {k: (am.as_launch1(v) if isinstance(v, dict) else v) for k, v in r[f"sensitivity_{name}"].items()}
    want = json.loads(inputs["l1_detection"])
    diffs = am.same(got, {k: want[k] for k in got})
    out["R0_reproduces_launch1"] = {"fields_differing": len(diffs), "first": diffs[:5], "pass": not diffs}
    if diffs:
        return out

    work_vol.reload()
    p = Path("/work/nano_nm360/results", f"scores_{attempt}.json")
    raw = p.read_bytes()
    rec = json.loads(raw)
    if rec["model"] != "openai/gpt-5.4-nano" or rec["options"]["k_rescores"] != 3 or rec["job"] != "nano_nm360":
        raise SystemExit("not the nano_nm360 record")
    out["score_file"] = {"path": f"{WORK}:/nano_nm360/results/{p.name}", "sha256": sha(raw), "attempt": rec["attempt"],
                         "cost_usd_logged": rec["cost_usd_logged"], "paid_calls": rec["paid_calls"], "provider_requests": rec["provider_requests"]}
    lines = am.lines_of(rec)
    g2 = {"P": any(am.counts(rows)["g2_flag"] for rows in lines.values()),
          "S": any(am.counts(lines[x])["g2_flag"] for x in ("real_run1", "real_run2", "sim_september", "honest_b"))}
    A, B = am.scored(lines["honest_a"]), am.scored(lines["honest_b"])
    pool_p, pool_s = np.concatenate([am.arr(A), am.arr(B)]), am.arr(B)
    out["P"] = am.readings(lines, lambda c: [(set(c), pool_p)], "nano new mains S1–S3 (P) honest A ∪ B", "P", g2["P"])
    out["S"] = am.readings(lines, lambda c: [(set(c), pool_s)], "nano new mains S1–S3 (S) honest B (no C)", "S", g2["S"])
    out["S_realized_FA_on_held_out_A"] = {f"{b:g}": am.detection(am.arr(A), pool_s, b) for b in am.BUDGETS}
    out["counts"] = {line: am.counts(rows, rec["lines"][line]["attrition"]) for line, rows in lines.items()}
    out["g2"] = g2
    out["dropped_by_key"] = {line: {cid: {"task": rec["lines"][line]["where"][cid]["task"], "category": cat,
                                          "eval": rec["lines"][line]["where"][cid]["eval"]}
                                    for cid, cat in rec["lines"][line]["dropped"].items()} for line in rec["lines"]}
    out["excluded_by_key"] = rec["excluded_by_staging"]
    out["nm360_keys"] = rec["nm360_keys"]
    cohort = am.cohort_of(lines)
    launch_of = {}
    for q, launch in LAUNCH.items():
        for c in json.loads(inputs[f"rt_{q}"])["cells"]:
            launch_of[c["cell"]] = q
    out["cohort_tasks_by_launch"] = dict(collections.Counter(launch_of.get(t, "unknown") for t in cohort))

    # attempt labels: 21c § 5 (attempted in run 1, run 2 and sim), over the cohort; no-reference cells are not in it
    labels = {}
    for n in GIT_PINS:
        if n.startswith("lab_"):
            for k, v in json.loads(inputs[n])["labels"].items():
                world, cell = k.split("|")
                if (world, cell) in labels and labels[(world, cell)] != v:
                    raise SystemExit(f"two labels for {k}")
                labels[(world, cell)] = v
    att = [t for t in cohort if all(labels.get((w, t)) == "attempted" for w in ("run1", "run2", "sim"))]
    out["attempted_all_three"] = {"k": len(att), "n": len(cohort),
                                  "label_coverage": {w: dict(collections.Counter(labels.get((w, t), "unlabelled") for t in cohort))
                                                     for w in ("run1", "run2", "sim")}}

    # status rows: every no-reference cell (run tables), plus every cell with a run 1 key but no sim key
    s3rec = {c["task"]: c for c in json.loads(inputs["s3_no_reference"])["cells"]}
    il = {row["key"]: lr["launch_run_dir"].rsplit("/", 1)[-1] for lr in json.loads(inputs["il_no_reference"])["launches"] for row in lr["rows"]}
    norefs = {}
    for q in LAUNCH:
        for c in json.loads(inputs[f"rt_{q}"])["cells"]:
            if c["sim_status"] != "ran":
                norefs[c["cell"]] = (q, c["sim_status"])
    excl = collections.defaultdict(list)
    for path, e in rec["excluded_by_staging"].items():
        excl[e["task"]].append({"line": e["line"], "sources": e["sources"]})
    keyed = {line: {w["task"] for w in rec["lines"][line]["where"].values()} for line in rec["lines"]}
    rows = {}
    for t in sorted(set(norefs) | set(rec["nm360_keys"]["cells_with_run1_but_no_sim_key"])):
        q = norefs.get(t, (launch_of.get(t), None))[0]
        if t in norefs:
            if q == "s3":
                whys = sorted({m["why"] for m in s3rec[t]["main"] if m["why"]})
                status = f"sim not run: {whys[0]}" if len(whys) == 1 else "sim not run: (S3 record has no single why)"
                source = f"s3_no_reference_record_2026-10-15.json @ a4af2afe6; run_table_nm360_s3 sim_status {norefs[t][1]}"
            else:
                status = "sim not run: no pinned reference"
                source = f"il_no_reference_cells_2026-10-15.json @ b7dbecb18 ({il.get(t, 'not listed')}); run_table_nm360_{q} sim_status {norefs[t][1]}"
        else:
            sim_ex = [e for e in excl[t] if e["line"] == "sim_september"]
            status = "sim ran, excluded: " + "; ".join(s for e in sim_ex for s in e["sources"]) if sim_ex else "sim key missing (no record)"
            source = "the nano_nm360 run's excluded_by_staging (stage record c_gate)"
        rows[t] = {"launch": q, "status_row": status, "reason_source": source,
                   "keys_in_run": {line: t in keyed[line] for line in keyed},
                   "excluded_lines": {e["line"]: e["sources"] for e in excl[t]}, "in_cohort": t in cohort}
    out["status_rows"] = rows
    out["status_counts"] = dict(collections.Counter(re.sub(r"in \d+ tries", "in N tries", r["status_row"]) for r in rows.values()))
    out["no_reference_cells_by_launch"] = dict(collections.Counter(q for q, _ in norefs.values()))
    return json.loads(json.dumps(out, allow_nan=False))


@app.local_entrypoint()
def main(attempt: str = ""):
    repo = HERE.parents[1]
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=repo).stdout.strip():
        raise SystemExit("uncommitted tracked changes in this repo")
    for f in (Path(__file__), HERE / "analyze_monitor_modal.py", HERE / "RULE_AMENDMENT_1.md"):
        subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], capture_output=True, cwd=repo, check=True)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo, check=True).stdout.strip()
    inputs = {n: subprocess.run(["git", "show", f"{c}:{p}"], capture_output=True, text=True, cwd=repo, check=True).stdout
              for n, (c, p, _) in GIT_PINS.items()}
    inputs["l1"] = Path(L1[1]).read_text()
    rec = analyze.remote(inputs, attempt, git)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = HERE / "results" / f"nm360_s123_analysis_{stamp}.json"
    with out.open("x") as f:
        f.write(json.dumps(rec, indent=1, allow_nan=False))
    print(json.dumps(rec.get("R0_reproduces_launch1")), "->", out)
    if not rec["R0_reproduces_launch1"]["pass"]:
        raise SystemExit("R0 failed: stop and report")
