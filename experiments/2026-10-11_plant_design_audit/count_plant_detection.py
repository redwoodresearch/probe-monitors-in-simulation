"""Counts only: how many planted checks each q3 result file records as detected or not, read from git refs (and one S3
file, sha256-checked). Reads only result JSON fields named "detected", "planted_detected", "constructible" and the
row-4 planted verdict; prints and writes counts, never any transcript or label text.

  python3 count_plant_detection.py --repo <repo> --out <results json>
"""

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

D = "experiments/2026-10-08_q3_onpolicy_figures/results"
SOURCES = {  # label: (ref, path) in git, or ("s3", uri, sha256)
    "stage-1 gate": ("origin/aswin/q3-sabotage-stage1-gate", "experiments/2026-10-11_sabotage_test/results/gate_result_2026-10-11_211446.json"),
    "stage-1 gate diagnostic (post hoc)": ("origin/aswin/q3-sabotage-stage1-gate", "experiments/2026-10-11_sabotage_test/results/gate_diag_POSTHOC_summary_2026-10-11_212359.json"),
    "generic probe part 1": ("s3", "s3://individual-aswin-surya-probe-shift-kill-2027-04-06/experiments/2026-10-11_generic_contrast_probe/results/part1_2026-10-11_212039.json",
                             "f0dbdbf392c6e2ec7840e8964b30340e34482d8b4949367b6377ab3fddcdb3a4"),
    "frozen family part 1": ("origin/aswin/q3-nabil-probes-plan", "experiments/2026-10-11_frozen_nabil_probes/results/part1_result_2026-10-11_212326.json"),
    "row 5 text classifier plants": ("origin/aswin/q3-text-classifier-catch", "experiments/2026-10-11_text_classifier_catch/results/scaleup_plants_2026-10-11_201235.json"),
    "row 6 catch by type": ("origin/aswin/q3-catch-by-command-type", "experiments/2026-10-11_catch_by_command_type/results/catch_by_type_2026-10-11_193855.json"),
    "row 4 tool-schema re-read": ("origin/aswin/q3-sim-toollist-swap", "experiments/2026-10-11_sim_toollist_swap/results/analysis_2026-10-11_203858.json"),
    "196-task 2x2 plants": ("origin/main", f"{D}/scaleup_plants_2026-10-10_164614.json"),
    "D196 plants": ("origin/main", f"{D}/D196_plants_2026-10-10_213831.json"),
    "Dcatch196": ("origin/main", f"{D}/Dcatch196_2026-10-10_215525.json"),
    "b honest (threshold in sim)": ("origin/main", f"{D}/b_honest_2026-10-10_234755.json"),
    "sim-honest calibration": ("origin/main", f"{D}/simhon_calibration_2026-10-10_234746.json"),
    "claim C (original)": ("origin/main", f"{D}/claim_c_POSTHOC_2026-10-10_225232.json"),
    "claim C content-only": ("origin/main", f"{D}/claim_c_content_POSTHOC_2026-10-11_000641.json"),
    "claim C first-k": ("origin/main", f"{D}/claim_c_firstk_POSTHOC_2026-10-10_233831.json"),
    "claim C no-agent": ("origin/main", f"{D}/claim_c_noagent_POSTHOC_2026-10-11_011248.json"),
    "claim C, honest_b swap (audit fix b)": ("origin/aswin/q3-sim-source-fixes", f"{D}/claim_c_POSTHOC_2026-10-11_211605.json"),
    "detector retrain": ("origin/main", f"{D}/retrain_2026-10-10_143949.json"),
    "retrain recalibrated": ("origin/main", f"{D}/retrain_recal_POSTHOC_2026-10-10_174409.json"),
    "retrain k=26": ("origin/aswin/q3-590-redo", f"{D}/retrain_k26_POSTHOC_2026-10-11_110203.json"),
    "k=26 extras (matched length)": ("origin/aswin/q3-590-redo", f"{D}/k26_extras_POSTHOC_2026-10-11_105644.json"),
    "D2": ("origin/aswin/q3-plans-d2-d3", f"{D}/d2_POSTHOC_2026-10-11_103358.json"),
    "D3 position": ("origin/aswin/q3-590-redo", f"{D}/d3_position_POSTHOC_2026-10-11_112843.json"),
    "D3 stripped": ("origin/aswin/q3-590-redo", f"{D}/d3_position_claimcnad3_POSTHOC_2026-10-11_120159.json"),
    "A1 run-level": ("origin/main", f"{D}/A1_runlevel_POSTHOC_2026-10-11_104814.json"),
    "A2 batches": ("origin/main", f"{D}/A2_batches_POSTHOC_2026-10-11_150751.json"),
    "late-run monitor": ("origin/main", f"{D}/late_run_monitor_POSTHOC_2026-10-11_115625.json"),
    "late-run false alarms": ("origin/main", f"{D}/late_run_false_alarms_POSTHOC_2026-10-11_120642.json"),
    "Phase 1 checks": ("origin/aswin/q3-phase1-checks", "experiments/2026-10-08_q3_swap_cue/results/phase1_checks_POSTHOC_2026-10-11_005336.json"),
    "fix939": ("origin/aswin/q3-phase1-checks", "experiments/2026-10-08_q3_swap_cue/results/fix939_POSTHOC_2026-10-11_020519.json"),
    "Q1 planted lines": ("origin/aswin/q3-q1-plants", "experiments/2026-10-10_q1_planted_checks/results/q1plants_verdicts_2026-10-10_213956.json"),
}
PAT = {k: re.compile(p) for k, p in {
    "detected_true": r'"detected": *true', "detected_false": r'"detected": *false',
    "planted_detected_true": r'"planted_detected": *true', "planted_detected_false": r'"planted_detected": *false',
    "not_constructible": r'"constructible": *false', "pass_true": r'"pass_[a-z_]+": *true', "pass_false": r'"pass_[a-z_]+": *false',
    "planted_explains": r'"verdict": *"explains the gap"'}.items()}


def text_of(repo, src):
    if src[0] == "s3":
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "f.json"
            subprocess.run(["aws", "s3", "cp", "--profile", "redwood-sandbox", "--only-show-errors", src[1], str(p)], check=True)
            b = p.read_bytes()
        if hashlib.sha256(b).hexdigest() != src[2]:
            raise SystemExit(f"sha256 mismatch for {src[1]}")
        return b.decode(), {"s3": src[1], "sha256": src[2]}
    ref, path = src
    commit = subprocess.run(["git", "-C", repo, "rev-parse", ref], capture_output=True, text=True, check=True).stdout.strip()
    b = subprocess.run(["git", "-C", repo, "show", f"{commit}:{path}"], capture_output=True, check=True).stdout
    return b.decode(), {"ref": ref, "commit": commit[:8], "path": path, "sha256": hashlib.sha256(b).hexdigest()}


def main(repo, out):
    res = {}
    for label, src in SOURCES.items():
        txt, where = text_of(repo, src)
        res[label] = {"source": where, **{k: len(p.findall(txt)) for k, p in PAT.items()}}
        print(label, {k: v for k, v in res[label].items() if k != "source" and v})
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    p = Path(out) / f"plant_detection_counts_{stamp}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as fh:  # one line per result file, so the record stays short
        fh.write('{"written": "%s", "counts": {\n' % stamp + ",\n".join(json.dumps(k) + ": " + json.dumps(v) for k, v in res.items()) + "\n}}\n")
    print("->", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    main(**vars(ap.parse_args()))
