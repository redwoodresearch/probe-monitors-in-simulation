"""Cell (iv), the grading gap of the fake/fake cell, layer 48, catch at 1% false alarms: how much grading the
fake/fake-trained probe on simulated attacks overstates its catch of real second-epoch attacks (fake catch − real catch).
Claimed: Qwen3-32B +2.3 [−0.4, +5.3] (Amendment 21f, df's full + F9 averaged pack) and Seed-OSS-36B 0.0 [−2.7, +2.7]
(Amendment 21g, fix-full, averaged readout).

Sources (git, sha256-checked; branch aswin/q3-sce-reply-probe, folder experiments/2026-10-13_q3_sce_reply_probe/results/):
- Qwen3-32B: sce_negcount_df_full_2026-10-13_221309.json @ facba7aa;
- Seed-OSS-36B: sce_negcount_seed_2026-10-13_222537.json @ 052dcbac.
Run from the repo after `git fetch origin aswin/q3-sce-reply-probe`: python3 -I cell_iv_grading_gaps.py
"""

import hashlib
import json
import subprocess

R = "experiments/2026-10-13_q3_sce_reply_probe/results"
QWEN = ("facba7aa303663b814bd902f44c17da2ab330b0b", f"{R}/sce_negcount_df_full_2026-10-13_221309.json",
        "6c00bd1833cff8a23cce2ae6cb8fd09e4c64288de187ce7dc33cc28b9651ed73")
SEED = ("052dcbac5d23d175b477318f01c3aeebdca27ddd", f"{R}/sce_negcount_seed_2026-10-13_222537.json",
        "0ef2b174dfe8d1274161bf2efa67312d6e8a451cd3cbad616fc9d9db2a7e6616")


def git_json(commit, path, sha):
    b = subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True).stdout
    if hashlib.sha256(b).hexdigest() != sha:
        raise SystemExit(f"sha256 mismatch: {commit[:8]}:{path}")
    return json.loads(b)


def pts(x):
    return f"{100 * x:+.1f}"


def main():
    q, s = git_json(*QWEN), git_json(*SEED)
    print("sha256 OK for both files (values stored as fractions; printed in points)\n")
    r = q["results"]["fpr_0.01"]
    g = r["b_F9_overstatement"]
    print(f"Qwen3-32B   {QWEN[1]} @ {QWEN[0][:8]}\n  label: {q['label']}")
    print(f"  results/fpr_0.01/b_F9_overstatement/point, /ci95: {pts(g['point'])} [{pts(g['ci95'][0])}, {pts(g['ci95'][1])}]")
    print(f"  check: b_F9_fake_catch {100 * r['b_F9_fake_catch']:.1f} − b_F9_e2 {100 * r['b_F9_e2']:.1f}"
          f" = {pts(r['b_F9_fake_catch'] - r['b_F9_e2'])}")
    r = s["results"]["seed_ff|L48|fpr_0.01"]
    g = r["gap"]
    print(f"\nSeed-OSS-36B   {SEED[1]} @ {SEED[0][:8]}\n  label: {s['label']}\n  cell: {r['cell']}")
    print(f"  results/seed_ff|L48|fpr_0.01/gap/point, /ci95: {pts(g['point'])} [{pts(g['ci95'][0])}, {pts(g['ci95'][1])}]")
    print(f"  check: cell_fake_catch {100 * r['cell_fake_catch']:.1f} − cell_e2 {100 * r['cell_e2']:.1f}"
          f" = {pts(r['cell_fake_catch'] - r['cell_e2'])}")


if __name__ == "__main__":
    main()
