"""Cell (iii), the fake/fake probe vs a real-trained probe at the same negative count, layer 48, catch at 1% false alarms.
D_m = the fake/fake-trained probe's catch of real second-epoch attacks − the count-matched real-trained probe's catch of
the same attacks (mean over its 5 count-matched draws). Claimed: Qwen3-32B +7.6 [−4.6, +15.0] (Amendment 21e, df's
full + F9 averaged pack) and Seed-OSS-36B +2.1 [−6.3, +9.4] (Amendment 21g, fix-full, averaged readout; interval
resamples tasks and the 5 draws).

Sources (git, sha256-checked; branch aswin/q3-sce-reply-probe, folder experiments/2026-10-13_q3_sce_reply_probe/results/):
- Qwen3-32B: sce_negcount_df_2026-10-13_215536.json @ c4b227cd;
- Seed-OSS-36B: sce_negcount_seed_2026-10-13_222537.json @ 052dcbac.
Run from the repo after `git fetch origin aswin/q3-sce-reply-probe`: python3 -I cell_iii_count_matched.py
"""

import hashlib
import json
import subprocess

R = "experiments/2026-10-13_q3_sce_reply_probe/results"
QWEN = ("c4b227cde29d5cbe0369407739c88c94727bf388", f"{R}/sce_negcount_df_2026-10-13_215536.json",
        "28ebe879e36d39a98d86ba0092d696147380d7bc3546a3d48525d6e19bb3fd08")
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
    r = q["results"]["fpr_0.01|percentile"]
    print(f"Qwen3-32B   {QWEN[1]} @ {QWEN[0][:8]}\n  label: {q['label']}")
    print(f"  results/fpr_0.01|percentile/point, /ci95: {pts(r['point'])} [{pts(r['ci95'][0])}, {pts(r['ci95'][1])}]")
    print(f"  check: b_F9_e2 {100 * r['b_F9_e2']:.1f} − a_m_e2_draw_mean {100 * r['a_m_e2_draw_mean']:.1f}"
          f" = {pts(r['b_F9_e2'] - r['a_m_e2_draw_mean'])}")
    r = s["results"]["seed_ff|L48|fpr_0.01"]
    print(f"\nSeed-OSS-36B   {SEED[1]} @ {SEED[0][:8]}\n  label: {s['label']}\n  cell: {r['cell']}")
    print(f"  results/seed_ff|L48|fpr_0.01/D_m/point, /ci95_v2: {pts(r['D_m']['point'])}"
          f" [{pts(r['D_m']['ci95_v2'][0])}, {pts(r['D_m']['ci95_v2'][1])}]")
    print(f"  check: cell_e2 {100 * r['cell_e2']:.1f} − a_m_e2_draw_mean {100 * r['a_m_e2_draw_mean']:.1f}"
          f" = {pts(r['cell_e2'] - r['a_m_e2_draw_mean'])}")


if __name__ == "__main__":
    main()
