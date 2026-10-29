"""Summary table of the uncut detector, read BY KEY from the committed records (no refit, no resampling): per (problem, readout,
layer), the per-run and per-action AUROC with 95% task-bootstrap interval and null p95 / p, beside the 32k value of the same row (this
folder's gate-2 records, equal to 2e's to 0.0). Prints markdown; every cell names its key path in the header notes.

  python3 -I analyze_uncut_summary.py results/honest_uncut_<stamp>.json results/attack_uncut_<stamp>.json results/honest_32k_<stamp>.json results/attack_32k_<stamp>.json
"""

import json
import sys

CONTROLS = {"c_vs_b", "d_run2_vs_run1", "hold:heldout_run2_vs_run1"}


def f(x):
    return "—" if x is None else f"{x:.3f}"


def ci(c):
    return "—" if c is None else f"[{c[0]:.3f}, {c[1]:.3f}]"


def main(hu, au, h32, a32):
    H, A, H3, A3 = (json.load(open(p)) for p in (hu, au, h32, a32))
    print(f"Sources: {hu}, {au} (uncut); {h32}, {a32} (32k, gate 2). PRELIMINARY.\n")
    print("| problem | readout | layer | run AUROC uncut [95% CI] | null p95 / p | run AUROC 32k | action AUROC uncut [95% CI] | null p95 / p | action AUROC 32k | n runs (1/0) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for ro in ("command", "reply"):
        for L in (48, 63):
            cell = f"{ro}|L{L}"
            for p in ("simh_vs_BuC", "simh_vs_Bmatched", "c_vs_b"):
                r, r3 = H["cells"][cell]["problems"][p], H3["cells"][cell]["problems"][p]
                o, n, nr = r["observed"], r["null"], r["null_row"]
                tag = " (control)" if p in CONTROLS else ""
                print(f"| honest {p}{tag} | {ro} | {L} | {f(o['auroc'])} {ci(o['auroc_boot']['ci95'])} | {f(n['p95'])} / {f(n['p_add_one'])} | {f(r3['observed']['auroc'])} | "
                      f"{f(o['row_auroc'])} {ci(o['row_auroc_boot']['ci95'])} | {f(nr['p95'])} / {f(nr['p_add_one'])} | {f(r3['observed']['row_auroc'])} | "
                      f"{r['n_runs']['class1']}/{r['n_runs']['class0']} |")
            for p in ("c_simatk_vs_run1", "d_run2_vs_run1"):
                r, r3 = A["cells"][cell]["problems"][p], A3["cells"][cell]["problems"][p]
                ns, nc = r["null"]["run_level_auroc_summary"], r["null"]["call_level_auroc_summary"]
                tag = " (control)" if p in CONTROLS else ""
                print(f"| attack {p}{tag} | {ro} | {L} | {f(r['run_level']['auroc'])} {ci(r['run_auroc_boot']['ci95'])} | {f(ns['p95'])} / {f(ns['p_add_one'])} | "
                      f"{f(r3['run_level']['auroc'])} | {f(r['call_level']['auroc'])} {ci(r['call_auroc_boot']['ci95'])} | {f(nc['p95'])} / {f(nc['p_add_one'])} | "
                      f"{f(r3['call_level']['auroc'])} | {r['n_runs']['class1']}/{r['n_runs']['class0']} |")
            for k in ("sim_vs_heldout_run2", "heldout_run2_vs_run1"):
                r, r3 = A["cells"][cell]["heldout_run2"]["contrasts"][k], A3["cells"][cell]["heldout_run2"]["contrasts"][k]
                tag = " (control)" if f"hold:{k}" in CONTROLS else ""
                print(f"| attack held-out {k}{tag} | {ro} | {L} | {f(r['auroc'])} {ci(r['boot']['ci95'])} | no null | {f(r3['auroc'])} | "
                      f"{f(r['call_level']['auroc'])} {ci(r['call_level']['boot']['ci95'])} | no null | {f(r3['call_level']['auroc'])} | {r['n_pos']}/{r['n_neg']} |")
    print("\nKeys: honest .observed.auroc / .auroc_boot.ci95 / .null.p95, .p_add_one; per action .observed.row_auroc / .row_auroc_boot.ci95 / .null_row; "
          "attack .run_level.auroc / .run_auroc_boot.ci95 / .null.run_level_auroc_summary; per action .call_level.auroc / .call_auroc_boot.ci95 / "
          ".null.call_level_auroc_summary; held-out .heldout_run2.contrasts[k].auroc / .boot.ci95 / .call_level. 32k columns: the same keys in the gate-2 records.")


if __name__ == "__main__":
    main(*sys.argv[1:5])
