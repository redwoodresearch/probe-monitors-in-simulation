# Appendix rules (kappa, R², Spearman), appended to apply_v84.py's rule set via exec in the same namespace.
KG = lambda f: f"21.kappa_group.headline_six|C.joint.{f}"
KF = lambda f: f"22.kappa_five.C.joint.{f}"
R2 = lambda blk, f: f"24.r2.{blk}.own_all_rs_uncut_L48 C.{f}"
SP, SPN = "23.spearman.sol_a_uncut C.commands L63.sim_score_vs_drop", "23.spearman.sol_a_uncut C.commands L63.null_run2_score_vs_run2_minus_run1"
sub("H1 kappa six", "real run 2 agrees with real run 1 at a kappa of 0.16, and the simulated run at 0.19, a difference of +0.03 (95% interval −0.05 to +0.10).",
    lambda P: f"real run 2 agrees with real run 1 at a kappa of {P[KG('mean_kappa_run2_vs_run1')]['pt']}, and the simulated run at {P[KG('mean_kappa_sim_vs_run1')]['pt']}, a difference of {P[KG('mean_kappa_sim_minus_run2')]['pt']} (95% interval {ci(P[KG('mean_kappa_sim_minus_run2')])}).",
    [KG("mean_kappa_run2_vs_run1"), KG("mean_kappa_sim_vs_run1"), KG("mean_kappa_sim_minus_run2")],
    check=lambda P: sign_flag("kappa six diff ('small positive average comes from the on-policy probe')", P[KG("mean_kappa_sim_minus_run2")], "+")
    + ["kappa: recheck 'No single probe's difference excludes zero' against 21.kappa per-probe slots"])
sub("H2 kappa five", "the other five probes give 0.20 for the second real run and 0.17 for the simulated run, a difference of −0.03 (95% interval −0.06 to +0.01), so the sign flips.",
    lambda P: f"the other five probes give {P[KF('mean_kappa_run2_vs_run1')]['pt']} for the second real run and {P[KF('mean_kappa_sim_vs_run1')]['pt']} for the simulated run, a difference of {P[KF('mean_kappa_sim_minus_run2')]['pt']} (95% interval {ci(P[KF('mean_kappa_sim_minus_run2')])}), so the sign flips.",
    [KF("mean_kappa_run2_vs_run1"), KF("mean_kappa_sim_vs_run1"), KF("mean_kappa_sim_minus_run2")],
    check=lambda P: sign_flag("kappa five diff ('so the sign flips')", P[KF("mean_kappa_sim_minus_run2")], "−"))
sub("H3 R2", "A second real run gives an R² of 0.21 and the simulated run 0.15, a difference of −0.05, computed before rounding (95% interval −0.10 to −0.01).",
    lambda P: f"A second real run gives an R² of {P[R2('lead','yardstick')]['pt']} and the simulated run {P[R2('lead','main')]['pt']}, a difference of {P[R2('lead','delta_main_minus_yardstick')]['pt']}, computed before rounding (95% interval {ci(P[R2('lead','delta_main_minus_yardstick')])}).",
    [R2("lead", "yardstick"), R2("lead", "main"), R2("lead", "delta_main_minus_yardstick")],
    check=lambda P: zero_flag("R² lead diff (Fig 8 caption 'Only the on-policy probe's difference excludes zero')", P[R2("lead", "delta_main_minus_yardstick")], False))
sub("H4 R2 run2", "the difference shrinks to −0.02 (95% interval −0.06 to +0.03).",
    lambda P: f"the difference shrinks to {P[R2('sensitivity_run2_anchor','delta_main_minus_yardstick')]['pt']} (95% interval {ci(P[R2('sensitivity_run2_anchor','delta_main_minus_yardstick')])}).",
    [R2("sensitivity_run2_anchor", "delta_main_minus_yardstick")],
    check=lambda P: ["R² run-2 anchor: check 'shrinks' still true vs lead"])
sub("C6 spearman", "(Spearman correlation −0.01, 95% interval −0.07 to +0.05, against −0.02 for two real runs).",
    lambda P: f"(Spearman correlation {P[SP]['pt']}, 95% interval {ci(P[SP])}, against {P[SPN]['pt']} for two real runs).",
    [SP, SPN], check=lambda P: zero_flag("Spearman ('no sign of it')", P[SP], True))
flags.append(("H5/Table 6", "Fig 8 caption per-probe claim, kappa per-probe claim, and Table 6 planted verdicts (25.plant.*) are checked BY HAND"))
RS10 = "09.catch.own_all_rs10_uncut_L48 C.mean"  # key name to be confirmed by 0d when the readout package lands
sub("C7 rs10", "the on-policy probe still catches 98% of real attacks on whole runs",
    lambda P: f"the on-policy probe still catches {P[RS10]['pt']}% of real attacks on whole runs, averaged over the two real runs",
    [RS10], check=lambda P: ["rs10: confirm 'so less training does not produce a mid-range monitor' still holds at this value"])
