# gate_d1.jq: D1 key and certificate checks on refit3's record for one L48 attack-detector cell (refit3-l48 PRESTATED §4 D1 + AMEND1 A3).
# Prints one boolean. Run with jq -e, so false exits 1:
#   jq -e --arg cell "command|L48" --arg wsha <c npz sha256> --arg wpath <c npz path> -f gate_d1.jq <record>
# Positive control, predicted false (exit 1): the same command with --arg cell "command|L63" (or "reply|L63"), keeping the L48 sha and path.
# It reads keys, counts and fixed settings only; no score leaves jq.
. as $r
| ($r.cells[$cell]) as $c
| ($c.problems.c_simatk_vs_run1) as $p
| ($c.saved_weights.c_simatk_vs_run1) as $w
| [
    $r.git == "8ea4a6a5d88fae66a6498e0f81250c433bdd8aaf",
    $r.volume == "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14",
    $r.weights_volume == "probe-shift_aswin-surya_sub-cmdsimdet_uncutall_w_kill-2026-11-14",
    $r.settings.C == 0.01,
    $r.epochs_per_batch == {"run1": 1313, "run2": 1313, "sim_attack": 1313},
    $r.inputs.folds.mains_used == 864,
    $r.inputs.cohort.sha256 == "4511f6c928c91f10dc2074a5d06bac4ab47ff57dca178f57fe14f8c106353477",
    $r.inputs.folds.sha256 == "c2fc5d40eb84a34a64cbb9ce7155572d1414f12ea1fbc2ec05708bd73061e299",
    ([$r.refit3.changed_paths[]
      | select(startswith("/cells/" + $cell + "/problems/c_simatk_vs_run1")
               or startswith("/cells/" + $cell + "/heldout_run2")
               or startswith("/cells/" + $cell + "/saved_weights"))] | length) == 0,
    ($c.certificates | length) == 20,
    ([$c.certificates[] | .C == 0.01] | all),
    ([$c.certificates[0:5][] | "\(.priority)/\(.problem)/\(.fold)"]
      == ["P1/c_simatk_vs_run1/0", "P1/c_simatk_vs_run1/1", "P1/c_simatk_vs_run1/2", "P1/c_simatk_vs_run1/3", "P1/c_simatk_vs_run1/4"]),
    ([$c.certificates[0:5][]
      | .columns == [0] and .certified_per_column == [true] and .uncertified_columns == []
        and (.max_grad | type) == "number" and .max_grad <= 1e-6] | all),  # Codex r2: in jq, null <= x is true, so require a number
    ([$c.certificates[0:5][] | (.n_evals | type) == "number" and .n_evals >= 6250 and .max_grad > 3e-7] | all),  # AMEND1 A3 (+ Codex r3: a number)
    $p.observed_complete == true,
    $p.observed_certified == true,
    ($p.run_scores | length) == 2620,
    ([$p.run_scores | keys[] | split("|")[0]] | group_by(.) | map({(.[0]): length}) | add) == {"run1": 1311, "sim": 1309},
    $p.n_runs == {"class1": 1309, "class0": 1311},
    $c.heldout_run2.n_runs == 1313,
    ($c.heldout_run2.run_scores | length) == 1313,
    ([$c.heldout_run2.run_scores | keys[] | startswith("run2|")] | all),
    ($c.runs_without_rows | length) == 6,
    ($c.chunks_read | length) == 829,
    $w.sha256 == $wsha,
    $w.path == $wpath,
    $w.folds == [0, 1, 2, 3, 4],
    $w.standardization == "torch_probe.standardize formulas, training rows of the fold",
    ($w.max_abs_saved_weights_score_diff | type) == "number",  # Codex r2 P2: null <= 1e-9 is true in jq
    $w.max_abs_saved_weights_score_diff <= 1e-9
  ]
| all
