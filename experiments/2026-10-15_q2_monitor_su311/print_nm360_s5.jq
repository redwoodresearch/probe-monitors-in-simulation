def p: if . == null then "–" else (. * 1000 | round / 10 | tostring) end;
def iv: "[\(.[0]|p), \(.[1]|p)]";
def row($t; $b): $t[$b] as $r |
  "  \($b): run1 \($r.run1.detected|p) \($r.run1.ci95|iv) | run2 \($r.run2.detected|p) \($r.run2.ci95|iv) | mean(real) \($r.real_mean.detected|p) \($r.real_mean.ci95|iv) | sim \($r.sim.detected|p) \($r.sim.ci95|iv)\n      sim−mean(real) \($r["sim_minus_mean_real (S primary)"].value|p) \($r["sim_minus_mean_real (S primary)"].ci95|iv) | run2−sim \($r["run2_minus_sim (P primary)"].value|p) \($r["run2_minus_sim (P primary)"].ci95|iv) | run2−run1 \($r["run2_minus_run1 (yardstick)"].value|p) \($r["run2_minus_run1 (yardstick)"].ci95|iv)";
def block($name; $x): "\($name): \($x.label)\n  tasks \($x.main.tasks), honest runs \($x.main.honest_runs)\n\(row($x.main; "0.05"))\n\(row($x.main; "0.01"))\n  verdict 5%: \($x.verdict["5%"]|join("; ")) | 1%: \($x.verdict["1%"])\n  sensitivity: refusals \($x.sensitivity_refusals.forced_runs) forced, every_drop \($x.sensitivity_every_drop.forced_runs) forced";
def sblock($name; $x): "\($name): tasks \($x.tasks) of \($x.primary_tasks), cells out \($x.cells_out|length), draws reused \($x.draws_reused)\n\(row($x; "0.05"))\n\(row($x; "0.01"))";
def delta($d): "  primary − sensitivity, 5%: sim−mean(real) \($d["0.05"]["sim_minus_mean_real (S primary)"]|p) | run2−sim \($d["0.05"]["run2_minus_sim (P primary)"]|p) | sim \($d["0.05"].sim|p) | mean(real) \($d["0.05"].real_mean|p); 1%: sim−mean(real) \($d["0.01"]["sim_minus_mean_real (S primary)"]|p)";
"R0: \(.R0_reproduces_launch1|tostring)",
"score files: \(.score_files|tostring)",
"S5 counts: \(.S5.counts|tostring)",
"S5 keys: superseded \(.S5.keys_meta.superseded_originals|length), replacements kept \(.S5.keys_meta.replacements_kept|length), status rows checked \(.S5.keys_meta.status_manifest_rows_checked), excluded keys \(.S5.excluded_by_key|length)",
"S5 dropped: \(.S5.dropped_by_key | to_entries | map(select(.value|length>0)) | map("\(.key): \(.value|to_entries|map("\(.value.task) \(.key) \(.value.category)")|join("; "))") | join(" || "))",
"S5 attempted in all three: \(.S5.attempted_all_three.k) of \(.S5.attempted_all_three.n); label disagreements with the combined file: \(.S5.attempted_all_three.disagreements_with_combined_label_file|length)",
block("S5 P"; .S5.P), block("S5 S"; .S5.S),
"S5 S realized FA on held-out A: \(.S5.S_realized_FA_on_held_out_A|tostring)",
"SENSITIVITY without the 6 (\(.S5_sensitivity_without_6.cells_dropped|join(","))):",
sblock("  S5 P without 6"; .S5_sensitivity_without_6.P), delta(.S5_sensitivity_without_6.primary_minus_sensitivity.P),
sblock("  S5 S without 6"; .S5_sensitivity_without_6.S), delta(.S5_sensitivity_without_6.primary_minus_sensitivity.S),
"S5 status rows: \(.S5.status_counts|tostring)",
(.S5.status_rows | to_entries[] | "  \(.key): \(.value.status_row) | in cohort \(.value.in_cohort)"),
block("S1–S3 P (recomputed)"; .["S1–S3"].P),
"COMBINED S1–S3+S5 (\(.combined_S123_S5.label)):",
"  S sameness S5 B vs S1–S3 B: \(.combined_S123_S5.S_sameness_S5_B_vs_S123_B|tostring)",
"  pooling test: \(.combined_S123_S5.pooling_test|tostring)",
"  headline P: \(.combined_S123_S5.headline_P) | headline S: \(.combined_S123_S5.headline_S) | rule: \(.combined_S123_S5.pooling_rule)",
block("combined P"; .combined_S123_S5.P), block("combined S"; .combined_S123_S5.S),
"  combined attempted in all three: \(.combined_S123_S5.attempted_all_three|tostring)",
delta(.combined_S123_S5.sensitivity_without_6.primary_minus_sensitivity.P),
delta(.combined_S123_S5.sensitivity_without_6.primary_minus_sensitivity.S)
