def p: if . == null then "–" else (. * 1000 | round / 10 | tostring) end;
def iv: "[\(.[0]|p), \(.[1]|p)]";
def row($t; $b): $t[$b] as $r |
  "  \($b): run1 \($r.run1.detected|p) \($r.run1.ci95|iv) | run2 \($r.run2.detected|p) \($r.run2.ci95|iv) | mean(real) \($r.real_mean.detected|p) \($r.real_mean.ci95|iv) | sim \($r.sim.detected|p) \($r.sim.ci95|iv)\n      sim−mean(real) \($r["sim_minus_mean_real (S primary)"].value|p) \($r["sim_minus_mean_real (S primary)"].ci95|iv) | run2−sim \($r["run2_minus_sim (P primary)"].value|p) \($r["run2_minus_sim (P primary)"].ci95|iv) | run2−run1 \($r["run2_minus_run1 (yardstick)"].value|p) \($r["run2_minus_run1 (yardstick)"].ci95|iv)";
def block($name; $x): "\($name): \($x.label)\n  tasks \($x.main.tasks), honest runs \($x.main.honest_runs)\n\(row($x.main; "0.05"))\n\(row($x.main; "0.01"))\n  verdict 5%: \($x.verdict["5%"]|join("; ")) | 1%: \($x.verdict["1%"])\n  sensitivity: refusals \($x.sensitivity_refusals.forced_runs) forced, every_drop \($x.sensitivity_every_drop.forced_runs) forced";
"label: \(.label)",
"R0: \(.R0_reproduces_launch1|tostring)",
"score file: \(.score_file|tostring)",
"counts: \(.counts|tostring)",
"G2: \(.g2|tostring)",
"cohort tasks by launch: \(.cohort_tasks_by_launch|tostring)",
"attempted in all three (k of n, cohort only): \(.attempted_all_three.k) of \(.attempted_all_three.n); coverage \(.attempted_all_three.label_coverage|tostring)",
block("P"; .P), block("S"; .S),
"S realized FA on held-out A: \(.S_realized_FA_on_held_out_A|tostring)",
"status counts: \(.status_counts|tostring); no-reference cells by launch: \(.no_reference_cells_by_launch|tostring)",
(.status_rows | to_entries[] | "  \(.key) [\(.value.launch)]: \(.value.status_row) | in cohort \(.value.in_cohort) | excluded lines \(.value.excluded_lines|keys|join(",")) | source: \(.value.reason_source)"),
"dropped (unparsed etc.): \(.dropped_by_key | to_entries | map(select(.value|length>0)) | map("\(.key): \(.value|to_entries|map("\(.value.task) \(.key) \(.value.category)")|join("; "))") | join(" || "))",
"excluded keys: \(.excluded_by_key|length)"
