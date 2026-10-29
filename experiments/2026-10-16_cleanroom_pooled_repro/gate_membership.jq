# Membership gate read for cleanroom_reduce_v2.py --stage membership (a1, 2026-10-18). Keys, booleans and counts only.
# Use (with set -o pipefail):  jq -f gate_membership.jq results/cleanroom_membership_<stamp>.json | tee <summary> | jq -e '.PASS == true' > /dev/null
# rc 0 only if PASS is true; a failed check, a missing field or an unreadable file gives rc != 0 (STOP, per 0d's pre-approval).
# Each label's "gate" is the compute-stage gate() evaluated dry: membership facts, C units by key (incl. zero-unit lead runs) and
# 09's pilot STOPs. The registered floor and the verdict need record A's comparison and are checked at compute, not here.
# Addendum 17: the expected label count comes in as --argjson nl (2 x (6 + sensitivity labels), from the run config).
def c_ok: (.c_units.runs_checked > 0) and (.c_units.n_mismatch == 0) and (.c_units.runs_equal == .c_units.runs_checked)
          and (.c_units.n_zero_mismatch == 0);
def empty_list: (type == "array") and (length == 0);  # a missing key (null) is NOT an empty list
def pilot_ok: (.pilot.one_root == true) and (.pilot.keys_eq_registry == true) and (.pilot.catalogue_eq_registry == true)
              and (.pilot.row_keys_outside_registry | empty_list) and (.pilot.other_files_emit_to_world | empty_list);
{
  n_labels: (.labels | length),
  labels_pass: ([.labels[] | select(.gate == "pass")] | length),
  failing: [.labels | to_entries[] | select(.value.gate != "pass") | {label: .key, gate: .value.gate}],
  n_c_labels: ([.labels | keys[] | select(endswith(" C"))] | length),
  c_units_ok: ([.labels | to_entries[] | select(.key | endswith(" C")) | .value | c_ok] | all),
  pilot_ok: ([.labels[] | pilot_ok] | all),
  fixed_ids_ok: (.fixed_slot.ids == [15, 77]),
  cohort_ok: (((.cohort.vs_nofilter | type) == "object") and ((.cohort.vs_nofilter | keys) == ["S_new", "S_newmains", "S_old"])
             and ([.cohort.vs_nofilter[] | empty_list] | all) and (.cohort.vs_inputs_final_v2_sp493 | empty_list)),
  env_ok: (((.environment.cpu_model | type) == "string") and ((.environment.cpus_usable | type) == "number") and ((.environment.threadpoolctl | type) == "array"))
}
| .PASS = ($nl > 0 and .n_labels == $nl and .labels_pass == $nl and .n_c_labels == ($nl / 2) and .c_units_ok and .pilot_ok and .fixed_ids_ok
           and .cohort_ok and .env_ok)
