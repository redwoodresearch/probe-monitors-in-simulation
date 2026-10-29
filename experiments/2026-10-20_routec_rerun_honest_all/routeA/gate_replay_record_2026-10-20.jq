# honest_all route A: shape and count gate on the phase-2 replay record (honest_all_replay_2026-10-20_174318.json, sha256 8583f990…).
# Run only as `jq -e -f <this file> <record> > /dev/null`: it outputs one boolean, never a value. Exit 0 = every assertion holds.
# Booleans may have been serialized by json.dumps(default=str) from numpy bools, so T/F accept exactly true/"True" and false/"False".
# Predicted on the real record: true. The results-commit RUN_ME runs it on 16 one-field fixtures, each predicted false (exit 1).
# v2 (Codex 18:07 REQUIRED): near_boundary must be T on rows 96 and 98 and F (false or "False", not missing or null) on the other six.
def T: . == true or . == "True";
def F: . == false or . == "False";
def tags: ["command|L48 a_vs_b", "command|L48 simh_vs_b", "reply|L48 a_vs_b", "reply|L48 simh_vs_b"];
def cells: ["command|L48", "reply|L48"];
def probs: ["a_vs_b", "simh_vs_b"];
def gates13: ["G-CLASS", "G-FOLD", "G-LAYOUT", "G-MANIFEST", "G-MANIFEST-FILE", "G-MYFIT", "G-MYW", "G-PIN", "G-RESULT",
              "G-RESULT-LOCAL", "G-STEMS", "G-STEMS-LOCAL", "G-WIDTH"];
def rows8: ["75", "76", "77", "78", "95", "96", "97", "98"];
def near8: [false, false, false, false, false, true, false, true];   # rows 96 and 98 (command|L48 a_vs_b), as printed in replay.log
. as $r
| type == "object"
  and $r.mode == "replay"
  and $r.stamp == "2026-10-20_174318"
  and $r.src_stamp == "2026-10-20_152036"
  and $r.meta.git_head == "8c051fb5a3d9ead23c36bce4a3b9aae5466968c5"
  and $r.meta.manifest_sha == "7d52ffc317f6a27b134c6bbc5d85f935f34c8ed68253ef0d929e048ab3846fd5"
  and $r.meta.script_sha256 == "b6c45a573054d05f728843eeca319782973494b6a8f03c8c08c77f2817a34f2f"
  and ($r | has("stopped_at") | not)
  and ($r | has("crashed") | not)
  and $r.exit_reasons == []
  and ($r.gates | type) == "object" and ($r.gates | keys) == (gates13 | sort)
  and (gates13 | all(. as $g | $r.gates[$g].ok | T))
  and ($r.fit_verdict | type) == "object" and ($r.fit_verdict | keys) == (tags | sort)
  and (tags | all(. as $t | $r.fit_verdict[$t].verdict == "PASS_RULING" and ($r.fit_verdict[$t].strict | F)))
  and ($r.printed | type) == "object" and ($r.printed | keys) == (rows8 | sort)
  and (rows8 | all(. as $k | ($r.printed[$k].refit_match | T) and ($r.printed[$k].record_match | T)))
  and ([range(0; 8)] | all(. as $i | $r.printed[rows8[$i]].near_boundary | if near8[$i] then T else F end))
  and ($r.link_rows95_98 | T)
  and $r.stored_cert.uncut_gates_failures_count == 0
  and ($r.stored_cert.by_cell_problem | type) == "object" and ($r.stored_cert.by_cell_problem | keys) == (tags | sort)
  and (tags | all(. as $t | $r.stored_cert.by_cell_problem[$t].ok | T))
  and ($r.values | type) == "object" and ($r.values | keys) == (cells | sort)
  and (cells | all(. as $c | ($r.values[$c] | type) == "object" and ($r.values[$c] | keys) == (probs | sort)))
  and ([cells[] as $c | probs[] as $p | $r.values[$c][$p]]
       | length == 4
         and all(.[]; (.replay.stored.match | T) and .replay.stored.tier == "tight" and (.replay.rotated.match | F)
                      and (.my_refit_reproduced.ok | T) and (.fit_quality.resid_explained | T) and (.fit_quality.rank_identical | F)
                      and (.fit_quality.fitter_ok | T) and (.fit_quality.stats_ok | T)))
