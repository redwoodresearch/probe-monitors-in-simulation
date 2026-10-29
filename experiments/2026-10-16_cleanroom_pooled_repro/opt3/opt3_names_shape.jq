# Shape gate for opt3_npz_names records (stage 0), applied with `jq -e -s -f` (slurped: the input is an array of documents).
# Exactly one document; exact keys and types; numbers only at n_members (Codex stage-0 pass, item 4).
def is_str_arr: type == "array" and all(.[]; type == "string");
def role_ok:
  type == "object"
  and (keys == ["members", "n_members", "path", "sha256_matches_pin"])
  and ((.path | type) == "string")
  and (.sha256_matches_pin == true)
  and (.members | is_str_arr)
  and ((.n_members | type) == "number") and (.n_members == (.n_members | floor)) and (.n_members == (.members | length));
def pin_ok: type == "object" and (keys == ["file", "sha256"]) and ((.file | type) == "string") and ((.sha256 | type) == "string") and (.sha256 | test("^[0-9a-f]{64}$"));
length == 1
and (.[0] | type == "object"
  and (keys == ["files", "pins_from"])
  and (.files | type == "object" and (keys == ["A_calib_five_sol_a_C", "A_calib_single_FIBS_C", "B_calib_five_sol_a_C", "B_gap_perdraw"]) and all(.[]; role_ok))
  and (.pins_from | type == "array" and length == 2 and all(.[]; pin_ok))
  and ([paths(type == "number")] | all(.[-1] == "n_members")))
