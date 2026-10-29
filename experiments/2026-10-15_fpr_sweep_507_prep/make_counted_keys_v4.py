"""Re-lock (09, 10-17 ~00:4x; data only): the S_old sim-honest counted-key manifest primary_v4 = primary_v3 u the 20 pilot keys, nothing else.
Keys, task names and sha256s only; no score is read. Inputs are read from git by <commit>:<path>@<sha256>.

POSITIVE CONTROL (09): from primary_v3's source (cont_v2 @ ae1ca69d7), the rule primary_v3 states (counted_tasks and keys = the usable ones;
the counted failure out) reproduces v3's 149 keys EXACTLY:
  - the run-key sets are equal;
  - the `keys` and `counted_tasks` arrays are byte-equal to the committed v3 (572f67dbb, 80753f44...) under its serialization (indent 1);
  - n / n_counted_tasks / usable are equal.
  The full v3 file is NOT re-derivable byte for byte: its `exclusions` text and `derived_from` were edited by hand (v3's own `changes`
  field says so). Those two blocks are carried from v3 as is.
v4 (09 (a), (b)):
  - v4.keys = v3.keys followed by the 20 pilot keys (@ 344829af6), each object VERBATIM;
  - the per-key world is kept: simh_full for v3, simh_pilot for the pilot, the pilot keyed as its own world as the fill-ins are;
  - counted_tasks = v3's followed by the pilot's tasks; n and n_counted_tasks = 169; usable = v3's usable + 20 (the pilot's gate12 states
    20 / 20 usable; its keys carry no per-key `usable` field and are not given one).
ASSERTS: no pilot task is a v3 task (t21-t40 have no simh_full run, asserted not assumed); run keys and .eval sha256s are unique across the
union; every v3 key is world simh_full and every pilot key world simh_pilot, stratum S_old; v4.keys[:149] is byte-equal to v3.keys.
Any failure STOPs before writing. The output is written with exclusive create; counts and shas are printed.

  uv run python make_counted_keys_v4.py --v2 <c>:<p>@<sha> --v3 <c>:<p>@<sha> --pilot <c>:<p>@<sha> --out config/relock_pilot/<name>.json
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def show(ref):
    """<commit>:<path>@<sha256> -> (parsed json, raw bytes); the sha is checked."""
    cp, h = ref.rsplit("@", 1)
    b = subprocess.run(["git", "show", cp], capture_output=True, check=True, cwd=HERE).stdout
    if hashlib.sha256(b).hexdigest() != h:
        raise SystemExit(f"{cp}: sha256 differs")
    return json.loads(b), b


def dump(x):
    return json.dumps(x, indent=1).encode()  # the committed v3's serialization (indent 1, ASCII, no trailing newline)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", required=True)  # ae1ca69d7:.../simh_counted_keys_cont_v2.json@688cc1cc...
    ap.add_argument("--v3", required=True)  # 572f67dbb:.../simh_counted_keys_primary_v3.json@80753f44...
    ap.add_argument("--pilot", required=True)  # 344829af6:.../simh_counted_keys_pilot_v3.json@2149c5cd...
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    (v2, _), (v3, v3b), (pl, _) = show(a.v2), show(a.v3), show(a.pilot)

    # positive control: v3's keys from its source by the stated rule
    keep = [k for k in v2["keys"] if k.get("usable") is True and "counted_failure" not in k]
    tasks = [t for t in v2["counted_tasks"] if t in {k["task"] for k in keep}]
    pc = {"run_keys_set_equal": {k["run_key"] for k in keep} == {k["run_key"] for k in v3["keys"]},
          "keys_bytes_equal": dump(keep) == dump(v3["keys"]), "counted_tasks_bytes_equal": dump(tasks) == dump(v3["counted_tasks"]),
          "counts_equal": (len(keep), len(tasks), sum(1 for k in keep if k.get("usable") is True)) == (v3["n"], v3["n_counted_tasks"], v3["usable"]),
          "v3_serialization_reproduced": dump(v3) == v3b}
    if not all(pc.values()):
        raise SystemExit(f"positive control FAILED (v3 not reproduced from cont_v2): {pc}")

    # v4 = v3 u pilot, asserts first
    v3t, plt = {k["task"] for k in v3["keys"]}, [k["task"] for k in pl["keys"]]
    ks = v3["keys"] + pl["keys"]
    bad = []
    if set(plt) & v3t:
        bad.append(f"pilot tasks with a simh_full run: {sorted(set(plt) & v3t)}")
    if len(plt) != 20 or len(set(plt)) != 20 or pl.get("n") != 20:
        bad.append(f"pilot: not 20 distinct tasks ({len(plt)}, {len(set(plt))}, n {pl.get('n')})")
    for f in ("run_key", "eval_sha256"):
        if len({k[f] for k in ks}) != len(ks):
            bad.append(f"{f} not unique across the union")
    if any(k.get("world") != "simh_full" for k in v3["keys"]) or any(k.get("world") != "simh_pilot" or k.get("stratum") != "S_old" for k in pl["keys"]):
        bad.append("worlds / strata not as expected (v3 simh_full; pilot simh_pilot, S_old)")
    if bad:
        raise SystemExit(f"v4 asserts FAILED: {bad}")
    v4 = {}
    for k, v in v3.items():  # v3's own key order; the changed fields only
        v4[k] = v
    v4 |= {"n": v3["n"] + 20, "usable": v3["usable"] + 20, "counted_tasks": v3["counted_tasks"] + plt,
           "n_counted_tasks": v3["n_counted_tasks"] + 20, "keys": ks,
           "derived_from": {"v3": {"ref": a.v3}, "pilot": {"ref": a.pilot}, "positive_control": pc,
                            "changes": "v3 u the 20 pilot keys (09 re-lock, 10-17): pilot keys verbatim (own world simh_pilot), appended after v3's; "
                                       "counted_tasks + the pilot's 20 tasks; n, n_counted_tasks, usable + 20; rule, launch, gate12, exclusions "
                                       "as v3's (the pilot's own gate12 / launch are in its manifest, pinned above)"}}
    if dump(v4["keys"][:len(v3["keys"])]) != dump(v3["keys"]):
        raise SystemExit("v4 keys do not start with v3's keys byte for byte")
    out = HERE / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as fh:
        fh.write(dump(v4))
    print(json.dumps({"written": str(out.relative_to(HERE)), "sha256": hashlib.sha256(dump(v4)).hexdigest(), "n": v4["n"],
                      "worlds": {w: sum(1 for k in ks if k["world"] == w) for w in ("simh_full", "simh_pilot")}, "positive_control": pc}))


if __name__ == "__main__":
    main()
