"""Keys-only proof that inventory_b_fix_v1 ⊂ inventory_b_fix_v2 (09's ruling via 2e, 10-17). fix_v2's "supersedes" field is a claim; this
record is the proof that the pilot-only build copy cites (its INVENTORY_B_CHECK pin).

Both inventories come from the committed INPUTS_FULL, never from stdout or a loose path:
- fix_v2 = inventories.S_old;
- fix_v1 = inventories.S_newmains, which must still be inventory_b_fix_v1 (the pin below).
Each is a {commit, path, sha256} pin. It is git-shown at its commit and checked against its sha256.

Checks, keys, shas and labels only:
1. every fix_v1 row is in fix_v2 and content-identical (canonical JSON: json.dumps(sort_keys=True), the same value AND type in every
   field), and no key appears twice. Rows only canonical-identical (same content, another key order) must number 0: the fix_v2
   builder preserves and checks row order (2e's ruling, 10-17), so any such row STOPs to 09 with the count and the row keys (keys
   only). The record carries both counts: rows_reserialized_identical (rows identical after re-serializing each parsed row with
   json.dumps; == n_fix_v1) and rows_canonical_only (== 0). Both inventories are parsed with a hook that refuses a repeated member
   name in any JSON object, and fix_v1's keys must be the ordered prefix of fix_v2's (the fix_v2 writer appends the pilot rows:
   make_inventory_b_fix_v2.py @ 67e04b0c5 L169).
2. fix_v2 minus fix_v1, by key, is EXACTLY the 20 counted pilot keys (simh_counted_keys_pilot_v3 @ 344829af6; staged key = volume_path
   below simh_evals_v1/ + "#e0");
3. each of those 20 rows has inventory world "sim_honest", and an eval_sha256 equal to its counted key's.
The record (results/inventory_b_fix_v2_subset_check_<stamp>.json, open "x") is written only when every check passes. Otherwise the reasons
are printed and nothing is written. It runs once the full fork's fix_v2 and the INPUTS_FULL that pins it are committed.
Direct-pin mode (09's ruling via 2e, 10-17: the step-1 INPUTS_FULL needs the pilot scores, which need this record): --fix-v2 and --fix-v1 give
the two {commit, path, sha256} pins directly, instead of --inputs-full (mutually exclusive; both or neither). The same checks and the same
record; its input_mode field names the mode and the two pins. 09's tie-back: route C's step-1 INPUTS_FULL must then pin, as
inventories.S_old and inventories.S_newmains, exactly the fix_v2 and fix_v1 this record checked (sha256 equal).

  uv run --no-project --python 3.12 python experiments/2026-10-17_step2_pilot_sold_equal/check_fix_v2_subset.py --inputs-full <path>@<commit>=<sha256>
  uv run --no-project --python 3.12 python experiments/2026-10-17_step2_pilot_sold_equal/check_fix_v2_subset.py --fix-v2 <path>@<commit>=<sha256> --fix-v1 <path>@<commit>=<sha256>
"""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIX_V1 = ("09b070617", "experiments/2026-10-15_cut_sensitivity/results/inventory_b_fix_v1_2026-10-16_025058.json",
          "4a5d0fb3cad32740f849df5def3c7a03a40a18aa4c42bc360709f0c740973d03")
COUNTED = ("344829af6", "experiments/2026-10-14_q3_q35_simhonest_purchase/results/simh_counted_keys_pilot_v3.json",
           "2149c5cde368aef9e5012c8d9cda5b12fe5a93db3d27ad2d525225bb9f511b27")
N, WORLD = 20, "sim_honest"
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731


def git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout


def shown(commit, path, sha256):
    try:
        b = git("show", f"{commit}:{path}")
    except subprocess.CalledProcessError:
        raise SystemExit(f"STOP: {path} @ {commit} is not in git; nothing written") from None
    if sha(b) != sha256:
        raise SystemExit(f"STOP: {path} @ {commit} differs from its pinned sha256; nothing written")
    return b


def canon(r):
    return json.dumps(r, sort_keys=True, separators=(",", ":"))


def no_dup(pairs):
    """json object_pairs_hook: a JSON object with a repeated member name is refused (json.loads would keep the last one silently)."""
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError(f"duplicate member {k!r}")
        d[k] = v
    return d


def pin_arg(s):
    """<path>@<commit>=<sha256> -> {commit, path, sha256} (the same split as --inputs-full)."""
    p, _, rest = s.rpartition("@")
    c, _, h = rest.partition("=")
    return {"commit": c, "path": p, "sha256": h}


def check(v1, v2, counted):
    """Pure: parsed fix_v1, fix_v2, counted keys -> (the STOP reasons, empty = pass; {rows_reserialized_identical, rows_canonical_only};
    the keys of the canonical-only rows)."""
    stops, counts, reordered = [], {"rows_reserialized_identical": 0, "rows_canonical_only": 0}, []
    k1, k2 = [r["key"] for r in v1["runs"]], [r["key"] for r in v2["runs"]]
    if len(set(k1)) != len(k1) or len(set(k2)) != len(k2):  # BEFORE any key map: last-wins could hide a changed row behind its copy
        return ["a key appears twice in fix_v1 or fix_v2"], counts, reordered
    r2 = {r["key"]: r for r in v2["runs"]}
    if any(r["key"] not in r2 for r in v1["runs"]):
        stops.append("a fix_v1 row is missing from fix_v2")
    elif any(canon(r2[r["key"]]) != canon(r) for r in v1["runs"]):
        stops.append("a fix_v1 row is not content-identical (canonical JSON) in fix_v2")
    elif k2[:len(k1)] != k1:
        stops.append("fix_v1's keys are not the ordered prefix of fix_v2's")
    else:
        reordered = sorted(r["key"] for r in v1["runs"] if json.dumps(r2[r["key"]]) != json.dumps(r))
        counts["rows_reserialized_identical"], counts["rows_canonical_only"] = len(v1["runs"]) - len(reordered), len(reordered)
    want = {k["volume_path"].removeprefix("simh_evals_v1/") + "#e0": k["eval_sha256"] for k in counted["keys"]}
    added = set(k2) - set(k1)
    if len(want) != N or added != set(want):
        stops.append(f"fix_v2 minus fix_v1 is not exactly the {N} counted pilot keys")
    elif any(r2[k].get("world") != WORLD for k in added):
        stops.append(f"a pilot row's inventory world is not {WORLD}")
    elif any(r2[k].get("eval_sha256") != want[k] for k in added):
        stops.append("a pilot row's eval_sha256 differs from its counted key's")
    return stops, counts, reordered


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--inputs-full", help="<path>@<commit>=<sha256> of the committed INPUTS_FULL")
    g.add_argument("--fix-v2", help="<path>@<commit>=<sha256> of inventory_b_fix_v2 (direct-pin mode; needs --fix-v1)")
    ap.add_argument("--fix-v1", help="<path>@<commit>=<sha256> of inventory_b_fix_v1 (direct-pin mode only)")
    a = ap.parse_args(argv)
    if (a.fix_v2 is None) != (a.fix_v1 is None):
        raise SystemExit("STOP: --fix-v2 and --fix-v1 go together, and never with --inputs-full; nothing written")
    if a.inputs_full is not None:
        ip, _, rest = a.inputs_full.rpartition("@")
        ic, _, ih = rest.partition("=")
        inputs_full = json.loads(shown(ic, ip, ih))
        inv = inputs_full.get("inventories") if isinstance(inputs_full.get("inventories"), dict) else {}
        p2, p1 = inv.get("S_old") or {}, inv.get("S_newmains") or {}
        src, mode = ("inputs_full.inventories.S_old", "inputs_full.inventories.S_newmains"), "inputs_full"
    else:
        p2, p1, src, mode = pin_arg(a.fix_v2), pin_arg(a.fix_v1), ("--fix-v2", "--fix-v1"), "direct pins"
    if not all(isinstance(p.get(f), str) and (p.get(f) or mode == "inputs_full") for p in (p1, p2) for f in ("commit", "path", "sha256")):
        raise SystemExit("STOP: INPUTS_FULL lacks inventories.S_old or inventories.S_newmains as {commit, path, sha256}; nothing written"
                         if mode == "inputs_full" else "STOP: --fix-v2 or --fix-v1 is not <path>@<commit>=<sha256>; nothing written")
    if (p1["path"], p1["sha256"]) != (FIX_V1[1], FIX_V1[2]) or git("rev-parse", p1["commit"]) != git("rev-parse", FIX_V1[0]):
        raise SystemExit("STOP: INPUTS_FULL's inventories.S_newmains is not inventory_b_fix_v1; nothing written" if mode == "inputs_full"
                         else "STOP: --fix-v1 is not inventory_b_fix_v1; nothing written")
    try:
        v1 = json.loads(shown(p1["commit"], p1["path"], p1["sha256"]), object_pairs_hook=no_dup)
        v2 = json.loads(shown(p2["commit"], p2["path"], p2["sha256"]), object_pairs_hook=no_dup)
    except ValueError as e:
        raise SystemExit(f"STOP: fix_v1 or fix_v2: {e}; nothing written") from None
    counted = json.loads(shown(*COUNTED))
    stops, counts, reordered = check(v1, v2, counted)
    if stops:
        raise SystemExit(f"STOP: {stops}; nothing written")
    if counts["rows_canonical_only"]:
        raise SystemExit(f"STOP (to 09): {counts['rows_canonical_only']} fix_v1 rows are only canonical-identical in fix_v2 (key order differs; "
                         f"rows_reserialized_identical {counts['rows_reserialized_identical']}, rows_canonical_only {counts['rows_canonical_only']}); keys {reordered}; nothing written")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"what": "keys-only proof that inventory_b_fix_v1 is a content-identical (canonical JSON) subset of inventory_b_fix_v2, plus exactly the 20 pilot rows (09 via 2e)",
           "git": git("rev-parse", "HEAD").decode().strip(), "stamp (PT)": stamp,
           "inputs_full": {"path": ip, "commit": git("rev-parse", ic).decode().strip(), "sha256": ih} if mode == "inputs_full" else None,
           "input_mode": {"mode": mode, "fix_v2": dict(p2), "fix_v1": dict(p1)},
           "fix_v1": p1 | {"from": src[1]}, "fix_v2": p2 | {"from": src[0]},
           "counted_keys": dict(zip(("commit", "path", "sha256"), COUNTED)),
           "n_fix_v1": len(v1["runs"]), "n_fix_v2": len(v2["runs"]), "n_added": len(v2["runs"]) - len(v1["runs"]),
           **counts,
           "checks": ["no JSON object in fix_v1 or fix_v2 repeats a member name", "no key twice; fix_v1's keys are the ordered prefix of fix_v2's",
                      "every fix_v1 row in fix_v2, content-identical (canonical JSON: sort_keys, same value and type)",
                      "rows only canonical-identical (another key order) == 0", f"fix_v2 - fix_v1 == the {N} counted pilot keys",
                      f"each added row: world {WORLD}, eval_sha256 == the counted key's"], "stops": [], "passes": True}
    dest = HERE / "results" / f"inventory_b_fix_v2_subset_check_{stamp}.json"
    body = json.dumps(rec, indent=1)
    dest.parent.mkdir(exist_ok=True)
    with dest.open("x") as fh:
        fh.write(body)
    print(dest.name, "PASS; sha256", sha(body.encode()))


if __name__ == "__main__":
    main()
