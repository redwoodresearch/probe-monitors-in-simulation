"""Route C expected-diff gate (507-prep, 10-17): the paid route C spec differs from the spec the 09-reviewed pre-flight read ONLY by the pinned
zero-unit list. Local and keys-only: it reads the spec_diff record (paths and counts, never a value) and the pinned pre-flight record.

PASS needs:
- the diff's side a is the pre-flight's spec, by sha256 (the record's spec.sha256), and side b a different spec;
- every counted difference (only_a, only_b, changed) is a pinned zero-unit key on a tagged S_old entry
  (jobs/<i>/1/simh_old/<label>/<n>/zero_unit_ok/<k>, side b only) or the meta pin (meta/full_length/s_old_simh_extra_roots/<root>/zero_unit_list,
  side b only, exactly once);
- zero_unit_ok paths appear iff the pre-flight lists any zero-unit key, and their count equals its expected_zero_unit_ok_paths;
- every tagged S_old entry on side b carries exactly the pinned list for its own file (zero_unit_by_file[its sha256]), by key, and on side
  a an empty one (spec_diff's a/b_routec_entries);
- side b pins exactly THIS pre-flight record (spec_diff's b_pins = [{path, sha256}] of --preflight) and side a pins none.
Side b is the gated spec-only pickle; after launch, spec_diff(that pickle, vol:<the stored driver spec>) must show 0 counted differences
(NOTE_routec_preflight.md), so the launched spec is the gated one.
ignored / normalized / order lists are reported, not counted. Exit 1 on any failure; the verdict JSON is printed (and written with --out).

  uv run python experiments/2026-10-15_fpr_sweep_507_prep/gate_spec_diff_routec.py --diff results/spec_diff/<x>.json \
      --preflight results/routec_preflight/<record>.json@<sha256> [--out results/spec_diff/<gate>.json]
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ZU_PAT = re.compile(r"jobs/\d+/1/simh_old/[^/]+/\d+/zero_unit_ok/\d+")  # one pinned key on a tagged S_old entry (side b)
PIN_PAT = re.compile(r"meta/full_length/s_old_simh_extra_roots/[^/]+/zero_unit_list")


def gate(d, pf, pin):
    """Pure: d = the spec_diff record, pf = the pre-flight record, pin = its {path, sha256} as --preflight names it. -> {pass, problems, counts}."""
    bad = []
    if [{k: x.get(k) for k in ("path", "sha256")} for x in d.get("b_pins") or []] != [pin] or d.get("a_pins") != []:
        bad.append("the paid spec does not pin exactly THIS pre-flight record (b_pins), or the pre-flight's own spec carries a pin (a_pins)")
    if (d.get("a") or {}).get("sha256") != (pf.get("spec") or {}).get("sha256"):
        bad.append("side a is not the spec the pre-flight read (sha256)")
    if (d.get("b") or {}).get("sha256") in (None, (d.get("a") or {}).get("sha256")):
        bad.append("side b is missing or the same spec as side a")
    if d.get("plant_b"):
        bad.append("a planted diff is not a gate input")
    zu_ = [p for p in d.get("only_b") or [] if ZU_PAT.fullmatch(p)]
    pin_ = [p for p in d.get("only_b") or [] if PIN_PAT.fullmatch(p)]
    other_ = sorted(set(d.get("only_a") or []) | set(d.get("changed") or []) | (set(d.get("only_b") or []) - set(zu_) - set(pin_)))
    if other_:
        bad.append(f"differences outside the pinned zero-unit list: {len(other_)}, e.g. {other_[:5]}")
    if len(pin_) != 1:
        bad.append(f"the meta pin appears {len(pin_)} times on side b (need exactly 1)")
    n_zu_keys = len({k for v in (pf.get("zero_unit_by_file") or {}).values() for k in v})
    if bool(zu_) != bool(n_zu_keys):
        bad.append(f"zero_unit_ok differences ({len(zu_)}) while the pre-flight lists {n_zu_keys} zero-unit keys")
    zbf_ = pf.get("zero_unit_by_file") or {}  # Astra r2 R1: the KEYS, per tagged entry, against the pinned list for its own file
    ea_, eb_ = d.get("a_routec_entries"), d.get("b_routec_entries")
    if not isinstance(ea_, list) or not isinstance(eb_, list) or sorted(x["at"] for x in ea_) != sorted(x["at"] for x in eb_) or not eb_:  # Astra
        # launch r1: the same entry paths as a SET (dict insertion order is not a difference)
        bad.append("the tagged S_old entries are missing or not the same on both sides")
    else:
        wrong_ = [x["at"] for x in eb_ if x.get("file_sha256") not in zbf_ or x.get("zero_unit_ok") != sorted(zbf_[x["file_sha256"]])]
        if wrong_ or any(x.get("zero_unit_ok") != [] for x in ea_):
            bad.append(f"tagged entries whose zero_unit_ok is not the pinned list for their file: {len(wrong_)}, e.g. {wrong_[:3]}; "
                       f"or the pre-flight's spec already carries one")
    if len(zu_) != pf.get("expected_zero_unit_ok_paths"):  # 507-prep (iii): the count, not just presence
        bad.append(f"zero_unit_ok paths {len(zu_)} != the pre-flight's expected {pf.get('expected_zero_unit_ok_paths')}")
    if d.get("n_differences") != len(d.get("only_a") or []) + len(d.get("only_b") or []) + len(d.get("changed") or []):
        bad.append("n_differences disagrees with the listed paths")
    if not isinstance(d.get("b_meta"), dict) or not d["b_meta"].get("git") or not str(d["b_meta"].get("dest", "")).endswith(".json"):
        bad.append("side b's launch metadata (build commit, record path) is missing")
    return {"pass": not bad, "problems": bad, "a_sha256": (d.get("a") or {}).get("sha256"), "b_sha256": (d.get("b") or {}).get("sha256"),  # the launch
            "b_meta": d.get("b_meta"),  # side b's build commit, record path, pooled flag and shipped inputs (no value): the launcher reads only these
            "counts": {"zero_unit_ok_paths": len(zu_), "meta_pin_paths": len(pin_), "other": len(other_),
                                                         "preflight_zero_unit_keys": n_zu_keys, "ignored": len(d.get("ignored_differing") or []),
                                                         "normalized": len(d.get("normalized_differing") or []), "order": len(d.get("order_differing") or [])}}


KEYS_DEST = re.compile(r"xfer/fpr507prep/records/results/nofilter_keys_FULL_TEST_\d{4}-\d{2}-\d{2}_\d{6}\.json")  # dispatch()'s census record
LISTS = ("only_a", "only_b", "changed", "ignored_differing", "normalized_differing", "order_differing")


def gate_keys_only(d):
    """The census launch (Aswin A1, 10-17): a keys-only spec pins no pre-flight record. Side a is a second, independent --spec-only build of the
    same flags at the same HEAD. PASS needs both sides keys-only (every job), side b naming the FULL TEST census record, side b's build commit
    set and equal to side a's (meta/git is ignored by the diff), no route C entry, pin or zero-unit path, two different specs, and 0 counted
    differences. Anything else fails closed. -> the same verdict fields as gate()."""
    bad = []
    am_, bm_ = d.get("a_meta") if isinstance(d.get("a_meta"), dict) else {}, d.get("b_meta") if isinstance(d.get("b_meta"), dict) else {}
    a_, b_ = (d.get("a") or {}).get("sha256"), (d.get("b") or {}).get("sha256")
    if am_.get("keys_only") is not True or bm_.get("keys_only") is not True:
        bad.append("a side is not keys-only (every job keys_only)")
    if not KEYS_DEST.fullmatch(str(bm_.get("dest", ""))):
        bad.append("side b does not name xfer/fpr507prep/records/results/nofilter_keys_FULL_TEST_<stamp>.json")
    if not bm_.get("git") or am_.get("git") != bm_.get("git"):
        bad.append("side b's build commit is missing, or side a was built at another commit")
    if any(d.get(k) != [] for k in ("a_routec_entries", "b_routec_entries", "a_pins", "b_pins")) or any(
            ZU_PAT.fullmatch(p) or PIN_PAT.fullmatch(p) for k in LISTS for p in d.get(k) or []):
        bad.append("a route C entry, pin or zero-unit path on a side")
    if not a_ or not b_ or a_ == b_:
        bad.append("side a or b is missing, or both are the same spec (side a must be a second, independent build)")
    if d.get("plant_b") or d.get("n_differences") != 0 or any(d.get(k) for k in LISTS[:3]):
        bad.append("counted differences between the two builds (or a planted diff)")
    return {"pass": not bad, "problems": bad, "a_sha256": a_, "b_sha256": b_, "b_meta": d.get("b_meta"),
            "counts": {"differences": d.get("n_differences")} | {k: len(d.get(k) or []) for k in LISTS[3:]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff", required=True)
    ap.add_argument("--preflight", required=True)  # <committed path>@<sha256>, the record --routec-zero-unit pins
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    pp_, ph_ = a.preflight.rsplit("@", 1)
    pb_ = subprocess.run(["git", "show", f"HEAD:./{pp_}"], capture_output=True, check=True, cwd=HERE).stdout
    if hashlib.sha256(pb_).hexdigest() != ph_:
        raise SystemExit("--preflight: not committed at HEAD at this sha256")
    v = gate(json.loads((HERE / a.diff).read_bytes()), json.loads(pb_), {"path": pp_, "sha256": ph_}) | {"diff": a.diff, "preflight": {"path": pp_, "sha256": ph_}}
    b = json.dumps(v, indent=1).encode()
    if a.out:
        with (HERE / a.out).open("xb") as fh:
            fh.write(b)
    print(b.decode())
    sys.exit(0 if v["pass"] else 1)


if __name__ == "__main__":
    main()
