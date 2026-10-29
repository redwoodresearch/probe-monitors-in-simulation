"""Full-length ("cutfull") sets: 09 RULE.md Append 41 amendment 2 @ bfc8742a6 (grammar, items 1-3) and item 4 @ 77a770eec (separate output trees), verbatim;
d9's routing format cutfull_routing_v1.json @ f7e47563f (top-level `original_folders` {registered arm token: authoritative original folder}); 2e's inventory
v3 @ b8a39f329 (09-OK'd) as the (a'') run and folder list (names from uncut_planned.folder). Off by default. A scorer takes the cutfull line only with its
--cutfull flag, and then ONLY cutfull folders on the cutfull Volume (never mixed with the registered 32k line). Local checks only; no data read."""

import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path

RULE = "09 RULE.md Append 41 amendment 2 @ bfc8742a6 (items 1-3) + item 4 @ 77a770eec; d9 cutfull_routing_v1.json @ f7e47563f; 2e inventory v3 @ b8a39f329"
NAME_RE = re.compile(r"^([a-z0-9_]+?)_cutfull_v1(?:_try[0-9]+)?__([A-Za-z0-9_@.]+)$")  # (b), verbatim: groups (target, arm)
VOLUME = "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14"  # §3: d9's cutfull staged sets and extraction roots, mounted read-only
LINE = "cutfull"  # item 4: every cutfull pack records line = "cutfull"; outputs go under /cutfull/<batch>/<scorer>/<cutfull folder>/
INVENTORY = ("b8a39f329", "experiments/2026-10-15_cut_sensitivity/results/inventory_v3_2026-10-15.json", "19306ffa7c42ab35a62715c1a18822a43c6273492e5ede9b2f62cd9e428ad9a3")
REPO = Path(__file__).resolve().parents[2]
_strip = lambda fo: re.sub(r"_cutfull_v1_try[0-9]+__", "_cutfull_v1__", fo)  # noqa: E731  d9: the cutfull line's own retries; the planned name has none


def original(name: str, accepted, manifest_original) -> str:
    """(a)-(d) for one folder: the authoritative original folder, or SystemExit. `accepted` is the scorer's own non-cutfull folder check."""
    if "_cutfull_gate_" in name:  # (a)
        raise SystemExit(f"{name}: a cutfull gate set is never scored into any pool ({RULE} (a))")
    m = NAME_RE.fullmatch(name)
    if not m:  # (b)
        raise SystemExit(f"{name}: not a cutfull folder name ({RULE} (b))")
    derived = f"{m[1]}_v1__{m[2]}"
    if not accepted(derived):  # (c)
        raise SystemExit(f"{name}: derived original {derived} is outside this scorer's accepted grammar ({RULE} (c))")
    mo = str(manifest_original or "")
    if re.sub(r"_v1_try[0-9]+__", "_v1__", mo) != derived or not accepted(mo):  # (d): the routing names the original by arm token; its own _try<n> is allowed
        raise SystemExit(f"{name}: the routing's original folder {mo or '(none)'} is not {derived} (or its _try<n>), or is outside the grammar ({RULE} (d))")
    return mo


def _inventory() -> dict:
    raw = subprocess.run(["git", "-C", str(REPO), "show", f"{INVENTORY[0]}:{INVENTORY[1]}"], capture_output=True, check=True).stdout
    if hashlib.sha256(raw).hexdigest() != INVENTORY[2]:
        raise SystemExit("inventory v3 differs from its pin")
    return {r["key"]: (r["uncut_planned"]["folder"], r["original_folder"]) for r in json.loads(raw)["runs"]}


def split(msg: dict, enabled: bool, accepted) -> dict:
    """The message as the scorer's name rules see it. Flag off: any cutfull name, the cutfull Volume or `original_folders` is refused (registered line,
    unchanged). Flag on: every folder must be a cutfull folder on the cutfull Volume whose original is msg["original_folders"][arm token] ((a)-(d)), one
    original per cutfull folder, and every epoch must be an inventory-v3 run whose planned folder and original folder match, by key. The returned copy
    carries the ORIGINAL folders, so every existing name rule (rulings, `_rlr`, batch, strata) applies to them; the caller keeps `msg` for its outputs."""
    eps = list(msg["epochs"]) + [x for p in msg.get("provenance", []) for x in p.get("excluded_by_staging", [])]
    if not enabled:
        if any("_cutfull" in str(e["folder"]) for e in eps) or msg.get("volume") == VOLUME or "original_folders" in msg:
            raise SystemExit(f"a cutfull folder, the cutfull Volume or original_folders without --cutfull ({RULE} item 4: registered readers never take the cutfull line)")
        return msg
    of = msg.get("original_folders")
    if msg.get("volume") != VOLUME or not isinstance(of, dict) or not of:
        raise SystemExit(f"--cutfull: the message Volume must be {VOLUME} and it must carry d9's top-level original_folders")
    out, orig = copy.deepcopy(msg), {}
    for e in list(out["epochs"]) + [x for p in out.get("provenance", []) for x in p.get("excluded_by_staging", [])]:
        fo = str(e["folder"])
        m = NAME_RE.fullmatch(fo)
        o = original(fo, accepted, of.get(m[2]) if m else None)
        if orig.setdefault(fo, o) != o or o in {v for k, v in orig.items() if k != fo}:
            raise SystemExit(f"{fo}: two cutfull folders share one original, or one folder has two originals")
        e["folder"] = o
    inv = _inventory()
    for e_cut, e_org in zip(msg["epochs"], out["epochs"], strict=True):  # by key: inventory v3's planned name and original for this run
        if inv.get(e_cut["key"]) != (_strip(e_cut["folder"]), e_org["folder"]):
            raise SystemExit(f"{e_cut['key']}: not an inventory-v3 run with planned folder {_strip(e_cut['folder'])} and original {e_org['folder']}")
    return out


def _selftest():  # 09's test vectors (cutfull_regex_test_output.txt @ bfc8742a6), keys only; fam = 09's tested families
    fam = re.compile(r"^(?:ep2_(t196|311_l1|311_l2|nm)_v1(_try[0-9]+)?__honest_(a|b)|simh311_q[1-4]_sim(?:_s[0-9]+|_rlr(?:[2-9]|[1-9][0-9]+)?)?_v1(?:_try[0-9]+)?__sim_honest"
                     r"|simh_nm360_(s1|s2|s3|s4d|s4r|s5)_sim(?:_s[0-9]+|_rlr(?:[2-9]|[1-9][0-9]+)?)?_v1(?:_try[0-9]+)?__sim_honest|simh_t196fill(?:_s[0-9]+)?_v1(?:_try[0-9]+)?__sim_honest"
                     r"|ilold_oldsel_(?:run3_v1(?:_try[0-9]+)?__attack|sim2_v1(?:_try[0-9]+)?__sim_attack))$")
    ok = fam.fullmatch
    want = {"ep2_t196_cutfull_v1__honest_a": "ep2_t196_v1__honest_a", "ep2_t196_cutfull_v1_try2__honest_b": "ep2_t196_v1__honest_b",
            "simh311_q1_sim_rlr2_cutfull_v1__sim_honest": "simh311_q1_sim_rlr2_v1__sim_honest", "simh311_q3_sim_s4_cutfull_v1__sim_honest": "simh311_q3_sim_s4_v1__sim_honest",
            "simh_nm360_s4r_sim_rlr_cutfull_v1__sim_honest": "simh_nm360_s4r_sim_rlr_v1__sim_honest", "simh_t196fill_cutfull_v1__sim_honest": "simh_t196fill_v1__sim_honest",
            "ilold_oldsel_sim2_cutfull_v1__sim_attack": "ilold_oldsel_sim2_v1__sim_attack"}
    for n, o in want.items():
        assert original(n, ok, o) == o
    assert original("simh311_q1_sim_cutfull_v1__sim_honest", ok, "simh311_q1_sim_v1_try2__sim_honest") == "simh311_q1_sim_v1_try2__sim_honest"
    for n, o in (("ep2_t196_cutfull_gate_v1__honest_a", "ep2_t196_v1__honest_a"), ("ep2_t196_v1__honest_a", "ep2_t196_v1__honest_a"),
                 ("ep2_t196_cutfull_v2__honest_a", "ep2_t196_v1__honest_a"), ("ep2_t196_cutfull__honest_a", "ep2_t196_v1__honest_a"),
                 ("ep2_t196_cutfull_v1__honest_c", "ep2_t196_v1__honest_c"), ("ep2_t196_cutfull_v1__honest_a", ""), ("ep2_t196_cutfull_v1__honest_a", "ep2_t196_v1__honest_b"),
                 ("simh311_q1_sim_rlr2_cutfull_v1__sim_honest", "simh311_q1_sim_rlr_v1__sim_honest"), ("su311_q1_main_cutfull_v1__attack", "su311_q1_main_v1__attack")):
        try:
            original(n, ok, o)
        except SystemExit:
            continue
        raise AssertionError(f"accepted {n} with original {o!r}")


_selftest()
