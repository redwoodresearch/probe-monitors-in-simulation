"""Pre-freeze reproduction check, npz part: the old all-tasks per-draw npz (key format v1) against the new one (v2), by key, on Modal (the npz
files never leave the Volume). Keys are normalised to one form (v1: '|'->'__'; v2: npzkey parts joined by '__', '|'->'_' inside a part; both:
'__' -> '_', the v2 markers '_pooled_gap_' and '_catch_' -> '_'); a normalised collision in either file stops the check. Every old key must map
to exactly one new key. Arrays under a combo or origin that holds 'without_S4' are the registered rule's expected change (51a612b29): reported
with counts, not compared exactly; every other array must be equal (NaN == NaN). New-only keys (per-origin gap_ arrays, the pooled_455 combo)
are listed by pattern.

  modal run npz_keymap_check.py --old xfer/fpr507prep/<old>.npz@<sha> --new xfer/fpr507prep/<new>.npz@<sha> --out results/checks/<name>.json \
      --expect-exact <N> --expect-nos4 <N>   (the expected key inventory; e.g. 1,992 and 1,992 for the 16:13 -> 18:40 pair, Codex r11's count;
      a run with fewer comparisons, an empty array, an unmapped old key or an unexpected new key fails and exits 1)
"""

import hashlib
import json
from pathlib import Path

import modal

app = modal.App("probe-shift_aswin-surya_sub-fpr507prep_npzkeymap_kill-2026-11-14")
vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", version=2)  # run_common455.VOLS["solavg"] (where the per-draw npz live)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")


def norm(k, v2):
    k = k.replace("__", "_")
    if v2:
        k = k.replace("_pooled_gap_", "_").replace("_catch_", "_")
    return k


@app.function(image=image, volumes={"/sol": vol}, cpu=2, memory=8192, timeout=1800)
def check(old, new, expect_exact, expect_nos4):
    import numpy as np

    out = {}
    z = {}
    for tag, ref in (("old", old), ("new", new)):
        p_, h_ = ref.rsplit("@", 1)
        b = (Path("/sol") / p_).read_bytes()
        if hashlib.sha256(b).hexdigest() != h_:
            raise SystemExit(f"{tag} npz sha256 differs")
        import io

        z[tag] = np.load(io.BytesIO(b))
    for tag in ("old", "new"):  # every array in both archives must be non-empty (matched or new-only)
        for k in z[tag].files:
            if z[tag][k].size == 0:
                raise SystemExit(f"{tag}: empty array {k}")
    nmap = {}
    for tag in ("old", "new"):
        m = {}
        for k in z[tag].files:
            nk = norm(k, tag == "new")
            if nk in m:
                raise SystemExit(f"{tag}: normalised key collision {k} / {m[nk]}")
            m[nk] = k
        nmap[tag] = m
    missing = sorted(set(nmap["old"]) - set(nmap["new"]))
    new_only = sorted(set(nmap["new"]) - set(nmap["old"]))
    exact_bad, s4_moved, s4_same, compared, empty = [], 0, 0, 0, []
    for nk, ko in nmap["old"].items():
        if nk not in nmap["new"]:
            continue
        a, b = z["old"][ko], z["new"][nmap["new"][nk]]
        if a.size == 0 or b.size == 0:
            empty.append(ko)
        same = a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))
        if "without_S4" in nk:
            s4_moved += not same
            s4_same += same
            continue
        compared += 1
        if not same:
            exact_bad.append(ko)
    pat = {}
    for nk in new_only:
        k = nmap["new"][nk]
        key = "pooled_455 combo" if "__pooled_455__" in k else "per-origin gap_" if "__gap_" in k else "other"
        pat[key] = pat.get(key, 0) + 1
    out = {"old": old, "new": new, "old_keys": len(z["old"].files), "new_keys": len(z["new"].files), "old_keys_without_a_new_key": missing[:40],
           "n_old_keys_without_a_new_key": len(missing), "compared_exact": compared, "exact_mismatch": exact_bad[:40], "n_exact_mismatch": len(exact_bad),
           "without_S4_arrays": {"changed": s4_moved, "unchanged": s4_same, "rule": "expected change (51a612b29: S4 out of attack estimation only)"},
           "new_only_by_pattern": pat, "empty_arrays": empty[:20], "expected_counts": {"exact": expect_exact, "without_S4": expect_nos4},
           "pass": (not missing and not exact_bad and not empty and pat.get("other", 0) == 0 and compared == expect_exact > 0
                    and s4_moved + s4_same == expect_nos4)}
    return out


@app.local_entrypoint()
def main(old: str, new: str, out: str, expect_exact: int, expect_nos4: int):
    res = check.remote(old, new, expect_exact, expect_nos4)
    dest = Path(__file__).resolve().parent / out
    blob = json.dumps(res, indent=1).encode()
    dest.open("xb").write(blob)
    print(json.dumps({k: v for k, v in res.items() if k not in ("old_keys_without_a_new_key", "exact_mismatch")} | {"file": out,
                      "sha256": hashlib.sha256(blob).hexdigest()}, indent=1))
    if not res["pass"]:
        raise SystemExit(1)
