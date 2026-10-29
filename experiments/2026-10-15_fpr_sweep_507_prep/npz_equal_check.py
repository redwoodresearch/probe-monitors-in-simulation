"""Byte-identity gate for an addendum record's per-draw npz (09 via 2e 23:3x): every array of the reference npz must exist in the new npz under the
same key and be EQUAL (NaN == NaN, same shape, non-empty); the new npz may add only keys holding --new-label. Runs on Modal (the npz never leave
the Volume); exit 1 on any failure.

  modal run npz_equal_check.py --ref …@<sha> --new …@<sha> --new-label "own_all_LOO_L48 C" --template-label "own_all_L48 C" --out results/checks/<name>.json
(the new label must add exactly the template label's key set, same shapes, no all-NaN array where the template's has a finite value)
  --new-label / --template-label may be comma-separated lists, paired in order (an all-tasks npz holds both readouts: "own_all_LOO_L48 C,own_all_LOO_L48 R").
  --drop-combo pooled_all: for an all-tasks record (key part 2 = combo), the template keys under that combo are not expected for the new label
  (own_all_LOO: pooled_all reads unavailable, 09 via 507-prep 00:1x); any new-label key under it fails.
"""

import hashlib
import json
from pathlib import Path

import modal

app = modal.App("probe-shift_aswin-surya_sub-fpr507prep_npzequal_kill-2026-11-14")
vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", version=2)  # run_common455.VOLS["solavg"]
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")


@app.function(image=image, volumes={"/sol": vol}, cpu=2, memory=8192, timeout=1800)
def check(ref, new, new_label, template_label, drop_combo=""):
    import io

    import numpy as np

    z = {}
    for tag, r in (("ref", ref), ("new", new)):
        p_, h_ = r.rsplit("@", 1)
        b = (Path("/sol") / p_).read_bytes()
        if hashlib.sha256(b).hexdigest() != h_:
            raise SystemExit(f"{tag} npz sha256 differs")
        z[tag] = np.load(io.BytesIO(b))
    pairs = list(zip([x.strip().replace(" ", "_") for x in new_label.split(",")], [x.strip().replace(" ", "_") for x in template_label.split(",")], strict=True))
    rk, nk = set(z["ref"].files), set(z["new"].files)
    # the exact key set the new labels must add: each template label's keys (same probe layout, same readouts) with the label part replaced
    expected_new, n_tpl, per_pair = {}, 0, {}
    for tag_, tpl_ in pairs:
        tpl_keys = [k for k in rk if k.startswith(tpl_ + "__")]
        n_tpl += len(tpl_keys)
        e_ = {tag_ + k[len(tpl_):]: k for k in tpl_keys if not (drop_combo and k.split("__")[1] == drop_combo)}
        per_pair[tag_] = {"template_keys": len(tpl_keys), "expected": len(e_), "dropped": len(tpl_keys) - len(e_)}
        expected_new |= e_
    dropped = n_tpl - len(expected_new)
    missing = sorted(rk - nk)
    mine_ = lambda k: any(k.startswith(t_ + "__") for t_, _ in pairs)  # noqa: E731
    extra_bad = sorted(k for k in nk - rk if not mine_(k))
    extra_ok = sorted(k for k in nk - rk if mine_(k))
    bad, empty = [], []
    for k in sorted(rk & nk):
        a, b_ = z["ref"][k], z["new"][k]
        if a.size == 0 or b_.size == 0:
            empty.append(k)
        if a.shape != b_.shape or not bool(np.array_equal(a, b_, equal_nan=True)):
            bad.append(k)
    empty += [k for k in extra_ok if z["new"][k].size == 0]
    wrong_set = sorted(set(extra_ok) ^ set(expected_new))
    wrong_shape = [k for k in extra_ok if k in expected_new and z["new"][k].shape != z["ref"][expected_new[k]].shape]
    all_nan = [k for k in extra_ok if k in expected_new and z["new"][k].size and not np.isfinite(z["new"][k]).any()
               and np.isfinite(z["ref"][expected_new[k]]).any()]
    all_nan_both = sum(1 for k in extra_ok if k in expected_new and z["new"][k].size and not np.isfinite(z["new"][k]).any()) - len(all_nan)
    return {"ref": ref, "new": new, "ref_keys": len(rk), "new_keys": len(nk), "compared_equal": len(rk & nk) - len(bad), "mismatch": bad[:40],
            "n_mismatch": len(bad), "missing_in_new": missing[:20], "unexpected_new_keys": extra_bad[:20], "new_label_keys": len(extra_ok),
            "empty_arrays": empty[:20], "expected_new_label_keys": len(expected_new), "new_label_key_set_diff": wrong_set[:20],
            "new_label_wrong_shape": wrong_shape[:20], "new_label_all_nan": all_nan[:20], "all_nan_in_template_too": all_nan_both,
            "drop_combo": drop_combo or None, "template_keys_dropped": dropped, "per_pair": per_pair,
            "pass": (not bad and not missing and not extra_bad and not empty and len(rk & nk) > 0 and len(expected_new) > 0 and not wrong_set
                     and not wrong_shape and not all_nan and len(per_pair) == len(pairs)
                     and all(v["expected"] > 0 and (not drop_combo or v["dropped"] > 0) for v in per_pair.values()))}


@app.local_entrypoint()
def main(ref: str, new: str, new_label: str, template_label: str, out: str, drop_combo: str = ""):
    res = check.remote(ref, new, new_label, template_label, drop_combo)
    dest = Path(__file__).resolve().parent / out
    blob = json.dumps(res, indent=1).encode()
    dest.open("xb").write(blob)
    print(json.dumps({k: v for k, v in res.items() if k not in ("mismatch",)} | {"file": out, "sha256": hashlib.sha256(blob).hexdigest()}, indent=1))
    if not res["pass"]:
        raise SystemExit(1)
