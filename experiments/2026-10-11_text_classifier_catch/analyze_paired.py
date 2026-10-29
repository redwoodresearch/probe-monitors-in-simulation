"""The probe's real − simulated gap minus the text classifier's, paired by draw (RULE.md, "Comparing the two
classifiers"). Local, $0. Both draw files use the same task resamples (run_score_onpolicy.task_weights, seed 0), so draw b
of one pairs with draw b of the other.

Per draw b = 1..500 and metric m (catch_frozen, auroc): d_b = [probe real->real − real->sim] − [text real->real −
real->sim]. A draw where either side fails is failed. 95% interval: the 2.5th / 97.5th percentiles of d_b, failed draws
at the floor (−2) for the lower and the ceiling (+2) for the upper (a difference of two gaps lies in [−2, 2]); more than
5% failed: invalid. Descriptive: the two classifiers' gaps differ only if the observed difference (draw 0) is defined
and this interval excludes 0; otherwise the comparison is inconclusive. The text file's seed, folds and calls must equal
the probe's, so the pairing holds.

  python3 analyze_paired.py <probe draws0-500 file> <text main draws0-500 file> --probe-calls <probe draws0-0 per-call file>
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
PROBE_SHA256 = "b8ddf7c20aecb210b90cb67544dcd7f9cdbae6ca9e9ead8f758fe405f895d18d"  # s3_manifest_scaleup196_v1.json on main
CALLS_SHA256 = "c2656658baaf3aa150f83f4d105e7572bfd854f98032831621419e1f8e8f36bf"  # its observed draw 0 with the call list
DRAWS, MAX_FAIL = 500, 0.05


def gap(d, metric):
    v = [((d["cells"].get(c) or {}).get("values") or {}).get(metric) for c in ("real->real", "real->sim")]
    bad = [c for c in ("real->real", "real->sim") if metric in ((d["cells"].get(c) or {}).get("failed") or {})]
    return None if bad or None in v else v[0] - v[1]


def main(probe: str, text: str, probe_calls: str):
    for f, want in ((probe, PROBE_SHA256), (probe_calls, CALLS_SHA256)):
        if hashlib.sha256(Path(f).read_bytes()).hexdigest() != want:
            raise SystemExit(f"{Path(f).name} does not match its manifest sha256")
    RP, RT = (json.loads(Path(f).read_text()) for f in (probe, text))
    for name, R in (("probe", RP), ("text", RT)):
        if sorted(d["draw"] for d in R["draws"]) != list(range(DRAWS + 1)):  # raw ids, so a duplicate fails
            raise SystemExit(f"the {name} file must hold draws 0..{DRAWS} exactly once")
    calls = json.loads(Path(probe_calls).read_text())["meta"]  # the probe's main draw file holds no call list
    if RT["config"]["seed"] != RP["config"]["seed"] or RT["folds"] != RP["folds"] or RT["meta"] != calls:
        raise SystemExit("the text file's seed, folds or calls differ from the probe's: draws would not pair")
    P, T = ({d["draw"]: d for d in R["draws"]} for R in (RP, RT))
    out = {"probe_file": Path(probe).name, "probe_sha256": PROBE_SHA256, "text_file": Path(text).name,
           "text_sha256": hashlib.sha256(Path(text).read_bytes()).hexdigest(), "draws": DRAWS, "metrics": {}}
    for m in ("catch_frozen", "auroc"):
        d = [None if gap(P[b], m) is None or gap(T[b], m) is None else gap(P[b], m) - gap(T[b], m) for b in range(1, DRAWS + 1)]
        v = np.array([np.nan if x is None else x for x in d])
        failed = np.isnan(v)
        lo, hi = float(np.percentile(np.where(failed, -2, v), 2.5)), float(np.percentile(np.where(failed, 2, v), 97.5))
        obs = None if gap(P[0], m) is None or gap(T[0], m) is None else gap(P[0], m) - gap(T[0], m)
        out["metrics"][m] = {"observed_probe_gap": gap(P[0], m), "observed_text_gap": gap(T[0], m), "observed_difference": obs,
                             "ci95": [round(lo, 4), round(hi, 4)], "failed": int(failed.sum()), "invalid": bool(failed.mean() > MAX_FAIL),
                             "differ": ("inconclusive" if obs is None or failed.mean() > MAX_FAIL else bool(lo > 0 or hi < 0))}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"paired_probe_minus_text_{stamp}.json"
    with path.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps(out["metrics"], indent=1))
    print("->", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("probe")
    ap.add_argument("text")
    ap.add_argument("--probe-calls", required=True)
    main(**vars(ap.parse_args()))
