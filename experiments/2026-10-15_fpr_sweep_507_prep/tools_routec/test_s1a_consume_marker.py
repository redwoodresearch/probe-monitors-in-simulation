"""Fixture checks of s1a_consume_marker.jq (10-18): both marker kinds, each presented to both gates, and single-field negatives. jq -f on
synthetic markers only; no real marker is read."""

import json
import subprocess
from pathlib import Path

JQ = Path(__file__).resolve().parent / "s1a_consume_marker.jq"
PS, CAT = "a" * 64, "b" * 64
E = "experiments/2026-10-15_fpr_sweep_507_prep/results/"


def ok(marker, kind, pickles):
    r = subprocess.run(["jq", "-e", "--arg", "kind", kind, "--argjson", "pickles", json.dumps(pickles), "-f", str(JQ)],
                       input=json.dumps(marker), capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() == "true"


def marker(kind, pickles, **kw):
    return {"signoff": kind, "signed": True, "by": "09", "time": "2026-10-18 12:30 PDT", "notes": "n", "items": {"3": "PASS"},
            "covered": [f"{E}routec_pscat_specs/x_{i}.pkl@{p}" for i, p in enumerate(pickles)] + [f"{E}spec_diff/r.json@{'c' * 64}"]} | kw


def main():
    full, cat = marker("S1a-consume", [PS, CAT]), marker("S1a-consume-CAT", [CAT])
    cases = {
        "S1A3M: the full marker with both launched pickles passes": ok(full, "S1a-consume", [PS, CAT]),
        "S1A3MC: the CAT-only marker with the CAT pickle passes": ok(cat, "S1a-consume-CAT", [CAT]),
        "S1A3MC, positive control: the FULL marker presented to it fails": not ok(full, "S1a-consume-CAT", [CAT]),
        "S1A3M, positive control: the CAT-only marker presented to it fails": not ok(cat, "S1a-consume", [PS, CAT]),
        "S1A3MC: a CAT-only marker that also covers a PS pickle fails": not ok(marker("S1a-consume-CAT", [PS, CAT]), "S1a-consume-CAT", [CAT]),
        "S1A3M: a full marker missing the PS pickle fails": not ok(marker("S1a-consume", [CAT]), "S1a-consume", [PS, CAT]),
        "an extra item (S1a-launch's 1) fails": not ok(cat | {"items": {"1": "PASS", "3": "PASS"}}, "S1a-consume-CAT", [CAT]),
        "signed false fails": not ok(cat | {"signed": False}, "S1a-consume-CAT", [CAT]),
        "another signer fails": not ok(cat | {"by": "2e"}, "S1a-consume-CAT", [CAT]),
        "an unknown kind fails even when signoff matches it": not ok(marker("S1a-launch", [CAT]), "S1a-launch", [CAT]),
    }
    for name, v in cases.items():
        print(("PASS " if v else "FAIL ") + name)
    assert all(cases.values()), "s1a_consume_marker tests failed"
    print(f"{len(cases)} checks pass")


if __name__ == "__main__":
    main()
