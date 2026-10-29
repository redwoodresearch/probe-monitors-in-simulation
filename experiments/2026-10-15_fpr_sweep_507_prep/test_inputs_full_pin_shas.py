"""Synthetic tests of inputs_full's pin-sha STOP (09, 10-17; no git, no record): a null or non-64-hex pin sha256 of each kind, in
INPUTS_FULL itself, E12, an override or an inventory, STOPs both build and verify; clean shapes (the ones 110447's files use) pass.
inputs_full's pin() and load() are replaced by in-memory fakes, so nothing is read.

  uv run --no-project --python 3.12 python experiments/2026-10-15_fpr_sweep_507_prep/test_inputs_full_pin_shas.py
"""

import copy
import json
import tempfile
from pathlib import Path

import inputs_full as IFU

H = "a" * 64
PIN = {"commit": "c0ffee0", "path": "x.json", "sha256": H}
# one clean file per kind, in the shapes the keys-only schema scan of 110447 found
E12 = {"artifact": dict(PIN), "epoch1_eval_sha256": [H, H], "epoch1_members_sha256": H, "pools": [{"key": "k", "eval_sha256": H}],
       "manifests": [{"commit": "c", "path": "m.json", "manifest_sha256": H, "sha256": H}], "script_sha256": H, "files": [{"path": "p", "sha256": H}],
       "arms": [{"arm": "a", "staged_sha": H}], "n_eval_shas": 3, "e2_shas": 2}
OVR = {"arms": {"f1": {"p2/fibs__command.npz": {"path": "cutfull/x.npz", "sha256": H, "staged_shas": [H],
                                                 "gate": {"commit": "c", "path": "g.json", "file": "f", "sha256": H}}}}, "simh": {},
       "manifests_without_staged_sha": ["a_manifest_name.json"]}
INV = {"runs": [{"key": "r#e0", "task_set": "T196", "world": "run1", "original_folder": "o", "eval_sha256": H,
                 "uncut_planned": {"staged_sha256": H}, "registered_source": {"source": "s", "sha256": H}}]}
CONTENT = {"e12.json": E12, "ov.json": OVR, "inv.json": INV}


def m_clean():
    return {"e12": dict(PIN, path="e12.json"), "overrides": {"S_old|C": dict(PIN, path="ov.json")},
            "inventories": {"S_old": dict(PIN, path="inv.json")}, "gates": {"S_old|C": dict(PIN)}, "cohort": {"pending": "t"},
            "pools": {"S_old": {"H_B": {"n": 2, "e2_members_sha256": H, "physical_members_sha256": H}}}}


def fake_load(content):
    return lambda e, what: json.dumps(content[e["path"]]).encode()


def stops(fn):
    """True only for the pin-sha STOP; any other outcome (another STOP, a later KeyError on the minimal fixture, a return) is False."""
    try:
        fn()
    except SystemExit as x:
        return "pin sha256" in str(x)
    except (KeyError, TypeError, ValueError):
        return False
    return False


def verify_stops(m, content):
    IFU.load = fake_load(content)
    return stops(lambda: IFU.verify(m))


def build_stops(content, null_pin_path=None):
    """build() up to its pin-sha check: pins resolve through a fake pin(); null_pin_path gets a null sha256 in INPUTS_FULL itself."""
    IFU.load = fake_load(content)
    IFU.pin = lambda c, p: {"commit": c, "path": p, "sha256": None if p == null_pin_path else H}
    P = {"auroc_subset": [], "base": {"pending": "t"}, "overrides": {"S_old|C": {"commit": "c", "path": "ov.json"}}, "gates": {},
         "inventories": {"S_old": {"commit": "c", "path": "inv.json"}}, "fits": {}, "ood": {"pending": "t"}, "ood_signature": {"pending": "t"},
         "e12": {"commit": "c", "path": "e12.json"}} | {k: {"pending": "t"} for k in ("cohort", "exclusions", "waitlist", "census", "ilarms",
                                                                                         "relaunch_members", "cutmap")}
    with tempfile.TemporaryDirectory() as d:
        pf = Path(d) / "pins.json"
        pf.write_text(json.dumps(P))
        return stops(lambda: IFU.build(str(pf), "", True))


def planted(path_in_content, fn):
    c = copy.deepcopy(CONTENT)
    fn(c[path_in_content])
    return c


cases = {
    "clean shapes: no problem (all kinds)": IFU.pin_sha_problems(m_clean()) == [] and all(IFU.pin_sha_problems(v) == [] for v in CONTENT.values()),
    "clean: require_pin_shas passes": (lambda: (setattr(IFU, "load", fake_load(CONTENT)), IFU.require_pin_shas(m_clean()), True)[2])(),
}
# planted nulls / bad shapes, one per pin kind, each must STOP verify AND build
PLANTS = {
    "E12 {commit,path,sha256} pin null": ("e12.json", lambda o: o["artifact"].__setitem__("sha256", None)),
    "E12 epoch1_eval_sha256 element null": ("e12.json", lambda o: o["epoch1_eval_sha256"].__setitem__(1, None)),
    "E12 epoch1_eval_sha256 empty list": ("e12.json", lambda o: o.__setitem__("epoch1_eval_sha256", [])),
    "E12 eval_sha256 non-hex": ("e12.json", lambda o: o["pools"][0].__setitem__("eval_sha256", "abc")),
    "E12 manifest_sha256 uppercase hex": ("e12.json", lambda o: o["manifests"][0].__setitem__("manifest_sha256", "A" * 64)),
    "override entry {path,sha256} null": ("ov.json", lambda o: o["arms"]["f1"]["p2/fibs__command.npz"].__setitem__("sha256", None)),
    "override gate pin sha256 missing": ("ov.json", lambda o: o["arms"]["f1"]["p2/fibs__command.npz"]["gate"].pop("sha256")),
    "inventory eval_sha256 null": ("inv.json", lambda o: o["runs"][0].__setitem__("eval_sha256", None)),
    "inventory staged_sha256 null": ("inv.json", lambda o: o["runs"][0]["uncut_planned"].__setitem__("staged_sha256", None)),
    "inventory sibling-less sha256 63 hex": ("inv.json", lambda o: o["runs"][0]["registered_source"].__setitem__("sha256", "a" * 63)),
    "inventory unknown *_sha256 field": ("inv.json", lambda o: o["runs"][0].__setitem__("new_sha256", H)),
    # second review R1 (10-17): sha fields not named *sha256
    "override staged_shas [H, null]": ("ov.json", lambda o: o["arms"]["f1"]["p2/fibs__command.npz"].__setitem__("staged_shas", [H, None])),
    "override staged_shas [null]": ("ov.json", lambda o: o["arms"]["f1"]["p2/fibs__command.npz"].__setitem__("staged_shas", [None])),
    "override staged_shas empty": ("ov.json", lambda o: o["arms"]["f1"]["p2/fibs__command.npz"].__setitem__("staged_shas", [])),
    "E12 arm staged_sha null": ("e12.json", lambda o: o["arms"][0].__setitem__("staged_sha", None)),
    "unknown key new_sha": ("inv.json", lambda o: o["runs"][0].__setitem__("new_sha", H)),
    "unknown key foo_shas": ("ov.json", lambda o: o.__setitem__("foo_shas", [H])),
    "unknown key Run_SHA (case-insensitive)": ("e12.json", lambda o: o.__setitem__("Run_SHA", H)),
    "a pin with a null commit": ("e12.json", lambda o: o["artifact"].__setitem__("commit", None)),
}
for name, (f, fn) in PLANTS.items():
    c = planted(f, fn)
    cases[f"verify STOPs: {name}"] = verify_stops(m_clean(), c)
    cases[f"build STOPs: {name}"] = build_stops(c)
# planted in INPUTS_FULL itself
m_null = m_clean()
m_null["gates"]["S_old|C"]["sha256"] = None
cases["verify STOPs: INPUTS_FULL gate pin sha256 null"] = verify_stops(m_null, CONTENT)
m_pools = m_clean()
m_pools["pools"]["S_old"]["H_B"]["e2_members_sha256"] = None
cases["verify STOPs: INPUTS_FULL e2_members_sha256 null"] = verify_stops(m_pools, CONTENT)
cases["build STOPs: INPUTS_FULL's own resolved pin with a null sha256 (inventory pin)"] = build_stops(CONTENT, null_pin_path="inv.json")
cases["build passes the pin-sha check when clean (then STOPs later for another reason, not pin sha256)"] = not build_stops(CONTENT)

for name, ok in cases.items():
    print(("PASS " if ok else "FAIL ") + name)
assert all(cases.values()), "pin-sha tests failed"
print(f"{len(cases)} checks pass")
