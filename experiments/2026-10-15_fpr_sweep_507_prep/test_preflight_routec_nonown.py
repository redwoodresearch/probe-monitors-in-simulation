"""Synthetic tests of 09's ruling (b) (10-18) in preflight_routec.analyze, run through the unchanged gate (gate_spec_diff_routec.gate). No
Modal call, no record, no score: fabricated specs and key arrays. The pre-flight as reviewed (blob a345aec8, read from git) is the "before".
Cases, each with its exact outcome:
- a pooled S_old job carrying another label's tagged entry (the Sol (a) C file inside an external family's row, rp L6748 / full_override
  L474): before, a CLEAR record whose zero_unit_by_file lacks that file, and the gate STOPs at L61; after, a CLEAR record listing it, and
  the gate PASSes on the spec the runner would build from it (full_override.s_old_simh L256: zero_unit_ok = the pinned list or []);
- the non-own file is annotated separately (labels, n_admitted, n_with_rows, n_zero_unit, the zero-unit keys by name), its keys are in the
  union, and the own-only part of the union equals the reviewed pre-flight's output;
- 09's tests: a tagged entry whose file is missing from the extended list: the gate STOPs; a planted wrong zero_unit (a key that HAS rows):
  the gate STOPs; a per-set (PS-shaped) spec, where every tagged file is some job's own: the record is IDENTICAL to the
  reviewed output (zero_unit_by_file_nonown is omitted when empty);
- 09's addition: one file own for one job and non-own for another, with different zero-unit sets: a DEFECT; with identical sets: none.
- a03f5d93's O1-O3 (10-18): the fake reader checks each read's sha against the entry's pin (a read without it dies); a non-own file
  whose fold array has fold -1 units gives no drop and no defect on a fold-model probe; one non-own file met twice with different
  zero-unit sets is a DEFECT (identical sets: none, listed once).

  uv run --no-project --python 3.12 --with modal==1.6.1 python experiments/2026-10-15_fpr_sweep_507_prep/test_preflight_routec_nonown.py [--pf <file>]
  (--pf: test this pre-flight file instead; the mutation runs: the test must FAIL)
"""
import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_spec_diff_routec as G  # noqa: E402

REVIEWED = "a345aec8443922a6b17270bcf5a5fc8baab8d1b0"  # preflight_routec.py as reviewed (package 4)
TASKS = [f"t{i}_task" for i in range(21, 41)]
KEYS = [f"simh_pilot|{t}" for t in TASKS]
MAIN_T = {t: f"main_{t}" for t in TASKS}
ROOT = "simh_q4f_v1_pilot__sim_honest"
PIN = f"meta/full_length/s_old_simh_extra_roots/{ROOT}/zero_unit_list"
PIN_ = {"path": "results/routec_preflight/p.json", "sha256": "9" * 64}
B1E = {"batch": "b1", "root": "simh_q4f_v1_full_b1__sim_honest", "path": "b1.npz", "sha256": "b" * 64}


def load(src, name):
    d = Path(tempfile.mkdtemp(prefix="pf_nonown_mod_"))
    (d / f"{name}.py").write_bytes(src)
    s = importlib.util.spec_from_file_location(name, d / f"{name}.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def ent(pf, sha, path, admitted=KEYS):
    return {"batch": "simh_q4f_v1_pilot", "root": ROOT, "path": path, "sha256": sha, pf.TAG: True, "admitted": list(admitted)}


def probe(label):
    return {"label": label, "folds": 1}  # single-model: no fold path, so only the zero-unit lists are in play


def job(lb, simh_old, **kw):
    return (lb, {"origin": "S_old", "probes": [probe(lb)], "simh_old": simh_old} | kw, None)


SHAS = {"ext.npz": "e" * 64, "extr.npz": "f" * 64, "sol.npz": "5" * 64}  # a03f5d93's O1 (M1): the fake reader checks each read's sha, as
# preflight_remote's read_kf does (sha256 of the bytes against the pin), so a read without the entry's own sha fails here too


def reader(files, shas=None):
    def read_kf(v, path, sha):
        if v != "conv" or (shas or SHAS).get(path) != sha:
            raise SystemExit(f"read_kf: {path} read without its pinned sha256")
        return files[path]
    return read_kf


def tagged_at(spec, pf, zu):
    """What the runner would build from a pre-flight's list (full_override.s_old_simh L256: each tagged entry carries its own file's pinned
    list, [] when none), seen as spec_diff's routec_entries (L170-178) on side a (no pin) and side b (the pin)."""
    a_, b_, ob = [], [], []
    for i, (_, cfg, _) in enumerate(spec["jobs"]):
        for lb, es in sorted((cfg.get("simh_old") or {}).items()):
            for n, e in enumerate(es):
                if e.get(pf.TAG):
                    at = f"jobs/{i}/1/simh_old/{lb}/{n}"
                    z = sorted(zu.get(e["sha256"], []))
                    a_.append({"at": at, "file_sha256": e["sha256"], "zero_unit_ok": []})
                    b_.append({"at": at, "file_sha256": e["sha256"], "zero_unit_ok": z})
                    ob += [f"{at}/zero_unit_ok/{k}" for k in range(len(z))]
    return a_, b_, ob


def diff_rec(a_, b_, ob):
    ob = ob + [PIN]
    return {"a": {"sha256": "a" * 64}, "b": {"sha256": "c" * 64}, "only_a": [], "only_b": ob, "changed": [], "n_differences": len(ob),
            "a_pins": [], "b_pins": [{"at": PIN} | PIN_], "a_routec_entries": a_, "b_routec_entries": b_,
            "b_meta": {"git": "h" * 40, "dest": "xfer/fpr507prep/records/results/x.json", "pooled": True}}


def gate_on(spec, pf, rec, zu=None):
    a_, b_, ob = tagged_at(spec, pf, rec["zero_unit_by_file"] if zu is None else zu)
    return G.gate(diff_rec(a_, b_, ob), rec | {"spec": {"sha256": "a" * 64}}, PIN_)


def main(pf_file):
    real = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()
    old = load(subprocess.run(["git", "-C", real, "cat-file", "-p", REVIEWED], capture_output=True, check=True).stdout, "pf_reviewed")
    new = load(Path(pf_file or HERE / "preflight_routec.py").read_bytes(), "pf_under_test")
    res = {}
    ea, es = ent(new, "e" * 64, "ext.npz"), ent(new, "5" * 64, "sol.npz")
    files = {"ext.npz": (KEYS, None), "sol.npz": (KEYS[1:], None)}  # the Sol (a) C pilot file has no rows for KEYS[0]: one zero-unit run
    pooled = {"jobs": [job("Ext C", {"Ext C": [B1E, ea], "Sol (a) C": [B1E, es]}, pw_combo="pooled_all_origins", full_as="Ext|uncut C"),
                       job("Ext C", {"Ext C": [B1E, ea], "Sol (a) C": [B1E, es]}, pw_combo="pooled_all_origins_without_S4", full_as="Ext|uncut C")]}
    r_old = old.analyze(json.loads(json.dumps(pooled)), reader(files), MAIN_T, ())
    r_new = new.analyze(json.loads(json.dumps(pooled)), reader(files), MAIN_T, ())
    g_old, g_new = gate_on(pooled, new, r_old), gate_on(pooled, new, r_new)
    res["before (the reviewed pre-flight): CLEAR, its list lacks the non-own file, and the gate STOPs at L61"] = \
        r_old["defects"] == [] and "5" * 64 not in r_old["zero_unit_by_file"] and not g_old["pass"] \
        and any(p.startswith("tagged entries whose zero_unit_ok is not the pinned list") for p in g_old["problems"])
    res["after: CLEAR, the union lists the non-own file, and the gate PASSes"] = \
        r_new["defects"] == [] and r_new["zero_unit_by_file"].get("5" * 64) == [KEYS[0]] and g_new["pass"] \
        and r_new["expected_zero_unit_ok_paths"] == 2
    res["the non-own file is annotated separately: labels, n_admitted 20, n_with_rows 19, n_zero_unit 1, its key by name"] = \
        r_new.get("zero_unit_by_file_nonown") == {"5" * 64: {"labels": ["Sol (a) C"], "n_admitted": 20, "n_with_rows": 19, "n_zero_unit": 1,
                                                             "zero_unit": [KEYS[0]]}}
    res["the non-own keys are in the union, and the own-only part of the union equals the reviewed output"] = \
        all(r_new["zero_unit_by_file"].get(h) == x["zero_unit"] for h, x in r_new.get("zero_unit_by_file_nonown", {}).items()) \
        and {h: v for h, v in r_new["zero_unit_by_file"].items() if h not in r_new.get("zero_unit_by_file_nonown", {})} == r_old["zero_unit_by_file"]
    a_, b_, ob = tagged_at(pooled, new, r_new["zero_unit_by_file"])
    extra = {"at": "jobs/1/1/simh_old/Other C/1", "file_sha256": "7" * 64, "zero_unit_ok": []}
    g_miss = G.gate(diff_rec(a_ + [extra | {"zero_unit_ok": []}], b_ + [extra], ob), r_new | {"spec": {"sha256": "a" * 64}}, PIN_)
    res["09 (i): a tagged entry whose file is missing from the extended list: the gate STOPs"] = not g_miss["pass"] \
        and any(p.startswith("tagged entries whose zero_unit_ok is not the pinned list") for p in g_miss["problems"])
    g_wrong = gate_on(pooled, new, r_new, zu=r_new["zero_unit_by_file"] | {"5" * 64: [KEYS[0], KEYS[1]]})  # KEYS[1] HAS rows in sol.npz
    res["09 (ii): a planted wrong zero_unit (a key that has rows): the gate STOPs"] = not g_wrong["pass"] \
        and any(p.startswith("tagged entries whose zero_unit_ok is not the pinned list") for p in g_wrong["problems"])
    per_set = {"jobs": [job("Ext C", {"Ext C": [B1E, ea], "Sol (a) C": [B1E, es]}), job("Sol (a) C", {"Ext C": [B1E, ea], "Sol (a) C": [B1E, es]}),
                        ("Sol (a) C", {"origin": "S_paper493", "probes": [probe("Sol (a) C")], "simh_old": {"Sol (a) C": [B1E]}}, None)]}
    p_old = old.analyze(json.loads(json.dumps(per_set)), reader(files), MAIN_T, ())
    p_new = new.analyze(json.loads(json.dumps(per_set)), reader(files), MAIN_T, ())
    res["09 (iii): a per-set (PS-shaped) spec: the record is identical to the reviewed output"] = \
        p_old["defects"] == [] and p_new == p_old and json.dumps(p_new, indent=1) == json.dumps(p_old, indent=1)
    es2 = ent(new, "5" * 64, "sol.npz", admitted=KEYS[1:])  # the same file, another admitted list: its zero-unit set differs (none vs KEYS[0])
    both = {"jobs": [job("Sol (a) C", {"Sol (a) C": [B1E, es]}), job("Ext C", {"Ext C": [B1E, ea], "Sol (a) C": [B1E, es2]})]}
    r_both = new.analyze(json.loads(json.dumps(both)), reader(files), MAIN_T, ())
    res["09's addition: a file own for one job and non-own for another, with different zero-unit sets: DEFECT"] = \
        any("own for one job and non-own for another" in d for d in r_both["defects"])
    same = {"jobs": [job("Sol (a) C", {"Sol (a) C": [B1E, es]}), job("Ext C", {"Ext C": [B1E, ea], "Sol (a) C": [B1E, es]})]}
    r_same = new.analyze(json.loads(json.dumps(same)), reader(files), MAIN_T, ())
    res["control: the same file on both sides with identical sets: no defect, and it is not listed as non-own"] = \
        r_same["defects"] == [] and "zero_unit_by_file_nonown" not in r_same
    res["analyze does not mutate the spec"] = (lambda s: (new.analyze(s, reader(files), MAIN_T, ()), s == json.loads(json.dumps(pooled)))[1])(
        json.loads(json.dumps(pooled)))
    # a03f5d93's O2 (M2b): a non-own file gives no drop even when its fold array has fold -1 units (the own file has none), on a fold-model probe
    folded = {"ext.npz": (KEYS, [1] * 20), "sol.npz": (KEYS[1:], [-1] + [1] * 18)}
    pf5 = {"jobs": [(lb, cfg | {"probes": [{"label": lb, "folds": 5}]}, b) for lb, cfg, b in pooled["jobs"]]}
    r_f = new.analyze(json.loads(json.dumps(pf5)), reader(folded), MAIN_T, ())
    res["a03f5d93 O2: a non-own file with fold -1 units gives no drop and no defect (fold-model probe); its zero-unit list is still listed"] = \
        r_f["defects"] == [] and r_f["drops"] == [] and r_f["zero_unit_by_file"].get("5" * 64) == [KEYS[0]] \
        and r_f["gate_view"]["drops"] == []
    # a03f5d93's O3 (M6): one non-own file met twice (two jobs) with different zero-unit sets: a DEFECT
    ear = ent(new, "f" * 64, "extr.npz")
    twice = {"jobs": [job("Ext C", {"Ext C": [B1E, ea], "Sol (a) C": [B1E, es]}),
                      job("Ext R", {"Ext R": [B1E, ear], "Sol (a) C": [B1E, es2]})]}
    r_tw = new.analyze(json.loads(json.dumps(twice)), reader(files | {"extr.npz": (KEYS, None)}), MAIN_T, ())
    res["a03f5d93 O3: one non-own file met twice with different zero-unit sets: DEFECT"] = \
        any("two zero-unit lists among its non-own entries" in d for d in r_tw["defects"])
    r_tw_ok = new.analyze(json.loads(json.dumps({"jobs": [twice["jobs"][0], job("Ext R", {"Ext R": [B1E, ear], "Sol (a) C": [B1E, es]})]})),
                          reader(files | {"extr.npz": (KEYS, None)}), MAIN_T, ())
    res["control: the same non-own file met twice with identical sets: no defect, listed once with both jobs' label"] = \
        r_tw_ok["defects"] == [] and list(r_tw_ok.get("zero_unit_by_file_nonown", {})) == ["5" * 64]
    # 09's extra-key rule (relayed by 2e 10-18; the reviewers' "(iii)"): a planted extra key (outside the admitted 20) in a non-own file STOPs, the extra key named
    r_x = new.analyze(json.loads(json.dumps(pooled)), reader(files | {"sol.npz": (KEYS[1:] + ["simh_pilot|PLANTED_EXTRA"], None)}), MAIN_T, ())
    res["09's extra-key rule: a planted extra key in a non-own file: DEFECT, the key named"] = \
        any("keys outside the 20 admitted" in d and "simh_pilot|PLANTED_EXTRA" in d for d in r_x["defects"])
    # the same rule: the clean synthetic shape of A (rp L6748 x full_override L474 / L481 / L484): 16 registered-pair rows (4 EXTERNAL x C/R x 2
    # combos), each with the 24 registered tagged entries, and 20 retrain rows (5 PW_FAMS retrains x C/R x 2), each with its base's 2
    reg = ["Sol (a)", "FIBS", "harmfulness", "refusal", "high-stakes", "own_all_rs_L48", "own_all_rs25_L48", "own_all_rs10_L48", "c_asis",
           "own_frac10_L48", "own_frac25_L48", "own_frac100_L48"]
    ext = ["FIBS", "harmfulness", "refusal", "high-stakes"]
    rts = {"sol_a_uncut": "Sol (a)", "own_all_rs_uncut_L48": "own_all_rs_L48", "own_all_rs25_uncut_L48": "own_all_rs25_L48",
           "own_all_rs10_uncut_L48": "own_all_rs10_L48", "c_asis_uncut": "c_asis"}
    shas, kf, n = {}, {}, 0

    def pent(name, own):
        nonlocal n
        n += 1
        p, h = f"{name}.npz", f"{n:064x}"
        shas[p], kf[p] = h, ((KEYS if own else KEYS[1:]), None)  # a non-own file has one zero-unit run, an own file none
        return ent(new, h, p)
    reg_so = {f"{f} {ro}": [B1E, pent(f"reg_{f}_{ro}", f in ext)] for f in reg for ro in "CR"}
    rt_so = {r: {f"{b} {ro}": [B1E, pent(f"rt_{r}_{ro}", True)] for ro in "CR"} for r, b in rts.items()}
    combos = ("pooled_all_origins", "paper_cells")
    shape = {"jobs": [job(f"{x} {ro}", reg_so, pw_combo=c, full_as=f"{x}|uncut {ro}") for x in ext for ro in "CR" for c in combos]
             + [job(f"{b} {ro}", rt_so[r], pw_combo=c, full_as=f"{r} {ro}") for r, b in rts.items() for ro in "CR" for c in combos]}
    s_old = old.analyze(json.loads(json.dumps(shape)), reader(kf, shas), MAIN_T, ())
    s_new = new.analyze(json.loads(json.dumps(shape)), reader(kf, shas), MAIN_T, ())
    g_s_old, g_s_new = gate_on(shape, new, s_old), gate_on(shape, new, s_new)
    res["09's extra-key rule: A's clean synthetic shape (36 jobs, 34 files = 18 own + 16 non-own): CLEAR and the gate PASSes; before, 256 at L61"] = \
        len(shape["jobs"]) == 36 and len(s_new["jobs"]) == 36 and s_new["defects"] == [] and len(s_new["zero_unit_by_file"]) == 34 \
        and len(s_new.get("zero_unit_by_file_nonown", {})) == 16 and s_new["expected_zero_unit_ok_paths"] == 256 and g_s_new["pass"] \
        and len(s_old["zero_unit_by_file"]) == 18 and not g_s_old["pass"] \
        and any(p.startswith("tagged entries whose zero_unit_ok is not the pinned list for their file: 256,") for p in g_s_old["problems"])
    for k, v in res.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("ALL PASS" if all(res.values()) else "FAILED")
    if not all(res.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pf", default=None)
    main(ap.parse_args().pf)
