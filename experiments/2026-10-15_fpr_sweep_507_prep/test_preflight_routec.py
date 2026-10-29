"""Synthetic tests of preflight_routec.analyze (no Modal, no git, no score): each STOP risk is a defect, zero-unit runs are listed and are not,
fold -1 drops are listed with their rule, and the constants match the runner's.

  uv run python experiments/2026-10-15_fpr_sweep_507_prep/test_preflight_routec.py
"""

import ast
from pathlib import Path

import full_override as FO
import preflight_routec as PF

HERE = Path(__file__).resolve().parent
TASKS = [f"t{i}_task" for i in range(21, 41)]
KEYS = [f"simh_pilot|{t}" for t in TASKS]
MAIN_T = {t: f"main_{t}" for t in TASKS} | {"t100_task": "main_t100"}
ENT = {"batch": "simh_q4f_v1_pilot", "root": next(iter(FO.S_OLD_EXTRA)), "path": "pilot.npz", "sha256": "p" * 64, PF.TAG: True, "admitted": KEYS}
B1E = {"batch": "b1", "root": "simh_q4f_v1_full_b1__sim_honest", "path": "b1.npz", "sha256": "b" * 64}
REAL = {"realF": {"path": "real.npz", "sha256": "r" * 64, "vol": "conv"}}


def probe(label="Sol (a) C", folds=5, family=None):
    return {"label": label, "folds": folds, "family": family, "old": {"kind": "uncut_list", "files": REAL}}


def spec(cfg_old=None, extra_jobs=()):
    co = cfg_old or {"origin": "S_old", "probes": [probe()], "simh_old": {"Sol (a) C": [B1E, ENT]}}
    sp = {"origin": "S_paper493", "probes": [probe()], "simh_old": {"Sol (a) C": [B1E]}}
    return {"jobs": [("Sol (a) C", co, None), ("Sol (a) C", sp, None), ("Sol (a) C", {"origin": "S_new", "probes": [probe()]}, None)] + list(extra_jobs)}


def reader(files):
    def read_kf(v, path, sha):
        assert v == "conv"
        return files[path]
    return read_kf


FOLD_OK = {"pilot.npz": (KEYS * 2, [1] * 40), "real.npz": ([f"honest_b|{t}" for t in TASKS], [2] * 20)}


def run(sp=None, files=None, nfd=FO.NO_FOLD_DROPS):  # Claude r1 R1: the REAL tuple
    return PF.analyze(sp or spec(), reader(files or FOLD_OK), MAIN_T, nfd)


def cap_and_excl():
    """Claude r1 NB: rp's EXC is the exclusion map only for S_new / union / S_paper493 (and new mains), so S_old's is empty; the cap filters
    units by their rank within the run (< 30), so every run with a row keeps one."""
    src = (HERE / "run_peraction311.py").read_text()
    exc = 'EXC = (cfg["excl"] if orig in ("S_new", "union", "S_paper493") else {}) | (dict(cfg.get("excl_nm", {})) if NM else {})'
    cap = 'kH = rank_in_run(H["run"]) < 30'
    nm = 'NM = orig in ("S_newmains", "union")'
    return src.count(exc) == 1 and src.count(cap) == 1 and src.count(nm) == 1


def cfg_with(**kw):
    return {"origin": "S_old", "probes": [probe()], "simh_old": {"Sol (a) C": [B1E, ENT]}} | kw


r0 = run()
cases = {
    "clean: one S_old job, no defect, no drop, no zero-unit run": r0["defects"] == [] and r0["drops"] == [] and len(r0["jobs"]) == 1
    and r0["jobs"][0]["pilot_keys_with_rows"] == 20 and r0["jobs"][0]["zero_unit"] == [],
    "an extra key in the pilot file: DEFECT": bool(run(files=FOLD_OK | {"pilot.npz": (KEYS + ["simh_pilot|NOT_A_PILOT_TASK"], [1] * 21)})["defects"]),
    "another namespace: DEFECT (extra keys)": bool(run(files=FOLD_OK | {"pilot.npz": ([k.replace("simh_pilot", "simh_full") for k in KEYS], [1] * 20)})["defects"]),
    "a zero-unit pilot run: listed by key, NOT a defect (09)": (lambda r: r["defects"] == [] and r["jobs"][0]["zero_unit"] == [KEYS[0]])(
        run(files=FOLD_OK | {"pilot.npz": (KEYS[1:], [1] * 19)})),
    "a merged job whose cell map lacks a pilot cell: DEFECT": bool(run(spec(cfg_with(merge={"main_of_cell": {t: MAIN_T[t] for t in TASKS[1:]}})))["defects"]),
    "a merged job whose cell map has every pilot cell: clean": run(spec(cfg_with(merge={"main_of_cell": dict(MAIN_T)})))["defects"] == [],
    "fold -1 units, per-set row: listed as drops with the 25h §1 rule, not a defect": (lambda r: r["defects"] == [] and len(r["drops"]) == 1
        and r["drops"][0]["key"] == KEYS[0] and r["drops"][0]["run_dropped"] and r["drops"][0]["filter"] == PF.FOLD_RULE)(
        run(files=FOLD_OK | {"pilot.npz": (KEYS, [-1] + [1] * 19)})),
    "fold -1 on some of a run's units: listed, run not dropped": (lambda r: r["drops"][0]["run_dropped"] is False)(
        run(files=FOLD_OK | {"pilot.npz": (KEYS + KEYS[:1], [-1] + [1] * 19 + [2])})),
    "fold -1 units in a pooled NO_FOLD_DROPS job (full_as sol_a_uncut, the real tuple): DEFECT (#6 STOPs), not a drop":
        (lambda r: bool(r["defects"]) and r["drops"] == [])(run(spec(cfg_with(pw_combo="pooled_all_origins", full_as="sol_a_uncut C")),
                                                              files=FOLD_OK | {"pilot.npz": (KEYS, [-1] + [1] * 19)})),
    "the same with the 32k label only (no full_as, not a NO_FOLD_DROPS name): a drop, as the runner reads full_as":
        (lambda r: r["defects"] == [] and len(r["drops"]) == 1)(run(spec(cfg_with(pw_combo="pooled_all_origins")),
                                                                    files=FOLD_OK | {"pilot.npz": (KEYS, [-1] + [1] * 19)})),
    "a pooled NO_FOLD_DROPS STOP risk is marked conditional when the label may apply 21b":
        any("unless the row applies 21b" in d for d in run(spec(cfg_with(pw_combo="pooled_all_origins", full_as="Sol (a)|uncut C", use21b=["Sol (a) C"])),
                                                            files=FOLD_OK | {"pilot.npz": (KEYS, [-1] + [1] * 19)})["defects"]),
    "zero_unit_by_file: the list the runner pins, by the pilot file's sha": run(files=FOLD_OK | {"pilot.npz": (KEYS[1:], [1] * 19)})["zero_unit_by_file"]
    == {"p" * 64: [KEYS[0]]},
    "the runner's rules the pre-flight relies on (by syntax tree): S_old's exclusion map is empty; the cap keeps <= 30 units per run (no run drop)":
        cap_and_excl(),
    "single-model probe: folds not read as drops": run(spec(cfg_with(probes=[probe(folds=1)])), files=FOLD_OK | {"pilot.npz": (KEYS, None)})["drops"] == [],
    "A25 path (no fold array), every task has a real fold: clean": run(files=FOLD_OK | {"pilot.npz": (KEYS, None)})["defects"] == [],
    "A25 path, a task without a real fold >= 0: DEFECT": bool(run(files={"pilot.npz": (KEYS, None),
                                                                        "real.npz": ([f"honest_b|{t}" for t in TASKS], [-1] + [2] * 19)})["defects"]),
    "own_all_rs, every pilot main on its listed fold: clean": run(spec(cfg_with(probes=[probe(family="own_all_rs")],
                                                                                own_all_rs_fold_of_main={MAIN_T[t]: 1 for t in TASKS})))["defects"] == [],
    "own_all_rs, a pilot main outside the fold list: DEFECT": bool(run(spec(cfg_with(probes=[probe(family="own_all_rs")],
                                                                                     own_all_rs_fold_of_main={MAIN_T[t]: 1 for t in TASKS[1:]})))["defects"]),
    "own_all_rs, a unit off its main's fold: DEFECT": bool(run(spec(cfg_with(probes=[probe(family="own_all_rs")],
                                                                             own_all_rs_fold_of_main={MAIN_T[t]: (2 if t == TASKS[0] else 1) for t in TASKS})))["defects"]),
    "a tagged entry on an S_paper493 job: DEFECT": bool(run(spec(extra_jobs=[("Sol (a) C", {"origin": "S_paper493", "probes": [probe()],
                                                                                            "simh_old": {"Sol (a) C": [B1E, ENT]}}, None)]))["defects"]),
    "no S_old job carries the entry: DEFECT": bool(run({"jobs": spec()["jobs"][1:]})["defects"]),
    "the S_old job's own label carries no tagged entry: DEFECT": bool(run(spec(cfg_with(simh_old={"Sol (a) C": [B1E], "c_asis C": [ENT]})))["defects"]),
    "gate view: a whole-run fold -1 drop, in every S_old job, reads 'all'; not conditional without 21b":
        run(files=FOLD_OK | {"pilot.npz": (KEYS, [-1] + [1] * 19)})["gate_view"]["drops"] == [
            {"run": KEYS[0], "filter": "fold -1 drop (25h §1)", "citation": "09 Append 25h amendment @ 0f9927f3e §1", "conditional_on_21b": False,
             "label_combos": "all"}],
    "gate view: a partial fold -1 drop is not a run drop": run(files=FOLD_OK | {"pilot.npz": (KEYS + KEYS[:1], [-1] + [1] * 19 + [2])})["gate_view"]["drops"] == [],
    "gate view: a label in use21b marks the drop conditional": run(spec(cfg_with(use21b=["Sol (a) C"])), files=FOLD_OK | {"pilot.npz": (KEYS, [-1] + [1] * 19)})[
        "gate_view"]["drops"][0]["conditional_on_21b"] is True,
    "gate view: zero-unit pilot runs listed": run(files=FOLD_OK | {"pilot.npz": (KEYS[1:], [1] * 19)})["gate_view"]["zero_unit_pilot_runs"] == [KEYS[0]],
    "gate view: 0 drops is an empty list": r0["gate_view"] == {"drops": [], "zero_unit_pilot_runs": []},
    "cap before fold (Astra r1 R1): 31 units, the 30 kept are fold -1, the 31st fold 0: the WHOLE run drops under cap 30":
        (lambda r: r["drops"][0]["run_dropped"] is True and r["drops"][0]["of_units"] == 30 and r["gate_view"]["drops"][0]["run"] == KEYS[0])(
            run(spec(cfg_with(cap=30)), files=FOLD_OK | {"pilot.npz": ([KEYS[0]] * 31 + KEYS[1:], [-1] * 30 + [0] + [1] * 19)})),
    "the same file without a cap: a partial drop (the run stays)": (lambda r: r["drops"][0]["run_dropped"] is False and r["gate_view"]["drops"] == [])(
        run(files=FOLD_OK | {"pilot.npz": ([KEYS[0]] * 31 + KEYS[1:], [-1] * 30 + [0] + [1] * 19)})),
    "cap taskmin behaves as cap 30": run(spec(cfg_with(cap="taskmin")), files=FOLD_OK | {"pilot.npz": ([KEYS[0]] * 31 + KEYS[1:], [-1] * 30 + [0] + [1] * 19)})[
        "drops"][0]["run_dropped"] is True,
    "expected_zero_unit_ok_paths counts each tagged entry's file list once per S_old job": run(
        spec(extra_jobs=[("Sol (a) C", {"origin": "S_old", "probes": [probe()], "simh_old": {"Sol (a) C": [B1E, ENT]}}, None)]),
        files=FOLD_OK | {"pilot.npz": (KEYS[2:], [1] * 18)})["expected_zero_unit_ok_paths"] == 4,
    "TAG is full_override's S_OLD_ONLY": PF.TAG == FO.S_OLD_ONLY,
    "VOL_CONV is run_common455's conv Volume": PF.VOL_CONV == ast.literal_eval(next(
        n.value for n in ast.parse((HERE / "run_common455.py").read_text()).body if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "VOLS"))["conv"],
    "analyze does not mutate the spec": (lambda s, s0: (run(s), s == s0)[1])(spec(), spec()),
}
for name, ok in cases.items():
    print(("PASS " if ok else "FAIL ") + name)
assert all(cases.values()), "pre-flight tests failed"
print(f"{len(cases)} checks pass")
