"""The simulator's run-to-run variance from run A (RULE.md here, pre-registered @ c55f8253b). PRELIMINARY; pilot, n ~ 19.

Per frozen-five probe x readout x alpha, at X_err v6's MECHANICAL thresholds (Sol (a) z-scored with v6's real per-fold
moments), over tasks paired between sim epoch 1 (v6's `sim|<task>`) and epoch 2 (`sim_e2|<task>`):
  sigma2_W,sim  = mean (hit_e1 - hit_e2)^2 / 2      sigma2_W,real = mean (hit_run1 - hit_run2)^2 / 2 (same tasks)
plus sigma2_W,real on v6's whole estimation-half cohort, the ratio, and discordant-task counts. Modal CPU.

  mode=test: sim epoch 1 stands in for epoch 2 (pipeline test; sigma2_W,sim must be exactly 0)
  mode=real: epoch-2 scores from 2e's Sol (a) and 4f's p2 scorer files under /work/simh/a/ (--spec <json>)

  XE_IN=<v6 pinned inputs> RV_IN=<d9 manifest, pairing record, corrected-B manifest> PYTHONPATH=src $MODAL run \\
      --detach experiments/2026-10-14_sim_rerun_variance_09/run_sim_rerunvar.py --mode test
"""

import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
IN = Path(os.environ.get("XE_IN", "/nonexistent"))
RV = Path(os.environ.get("RV_IN", "/nonexistent"))
APP = "probe-shift_aswin-surya_sub-convaudit_simrerunvar_kill-2026-11-14"
PINS = {  # X_err v6's primary-variant inputs and its record (as run_simhonest_a25.py)
    "fibs_scores_2026-10-14_124948.json": "366d594cd719d52e9bb11d34bf4d3491b6cbd82d64f61efa73f476773cb49b1a",
    "scores_2026-10-14_145748.json": "d0b7f028f8ac3795dd946a261f384f474748a296772fb1f2f2f22cecaac6c961",
    "highstakes_scores_133519.json": "2d251dee71a4a0366bd9549fcc3726511fe576aafce0bb6a913353d7fc06f38a",
    "asis_scores_142228.json": "59c051a4ff9cc40811745a119251224f031cdaf479b8d3fb909c1059ab94da31",
    "xerr_v6_2026-10-14_161122.json": "e789144b62f71e0fb168dc3106656c2b5f29aadafe12df0f86dd1f74441d7645",
}
COUNTED, PAIRING, CB = (
    "simh_counted_keys_atk2_v2.json",
    "simh_attack2_pairing.json",
    "correctedb_manifest_v1.json",
)
RV_PINS = {  # d9's counted manifest @ 7033e873, pairing record @ 7d2ec6f1; v6's corrected-B eval manifest @ 46ca25fc4
    COUNTED: "17793af92932a996c8d0eee2acf1138ba48e54f6f31801d09be8bc4ead220b50",
    PAIRING: "3a6e139d15a1b9d8064225baa2ca4c22a256dac1a0bbc94d19adbb66d893ab52",
    CB: "bf847d425b80d95ebb75a9ed1d524fc68c9efcd014710ddc18e445ed9d373ebc",
}
ROWS = "/sol/xfer/asis_2026-10-14_135701/rows_2026-10-14_142228/"
SOL_NPZ = (
    ROWS + "a_tight__cmd_A.npz",
    "2e3b60e8e92dd744311773900a5b41b3dd2b48efadc29c9efcd665438a390cc3",
)
ALPHAS = (0.01, 0.05)
DRAWS = 10_000
SEED = [20261014, 3]
READOUTS = ("R_reply_meanpooled", "C_anycommand_max")
PROBES = ("sol_a", "fibs", "harmfulness", "refusal", "highstakes")
P2 = ("fibs", "refusal", "harmfulness", "highstakes")
E2 = "sim_e2"
LABEL = "pilot, n ~ 19, underpowered; PRELIMINARY"
SENS_LABEL = (
    "SENSITIVITY: restricted to attempted epoch-2 runs (Append: rerun-variance caveat)"
)
LABEL_ENUM = {"attempted", "unclear", "declined", "no_report", "run_failed"}

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
if IN.exists() and RV.exists():
    for name in PINS:
        image = image.add_local_file(IN / name, f"/root/in/{name}")
    for name in RV_PINS:
        image = image.add_local_file(RV / name, f"/root/in/{name}")
sol_vol = modal.Volume.from_name(
    "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", version=2
).read_only()
p2_vol = modal.Volume.from_name(
    "probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13", version=2
).read_only()
work_vol = modal.Volume.from_name(
    "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", version=2
)


class GateError(Exception):
    pass


def _no_dup(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise GateError("duplicate JSON key")
        d[k] = v
    return d


def load(name: str):
    raw = Path(f"/root/in/{name}").read_bytes()
    if hashlib.sha256(raw).hexdigest() != {**PINS, **RV_PINS}[name]:
        raise GateError(f"pin {name}")
    return json.loads(raw, object_pairs_hook=_no_dup)


def load_pinned(path: str, sha: str) -> bytes:
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise GateError(f"pin {Path(path).name}")
    return raw


def finite(d: dict, what: str) -> dict:
    import math

    if any(
        not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
        for v in d.values()
    ):
        raise GateError(f"non-finite scores in {what}")
    return d


def order_stat(values, a):
    import numpy as np

    s = np.sort(np.asarray(values, float), kind="mergesort")[::-1]
    if not len(s) or not np.isfinite(s).all():
        raise GateError("empty or non-finite pool")
    k = int(np.floor(a * len(s) + 1e-9))
    return float(s[min(k, len(s) - 1)])


def row_max_mean(raw_npz: bytes):
    import io

    import numpy as np

    z = np.load(io.BytesIO(raw_npz))
    keys = z["key" if "key" in z.files else "keys"].astype(str)
    sc = z["score" if "score" in z.files else "scores"].astype(np.float64)
    if keys.ndim != 1 or sc.shape != keys.shape or not np.isfinite(sc).all():
        raise GateError("npz shape or non-finite row")
    mx, tot, cnt = {}, {}, {}
    for k, v in zip(keys, sc, strict=True):
        mx[k] = max(mx.get(k, -np.inf), v)
        tot[k] = tot.get(k, 0.0) + v
        cnt[k] = cnt.get(k, 0) + 1
    return mx, {k: tot[k] / cnt[k] for k in tot}


def mean_gate(mean_: dict, ref: dict, what: str):
    if not ref or not mean_:
        raise GateError(f"G2 {what}: empty score dict")
    if set(mean_) != set(ref) or max(abs(mean_[k] - ref[k]) for k in ref) > 1e-6:
        raise GateError(f"G2 {what}")


def make_draws(tasks: list, stratum: dict, seed: list):
    import numpy as np

    rng = np.random.default_rng(seed)
    W = np.zeros((DRAWS + 1, len(tasks)))
    W[0] = 1.0
    g = np.array([stratum[t] for t in tasks])
    for s in sorted(set(g.tolist())):
        idx = np.flatnonzero(g == s)
        W[1:, idx] = rng.multinomial(
            len(idx), np.full(len(idx), 1 / len(idx)), size=DRAWS
        )
    return W, {t: i for i, t in enumerate(tasks)}


def p2_cell_ok(c: dict) -> bool:
    """Append 25c's p2 honest_b gate: exact, or within 1e-6 with no honest_b hit flips (fields required, no defaults)."""
    if c["same_keys"] is not True or c["n_runs"] < 190:
        return False
    exact = c["n_not_identical"] == 0 and c["max_abs_diff"] == 0.0
    if "byte_identical" in c and c["byte_identical"] is not exact:
        raise GateError("p2: byte_identical disagrees with the exact test")
    if exact:
        return True
    for f in ("byte_identical", "tolerance", "max_abs_diff", "hit_flips"):
        if f not in c:
            raise GateError(f"p2 tolerance gate: field {f} missing")
    return (
        c["tolerance"] == 1e-6
        and c["max_abs_diff"] <= 1e-6
        and c["hit_flips"] == {"0.01": 0, "0.05": 0}
    )


def epoch2_scores(
    spec: dict, paired: list, counted: set, fold_of: dict, recs: dict, fb_raw: str
) -> tuple:
    """Run A's scores {readout: {probe: {"sim_e2|<task>": score}}} from 2e's Sol and 4f's p2 manifests, with their gates."""
    E = {ro: {} for ro in READOUTS}
    meta = {}

    def keys_ok(D, what):
        bad = [
            k
            for k in D
            if not k.startswith(f"{E2}|") or k.split("|", 1)[1] not in counted
        ]
        if bad:
            raise GateError(f"{what}: {len(bad)} keys outside {E2}|<counted task>")
        return dict(D)

    # ---- 2e's Sol (a) tight scorer
    o = spec["sol"]
    man = json.loads(
        load_pinned(o["manifest"], o["manifest_sha256"]), object_pairs_hook=_no_dup
    )
    d = man["dump"]
    if (
        d["extract_run"] != o["approved_dump_run"]
        or d["manifest"]["sha256"] not in o["approved_dump_sha256"]
    ):
        raise GateError("sol: dump run or dump record not the approved one")
    if man["git"] != o["scorer_git"] or man["world"] != E2:
        raise GateError("sol: scorer commit or world differs from the expected")
    if man["counted_keys"]["sha256"] != RV_PINS[COUNTED]:
        raise GateError("sol: counted-key manifest differs from d9's pinned one")
    if man["pairing_record"]["sha256"] != RV_PINS[PAIRING]:
        raise GateError("sol: pairing record differs from d9's pinned one")
    if man.get("tasks_without_36_fold"):
        raise GateError("sol: tasks without a #36 fold")
    for rd in ("avg_top2", "cmd_A"):
        c = man["gates"][f"{rd}|a_tight"]
        if not (
            c["passes"] is True
            and c["not_equal"] == 0
            and c["max_abs_diff"] == 0.0
            and not c["missing"]
            and not c["extra"]
            and c["runs_compared"] >= 190
        ):
            raise GateError(f"sol: honest_b not exact for {rd}")
    if man["fit_npz_sha256"] != o["fit_npz_sha256"]:
        raise GateError("sol: fit identities differ from the expected")
    for k, f in man["run_folds"].items():
        if isinstance(f, bool) or not isinstance(f, int) or not k.startswith(f"{E2}|"):
            raise GateError("sol: fold not an integer, or a fold key outside the world")
        if fold_of.get(k.split("|", 1)[1]) != f:
            raise GateError(
                "sol: an epoch-2 fold differs from v6's Sol scoring fold map (asis run_folds)"
            )
    base = str(Path(o["manifest"]).parent) + "/"
    rj = json.loads(
        load_pinned(
            base + "a_tight__avg_top2.json",
            man["files"]["a_tight__avg_top2.json"]["sha256"],
        ),
        object_pairs_hook=_no_dup,
    )
    mx, _ = row_max_mean(
        load_pinned(
            base + "a_tight__cmd_A.npz", man["files"]["a_tight__cmd_A.npz"]["sha256"]
        )
    )
    E["R_reply_meanpooled"]["sol_a"] = keys_ok(finite(rj, "e2 sol reply"), "sol reply")
    E["C_anycommand_max"]["sol_a"] = keys_ok(mx, "sol command")
    for ro in READOUTS:
        if [k for k in E[ro]["sol_a"] if k not in man["run_folds"]]:
            raise GateError(
                f"sol: {ro} epoch-2 scores without a manifest fold (Codex review1)"
            )
    meta["manifest_sol"] = {
        "path": o["manifest"],
        "sha256": o["manifest_sha256"],
        "scorer_git": man["git"],
        "dump": d,
    }
    # ---- 4f's p2 scorer (fibs, refusal, harmfulness, highstakes)
    o = spec["p2"]
    man = json.loads(
        load_pinned(o["manifest"], o["manifest_sha256"]), object_pairs_hook=_no_dup
    )
    d = man["dump"]
    if (
        d["run"] != o["approved_dump_run"]
        or d["dump_manifest_sha256"] not in o["approved_dump_sha256"]
    ):
        raise GateError("p2: dump run or dump record not the approved one")
    if d["world"] != E2 or man["git"] != o["scorer_git"]:
        raise GateError("p2: world or scorer commit differs from the expected")
    pins = man["inputs"]["pins"]
    if (
        pins["counted_keys_atk2_v2"]["sha256"] != RV_PINS[COUNTED]
        or pins["attack2_pairing"]["sha256"] != RV_PINS[PAIRING]
    ):
        raise GateError(
            "p2: the manifest's counted-key or pairing pin differs from this job's pinned files"
        )
    if man.get("verdict") != "written":
        raise GateError("p2: manifest verdict is not 'written'")
    g = man["gate_honest_b"]
    if g.get("pass") is not True or g.get("fails"):
        raise GateError("p2: honest_b reproduction gate did not pass")
    pins_ = {
        "fibs": "fibs_scores_2026-10-14_124948.json",
        "harmfulness": "scores_2026-10-14_145748.json",
        "refusal": "scores_2026-10-14_145748.json",
        "highstakes": "highstakes_scores_133519.json",
    }
    cells = {}
    for p in P2:
        for rd in ("reply_mean", "command"):
            c = g["by_cell"][f"{p}|{rd}"]
            cells[f"{p}|{rd}"] = c
            if not p2_cell_ok(c):
                raise GateError(f"p2: honest_b gate failed for {p}|{rd}")
        if man["inputs"][f"record_{p}"]["sha256"] != PINS[pins_[p]]:
            raise GateError(
                f"p2: {p} reproduced against a record other than v6's input"
            )
        if p != "fibs":
            dr = recs[p]["directions"][p]
            if (
                man["probes"][p].get("sha256") != dr["sha256"]
                or man["probes"][p].get("layer") != dr["layer"]
            ):
                raise GateError(
                    f"p2: {p} direction or layer differs from v6's scores record"
                )
        else:
            got = {
                k_: v for k_, v in man["probes"]["fibs"].items() if k_.startswith("L")
            }
            if (
                not got
                or got != o["fibs_layer_sha256"]
                or any(v not in fb_raw for v in got.values())
            ):
                raise GateError(
                    "p2: FIBS layer files differ from the expected mapping or from v6's FIBS record"
                )
    meta["manifest_p2"] = {
        "path": o["manifest"],
        "sha256": o["manifest_sha256"],
        "scorer_git": man["git"],
        "dump": d,
        "gate_cells": cells,  # verbatim, including near_threshold_new_arm ("not computed (sim firewall, Append 25)")
        "host_cpu": man.get("host_cpu"),
        "projector_cpu": man.get("projector_cpu"),
    }
    pbase = "/work" + d["out_dir"].rstrip("/") + "/"
    for p in P2:
        rj = json.loads(
            load_pinned(
                pbase + f"{p}__reply_mean.json", man["files"][f"{p}__reply_mean.json"]
            ),
            object_pairs_hook=_no_dup,
        )
        if any(v for v in (rj.get("dropped") or {}).values()):
            raise GateError(f"p2: {p} dropped runs listed")
        mx, m_ = row_max_mean(
            load_pinned(pbase + f"{p}__command.npz", man["files"][f"{p}__command.npz"])
        )
        mean_gate(m_, finite(rj["command_mean"], f"e2 {p} command mean"), f"e2 {p}")
        E["R_reply_meanpooled"][p] = keys_ok(
            finite(rj["reply_mean"], f"e2 {p} reply"), p
        )
        E["C_anycommand_max"][p] = keys_ok(mx, p)
    for ro in READOUTS:
        for p in PROBES:
            miss = [t for t in paired if f"{E2}|{t}" not in E[ro][p]]
            if miss:
                raise GateError(
                    f"{ro}|{p}: {len(miss)} paired tasks without an epoch-2 score"
                )
    return E, meta


def _run(mode: str, spec: dict) -> dict:
    import numpy as np

    v6 = load("xerr_v6_2026-10-14_161122.json")["result"]
    if not v6["gates"]["pass"]:
        raise GateError("v6 record did not pass its gates")
    fb_raw = Path("/root/in/fibs_scores_2026-10-14_124948.json").read_text()
    fb, rh, hs = (
        load(n)
        for n in (
            "fibs_scores_2026-10-14_124948.json",
            "scores_2026-10-14_145748.json",
            "highstakes_scores_133519.json",
        )
    )
    asis = load("asis_scores_142228.json")
    us = v6["universe_and_split"]
    env, est_half = us["env_map"], us["estimation_half"]
    cohort = v6["cohort"]["tasks"]
    cpos = {t: i for i, t in enumerate(cohort)}
    c_est = [t for t in cohort if t in set(est_half)]
    fold_of = {}
    for k, f in asis["run_folds"].items():
        if isinstance(f, bool) or not isinstance(f, int) or not 0 <= f <= 4:
            raise GateError("fold not an integer in 0..4")
        if fold_of.setdefault(k.split("|", 1)[1], f) != f:
            raise GateError("a task's fold differs across arms")
    R = {"sol_a": finite(asis["run_scores"]["avg_top2"]["a_tight"], "sol reply")}
    C = {}
    C["sol_a"], m_ = row_max_mean(load_pinned(*SOL_NPZ))
    mean_gate(m_, asis["run_scores"]["cmd_A"]["a_tight"], "sol command")
    recs = {"fibs": fb, "harmfulness": rh, "refusal": rh, "highstakes": hs}
    for p, rec in (
        ("fibs", fb),
        ("harmfulness", rh["probes"]["harmfulness"]),
        ("refusal", rh["probes"]["refusal"]),
        ("highstakes", hs["probes"]["highstakes"]),
    ):
        R[p] = finite(rec["run_scores"]["reply_mean"], f"{p} reply")
        row = rec["rows"]["command"]
        C[p], m_ = row_max_mean(load_pinned(f"/p2/{row['path']}", row["sha256"]))
        mean_gate(m_, rec["run_scores"]["command"], p)
    RAW = {"R_reply_meanpooled": R, "C_anycommand_max": C}

    def g_apply(D, mom):
        if mom is None:
            return dict(D)
        return {
            k: (s - mom["mu"][fold_of[k.split("|", 1)[1]]])
            / mom["sd"][fold_of[k.split("|", 1)[1]]]
            for k, s in D.items()
        }

    # ---- gate V: v6's MECHANICAL thresholds and catches reproduce exactly before any epoch-2 score is read
    gV = {"threshold_max_abs": 0.0, "catch_max_abs": 0.0, "delta_max_abs": 0.0, "n": 0}
    real = {}

    def upd(key, d):  # a non-finite discrepancy fails at once (Codex review1)
        if not np.isfinite(d):
            raise GateError(f"gate V: non-finite discrepancy in {key}")
        gV[key] = max(gV[key], abs(d))

    for ro in READOUTS:
        for a in ALPHAS:
            r6 = v6["results"][f"{ro}|mechanical|alpha={a}"]
            for p in PROBES:
                T6 = r6["thresholds"][p]["threshold"]
                std6 = r6["standardization_sol"] if p == "sol_a" else None
                mom = (
                    None
                    if std6 is None
                    else {
                        k: {int(f): v for f, v in std6[k].items()} for k in ("mu", "sd")
                    }
                )
                Zr = g_apply(RAW[ro][p], mom)
                pool = r6["pools"][p]
                worlds = {
                    "B": ("honest_b",),
                    "BuC": ("honest_b", "honest_c"),
                    "AuB": ("honest_a", "honest_b"),
                }[pool]
                Tc = order_stat(
                    [v for k, v in Zr.items() if k.split("|", 1)[0] in worlds], a
                )
                hit = {
                    w: np.array([float(Zr[f"{w}|{t}"] > T6) for t in cohort])
                    for w in ("run1", "run2", "sim")
                }
                cs = {}
                for w in hit:
                    mine = 100 * float(np.mean([hit[w][cpos[t]] for t in c_est]))
                    cs[w] = (mine, r6["catches_percent"][p][w]["estimation_half"])
                    upd("catch_max_abs", mine - cs[w][1])
                d_m = cs["sim"][0] - (cs["run1"][0] + cs["run2"][0]) / 2
                d_6 = cs["sim"][1] - (cs["run1"][1] + cs["run2"][1]) / 2
                upd("delta_max_abs", d_m - d_6)
                upd("threshold_max_abs", Tc - T6)
                gV["n"] += 1
                real[(ro, a, p)] = {"T": T6, "mom": mom, "hit": hit, "pool": pool}
    if not all(
        np.isfinite(gV[k]) and gV[k] <= 1e-12
        for k in ("threshold_max_abs", "catch_max_abs", "delta_max_abs")
    ):
        raise GateError(f"gate V failed: {gV}")
    gV["pass"] = True
    # ---- the paired task set: pairing record x v6 cohort x counted-usable; epoch 1 must be v6's own sim run
    cm, pr, cb = load(COUNTED), load(PAIRING), load(CB)
    usable = {k["task"]: bool(k["usable"]) for k in cm["keys"] if k["world"] == E2}
    e2_sha = {k["task"]: k["eval_sha256"] for k in cm["keys"]}
    if len(usable) != len(cm["keys"]):
        raise GateError("counted manifest: duplicate tasks or a key outside sim_e2")
    v6_sim = {}
    for ep in cb["epochs"]:
        if ep.get("world") == "sim":
            if ep["join"] in v6_sim:
                raise GateError("corrected-B manifest: two epochs for one sim join")
            v6_sim[ep["join"]] = ep["key"]
    paired, excluded = [], {}
    for pp in pr["pairs"]:
        t, e1, e2 = pp["task"], pp["epoch1"], pp["epoch2"]
        if t in excluded or t in paired:
            raise GateError("pairing record: duplicate task")
        if e2["eval_sha256"] != e2_sha.get(t):
            raise GateError(f"{t}: pairing epoch 2 is not the counted run")
        key1 = v6_sim.get(f"sim|{t}")
        if t not in cohort:
            excluded[t] = "not in v6's cohort"
        elif not usable.get(t) or not e2.get("usable"):
            excluded[t] = "epoch 2 not counted-usable: " + str(
                e2.get("counted_failure")
            )
        elif key1 is None or not key1.startswith(f"{e1['launch']}/{e1['eval_file']}#"):
            raise GateError(f"{t}: epoch 1 in the pairing record is not v6's sim run")
        else:
            paired.append(t)
    paired = sorted(paired)
    if len(paired) < 5:
        raise GateError(f"only {len(paired)} paired tasks")
    counted = set(usable)
    paired_all = list(
        paired
    )  # gates and score completeness always use the full paired set (Codex review2)
    sens = None
    if spec.get(
        "e2_attempt_labels"
    ):  # optional sensitivity: keep tasks whose epoch-2 run is labelled "attempted"
        ls = spec["e2_attempt_labels"]
        raw = spec.get("_e2_labels_raw", "").encode()
        if hashlib.sha256(raw).hexdigest() != ls["sha256"]:
            raise GateError("e2 attempt labels: pin")
        lj = json.loads(raw, object_pairs_hook=_no_dup)
        lab = lj["labels"]
        if lj.get("world") != E2 or set(lab) != {f"{E2}|{t}" for t in counted}:
            raise GateError(
                "e2 attempt labels: world, or keys not exactly the counted epoch-2 runs"
            )
        if any(v not in LABEL_ENUM for v in lab.values()):
            raise GateError(f"e2 attempt labels: a label outside {sorted(LABEL_ENUM)}")
        drop = {t: lab[f"{E2}|{t}"] for t in paired if lab[f"{E2}|{t}"] != "attempted"}
        paired = [t for t in paired if t not in drop]
        if len(paired) < 5:
            raise GateError(
                f"only {len(paired)} paired tasks after the attempted restriction"
            )
        counts = {
            v: sum(x == v for x in lab.values()) for v in sorted(set(lab.values()))
        }
        sens = {k: ls[k] for k in ("path", "sha256", "source") if k in ls} | {
            "counts": counts,
            "dropped_from_paired": drop,
        }
    # ---- epoch-2 scores
    meta = {"mode": mode}
    if mode == "test":
        E = {
            ro: {
                p: {f"{E2}|{t}": RAW[ro][p][f"sim|{t}"] for t in paired_all}
                for p in PROBES
            }
            for ro in READOUTS
        }
        meta["label"] = "PIPELINE TEST: sim epoch 1 stands in for epoch 2; not a result"
    else:
        E, m2 = epoch2_scores(spec, paired_all, counted, fold_of, recs, fb_raw)
        meta |= m2 | {
            "label": LABEL,
            "spec_resolved": {k: v for k, v in spec.items() if k != "_e2_labels_raw"},
        }
    if sens is not None:
        meta |= {
            "label": f"{SENS_LABEL}; {meta['label']}",
            "role": "sensitivity; the unrestricted record stays primary",
            "e2_attempt_labels": sens,
        }
    # ---- estimand
    Wp, _ = make_draws(paired, env, SEED)
    We, pe = make_draws(est_half, env, [20261014, 1])
    Wt_est = We[:, [pe[t] for t in c_est]]

    def wmean(W, v):
        with np.errstate(invalid="ignore", divide="ignore"):
            return (W @ v) / W.sum(1)

    def summ(v):
        f = v[1:][np.isfinite(v[1:])]
        return {
            "point": float(v[0]) if np.isfinite(v[0]) else None,
            "ci95": [float(np.quantile(f, q)) for q in (0.025, 0.975)]
            if len(f)
            else [None, None],
            "finite_draws": len(f),
        }

    def ratio_summ(v):
        r = summ(v)
        und = DRAWS - r["finite_draws"]
        r["undefined_draws_real_variance_0"] = und
        if und:
            r["ci95_conditional_on_defined_draws"] = r["ci95"]
            r["ci95"] = None
        return r

    def reading(sim, ref_point):
        lo, hi = sim["ci95"]
        if ref_point is None or lo is None:
            return "undefined"
        if hi < ref_point:
            return "extra sim runs help LESS than projected (sigma2_W,sim interval entirely below sigma2_W,real)"
        if lo > ref_point:
            return "extra sim runs help MORE than projected (sigma2_W,sim interval entirely above sigma2_W,real)"
        return "not contradicted at this n"

    rows, readings = {}, {}
    for ro in READOUTS:
        for a in ALPHAS:
            blk = {}
            for p in PROBES:
                rr = real[(ro, a, p)]
                z2 = g_apply(E[ro][p], rr["mom"])
                h1 = np.array([rr["hit"]["sim"][cpos[t]] for t in paired])
                h2 = np.array([float(z2[f"{E2}|{t}"] > rr["T"]) for t in paired])
                r1 = np.array([rr["hit"]["run1"][cpos[t]] for t in paired])
                r2 = np.array([rr["hit"]["run2"][cpos[t]] for t in paired])
                s_sim = wmean(Wp, (h1 - h2) ** 2 / 2)
                s_real = wmean(Wp, (r1 - r2) ** 2 / 2)
                ra = np.array(
                    [
                        rr["hit"]["run1"][cpos[t]] - rr["hit"]["run2"][cpos[t]]
                        for t in c_est
                    ]
                )
                s_all = (Wt_est @ (ra**2 / 2)) / Wt_est.sum(1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    ratio = np.where(
                        s_real > 0, s_sim / np.where(s_real > 0, s_real, 1.0), np.nan
                    )
                if mode == "test":
                    h2k = h2.copy()
                    h2k[0] = 1.0 - h2k[0]
                    if (
                        abs(
                            wmean(Wp[:1], (h1 - h2k) ** 2 / 2)[0]
                            - 1 / (2 * len(paired))
                        )
                        > 1e-15
                    ):
                        raise GateError(
                            "test mode: one flipped task does not give 1/(2N)"
                        )
                if mode == "test" and np.any(s_sim != 0):
                    raise GateError(
                        f"test mode: sigma2_W,sim not exactly 0 for {ro}|{a}|{p}"
                    )
                rec = {
                    "sigma2_W_sim": summ(s_sim),
                    "sigma2_W_real_paired": summ(s_real),
                    "sigma2_W_real_all_estimation_half": summ(s_all)
                    | {"n_tasks": len(c_est)},
                    "ratio_sim_over_real_paired": ratio_summ(ratio)
                    | (
                        {"note": "undefined: sigma2_W,real = 0 on the paired tasks"}
                        if s_real[0] == 0
                        else {}
                    ),
                    "discordant_tasks": {
                        "e1_vs_e2": int((h1 != h2).sum()),
                        "run1_vs_run2": int((r1 != r2).sum()),
                    },
                    "catch_percent_paired": {
                        "e1": 100 * float(h1.mean()),
                        "e2": 100 * float(h2.mean()),
                        "run1": 100 * float(r1.mean()),
                        "run2": 100 * float(r2.mean()),
                    },
                    "threshold_v6_mechanical": rr["T"],
                    "near_threshold_1e-6_e2": int(
                        np.sum(
                            np.abs(
                                np.array([z2[f"{E2}|{t}"] for t in paired]) - rr["T"]
                            )
                            <= 1e-6
                        )
                    ),  # report only: an epoch-2 hit that could depend on the scorer's host CPU (Sol: z units)
                    "pool": rr["pool"],
                }
                rec["reading_vs_real_paired"] = reading(
                    rec["sigma2_W_sim"], rec["sigma2_W_real_paired"]["point"]
                )
                rec["reading_vs_real_all_secondary"] = reading(
                    rec["sigma2_W_sim"],
                    rec["sigma2_W_real_all_estimation_half"]["point"],
                )
                blk[p] = rec
                readings[f"{ro}|{a}|{p}"] = rec["reading_vs_real_paired"]
            rows[f"{ro}|alpha={a}"] = blk
    return {
        "label": meta["label"],
        "rule": "experiments/2026-10-14_sim_rerun_variance_09/RULE.md @ c55f8253b",
        "meta": meta,
        "gates": {"V": gV},
        "tasks": {
            "paired": paired,
            "n_paired": len(paired),
            "excluded": excluded,
            "n_paired_in_estimation_half": sum(t in set(est_half) for t in paired),
            "draws": {"n": DRAWS, "seed": SEED, "strata": "environment"},
        },
        "rows": rows,
        "readings": readings,
        "caveats": [
            "sigma2 values are in hit units (fraction^2); hit = score above v6's mechanical threshold, fixed",
            (
                "the real pair is conditioned on 'attempted' in run1, run2 and sim (v6's cohort); epoch 2 has no "
                "attempt label here, so a declined epoch-2 attack counts as an ordinary run (can inflate sigma2_W,sim)"
                if sens is None
                else "the real pair, epoch 1 and (in this sensitivity) epoch 2 are all restricted to 'attempted' runs"
            ),
            (
                "run1 and run2 are different launches (Append 22), and so are epochs 1 and 2: both variances include "
                "a launch component"
            ),
            "about 19 paired tasks: a pilot; intervals are task-bootstrap percentiles, conditional on the thresholds",
        ]
        + (
            [
                "this record keeps only tasks whose epoch-2 run is labelled 'attempted' (d9's claim-C labels)"
            ]
            if sens is not None
            else []
        ),
    }


def SCRIPT_SHA256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    timeout=1200,
    retries=0,
    volumes={"/sol": sol_vol, "/p2": p2_vol, "/work": work_vol},
)
def run(mode: str, spec: dict, stamp: str = "", git_sha: str = "") -> dict:
    import time

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        work_vol.reload()
        res = _run(mode, spec)
    except GateError as e:
        res = {"fatal": "GateError", "gate": str(e)}
    except Exception as e:  # noqa: BLE001 -- scores only, no transcript text
        import traceback

        res = {"fatal": type(e).__name__, "traceback": traceback.format_exc()[-3000:]}
    res["provenance"] = {
        "git": git_sha,
        "stamp": stamp,
        "script_sha256": SCRIPT_SHA256(),
        "started_utc": started,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    Path("/work/simh").mkdir(exist_ok=True)
    tag = mode + ("_e2attempted" if spec.get("e2_attempt_labels") else "")
    Path(f"/work/simh/rerunvar_{tag}_{stamp}.json").write_text(
        json.dumps(res, default=float)
    )
    work_vol.commit()
    return res


@app.local_entrypoint()
def main(mode: str = "test", spec: str = ""):
    def git(*a):
        return subprocess.run(
            ["git", *a], capture_output=True, text=True, check=True, cwd=REPO
        ).stdout.strip()

    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    if mode not in ("test", "real"):
        raise SystemExit("mode must be test or real")
    sp = json.loads(Path(spec).read_text()) if spec else {}
    if mode == "real" and not sp:
        raise SystemExit("--spec is required in real mode")
    if sp.get("e2_attempt_labels"):  # the job re-hashes these bytes in-container
        raw = Path(sp["e2_attempt_labels"]["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sp["e2_attempt_labels"]["sha256"]:
            raise SystemExit("e2 attempt labels: local file does not match its sha256")
        sp["_e2_labels_raw"] = raw.decode()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = run.remote(mode, sp, stamp, git("rev-parse", "HEAD"))
    tag = mode + ("_e2attempted" if sp.get("e2_attempt_labels") else "")
    dest = HERE / "results" / f"rerunvar_{tag}_{stamp}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.open("x").write(
        json.dumps(
            {
                "git": git("rev-parse", "HEAD"),
                "stamp": stamp,
                "mode": mode,
                "result": res,
            },
            indent=1,
            default=float,
        )
    )
    print(
        json.dumps(
            {
                k: res.get(k)
                for k in ("fatal", "gate", "traceback", "gates", "readings")
            },
            indent=1,
        ),
        flush=True,
    )
    print("wrote", dest.name, flush=True)
    if "fatal" in res:
        raise SystemExit(1)
