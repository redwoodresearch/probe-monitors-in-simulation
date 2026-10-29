"""Follow-up 1 (q2 / 0d, 2026-10-15 ~23:40 PT): split the threshold-re-estimation part of the pooled 455 Delta_full joint interval into tau_sim
(sim-honest pool) and tau_real (real-honest pool), and predict the joint half-width with larger honest pools (POST HOC, descriptive; PRELIMINARY).

Part A (pooled 455, Stage A pooled455v2 C/R @ 106262746, per action 1e-2). In Stage A the sim catch depends only on the sim-honest pool and the
real catches only on the real pool, and the joint and fixed modes share the same task draws (only the pools' weights differ). So
  only tau_sim re-drawn  = sim(joint) - real_mean(fixed),   only tau_real re-drawn = sim(fixed) - real_mean(joint),
per draw, with no recompute. S_old also carries run 1 at tau on honest B alone and on honest C alone (one epoch per task) beside B u C (two),
which measures what a second honest epoch on the SAME tasks buys.
Part B (validation, S_new five-arm cells, the four single-model probes x C/R): the joint bootstrap recomputed from the per-row files the S_new
hit export read (gates: draw-0 thresholds and hits equal the export and Stage A), then the honest pools subsampled to half their tasks.
Method and predictions written before the run: README.md, "Follow-up 1".

  uv run --no-project --with modal==1.6.1 modal run --detach experiments/2026-10-15_q2_more_runs_variance/run_threshold_split.py
"""

import hashlib
import io
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-q2-morerunsvar_check_kill-2026-11-15"
PINS = {  # name -> (commit, path, sha256 of the file at that commit)
    "rec_C": (
        "106262746",
        "experiments/2026-10-15_fpr_sweep_507_prep/results/pooled455v2_C_2026-10-15_221159.json",
        "631a23dd95ffad1a1cbd7dbf31ff4627070774e1d48891a5a9d5db6b2ccb3783",
    ),
    "rec_R": (
        "106262746",
        "experiments/2026-10-15_fpr_sweep_507_prep/results/pooled455v2_R_2026-10-15_221200.json",
        "0209f4aac394c6ae84692f90e368699efacb753d93e3cd0e9b29080792d2d43c",
    ),
    "rms_S_new": (
        "07dda0021",
        "experiments/2026-10-15_il_z_e2e/results/rmshits_S_new_2026-10-15_214500_ap-7XDyyK8ba6AjMTYvg0VUXB.json",
        "a50d0807e251cc9e1ba5126936b886c9f439f43e97bc741ab491bb43bd8892f4",
    ),
    "cfg_S_new": (
        "91ffc602b",
        "experiments/2026-10-15_il_z_e2e/config/inputs_rms_S_new_final.json",
        "3baef107eefdf9b145485a1fe6c1f4734b954769ebab02f676c024913d572cac",
    ),
    "plus311": (
        "d75f997f3",
        "experiments/2026-10-14_q35_sol_avg_POSTHOC/config/scaleup_tasks_plus311_2026-10-14_125135.json",
        "69f1f75313c9610f7d16b18225c1166268c0b6e084859c641d0213f293896a8b",
    ),
}
F = "0.01"
FF = 0.01
COST_REAL, COST_SIM = 0.22, (12.0, 23.0)  # $ per honest run (q2's figures)
SCEN = {  # name -> (sim pool factor, real pool factor), the same on both origins; "epoch-2 actual" scenarios are added from the counts
    "e_real_x2": (1, 2),
    "f_sim_x2": (2, 1),
    "g_both_x2": (2, 2),
    "h_real_x4": (1, 4),
    "sim_x4": (4, 1),
}
EP2 = (  # epoch-2 pool-key artifact (keys, launch ids and .eval sha256s only; COUNT fields read by key, nothing else)
    "ef4af11c9",
    "experiments/2026-10-15_epoch2_poolkeys/results/epoch2_poolkeys_2026-10-15_181342.json",
    "e1979dc72e90c19beba24c47036156ec026a26330a159a2eba0718ff67f61414",
    {"S_old": "S_old | BuC | H_B real line (B u C)", "S_new": "S_new | B | H_B (B)"},
)
N_DRAWS_B, N_HALVES = 4000, 8

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
solavg = modal.Volume.from_name(
    "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", version=2
)
conv = modal.Volume.from_name(
    "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"
)


def npzkey(*parts):  # run_peraction311.npzkey (v2 key format)
    return "__".join(
        str(p)
        .replace("@", "_at_")
        .replace("|", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        for p in parts
    )


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=1800,
    retries=0,
    volumes={"/solavg": solavg, "/conv": conv.read_only()},
)
def analyze(blobs: dict, meta: dict) -> dict:
    import numpy as np

    rng = np.random.default_rng([20261015, 101])
    gates = []

    def gate(ok, msg):
        if not ok:
            raise SystemExit(f"gate failed: {msg}")
        gates.append(msg)

    def var(x):  # unbiased variance over draws 1..D, finite only
        x = x[1:][np.isfinite(x[1:])]
        return float(np.var(x, ddof=1))

    def cov(x, y):
        m = np.isfinite(x[1:]) & np.isfinite(y[1:])
        return float(np.cov(x[1:][m], y[1:][m])[0, 1])

    def hw(x):
        f = x[1:][np.isfinite(x[1:])]
        lo, hi = float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))
        return [lo, hi], (hi - lo) / 2

    def safe_div(a, b):
        return a / b if b != 0 else None

    def components(simF, simJ, rmF, rmJ):
        """g0 = fixed gap (attack sampling), da = sim threshold shift, dc = real threshold shift (per draw).
        V_joint = V_F + Var(da) + Var(dc) + 2Cov(g0,da) - 2Cov(g0,dc) - 2Cov(da,dc)."""
        g0, da, dc = simF - rmF, simJ - simF, rmJ - rmF
        gate(
            all(bool(np.isfinite(x).all()) for x in (g0, da, dc)),
            "every draw finite in a decomposition (one shared sample for all its variances and covariances)",
        )
        c = {
            "V_F": var(g0),
            "V_da": var(da),
            "V_dc": var(dc),
            "C_g0_da": cov(g0, da),
            "C_g0_dc": cov(g0, dc),
            "C_da_dc": cov(da, dc),
        }
        c["V_J"] = var(g0 + da - dc)
        c["V_J_from_parts"] = (
            c["V_F"]
            + c["V_da"]
            + c["V_dc"]
            + 2 * c["C_g0_da"]
            - 2 * c["C_g0_dc"]
            - 2 * c["C_da_dc"]
        )
        return c

    def vj_scaled(
        c, gs, gr
    ):  # threshold-shift SDs scale by sqrt(g) (variance factor g) for each pool
        return (
            c["V_F"]
            + c["V_da"] * gs
            + c["V_dc"] * gr
            + 2 * c["C_g0_da"] * np.sqrt(gs)
            - 2 * c["C_g0_dc"] * np.sqrt(gr)
            - 2 * c["C_da_dc"] * np.sqrt(gs * gr)
        )

    def epoch_factor(
        m, k, phi
    ):  # threshold variance V(m) = V1 (phi + (1 - phi)/m): factor from m epochs per task to k*m
        return (phi + (1 - phi) / (m * k)) / (phi + (1 - phi) / m)

    # ---------------- Part A: the pooled 455 ----------------
    def part_a(R):
        rec = json.loads(blobs[f"rec_{R}"])
        b = (Path("/solavg") / rec["npz"]["path"]).read_bytes()
        gate(
            hashlib.sha256(b).hexdigest() == rec["npz"]["sha256"],
            f"{R} npz sha256 equals the record's pin",
        )
        z = np.load(io.BytesIO(b), allow_pickle=False)
        out, pools = {}, {}
        for lb, pr in rec["pooled"].items():
            P = pr.get("pooled")
            if not isinstance(P, dict):
                out[lb] = {"unavailable": P}
                continue
            arr = {}
            for mode in ("joint", "fixed"):
                for org in ("S_old", "S_new"):
                    names = [
                        ("sim", "sim@S|primary"),
                        ("r1", "run1@B"),
                        ("r2", "run2@B"),
                        ("rm", "real_mean@B"),
                    ]
                    if org == "S_old":
                        names += [
                            ("r1B", "run1@B2"),
                            ("r1C", "run1@A"),
                            ("r2C", "run2@A"),
                        ]
                    for nm, k in names:
                        arr[(mode, org, nm)] = z[
                            npzkey(lb, mode, F, org, "catch_" + k)
                        ].astype(np.float64)
                    arr[(mode, org, "D")] = z[npzkey(lb, mode, F, org, "D")].astype(
                        np.float64
                    )
            for org in ("S_old", "S_new"):
                gate(
                    np.array_equal(arr[("joint", org, "D")], arr[("fixed", org, "D")]),
                    f"{lb} {org}: joint and fixed share the task draws (D)",
                )
                for nm in ("sim", "r1", "r2", "rm"):
                    gate(
                        abs(arr[("joint", org, nm)][0] - arr[("fixed", org, nm)][0])
                        <= 1e-12,
                        f"{lb} {org} {nm}: joint equals fixed at draw 0",
                    )
            D = {org: arr[("fixed", org, "D")] for org in ("S_old", "S_new")}

            def pool(
                mode, nm, arr=arr, D=D
            ):  # Stage A's pool_reduce: equal-main weights over both origins per draw
                return (
                    arr[(mode, "S_old", nm)] * D["S_old"]
                    + arr[(mode, "S_new", nm)] * D["S_new"]
                ) / (D["S_old"] + D["S_new"])

            dJ = pool("joint", "sim") - pool("joint", "rm")
            ci, h = hw(dJ)
            gate(
                abs(ci[0] - P["joint"][F]["gap"]["delta_full"]["ci95"][0]) <= 1e-12
                and abs(ci[1] - P["joint"][F]["gap"]["delta_full"]["ci95"][1]) <= 1e-12,
                f"{lb}: pooled joint Delta_full interval equals the record",
            )
            dF = pool("fixed", "sim") - pool("fixed", "rm")
            d_sim_only = pool("joint", "sim") - pool("fixed", "rm")
            d_real_only = pool("fixed", "sim") - pool("joint", "rm")
            VJ, VF, Vs, Vr = var(dJ), var(dF), var(d_sim_only), var(d_real_only)
            Vthr = VJ - VF
            cpool = components(
                pool("fixed", "sim"),
                pool("joint", "sim"),
                pool("fixed", "rm"),
                pool("joint", "rm"),
            )
            per = {}
            for org in ("S_old", "S_new"):

                def g(mode, nm, arr=arr, org=org):
                    return arr[(mode, org, nm)]

                per[org] = components(
                    g("fixed", "sim"),
                    g("joint", "sim"),
                    g("fixed", "rm"),
                    g("joint", "rm"),
                )
                per[org]["mains"] = float(D[org][0])
            om = {
                org: per[org]["mains"] / (per["S_old"]["mains"] + per["S_new"]["mains"])
                for org in per
            }

            # S_old real pool: one honest epoch per task (B alone, C alone) vs two (B u C), same tasks, same draws
            def thr_var(nm, arr=arr):
                return var(arr[("joint", "S_old", nm)] - arr[("fixed", "S_old", nm)])

            ep = {
                "run1_B_only": thr_var("r1B"),
                "run1_C_only": thr_var("r1C"),
                "run1_B_and_C": thr_var("r1"),
                "run2_C_only": thr_var("r2C"),
                "run2_B_and_C": thr_var("r2"),
            }
            ratio1 = safe_div(
                (ep["run1_B_only"] + ep["run1_C_only"]) / 2, ep["run1_B_and_C"]
            )
            ratio2 = safe_div(ep["run2_C_only"], ep["run2_B_and_C"])
            gate(
                ratio1 is not None and ratio2 is not None and ratio1 > 0 and ratio2 > 0,
                f"{lb}: S_old epoch ratios identified",
            )
            ratio = (
                (ratio1 + ratio2) / 2
            )  # 2 if a 2nd epoch acted like new tasks (1/n), 1 if the threshold noise were all between-task
            phi_raw = (
                2 / ratio - 1
            )  # between-task fraction of a one-epoch pool's threshold variance: V(m) = V1 (phi + (1 - phi)/m)
            phi = min(max(phi_raw, 0.0), 1.0)
            m_epochs = {
                "S_old": {"real": 2, "sim": 1},
                "S_new": {"real": 1, "sim": 1},
            }  # B u C on S_old; one honest run per task elsewhere
            cal = VJ / sum(om[o] ** 2 * per[o]["V_J"] for o in per)
            preds = {}
            for model in ("repeat_epochs_same_tasks", "new_tasks_1_over_n"):
                for sc, fk in scen.items():
                    parts = {
                        "V_F": 0.0,
                        "tau_sim": 0.0,
                        "tau_real": 0.0,
                        "interaction": 0.0,
                    }
                    for o in ("S_old", "S_new"):
                        ks, kr = fk[o]
                        if model == "new_tasks_1_over_n":
                            gs, gr = 1 / ks, 1 / kr
                        else:
                            gs, gr = (
                                epoch_factor(m_epochs[o]["sim"], ks, phi),
                                epoch_factor(m_epochs[o]["real"], kr, phi),
                            )
                        c = per[o]
                        w2 = om[o] ** 2 * cal
                        parts["V_F"] += w2 * c["V_F"]
                        parts["tau_sim"] += w2 * (
                            c["V_da"] * gs + 2 * c["C_g0_da"] * np.sqrt(gs)
                        )
                        parts["tau_real"] += w2 * (
                            c["V_dc"] * gr - 2 * c["C_g0_dc"] * np.sqrt(gr)
                        )
                        parts["interaction"] += w2 * (
                            -2 * c["C_da_dc"] * np.sqrt(gs * gr)
                        )
                    vj = sum(parts.values())
                    preds.setdefault(model, {})[sc] = {
                        "joint_hw": h * np.sqrt(vj / VJ),
                        "change_pct": 100 * (np.sqrt(vj / VJ) - 1),
                        "share_of_joint_var": {k_: v_ / vj for k_, v_ in parts.items()},
                    }
            today_rebuilt = cal * sum(
                om[o] ** 2 * vj_scaled(per[o], 1.0, 1.0) for o in per
            )
            rp = rec["rows"][lb]
            pools[lb] = {
                o: {
                    "H_sim_admitted": rp[o]["pool_runs"]["H_sim"]["admitted_runs"],
                    "H_real_admitted": rp[o]["pool_runs"]["H_B"]["admitted_runs"],
                }
                for o in ("S_old", "S_new")
            }
            out[lb] = {
                "label": "estimate (POST HOC)",
                "today": {
                    "joint_hw": h,
                    "joint_ci95": ci,
                    "fixed_hw": hw(dF)[1],
                    "V_J": VJ,
                    "V_F": VF,
                    "V_thr": Vthr,
                },
                "split": {
                    "method": "per-draw hybrids of Stage A's joint and fixed modes (same task draws): only tau_sim = sim(joint) - real(fixed); only tau_real = sim(fixed) - real(joint)",
                    "hw_only_tau_sim_redrawn": hw(d_sim_only)[1],
                    "hw_only_tau_real_redrawn": hw(d_real_only)[1],
                    "V_sim_part": Vs - VF,
                    "V_real_part": Vr - VF,
                    "V_interaction": VJ - Vs - Vr + VF,
                    "share_of_V_thr": {
                        "tau_sim": safe_div(Vs - VF, Vthr),
                        "tau_real": safe_div(Vr - VF, Vthr),
                        "interaction": safe_div(VJ - Vs - Vr + VF, Vthr),
                    },
                    "pooled_components": cpool,
                },
                "per_origin_components": per | {"weights": om},
                "S_old_real_epoch_check": ep
                | {
                    "ratio_one_over_two_epochs_run1": ratio1,
                    "ratio_one_over_two_epochs_run2": ratio2,
                    "ratio": ratio,
                    "phi_raw": phi_raw,
                    "phi_used": phi,
                },
                "predicted": preds,
                "positive_controls": {
                    "today_joint_hw_rebuilt_from_per_origin_parts (identity via the calibration factor)": h
                    * np.sqrt(today_rebuilt / VJ),
                    "today_joint_hw": h,
                    "calibration_pooled_over_independent_sum": cal,
                    "pooled_V_J_parts_identity": [
                        cpool["V_J"],
                        cpool["V_J_from_parts"],
                    ],
                    "normal_vs_percentile_joint_hw": [
                        1.959963984540054 * np.sqrt(VJ),
                        h,
                    ],
                },
            }
        return out, pools

    ep2 = json.loads(blobs["ep2_counts"])
    real_now = {o: ep2[o]["epoch1_members"] for o in ("S_old", "S_new")}
    real_ep2 = {
        o: ep2[o]["epoch1_members"] + ep2[o]["epoch2_admitted"]
        for o in ("S_old", "S_new")
    }
    scen = {sc: {o: v for o in ("S_old", "S_new")} for sc, v in SCEN.items()}
    for sc, ks in (
        ("e_actual_epoch2_real", 1),
        ("e_actual_plus_sim_x2", 2),
        ("e_actual_plus_sim_x4", 4),
    ):
        scen[sc] = {o: (ks, real_ep2[o] / real_now[o]) for o in ("S_old", "S_new")}
    result = {
        "label": "estimate (POST HOC); PRELIMINARY until Aswin reviews",
        "meta": meta,
        "readouts": {},
        "epoch2_counts_used": ep2,
        "scenario_factors": scen,
    }
    allpools = {}
    for R in ("C", "R"):
        result["readouts"][R], pl = part_a(R)
        allpools |= pl
    first = next(iter(allpools.values()))
    gate(
        all(v == first for v in allpools.values()),
        "admitted honest pool sizes are the same for every probe row",
    )
    gate(
        all(real_now[o] == first[o]["H_real_admitted"] for o in real_now),
        "epoch-1 real pool sizes in the epoch-2 artifact equal Stage A's admitted pools",
    )
    runs = {}
    for sc, fk in scen.items():
        actual = sc.startswith("e_actual")
        xr = {
            o: (
                ep2[o]["epoch2_admitted"]
                if actual
                else round((fk[o][1] - 1) * first[o]["H_real_admitted"])
            )
            for o in first
        }
        xs = {o: round((fk[o][0] - 1) * first[o]["H_sim_admitted"]) for o in first}
        n_real, n_sim = sum(xr.values()), sum(xs.values())
        new_real = (
            0 if actual else n_real
        )  # epoch-2 real honest runs already exist (scored, sealed): no new spend
        runs[sc] = {
            "extra_real_honest_runs": xr | {"total": n_real, "already_exist": actual},
            "extra_sim_honest_runs": xs | {"total": n_sim},
            "new_cost_usd": [
                new_real * COST_REAL + n_sim * COST_SIM[0],
                new_real * COST_REAL + n_sim * COST_SIM[1],
            ],
        }
    result["pool_sizes_admitted"] = first
    result["scenario_runs_and_cost"] = runs | {
        "unit_costs": {"real_honest_run": COST_REAL, "sim_honest_run": list(COST_SIM)}
    }

    # ---------------- Part B: S_new five-arm validation of pool scaling (single-model probes) ----------------
    cfg = json.loads(blobs["cfg_S_new"])
    man = json.loads(blobs["rms_S_new"])
    hz = np.load(
        io.BytesIO((Path("/solavg") / f"xfer/il_z/rms_hits/{man['npz']}").read_bytes()),
        allow_pickle=False,
    )
    gate(
        hashlib.sha256(
            (Path("/solavg") / f"xfer/il_z/rms_hits/{man['npz']}").read_bytes()
        ).hexdigest()
        == man["npz_sha256"],
        "S_new hits npz sha256",
    )
    gate(
        list(cfg["cells"]) == list(man["cells"]),
        "config cells equal the export's, in order",
    )
    main_of = {c["key"]: c["main"] for c in json.loads(blobs["plus311"])["cells"]}
    recs = {R: json.loads(blobs[f"rec_{R}"]) for R in ("C", "R")}

    def loadc(path, pin):
        b = (Path("/conv") / path.lstrip("/").removeprefix("work/")).read_bytes()
        gate(
            hashlib.sha256(b).hexdigest() == pin,
            "per-row file sha256 equals the export config's pin",
        )
        return b

    def load_row(
        row,
    ):  # run_rms_hits.compute (single model) @ 91ffc602b: H_B units, H_sim units, arm run maxima (raw scores)
        cache = {}

        def folder(fol):
            if fol not in cache:
                mf = cfg["folders"][fol][row["sub"]]
                ent = json.loads(loadc(mf["path"], mf["sha256"]))["files"][row["file"]]
                d = mf["path"].replace("/work/", "").rsplit("/", 1)[0]
                cache[fol] = dict(
                    np.load(
                        io.BytesIO(
                            loadc(
                                f"{d}/{row['file']}",
                                ent if isinstance(ent, str) else ent["sha256"],
                            )
                        ),
                        allow_pickle=False,
                    )
                )
            return cache[fol]

        def root(ent):
            d = ent["path"].strip("/")
            if "files" in ent:
                b = loadc(f"{d}/{row['file']}", ent["files"][row["file"]])
            else:
                e = json.loads(
                    loadc(f"{d}/{ent['manifest']['name']}", ent["manifest"]["sha256"])
                )["files"][row["file"]]
                b = loadc(
                    f"{d}/{row['file']}", e if isinstance(e, str) else e["sha256"]
                )
            return dict(np.load(io.BytesIO(b), allow_pickle=False))

        kB, sB, owners = [], [], set()
        for fol in cfg["H_B"]["folders"]:
            zz = folder(fol)
            k = zz["key"].astype(str)
            m = np.isin(k, sorted(cfg["H_B"]["keys"]))
            gate(
                not (set(k[m].tolist()) & owners),
                f"{row['label']}: each H_B key in one folder",
            )
            owners |= set(k[m].tolist())
            kB.append(k[m])
            sB.append(zz["score"].astype(np.float64)[m])
        gate(owners == set(cfg["H_B"]["keys"]), f"{row['label']}: H_B keys all present")
        kS, sS, seen = [], [], set()
        for rt in cfg["H_sim"]["roots"]:
            zz = root(rt[row["sub"]])
            k = zz["key"].astype(str)
            m = ~np.isin(k, sorted(cfg["H_sim"].get("exclude_keys", [])))
            gate(
                not (set(k[m].tolist()) & seen),
                f"{row['label']}: each sim-honest run in one root",
            )
            seen |= set(k[m].tolist())
            kS.append(k[m])
            sS.append(zz["score"].astype(np.float64)[m])
        P = {
            "B": (np.concatenate(kB), np.concatenate(sB)),
            "S": (np.concatenate(kS), np.concatenate(sS)),
        }
        for nm in P:
            gate(
                bool(np.isfinite(P[nm][1]).all()),
                f"{row['label']} {nm}: finite pool scores",
            )
        mx = {}
        for arm in ("sim1", "run1", "run2"):
            v = np.full(len(cfg["cells"]), np.nan)
            for i, c in enumerate(cfg["cells"]):
                fol, key = cfg["arm_rows"][arm][c]
                zz = folder(fol)
                ix = zz["key"].astype(str) == key
                gate(
                    bool(ix.any()) and bool(np.isfinite(zz["score"][ix]).all()),
                    f"{row['label']}: arm rows present and finite",
                )
                v[i] = float(zz["score"].astype(np.float64)[ix].max())
            mx[arm] = v
        return P, mx

    def thr_draws(
        s, cols, W
    ):  # A24 / wthr weighted order statistic per draw; s raw unit scores, cols their main columns
        o = np.argsort(-s, kind="mergesort")
        zs, cc = s[o], cols[o]
        T = np.empty(W.shape[0])
        for b0 in range(0, W.shape[0], 500):
            cw = np.cumsum(W[b0 : b0 + 500][:, cc], 1)
            k = np.floor(FF * cw[:, -1] + 1e-9)
            idx = np.minimum((cw <= k[:, None]).sum(1), len(zs) - 1)
            T[b0 : b0 + 500] = np.where(k >= 1, zs[idx], np.nan)
        return T

    def catches(x, T, wc):  # a NaN threshold (k < 1) gives a NaN catch, as in Stage A
        c = (wc * (x[None, :] > T[:, None])).sum(1) / wc.sum(1)
        return np.where(np.isnan(T), np.nan, c)

    part_b = {}
    rows_b = [r_ for r_ in cfg["rows"] if r_["folds"] == 1]
    gate(len(rows_b) == 8, "Part B: exactly the 8 single-model probe x readout rows")
    for row in rows_b:
        lab = row["label"]
        P, mx = load_row(row)
        cells = cfg["cells"]
        mains = sorted(
            {main_of[c] for c in cells}
            | {main_of[k.split("|", 1)[1]] for nm in P for k in P[nm][0]}
        )
        col = {m_: i for i, m_ in enumerate(mains)}
        W = np.vstack(
            [
                np.ones((1, len(mains))),
                rng.multinomial(
                    len(mains), np.full(len(mains), 1 / len(mains)), size=N_DRAWS_B
                ),
            ]
        ).astype(np.float64)
        wc = W[:, [col[main_of[c]] for c in cells]]
        pc = {
            nm: np.array([col[main_of[k.split("|", 1)[1]]] for k in P[nm][0]])
            for nm in P
        }

        def rm_hit(T, mx=mx, wc=wc):
            return (catches(mx["run1"], T, wc) + catches(mx["run2"], T, wc)) / 2

        def run(keepB=None, keepS=None, P=P, pc=pc, W=W, mx=mx, wc=wc, rm_hit=rm_hit):
            sel = {
                nm: (
                    np.ones(len(P[nm][1]), bool)
                    if keep is None
                    else np.isin(P[nm][0], keep)
                )
                for nm, keep in (("B", keepB), ("S", keepS))
            }
            TJ = {nm: thr_draws(P[nm][1][sel[nm]], pc[nm][sel[nm]], W) for nm in P}
            TF = {nm: np.full(W.shape[0], TJ[nm][0]) for nm in P}
            return components(
                catches(mx["sim1"], TF["S"], wc),
                catches(mx["sim1"], TJ["S"], wc),
                rm_hit(TF["B"]),
                rm_hit(TJ["B"]),
            ), {nm: TJ[nm][0] for nm in P}

        full, T0 = run()
        # gates: draw-0 thresholds equal the export's and Stage A's S_new pools; draw-0 separate-line hits equal the export
        gate(
            T0["B"] == man["pools"][lab]["H_B"][F]["T"]
            and T0["S"] == man["pools"][lab]["H_sim"][F]["T"],
            f"{lab}: draw-0 thresholds equal the export",
        )
        st = recs[lab[-1]]["rows"][lab]["S_new"]["pools_at_draw0"][F]
        gate(
            abs(T0["B"] - st["B"]["T"]) <= 1e-12
            and abs(T0["S"] - st["S"]["T"]) <= 1e-12,
            f"{lab}: draw-0 thresholds equal Stage A's S_new pools",
        )
        H = hz[f"S_new|separate|{F}|{lab}"]
        gate(
            np.array_equal(H[:, 0], (mx["sim1"] > T0["S"]).astype(np.int8))
            and np.array_equal(H[:, 2], (mx["run1"] > T0["B"]).astype(np.int8))
            and np.array_equal(H[:, 3], (mx["run2"] > T0["B"]).astype(np.int8)),
            f"{lab}: draw-0 hits equal the export",
        )
        halves = {"B": [], "S": []}
        for _ in range(N_HALVES):
            for nm in ("B", "S"):
                runs_ = sorted(set(P[nm][0].tolist()))
                keep = list(rng.choice(runs_, size=len(runs_) // 2, replace=False))
                c_, _t = run(
                    keepB=keep if nm == "B" else None, keepS=keep if nm == "S" else None
                )
                pred = vj_scaled(
                    full, 2.0 if nm == "S" else 1.0, 2.0 if nm == "B" else 1.0
                )
                halves[nm].append(
                    {
                        "V_shift": c_["V_dc" if nm == "B" else "V_da"],
                        "V_J": c_["V_J"],
                        "V_J_predicted_1_over_n": pred,
                    }
                )
        sn = recs[lab[-1]]["rows"][lab]
        part_b[lab] = {
            "n_cells": len(cells),
            "n_mains": len(mains),
            "draws": N_DRAWS_B,
            "full": full,
            "halves": {
                nm: {
                    "shift_var_ratio_half_over_full": [
                        safe_div(h_["V_shift"], full["V_dc" if nm == "B" else "V_da"])
                        for h_ in halves[nm]
                    ],
                    "V_J_ratio_actual_over_predicted": [
                        h_["V_J"] / h_["V_J_predicted_1_over_n"] for h_ in halves[nm]
                    ],
                    "hw_ratio_half_over_full": [
                        float(np.sqrt(h_["V_J"] / full["V_J"])) for h_ in halves[nm]
                    ],
                    "hw_ratio_predicted": float(
                        np.sqrt(halves[nm][0]["V_J_predicted_1_over_n"] / full["V_J"])
                    ),
                }
                for nm in halves
            },
            "pool_runs": {nm: len(set(P[nm][0].tolist())) for nm in P},
            "stage_a_S_new_reference": {  # same probe, Stage A's 272-cell S_new cohort and stratified draws (approximate comparison)
                "cohort_cells": sn["S_new"]["coverage"]["cohort_cells"],
                "V_da_V_dc_stage_a": [
                    result["readouts"][lab[-1]][lab]["per_origin_components"]["S_new"][
                        k_
                    ]
                    for k_ in ("V_da", "V_dc")
                ],
                "V_da_V_dc_here": [full["V_da"], full["V_dc"]],
            },
        }
    result["part_b_S_new_pool_scaling"] = part_b
    result["gates_passed"] = {"n": len(gates), "distinct": sorted(set(gates))}

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, (np.floating, float)):
            return float(o) if np.isfinite(o) else None
        if isinstance(o, np.integer):
            return int(o)
        return o

    b = json.dumps(clean(result), indent=1).encode()
    name = f"thrsplit_{meta['stamp_pt']}_{meta['app_id']}.json"
    d = Path("/solavg/xfer/morerunsvar")
    d.mkdir(parents=True, exist_ok=True)
    with (d / name).open("xb") as fh:
        fh.write(b)
    solavg.commit()
    return {
        "name": name,
        "volume_path": f"xfer/morerunsvar/{name}",
        "sha256": hashlib.sha256(b).hexdigest(),
        "bytes": b,
    }


@app.local_entrypoint()
def main():
    def git(*a):
        return subprocess.run(
            ["git", *a], capture_output=True, check=True, cwd=HERE
        ).stdout

    if git("status", "--porcelain", "--untracked-files=no", "--", ".").strip():
        raise SystemExit("commit the experiment folder before running")
    blobs = {}
    for k, (c, p, sha) in PINS.items():
        b = git("show", f"{c}:{p}")
        if hashlib.sha256(b).hexdigest() != sha:
            raise SystemExit(f"{k}: sha256 differs from the pin")
        blobs[k] = b
    c, p, sha, keys = EP2  # counts only: the two summary rows' run-count fields
    b = git("show", f"{c}:{p}")
    if hashlib.sha256(b).hexdigest() != sha:
        raise SystemExit("epoch-2 pool keys: sha256 differs from the pin")
    summ = json.loads(b)["summary"]
    blobs["ep2_counts"] = json.dumps(
        {
            o: {
                f: summ[key][f]
                for f in ("epoch1_members", "epoch2_admitted", "members_without_epoch2")
            }
            for o, key in keys.items()
        }
        | {"source": {"commit": c, "path": p, "sha256": sha, "summary_keys": keys}}
    ).encode()
    meta = {
        "git": git("rev-parse", "HEAD").decode().strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pins": {
            k: {"commit": c, "path": p, "sha256": s} for k, (c, p, s) in PINS.items()
        },
        "app": APP,
        "app_id": app.app_id,
        "stamp_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime(
            "%Y-%m-%d_%H%M%S"
        ),
        "rule": "experiments/2026-10-15_q2_more_runs_variance/README.md (Follow-up 1)",
    }
    out = analyze.remote(blobs, meta)
    dest = HERE / "results" / out["name"]
    with dest.open("xb") as fh:
        fh.write(out["bytes"])
    print(
        json.dumps(
            {
                "result": str(dest.relative_to(HERE.parents[1])),
                "sha256": out["sha256"],
                "volume_path": out["volume_path"],
            }
        )
    )
