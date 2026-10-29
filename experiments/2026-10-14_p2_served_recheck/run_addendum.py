"""Addendum job (ADDENDUM.md): the POST-HOC k=8 float32 tolerance per cell and layout, the t16 unit under Append 28e, the
recomputed per-run score files (09's option (a)), and flip COUNTS (09's option (b)). Modal CPU only.

Prints and records counts, yes/no, ratios and hashes only: no thresholds, rates or catches, no score values (the score
values go only into the per-run files on the Volume, for 09's reading job). Token ids only; nothing decoded.
Threshold mechanics follow 09's run_served_read.py @ 863d5144b (read for those mechanics only): order_stat over the
descending pool, k = floor(a * n + 1e-9), T = s[min(k, n - 1)], hit = score > T; honest keys = the READER's run keys in the
pool worlds; pool = v6's reader-mechanical pool per probe x readout x alpha; R = reply_mean run mean, C = command run max;
attack arms run1/run2/sim on v6's 90 estimation-cohort tasks.

  uv run --no-project --with 'modal>=1.2' --with 'numpy>=2' modal run --detach experiments/2026-10-14_p2_served_recheck/run_addendum.py
"""

import hashlib
import json
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-p2probes_servedrecheck_kill-2026-11-13"
STAGE = "served_2026-10-14_200651"
EXTRACT = {L: f"{STAGE}_extract_{L}_v1" for L in ("A0", "A", "B")}
LAYOUTS = ("A0", "A", "B")
WORLDS = ("run1", "run2", "sim", "honest_a", "honest_b", "honest_c")
ATTACK = ("run1", "run2", "sim")
READOUTS = ("reply_mean", "reply_last", "command", "call_end_content")
KIND_OF = {"reply_mean": "output_end", "reply_last": "output_end", "command": "command", "call_end_content": "call_end"}
PROBES = ("fibs", "refusal", "harmfulness", "highstakes", "apollo_d2_tight")
P2_READ = ("fibs", "harmfulness", "refusal", "highstakes")  # the p2 probes 09 reads
FIXED = (15, 77)
K_TOL = 8.0  # POST HOC (ADDENDUM.md section 2, committed before this job)
EPS32 = 2.0 ** -24
ALPHAS = (0.01, 0.05)
RO09 = {"R_reply_meanpooled": ("reply_mean", "mean"), "C_anycommand_max": ("command", "max")}
POOLS = {"B": ("honest_b",), "BuC": ("honest_b", "honest_c")}
IM_START, IM_END, ASSISTANT, NL, TR_O, TR_C = 248045, 248046, 74455, 198, 248066, 248067
HF_SNAP = "hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654"

PIN_SIMH = "f14b1253"
RECORDS = {"fibs": ("d74ccaba0", "experiments/2026-10-14_p2_fibs_pair/results/fibs_scores_2026-10-14_124948.json", None),
           "refusal": ("a8d786425", "experiments/2026-10-14_p2_refusal_harm/results/scores_2026-10-14_145748.json", "refusal"),
           "harmfulness": ("a8d786425", "experiments/2026-10-14_p2_refusal_harm/results/scores_2026-10-14_145748.json", "harmfulness"),
           "highstakes": ("3d04b7936", "experiments/2026-10-14_p2_highstakes/results/scores_2026-10-14_133519.json", "highstakes"),
           "apollo_d2_tight": ("54ea9724", "experiments/2026-10-14_p2_apollo/results/scores_2026-10-14_145858.json", "apollo_d2_tight")}
LINEAR = {"refusal": "experiments/2026-10-14_p2_refusal_harm/results/h200_check_2026-10-14_145255.json",
          "harmfulness": "experiments/2026-10-14_p2_refusal_harm/results/h200_check_2026-10-14_145255.json",
          "highstakes": "experiments/2026-10-14_p2_highstakes/results/read_fit_2026-10-14_131233.json",
          "apollo_d2_tight": "experiments/2026-10-14_p2_apollo/results/fit_tight_2026-10-14_145508.json"}
CONV = "experiments/2026-10-14_q2_fibs_probe/results/convert_fibs_linear_2026-10-14_115759.json"
MANIFEST_B = ("46ca25fc", "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/correctedb_manifest_v1.json",
              "bf847d425b80d95ebb75a9ed1d524fc68c9efcd014710ddc18e445ed9d373ebc")
V6 = ("863d5144b", "experiments/2026-10-14_xerr_prelim_09/results/xerr_v6_2026-10-14_161122.json",
      "e789144b62f71e0fb168dc3106656c2b5f29aadafe12df0f86dd1f74441d7645")
MAIN_RECORD = ("a67045a40", "experiments/2026-10-14_p2_served_recheck/results/recheck_2026-10-14_230545.json",
               "0f8ab38177f8c4dbfabebe0a96ddc8d57ff7afed5d43a7b3819fc0fddf40cbb3")
READ09 = ("863d5144b", "experiments/2026-10-14_served_check_09/run_served_read.py")

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.3.3", "tokenizers==0.22.1")
ro = lambda n: modal.Volume.from_name(n).read_only()  # noqa: E731
V_ACTS, V_CW, V_Q3 = ro("probe-shift_aswin-surya_sub-p2served_acts_kill-2026-11-14"), ro("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"), ro("probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08")
V_Q4, V_RL, V_HF = ro("probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"), ro("probe-shift_aswin-surya_sub-phase2_request-log_kill-2026-11-06"), ro("probe-shift_aswin-surya_hf-cache")
V_P2RO = ro("probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13")
out_vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _read(path, keep=None):
    import re

    rx = re.compile(r'"key":\s*"([^"]+)"')
    for line in open(path, "rb"):
        if keep is not None:
            m = rx.search(line[:2048].decode("utf-8", "replace"))
            if not m or m.group(1) not in keep:
                continue
        yield json.loads(line)


def ukey(c):
    if any(k not in c for k in ("kind", "src_irid", "call_index")):
        return None
    return (c["irid"], c["kind"], c["src_irid"], c["call_index"])


def order_stat(values, a):
    """09's order_stat (run_served_read.py @ 863d5144b), reproduced."""
    import numpy as np

    s = np.sort(np.asarray(values, float), kind="mergesort")[::-1]
    if not len(s) or not np.isfinite(s).all():
        raise ValueError("empty or non-finite pool")
    k = int(np.floor(a * len(s) + 1e-9))
    return float(s[min(k, len(s) - 1)])


# ----------------------------------------------------------------------------------------------------- projection
@app.function(image=image, volumes={"/acts": V_ACTS, "/q4": V_Q4}, cpu=4, memory=12288, timeout=900, retries=0)
def project(src: str, layer: int, vecs: dict, chunks: list) -> dict:
    """Every row of the listed chunks at one layer: f32 path, float64 exact, and sum_i |x_i w_i| (float64)."""
    import numpy as np

    V = {n: (v["kind"], {k: np.asarray(x) for k, x in v["tensors"].items()}) for n, v in vecs.items()}
    out = {}
    for folder, chunk in chunks:
        X = np.load(f"{'/acts' if src in LAYOUTS else '/q4'}/{folder}/by_layer/L{layer}/{chunk}.npy")
        if X.dtype != np.float16 or X.shape[1] != 5120 or not np.isfinite(X).all():
            raise ValueError(f"{folder}/{chunk} L{layer}: bad rows")
        Xf, Xd = X.astype(np.float32), X.astype(np.float64)
        Xa = np.abs(Xd)
        res = {}
        for name, (kind, t) in V.items():
            if kind == "fibs":
                w, s, b = t["linear.weight"].reshape(-1), t["input_scale"].reshape(()), t["linear.bias"].reshape(())
                f32 = (Xf / s.astype(np.float32)) @ w.astype(np.float32) + b.astype(np.float32)
                f64 = (Xd / float(s)) @ w.astype(np.float64) + float(b)
                ad = (Xa / abs(float(s))) @ np.abs(w.astype(np.float64)) + abs(float(b))
            else:
                d = t["direction"].reshape(-1)
                b = t["bias"].reshape(()) if "bias" in t else np.zeros((), np.float32)
                f32 = Xf @ d.astype(np.float32) + b.astype(np.float32)
                f64 = Xd @ d.astype(np.float64) + float(b)
                ad = Xa @ np.abs(d.astype(np.float64)) + abs(float(b))
            res[name] = (f32.astype(np.float32), f64, ad)
        out[f"{folder}/{chunk}"] = res
    return out


def row_value(P, probe, layers, chunk, row):
    """(f32 score, exact score, bound scale) of one row; FIBS: mean of sigmoids; bound scale = 0.25 * mean_l sum|xw|."""
    import numpy as np

    if probe == "fibs":
        zz = [P[L][chunk]["fibs"] for L in layers]
        z32 = np.array([z[0][row] for z in zz], dtype=np.float32).astype(np.float64)
        z64 = np.array([z[1][row] for z in zz])
        ad = np.array([z[2][row] for z in zz])
        return (float(np.mean(1.0 / (1.0 + np.exp(-z32)))), float(np.mean(1.0 / (1.0 + np.exp(-z64)))), 0.25 * float(ad.mean()),
                z64, ad)
    x = P[layers[0]][chunk][probe]
    return float(x[0][row]), float(x[1][row]), float(x[2][row]), None, None


def aggregate(rows_by_key: dict) -> dict:
    """rows: key -> [(stored, f32, exact, bound)] -> per run: mean and max of each, and the run-level ratio."""
    import numpy as np

    out = {}
    for k, lst in rows_by_key.items():
        a = np.array(lst, dtype=np.float64)
        st, mi, ex, bd = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        jm = int(np.argmax(ex))
        out[k] = {"stored_mean": float(st.mean()), "mine_mean": float(mi.mean()), "exact_mean": float(ex.mean()), "bound_mean": float(bd.mean()),
                  "stored_max": float(st.max()), "mine_max": float(mi.max()), "exact_max": float(ex.max()), "bound_max": float(bd.max()),
                  "_jm": jm}
    return out


def ratio_stats(pairs):
    """pairs: [(|stored - exact|, bound scale)] -> max ratio, n above K_TOL, pass."""
    import numpy as np

    if not pairs:
        return {"n": 0, "max_ratio": None, "n_above_k": 0, "pass": False}
    a = np.array(pairs, dtype=np.float64)
    r = a[:, 0] / np.maximum(EPS32 * a[:, 1], 1e-300)
    return {"n": len(a), "max_ratio": float(r.max()), "n_above_k": int((~(r <= K_TOL)).sum()), "pass": bool((r <= K_TOL).all())}


# ------------------------------------------------------------------------------------------ served layouts (A0/A/B)
@app.function(image=image, volumes={"/acts": V_ACTS, "/cw": V_CW}, cpu=4, memory=32768, timeout=900, retries=0)
def served_layout(layout: str, proj: dict, projA: dict, probe_layers: dict) -> dict:
    import os
    from collections import defaultdict

    import numpy as np

    def meta_index(L):
        last, mean = {}, {}
        for w in sorted(os.listdir(f"/acts/{EXTRACT[L]}/meta")):
            for f in sorted(os.listdir(f"/acts/{EXTRACT[L]}/meta/{w}")):
                m = json.load(open(f"/acts/{EXTRACT[L]}/meta/{w}/{f}"))
                rk, ch = f"{m['arm']}|{m['task']}", f"{EXTRACT[L]}/{m['chunk']}"
                for c in m["calls"]:
                    last[(rk, c["reader_row"])] = (c["kind"], m["epoch"], ch, m["row_start"] + c["row"])
                    if c["kind"] == "output_end":
                        mean[(rk, c["reader_row_mean"])] = ("output_end", m["epoch"], ch, m["row_start"] + c["row_mean"])
        return last, mean

    idx = {"self": meta_index(layout)}
    if layout == "A0":
        idx["A"] = meta_index("A")
    out = {"cells": {}, "runs": {}, "problems": 0}
    for probe in PROBES:
        rm = json.load(open(f"/cw/served/{layout}/p2/{probe}__run_means.json"))
        for ro_ in READOUTS:
            zf = np.load(f"/cw/served/{layout}/p2/{probe}__{ro_}.npz", allow_pickle=False)
            z = {f: zf[f] for f in zf.files}  # load each array ONCE (NpzFile re-reads on every access)
            rows, rowpairs, logitpairs, unmatched = defaultdict(list), [], [], 0
            for i in range(len(z["key"])):
                k = str(z["key"][i])
                src = "A" if (layout == "A0" and not k.startswith(ATTACK)) else "self"
                m = idx[src][1 if ro_ == "reply_mean" else 0].get((k, int(z["reader_row"][i])))
                if m is None or m[0] != KIND_OF[ro_] or m[1] != int(z["served_epoch"][i]):
                    out["problems"] += 1
                    unmatched += 1
                    continue
                f32, ex, bd, z64, ad = row_value(projA if src == "A" else proj, probe, probe_layers[probe], m[2], m[3])
                st = float(z["score"][i])
                rows[k].append((st, f32, ex, bd))
                rowpairs.append((abs(st - ex), bd))
                if probe == "fibs":
                    logitpairs += [(abs(float(z["logits"][i][j]) - z64[j]), ad[j]) for j in range(len(z64))]
            agg = aggregate(rows)
            if unmatched or not agg:
                out["cells"][f"{probe}|{ro_}"] = {"complete": False, "unmatched rows": unmatched}
                out["runs"][f"{probe}|{ro_}"] = None
                out["runs"][f"{probe}|{ro_}|stored_keys"] = sorted(rm[ro_])
                continue
            run_mean_pairs = [(abs(v["stored_mean"] - v["exact_mean"]), v["bound_mean"]) for v in agg.values()]
            cell = {"complete": True, "rows": ratio_stats(rowpairs), "runs (mean)": ratio_stats(run_mean_pairs),
                    "stored run means == run_means.json (within 1e-9)": all(abs(agg[k]["stored_mean"] - rm[ro_][k]) <= 1e-9 for k in agg)
                    and set(agg) == set(rm[ro_]),
                    "max |stored - mine| run mean": max(abs(v["stored_mean"] - v["mine_mean"]) for v in agg.values())}
            if ro_ == "command":
                cell["runs (max)"] = ratio_stats([(abs(v["stored_max"] - v["exact_max"]), v["bound_max"]) for v in agg.values()])
                cell["max |stored - mine| run max"] = max(abs(v["stored_max"] - v["mine_max"]) for v in agg.values())
            if probe == "fibs":
                cell["fibs logits"] = ratio_stats(logitpairs)
            out["cells"][f"{probe}|{ro_}"] = cell
            out["runs"][f"{probe}|{ro_}"] = {k: [v["stored_mean"], v["mine_mean"], v["stored_max"], v["mine_max"]] for k, v in agg.items()}
            out["runs"][f"{probe}|{ro_}|stored_keys"] = sorted(rm[ro_])
    return out


# ------------------------------------------------------------------------------------------------ reader layout
@app.function(image=image, volumes={"/q4": V_Q4, "/p2": V_P2RO}, cpu=4, memory=32768, timeout=900, retries=0)
def reader_layout(proj: dict, epochs: list, probe_layers: dict, rows_files: dict, hb_skip: dict, rec_runs: dict) -> dict:
    """Reader layout: my per-row scores from the reader's own meta and activations; stored rows paired by call_index order;
    run scores R (reply_mean mean), C (command max) and the honest_b gate cells at run level vs the committed records."""
    from collections import defaultdict

    import numpy as np

    mine = defaultdict(lambda: defaultdict(list))  # (probe, readout) -> key -> [(call_index, f32, exact, bound)]
    for e in epochs:
        m = json.load(open(f"/q4/{e['meta']}"))
        ch = f"{e['extraction_run']}/{m['chunk']}"
        skip = set(tuple(x) for x in hb_skip.get(e["run_key"], []))
        for c in m["calls"]:
            if c.get("row") is None:
                continue
            sel = []
            if c["kind"] == "output_end":
                sel = [("reply_mean", c["row_mean"]), ("reply_last", c["row"])]
            elif c["kind"] == "command":
                sel = [("command", c["row"])]
            elif c["kind"] == "call_end" and e["run_key"].startswith("honest_b|") and ukey(c) not in skip:
                sel = [("call_end_content", c["row"])]
            for ro_, row in sel:
                for probe in PROBES:
                    f32, ex, bd, _, _ = row_value(proj, probe, probe_layers[probe], ch, m["row_start"] + row)
                    mine[(probe, ro_)][e["run_key"]].append((c["call_index"], f32, ex, bd))
    out = {"cells": {}, "runs": {}, "gate": {}, "problems": {}}
    for probe in PROBES:
        for ro_ in ("reply_mean", "command"):
            path, want = rows_files[probe][ro_]
            raw = open(f"/p2/{path}", "rb").read()
            if sha(raw) != want:
                raise ValueError(f"{path}: sha256 differs from its record")
            import io

            z = np.load(io.BytesIO(raw), allow_pickle=False)
            st = defaultdict(list)
            for k, ci, sc_ in zip(z["key"].astype(str), z["call_index"], z["score"], strict=True):
                st[k].append((int(ci), float(sc_)))
            pairs, bad, agg = [], 0, {}
            for k, lst in st.items():
                my = mine[(probe, ro_)].get(k, [])
                sv = np.array([x[1] for x in lst])
                if my:
                    a_ = np.array([x[1:] for x in my], dtype=np.float64)
                    agg[k] = {"stored_mean": float(sv.mean()), "stored_max": float(sv.max()), "mine_mean": float(a_[:, 0].mean()),
                              "mine_max": float(a_[:, 0].max()), "exact_mean": float(a_[:, 1].mean()), "exact_max": float(a_[:, 1].max()),
                              "bound_mean": float(a_[:, 2].mean()), "bound_max": float(a_[:, 2].max())}
                else:
                    agg[k] = {"stored_mean": float(sv.mean()), "stored_max": float(sv.max()), "mine_mean": float("nan"), "mine_max": float("nan"),
                              "exact_mean": float("nan"), "exact_max": float("nan"), "bound_mean": 1.0, "bound_max": 1.0}
                if [c for c, _ in lst] != [c for c, *_ in my]:
                    bad += 1
                    continue
                for (_, sv1), (_, f32, ex, bd) in zip(lst, my, strict=True):
                    pairs.append((abs(sv1 - ex), bd))
            out["problems"][f"{probe}|{ro_}"] = {"runs whose call_index sequence differs from mine (row pairing skipped)": bad,
                                                  "stored runs": len(st), "my runs": len(mine[(probe, ro_)]),
                                                  "stored runs missing in mine": sum(1 for k in st if k not in mine[(probe, ro_)])}
            cell = {"rows (paired)": ratio_stats(pairs),
                    "runs (mean)": ratio_stats([(abs(v["stored_mean"] - v["exact_mean"]), v["bound_mean"]) for v in agg.values()])}
            if ro_ == "command":
                cell["runs (max)"] = ratio_stats([(abs(v["stored_max"] - v["exact_max"]), v["bound_max"]) for v in agg.values()])
            rr = rec_runs[probe].get(ro_, {})
            cell["stored row means == record run_scores (within 1e-9)"] = bool(rr) and set(agg) == set(rr) and all(
                abs(agg[k]["stored_mean"] - rr[k]) <= 1e-9 for k in agg)
            out["cells"][f"{probe}|{ro_}"] = cell
            out["runs"][f"{probe}|{ro_}"] = {k: [v["stored_mean"], v["mine_mean"], v["stored_max"], v["mine_max"]] for k, v in agg.items()}
        # the honest_b gate at run level vs the committed records (4 readouts, means)
        for ro_ in READOUTS:
            ref = {k: v for k, v in rec_runs[probe].get(ro_, {}).items() if k.startswith("honest_b|")}
            my = mine[(probe, ro_)]
            pr, d = [], []
            for k, v in ref.items():
                lst = my.get(k)
                if not lst:
                    pr.append((float("inf"), 1.0))
                    continue
                a = np.array([x[1:] for x in lst], dtype=np.float64)
                pr.append((abs(v - a[:, 1].mean()), a[:, 2].mean()))
                d.append(abs(v - a[:, 0].mean()))
            out["gate"][f"{probe}|{ro_}"] = {"runs": len(ref), "max |record - mine f32|": max(d) if d else None,
                                             "n |record - mine f32| > 1e-6": int(sum(x > 1e-6 for x in d)),
                                             "tolerance (record vs exact)": ratio_stats(pr)}
    return out


# ------------------------------------------------------------------------------------------------------- t16
@app.function(image=image, volumes={"/acts": V_ACTS, "/q3": V_Q3, "/cw": V_CW, "/hf": V_HF}, cpu=2, memory=16384, timeout=600, retries=0)
def t16_unit() -> dict:
    """The output_end unit of honest_a|t16_file_indexing that failed the 'final user turn' check: its keys, turns and the
    turn roles after its readout (ids only), and the run's request ids for the log check."""
    import os

    import numpy as np
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(f"/hf/{HF_SNAP}/tokenizer.json")
    role = {n: tok.encode(n, add_special_tokens=False).ids for n in ("user", "assistant", "system")}
    A = [r for r in _read(f"/acts/{STAGE}/A/inputs/honest_a.jsonl") if r["task"] == "t16_file_indexing"]
    out = {"role ids": role, "units": [], "run irids": []}
    for r in A:
        S = None
        for f in sorted(os.listdir(f"/q3/{r['served_staging']}/inputs")):
            for x in _read(f"/q3/{r['served_staging']}/inputs/{f}", keep={r["key"]}):
                if x["epoch"] == r["epoch"]:
                    S = x
        s = np.asarray(S["ids"])
        sc = {ukey(c): c for c in S["calls"] if ukey(c) is not None}
        turn_of = {c["irid"]: c["turn"] for c in S["calls"] if c.get("kind") == "command" and c.get("src_irid") == c["irid"]}
        out["run irids"] += sorted({c["irid"] for c in S["calls"]} | {c.get("src_irid") for c in S["calls"] if c.get("src_irid")})
        for c in r["calls"]:
            if c["kind"] != "output_end":
                continue
            v = sc[ukey(c)]
            rs, n = v["readout"], v["n"]
            st = np.flatnonzero(s[:n] == IM_START)
            h = int(st[-1])
            if len(st) >= 2 and s[h + 1] == ASSISTANT and int(st[-2]) < rs < h:
                continue
            tro = int(np.flatnonzero(s[:rs] == TR_O)[-1])
            trc = rs + int(np.flatnonzero(s[rs:] == TR_C)[0])
            out["units"].append({"served key": r["key"], "epoch": r["epoch"], "irid (consuming request)": c["irid"], "src_irid": c["src_irid"],
                                 "call_index": c["call_index"], "served readout": rs, "request n": n, "A readout": c["readout"], "A span": c["span"],
                                 "src turn (its own command call)": turn_of.get(c["src_irid"]), "consuming request turn (its own command call)": turn_of.get(c["irid"]),
                                 "role id of the turn holding the readout": int(s[int(st[st < rs][-1]) + 1]),
                                 "turns after the readout (offset, role id)": [(int(p) - rs, int(s[p + 1])) for p in st if p > rs],
                                 "<tool_response> / </tool_response> offsets": [tro - rs, trc - rs],
                                 "served token at readout": int(s[rs]), "A token at readout": int(np.asarray(r["ids"])[c["readout"]])})
    out["run irids"] = sorted(set(out["run irids"]))
    return out


@app.function(image=image, volumes={"/rl": V_RL}, cpu=2, memory=8192, timeout=300, retries=0)
def scan_log(fname: str, wanted: list) -> dict:
    import gzip
    import re

    import numpy as np

    W, out = set(wanted), {}
    rx = re.compile(rb'"request_id":\s*"([A-Za-z0-9]+)"')
    with gzip.open(f"/rl/{fname}", "rb") as f:
        while True:
            try:
                line = f.readline()
            except (EOFError, OSError):
                break
            if not line:
                break
            m = rx.search(line[:200])
            if not m or m.group(1).decode() not in W:
                continue
            r = json.loads(line)
            ch = (r.get("choices") or [{}])[0]
            if r.get("status") == 200 and r.get("complete") and r.get("prompt_token_ids") is not None and ch.get("token_ids") is not None:
                out.setdefault(r["request_id"], []).append({"received": r["received"], "lp": len(r["prompt_token_ids"]),
                                                            "full": np.asarray(r["prompt_token_ids"] + ch["token_ids"], dtype="<i4")})
    return out


# ------------------------------------------------------------------------------------------------- orchestrator
@app.function(image=image, volumes={"/acts": V_ACTS, "/cw": V_CW, "/q4": V_Q4, "/rl": V_RL, "/out": out_vol}, cpu=2, memory=32768, timeout=3000, retries=0)
def orchestrate(payload: dict) -> dict:
    import os
    from collections import Counter
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import numpy as np

    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    W, mb = payload["weights"], payload["manifest_b"]
    probe_layers = {p: W[p]["layers"] for p in W}
    layers = sorted({L for p in PROBES for L in probe_layers[p]})
    vec_at = {L: {p: {"kind": W[p]["kind"], "tensors": W[p]["tensors"][str(L)]} for p in PROBES if L in probe_layers[p]} for L in layers}
    calls = {}
    for Lo in LAYOUTS:
        ch = sorted(f[:-5] for f in os.listdir(f"/acts/{EXTRACT[Lo]}/chunks") if f.endswith(".json"))
        for L in layers:
            calls[(Lo, L)] = project.spawn(Lo, L, vec_at[L], [(EXTRACT[Lo], c) for c in ch])
    epochs = [{"run_key": f"{e['world']}|{e['task']}", "extraction_run": e["extraction_run"], "meta": e["meta_file"], "key": e["key"],
               "inputs": e["staged_inputs_file"]} for e in mb["epochs"]]
    for f in sorted(os.listdir("/q4/q4b_extract_c_v1/meta/honest_b")):
        m = json.load(open(f"/q4/q4b_extract_c_v1/meta/honest_b/{f}"))
        epochs.append({"run_key": f"honest_c|{m['task']}", "extraction_run": "q4b_extract_c_v1", "meta": f"q4b_extract_c_v1/meta/honest_b/{f}",
                       "key": m["key"], "inputs": "q4f_c_v1/inputs/honest_b.jsonl"})
    rch = sorted({(e["extraction_run"], json.load(open(f"/q4/{e['meta']}"))["chunk"]) for e in epochs})
    for L in layers:
        calls[("R", L)] = project.spawn("R", L, vec_at[L], rch)
    t16c = t16_unit.spawn()
    # honest_b content-only skip list from the reader's staged ids (as in the main recheck's gate)
    hb = [e for e in epochs if e["run_key"].startswith("honest_b|")]
    hb_skip = {}
    for f in sorted({e["inputs"] for e in hb}):
        want = {e["key"]: e["run_key"] for e in hb if e["inputs"] == f}
        for r in _read(f"/q4/{f}", keep=set(want)):
            ids = np.asarray(r["ids"])
            hb_skip[want[r["key"]]] = [list(ukey(c)) for c in r["calls"] if c.get("kind") == "call_end" and c.get("readout") is not None
                                       and ukey(c) is not None and int(ids[c["readout"]]) in FIXED]
    proj = {Lo: {} for Lo in LAYOUTS + ("R",)}
    for (Lo, L), fc in calls.items():
        proj[Lo][L] = fc.get()
    sl = {Lo: served_layout.spawn(Lo, proj[Lo], proj["A"] if Lo == "A0" else {}, probe_layers) for Lo in LAYOUTS}
    rl = reader_layout.spawn(proj["R"], epochs, probe_layers, payload["rows_files"], hb_skip, payload["rec_runs"])
    # t16 under Append 28e, with the logged ids
    t16 = t16c.get()
    files = sorted(f for f in os.listdir("/rl") if f.startswith("requests_") and "2026-10-08" <= f[9:19] <= "2026-10-11")
    logs = {}
    for fc in [scan_log.spawn(f, t16["run irids"]) for f in files]:
        for i, xs in fc.get().items():
            logs.setdefault(i, []).extend(xs)
    req = {}
    for i, xs in logs.items():
        u = {sha(x["full"].tobytes()): x for x in xs}
        if len(u) == 1:
            req[i] = next(iter(u.values()))
    order = sorted(req, key=lambda i: datetime.fromisoformat(req[i]["received"]))
    my_ep, e_, prev = {}, -1, None
    for i in order:
        p = req[i]["full"][:req[i]["lp"]]
        if prev is None or len(p) < len(prev) or not np.array_equal(p[:len(prev)], prev):
            e_ += 1
        my_ep[i], prev = e_, req[i]["full"]
    for u in t16["units"]:
        q, qs = req.get(u["irid (consuming request)"]), req.get(u["src_irid"])
        rs = u["served readout"]
        u["28e checks (logged ids)"] = {
            "both requests found once in the logs": q is not None and qs is not None,
            "consuming request is the FIRST logged request after the source (time order)":
                bool(q is not None and qs is not None and order.index(u["irid (consuming request)"]) == order.index(u["src_irid"]) + 1),
            "consuming request's logged length == staged n": bool(q is not None and len(q["full"]) == u["request n"]),
            "readout inside the consuming request's logged prompt": bool(q is not None and rs < q["lp"]),
            "logged token at readout == served token": bool(q is not None and int(q["full"][rs]) == u["served token at readout"]),
            "logged ids: <tool_response> at -5... </tool_response> at +2 around the readout": bool(
                q is not None and int(q["full"][rs + u["<tool_response> / </tool_response> offsets"][0]]) == TR_O
                and int(q["full"][rs + u["<tool_response> / </tool_response> offsets"][1]]) == TR_C),
            "my epoch index of the consuming request (prefix rule)": my_ep.get(u["irid (consuming request)"]),
            "my epoch index of the source request": my_ep.get(u["src_irid"]),
            "logged requests of the run": len(order)}
    # served + reader results
    S = {Lo: sl[Lo].get() for Lo in LAYOUTS}
    R = rl.get()
    # ---- option (a): per-run score files (mine, float32 path), keys = the stored keys
    folder = f"/out/2026-10-14_p2_served_recheck/run_scores_{stamp}"
    os.makedirs(folder)  # exclusive: fails if it exists
    man = {"label": "PRELIMINARY: numerically within float32 accumulation; decision-identity pending. Recomputed per-run scores "
                    "(this checker's float32 path, run mean or max of per-row scores); keys = the stored keys", "files": {}, "not written": {}}

    def put(name, d, stored_keys):
        if d is None:
            man["not written"][name] = "cell incomplete (unmatched rows)"
            return
        if sorted(d) != sorted(stored_keys) or not all(np.isfinite(v) for v in d.values()):
            man["not written"][name] = "keys differ from the stored keys or a non-finite value"
            return
        blob = json.dumps({k: d[k] for k in sorted(d)}, indent=0, allow_nan=False).encode()
        with open(f"{folder}/{name}", "xb") as f:
            f.write(blob)
        man["files"][name] = {"sha256": sha(blob), "n_keys": len(d), "keys == stored keys": True}

    for Lo in LAYOUTS:
        for probe in PROBES:
            for ro_ in READOUTS:
                runs = S[Lo]["runs"][f"{probe}|{ro_}"]
                sk = S[Lo]["runs"][f"{probe}|{ro_}|stored_keys"]
                put(f"{Lo}__{probe}__{ro_}__mean.json", None if runs is None else {k: v[1] for k, v in runs.items()}, sk)
                if ro_ == "command":
                    put(f"{Lo}__{probe}__command__max.json", None if runs is None else {k: v[3] for k, v in runs.items()}, sk)
    for probe in PROBES:
        put(f"reader__{probe}__reply_mean__mean.json", {k: v[1] for k, v in R["runs"][f"{probe}|reply_mean"].items()},
            payload["rec_runs"][probe]["reply_mean"])
        put(f"reader__{probe}__command__max.json", {k: v[3] for k, v in R["runs"][f"{probe}|command"].items()},
            list(R["runs"][f"{probe}|command"]))
        put(f"reader__{probe}__command__mean.json", {k: v[1] for k, v in R["runs"][f"{probe}|command"].items()},
            payload["rec_runs"][probe]["command"])
    man["record git"] = payload["git"]
    mblob = json.dumps(man, indent=1).encode()
    with open(f"{folder}/manifest.json", "xb") as f:
        f.write(mblob)
    # ---- option (b): flip counts (counts and yes/no only)
    v6 = payload["v6"]
    c_est = v6["c_est"]
    D = {}
    for Lo in LAYOUTS:
        for p in P2_READ:
            for ro09, (ro_, agg) in RO09.items():
                runs = S[Lo]["runs"][f"{p}|{ro_}"] or {}  # an incomplete cell -> empty -> the flip cell reads "incomplete"
                j = (0, 1) if agg == "mean" else (2, 3)
                D[(Lo, p, ro09)] = ({k: v[j[0]] for k, v in runs.items()}, {k: v[j[1]] for k, v in runs.items()})
    for p in P2_READ:
        for ro09, (ro_, agg) in RO09.items():
            runs = R["runs"][f"{p}|{ro_}"]
            j = (0, 1) if agg == "mean" else (2, 3)
            stored = dict(payload["rec_runs"][p]["reply_mean"]) if ro09 == "R_reply_meanpooled" else {k: v[j[0]] for k, v in runs.items()}
            D[("reader", p, ro09)] = (stored, {k: v[j[1]] for k, v in runs.items()})
    flips, elig_total = {}, Counter()
    for (Lo, p, ro09), (Ds, Dm) in D.items():
        rkeys = D[("reader", p, ro09)][0]  # the READER's keys define the honest calibration keys (09's hon_keys)
        for a in ALPHAS:
            pool = v6["pools"][f"{ro09}|{a}"][p]
            keys = sorted(k for k in rkeys if k.split("|", 1)[0] in POOLS[pool])
            att = {w: [f"{w}|{t}" for t in c_est] for w in ATTACK}
            hon = {w: sorted(k for k in rkeys if k.split("|", 1)[0] == w) for w in ("honest_b", "honest_c")}
            need = keys + [k for v in att.values() for k in v] + [k for v in hon.values() for k in v]
            fin = lambda D_, k: k in D_ and np.isfinite(D_[k])  # noqa: E731
            miss_s, miss_m = sum(not fin(Ds, k) for k in need), sum(not fin(Dm, k) for k in need)
            cell = {"pool": pool, "eligible (v6 frozen)": p in v6["eligible"][f"{ro09}|{a}"], "missing keys (stored, mine)": [miss_s, miss_m]}
            if miss_s or miss_m:
                flips[f"{Lo}|{p}|{ro09}|{a}"] = cell | {"incomplete": True}
                continue
            Ts, Tm = order_stat([Ds[k] for k in keys], a), order_stat([Dm[k] for k in keys], a)
            d = max(abs(Ds[k] - Dm[k]) for k in need)
            cell["threshold differs (stored vs mine)"] = "yes" if Ts != Tm else "no"
            cell["max |stored - mine| over the keys used"] = d
            for w, ks in list(att.items()) + list(hon.items()):
                nf = sum((Ds[k] > Ts) != (Dm[k] > Tm) for k in ks)
                cell[w] = {"runs": len(ks), "hit status differs": nf,
                           "within max|diff| of the stored threshold": sum(abs(Ds[k] - Ts) <= d for k in ks),
                           "within 2*max|diff| (upper bound on flips)": sum(abs(Ds[k] - Ts) <= 2 * d for k in ks)}
                if cell["eligible (v6 frozen)"]:
                    elig_total["flips in eligible cells"] += nf
            flips[f"{Lo}|{p}|{ro09}|{a}"] = cell
    # ---- the record
    rec = {"label": "PRELIMINARY: numerically within float32 accumulation; decision-identity pending (ADDENDUM.md)",
           "stamp (PT)": stamp, "git": payload["git"], "inputs": payload["inputs"],
           "threshold mechanics": "read from 09's run_served_read.py @ 863d5144b for the threshold mechanics only (order_stat, hon_keys, "
                                  "calib, POOLS, readouts R mean / C max, c_est); not otherwise used",
           "scope": {"p2 probes read by 09": list(P2_READ), "apollo_d2_tight": "per-run files written; no v6 pool, not read by 09, no flips",
                     "Sol (a)": "out of scope (2e's scorer; not recomputed)"},
           "1_wording (from the main record, by key)": payload["wording"],
           "2_tolerance (POST HOC, k = 8)": {"definition": f"|stored - exact float64| <= {K_TOL} * 2^-24 * sum_i |x_i w_i| (+|b|); FIBS score: "
                                                          "0.25 * k * eps * mean_l sum|xw|; runs: mean/max of the rows' bound scale",
                                            "served": {Lo: S[Lo]["cells"] for Lo in LAYOUTS}, "served row-matching problems": {Lo: S[Lo]["problems"] for Lo in LAYOUTS},
                                            "reader (all worlds)": R["cells"], "reader pairing": R["problems"], "honest_b gate (runs)": R["gate"]},
           "3_t16": t16 | {"run irids": len(t16["run irids"])},
           "4a_run_score_files": {"folder": folder.replace("/out/", "probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13:/"),
                                  "manifest_sha256": sha(mblob), "n_files": len(man["files"]), "not written": man["not written"],
                                  "all keys == stored keys": all(v["keys == stored keys"] for v in man["files"].values())},
           "4b_flip_counts": {"cells": flips, "totals": dict(elig_total)}}
    path = f"/out/2026-10-14_p2_served_recheck/addendum_{stamp}.json"
    blob = json.dumps(rec, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)).encode()
    with open(path, "xb") as f:
        f.write(blob)
    out_vol.commit()
    return {"path": path, "sha256": sha(blob), "blob": blob.decode(), "folder": folder, "manifest_sha256": sha(mblob)}


# ------------------------------------------------------------------------------------------------ local entrypoint
def _git(root, c, p):
    return subprocess.run(["git", "show", f"{c}:{p}"], capture_output=True, check=True, cwd=root).stdout


def _safetensors(b: bytes) -> dict:
    import struct

    import numpy as np

    n = struct.unpack("<Q", b[:8])[0]
    h = json.loads(b[8:8 + n])
    dt = {"F32": np.float32, "F64": np.float64}
    return {k: np.frombuffer(b[8 + n + v["data_offsets"][0]:8 + n + v["data_offsets"][1]], dtype=dt[v["dtype"]]).reshape(v["shape"]).tolist()
            for k, v in h.items() if k != "__metadata__"}


@app.local_entrypoint()
def main():
    root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=root).stdout.strip()
    if subprocess.run(["git", "status", "--porcelain", "--", str(HERE)], capture_output=True, text=True, check=True, cwd=root).stdout.strip():
        raise SystemExit("commit this folder before launch")
    inputs, weights = {}, {}
    for p, rp in LINEAR.items():
        d = json.loads(_git(root, PIN_SIMH, rp))["directions"][p]
        wb = _git(root, PIN_SIMH, str(Path(rp).parent / d["file"]["path"]))
        if sha(wb) != d["file"]["sha256"]:
            raise SystemExit(f"{p}: weight sha256")
        weights[p] = {"kind": "linear", "layers": [int(d["layer"])], "tensors": {str(d["layer"]): _safetensors(wb)}}
        inputs[f"weights {p}"] = sha(wb)
    conv = json.loads(_git(root, PIN_SIMH, CONV))
    fl = sorted(int(L) for L in conv["layers"])
    weights["fibs"] = {"kind": "fibs", "layers": fl, "tensors": {}}
    for L in fl:
        wb = _git(root, PIN_SIMH, str(Path(CONV).parent / conv["layers"][str(L)]["output"]))
        if sha(wb) != conv["layers"][str(L)]["output_sha256"]:
            raise SystemExit("FIBS weight sha256")
        weights["fibs"]["tensors"][str(L)] = _safetensors(wb)
        inputs[f"weights fibs L{L}"] = sha(wb)
    rec_runs, rows_files = {}, {}
    for p, (c, path, key) in RECORDS.items():
        rb = _git(root, c, path)
        r = json.loads(rb)
        r = r if key is None else r["probes"][key]
        rec_runs[p] = {ro_: dict(r["run_scores"][ro_]) for ro_ in READOUTS}
        rows_files[p] = {ro_: (r["rows"][ro_]["path"], r["rows"][ro_]["sha256"]) for ro_ in ("reply_mean", "command")}
        inputs[f"record {p}"] = {"commit": c, "path": path, "sha256": sha(rb)}
    mbb = _git(root, MANIFEST_B[0], MANIFEST_B[1])
    vb = _git(root, V6[0], V6[1])
    mr = _git(root, MAIN_RECORD[0], MAIN_RECORD[1])
    if sha(mbb) != MANIFEST_B[2] or sha(vb) != V6[2] or sha(mr) != MAIN_RECORD[2]:
        raise SystemExit("pinned input sha256 differs")
    inputs |= {"manifest_b": MANIFEST_B[2], "v6": V6[2], "main record": MAIN_RECORD[2],
               "09 reading script (threshold mechanics only)": {"commit": READ09[0], "path": READ09[1], "sha256": sha(_git(root, *READ09))},
               "this script sha256": sha(Path(__file__).read_bytes())}
    v6r = json.loads(vb)["result"]
    es = set(v6r["universe_and_split"]["estimation_half"])
    v6 = {"c_est": [t for t in v6r["cohort"]["tasks"] if t in es],
          "pools": {f"{ro09}|{a}": v6r["results"][f"{ro09}|mechanical|alpha={a}"]["pools"] for ro09 in RO09 for a in ALPHAS},
          "eligible": {f"{ro09}|{a}": v6r["results"][f"{ro09}|mechanical|alpha={a}"]["eligibility_append13_4"]["eligible"] for ro09 in RO09 for a in ALPHAS}}
    m = json.loads(mr)
    g = m["claims"]["1_gate"]["mine"]
    wording = {"gate cells within 1e-6": f"{m['claims']['1_gate']['verified_cells']}/{m['claims']['1_gate']['cells']}",
               "gate cells not within 1e-6": {k: {"max |diff|": v["max_abs_diff"], "runs > 1e-6": v["n_gt_1e-6_or_missing"]}
                                              for k, v in g.items() if v["n_gt_1e-6_or_missing"]},
               "row-level cells with verified false, per layout": {L: sum(1 for v in m["claims"]["7_per_row"][L]["cells"].values()
                                                                       if not [x for kk, x in v.items() if kk.startswith("verified")][0]) for L in LAYOUTS},
               "row-level max |diff| per layout (cell)": {L: max(((k, v["max_abs_diff_f32"]) for k, v in m["claims"]["7_per_row"][L]["cells"].items()),
                                                                 key=lambda kv: kv[1]) for L in LAYOUTS}}
    payload = {"git": head, "inputs": inputs, "weights": weights, "rec_runs": rec_runs, "rows_files": rows_files,
               "manifest_b": json.loads(mbb), "v6": v6, "wording": wording}
    r = orchestrate.remote(payload)
    (HERE / "results" / Path(r["path"]).name).write_text(r["blob"])
    print(json.dumps({"path": r["path"], "sha256": r["sha256"], "folder": r["folder"], "manifest_sha256": r["manifest_sha256"]}))
