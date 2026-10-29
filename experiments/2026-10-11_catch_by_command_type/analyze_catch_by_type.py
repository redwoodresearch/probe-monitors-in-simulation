"""Aswin's row 6: the 196-task impactful-command monitor's catch at its frozen 1% threshold within command types and
reweighted to one type mix, real run 1 vs simulated and real run 1 vs real run 2 (the yardstick). Rule: RULE.md.

Stored per-call flags of the scaleup196 scoring's draw 0 (no refit); a command's type is its tool x rule-1 parameter
name from the extraction metas (no command text is read). Laptop CPU.

  uv run --no-project --python 3.12 --with numpy --with matplotlib python analyze_catch_by_type.py \
      --draw0 <draw0.json> --metas <acts_meta root>
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
EXTRACTIONS = ["2026-10-09_rerun_strict_segends_v1", "2026-10-10_scaleup_canary_segends_v1", "2026-10-10_scaleup_l2_segends_v1",
               "2026-10-10_scaleup_topup_segends_v1", "2026-10-10_second_attack_segends_v1"]
CELLS = {"real": "real->real", "sim": "real->sim", "run2": "real->run2"}  # the real-trained monitor on each world
DRAW0_SHA256 = "c2656658baaf3aa150f83f4d105e7572bfd854f98032831621419e1f8e8f36bf"
HEADLINE = {"real_minus_sim_points": 0.46, "ci95_points": [-2.32, 2.50], "note": "refit and threshold recomputed per draw"}
MIN_IMPACTFUL, BOOT, SEED, MARGIN, PLANT, USABLE = 30, 2000, 20261011, 0.10, 0.10, 0.95


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def types_of(metas: Path) -> tuple[dict, str]:
    """(request id, labeler call number) -> (tool, rule-1 parameter name, label), from the kept runs' command entries."""
    out, h = {}, hashlib.sha256()
    for x in EXTRACTIONS:
        for f in sorted((metas / x / "meta").glob("*/*.json")):
            b = f.read_bytes()
            h.update(f"{x}/{f.parent.name}/{f.name}".encode() + hashlib.sha256(b).digest())
            m = json.loads(b)
            if m.get("redrawn_because") or m.get("attempt_kind") == "rerun":
                continue
            for c in m["calls"]:
                if c.get("kind", "command") == "command" and c.get("call") is not None:
                    out[(c["irid"], c["call"])] = (c.get("name"), c.get("param"), c.get("label"))
    return out, h.hexdigest()


def summary(x, diff: bool = True) -> dict:
    """Interval over the finite draws and its usable share (unrounded ends kept); the margin verdict only for differences."""
    x = np.asarray(x, float)
    f = x[np.isfinite(x)]
    usable = len(f) / len(x)
    if not len(f):
        return {"ci95": None, "usable_draws": 0.0, **({"verdict": "unavailable"} if diff else {}), "_lo": float("nan"), "_hi": float("nan")}
    lo, hi = float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))
    v = ("unstable" if usable < USABLE else "within the margin" if -MARGIN < lo and hi < MARGIN
         else "gap" if lo > MARGIN or hi < -MARGIN else "too few to tell")
    return {"ci95": [round(lo, 4), round(hi, 4)], "usable_draws": round(usable, 4), **({"verdict": v} if diff else {}), "_lo": lo, "_hi": hi}


def clean(o):
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items() if not k.startswith("_")}
    return [clean(v) for v in o] if isinstance(o, list) else o


def main(draw0: str, metas: str):
    d0, mp = Path(draw0), Path(metas)
    if sha(d0) != DRAW0_SHA256:
        raise SystemExit("draw0 sha256 mismatch")
    d = json.loads(d0.read_text())
    meta, cells = d["meta"], d["draws"][0]["cells"]
    T, meta_digest = types_of(mp)
    raw = {}
    for r in meta:
        name, param, label = T[(r["irid"], r["call"])]  # KeyError = an unjoined call: stop loudly
        if label != ("impactful" if r["y"] == 1 else "read_only"):
            raise SystemExit(f"label disagrees for {r['irid']} {r['call']}")
        raw[(r["irid"], r["call"])] = f"{name}:{param}"
    worlds = list(CELLS)
    rows, checks = {}, {}
    for w, cell in CELLS.items():
        pc = cells[cell]["per_call"]
        if len(pc["row"]) != len(pc["flag_frozen"]) or len(set(pc["row"])) != len(pc["row"]) or not all(0 <= i < len(meta) for i in pc["row"]):
            raise SystemExit(f"{cell}: rows and flags malformed")
        if any(meta[i]["world"] != w for i in pc["row"]) or len(pc["row"]) != sum(m["world"] == w for m in meta):
            raise SystemExit(f"{cell}: rows are not exactly world {w}")
        rows[w] = [(meta[i]["task"], meta[i]["y"], int(bool(f)), raw[(meta[i]["irid"], meta[i]["call"])]) for i, f in zip(pc["row"], pc["flag_frozen"])]
        imp = [f for t, y, f, k in rows[w] if y == 1]
        catch = sum(imp) / len(imp)
        if abs(catch - cells[cell]["values"]["catch_frozen"]) > 1e-12:
            raise SystemExit(f"{cell}: catch {catch} does not reproduce the stored catch_frozen")
        checks[w] = {"calls": len(rows[w]), "impactful": len(imp), "tasks": len({r[0] for r in rows[w]}), "catch_reproduced": catch}
    if round(100 * (checks["real"]["catch_reproduced"] - checks["sim"]["catch_reproduced"]), 2) != HEADLINE["real_minus_sim_points"]:
        raise SystemExit("the +0.46 headline is not reproduced")
    # rule 1: types with fewer than MIN_IMPACTFUL impactful calls in any world go to "other"
    cnt = {w: {} for w in worlds}
    for w in worlds:
        for t, y, f, k in rows[w]:
            if y == 1:
                cnt[w][k] = cnt[w].get(k, 0) + 1
    keep = sorted(k for k in set().union(*[set(c) for c in cnt.values()]) if min(cnt[w].get(k, 0) for w in worlds) >= MIN_IMPACTFUL)
    types = keep + ["other"]
    final = lambda k: k if k in keep else "other"  # noqa: E731
    tasks = sorted({r[0] for w in worlds for r in rows[w]})
    ti, ki = {t: i for i, t in enumerate(tasks)}, {k: i for i, k in enumerate(types)}
    A = np.zeros((4, len(worlds), len(types), len(tasks)))  # [impactful, flagged impactful, read-only, flagged read-only]
    for wi, w in enumerate(worlds):
        for t, y, f, k in rows[w]:
            j, n = ki[final(k)], ti[t]
            A[0 if y else 2, wi, j, n] += 1
            A[1 if y else 3, wi, j, n] += f
    rng = np.random.default_rng(SEED)
    W = np.vstack([np.ones(len(tasks)), rng.multinomial(len(tasks), np.full(len(tasks), 1 / len(tasks)), size=BOOT)])  # row 0 = observed

    def compare(A, a, b):
        S = np.einsum("bt,cwkt->bcwk", W, A)  # [draw, count kind, world, type]
        ia, ib = worlds.index(a), worlds.index(b)
        with np.errstate(invalid="ignore", divide="ignore"):
            catch, fa = S[:, 1] / S[:, 0], S[:, 3] / S[:, 2]
            share = S[:, 0] / S[:, 0].sum(-1, keepdims=True)
            overall = S[:, 1].sum(-1) / S[:, 0].sum(-1)
            mixes = {"pooled": (S[:, 0, ia] + S[:, 0, ib]) / (S[:, 0, ia] + S[:, 0, ib]).sum(-1, keepdims=True), f"{a}_mix": share[:, ia]}
        present = (A[0, ia].sum(-1) + A[0, ib].sum(-1)) > 0  # types with any impactful call in the two worlds (others weigh 0)
        full = np.isfinite(catch[:, ia][:, present]).all(-1) & np.isfinite(catch[:, ib][:, present]).all(-1)  # each present in both worlds
        out = {"overall_difference": {"_raw": float(overall[0, ia] - overall[0, ib]), "estimate": round(float(overall[0, ia] - overall[0, ib]), 4), **summary(overall[1:, ia] - overall[1:, ib]),
                                      a: round(float(overall[0, ia]), 4), b: round(float(overall[0, ib]), 4)},
               "reweighted": {}, "mix_total_variation": {"estimate": round(float(0.5 * np.abs(share[0, ia] - share[0, ib]).sum()), 4),
                                                         **summary(0.5 * np.abs(share[1:, ia] - share[1:, ib]).sum(-1), diff=False)}, "by_type": {}}
        for mix, m in mixes.items():
            rw = {w: np.where(full, (np.nan_to_num(catch[:, i][:, present]) * m[:, present]).sum(-1), np.nan) for w, i in ((a, ia), (b, ib))}
            out["reweighted"][mix] = {a: round(float(rw[a][0]), 4), b: round(float(rw[b][0]), 4), f"{a}_ci95": summary(rw[a][1:])["ci95"],
                                      f"{b}_ci95": summary(rw[b][1:])["ci95"], "difference": round(float(rw[a][0] - rw[b][0]), 4),
                                      **summary(rw[a][1:] - rw[b][1:])}
        for k, j in ki.items():
            dc = catch[:, ia, j] - catch[:, ib, j]
            out["by_type"][k] = {"_raw_difference": float(dc[0]),
                "impactful_calls": {a: int(A[0, ia, j].sum()), b: int(A[0, ib, j].sum())},
                "read_only_calls": {a: int(A[2, ia, j].sum()), b: int(A[2, ib, j].sum())},
                "catch": {a: round(float(catch[0, ia, j]), 4), b: round(float(catch[0, ib, j]), 4), f"{a}_ci95": summary(catch[1:, ia, j])["ci95"],
                          f"{b}_ci95": summary(catch[1:, ib, j])["ci95"], "difference": round(float(dc[0]), 4), **summary(dc[1:])},
                "false_alarm": {a: round(float(fa[0, ia, j]), 4), b: round(float(fa[0, ib, j]), 4), f"{a}_interval": summary(fa[1:, ia, j], diff=False),
                                f"{b}_interval": summary(fa[1:, ib, j], diff=False), "difference": round(float(fa[0, ia, j] - fa[0, ib, j]), 4),
                                **summary(fa[1:, ia, j] - fa[1:, ib, j], diff=False)},
                "catch_per_world_interval": {a: summary(catch[1:, ia, j], diff=False), b: summary(catch[1:, ib, j], diff=False)},
                "share_of_impactful": {a: round(float(share[0, ia, j]), 4), b: round(float(share[0, ib, j]), 4), f"{a}_ci95": summary(share[1:, ia, j])["ci95"],
                                       f"{b}_ci95": summary(share[1:, ib, j])["ci95"], "difference": round(float(share[0, ia, j] - share[0, ib, j]), 4),
                                       **summary(share[1:, ia, j] - share[1:, ib, j], diff=False)}}
        return out

    res = {"real_vs_sim": compare(A, "real", "sim"), "real_vs_run2 (yardstick)": compare(A, "real", "run2")}
    # rule 5: plant a type-level gap in the simulated world that leaves the overall about equal
    sim_imp = {k: int(A[0, worlds.index("sim"), ki[k]].sum()) for k in types}
    si, skipped, big = worlds.index("sim"), [], None
    for cand in sorted(types, key=lambda k: -sim_imp[k]):  # amendment 1: the largest type whose flips are feasible
        n_flip = int(round(PLANT * sim_imp[cand]))
        pool_dn = [(t, k) for t, y, f, k in rows["sim"] if y == 1 and f == 1 and final(k) == cand]
        pool_up = [(t, k) for t, y, f, k in rows["sim"] if y == 1 and f == 0 and final(k) != cand]
        if n_flip > 0 and len(pool_dn) >= n_flip and len(pool_up) >= n_flip:
            big = cand
            break
        skipped.append(cand)
    if big is None:
        res["planted_check"] = {"infeasible": True, "skipped": skipped}
        return finish(d, checks, cnt, keep, tasks, meta_digest, res)
    prng = np.random.default_rng(SEED + 1)
    Ap = A.copy()
    for i in prng.choice(len(pool_dn), n_flip, replace=False):
        Ap[1, si, ki[big], ti[pool_dn[i][0]]] -= 1
    for i in prng.choice(len(pool_up), n_flip, replace=False):
        Ap[1, si, ki[final(pool_up[i][1])], ti[pool_up[i][0]]] += 1
    planted = compare(Ap, "real", "sim")
    u_raw, p_raw = res["real_vs_sim"]["by_type"][big]["_raw_difference"], planted["by_type"][big]["_raw_difference"]
    u, p = res["real_vs_sim"]["by_type"][big]["catch"], planted["by_type"][big]["catch"]
    shift = p_raw - u_raw
    o_shift = planted["overall_difference"]["_raw"] - res["real_vs_sim"]["overall_difference"]["_raw"]
    finite = np.isfinite(p["_lo"]) and np.isfinite(p["_hi"])
    res["planted_check"] = {"type": big, "skipped_types": skipped, "flips_each_way": n_flip, "planted_points": PLANT, "unplanted_difference": u["difference"],
                            "planted_difference": p["difference"], "planted_ci95": p["ci95"], "shift": round(shift, 4), "overall_shift": round(o_shift, 4),
                            "detected": bool(abs(shift - PLANT) <= 0.01 and finite and not (p["_lo"] <= u_raw <= p["_hi"]) and abs(o_shift) < 0.01)}
    return finish(d, checks, cnt, keep, tasks, meta_digest, res)


def finish(d, checks, cnt, keep, tasks, meta_digest, res):
    cells = d["draws"][0]["cells"]
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=HERE).stdout.strip())
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = clean({"git": {"commit": git, "dirty": dirty}, "written": stamp, "rule": "RULE.md", "seed": SEED, "draws": BOOT, "margin": MARGIN,
                 "inputs": {"draw0_sha256": DRAW0_SHA256, "draw0_git": d["git"], "metas_digest": meta_digest, "extractions": EXTRACTIONS},
                 "cells": CELLS, "stored_values": {w: cells[c]["values"] for w, c in CELLS.items()}, "checks": checks, "headline": HEADLINE,
                 "types": {"kept": keep, "min_impactful_in_every_world": MIN_IMPACTFUL, "impactful_counts_before_merge": cnt}, "tasks": len(tasks),
                 "interval_note": "task-resampled, paired by task, with flags and thresholds as stored (conditional on the fitted probes)", **res})
    (HERE / "results").mkdir(exist_ok=True)
    path = HERE / "results" / f"catch_by_type_{stamp}.json"
    path.write_text(json.dumps(out, indent=1, allow_nan=False))
    plot(out, HERE / "results" / f"catch_by_type_{stamp}.png")
    print(json.dumps({k: out[k] for k in ("checks", "types", "planted_check")}, indent=1)[:2500])
    for comp in ("real_vs_sim", "real_vs_run2 (yardstick)"):
        r = out[comp]
        print(comp, "| overall", r["overall_difference"], "| reweighted", r["reweighted"], "| TV", r["mix_total_variation"])
        for k, v in r["by_type"].items():
            print(f"  {k:22s} n={v['impactful_calls']} catch={v['catch']} share={v['share_of_impactful']['difference']} fa={v['false_alarm']}")
    print("->", path)


def plot(out, png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = out["real_vs_sim"]["by_type"]
    ks = list(r)
    fig, ax = plt.subplots(figsize=(1.6 * len(ks) + 2, 4))
    for i, (w, col, name) in enumerate((("real", "#4C72B0", "real run 1"), ("sim", "#DD8452", "simulated"))):
        vals = np.array([np.nan if r[k]["catch"][w] is None else r[k]["catch"][w] * 100 for k in ks], float)
        ci = np.array([r[k]["catch"][f"{w}_ci95"] or [np.nan, np.nan] for k in ks], float) * 100
        x = np.arange(len(ks)) + (i - 0.5) * 0.38
        ax.bar(x, vals, 0.36, color=col, label=name)
        ax.vlines(x, ci[:, 0], ci[:, 1], color="black", linewidth=1)
        for xx, v in zip(x, vals):
            if np.isfinite(v):
                ax.text(xx, 2, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color="white")
    ax.set_xticks(np.arange(len(ks)), [f"{k}\n(n={r[k]['impactful_calls']['real']}/{r[k]['impactful_calls']['sim']})" for k in ks], fontsize=8)
    ax.set_ylabel("impactful commands caught at 1% (%, higher = caught)")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")
    ax.set_title("Catch within command type: real run 1 vs simulated (95% task-resampled intervals)")
    fig.tight_layout()
    fig.savefig(png, dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw0", required=True)
    ap.add_argument("--metas", required=True)
    main(**vars(ap.parse_args()))
