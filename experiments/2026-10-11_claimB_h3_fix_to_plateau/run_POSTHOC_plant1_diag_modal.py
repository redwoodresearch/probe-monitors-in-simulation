"""POST HOC (after H3's result, h3_small_2026-10-11_155930.json: planted check 1 at 0.794 / 0.829 vs the 0.833 floor;
Aswin's option (ii) relayed by 0d, the free diagnosis; under the standing under-$5 rule): why did plant 1 drop?
One Modal CPU container, keys and numbers only (no transcript or command text leaves it). For all plant-1 pairs on
the test environments (the staged copy and its fixed-sim twin, step B's level-1 inputs), with H3's own code
(run_h3_small_modal.py at 23b5c37a, imported unchanged; its cells.json):

1. Re-insertion: does the staged copy equal its twin plus the planted line at the STAGED_A manifest site (msg,
   after_line), differing in that one tool message only? Was the site message itself changed by the evening? Does the
   re-made copy carry the line after the same content line as the staged copy (by hash)? Does evening the staged copy
   directly give the re-made copy? Is the twin's and the copy's changed status (cells.json) consistent? Is the site's
   output (step, action) inside the copy's rows H3 used (its evened read, or the pack rows if unchanged) and outside
   the range exclusion?
2. Probe vs data: B-dev's frozen L63 probes (the meaning control's refit, C 0.001, refit here on CPU, float64) score the
   pairs on the UNEVENED pack rows and on the EVENED rows H3 assembled; the evened scores are checked against H3's
   frozen_run_scores.json (it must agree within 1e-4, or the result says so). Plant-1 AUROC under the frozen probe on
   each version separates a change in the data from a change in the refit probe.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy \\
      modal run experiments/2026-10-11_claimB_h3_fix_to_plateau/run_POSTHOC_plant1_diag_modal.py
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE) if modal.is_local() else "/root")
import run_h3_small_modal as R  # noqa: E402

H3_RUNNER_SHA = "04c3953b139ec5a32cc2e822ec8d4310aa3583384950a0010ae15bf0d048b77e"  # run_h3_small_modal.py at 23b5c37a (the run)
PL1 = "sim-attack-fixed-plant1"
app = modal.App("probe-shift_aswin-surya_sub-claimb-h3p1diag_kill-2026-11-11")
image = R.image.add_local_file(HERE / "run_h3_small_modal.py", "/root/run_h3_small_modal.py")


def sha16(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


@app.function(image=image, cpu=8, memory=65536, volumes={"/acts": R.B.P.vol, "/hf": R.B.P.hf}, timeout=45 * 60, retries=0)
def diag(env_to_fold: dict) -> dict:
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    from probe_shift import torch_probe as tp
    from probe_shift.models import QWEN3_32B
    from probe_shift.q1_eval import step_table
    from probe_shift.reader import segment_end_positions

    if R.SCRIPT_SHA != H3_RUNNER_SHA:
        raise ValueError("H3's runner here differs from the one that ran")
    R.check_b()
    torch.set_num_threads(8)
    M, B = R.M, R.B
    plan = R.load_plan()
    changed = {tuple(x) for x in plan["changed"]}
    runs = R.staged()
    man = json.loads((Path("/acts") / R.STAGED_A / "manifest.json").read_text())["plant1"]
    sites, line = {x["key"]: x for x in man["sites"]}, man["line"]
    tok = AutoTokenizer.from_pretrained(QWEN3_32B["name"], revision=QWEN3_32B["revision"])
    pruns = json.loads((Path("/acts") / M.PACK / "runs.L63.json").read_text())
    key_of = {(m["cell"], m["arm"]): i for i, m in enumerate(pruns)}
    t = step_table(pruns, "output_end")
    rows_of = {}
    for k in range(len(t["run"])):
        rows_of.setdefault(int(t["run"][k]), {})[(int(t["step"][k]), int(t["action"][k]))] = k
    XP = np.load(Path("/acts") / M.PACK / "L63_output_end.npy")

    def rows(cell, arm):
        """(L63 rows, kept (step, action) set, range set, source) exactly as H3's refit assembled them: a validated evened
        read if one exists (H3's files()), else the pack rows."""
        f = Path("/acts") / R.OUT / "runs" / f"{sha16(f'{cell}|{arm}|evened')}.npz"
        if f.exists():
            z = np.load(f)
            m = json.loads(str(z["meta"]))
            if (m["version"], m["cell"], m["arm"], m["script_sha256"], m.get("deps_sha256"), m["model"], m["revision"]) != (
                    "evened", cell, arm, B.SCRIPT_SHA, B.DEPS_SHA, QWEN3_32B["name"], QWEN3_32B["revision"]):
                raise ValueError(f"{f.name}: evened read from another version, run or model")
            rng = {tuple(x) for x in m["range"]}
            keep = [k for k, s in enumerate(m["steps"]) if tuple(s) not in rng]
            return np.asarray(z["L63"][keep], np.float32), {tuple(m["steps"][k]) for k in keep}, rng, "evened"
        pr = rows_of.get(key_of[(cell, arm)], {})
        return np.asarray(XP[sorted(pr.values())], np.float32), set(pr), set(), "pack"

    cells = sorted(c for (c, a) in runs if a == PL1 and runs[(c, a)]["environment"] in R.TEST)
    recs = []
    for cell in cells:
        twin, copy = runs[(cell, M.SIM)], runs[(cell, PL1)]
        site = sites[twin["key"]]
        tm, cm, i, al = twin["messages"], copy["messages"], site["msg"], site["after_line"]
        diff = [k for k in range(max(len(tm), len(cm))) if k >= len(tm) or k >= len(cm) or tm[k] != cm[k]]
        lt = tm[i]["content"].split("\n")
        staged_ok = len(tm) == len(cm) and diff == [i] and cm[i]["content"] == "\n".join(lt[:al] + [line] + lt[al:])
        et, _, rng_t = B.evened(tm)
        ec, _ = R.evened_version(runs, cell, PL1)
        ed, _, _ = B.evened(cm)
        le, lc, ld = et[i]["content"].split("\n"), ec[i]["content"].split("\n"), ed[i]["content"].split("\n")
        h = lambda x: hashlib.sha256(x.encode()).hexdigest()  # noqa: E731
        lcs = cm[i]["content"].split("\n")
        pos_s, pos_r = [k for k, x in enumerate(lcs) if x == line], [k for k, x in enumerate(lc) if x == line]
        anchor_staged = h(lcs[pos_s[0] - 1]) if len(pos_s) == 1 and pos_s[0] > 0 else None
        anchor_remade = h(lc[pos_r[0] - 1]) if len(pos_r) == 1 and pos_r[0] > 0 else None
        tools = [k for k, m in enumerate(ec) if m["role"] == "tool"]
        seg = segment_end_positions(tok, ec, M.MAX_TOKENS)
        o = seg["outputs"][tools.index(i)]
        sa = (o["step"], o["action"])
        _, c_kept, c_rng, c_src = rows(cell, PL1)
        _, t_kept, t_rng, t_src = rows(cell, M.SIM)
        recs.append({"cell": cell, "environment": twin["environment"], "main_task": cell.split(":")[0],
                     "staged_insertion_matches_manifest": bool(staged_ok), "staged_diff_messages": len(diff),
                     "twin_changed_by_evening": bool(et != tm or rng_t), "twin_changed_in_plan": (cell, M.SIM) in changed,
                     "copy_changed_in_plan": (cell, PL1) in changed, "site_message_changed_by_evening": et[i]["content"] != tm[i]["content"],
                     "anchor_line_same": anchor_staged is not None and anchor_staged == anchor_remade,
                     "line_index_staged": pos_s, "line_index_remade_equals_after_line": pos_r == [al], "remade_equals_direct_evening": ed == ec,
                     "line_positions_remade": [k for k, x in enumerate(lc) if x == line], "line_positions_direct": [k for k, x in enumerate(ld) if x == line],
                     "line_count_in_evened_twin_site": sum(1 for x in le if x == line), "site_step_action": list(sa),
                     "site_in_copy_rows": sa in c_kept, "site_in_copy_range": sa in c_rng, "site_in_twin_rows": sa in t_kept,
                     "copy_rows_source": c_src, "twin_rows_source": t_src, "copy_kept": len(c_kept), "twin_kept": len(t_kept),
                     "kept_sets_equal": c_kept == t_kept})
    # frozen L63 probes (CPU refit of B-dev's refit), scored on unevened pack rows and on H3's evened rows
    arm_of = np.array([pruns[r]["arm"] for r in t["run"]])
    efold = np.array([env_to_fold[pruns[r]["environment"]] for r in t["run"]])
    fz = json.loads((Path("/acts") / R.OUT / "refit" / "frozen_run_scores.json").read_text())
    if fz["script_sha256"] != R.SCRIPT_SHA:
        raise ValueError("frozen_run_scores.json from another H3 version")
    frozen_h3 = fz["scores"]
    scores, fits = {}, {}
    for e in R.TEST:
        tr = np.isin(arm_of, [M.REAL, M.SIM]) & (efold != env_to_fold[e])
        Xa = torch.from_numpy(np.asarray(XP[tr], np.float32)).double()
        Ya = torch.from_numpy((arm_of[tr] == M.SIM).astype(float)[:, None]).double()
        mean, std = Xa.mean(0), Xa.std(0, unbiased=False)
        std = std.where(std > 0, std.new_ones(()))
        Xa = (Xa - mean) / std * M.C**0.5
        tol, mi, prev = 1e-4 * min(M.C**0.5, 1.0), 5000, None
        while True:
            W, b, evals = tp.fit(Xa, Ya, 1.0, mi, tol=tol)
            if evals < mi or evals == prev or mi * 4 > 80000:
                break
            mi, prev = mi * 4, evals
        Wg, bg = W.clone().requires_grad_(True), b.clone().requires_grad_(True)
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(Xa @ Wg + bg, Ya, reduction="sum") + 0.5 * (Wg * Wg).sum()) / len(Xa)
        loss.backward()
        grad = float(max(Wg.grad.abs().max(), bg.grad.abs().max()))
        fits[e] = {"evals": int(evals), "grad_max": grad, "converged": grad <= tol}
        del Xa, Ya, Wg, bg

        def s(X):
            if len(X) == 0:
                raise ValueError("a run with no rows")
            return float(((((torch.from_numpy(X).double() - mean) / std * M.C**0.5) @ W + b)[:, 0]).mean())
        for r in recs:
            if r["environment"] != e:
                continue
            for arm, tag in ((PL1, "copy"), (M.SIM, "twin")):
                pr = rows_of.get(key_of[(r["cell"], arm)], {})
                X_ev, _, _, _ = rows(r["cell"], arm)
                ev = s(X_ev)
                scores[(r["cell"], tag)] = {"unevened": s(np.asarray(XP[sorted(pr.values())], np.float32)), "evened": ev,
                                            "h3_frozen_evened": frozen_h3.get(str(key_of[(r["cell"], arm)]))}
    for r in recs:
        r["frozen"] = {tag: scores[(r["cell"], tag)] for tag in ("copy", "twin")}
    return {"pairs": recs, "frozen_fit_evals": fits, "line_sha256": hashlib.sha256(line.encode()).hexdigest()}


@app.local_entrypoint()
def main():
    if subprocess.run(["git", "status", "--porcelain", str(Path(__file__))], capture_output=True, text=True, check=True).stdout:
        raise SystemExit(f"commit {Path(__file__).name} first")
    folds = json.loads((HERE.parents[1] / "splits" / "folds_v2.json").read_text())["environment_folds"]["environment_to_fold"]
    res = diag.remote(folds)
    P = res["pairs"]
    n = lambda f: sum(1 for r in P if f(r))  # noqa: E731
    summary = {"pairs": len(P), "staged_insertion_matches_manifest": n(lambda r: r["staged_insertion_matches_manifest"]),
               "twin_changed_by_evening": n(lambda r: r["twin_changed_by_evening"]), "copy_changed_in_plan": n(lambda r: r["copy_changed_in_plan"]),
               "changed_status_consistent": n(lambda r: r["twin_changed_in_plan"] == r["copy_changed_in_plan"] == r["twin_changed_by_evening"]),
               "site_message_changed_by_evening": n(lambda r: r["site_message_changed_by_evening"]), "anchor_line_same": n(lambda r: r["anchor_line_same"]),
               "remade_equals_direct_evening": n(lambda r: r["remade_equals_direct_evening"]),
               "line_index_remade_equals_after_line": n(lambda r: r["line_index_remade_equals_after_line"]),
               "copy_source_matches_plan": n(lambda r: (r["copy_rows_source"] == "evened") == r["copy_changed_in_plan"]),
               "twin_source_matches_plan": n(lambda r: (r["twin_rows_source"] == "evened") == r["twin_changed_in_plan"]),
               "frozen_fits_converged": sum(1 for v in res["frozen_fit_evals"].values() if v["converged"]),
               "line_present_once_in_remade": n(lambda r: len(r["line_positions_remade"]) == 1),
               "site_in_copy_rows": n(lambda r: r["site_in_copy_rows"]), "site_in_copy_range": n(lambda r: r["site_in_copy_range"]),
               "copy_and_twin_same_source": n(lambda r: r["copy_rows_source"] == r["twin_rows_source"]),
               "kept_sets_equal": n(lambda r: r["kept_sets_equal"]),
               "frozen_evened_matches_h3_within_1e-4": n(lambda r: all(v["h3_frozen_evened"] is not None and abs(v["evened"] - v["h3_frozen_evened"]) <= 1e-4
                                                                        for v in r["frozen"].values()))}
    import numpy as np

    def auroc(pos, neg):
        x = np.concatenate([pos, neg])
        if not np.isfinite(x).all():
            raise ValueError("non-finite scores")
        r = np.argsort(np.argsort(x, kind="stable"), kind="stable") + 1.0
        for v in np.unique(x):  # average ranks for ties
            r[x == v] = r[x == v].mean()
        return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    au = {}
    for name, sel in (("all", lambda r: True), ("copy_changed", lambda r: r["copy_changed_in_plan"]), ("copy_unchanged", lambda r: not r["copy_changed_in_plan"])):
        Q = [r for r in P if sel(r)]
        if len(Q) >= 2:
            au[name] = {"pairs": len(Q)} | {v: auroc(np.array([r["frozen"]["copy"][v] for r in Q]), np.array([r["frozen"]["twin"][v] for r in Q]))
                                            for v in ("unevened", "evened")} | {
                f"mean_shift_{v}": float(np.mean([r["frozen"]["copy"][v] - r["frozen"]["twin"][v] for r in Q])) for v in ("unevened", "evened")} | {
                f"pairs_shift_positive_{v}": sum(1 for r in Q if r["frozen"]["copy"][v] > r["frozen"]["twin"][v]) for v in ("unevened", "evened")}
    summary["frozen_probe_plant1"] = au
    res |= {"label": "POST HOC: H3 plant-1 diagnosis (re-insertion checks; frozen probe on unevened vs evened rows); keys and numbers only",
            "summary": summary, "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"plant1_diag_POSTHOC_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(res, indent=1))
    print(json.dumps(summary, indent=1))
    print("->", dst)
