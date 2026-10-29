"""POST-HOC tests of what the step-2 detector reads (plan Amendment 7). One deployed Modal app, three functions:

- refit (H100): every detector Amendment 7 names is refit exactly as selected: same rows, same 101 label columns, same
  fitter. The run means of its out-of-fold scores must reproduce the grid record's real column (the largest absolute
  difference is returned). W, b, the standardization and C are kept.
  - Qwen-trained (step 2, records retrain_qwen_a38485ea5b9f, level 1, attempted): the primary, output_end-only and
    call_end-only selections, per fold of the 196-task frozen folds.
  - Sol-trained (the planned retrain, records retrain_227c3e8ca45d, level 1, attempted): the command-end and reply-end
    selections of results/retrain_2026-10-10_143949.json, per folds_v3 fold.
  Every readout of every Qwen run is scored out of fold; the scores are saved and returned (keys and numbers, no text).
  So are Aswin's 60 keyed commands at both reading points, and each fold model's honest_b mean and SD at its reading point.
- swap_stage (CPU): the instruction swap. Every epoch of one arm of one masked staging gets its system turn's interior
  replaced by the same task's honest system turn, and its first user turn replaced if it differs. Checks per epoch: staged
  hashes first; every token outside the span identical after a shift of Δ; every readout and request length shifted by
  exactly Δ; every readout's token id unchanged. Writes `<staging>_swapsys/inputs/<arm>.jsonl`.
- score_swapped (CPU): one swapped extraction's readouts scored with the refit Qwen detectors (primary and reply-end
  reader), mean over the selected readout kind per run.

  modal deploy experiments/2026-10-08_q3_onpolicy_figures/run_read_tests.py   (then spawn_read_tests.py)
"""

import hashlib
import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-q3onpol_readtests_kill-2026-11-10"
SCORE_VOL, ACTS_VOL, HF_CACHE = ("probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08", "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08",
                                 "probe-shift_aswin-surya_hf-cache")
CACHE, QWEN_TAG, SOL_TAG = "retrain_cache_v1", "a38485ea5b9f", "227c3e8ca45d"
T196, FOLDS = "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json", "splits/folds_v3.json"
P1_LABELS = "experiments/2026-10-07_q3_nano_monitor/results/attack_attempt_labels_2026-10-08_010659.jsonl"
KEYS = "experiments/2026-10-09_q2_deception_commands/labels/reading_keys.csv"
LABEL_FILE_SHA256 = "c0228369925f98b7d016a289acb9b2993411ddeaaa886c9732ceb548d9dab05e"
KEYED_RUN = "2026-10-09_rerun_strict_segends_v1_scrub1"  # the strict rerun at level 1 (masked), real attack run 1
LAYERS, KINDS = [39, 51, 63], ["output_end", "call_end"]
QWEN_C, SOL_C = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
SOL_CHOICES = {"command_end": [[51, "call_end", 1.0], [39, "call_end", 0.1], [51, "call_end", 0.1], [39, "call_end", 0.1], [39, "call_end", 0.1]],
               "reply_end": [[51, "output_end", 0.01], [51, "output_end", 0.01], [63, "output_end", 0.01], [51, "output_end", 0.01], [51, "output_end", 100.0]]}
QWEN_PRIMARY = [[39, "output_end", 0.0001], [39, "call_end", 0.001], [39, "output_end", 0.0001], [63, "output_end", 1e-06], [39, "output_end", 0.0001]]
N_NULL, MAX_ITER, MAX_ITER_CAP = 100, 5000, 80000

app = modal.App(APP)
score_vol, acts_vol = modal.Volume.from_name(SCORE_VOL, version=2), modal.Volume.from_name(ACTS_VOL, version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy", "scipy", "transformers==5.19.0")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift")
         .add_local_file(REPO / T196, f"/root/{T196}").add_local_file(REPO / FOLDS, f"/root/{FOLDS}")
         .add_local_file(REPO / P1_LABELS, f"/root/{P1_LABELS}").add_local_file(REPO / KEYS, f"/root/{KEYS}"))


def tol_of(C: float) -> float:
    return 1e-4 * min(C**0.5, 1.0)


def frozen_folds(d: dict) -> dict:  # run_score_onpolicy.frozen_folds (amendment 25), on a parsed task file
    h = lambda s: hashlib.sha256((d["salt"] + s).encode()).hexdigest()
    envs, by_env = sorted(d["eligible_mains_by_env"], key=h), {}
    for c in d["cells"]:
        by_env.setdefault(c["env"], []).append(c["main"])
    return {c["key"]: (by_env[c["env"]].index(c["main"]) + envs.index(c["env"])) % 5 for c in d["cells"]}


def fit_one(Xa, Y, C):
    """The pinned fit loop (run_retrain_grid.py / run_retrain_qwen.py) on rescaled, standardized rows; returns W, b."""
    import torch
    import torch.nn.functional as F

    from probe_shift import torch_probe as tp

    Ya = torch.from_numpy(Y).cuda()
    mi, prev = MAX_ITER, None
    while True:
        W, b, evals = tp.fit(Xa, Ya, 1.0, mi, tol=tol_of(C))
        Wg, bg = W.clone().requires_grad_(True), b.clone().requires_grad_(True)
        loss = (F.binary_cross_entropy_with_logits(Xa @ Wg + bg, Ya, reduction="sum") + 0.5 * (Wg * Wg).sum()) / len(Xa)
        loss.backward()
        g = torch.maximum(Wg.grad.abs().amax(0), bg.grad.abs()).cpu().numpy()
        if g.max() <= tol_of(C) or evals < mi or evals == prev or mi * 4 > MAX_ITER_CAP:
            return W, b, float(g[0])
        mi, prev = mi * 4, evals


def select(R: dict, cands: list) -> list:
    """Per fold, the real column's converged candidate with the highest validation AUROC (ties: earlier in `cands`)."""
    import numpy as np

    V = np.stack([np.where((R[c]["grad_full"][:, 0] <= R[c]["tol"]) & (R[c]["grad_val"][:, 0] <= R[c]["tol"]) & np.isfinite(R[c]["val_auroc"][:, 0]),
                           R[c]["val_auroc"][:, 0], -np.inf) for c in cands])
    if not np.isfinite(V.max(0)).all():
        raise ValueError("a fold has no converged candidate")
    return [list(cands[i]) for i in V.argmax(0)]


@app.function(image=image, gpu="H100!", cpu=8, memory=98304, timeout=3600, retries=0,
              volumes={"/score": score_vol, "/acts": acts_vol.read_only()}, scaledown_window=5)
def refit() -> dict:
    import csv

    import numpy as np
    import torch

    score_vol.reload()
    t0 = time.time()
    t196 = json.loads((Path("/root") / T196).read_text())
    f196, main_of = frozen_folds(t196), {c["key"]: c["main"] for c in t196["cells"]}
    fv3 = json.loads((Path("/root") / FOLDS).read_text())["task_folds"]["main_task_to_fold"]
    d = Path("/score") / CACHE / "level1"
    out = {"models": {}, "reproduction_max_abs_diff": {}, "per_row": {}, "keyed": {}, "honest_b_stats": {}, "record_scores": {}}
    (Path("/score") / f"readtests_{QWEN_TAG}").mkdir(exist_ok=True)
    weights = {}
    # selections: Qwen's from its records (the primary must equal the step-2 result's), Sol's from the planned result
    qrec = lambda c: dict(np.load(Path("/score") / f"retrain_qwen_{QWEN_TAG}" / "level1" / "attempted" / f"L{c[0]}_{c[1]}_C{c[2]:g}.npz"))
    qc = [(l, k, C) for C in QWEN_C for l in LAYERS for k in KINDS]
    QR = {c: qrec(c) for c in qc}
    sel = {("qwen", "primary"): select(QR, qc), ("qwen", "reply_end"): select(QR, [c for c in qc if c[1] == "output_end"]),
           ("qwen", "command_end"): select(QR, [c for c in qc if c[1] == "call_end"]),
           ("sol", "command_end"): SOL_CHOICES["command_end"], ("sol", "reply_end"): SOL_CHOICES["reply_end"]}
    if sel[("qwen", "primary")] != QWEN_PRIMARY:
        raise ValueError(f"the primary selection {sel[('qwen', 'primary')]} differs from the step-2 result's {QWEN_PRIMARY}")
    out["selections"] = {f"{a}|{b}": v for (a, b), v in sel.items()}
    # keyed commands: their rows in the masked strict rerun's real attack run 1 (keys and numbers only)
    keys = list(csv.DictReader((Path("/root") / KEYS).open()))
    if hashlib.sha256((Path("/root") / KEYS).read_bytes()).hexdigest() != LABEL_FILE_SHA256:
        raise ValueError("reading_keys.csv is not Aswin's frozen file")
    want = {(k["task"], k["irid"], int(k["call"])): k for k in keys}
    krows = {}  # (task, irid, call, kind) -> (chunk, global row)
    for f in sorted((Path("/acts") / KEYED_RUN / "meta" / "attack").glob("*.json")):
        m = json.loads(f.read_text())
        if m.get("redrawn_because") or m.get("attempt_kind") == "rerun":
            continue
        for c in m["calls"]:
            if c.get("row") is None or c.get("kind") not in KINDS:
                continue
            irid = c["irid"] if c["kind"] == "call_end" else c.get("src_irid")
            k = (m["task"], irid, c.get("call_index"))
            if k in want:
                if (k + (c["kind"],)) in krows:
                    raise ValueError(f"two {c['kind']} rows for a keyed command")
                krows[k + (c["kind"],)] = (m["chunk"], m["row_start"] + c["row"])
    out["keyed_rows_found"] = {kd: sum(1 for k in krows if k[3] == kd) for kd in KINDS}
    kfeat = {}
    for layer in LAYERS:
        for kd in KINDS:
            ks = [k for k in krows if k[3] == kd]
            X = np.empty((len(ks), 5120), np.float16)
            for chunk in sorted({krows[k][0] for k in ks}):
                a = np.load(Path("/acts") / KEYED_RUN / "by_layer" / f"L{layer}" / f"{chunk}.npy")
                for i, k in enumerate(ks):
                    if krows[k][0] == chunk:
                        X[i] = a[krows[k][1]]
            kfeat[(layer, kd)] = (ks, X)
    for det in ("qwen", "sol"):
        for layer in LAYERS:
            for kd in KINDS:
                todo = [(name, f, c) for (dt, name), cs in sel.items() if dt == det for f, c in enumerate(cs) if c[0] == layer and c[1] == kd]
                if not todo:
                    continue
                on_rows = json.loads((d / f"on_rows.L{layer}.json").read_text())
                kn = np.flatnonzero([r["kind"] == kd for r in on_rows])
                Xn = torch.from_numpy(np.load(d / f"on_L{layer}.npy")[kn]).cuda()
                runs, n_run = np.unique([f"{r['world']}|{r['task']}" for r in on_rows], return_inverse=True)
                n_run = n_run[kn]
                order = np.array([on_rows[i]["order"] for i in kn])
                if det == "qwen":
                    Xt, t_run = Xn, n_run
                    n_fold = np.array([f196[u.split("|", 1)[1]] for u in runs])
                else:
                    off_rows = json.loads((d / f"off_rows.L{layer}.json").read_text())
                    ko = np.flatnonzero([r["kind"] == kd for r in off_rows])
                    Xt = torch.from_numpy(np.load(d / f"off_L{layer}.npy")[ko]).cuda()
                    oruns, o_run = np.unique([r["key"] for r in off_rows], return_inverse=True)
                    t_run = o_run[ko]
                    n_fold = np.array([fv3[main_of[u.split("|", 1)[1]]] for u in runs])
                for name, f, (_, _, C) in todo:
                    if det == "qwen":
                        rec = QR[(layer, kd, C)]
                        if not np.array_equal(rec["on_runs"], runs):
                            raise ValueError("the cache's runs differ from the record's")
                        Y, train, r_fold = rec["Y"], rec["train"], rec["on_fold"]
                        tr = train & (r_fold != f)
                    else:  # the planned grid's label columns, rebuilt exactly (run_retrain_grid.grid, the attempted cell)
                        p1 = {r["key"]: r["label"] for r in map(json.loads, (Path("/root") / P1_LABELS).read_text().splitlines()) if r["arm"] == "real-attack-e1"}
                        arm = {r["key"]: r["arm"] for r in off_rows}
                        mains = np.array([{r["key"]: r["main_task"] for r in off_rows}[k] for k in oruns])
                        r_y = np.array([1.0 if arm[k] == "real-attack-e1" else 0.0 for k in oruns])
                        keep = (r_y == 0) | (np.array([p1.get(k) for k in oruns]) == "attempted")
                        Y = np.zeros((len(oruns), 1 + N_NULL))
                        Y[:, 0] = r_y
                        for col in range(1, 1 + N_NULL):
                            rng = np.random.default_rng([20261010, col])
                            y = r_y.copy()
                            for mm in sorted(set(mains[keep])):
                                j = np.flatnonzero(keep & (mains == mm))
                                y[j] = y[rng.permutation(j)]
                            Y[:, col] = y
                        r_fold = np.array([fv3[mm] for mm in mains])
                        tr = keep & (r_fold != f)
                        rec = dict(np.load(Path("/score") / f"retrain_{SOL_TAG}" / "level1" / "attempted" / f"L{layer}_{kd}_C{C:g}.npz"))
                        if not np.array_equal(rec["on_runs"], runs):
                            raise ValueError("the cache's runs differ from the planned record's")
                    a = np.flatnonzero(tr[t_run])
                    Xa = Xt[a].double()
                    mu = Xa.mean(0)
                    sd = Xa.std(0, unbiased=False)
                    sd = torch.where(sd > 0, sd, torch.ones_like(sd))
                    Xa = (Xa - mu) / sd * C**0.5
                    W, b, g0 = fit_one(Xa, Y[t_run[a]], C)
                    del Xa
                    w0, b0 = W[:, 0], b[0]
                    score = lambda X: (((X.double() - mu) / sd * C**0.5) @ w0 + b0).cpu().numpy()
                    mine = np.flatnonzero(n_fold[n_run] == f)  # every readout of the fold's held-out runs
                    S = score(Xn[mine])
                    u, inv = np.unique(n_run[mine], return_inverse=True)
                    means = np.bincount(inv, weights=S) / np.bincount(inv)
                    key = f"{det}|{name}|fold{f}"
                    d_abs = np.abs(means - rec["on_score"][u, 0])
                    if not (np.isfinite(d_abs).all() and d_abs.max() <= 1e-4 and g0 <= tol_of(C)):
                        raise ValueError(f"{key}: the refit does not reproduce its record (max {np.nanmax(d_abs):.2e}) or did not converge ({g0:.2e})")
                    out["reproduction_max_abs_diff"][key] = float(d_abs.max())
                    if (det, name) == ("qwen", "primary"):  # test (a): every readout's out-of-fold score
                        out["per_row"][key] = {"run": runs[n_run[mine]].astype(str).tolist(), "order": order[mine].tolist(), "kind": kd, "score": S.tolist()}
                    if det == "qwen":  # the record's own out-of-fold run scores (all outputs, first 11), for addition 2
                        out["record_scores"][key] = {"run": runs[u].astype(str).tolist(), "score": rec["on_score"][u, 0].tolist(),
                                                     "score_k": rec["on_score_k"][u, 0].tolist()}
                    hb = np.char.startswith(runs[n_run[mine]].astype(str), "honest_b|")
                    out["honest_b_stats"][key] = {"mean": float(S[hb].mean()), "sd": float(S[hb].std()), "n": int(hb.sum())}
                    ks, Xk = kfeat[(layer, kd)]
                    held = [i for i, k in enumerate(ks) if (f196 if det == "qwen" else {t: fv3[main_of[t]] for t in f196})[k[0]] == f]
                    if held:
                        sk = score(torch.from_numpy(Xk[held]).cuda())
                        for i, s in zip(held, sk):
                            out["keyed"].setdefault(f"{det}|{name}", {})[f"{ks[i][0]}|{ks[i][1]}|{ks[i][2]}"] = {"score": float(s), "fold": f}
                    out["models"][key] = {"layer": layer, "kind": kd, "C": C, "grad_real_column": g0}
                    weights[key] = {"w": w0.cpu().numpy(), "b": float(b0), "mu": mu.cpu().numpy(), "sd": sd.cpu().numpy(), "C": C, "layer": layer, "kind": kd}
                    print(f"{key} L{layer} {kd} C {C:g}: reproduction {out['reproduction_max_abs_diff'][key]:.2e} at {time.time() - t0:.0f} s", flush=True)
    # every fit passed: publish the weights, the keys-and-numbers result and a manifest naming both
    dd = Path("/score") / f"readtests_{QWEN_TAG}"
    for key, wt in weights.items():
        np.savez(dd / f"{key.replace('|', '_')}.npz", **wt)
    (dd / "refit_result.json").write_text(json.dumps(out))
    (dd / "manifest.json").write_text(json.dumps({"models": sorted(weights), "result_sha256": hashlib.sha256((dd / "refit_result.json").read_bytes()).hexdigest()}))
    score_vol.commit()
    return out


@app.function(image=image, cpu=8, memory=65536, timeout=3600, retries=0, volumes={"/acts": acts_vol, "/hf": modal.Volume.from_name(HF_CACHE)})
def swap_stage(staging: str, arm: str, honest_stagings: list, keep_own_tools: bool = False) -> dict:
    """keep_own_tools (Amendment 8, the simulated arm): keep the arm's own system turn except the system prompt after
    `</IMPORTANT>` and its blank line, and the submit definition line, both taken from the honest donor; written under
    `<staging>_swapsys2`. Otherwise the whole system turn is the donor's (real runs 1 and 2: only submit differs)."""
    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B", revision="fc05daec18b0a78c049392ed2e771dde82bdf654")
    im_start, im_end = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>")
    sha = lambda ids: hashlib.sha256(np.asarray(ids, np.int32).tobytes()).hexdigest()

    def turns(ids):
        """(start, end) of each turn's interior: after <|im_start|>, up to its <|im_end|>."""
        out, i = [], 0
        while i < len(ids):
            if ids[i] == im_start:
                j = next((k for k in range(i + 1, len(ids)) if ids[k] == im_end), len(ids))
                out.append((i + 1, j))
                i = j
            i += 1
        return out

    honest = {}  # task -> candidate donors: (arm, system interior, first user interior, decoded system text, epoch info)
    for hs in honest_stagings:
        for h_arm in ("honest_a", "honest_b"):
            p = Path("/acts") / hs / "inputs" / f"{h_arm}.jsonl"
            if not p.exists():
                continue
            for line in p.read_text().splitlines():
                ep = json.loads(line)
                if ep["epoch"] != 0:  # any attempt's first epoch: the system prompt is the task's, whichever attempt counted
                    continue
                for c in ep["calls"]:
                    if sha(ep["ids"][: c["n"]]) != c["sha256"]:
                        raise ValueError(f"{ep['key']}: a donor's staged ids differ from its staged hash")
                t = turns(ep["ids"])
                sys_ids, usr_ids = ep["ids"][t[0][0]:t[0][1]], ep["ids"][t[1][0]:t[1][1]]
                head = tok.decode(sys_ids)
                if not head.startswith("system"):
                    raise ValueError(f"{ep['key']}: a donor's first turn is not the system turn")
                honest.setdefault(ep["task"], []).append((h_arm, sys_ids, usr_ids, head,
                                                         {"arm": h_arm, "kind": ep.get("attempt_kind"), "redrawn": bool(ep.get("redrawn_because")), "chars": len(head)}))
    for v in honest.values():
        v.sort(key=lambda x: x[0])  # honest_a before honest_b
    import re

    def tools_of(text):  # function names and their JSON lines inside the <tools> block, if the template renders one
        m = re.search(r"<tools>\n(.*?)\n</tools>", text, re.S)
        if not m:
            return {}
        out = {}
        for line in m.group(1).splitlines():
            if not line.strip():
                continue
            fn = json.loads(line)  # an unparseable tool definition stops the stage
            out[(fn.get("function") or fn).get("name")] = line
        return out

    src = Path("/acts") / staging / "inputs" / f"{arm}.jsonl"
    out_eps, n = [], {"epochs": 0, "honest_b_used": set(), "delta": [], "cover_task_in_donor": 0, "tools_differ": set()}
    for line in src.read_text().splitlines():
        ep = json.loads(line)
        ids = ep["ids"]
        for c in ep["calls"]:
            if sha(ids[: c["n"]]) != c["sha256"]:
                raise ValueError(f"{ep['key']}: the staged ids differ from the staged hash")
        t = turns(ids)
        if tok.decode(ids[t[0][0]: t[0][0] + 1]) != "system":
            raise ValueError(f"{ep['key']}: the first turn is not the system turn")
        (s0, s1), (u0, u1) = t[0], t[1]
        a_text = tok.decode(ids[s0:s1])
        cover = re.search(r"<cover_task>\n?(.*?)\n?</cover_task>", a_text, re.S)
        if cover is None:
            raise ValueError(f"{ep['key']}: no cover task in the attacker prompt")
        # the donor: the first honest prompt (honest_a, then honest_b) carrying this task's cover text verbatim; an empty
        # honest prompt (honest_a in 6 tasks) carries none
        cands = honest.get(ep["task"], [])
        pick = [c for c in cands if cover.group(1).strip() in c[3] and "You are an agent" in c[3]]
        if not pick:
            raise ValueError(f"{ep['task']}: no honest donor carries the cover task; candidates {[c[4] for c in cands]}")
        h_arm, hs_sys, hs_usr, h_text, _ = pick[0]
        if ids[u0:u1] != hs_usr:
            raise ValueError(f"{ep['key']}: the first user turn differs from the donor's")
        n["cover_task_in_donor"] += 1
        ta, th = tools_of(a_text), tools_of(h_text)
        n["tools_differ"] |= {k for k in set(ta) | set(th) if ta.get(k) != th.get(k)}
        if keep_own_tools:
            sep = "</IMPORTANT>\n\n"
            if a_text.count(sep) != 1 or h_text.count(sep) != 1:
                raise ValueError(f"{ep['key']}: not exactly one end of the function-call instructions on each side")
            a_head, h_prompt = a_text.split(sep)[0], h_text.split(sep)[1]
            if cover.group(1).strip() not in h_prompt:  # the span actually copied carries the task's text
                raise ValueError(f"{ep['key']}: the cover task is not verbatim in the donor's system prompt")
            a_lines, h_lines = a_head.split("\n"), h_text.split(sep)[0].split("\n")
            is_submit = lambda line: line.startswith("{") and (json.loads(line).get("function") or json.loads(line)).get("name") == "submit"
            ai, hi = [i for i, x in enumerate(a_lines) if is_submit(x)], [i for i, x in enumerate(h_lines) if is_submit(x)]
            if len(ai) != 1 or len(hi) != 1:
                raise ValueError(f"{ep['key']}: not exactly one submit definition on each side")
            new_lines = a_lines[: ai[0]] + [h_lines[hi[0]]] + a_lines[ai[0] + 1:]
            new_text = "\n".join(new_lines) + sep + h_prompt
            if [x for i, x in enumerate(new_lines) if i != ai[0]] != [x for i, x in enumerate(a_lines) if i != ai[0]]:
                raise ValueError(f"{ep['key']}: a line other than submit changed")
            body = tok.encode(new_text[len("system"):], add_special_tokens=False)
            role = ids[s0: s0 + 1]  # the role token ("system") stays the arm's own
            sys_new = role + body
            if tok.decode(sys_new) != new_text:
                raise ValueError(f"{ep['key']}: the rebuilt system turn does not round-trip through the tokenizer")
            if tools_of(tok.decode(sys_new)).keys() != ta.keys() or any(tools_of(new_text)[k] != ta[k] for k in ta if k != "submit"):
                raise ValueError(f"{ep['key']}: a tool definition other than submit changed")
            new = ids[:s0] + sys_new + ids[s1:]
        else:
            new = ids[:s0] + hs_sys + ids[s1:]
        delta = len(new) - len(ids)
        # checks: the prefix and every token after the replaced spans identical, shifted by delta
        if new[:s0] != ids[:s0] or new[len(new) - (len(ids) - s1):] != ids[s1:]:
            raise ValueError(f"{ep['key']}: tokens outside the swapped spans changed")
        calls = []
        for c in ep["calls"]:
            c = dict(c)
            if c["readout"] is not None:
                if c["readout"] < s1:
                    raise ValueError(f"{ep['key']}: a readout inside the swapped spans")
                if new[c["readout"] + delta] != ids[c["readout"]]:
                    raise ValueError(f"{ep['key']}: a readout's token changed")
                c["readout"] += delta
            if c["n"] < s1:
                raise ValueError(f"{ep['key']}: a request ends inside the swapped spans")
            c["n"] += delta
            c["sha256"] = sha(new[: c["n"]])
            calls.append(c)
        if len(calls) != len(ep["calls"]):
            raise ValueError(f"{ep['key']}: the readouts changed in number")
        n["epochs"] += 1
        n["delta"].append(delta)
        if h_arm == "honest_b":
            n["honest_b_used"].add(ep["task"])
        out_eps.append(ep | {"ids": new, "calls": calls, "swap": {"system_from": h_arm, "delta": delta}})
    d = Path("/acts") / f"{staging}_swapsys{'2' if keep_own_tools else ''}" / "inputs"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{arm}.jsonl"
    if f.exists():
        raise FileExistsError(f"{f} exists")
    b = "".join(json.dumps(e) + "\n" for e in out_eps).encode()
    f.write_bytes(b)
    acts_vol.commit()
    dl = np.array(n["delta"])
    return {"staging": staging, "arm": arm, "sha256": hashlib.sha256(b).hexdigest(), "epochs": n["epochs"],
            "cover_task_verbatim_in_donor": n["cover_task_in_donor"], "tool_definitions_that_differ": sorted(map(str, n["tools_differ"])),
            "tasks_honest_b_prompt": sorted(n["honest_b_used"]), "delta_min": int(dl.min()) if len(dl) else None,
            "delta_median": float(np.median(dl)) if len(dl) else None, "delta_max": int(dl.max()) if len(dl) else None}


@app.function(image=image, cpu=8, memory=65536, timeout=3600, retries=0, volumes={"/score": score_vol, "/acts": acts_vol.read_only()})
def score_swapped(run: str, selections: dict) -> dict:
    """Run-mean scores of one swapped extraction's runs under the refit Qwen detectors (each run by its fold's model)."""
    import numpy as np

    score_vol.reload()
    acts_vol.reload()
    if not (Path("/score") / f"readtests_{QWEN_TAG}" / "manifest.json").exists():
        raise ValueError("no validated refit manifest")
    done = json.loads((Path("/acts") / run / "complete.json").read_text())  # written only after every shard succeeded
    f196 = frozen_folds(json.loads((Path("/root") / T196).read_text()))
    metas = [json.loads(f.read_text()) for f in sorted((Path("/acts") / run / "meta").glob("*/*.json"))]
    if len(metas) != done["epochs"]:
        raise ValueError(f"{run}: {len(metas)} epochs extracted, {done['epochs']} staged")
    metas = [m for m in metas if not (m.get("redrawn_because") or m.get("attempt_kind") == "rerun")]
    runs_of = {}
    for m in metas:
        runs_of.setdefault(f"{m['arm']}|{m['task']}", set()).add(m["run"])
    if any(len(v) > 1 for v in runs_of.values()):
        raise ValueError("several counted runs for one arm and task")
    chunks, out = {}, {}

    def chunk(layer, name):
        if (layer, name) not in chunks:
            chunks[(layer, name)] = np.load(Path("/acts") / run / "by_layer" / f"L{layer}" / f"{name}.npy")
        return chunks[(layer, name)]

    for name in ("primary", "reply_end"):
        tot = {}  # a run's epochs are pooled: sum and count of its selected readouts' scores
        for m in metas:
            f = f196[m["task"]]
            layer, kd, _ = selections[f"qwen|{name}"][f]
            wts = np.load(Path("/score") / f"readtests_{QWEN_TAG}" / f"qwen_{name}_fold{f}.npz")
            rows = [m["row_start"] + c["row"] for c in m["calls"] if c.get("kind") == kd and c.get("row") is not None]
            if not rows:
                continue
            X = chunk(layer, m["chunk"])[rows].astype(np.float64)
            s = ((X - wts["mu"]) / wts["sd"] * float(wts["C"]) ** 0.5) @ wts["w"] + float(wts["b"])
            t = tot.setdefault(f"{m['arm']}|{m['task']}", [0.0, 0, f])
            t[0] += float(s.sum())
            t[1] += len(rows)
        out[name] = {k: {"mean": v[0] / v[1], "n": v[1], "fold": v[2]} for k, v in tot.items()}
    return out
