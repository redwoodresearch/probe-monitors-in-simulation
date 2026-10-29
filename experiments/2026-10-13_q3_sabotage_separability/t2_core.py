"""fit_dual, wauc_draws, aucs and prep_qwen35, copied byte-for-byte from run_t2_realism_v2.py @ 599fa553
(aswin/q3-sabotage-t2-realism); prep_qwen35 is a plain function here (its Modal decorator dropped). Shipped into the Modal image
next to run_sab_separability.py, which imports these inside its functions."""

import json
from collections import Counter
from pathlib import Path

AVG = "2026-10-13_sabavg_"


def fit_dual(K, y, w, Cs, iters=100, tol=1e-9):
    """run_grid.py's fit_dual (aswin/q3-sabotage-scaleup @ d4be80f6), with tensors created on K's device."""
    import torch
    import torch.nn.functional as F

    L, n, _ = K.shape
    C = torch.tensor(Cs, dtype=K.dtype, device=K.device)[None, :, None]
    Kb, eye = K[:, None], torch.eye(n, dtype=K.dtype, device=K.device)
    a = torch.zeros(L, len(Cs), n, dtype=K.dtype, device=K.device)

    def obj(a):
        f = (Kb @ a[..., None])[..., 0]
        return (C * w * (F.softplus(f) - y * f)).sum(-1) + 0.5 * (a * f).sum(-1)

    for _ in range(iters):
        f = (Kb @ a[..., None])[..., 0]
        p = torch.sigmoid(f)
        r = a - C * w * (y - p)
        res = r.abs().amax(-1)
        if float(res.max()) < tol:
            return a
        delta = torch.linalg.solve(eye + (C * w * p * (1 - p))[..., :, None] * Kb, r[..., None])[..., 0]
        step, j0 = torch.ones(L, len(Cs), 1, dtype=K.dtype, device=K.device), obj(a)
        for _ in range(40):
            worse = obj(a - step * delta) > j0 + 1e-12 * j0.abs()
            if not worse.any():
                break
            step = torch.where(worse[..., None], step / 2, step)
        a = a - step * delta
    raise RuntimeError(f"dual Newton did not converge: max residual {float(res.max()):.3g}")


def wauc_draws(s, y, W):
    """AUROC of scores s (n,) for labels y under each row of task-multiplicity weights W (d, n); ties count half. (d,)"""
    import numpy as np

    u, inv = np.unique(s, return_inverse=True)
    P = np.zeros((W.shape[0], len(u)))
    N = np.zeros((W.shape[0], len(u)))
    np.add.at(P.T, inv[y == 1], W[:, y == 1].T)
    np.add.at(N.T, inv[y == 0], W[:, y == 0].T)
    below = np.cumsum(N, 1) - N
    num = (P * (below + 0.5 * N)).sum(1)
    pairs = P.sum(1) * N.sum(1)
    return num, pairs


def aucs(scores, y, fold, tw):
    """Per-fold-combined (pair-weighted) and pooled AUROC per draw: tw (d, n) row weights. Returns (combined, pooled, per-fold full)."""
    import numpy as np

    num, den, per = 0, 0, {}
    for f in sorted(set(fold.tolist())):
        m = fold == f
        if len(set(y[m].tolist())) < 2:
            continue
        a, b = wauc_draws(scores[m], y[m], tw[:, m])
        num, den = num + a, den + b
        per[int(f)] = float(a[0] / b[0])
    pa, pb = wauc_draws(scores, y, tw)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den, pa / pb, per


def prep_qwen35(calls: list) -> dict:
    """calls: [call_id, ce(x, chunk, row), cmd(x, chunk, row) or None]. -> output_end and averaged-command locations."""
    import numpy as np

    xs = sorted({c[1][0] for c in calls} | {c[2][0] for c in calls if c[2]})
    ce_owner, oe, cmd_key, avg, pooled_keys = {}, {}, {}, {}, Counter()
    for x in xs:
        for f in sorted(Path(f"/acts/{x}/meta").glob("*/*.json")):
            m = json.loads(f.read_text())
            if m.get("chunk") is None:
                continue
            fid = f"{x}|{f.parent.name}|{f.name}"
            for c in m["calls"]:
                if c.get("row") is None or "call_index" not in c:
                    continue
                loc, kind = (x, m["chunk"], m["row_start"] + c["row"]), c.get("kind", "command")
                k = (fid, c["irid"], c["call_index"])
                if kind == "call_end":
                    ce_owner[loc] = k
                elif kind == "output_end":
                    oe.setdefault(k, []).append(loc)
                elif kind == "command":
                    cmd_key[loc] = (x, m["arm"], m["key"], c["irid"], c["call_index"], c["readout"])
        for f in sorted(Path(f"/acts/{AVG}{x}/meta").glob("*/*.json")):
            m = json.loads(f.read_text())
            for c in m["calls"]:
                k5 = (m["arm"], m["key"], c["irid"], c.get("call_index", 0), c["readout"])
                pooled_keys[k5] += 1
                avg[(x, *k5)] = (AVG + x, m["chunk"], m["row_start"] + c["row"])
    out, cnt, gate = {}, Counter(), []
    for cid, ce, cmd in calls:
        k = ce_owner.get(tuple(ce))
        lst = oe.get(k, []) if k else []
        cnt["ce_found" if k else "ce_missing"] += 1
        cnt[f"output_end_{min(len(lst), 2)}"] += 1
        a = avg.get(cmd_key.get(tuple(cmd))) if cmd else None
        cnt["cmd_mean_found" if a else "cmd_mean_missing"] += 1
        if a:
            gate.append((tuple(cmd), a))
        out[cid] = {"reply_last": list(lst[0]) if len(lst) == 1 else None, "command_mean": list(a) if a else None}
    by = {}
    for o, a in gate:
        by.setdefault((o[0], o[1], a[0], a[1]), []).append((o[2], a[2]))
    mism = 0
    for (ox, oc, ax, ac), lst in by.items():
        A0 = np.load(f"/acts/{ox}/by_layer/L30/{oc}.npy", mmap_mode="r")
        A1 = np.load(f"/acts/{ax}/last/L30/{ac}.npy", mmap_mode="r")
        mism += sum(not np.array_equal(A0[r], A1[q]) for r, q in lst)
    return {"locs": out, "counts": dict(cnt), "avg_gate_checked": len(gate), "avg_gate_mismatches": int(mism),
            "v1_pooled_key_collisions": int(sum(v > 1 for v in pooled_keys.values()))}
