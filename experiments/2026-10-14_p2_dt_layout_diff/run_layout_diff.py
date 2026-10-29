"""RULE.md: on the T196 calls both layouts hold, the DT monitors' q4b-vs-replay Δz against prefix length and the token-level
differences of each call's two prefixes, by category (special ids and segment boundaries only). Token ids are never decoded,
printed or logged; only counts, positions, categories and statistics leave a container. Modal CPU, detached.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy --with pyyaml modal run --detach \\
      experiments/2026-10-14_p2_dt_layout_diff/run_layout_diff.py --manifest <correctedb_manifest_v1.json>
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-p2probes_dtlayout_kill-2026-11-13"
VOLS = {"/ev": "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13", "/acts": "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08",
        "/p2": "probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13", "/hf": "probe-shift_aswin-surya_hf-cache"}
WVOL = "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"
TOK = "hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654/tokenizer.json"
DESIGNS = ("destructive_primary", "transfer_primary", "destructive_secondary", "transfer_secondary")
REPLAY_NPZ = {"destructive_primary": "e142c828bb20f16305621af64c735635f863964731bfe42a0dd3b5f9147810ed",
              "transfer_primary": "db7b5e9c2ec84b078452472aa835e8b79181facd6a0178f18f426a3677c35c63",
              "destructive_secondary": "aadd98e5bcccda6874564b5c43a1db621578c634f786268f2603e305729519f9",
              "transfer_secondary": "a1bf1e3ac03d868bd3bf075dea073fda5b82fb187a8b73793e1de0a43d43a20b"}
T196_MANIFEST_SHA = "403dcdda61bfb03afb67062a22421e35b24ac9c8f9f0999d37e6b0d82eb5a634"  # = results/dt_q4b_t196_t196_2026-10-14_174544.json (the same json.dumps)
N_SHARED = 44582  # the T196 bridge's matched calls (every design)
MAIN = ["rerun_segends", "scaleup_canary", "scaleup_l2", "scaleup_topup"]
WORLD_FILES = {"run1": [(c, "attack.jsonl") for c in MAIN], "sim": [(c, "sim_attack@september.jsonl") for c in MAIN],
               "honest_a": [(c, "honest_a.jsonl") for c in MAIN], "honest_b": [(c, "honest_b.jsonl") for c in MAIN],
               "run2": [("second_attack", "attack.jsonl")], "honest_c": [("honest_c", "honest_b.jsonl")]}
CATS = ("special", "system", "think", "role_header", "content")
SHARDS = 4

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
RO = {m: (modal.Volume.from_name(n) if m == "/hf" else modal.Volume.from_name(n, version=2)).read_only() for m, n in VOLS.items()}  # the HF cache is a v1 Volume
wv = modal.Volume.from_name(WVOL, version=2)
LIM = dict(cpu=(2, 2), memory=(4096, 4096), retries=0, scaledown_window=10, startup_timeout=120)


def _cats(S, sp: dict):
    """Per position of token-id array S, its category index (CATS), from special-token ids and segment boundaries only."""
    import numpy as np

    c = np.full(len(S), 4, np.int8)  # content
    st = np.flatnonzero(S == sp["<|im_start|>"])
    for k in (1, 2):
        c[st[st + k < len(S)] + k] = 3  # role header: the two tokens after <|im_start|>
    t0, t1 = np.flatnonzero(S == sp["<think>"]), np.flatnonzero(S == sp["</think>"])
    for i in t0:
        j = t1[t1 > i]
        c[i + 1: (j[0] if len(j) else len(S))] = 2  # think
    e = np.flatnonzero(S == sp["<|im_end|>"])
    c[: (e[0] if len(e) else len(S))] = 1  # the first message: the system turn, with any offered-tool block
    c[np.isin(S, np.array(sorted(sp["all"]), np.int64))] = 0  # special tokens
    return c


def _special() -> dict:
    """The named special-token ids and every added token's id, from the pinned tokenizer, built inside each container (never returned)."""
    tok = json.loads((Path("/hf") / TOK).read_text())
    sp = {t["content"]: t["id"] for t in tok["added_tokens"]}
    if not all(k in sp for k in ("<|im_start|>", "<|im_end|>", "<think>", "</think>")):
        raise ValueError("a required special token is missing")
    return {k: sp[k] for k in ("<|im_start|>", "<|im_end|>", "<think>", "</think>")} | {"all": sorted({t["id"] for t in tok["added_tokens"]})}


@app.function(image=image, timeout=900, volumes=RO, **LIM)
def token_diff(world: str, need: dict, q4b_files: dict, replay_files: list) -> dict:
    """Withholds every exception's message (an input line could ride along); see _token_diff."""
    try:
        return _token_diff(need, q4b_files, replay_files)
    except Exception as x:
        raise RuntimeError(f"token_diff {world}: {type(x).__name__} (message withheld)") from None


def _token_diff(need: dict, q4b_files: dict, replay_files: list) -> dict:
    """need: {irid: [ranks]} of one shard's shared calls. Streams the world's staged files (sha256 checked while reading), keeps
    only epochs holding a needed irid, and per (q4b epoch, replay context epoch) pair: the exact positional first difference, and one
    difflib alignment on prefixes up to the longest needed call. Per call: the identical flag (exact comparison), the after-first-
    difference flag, and differing tokens by category on each side, a token counting as matched only if its partner lies inside the
    other side's prefix. Returns counts only."""
    import difflib

    import numpy as np

    sp = _special()

    def stream(path, want):
        h, keep = hashlib.sha256(), []
        with open(path, "rb") as fh:
            for line in fh:
                h.update(line)
                ep = json.loads(line)
                cmd = {}
                for c in ep["calls"]:
                    if c.get("readout") is None:  # requests without a readout carry no kind or call_index (Codex r1)
                        continue
                    if c["kind"] == "command" and c["irid"] in need:
                        cmd.setdefault(c["irid"], []).append((c["call_index"], c["readout"]))
                if cmd:
                    keep.append((np.asarray(ep["ids"], np.int64), {k: [r for _, r in sorted(v)] for k, v in cmd.items()}))
        if h.hexdigest() != want:
            raise ValueError("a staged file's sha256 differs from its pin")
        return keep

    Q, R = {}, {}
    for f, sha in q4b_files.items():
        for ids, cmd in stream(Path("/ev") / f, sha):
            Q.update({irid: (ids, ro) for irid, ro in cmd.items()})
    for f, sha in replay_files:
        for ids, cmd in stream(Path("/acts") / f, sha):
            R.update({irid: (ids, ro) for irid, ro in cmd.items()})
    pairs, out, missing, runs = {}, {}, 0, []
    for irid, ranks in need.items():
        if irid not in Q or irid not in R or max(ranks) >= min(len(Q[irid][1]), len(R[irid][1])):
            missing += len(ranks)
            continue
        pairs.setdefault((id(Q[irid][0]), id(R[irid][0])), []).append(irid)
    for irids in pairs.values():
        qa, ra = Q[irids[0]][0], R[irids[0]][0]
        lq = max(Q[i][1][k] for i in irids for k in need[i]) + 1
        lr = max(R[i][1][k] for i in irids for k in need[i]) + 1
        n = min(lq, lr)
        d = np.flatnonzero(qa[:n] != ra[:n])
        fd = int(d[0]) if len(d) else (n if lq != lr else None)  # positional first difference within the needed prefixes
        mq, mr = np.full(lq, -1, np.int64), np.full(lr, -1, np.int64)
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, qa[:lq].tolist(), ra[:lr].tolist(), autojunk=False).get_opcodes():
            if tag == "equal":
                mq[i1:i2], mr[j1:j2] = np.arange(j1, j2), np.arange(i1, i2)
        cq, cr = _cats(qa[:lq], sp), _cats(ra[:lr], sp)
        runs.append(None if fd is None else CATS[int(cq[fd])] if fd < lq else CATS[int(cr[fd])])
        for irid in irids:
            for k in need[irid]:
                pq, pr = Q[irid][1][k] + 1, R[irid][1][k] + 1
                dq = (mq[:pq] < 0) | (mq[:pq] >= pr)
                dr = (mr[:pr] < 0) | (mr[:pr] >= pq)
                out[f"{irid}|{k}"] = {"len_q": int(pq), "len_r": int(pr), "identical": bool(pq == pr and np.array_equal(qa[:pq], ra[:pr])),
                                      "after_first_diff": bool(fd is not None and max(pq, pr) - 1 >= fd),
                                      "nq": np.bincount(cq[:pq][dq], minlength=5).tolist(), "nr": np.bincount(cr[:pr][dr], minlength=5).tolist()}
    return {"calls": out, "missing": missing, "first_diff_by_pair": runs}


def _rank(v):  # average ranks (ties share their mean)
    import numpy as np

    _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    return (np.concatenate([[0], np.cumsum(cnt)[:-1]]) + (cnt - 1) / 2.0)[inv]


def _finite(x):  # every non-finite float (NaN, ±Infinity) becomes None (Codex r2), so the record serializes with allow_nan=False
    import math

    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    return None if isinstance(x, float) and not math.isfinite(x) else x


def _summ(x):
    import numpy as np

    return {"n": int(len(x)), "median": float(np.median(x)) if len(x) else None, "p90": float(np.percentile(x, 90)) if len(x) else None}


@app.function(image=image, timeout=1500, volumes=RO | {"/work": wv}, **LIM)
def orchestrate(q4b_files: dict, replay_files: dict, meta: dict) -> dict:
    """Withholds every exception's message; see _orchestrate."""
    try:
        return _orchestrate(q4b_files, replay_files, meta)
    except Exception as x:
        raise RuntimeError(f"orchestrate: {type(x).__name__} (message withheld)") from None


def _orchestrate(q4b_files: dict, replay_files: dict, meta: dict) -> dict:
    """Matches the shared calls (the bridge's key), computes Δz per design, fans out the token diff, and writes the record."""
    import numpy as np

    wv.reload()
    out = Path("/work/layoutdiff") / meta["stamp (PT)"]
    if out.exists():
        raise ValueError(f"{out} exists (append-only)")
    mraw = (Path("/work/scaleup/t196_q4b") / "manifest_dt.json").read_bytes()
    if hashlib.sha256(mraw).hexdigest() != T196_MANIFEST_SHA:
        raise ValueError("the T196 manifest differs from its pin")
    man = json.loads(mraw)

    def ranked(keys):
        grp, res = {}, [None] * len(keys)
        for n, k in enumerate(keys):
            grp.setdefault(k[:3], []).append((k[3], n))
        for g, v in grp.items():
            for r, (_, n) in enumerate(sorted(v)):
                res[n] = g + (r,)
        return res

    dz, calls = {}, None
    for des in DESIGNS:
        q = {}
        for w in WORLD_FILES:
            f = man["files"][f"{w}/DT_{des}"]
            raw = Path(f["path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != f["sha256"]:
                raise ValueError(f"{w}/{des}: q4b npz sha256 differs from the T196 manifest")
            with np.load(Path(f["path"])) as z:
                q[w] = {k: z[k] for k in ("key", "irid", "call_index", "score")}
        qk = ranked([(str(k).split("|")[0], str(k).split("|")[1], str(i), int(c)) for w in q for k, i, c in zip(q[w]["key"], q[w]["irid"], q[w]["call_index"])])
        qs = np.concatenate([q[w]["score"] for w in q])
        fp = Path("/p2/2026-10-14_p2_callfacet/2026-10-14_141350") / f"DT_{des}__rule1_command.npz"
        if hashlib.sha256(fp.read_bytes()).hexdigest() != REPLAY_NPZ[des]:
            raise ValueError(f"{fp.name}: sha256 differs from its pin")
        with np.load(fp) as zf:
            z = {k: zf[k] for k in zf.files}
        wz = np.array([str(k).split("|")[0] for k in z["key"]])
        hon = np.isin(wz, ["honest_b", "honest_c"]) & (z["impactful"] == 1)
        sdk = np.array([z["score"][hon & (z["fold"] == k)].std() for k in range(5)])
        kr = np.flatnonzero(np.isin(wz, list(WORLD_FILES)))
        rk = ranked([(wz[r], str(z["key"][r]).split("|")[1], str(z["irid"][r]), int(z["call_index"][r])) for r in kr])
        idx = {k: n for n, k in enumerate(qk)}
        m = {k: (r, idx[k]) for r, k in zip(kr, rk) if k in idx}
        dz[des] = {k: float((qs[n] - z["score"][r]) / sdk[z["fold"][r]]) for k, (r, n) in m.items()}
        if calls is not None and set(m) != calls:
            raise ValueError("the matched calls differ across designs")
        calls = set(m)
    if len(calls) != N_SHARED:
        raise ValueError("the matched calls differ from the bridge's count")
    need, tix = {}, {}
    for w in WORLD_FILES:
        tix[w] = {t: i for i, t in enumerate(sorted({c[1] for c in calls if c[0] == w}))}
    for w, t, irid, k in sorted(calls):
        need.setdefault((w, tix[w][t] % SHARDS), {}).setdefault(irid, []).append(k)
    args = [(w, nd, q4b_files[w], replay_files[w]) for (w, _), nd in sorted(need.items())]
    res, missing, fdc = {}, 0, []
    for r in token_diff.starmap(args):
        res |= r["calls"]
        missing += r["missing"]
        fdc += r["first_diff_by_pair"]
    keys = [k for k in sorted(calls) if f"{k[2]}|{k[3]}" in res]
    if missing or len(keys) != len(calls):
        raise ValueError("some shared calls were not found in the staged inputs")
    T = [res[f"{k[2]}|{k[3]}"] for k in keys]
    lenq = np.array([t["len_q"] for t in T])
    ident = np.array([t["identical"] for t in T])
    after = np.array([t["after_first_diff"] for t in T])
    cnt = np.array([np.add(t["nq"], t["nr"]) for t in T], np.float64)
    stats = {"shared_calls": len(calls), "with_token_stats": len(keys), "missing_in_staged_inputs": missing,
             "share_identical_prefix": float(ident.mean()) if len(T) else None,
             "first_diff_category_by_epoch_pair": {str(c): fdc.count(c) for c in CATS + (None,)},
             "diff_counts": {c: {"median": float(np.median(cnt[:, i])), "p90": float(np.percentile(cnt[:, i], 90)), "share_any": float((cnt[:, i] > 0).mean())} for i, c in enumerate(CATS)},
             "per_design": {}}
    q5 = np.quantile(lenq, [0.2, 0.4, 0.6, 0.8]) if len(T) else []
    for des in DESIGNS:
        d = np.abs(np.array([dz[des][k] for k in keys]))
        s = np.array([dz[des][k] for k in keys])
        X = np.column_stack([np.ones(len(d)), cnt])
        fit = {}
        for nm, M in (("raw", X), ("log1p", np.column_stack([np.ones(len(d)), np.log1p(cnt)]))):
            b, *_ = np.linalg.lstsq(M, d, rcond=None)
            fit[nm] = {"coef": dict(zip(("intercept",) + CATS, b.tolist())), "r2": float(1 - ((d - M @ b) ** 2).sum() / ((d - d.mean()) ** 2).sum())}
        stats["per_design"][des] = {
            "spearman_absdz_vs_prefix_len": float(np.corrcoef(_rank(d), _rank(lenq))[0, 1]),
            "absdz_by_prefix_len_quintile": [_summ(d[np.digitize(lenq, q5) == i]) for i in range(5)],
            "prefix_len_quintile_edges": [float(x) for x in q5],
            "before_first_diff": _summ(d[~after]), "after_first_diff": _summ(d[after]),
            "identical_prefix": _summ(d[ident]) | {"mean_signed": float(s[ident].mean()) if ident.any() else None},
            "non_identical_prefix": _summ(d[~ident]) | {"mean_signed": float(s[~ident].mean()) if (~ident).any() else None},
            "ols_absdz_on_diff_counts": fit}
    rec = meta | {"t196_manifest_sha256": T196_MANIFEST_SHA, "t196_reading": man.get("reading"), "categories": CATS,
                  "stats": _finite(stats)}
    out.mkdir(parents=True)
    with (out / "record.json").open("x") as fh:
        fh.write(json.dumps(rec, indent=1, allow_nan=False))
    wv.commit()
    return rec


@app.local_entrypoint()
def main(manifest: str):
    import yaml

    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
    head, here = git("rev-parse", "HEAD"), "experiments/2026-10-14_p2_dt_layout_diff"
    for f in ("RULE.md", "run_layout_diff.py"):
        if git("status", "--porcelain", "--", f"{here}/{f}") or not git("ls-files", "--", f"{here}/{f}"):
            raise SystemExit(f"{f} is not committed unchanged")
    if not git("branch", "-r", "--contains", head):
        raise SystemExit("HEAD is not pushed")
    raw = Path(manifest).read_bytes()
    if hashlib.sha256(raw).hexdigest() != "bf847d425b80d95ebb75a9ed1d524fc68c9efcd014710ddc18e445ed9d373ebc":
        raise SystemExit("corrected-B manifest sha256 differs from its pin")
    q4b = {w: {} for w in WORLD_FILES}
    for e in json.loads(raw)["epochs"]:
        if q4b[e["world"]].setdefault(e["staged_inputs_file"], e["staged_inputs_sha256"]) != e["staged_inputs_sha256"]:
            raise SystemExit("a staged file with two hashes")
    q4b["honest_c"] = {"q4f_c_v1/inputs/honest_b.jsonl": "513f621a9e71e91f8188cccc76cb222e9c9f4acb5a91fc90c45a283964729b31"}
    cfgdir = REPO / "experiments/2026-10-08_q3_onpolicy_figures/config"
    rep = {w: [] for w in WORLD_FILES}
    for w, items in WORLD_FILES.items():
        for c, fname in items:
            cfg = yaml.safe_load((cfgdir / f"extract_{c}.yaml").read_text())
            rep[w].append((f"{cfg['inputs']}/inputs/{fname}", cfg["inputs_sha256"][fname]))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    meta = {"label": "PRELIMINARY; RULE.md in this folder; counts and statistics only (no token text)", "git": head, "stamp (PT)": stamp,
            "app": APP, "q4b_staged_files": q4b, "replay_staged_files": rep}
    r = orchestrate.remote(q4b, rep, meta)
    dest = HERE / "results" / f"layout_diff_{stamp}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.open("x").write(json.dumps(r, indent=1))
    print(json.dumps(r["stats"], indent=1)[:4000])
