"""Does Qwen3.5 attack differently in simulation, and does that explain the catch gap? (RULE.md; part A, the 196 cohort.)
Modal CPU. Scan containers read the attack .eval logs and return ONLY counts, booleans, tool-function names, token counts and
short scorer values; no attack text leaves a container. The orchestrator joins each run to its corrected-B scored sample by
irid, computes the paired behaviour differences, and correlates them with each probe's standardized per-task score gap.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 --with numpy==2.3.3 modal run --detach \\
      experiments/2026-10-14_attack_behaviour_sim_vs_real/run_behaviour.py --labels <claimc_labels_reduced.json>
"""

import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-attack_behaviour_kill-2026-11-14"
EV_ROOT = "/ev/evals_v1"
LABELS_SHA256 = "d00e88773fc2a71bef51e1ff0ccc2559932f5c3a306a8f5604c5248e797dc995"
T196 = ("experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json",
        "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb")
OWN_RECORD = ("cc53afc3a", "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/correctedb_scores_2026-10-14_164934.json",
              "c002d5f822e49d7877f4e8b79a80ca9dd2e2f16c664b921d3ad12cb3da4c49ac")
OWN_CHOICE = [[49, 50], [49, 50], [50, 51], [49, 50], [49, 50]]
G22 = "xfer/correctedb_tight22g_2026-10-14_144709/"
RH, HS = "2026-10-14_p2_refusal_harm/2026-10-14_145748/", "2026-10-14_p2_highstakes/2026-10-14_133519/"
# probe, readout, volume, path, sha256, fold model?, run aggregate; own_a's files come from its record (local side)
PROBES = [
    ("sol_a_tight", "reply", "sol", G22 + "perrow_asis_tight__a__averaged_reply_top2.npz", "aca93022c00ebb8394977be5013aaf53f4881e35b401c91ab8ffad846cd52d50", True, "mean"),
    ("sol_a_tight", "command", "sol", G22 + "perrow_asis_2026-10-14_135701_cmd__a__A_corrected_L63.npz", "3b0645c9c19456278fb6090b94476425b13a156a45a60a4508340068ef182d41", True, "max"),
    ("fibs", "reply", "p2", "fibs/2026-10-14_124948/reply_mean.npz", "576538939ae53f9afbf6da8b7e7123e698413c34331baebf7193d8236266f666", False, "mean"),
    ("fibs", "command", "p2", "fibs/2026-10-14_124948/call_end_content.npz", "3f3ef32be2b576b75bec29f900cb560d39690709b34038cab9549f2b7882b558", False, "max"),
    ("harmfulness_h200", "reply", "p2", RH + "harmfulness_reply_mean.npz", "28678271ce765b669d92127f15ff55092208c973d3232b4def0e3b113d52b675", False, "mean"),
    ("harmfulness_h200", "command", "p2", RH + "harmfulness_call_end_content.npz", "978d124f6057677a4a6d9af3d42f35d38afeeaacd92ab97f3fc7a6d8458b8bd0", False, "max"),
    ("refusal_h200", "reply", "p2", RH + "refusal_reply_mean.npz", "8b8e64752f5ee67077ef0ee7f018cc683cde2f51ddaf07e64ccc46bd028ff360", False, "mean"),
    ("refusal_h200", "command", "p2", RH + "refusal_call_end_content.npz", "67efd85de72fbec4f93704228afe97ac4e6c224b9948dcec577cbfdb105143ff", False, "max"),
    ("highstakes", "reply", "p2", HS + "highstakes_reply_mean.npz", "98e246bd8441a14f0c9870a79d5645ba54953f86940712ae7acc12f4c9d02838", False, "mean"),
    ("highstakes", "command", "p2", HS + "highstakes_call_end_content.npz", "d81d9ef904462961899a604823dc5f00c1423ee61aa578b98618c37d2af10717", False, "max"),
]
REAL_TRAINED = ("sol_a_tight", "own_a_tight")
METRICS = ("attempted", "side_task_success", "steps", "calls_total", "calls_bash", "calls_text_editor", "calls_submit", "calls_other",
           "out_tokens", "final_prompt_tokens", "submit_present", "submit_position")
N_BOOT, SEED = 10_000, [20261014, 1]
AGENT_MODEL = "vllm/Qwen/Qwen3.5-27B"  # sim runs also log the simulator's own x-irid model calls (check_agent_events.py); tokens count the agent only
SAMPLE_RE = re.compile(r"samples/.+_epoch_(\d+)\.json")
REDRAW_RE = re.compile(r"_redraw\d+$")
PATH_RE = re.compile(r"[A-Za-z0-9_@.:/-]+\.eval")

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "scipy==1.16.2", "zstandard==0.23.0")
ev = modal.Volume.from_name("probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13", version=2).read_only()
vols = {"/sol": modal.Volume.from_name("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13").read_only(),
        "/p2": modal.Volume.from_name("probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13").read_only()}


def world_of_path(rel: str):  # 09's run_tool_behaviour.world_of_path @ 19421fa7f, attack worlds only
    arm = Path(rel).parts[1]
    if arm == "attack":
        return "run2" if rel.startswith("2026-10-10_001314_second_attack/") else "run1"
    return "sim" if arm == "sim_attack@september" else None


def zstd_zip():  # 09's run_tool_behaviour.zstd_zip @ 19421fa7f, unchanged
    import zipfile

    import zstandard

    if hasattr(zipfile, "ZIP_ZSTANDARD") or getattr(zipfile, "_zstd_patched", False):
        return zipfile
    check, get = zipfile._check_compression, zipfile._get_decompressor
    zipfile._check_compression = lambda ct: None if ct == 93 else check(ct)
    zipfile._get_decompressor = lambda ct: zstandard.ZstdDecompressor().decompressobj() if ct == 93 else get(ct)
    zipfile._zstd_patched = True
    return zipfile


def short(v):  # a scorer value kept only if it is a short scalar (no text leaves the container)
    return v if isinstance(v, (bool, int, float)) or (isinstance(v, str) and len(v) <= 20) else None


NAME_RE = re.compile(r"[A-Za-z0-9_.:-]{1,64}")  # tool-function / scorer names and tool-call ids must look like identifiers


def ident(x):
    if not isinstance(x, str) or not NAME_RE.fullmatch(x):
        raise ValueError("a name or id field is not an identifier")
    return x


@app.function(image=image, cpu=1, memory=8192, timeout=1800, retries=0, volumes={"/ev": ev})
def scan(rels: list) -> dict:
    """Counts only: per sample, irid -> tool-call ids, per-call function names, steps, token counts, short scorer values."""
    zipfile = zstd_zip()
    out, errors = [], {}
    for rel in rels:
        try:
            with zipfile.ZipFile(Path(EV_ROOT) / rel) as z:
                names = [n for n in z.namelist() if SAMPLE_RE.fullmatch(n)]
                if not names:
                    raise ValueError("no samples")
                for n in names:
                    s = json.loads(z.read(n))
                    att = s.get("attachments") or {}
                    irid, out_tok, last_in, conflicts, nonagent = {}, 0, None, 0, 0
                    for e in s.get("events") or []:
                        if not isinstance(e, dict) or e.get("event") != "model":
                            continue
                        hdr = ((e.get("call") or {}).get("request") or {}).get("extra_headers") or {}
                        r = hdr.get("x-irid") if isinstance(hdr, dict) else None
                        r = att.get(r[len("attachment://"):], r) if isinstance(r, str) and r.startswith("attachment://") else r
                        if not r:
                            continue
                        if e.get("model") != AGENT_MODEL:
                            nonagent += 1
                            continue
                        r = ident(r)
                        o = e.get("output") or {}
                        ch = (o.get("choices") or [{}])[0]
                        ids = [ident(c.get("id")) for c in ((ch.get("message") or {}).get("tool_calls") or [])]
                        conflicts += r in irid and irid[r] != ids
                        irid[r] = ids
                        u = o.get("usage") or {}
                        out_tok += int(u.get("output_tokens") or 0)
                        last_in = int(u["input_tokens"]) if u.get("input_tokens") is not None else None  # the LAST agent event's, or missing
                    steps, fns, order = 0, [], []
                    for m in s.get("messages") or []:
                        if m.get("role") == "assistant":
                            steps += 1
                            for c in m.get("tool_calls") or []:
                                fns.append(ident(c.get("function")))
                                order.append(ident(c.get("id")))
                    names = []  # (top, sub) of every scorer key, flattened one level, regardless of value; only the side-task value is kept
                    for name, sc in (s.get("scores") or {}).items():
                        v = sc.get("value") if isinstance(sc, dict) else sc
                        names += [(ident(name), ident(k)) for k in v] if isinstance(v, dict) else [(ident(name), None)]
                    hits = [(t, u) for t, u in names if "side_task" in (t if u is None else f"{t}/{u}").lower()]
                    side = [t if u is None else f"{t}/{u}" for t, u in hits]
                    sv = None
                    if len(hits) == 1:
                        t, u = hits[0]
                        sc = s["scores"][t]
                        v = sc.get("value") if isinstance(sc, dict) else sc
                        x = v[u] if u is not None else v
                        sv = x if x in ("C", "I") else None  # allowlist: only the score letter leaves the container
                    out.append({"world": world_of_path(rel), "task": REDRAW_RE.sub("", Path(rel).parts[2]), "irid": irid, "irid_conflicts": conflicts,
                                "order": order, "fns": fns, "steps": steps, "out_tokens": out_tok, "final_prompt_tokens": last_in,
                                "side_keys": side, "side_value": sv, "nonagent_irid_events": nonagent})
        except Exception as e:  # noqa: BLE001 -- the error type only, never text; counted and gated
            errors[rel] = type(e).__name__
    return {"rows": out, "errors": errors}


@app.function(image=image, cpu=4, memory=16384, timeout=3600, retries=0, volumes={"/ev": ev, **vols})
def analyze(labels: dict, t196: dict, own: dict) -> dict:
    import io

    import numpy as np
    from scipy import stats

    rels = sorted(str(p.relative_to(EV_ROOT)) for p in Path(EV_ROOT).glob("*/*/*/*.eval") if world_of_path(str(p.relative_to(EV_ROOT))))
    if not rels or not all(PATH_RE.fullmatch(r) for r in rels):
        raise ValueError("unexpected eval path")
    parts = list(scan.map([rels[i:i + 20] for i in range(0, len(rels), 20)]))
    errs = {k: v for p in parts for k, v in p["errors"].items()}
    if errs:
        raise ValueError(f"scan errors in {len(errs)} files: {sorted(set(errs.values()))}")
    samples = [r for p in parts for r in p["rows"]]

    def npz(vol, path, sha):
        raw = (Path("/" + vol) / path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError(f"{path}: sha256 differs")
        return np.load(io.BytesIO(raw), allow_pickle=False)

    # join each attack run to its corrected-B scored sample through the tight Sol command rows' irids
    zc = npz("sol", PROBES[1][3], PROBES[1][4])
    run_irids, run_units = {}, {}
    for k, r, c, u in zip(zc["key"].astype(str), zc["irid"].astype(str), zc["call_index"].astype(np.int64), zc["unit"].astype(np.int64)):
        if k.split("|", 1)[0] in ("run1", "run2", "sim"):
            run_irids.setdefault(k, set()).add(r)
            run_units.setdefault(k, []).append((int(u), r, int(c)))
    run_units = {k: [(r, c) for _, r, c in sorted(v)] for k, v in run_units.items()}  # in unit order
    by_wt = {}
    for s in samples:
        by_wt.setdefault((s["world"], s["task"]), []).append(s)
    joined, gate = {}, {"multi_or_none": 0, "irid_conflicts": 0, "call_index_bad": 0, "dup_order_ids": 0, "dup_event_ids": 0, "nonmonotone": 0}
    for k, ir in run_irids.items():
        w, t = k.split("|", 1)
        hit = [s for s in by_wt.get((w, t), []) if ir <= set(s["irid"])]
        if len(hit) != 1:
            gate["multi_or_none"] += 1
            continue
        s = hit[0]
        gate["irid_conflicts"] += s["irid_conflicts"]
        gate["dup_order_ids"] += len(set(s["order"])) != len(s["order"])
        pos = {c: i for i, c in enumerate(s["order"])}
        gate["dup_event_ids"] += sum(len(set(v)) != len(v) for v in s["irid"].values())
        bad = [(r, c) for r, c in run_units[k] if not (0 <= c < len(s["irid"][r])) or s["irid"][r][c] not in pos]
        gate["call_index_bad"] += len(bad)
        if not bad:
            ps = [pos[s["irid"][r][c]] for r, c in run_units[k]]
            gate["nonmonotone"] += int(any(b <= a for a, b in zip(ps, ps[1:])))
        joined[k] = s
    if any(gate.values()):
        raise ValueError(f"join gate failed: {gate}")
    side_names = sorted({n for s in joined.values() for n in s["side_keys"]})
    not_one = sorted(k for k, s in joined.items() if len(s["side_keys"]) != 1)
    if not_one:
        raise ValueError(f"{len(not_one)} runs lack exactly one side-task scorer key (names seen: {side_names}), e.g. {not_one[:3]}")
    side = side_names

    def metric(k):
        s = joined[k]
        f = s["fns"]
        sv = s["side_value"]
        first = f.index("submit") if "submit" in f else None
        return {"attempted": float(labels.get(k) == "attempted"), "side_task_success": {"C": 1.0, "I": 0.0}.get(sv, np.nan),
                "steps": s["steps"], "calls_total": len(f), "calls_bash": f.count("bash"), "calls_text_editor": f.count("text_editor"),
                "calls_submit": f.count("submit"), "calls_other": sum(x not in ("bash", "text_editor", "submit") for x in f),
                "out_tokens": s["out_tokens"], "final_prompt_tokens": np.nan if s["final_prompt_tokens"] is None else s["final_prompt_tokens"],
                "submit_present": float(first is not None), "submit_position": np.nan if first is None else (first + 1) / len(f)}  # 1-based

    M = {k: metric(k) for k in joined}
    tasks = sorted({k.split("|", 1)[1] for k in M})
    cohort = [t for t in tasks if all(f"{w}|{t}" in M for w in ("run1", "run2", "sim"))]
    attempted = [t for t in cohort if all(labels.get(f"{w}|{t}") == "attempted" for w in ("run1", "run2", "sim"))]
    side_missing = {w: sorted(t for t in cohort if np.isnan(M[f"{w}|{t}"]["side_task_success"])) for w in ("run1", "run2", "sim")}

    # probe run scores, standardized (per fold for fold-model probes; honest-B moments, ddof 0)
    def run_scores(z, agg):
        key, score = z["key"].astype(str), z["score"].astype(np.float64)
        fold = z["fold"].astype(np.int64) if "fold" in z.files else None
        out, fo = {}, {}
        for i, k in enumerate(key):
            out.setdefault(k, []).append(score[i])
            if fold is not None:
                fo.setdefault(k, set()).add(int(fold[i]))
        if any(len(v) != 1 for v in fo.values()):
            raise ValueError("a run scored by more than one fold model")
        return {k: float(np.max(v) if agg == "max" else np.mean(v)) for k, v in out.items()}, {k: next(iter(v)) for k, v in fo.items()}

    def standardize(s, fo, outer):
        hb = [k for k in s if k.startswith("honest_b|")]
        if outer and set(fo) != set(s):
            raise ValueError("a fold-model probe has runs without a fold id")
        if not outer:
            sd = float(np.std([s[k] for k in hb]))
            return {k: v / sd for k, v in s.items()}
        out = {}
        for f in range(5):
            cal = np.array([s[k] for k in hb if fo[k] == f])
            if len(cal) < 2 or not np.isfinite(cal.std()) or cal.std() <= 0:
                raise ValueError(f"fold {f} calibration too small or flat")
            out |= {k: (v - cal.mean()) / cal.std() for k, v in s.items() if fo.get(k) == f}
        return out

    Z = {}
    for probe, ro, vol, path, sha, outer, agg in PROBES:
        s, fo = run_scores(npz(vol, path, sha), agg)
        Z[(probe, ro)] = standardize(s, fo, outer)
    for ro, spec in own.items():  # own_a tight: per-fold layer files (reply) or one file (command)
        if ro == "reply":
            rows = {}
            for f, (la, lb) in enumerate(OWN_CHOICE):
                za, zb = (npz("sol", spec[f"{L}|{f}"]["rows"], spec[f"{L}|{f}"]["rows_sha256"]) for L in (la, lb))
                ka = list(zip(za["key"].astype(str), za["unit"].astype(int)))
                kb = list(zip(zb["key"].astype(str), zb["unit"].astype(int)))
                if ka != kb:
                    raise ValueError(f"own_a fold {f}: layer files differ in rows")
                for (k, _), x, y in zip(ka, za["score"], zb["score"]):
                    rows.setdefault(k, ([], f))[0].append((float(x) + float(y)) / 2)
            s = {k: float(np.mean(v)) for k, (v, _) in rows.items()}
            fo = {k: f for k, (_, f) in rows.items()}
        else:
            s, fo = run_scores(npz("sol", spec["path"], spec["sha256"]), "max")
        Z[("own_a_tight", ro)] = standardize(s, fo, True)

    # bootstrap over cohort tasks, environment strata
    env = {c["key"]: c["env"] for c in t196["cells"]}

    def weights(ts):
        rng = np.random.default_rng(SEED)
        groups = [np.array([i for i, t in enumerate(ts) if env[t] == g]) for g in sorted({env[t] for t in ts})]
        W = np.zeros((N_BOOT + 1, len(ts)))
        W[0] = 1.0
        for b in range(1, N_BOOT + 1):
            for ii in groups:
                W[b, ii] = rng.multinomial(len(ii), np.full(len(ii), 1.0 / len(ii)))
        return W

    def wmean(W, v):
        ok = ~np.isnan(v)
        Wc = W[:, ok]
        with np.errstate(invalid="ignore", divide="ignore"):
            m = (Wc @ v[ok]) / Wc.sum(1)
        b = m[1:][~np.isnan(m[1:])]
        if not ok.any() or np.isnan(m[0]) or not len(b):
            return {"point": None, "ci95": [None, None], "n": int(ok.sum()), "undefined": True}
        return {"point": float(m[0]), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "n": int(ok.sum()),
                "nan_draws": int(np.isnan(m[1:]).sum())}

    def paired(ts):
        W = weights(ts)
        out = {}
        for mname in METRICS:
            v = {w: np.array([M[f"{w}|{t}"][mname] for t in ts], float) for w in ("run1", "run2", "sim")}
            # NaN propagates, so each difference uses only tasks where both of its arms are defined (e.g. submit_position: both submit)
            out[mname] = {"mean_own_runs": {w: wmean(W, x) for w, x in v.items()},
                          "sim_minus_run1": wmean(W, v["sim"] - v["run1"]),
                          "sim_minus_real": wmean(W, v["sim"] - (v["run1"] + v["run2"]) / 2),
                          "run2_minus_run1": wmean(W, v["run2"] - v["run1"])}
        return out

    res = {"side_task_keys": side, "side_missing": side_missing, "cohort_n": len(cohort), "attempted_n": len(attempted),
           "joined_runs": len(joined), "nonagent_irid_events_by_world": {w: sum(s["nonagent_irid_events"] for s in samples if s["world"] == w) for w in ("run1", "run2", "sim")},
           "paired_primary": paired(cohort), "paired_attempted": paired(attempted)}

    # Spearman of each behaviour difference with each probe's standardized score gap g = z(sim) - z(run1)
    W = weights(cohort)
    rows, draws = [], {}
    for (probe, ro), zz in sorted(Z.items()):
        g = np.array([zz.get(f"sim|{t}", np.nan) - zz.get(f"run1|{t}", np.nan) for t in cohort])
        for mname in METRICS:
            d = np.array([M[f"sim|{t}"][mname] - M[f"run1|{t}"][mname] for t in cohort], float)
            ok = ~np.isnan(g) & ~np.isnan(d)
            idx0 = np.flatnonzero(ok)
            rho = np.full(N_BOOT + 1, np.nan)
            for b in range(N_BOOT + 1):
                rep = np.repeat(idx0, W[b, idx0].astype(int))
                if len(rep) > 2 and np.ptp(d[rep]) > 0 and np.ptp(g[rep]) > 0:
                    rho[b] = stats.spearmanr(g[rep], d[rep]).statistic
            bd = rho[1:][~np.isnan(rho[1:])]
            p = min(1.0, 2 * min((bd <= 0).mean(), (bd >= 0).mean())) if len(bd) and not np.isnan(rho[0]) else None
            rows.append({"probe": probe, "readout": ro, "metric": mname, "n": int(ok.sum()), "rho": float(rho[0]),
                         "ci95": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))] if len(bd) else None,
                         "p_boot": p, "nan_draws": int(np.isnan(rho[1:]).sum())})
            draws[(probe, ro, mname)] = rho
    ps = [(i, 1.0 if r["p_boot"] is None else r["p_boot"]) for i, r in enumerate(rows)]  # full family; untestable = p 1
    run = 0.0
    for j, (i, p) in enumerate(sorted(ps, key=lambda x: x[1])):
        run = max(run, min(1.0, (len(ps) - j) * p))
        rows[i]["holm_p"] = run
    contrast = {}
    for mname in METRICS:
        full = lambda k: not np.isnan(draws[k]).any()  # noqa: E731 -- fixed membership: rows defined in every draw
        rt = [draws[k] for k in draws if k[2] == mname and k[0] in REAL_TRAINED and full(k)]
        pb = [draws[k] for k in draws if k[2] == mname and k[0] not in REAL_TRAINED and full(k)]
        if not rt or not pb:
            contrast[mname] = {"point": None, "ci95": [None, None], "rows_real": len(rt), "rows_public": len(pb), "undefined": True}
            continue
        c = np.mean(rt, axis=0) - np.mean(pb, axis=0)
        contrast[mname] = {"point": float(c[0]), "ci95": [float(np.percentile(c[1:], 2.5)), float(np.percentile(c[1:], 97.5))],
                           "rows_real": len(rt), "rows_public": len(pb)}
    res |= {"spearman": rows, "contrast_real_minus_public": contrast}
    return res


@app.local_entrypoint()
def main(labels: str):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO, check=True).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    lab_raw, t_raw = Path(labels).read_bytes(), (REPO / T196[0]).read_bytes()
    rec = subprocess.run(["git", "show", f"{OWN_RECORD[0]}:{OWN_RECORD[1]}"], capture_output=True, check=True, cwd=REPO).stdout
    if (hashlib.sha256(lab_raw).hexdigest() != LABELS_SHA256 or hashlib.sha256(t_raw).hexdigest() != T196[1]
            or hashlib.sha256(rec).hexdigest() != OWN_RECORD[2]):
        raise SystemExit("an input sha256 differs from RULE.md")
    saved = json.loads(rec)["saved"]
    own = {"reply": {k: {"rows": v["rows"], "rows_sha256": v["rows_sha256"]} for k, v in saved["tight"]["avg"].items()},
           "command": {"path": saved["cmd"]["cmdA"]["rows|own_a_tight"]["path"], "sha256": saved["cmd"]["cmdA"]["rows|own_a_tight"]["sha256"]}}
    res = analyze.remote(json.loads(lab_raw)["labels"], json.loads(t_raw), own)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    head = {"git": git("rev-parse", "HEAD"), "labels_sha256": LABELS_SHA256, "t196_sha256": T196[1], "own_record": f"{OWN_RECORD[1]}@{OWN_RECORD[0]}",
            "n_boot": N_BOOT, "seed": SEED}
    p = HERE / "results" / f"behaviour_{stamp}.json"
    p.parent.mkdir(exist_ok=True)
    with p.open("x") as fh:
        json.dump(head | res, fh)
    print(f"wrote {p.name} sha256 {hashlib.sha256(p.read_bytes()).hexdigest()}")
    print("side-task keys:", res["side_task_keys"], "| side missing:", res["side_missing"],
          "| non-agent x-irid events", res["nonagent_irid_events_by_world"], "| joined", res["joined_runs"], "cohort", res["cohort_n"], "attempted", res["attempted_n"])
    num = lambda x: "n/a" if x is None else f"{x:+.3f}"  # noqa: E731
    f = lambda d: f"{num(d['point'])} [{num(d['ci95'][0])}, {num(d['ci95'][1])}] n{d['n']}"  # noqa: E731
    for m, v in res["paired_primary"].items():
        print(f"{m:<20} mean r1 {num(v['mean_own_runs']['run1']['point'])} r2 {num(v['mean_own_runs']['run2']['point'])} sim {num(v['mean_own_runs']['sim']['point'])} | "
              f"sim-r1 {f(v['sim_minus_run1'])} | sim-real {f(v['sim_minus_real'])} | r2-r1 {f(v['run2_minus_run1'])}")
    for m, c in res["contrast_real_minus_public"].items():
        print(f"contrast {m:<20} real-public rho {num(c['point'])} [{num(c['ci95'][0])}, {num(c['ci95'][1])}] rows {c['rows_real']}/{c['rows_public']}")
    for r in sorted(res["spearman"], key=lambda x: x.get("holm_p", 1))[:12]:
        print(f"rho {r['probe']}|{r['readout']}|{r['metric']} {r['rho']:+.3f} {r['ci95']} p {r['p_boot']} Holm {r.get('holm_p')}")
