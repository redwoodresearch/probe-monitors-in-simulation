"""Price and attribute the OpenAI usage found by run_openai_usage_scan.py (keys and numbers only), and find repeated simulated runs.
Prices: the sims harness's own table, src/sim/prices.py @ sims ba990399 (OpenAI rows verified by its authors 2026-08-11 against the
vendor's Standard tier), USD per million tokens (input, output, cache read); the harness prices input_tokens + output_tokens +
input_tokens_cache_read exactly this way (its sim_usd), treating Inspect's input_tokens as excluding cache reads.
Repeats: simulated runs are grouped by (world family, arm, cell) where the cell is the task folder with launch suffixes removed; the
attempt the INPUTS_FULL inventories route is the "routed" one; others are earlier (before it started) or later attempts.
Runs INSIDE the Modal job (run_openai_usage_scan.py imports analyze()); never on the laptop (Aswin's 10-13 rule).
"""
import copy
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

PRICES = {"openai/gpt-6-astra": (10.0, 50.0, 1.0), "openai/gpt-5.6-sol": (5.0, 30.0, 0.5), "openai/gpt-5.6-terra": (2.0, 12.0, 0.2),
          "openai/gpt-5.6-luna": (0.2, 1.2, 0.02), "openai/gpt-5.4": (2.5, 15.0, 0.25), "openai/gpt-5.4-mini": (0.75, 4.5, 0.075),
          "openai/gpt-5.4-nano": (0.2, 1.25, 0.02), "openai/gpt-5-nano-2025-08-07": (0.05, 0.4, 0.005), "openai/gpt-4.1-mini": (0.4, 1.6, 0.1)}
ROLE = {"openai/gpt-6-astra": "simulator", "openai/gpt-5.6-sol": "simulator fallback / Sol policy", "openai/gpt-5.6-terra": "simulator judges",
        "openai/gpt-5-nano-2025-08-07": "side-task judge"}
VOL_ORDER = ("chain", "q4own", "pilots", "q2fresh")
SUFFIX = re.compile(r"_(?:r\d+|redraw\d+|e2[ab]|try\d+)$")


def strip_all(s):  # strip launch suffixes until none is left ("_r2_try1" -> base) (review r2 #10)
    while True:
        t = SUFFIX.sub("", s)
        if t == s:
            return s
        s = t
PT = timezone(timedelta(hours=-7))


def usd(model, u):
    p = PRICES.get(model)
    if p is None:
        return None
    return (u["input_tokens"] * p[0] + u["output_tokens"] * p[1] + u["input_tokens_cache_read"] * p[2]) / 1e6


def family(launch, arm):
    lo = launch.lower()
    if not arm.startswith("sim"):
        return "real"
    if "_il_" in lo or "sim2" in lo or "ilrr" in lo or "_ilc" in lo:
        return "sim2 (IL)"
    if arm.startswith("sim_honest") or "simh" in lo:
        return "sim honest"
    return "sim attack"


def covers(later, earlier):
    """True when `later`'s usage is >= `earlier`'s for every model and field (a retry carries prior usage forward)"""
    return all(later.get(m, {}).get(f, 0) >= v for m, u in earlier.items() for f, v in u.items())


def analyze(scan, routed, label):
    routed = set(routed)
    good = [copy.deepcopy(r) for r in scan["rows"] if "read_error" not in r]  # normalize a copy; the saved rows stay raw (review r3 #7)
    for r in good:  # "<non-id>" never groups rows together (review r2 #7)
        for k in ("eval_id", "run_id", "task_id", "eval_set_id"):
            if r.get(k) == "<non-id>":
                r[k] = None
    headerless = [r for r in good if not r.get("model_usage") and not r.get("started_at")]  # killed / in-progress: usage unknown, counted
    hl = {id(r) for r in headerless}
    rows = sorted([r for r in good if id(r) not in hl], key=lambda r: VOL_ORDER.index(r["vol"]))
    by_eval, copy_paths = {}, {}
    for r in rows:  # copies of one log on two Volumes share eval_id: keep the first Volume's copy
        k_ = r.get("eval_id") or ("path", "/".join(r["parts"]))
        by_eval.setdefault(k_, r)
        copy_paths.setdefault(k_, set()).add("/".join(r["parts"]))  # every Volume copy's path (the inventory may name either; 0d recheck note 2)
    groups_t = defaultdict(list)
    for r in by_eval.values():
        groups_t[r.get("task_id") or ("eval", r.get("eval_id"), "/".join(r["parts"]))].append(r)
    keep, n_folded, n_task_id_conflicts = [], 0, 0
    for k, rs in groups_t.items():  # fold an earlier attempt into the last ONLY if the last's usage covers it (review r2 #1)
        rs.sort(key=lambda r: r.get("started_at") or "")
        last_ = rs[-1]
        folded = [r for r in rs[:-1] if covers(last_["model_usage"], r["model_usage"])]
        kept = [r for r in rs[:-1] if not covers(last_["model_usage"], r["model_usage"])]
        n_folded += len(folded)
        n_task_id_conflicts += bool(kept)
        keep += kept + [last_]
    n_copies = len(rows) - len(by_eval)
    runs = []
    for r in keep:
        parts = r["parts"]
        if len(parts) >= 5 and parts[0] == "runs":  # chain keeps some launches under evals_v1/runs/<launch>/... (S_paper493 sims)
            parts = parts[1:]
        launch, arm = parts[0], (parts[1] if len(parts) >= 3 else "")
        cell = strip_all(parts[2]) if len(parts) >= 4 else (parts[1] if len(parts) == 3 else "")
        t = datetime.fromisoformat(r["started_at"]) if r.get("started_at") else None
        day = t.astimezone(PT).strftime("%m-%d") if t else "?"
        day_utc = t.astimezone(timezone.utc).strftime("%m-%d") if t else "?"
        cost = {m: usd(m, u) for m, u in r["model_usage"].items()}
        runs.append({"vol": r["vol"], "path": "/".join(r["parts"]), "launch": launch, "arm": arm, "cell": cell, "status": r["status"], "day": day,
                     "started_at": r.get("started_at"), "completed_at": r.get("completed_at"), "day_utc": day_utc,
                     "family": family(launch, arm), "routed": bool(copy_paths.get(r.get("eval_id") or ("path", "/".join(r["parts"])), {"/".join(r["parts"])}) & routed), "depth": len(parts),
                     "usd": sum(v for v in cost.values() if v), "unpriced": sorted(m for m, v in cost.items() if v is None and m.startswith("openai/")),
                     "usage": r["model_usage"]})
    by_model, by_launch, by_day, by_family, by_day_utc = Counter(), Counter(), Counter(), Counter(), Counter()
    tok = defaultdict(Counter)
    for x in runs:
        for m, u in x["usage"].items():
            c = usd(m, u)
            if c:
                by_model[m] += c
            for f, v in u.items():
                tok[m][f] += v
        by_launch[x["launch"]] += x["usd"]
        by_day[x["day"]] += x["usd"]
        by_day_utc[x["day_utc"]] += x["usd"]
        by_family[x["family"]] += x["usd"]
    sim = [x for x in runs if x["family"] != "real" and x["usage"].get("openai/gpt-6-astra")]
    astra = [x["usage"]["openai/gpt-6-astra"] for x in sim]
    per_run = {"n_sim_runs_with_astra": len(sim), "usd_mean": statistics.mean(x["usd"] for x in sim) if sim else None,
               "usd_median": statistics.median(x["usd"] for x in sim) if sim else None,
               "astra_input_median": statistics.median(u["input_tokens"] for u in astra) if astra else None,
               "astra_cache_read_median": statistics.median(u["input_tokens_cache_read"] for u in astra) if astra else None,
               "astra_output_median": statistics.median(u["output_tokens"] for u in astra) if astra else None,
               "astra_reasoning_median": statistics.median(u["reasoning_tokens"] for u in astra) if astra else None,
               "astra_cached_share": (sum(u["input_tokens_cache_read"] for u in astra) /
                                      max(1, sum(u["input_tokens"] + u["input_tokens_cache_read"] for u in astra))) if astra else None}
    groups = defaultdict(list)
    for x in runs:
        if x["family"] != "real" and x["cell"]:
            groups[(x["family"], x["arm"].split("@")[0], x["cell"])].append(x)
    rep = Counter()
    rep_usd = Counter()
    examples = defaultdict(list)
    n_multi_routed = 0
    for g, xs in groups.items():
        xs.sort(key=lambda x: x["started_at"] or "")
        r_ = [x for x in xs if x["routed"]]
        n_multi_routed += len(r_) > 1
        t_first = r_[0]["started_at"] if r_ else None
        t_done = max((x.get("completed_at") or x["started_at"] or "") for x in r_) if r_ else None  # last routed run's END (review r2 #3, #6)
        for x in xs:
            st = x["started_at"] or ""
            if x["routed"]:
                k = "routed"
            elif t_first is None:
                k = "no routed attempt in group"
            elif st < t_first:
                k = "earlier attempt, not success (retry)" if x["status"] != "success" else "earlier attempt, success (superseded)"
            elif st < t_done:
                k = "between / overlapping routed runs"
            else:
                k = "later attempt after the routed run existed"
            rep[(x["family"], k)] += 1
            rep_usd[(x["family"], k)] += x["usd"]
            if k.startswith("later") and len(examples[k]) < 15:
                examples[k].append({"launch": x["launch"], "arm": x["arm"], "cell": x["cell"], "usd": round(x["usd"], 2)})
    later_by_launch = Counter()
    for g, xs in groups.items():  # += per attempt, so two later attempts from one launch both count (review r2 #2)
        r_ = [x for x in xs if x["routed"]]
        if r_:
            t_done = max((x.get("completed_at") or x["started_at"] or "") for x in r_)
            for x in xs:
                if not x["routed"] and (x["started_at"] or "") >= t_done:
                    later_by_launch[x["launch"]] += x["usd"]
    fam_check = Counter((x["family"], bool(x["usage"].get("openai/gpt-6-astra"))) for x in runs)
    fam_map = sorted({(x["launch"], x["arm"], x["family"]) for x in runs})
    unpriced_tok = defaultdict(Counter)
    for x in runs:
        for m, u in x["usage"].items():
            if m not in PRICES:
                for f, v in u.items():
                    unpriced_tok[m][f] += v
    out = {"what": __doc__.strip().splitlines()[0], "inputs_full": label, "prices_per_M": PRICES,
           "n_rows": len(scan["rows"]), "n_read_errors": len(scan["rows"]) - len(good), "n_headerless_usage_unknown": len(headerless),
           "headerless_by_launch": dict(Counter(r["parts"][0] for r in headerless)), "n_volume_copies_dropped": n_copies,
           "n_retry_attempts_folded_into_last": n_folded, "n_task_id_groups_not_folded": n_task_id_conflicts, "n_groups_multi_routed": n_multi_routed,
           "total_is_lower_bound": "headerless and unreadable logs and non-Inspect calls are not included",
           "family_vs_astra_usage": {f"{f} | astra={a_}": n for (f, a_), n in sorted(fam_check.items())}, "family_map": fam_map,
           "depth_by_volume": {f"{v}|{d}": n for (v, d), n in sorted(Counter((x["vol"], x["depth"]) for x in runs).items())},
           "by_day_utc": {k: round(v, 2) for k, v in sorted(by_day_utc.items())},
           "unpriced_model_tokens": {m: dict(c) for m, c in unpriced_tok.items()},
           "price_source": "sims src/sim/prices.py @ ba990399", "n_runs": len(runs), "n_routed_matched": sum(x["routed"] for x in runs),
           "n_routed_keys_unmatched": len(routed - set().union(*copy_paths.values())) if copy_paths else len(routed), "total_usd": sum(by_model.values()), "by_model": {m: {"usd": round(v, 2), "role": ROLE.get(m, "?"), **tok[m]} for m, v in by_model.most_common()},
           "by_family": {k: round(v, 2) for k, v in by_family.most_common()}, "by_day_pt": {k: round(v, 2) for k, v in sorted(by_day.items())},
           "top_launches": [(k, round(v, 2)) for k, v in by_launch.most_common(15)], "per_sim_run": per_run,
           "repeats": {f"{f} | {k}": {"n": n, "usd": round(rep_usd[(f, k)], 2)} for (f, k), n in sorted(rep.items())},
           "later_attempt_usd_by_launch": [(k, round(v, 2)) for k, v in later_by_launch.most_common(15)], "later_examples": examples,
           "unpriced_openai_models": sorted({m for x in runs for m in x["unpriced"]}), "runs": runs}
    return out
