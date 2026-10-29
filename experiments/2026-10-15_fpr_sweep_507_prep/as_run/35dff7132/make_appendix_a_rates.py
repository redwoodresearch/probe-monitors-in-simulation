"""Appendix A failure RATES (09; 0d's recheck items 1-7 via 2e): failed / LAUNCHED honest RUNS, per world x set x arm, with one failure
definition and one category scheme on both sides. Keys and counts only; no score or transcript.

  python3 -I make_appendix_a_rates.py   # writes results/appendix_a_rates_<stamp>.json (exclusive create)

Inputs (git show, pinned by sha256):
  v3      Appendix A v3 (6cc962493): like_for_like (real runs with a named failure, by run key; sim-honest pool exclusions, by cell), the
          honest_launch denominators (pool members by set|arm from d9's counted manifests @ 1df288e61, deduplicated by run path), the
          non-member entries with their own .eval scan (n_model_events, log_status, sample_error), the earlier attempts it located, and the
          redraws whose earlier attempts no manifest lists.
  chain   d9's counted manifests at 1df288e61: every sim_honest row (TEST, _dry and sample_error_v1 files excluded, the same rule as v2's real
          side), deduplicated by (launch, rel); set by manifest family. A sim exclusion's FAILED ATTEMPTS are its cell's rows in its batch's
          manifests (originals and relaunches), minus the admitted replacement when there is one (Codex APPXA-R1: count attempts, not cells).
  pools   the frozen honest pools by row id (allowlist @ 8608dd0df: S_old / S_new / S_newmains H_A, H_B) and M1's S_paper493 block (H_B keys,
          pool_exclusions) for the pool-level loss.

FAILURE (both sides): a launched honest run that ended without a usable scored run because of an error: a sample error of any class, an
infrastructure error, or a failure of unknown cause. NOT a failure: a pool rule (paired-pool rule; non-membership by pool rule), a completed
run without rows and without an error (structural no-row), and a run that is 'stopped' only in the manifest (its own log: status success, no
sample error; 0d item 4), listed apart.
CATEGORIES (both sides; 0d item 3): setup (0 model events), mid-run model error (EmptyPolicyCompletionError / OutputContractError),
infrastructure (rate limit), other error (unclassified, model events > 0), unknown cause.
REAL is a LOWER BOUND: earlier attempts that no counted manifest lists are not seen; the record states the bound computed from the redraw
indices of the redraws whose earlier attempts were not located.
"""

import collections
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
V3 = ("6cc962493", "experiments/2026-10-15_fpr_sweep_507_prep/results/appendix_a_exclusions_v3_2026-10-16_115320.json", "84e71ac387c78a70")
CHAIN = ("1df288e61", "experiments/2026-10-14_q3_overnight_chain/config")
ALLOW = ("8608dd0df", "experiments/2026-10-15_q3_sp493/results/frozen_honest_scan_inputs_2026-10-16/allowlist_row_id.json")
M1 = ("a6bde80a2", "experiments/2026-10-15_fpr_sweep_507_prep/config/INPUTS_FINAL_V2_FROZEN_2026-10-16_044120.json",
      "0e065958b206dbceba8afeb3332ec68b898481da520fe3a3654ab2b6913eaadd")
COMMON = ("S_new", "S_newmains", "S_paper493")  # 0d item 7: the sets with both sides
REAL_FAIL = re.compile(r"^real\|(?P<set>[^|]+)\|(?P<arm>honest_[ab])\|(?P<cls>.+)$")
SIM_ENTRY = re.compile(r"^sim\|(?P<set>[^|]+)\|sim_honest\|(?P<cls>.+)$")
SIM_NOT_FAILURE = {"structural no-row (ledger.structural_no_rows)": "completed run without rows and without an error",
                   "H_sim cell without an admitted H_A member (paired-pool rule)": "pool rule"}
SIM_SET_OF = (("manifest_simh311_", "S_new"), ("manifest_simh_nm360_", "S_newmains"), ("manifest_sp493_g4_simh", "S_paper493"),
              ("manifest_sp493_g5_simh", "S_paper493"), ("manifest_simh_t196fill", "S_old"), ("manifest_q4b_eval_upload_v1", "S_old"))


def git_show(c, p):
    return subprocess.run(["git", "show", f"{c}:{p}"], capture_output=True, check=True, cwd=REPO).stdout


def cell_of(task):
    return re.sub(r"(_e2[ab]|_redraw\d+)+$", "", task)


def rate(f, n):
    return {"failed": f, "launched": n, "rate": f / n if n else None}


def real_category(cls, scan):
    if cls == "EmptyPolicyCompletionError":
        return "mid-run model error (EmptyPolicyCompletionError)"
    if cls == "unclassified" and scan.get("n_model_events") == 0:
        return "setup (0 model events)"
    if cls == "unclassified":
        return "other error (unclassified, model events > 0)"
    raise SystemExit(f"real class not mapped: {cls}")


def sim_category(reason, text):
    """By the error named in the ledger's reason text, or by its source location when only that is given: step_limit.py:72 / ':72' is
    EmptyPolicyCompletionError (nm258's own text: 'terminal error :72 EmptyPolicyCompletionError'); contract.py:313 / :344 is the output contract
    check (OutputContractError; t237's text names it)."""
    if "EmptyPolicyCompletionError" in text or re.search(r"(step_limit\.py)?:72\b", text):
        return "mid-run model error (EmptyPolicyCompletionError)"
    if "OutputContractError" in text or re.search(r"(contract\.py)?:3(13|44)\b", text):
        return "mid-run model error (OutputContractError)"
    if reason == "infrastructure" or "RL infrastructure" in text:
        return "infrastructure (rate limit)"
    return "unknown cause"


def main():
    b = git_show(*V3[:2])
    h = hashlib.sha256(b).hexdigest()
    if not h.startswith(V3[2]):
        raise SystemExit("v3 sha256")
    v3 = json.loads(b)
    hl = v3["sources"]["honest_launch"]
    if hl["chain_commit"] != CHAIN[0] or hl["launch_rows"] != 3717:
        raise SystemExit("v3 honest_launch: chain commit or launch rows not as relayed")
    E = {e["key"]: e for e in v3["entries"] if e["source"] in ("honest_launch", "honest_earlier_attempt")}
    gates = {}
    # ---------------- real: runs launched per set|arm = members + non-members (manifest rows) + located earlier attempts
    launched = collections.Counter({tuple(k.split("|")): v for k, v in hl["members_by_set_arm"].items()})
    for e in E.values():
        if e["world"] == "real":
            launched[(e["set"], e["arm"])] += 1
    gates["real_launched_rows"] = {"n": sum(launched.values()), "want": 3717 + v3["n_entries_by_source"]["honest_earlier_attempt"]}
    if gates["real_launched_rows"]["n"] != gates["real_launched_rows"]["want"]:
        raise SystemExit("real launched rows != 3,717 manifest rows + located earlier attempts")
    # real failed runs, by key, with category from each run's own scan
    rf, stopped, pool_rule, n_named = collections.defaultdict(list), [], {}, 0
    for k, v in v3["v3"]["like_for_like"].items():
        if k.startswith("real_by_pool_rule|"):
            pool_rule[k] = {"n": v["n"], "keys": v["keys"]}
            continue
        m = REAL_FAIL.match(k)
        if not m:
            continue
        n_named += v["n"]
        for key in v["keys"]:
            sc = E[key]["scan"]
            if m["cls"] == "state 'stopped'":
                if not (sc.get("log_status") == "success" and sc.get("sample_error") is None):
                    raise SystemExit(f"{key}: 'stopped' with an error in its own log; not handled")
                stopped.append({"key": key, "set": m["set"], "arm": m["arm"], "label": "stopped by manifest only (own log: success, no sample error)"})
                continue
            rf[(m["set"], m["arm"])].append({"key": key, "cell": E[key]["cell"], "category": real_category(m["cls"], sc), "class": m["cls"],
                                             "n_model_events": sc.get("n_model_events")})
    gates["real_named"] = {"n": n_named, "want": 76}
    if n_named != 76:
        raise SystemExit("real named failures != 76")
    # ---------------- sim: runs launched per set from the counted manifests; failed attempts per excluded cell
    ls = subprocess.run(["git", "ls-tree", "--name-only", CHAIN[0], CHAIN[1] + "/"], capture_output=True, text=True, check=True, cwd=REPO).stdout.split()
    srows, read_ = collections.defaultdict(dict), {}
    for p in sorted(ls):
        n = p.rsplit("/", 1)[1]
        if not (n.startswith("manifest_") and n.endswith(".json")) or "TEST" in n or "sample_error_v1" in n or "_dry" in n:
            continue
        m = json.loads(git_show(CHAIN[0], p))
        rows = [x for x in m.get("files", []) if str(x.get("arm_dir", "")).startswith("sim_honest")]
        st = next((s for pre, s in SIM_SET_OF if n.startswith(pre)), None)
        if rows and st is None:
            raise SystemExit(f"{n}: sim_honest rows in a manifest outside the mapped families")
        if st is None:
            continue
        read_[n] = len(rows)
        for x in rows:
            srows[st][(x["launch"], x["rel"])] = (n, cell_of(x["task"]))
    sl = {s: len(v) for s, v in srows.items()}
    sf, s_not, cells_lost, sim_cells = collections.defaultdict(list), {}, collections.defaultdict(list), {}
    hs = {e["key"]: e for e in v3["entries"] if e["source"] in ("hsim_base", "paired")}
    for k, v in v3["v3"]["like_for_like"].items():
        m = SIM_ENTRY.match(k)
        if not m:
            continue
        if m["cls"] in SIM_NOT_FAILURE:
            s_not[k] = {"n": v["n"], "why_not_a_failure": SIM_NOT_FAILURE[m["cls"]], "keys": v["keys"]}
            continue
        if m["cls"] not in ("infrastructure", "unknown"):
            raise SystemExit(f"sim class not mapped: {m['cls']}")
        for ck in v["keys"]:
            e = hs[ck]
            batch, cell = ck.split("|", 1)
            att = sorted(rk for rk, (mn, c) in srows[m["set"]].items() if mn.startswith(f"manifest_{batch}_") and c.split("_", 1)[0] == cell.split("_", 1)[0])
            n_fail = len(att) - (1 if e.get("replacement_admitted") else 0)
            if n_fail < 1:
                raise SystemExit(f"{ck}: no failed attempt among its {len(att)} manifest rows")
            sim_cells[ck] = {"rows": len(att), "replacement_admitted": bool(e.get("replacement_admitted")), "failed_attempts": n_fail}
            cat = sim_category(m["cls"], e.get("reason_text", ""))
            for i in range(n_fail):
                sf[m["set"]].append({"cell": ck, "category": cat, "class": m["cls"], "attempt": i + 1})
            if not e.get("replacement_admitted"):
                cells_lost[m["set"]].append(ck)
    mb = git_show(*M1[:2])
    if hashlib.sha256(mb).hexdigest() != M1[2]:
        raise SystemExit("M1 sha256")
    sp = json.loads(mb)["extensions"]["S_paper493"]
    added = []
    for x in sp["pool_exclusions"]:  # sim-honest worlds of S_paper493 (sp493_new / sp493_t196); errors only (the paired-pool rule is not a failure)
        w, c = x["key"].split("|", 1)
        if not w.startswith("sp493_") or "error" not in x["class"] or any(c.split("_", 1)[0] == k.split("|", 1)[1].split("_", 1)[0]
                                                                          for v in sf.values() for k in [r["cell"] for r in v]):
            continue
        att = sorted(rk for rk, (mn, cc) in srows["S_paper493"].items() if cc == c)
        if len(att) != 1:
            raise SystemExit(f"{x['key']}: {len(att)} launch rows (want 1)")
        sf["S_paper493"].append({"cell": x["key"], "category": sim_category("", x["class"]), "class": x["class"], "attempt": 1,
                                 "source": "M1 extensions.S_paper493.pool_exclusions (not in v3's like_for_like)"})
        cells_lost["S_paper493"].append(x["key"])
        added.append(x["key"])
    gates["sim_failures_added_from_M1"] = added
    gates["sim_like_for_like_entries"] = {"n": sum(len(v["keys"]) for k, v in v3["v3"]["like_for_like"].items() if SIM_ENTRY.match(k)), "want": 25}
    gates["sim_failed_attempts"] = {"n": sum(len(v) for v in sf.values()), "cells": len(sim_cells),
                                    "cells_with_two_failed_attempts": sorted(c for c, x in sim_cells.items() if x["failed_attempts"] == 2)}
    if gates["sim_like_for_like_entries"]["n"] != 25:
        raise SystemExit("sim like-for-like entries != 25")
    # ---------------- pool-level loss (0d item 5): cells whose pool lost the run (no member / no admitted replacement)
    allow = json.loads(git_show(*ALLOW))
    members = collections.defaultdict(set)
    for r in allow["rows"]:
        members[r["pool"]].add(r["row_id"])
    sp_members = {"honest_b": set(sp["pools"]["H_B"]) - {x["key"] for x in sp["pool_exclusions"]},
                  "honest_a": None}  # S_paper493 H_A = honest_a on H_sim(P)'s cells by key: a failed honest_a whose key is a pool exclusion is lost
    sp_excl = {x["key"] for x in sp["pool_exclusions"]}
    world_of_launch = {}  # allowlist: each launch's logical world (S_old's honest_c runs sit under arm dir honest_b); one world per launch and arm dir
    for r in allow["rows"]:
        if not r["pool"].startswith("S_old_"):  # only S_old stores two logical worlds (B, C) under one arm dir
            continue
        lw = (r["path"].split("/", 1)[0], r["path"].split("/")[1])
        if world_of_launch.setdefault(lw, r["row_id"].split("|", 1)[0]) != r["row_id"].split("|", 1)[0]:
            raise SystemExit(f"allowlist: launch {lw} holds two worlds")
    real_lost = collections.defaultdict(list)
    for (st, arm), runs in rf.items():
        for r in runs:
            task = E[r["key"]]["task"]
            parts_ = r["key"].split("/")
            rid = f"{world_of_launch.get((parts_[0], parts_[1]), arm)}|{cell_of(task)}"  # logical identity (honest_c under honest_b)
            if st in ("S_old", "S_new", "S_newmains"):
                pool = f"{st}_{'H_A' if arm == 'honest_a' else 'H_B'}" if not (st == "S_old" and arm == "honest_a") else None
                lost = pool is not None and rid not in members[pool]
            elif st == "S_paper493":
                lost = rid in sp_excl or (arm == "honest_b" and rid not in sp_members["honest_b"])
            elif st == "ep2":
                lost = False  # the 10 are earlier attempts whose redraw is the admitted member (v3 honest_earlier_attempt.redraw_key)
                if not E[r["key"]].get("redraw_key"):
                    raise SystemExit(f"{r['key']}: ep2 failure without a redraw key")
            else:
                raise SystemExit(f"real set not mapped: {st}")
            if lost:
                real_lost[(st, arm)].append(rid)
    # ---------------- tables
    cats = sorted({r["category"] for v in rf.values() for r in v} | {r["category"] for v in sf.values() for r in v})
    by_cat = lambda runs: {c: sum(1 for r in runs if r["category"] == c) for c in cats}  # noqa: E731
    real_rows = {f"{s}|{a}": rate(len(rf[(s, a)]), n) | {"by_category": by_cat(rf[(s, a)]), "pool_cells_lost": sorted(set(real_lost[(s, a)]))}
                 for (s, a), n in sorted(launched.items()) if not (s == "S_old" and a == "honest_a")}
    sim_rows = {f"{s}|sim_honest": rate(len(sf[s]), n) | {"by_category": by_cat(sf[s]), "pool_cells_lost": sorted(cells_lost[s]),
                                                          "cells_launched": len({c for _, c in srows[s].values()})}
                for s, n in sorted(sl.items())}
    agg = lambda rows, sets: rate(sum(r["failed"] for k, r in rows.items() if k.split("|")[0] in sets),  # noqa: E731
                                  sum(r["launched"] for k, r in rows.items() if k.split("|")[0] in sets))
    agg_cat = lambda rows, sets: {c: sum(r["by_category"][c] for k, r in rows.items() if k.split("|")[0] in sets) for c in cats}  # noqa: E731
    gap = v3["v3"]["gap_redraws_with_no_earlier_attempt_located"]
    idx = collections.Counter(int(re.search(r"_redraw(\d+)/", k).group(1)) for k in gap["keys"])
    unseen = sum(i * n for i, n in idx.items())
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"label": "Appendix A failure RATES (09; 0d recheck items 1-7): failed / launched honest RUNS per world x set x arm; keys and counts only; "
                    "POST HOC; PRELIMINARY until 0d's recheck",
           "definitions": {"failure": __doc__.split("FAILURE (both sides): ")[1].split("\nCATEGORIES")[0].replace("\n", " "),
                           "categories": __doc__.split("CATEGORIES (both sides; 0d item 3): ")[1].split("\nREAL is")[0].replace("\n", " "),
                           "unit": "runs (attempts) on both sides: real = honest_a and honest_b runs; sim = sim_honest runs (failed attempts per cell from its manifest rows)"},
           "headline_common_sets": {"sets": list(COMMON),
                                    "X_real": agg(real_rows, COMMON) | {"by_category": agg_cat(real_rows, COMMON), "note": "lower bound (see real_lower_bound)"},
                                    "Y_sim": agg(sim_rows, COMMON) | {"by_category": agg_cat(sim_rows, COMMON)},
                                    "pool_level_loss": {"real_cells_lost": sum(len(set(r["pool_cells_lost"])) for k, r in real_rows.items() if k.split("|")[0] in COMMON),
                                                        "sim_cells_lost": sum(len(r["pool_cells_lost"]) for k, r in sim_rows.items() if k.split("|")[0] in COMMON),
                                                        "rule": "a failed run's cell is lost when its pool has no member for it (real: allowlist / M1's S_paper493 "
                                                                "block; sim: no admitted replacement)"}},
           "other_sets_apart": {"real": {k: v for k, v in real_rows.items() if k.split("|")[0] not in COMMON},
                                "sim": {k: v for k, v in sim_rows.items() if k.split("|")[0] not in COMMON},
                                "why": "only S_new, S_newmains and S_paper493 have both sides (0d item 7); S_old's sim count is partial (A25 primary_v3 "
                                       "predates the chain manifests: only its q4b eval-upload rows and the A25 fill-in are counted)"},
           "real": {"by_set_arm": real_rows, "failed_runs": {f"{s}|{a}": v for (s, a), v in sorted(rf.items())},
                    "stopped_by_manifest_only": stopped,
                    "S_old_honest_a_apart": {"launched": launched[("S_old", "honest_a")], "named_failures_by_pool_rule": pool_rule,
                                             "why": "S_old has no honest_a pool (H_A = honest_c runs of H_B)"}},
           "real_lower_bound": {"statement": (f"real is a LOWER BOUND: {gap['n']} redraws have no earlier attempt located ({dict(sorted(idx.items()))} by "
                                              f"redraw index), so at least {unseen} earlier attempts are unseen; "
                                              f"{v3['v3']['earlier_attempts_located']} were located and are counted as launched and failed"),
                                "unseen_attempts_at_least": unseen, "gap_keys": gap["keys"]},
           "sim": {"by_set_arm": sim_rows, "failed_attempts": {s: v for s, v in sorted(sf.items())}, "cells": sim_cells, "not_failures": s_not,
                   "launch_source": {"commit": CHAIN[0], "manifests_rows": read_, "rule": "sim_honest rows of d9's counted manifests (TEST, _dry and "
                                     "sample_error_v1 excluded), deduplicated by (launch, rel)"}},
           "gates": gates,
           "sources": {"v3": {"commit": V3[0], "path": V3[1], "sha256": h}, "chain_commit": CHAIN[0],
                       "allowlist": {"commit": ALLOW[0], "path": ALLOW[1], "sha256": hashlib.sha256(git_show(*ALLOW)).hexdigest()},
                       "M1": {"commit": M1[0], "path": M1[1], "sha256": M1[2]}},
           "meta": {"stamp_pt": stamp, "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
                    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    out = HERE / "results" / f"appendix_a_rates_{stamp}.json"
    blob = json.dumps(rec, indent=1).encode()
    with out.open("xb") as fh:
        fh.write(blob)
    print(json.dumps({"file": str(out.relative_to(REPO)), "sha256": hashlib.sha256(blob).hexdigest(), "gates": gates}, indent=1))


if __name__ == "__main__":
    main()
