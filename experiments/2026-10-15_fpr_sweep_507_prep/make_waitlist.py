"""Waitlist of in-flight S_paper493 cells for the second-refresh final (09 RULE @ 07ad588be, amendment to the cut-off rule §7): the final WAITS
for every listed cell; a cell leaves the list only when every headline probe has scored it on every arm the metric uses (run1, run2, sim attack
arms), or when it terminally failed and is classed in the exclusions file. Slowness never resolves a cell. KEYS AND COUNTS ONLY (no score is read).

generate: from COMMITTED routing state at one chain commit (d9's aswin/q3-overnight-chain; every source pinned commit + path + sha256):
  - universe       the 493 S_paper493 cells (facts file); per cell and attack world (run1 = <box>_main attack, run2 = <box>_run2 attack, sim)
  - staged         keys in d9's routing records (results/hygiene/sp493_routing_*.json; per staged-set lineage, its highest _try only)
  - terminal       keys excluded/refused WITH a sample_error (routing records, manifests), a registered exclusion (manifest excluded_keys), or
                   a stager refusal with no re-stage registered (a registered gate): the CELL is out (already failed; one arm suffices), listed
                   under terminal_cells, never waitlisted; check STOPs unless the exclusions file classes every terminal cell
  - waitlisted     every other cell: re-stage pending (manifest index_donors.fallback cells not staged yet), held set (a manifest with no
                   routing record: b_run2), sim unrouted (X/Y tail groups' not-admitted cells), staged but scoring unconfirmed (no --census
                   given, or the census lacks a headline row), and anything unaccounted (fail closed)
  python make_waitlist.py generate --out results/waitlist_<stamp>.json [--chain-ref <commit>] [--census <path>@<sha256>]
      [--authority <commit>:<path>@<sha256>]   (d9's routing authority: its S_paper493 batches' staged sets must equal the lineage, else STOP)

generate-full: the full-length (uncut) waitlist over all four origins: every no-filter cohort cell (--cohort: origins.<O>.headline_cells; S_paper493
from the committed 32k waitlist until the cohort file carries it) waits on run1, run2, sim, plus sim2 / run3 for the IL (B) cells (--il-scope,
extra arms REPORT-ONLY by default, 0d/09), until the UNCUT census (meta.full_length) has rows in all FULL_SIX labels x C/R. States from INPUTS_FULL's pins:
uncut inventory pending, uncut override pending, or uncut scoring unconfirmed (per-run uncut folder from the inventories; nm360_s2_run2_s1 noted).
  python make_waitlist.py generate-full --out results/waitlist_full_<stamp>.json --cohort <c>:<p>@<sha> --inputs-full <c>:<p>@<sha>
      --il-scope <c>:<p>@<sha> (CUT's il_scope_v2, or il_five_arm_cells v2) [--ilarms <c>:<p>@<sha> (uncut)] [--sp493-waitlist <w>@<sha>]

check (build time; exit 1 = STOP; every origin in the waitlist, or only --origin): every waitlisted cell resolved by the census (rows in every headline label C and R, source S_paper493, on
every attack world; a structural-zero run counts as scored) or by a terminal class in the exclusions file (origins.S_paper493
.measurement_excluded; only an affirmative terminal class resolves: TERMINAL and not NOT_TERMINAL below). Prints {"pending": [...]} for meta.waitlist_pending.
  python make_waitlist.py check --waitlist results/<w>.json@<sha256> --exclusions results/<x>.json@<sha256> [--census results/<c>.json@<sha256>]
      [--origin S_paper493] [--probes "sol_a_uncut,FIBS|uncut,..."] [--readouts C,R]   (full-length build: its own labels and uncut census)
      [--final]   (required iff the exclusions file's label/status does not say TEST; "zero-row, class pending 09" cells pass in TEST, STOP in final)
"""

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RULE = "09 RULE @ 07ad588be"
CHAIN = "experiments/2026-10-14_q3_overnight_chain"
CHAIN_BRANCH = "origin/aswin/q3-overnight-chain"
FACTS = ("d800edc6a", "experiments/2026-10-15_q3_sp493/results/sp493_facts_2026-10-15_165730.json")
HEADLINE = {"probes": ["Sol (a)", "FIBS", "harmfulness", "refusal", "high-stakes", "own_all_rs_L48"], "readouts": ["C", "R"],
            "source": "07ad588be:experiments/2026-10-14_priority1_spec_09/RULE.md l.4402-4403 (six-probe signed S; own_all_rs = amendment 3/3a)"}
SCORER = {"Sol (a)": "sol (sol/a_tight__*)", "FIBS": "p2 (p2/fibs__*)", "harmfulness": "p2 (p2/harmfulness__*)", "refusal": "p2 (p2/refusal__*)",
          "high-stakes": "p2 (p2/highstakes__*)", "own_all_rs_L48": "own_all_rs (own_all_rs/own_all_rs_L48__*)"}
WORLDS = ("run1", "run2", "sim")
EXTRA_WORLDS = ("sim2", "run3")  # the IL (B) extra arms (09 @ 36b9b47df §3/§8; the full fork 03:0x)
SETS = ("S_old", "S_new", "S_newmains", "S_paper493")
FULL_SIX = ("sol_a_uncut", "FIBS|uncut", "harmfulness|uncut", "refusal|uncut", "high-stakes|uncut", "own_all_rs_uncut_L48")
FULL_SIX_SOURCE = "full_override.py @ 7b059dc6c HEADLINE_SIX (labels '<probe> <C|R>' as the full fork's census writes them)"
TASK_SET = {"S_old": "T196", "S_newmains": "new mains"}  # inventory_b_v2 rows carry both sets; S_new's inventory v3 is S_new only
IL_KEYS = {"S_old": "S_old_five_arm", "S_new": "S_new_on_M", "S_newmains": "S_newmains_five_arm"}  # il_five_arm_cells v2 @ 668a3eb91
NM360_S2_RUN2_S1 = "nm360_s2_run2_s1_cutfull"
NM360_NOTE = "uncut root re-extracted; writer attestation passed (r3 @ 8fb551db944f, pointer 41c7df8ce476); CUT decides its inventory pin"
EXTRA_NOTE = ("IL extra arm (B), REPORT-ONLY (0d/09 03:3x): never gates the freeze; included by availability at the freeze, never by value; "
              "a failed or unbuilt extra run leaves that task's (B) mean only, never the cohort")
# a class resolves a cell only when it STARTS with a canonical terminal root cause (a sample-level error or a registered gate's refusal; 09 §7);
# anything else waits (Codex r1). "no usable donor" never resolves a WAITLISTED cell: it is the pre-re-stage state of the index-donor cells.
TERMINAL = re.compile(r"^(?:setup failure|sim never launched|never launched|no run for an arm|errored run|sample error[:\s(]+\(?(?:EPCE|[A-Za-z]+Error)|"
                      r"EmptyPolicyCompletionError|"
                      r"EPCE|OutputContractError|OutputContract|approach-assert|malformed call|staging exclusion|stager refusal|"
                      r"registered exclusion|excluded by key|no usable (?:honest_a )?reference|registered gate refusal|attempt timeout)(?=$|[\s(:;,)])", re.IGNORECASE)  # class HEADS + separator
TIME_LIMIT = re.compile(r"^time-limit end \(7,200 s\); messages not saved, so it can't be scored(?=$|[\s(:;,)])", re.IGNORECASE)  # t706, 09 10-16
NOT_TERMINAL = re.compile(r"recoverable|cut-off|pending|not scored|no scored row|slow|not yet|in flight|no usable donor|\bnull\b|\bnone\b|\bn/?a\b|"
                          r"\bunknown\b|\bfalse\b|\bno error\b|\bnot\b|\babsent\b|\bno\W*$|\bexpected\b|\bwould\b|\bmaybe\b|\?",
                          re.IGNORECASE)  # negations and hedges wait; the exclusions file itself is script-generated and reviewed (this is a guard)


PENDING_09 = "zero-row, class pending 09"  # 507-prep 02:5x: exact label for zero-row runs awaiting 09's class
PENDING_LABELS = (PENDING_09, "zero-row, class pending the eligible-unit count")  # 09 @ 3685810b4 item 11 renamed it; both recognised exactly
OTHER = re.compile(r"^(?:other|attempt timeout) \(cause unknown\)(?=$|[\s(:;,)])", re.IGNORECASE)  # 09 @ 806da4823 / 3685810b4 (t899, nm290)


def is_terminal(cls):
    if not isinstance(cls, str):
        return False
    m = OTHER.match(cls)
    if m:  # terminal, non-structural; the rest of the class must not hedge
        return not NOT_TERMINAL.search(cls[m.end():])
    m = TIME_LIMIT.match(cls)
    if m:  # t706 (09): only this exact negation is allowed after the head; anything else after it must not hedge
        return not NOT_TERMINAL.search(cls[m.end():])
    return bool(TERMINAL.search(cls)) and not NOT_TERMINAL.search(cls)


TRY = re.compile(r"^(?P<base>.+?)_v1(?:_try(?P<n>[0-9]+))?$")
BOX = re.compile(r"^sp493_(?P<box>[abc])_(?P<arm>main|run2)$")


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, check=True, cwd=REPO).stdout


def pin(c, p):
    b = git("show", f"{c}:{p}")
    return {"commit": c, "path": p, "sha256": hashlib.sha256(b).hexdigest()}, json.loads(b)


def load_ref(ref):
    """<path>@<sha256> (path relative to this folder): bytes sha-asserted."""
    p, h = ref.rsplit("@", 1)
    b = (HERE / p).read_bytes()
    if hashlib.sha256(b).hexdigest() != h:
        raise SystemExit(f"STOP: {p}: sha256 differs from the pin {h}")
    return json.loads(b)


def cell_of(task):
    return re.sub(r"(?:_redraw[0-9]+|_r2)+$", "", task)  # run2 marker and redraw attempts (uncut inventories: t571_auto_workflow_redraw1)


def census_rows(census, source="S_paper493"):
    """{label: {world: {cell: n}}} over one census source (any origin row of the label)."""
    out = {}
    for lb, by_o in census["rows"].items():
        for v in by_o.values():
            for w, cells in v.get("attack_rows_by_source_world_cell", {}).get(source, {}).items():
                if "|sentinel" in w:
                    continue
                d = out.setdefault(lb, {}).setdefault(w, {})
                for c, n in cells.items():
                    d[c] = d.get(c, 0) + n
    return out


def labels_of(probes=None, readouts=None):
    return [f"{p} {r}" for p in (probes or HEADLINE["probes"]) for r in (readouts or HEADLINE["readouts"])]


def scored_missing(cell, worlds, rows, szero, labels=None):
    """(scorer x arm) pairs without a row for this cell; a structural-zero run (no rows in any label, no sample error) counts as scored."""
    miss = []
    for w in worlds:
        if f"{w}|{cell}" in szero:
            continue
        for lb in labels or labels_of():
            if rows.get(lb, {}).get(w, {}).get(cell, 0) < 1:
                miss.append(f"{lb} x {w}")
    return miss


def generate(a):
    ref = git("rev-parse", a.chain_ref).decode().strip()
    src = {}
    src["facts"], facts = pin(*FACTS)
    U = sorted(r["key"] for r in facts["rows"])
    assert len(U) == len(set(U)) == 493, "facts: not 493 distinct cells"
    names = git("ls-tree", "--name-only", ref, f"{CHAIN}/results/hygiene/").decode().split()
    recs = sorted(n for n in names if re.fullmatch(rf"{CHAIN}/results/hygiene/sp493_routing_[a-z0-9_]+_2026-[0-9-]+_[a-z]\.json", n))
    lineage = {}  # staged-set base -> (try, set dict, record name)
    auth = None
    if a.authority:  # d9's routing authority (authoritative root per logical batch): every S_paper493 batch's staged set must equal ours
        loc, h_ = a.authority.rsplit("@", 1)
        c_, p_ = loc.split(":", 1)
        src["routing_authority"], auth = pin(c_, p_)
        if src["routing_authority"]["sha256"] != h_:
            raise SystemExit("STOP: routing authority sha256 differs from the pin")
    for n in recs:
        src[f"routing:{Path(n).name}"], r = pin(ref, n)
        for s in r["sets"]:
            m = TRY.match(s["staged_set"])
            assert m, s["staged_set"]
            t = int(m["n"] or 0)
            if m["base"] in lineage and lineage[m["base"]][0] == t:  # cumulative records (tail _s2 repeats _s1's sets): identical content only
                o = lineage[m["base"]][1]
                if (o["staged"], o["excluded"]) != (s["staged"], s["excluded"]):
                    raise SystemExit(f"STOP: {s['staged_set']} routed in two records with different keys")
                continue
            if t >= lineage.get(m["base"], (-1,))[0]:
                lineage[m["base"]] = (t, s, Path(n).name)
    if auth is not None:
        A_ = {}
        for k_, v_ in auth["batches"].items():
            if k_.startswith("sp493-"):
                m_ = TRY.match(v_["staged_set"])
                assert m_, v_["staged_set"]
                A_[m_["base"]] = v_["staged_set"]
        ours = {b_: s_["staged_set"] for b_, (_, s_, _) in lineage.items() if b_.startswith("sp493_")}
        if A_ != ours:
            diff = sorted(set(A_.items()) ^ set(ours.items()))
            raise SystemExit(f"STOP: routing authority vs routing records (base, staged set) differ: {diff}")
    state = {}  # world|cell -> (state, detail, source)
    sim_group = {}

    def put(k, st, det, so, over=False):
        if k in state and not over:
            raise SystemExit(f"STOP: {k} has two states: {state[k]} / {(st, det, so)}")
        state[k] = (st, det, so)

    for base, (t, s, rn) in sorted(lineage.items()):
        mb = BOX.match(base)
        for k in s["staged"]:
            arm, task = k.split("|", 1)
            if arm.startswith("sim_attack"):
                w = "sim"
            elif mb and arm == "attack":
                w = "run1" if mb["arm"] == "main" else "run2"
            else:
                continue  # honest arms: pool keys (refresh2's pool-key gates), not cells
            k_ = f"{w}|{cell_of(task)}"  # a staging in another set (a later snapshot or re-stage) overrides that set's exclusion; two stagings STOP
            put(k_, "staged", s["staged_set"], rn, over=state.get(k_, ("",))[0] in ("terminal_seen", "stager_refusal"))
            if w == "sim":
                gm = re.match(r"sp493_(g[0-9])_", s["staged_set"])
                sim_group[cell_of(task)] = gm[1] if gm else s["staged_set"]
        for e in s["excluded"]:
            arm, task = e["key"].split("|", 1)
            w = "sim" if arm.startswith("sim_attack") else ("run1" if mb and mb["arm"] == "main" else "run2") if mb and arm == "attack" else None
            k_ = f"{w}|{cell_of(task)}" if w else None
            if not w or state.get(k_, ("",))[0] == "staged":
                continue  # honest rows are pool keys; a key staged in another set stays staged
            if e.get("sample_error"):
                put(k_, "terminal_seen", f"{e['sample_error']} ({e['staging_reason']})", rn, over=k_ in state)
            else:
                put(k_, "stager_refusal", f"stager refusal, a registered gate (no sample error): {e['staging_reason']}", rn, over=k_ in state)
    # real-arm manifests at the chain commit: re-stage fallback cells, registered exclusions, refusals, held sets (no routing record)
    for box in "abc":
        for arm in ("main", "run2"):
            p = f"{CHAIN}/config/manifest_sp493_{box}_{arm}.json"
            src[f"manifest:{box}_{arm}"], m = pin(ref, p)
            w = "run1" if arm == "main" else "run2"
            routed = any(BOX.match(b) and BOX.match(b)["box"] == box and BOX.match(b)["arm"] == arm for b in lineage)
            refused = {r_["key"]: r_["class"] for r_ in m.get("sample_error_rule", {}).get("refused", [])}
            excl = set(m.get("excluded_keys", {}).get("keys", []))
            fb = set((m.get("index_donors", {}).get("fallback") or {}).get("cells", []))
            for ak, why in [(x, f"{c_} (manifest sample_error refusal)") for x, c_ in refused.items()] + [(x, "registered exclusion (manifest excluded_keys)") for x in sorted(excl)]:
                arm_, task_ = ak.split("|", 1)
                k = f"{w}|{cell_of(task_)}"
                if arm_ != "attack" or state.get(k, ("",))[0] == "staged":
                    continue  # honest rows are pool keys; a staged key (a later clean attempt) stays staged
                put(k, "terminal_seen", why, p, over=True)
            for f in m["files"]:
                if f["arm_dir"] != "attack" or f.get("attempt_kind") == "reference":
                    continue
                k = f"{w}|{cell_of(f['task'])}"
                if state.get(k, ("",))[0] in ("staged", "terminal_seen"):
                    continue  # (a stager_refusal key falls through: a registered fallback re-stage makes it restage_pending)
                if not routed:
                    put(k, "held_set_pending", f"sp493_{box}_{arm}: manifest written, no routing record", p, over=k in state)
                elif cell_of(f["task"]) in fb:
                    put(k, "restage_pending", f"index-donor fallback (manifest index_donors.fallback) for sp493_{box}_{arm}", p, over=True)
    # sim: the X/Y tail groups' universe (d9's asserted record) and its not-admitted cells
    tails = [n for n in recs if "tail_xy" in n]
    tail_cells = {}
    for n in tails:
        _, r = pin(ref, n)
        for g, v in r["tail_universe"]["groups"].items():
            for c in v["not_admitted_here"]:
                tail_cells.setdefault(c, g)
    for c, g in sorted(tail_cells.items()):
        k = f"sim|{c}"
        if k not in state:
            put(k, "sim_unrouted", f"X/Y tail {g}: not admitted in any routed snapshot", "tail_universe")
    # staged attack runs with 0 tool calls (d9's per-set tool-error files): no rows in any label, so check waits on them unless the exclusions
    # file lists them as structural zeros (refresh2 checks sample_error null first); listed apart for that, keys only
    chosen = {s_["staged_set"] for (_, s_, _) in lineage.values()}
    no_row = []
    for n in sorted(x for x in names if re.fullmatch(rf"{CHAIN}/results/hygiene/toolerr_sp493_[A-Za-z0-9_-]+\.json", x)):
        pn, d = pin(ref, n)
        ss = d["staged_set"].split("/inputs/", 1)[0]
        if ss not in chosen:
            continue
        for r_ in d["rows"]:
            wk, c_ = r_["key"].split("|", 1)
            w_ = "sim" if wk == "sp493_sim" else wk
            k_ = f"{w_}|{cell_of(c_)}"
            if w_ in WORLDS and r_["n_tool_calls"] == 0 and state.get(k_, ("",))[0] == "staged":
                src[f"toolerr:{Path(n).name}"] = pn
                no_row.append({"key": k_, "staged_set": ss, "n_tool_calls": 0, "source": f"{pn['commit'][:12]}:{n}@{pn['sha256']}"})
    # every remaining universe key: staged keys need scoring; everything else is unaccounted (fail closed)
    rows, szero = {}, set()
    if a.census:
        cen = load_ref(a.census)
        src["census"] = {"path": a.census.rsplit("@", 1)[0], "sha256": a.census.rsplit("@", 1)[1]}
        rows = census_rows(cen)
    wl, terminal = {}, {}
    for c in U:
        for w in WORLDS:
            k = f"{w}|{c}"
            st, det, so = state.get(k, ("unaccounted", "no routing record, manifest or tail list names this key", "-"))
            if st in ("terminal_seen", "stager_refusal"):  # stager_refusal: no re-stage registered for it (else restage_pending)
                terminal[k] = {"state": st, "detail": det, "source": so}
                continue
            if st == "staged":
                miss = scored_missing(c, [w], rows, szero) if rows else ["census not given"]
                if not miss:
                    continue
                st, det = "staged_scoring_unconfirmed", f"{det}; missing: {len(miss)} label rows" if rows else f"{det}; no census given"
            wl.setdefault(c, {})[w] = {"state": st, "detail": det, "source": so}
    tcells = {}  # one terminally failed arm takes the whole cell out of the cohort: not in flight; check asserts the exclusions file classes it
    for k, v in terminal.items():
        w, c = k.split("|", 1)
        tcells.setdefault(c, {})[w] = v
    for c in tcells:
        wl.pop(c, None)
    pairs = labels_of()
    cells = [{"origin": "S_paper493", "cell": c, "sim_group": sim_group.get(c), "arms": sorted(v), "needs": {w: pairs for w in sorted(v)},
              "state_by_arm": v} for c, v in sorted(wl.items())]
    by = {}
    for x in cells:
        for w, v in x["state_by_arm"].items():
            by.setdefault(v["state"], {}).setdefault(w, []).append(x["cell"])
    out = {"what": "waitlist of in-flight S_paper493 cells (keys only)", "rule": RULE, "generated_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S %Z"),
           "chain_ref": ref, "headline": HEADLINE, "scorer_of_probe": SCORER, "arms_rule": "every waitlisted cell needs rows in every headline label (C and R) on "
           "every attack world run1, run2, sim; honest arms are pool keys (refresh2's pool-key gates), not waitlist cells",
           "sources": src, "counts": {"universe_cells": len(U), "waitlist_cells": len(cells), "by_state": {s: {w: len(v) for w, v in d.items()} for s, d in sorted(by.items())},
                                      "terminal_keys": len(terminal), "terminal_cells": len(tcells),
                                      "no_row_runs": len(no_row)},
           "dependency": {s: {w: sorted(v) for w, v in d.items()} for s, d in sorted(by.items()) if s != "staged_scoring_unconfirmed"},
           "staged_scoring_unconfirmed_by_set": {}, "terminal_cells": tcells, "no_row_runs": no_row, "waitlist": cells, "pending": [x["cell"] for x in cells]}
    for x in cells:
        for w, v in x["state_by_arm"].items():
            if v["state"] == "staged_scoring_unconfirmed":
                s_ = v["detail"].split(";", 1)[0]
                out["staged_scoring_unconfirmed_by_set"][s_] = out["staged_scoring_unconfirmed_by_set"].get(s_, 0) + 1
    dest = HERE / a.out
    blob = json.dumps(out, indent=1).encode()
    with dest.open("xb") as fh:
        fh.write(blob)
    print(json.dumps({"file": a.out, "sha256": hashlib.sha256(blob).hexdigest(), "chain_ref": ref} | out["counts"]
                     | {"staged_scoring_unconfirmed_by_set": out["staged_scoring_unconfirmed_by_set"]}, indent=1))


def pinref(ref):
    """<commit>:<path>@<sha256> -> (pin, json); the committed bytes, sha-asserted."""
    loc, h = ref.rsplit("@", 1)
    c, p = loc.split(":", 1)
    pn, d = pin(c, p)
    if pn["sha256"] != h:
        raise SystemExit(f"STOP: {loc}: sha256 differs from the pin {h}")
    return pn, d


def generate_full(a):
    """Full-length (uncut) waitlist over all four origins (09 RULE @ 07ad588be item 2; the full fork's inputs, 03:0x): every no-filter cohort cell
    waits until the UNCUT census has rows in every FULL_SIX label (C and R) on run1, run2, sim, and on sim2 / run3 for the IL (B) cells
    (extra arms report-only by default). States come from INPUTS_FULL's override and inventory pins and the per-run uncut inventories."""
    src = {}
    src["cohort"], coh = pinref(a.cohort)
    src["inputs_full"], IFU = pinref(a.inputs_full)
    cohort = {o: sorted(coh["origins"][o]["headline_cells"]) for o in SETS if o in coh.get("origins", {})}
    notes = []
    if "S_paper493" not in cohort:
        if not a.sp493_waitlist:
            raise SystemExit("STOP: the cohort file has no S_paper493 origin; give --sp493-waitlist (the committed 32k waitlist)")
        W32 = load_ref(a.sp493_waitlist)
        src["sp493_waitlist"] = {"path": a.sp493_waitlist.rsplit("@", 1)[0], "sha256": a.sp493_waitlist.rsplit("@", 1)[1]}
        cohort["S_paper493"] = sorted(x["cell"] for x in W32["waitlist"])
        notes.append("S_paper493 cohort = the 32k waitlist's non-terminal cells (until the cohort file carries S_paper493)")
    if set(cohort) != set(SETS):
        raise SystemExit(f"STOP: cohort origins {sorted(cohort)} != {list(SETS)}")
    planned, inv_state = {}, {}
    for o in SETS:  # an origin's inventories: "<O>" plus any wave parts "<O>_w<n>" (S_paper493: w1 + w23)
        ks = sorted(k for k in IFU["inventories"] if k == o or re.fullmatch(rf"{re.escape(o)}_w[0-9]+", k))
        es = [IFU["inventories"][k] or {} for k in ks]
        if not es or any("pending" in e or "path" not in e for e in es):
            inv_state[o] = "uncut inventory pending (" + "; ".join(str(e.get("pending", "not in INPUTS_FULL")) for e in es or [{}]) + ")"
            continue
        ts = TASK_SET.get(o)
        for k, e in zip(ks, es):
            src[f"inventory:{k}"], inv = pin(e["commit"], e["path"])
            if src[f"inventory:{k}"]["sha256"] != e["sha256"]:
                raise SystemExit(f"STOP: inventory {k}: sha256 differs from INPUTS_FULL's pin")
            for r in inv["runs"]:
                if r["world"] in WORLDS and (ts is None or r.get("task_set") == ts):
                    planned.setdefault(o, {})[(r["world"], cell_of(r["task"]))] = (r.get("uncut_planned") or {}).get("folder")
    il_routed = set()  # (origin, "<world>|<cell>"): IL extra-arm runs in INPUTS_FULL's uncut "il" inventory with an uncut folder
    e = IFU["inventories"].get("il") or {}
    if "path" in e:
        src["inventory:il"], inv = pin(e["commit"], e["path"])
        if src["inventory:il"]["sha256"] != e["sha256"]:
            raise SystemExit("STOP: inventory il: sha256 differs from INPUTS_FULL's pin")
        il_routed = {(r["origin"], f"{r['world']}|{r['cell']}") for r in inv["runs"] if r["world"] in EXTRA_WORLDS and (r.get("uncut_planned") or {}).get("folder")}
    ovp = {o: [f"{o}|{ro}: {IFU['overrides'][f'{o}|{ro}']['pending']}" for ro in "CR" if "pending" in IFU["overrides"].get(f"{o}|{ro}", {"pending": "absent"})]
           for o in SETS}
    src["il_scope"], ILS = pinref(a.il_scope)  # required: omitting it would drop every extra arm silently (Codex r16)
    il = {}  # origin -> cell -> admitted extra worlds
    if ILS.get("schema") == "ilarms/v1":  # the LOADER's block: admitted keys "<world>|<cell>" per set and folder (scope v4 r2 and later)
        for o, fs in ILS["sets"].items():
            for v in fs.values():
                for k in v["keys"]:
                    w, c = k.split("|", 1)
                    if w not in EXTRA_WORLDS or o not in SETS:
                        raise SystemExit(f"STOP: ilarms key {o}|{k}: world or set unknown")
                    if w in il.setdefault(o, {}).setdefault(c, set()):
                        raise SystemExit(f"STOP: ilarms admits two runs for {o}|{k}")
                    il[o][c].add(w)
    elif "rows" in ILS and "admitted_keys" in ILS:  # CUT's il_scope_v2: one canonical run per cell x extra arm, admitted rows by key
        for r in ILS["rows"]:
            if r.get("admitted"):
                if r["world"] not in EXTRA_WORLDS or r["set"] not in SETS:
                    raise SystemExit(f"STOP: il scope row {r['key']}: world {r['world']} / set {r['set']} unknown")
                if r["world"] in il.setdefault(r["set"], {}).setdefault(r["cell"], set()):
                    raise SystemExit(f"STOP: il scope admits two runs for {r['set']}|{r['world']}|{r['cell']}")
                il[r["set"]][r["cell"]].add(r["world"])
        if sum(len(v) for d_ in il.values() for v in d_.values()) != len(set(ILS["admitted_keys"])):
            raise SystemExit("STOP: il scope admitted rows != admitted_keys")
    else:  # il_five_arm_cells v2 (both extra arms per listed cell)
        for o, k in IL_KEYS.items():
            il[o] = {c: set(EXTRA_WORLDS) for c in ILS[k]["cells"]}
    routed = set(il_routed)  # (origin, "<world>|<cell>") with an uncut extra-arm folder (INPUTS_FULL's il inventory, or an uncut ilarms block)
    if a.ilarms:
        src["ilarms"], IB = pinref(a.ilarms)
        if IB.get("schema") != "ilarms/v1":
            raise SystemExit("STOP: --ilarms is not an ilarms/v1 block")
        if "32768" in json.dumps(IB.get("window")):
            raise SystemExit(f"STOP: the ilarms block's window {IB.get('window')!r} is the 32k cut; a full-length extra arm needs the uncut block")
        for o, fs in IB["sets"].items():
            for v in fs.values():  # optional_pending_keys: admitted, uncut files not routed yet (the loader's own definition): not routed
                routed |= {(o, k) for k in v["keys"] if k not in set(v.get("optional_pending_keys", []))}
    wl = []
    for o in SETS:
        for c in cohort[o]:
            st = {}
            for w in WORLDS:
                if o in inv_state:
                    st[w] = {"state": "uncut_inventory_pending", "detail": inv_state[o]}
                    continue
                fo = planned.get(o, {}).get((w, c), "absent")
                if fo == "absent":
                    st[w] = {"state": "unaccounted", "detail": "no uncut inventory row for this run (fail closed)"}
                elif ovp[o]:
                    st[w] = {"state": "uncut_override_pending", "detail": "; ".join(ovp[o]), "uncut_folder": fo}
                else:
                    st[w] = {"state": "uncut_scoring_unconfirmed", "detail": "override pinned; waits for the uncut census", "uncut_folder": fo}
                if fo and fo.startswith(NM360_S2_RUN2_S1):
                    st[w]["note"] = NM360_NOTE
            x = {"origin": o, "cell": c, "arms": list(WORLDS), "state_by_arm": st}
            ex = sorted(il.get(o, {}).get(c, ()))
            if ex:
                x["extra_arms"] = ex
                x["extra_state"] = {w: {"state": "routed" if (o, f"{w}|{c}") in routed else "in scope, not built", "detail": EXTRA_NOTE} for w in ex}
            wl.append(x)
    labels = labels_of(list(FULL_SIX), HEADLINE["readouts"])
    by = {}
    for x in wl:
        for w, v in list(x["state_by_arm"].items()) + list(x.get("extra_state", {}).items()):
            by.setdefault(x["origin"], {}).setdefault(v["state"], {}).setdefault(w, 0)
            by[x["origin"]][v["state"]][w] += 1
    out = {"what": "full-length (uncut) waitlist: every no-filter cohort cell of the four origins (keys only)", "rule": RULE, "mode": "full-length",
           "generated_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S %Z"), "labels": labels,
           "labels_source": FULL_SIX_SOURCE, "arms_rule": "run1, run2, sim on every cohort cell; sim2 and run3 on the IL (B) cells (extra_arms): the "
           "primary arms alone define the cohort; a terminally failed extra arm leaves that task's (B) mean only (classed under extra_arm_excluded)",
           "notes": notes, "sources": src, "counts": {"cells_by_origin": {o: len(cohort[o]) for o in SETS}, "il_b_cells_by_origin": {o: sum(1 for x in wl if x["origin"] == o and "extra_arms" in x) for o in SETS},
                                                      "il_scope_cells_outside_cohort": {o: len(set(il.get(o, {})) - set(cohort[o])) for o in SETS},
                                                      "extra_arms": {o: {w: {"admitted": sum(1 for x in wl if x["origin"] == o and w in x.get("extra_arms", [])),
                                                                             "routed": sum(1 for x in wl if x["origin"] == o and x.get("extra_state", {}).get(w, {}).get("state") == "routed"),
                                                                             "in_scope_not_built": sum(1 for x in wl if x["origin"] == o and x.get("extra_state", {}).get(w, {}).get("state") == "in scope, not built")}
                                                                         for w in EXTRA_WORLDS} for o in SETS},
                                                      "states": by}, "terminal_cells": {}, "waitlist": wl, "pending": [f"{x['origin']}|{x['cell']}" for x in wl]}
    blob = json.dumps(out, indent=1).encode()
    with (HERE / a.out).open("xb") as fh:
        fh.write(blob)
    print(json.dumps({"file": a.out, "sha256": hashlib.sha256(blob).hexdigest()} | out["counts"], indent=1))


def check_origin(W, X, O, labels, census, final, extra_gate):
    if O not in X.get("origins", {}):
        raise SystemExit(f"STOP: the exclusions file has no {O} origin")
    xo = X["origins"][O]
    cells = [x for x in W["waitlist"] if x.get("origin") == O]
    term = {c: v["class"] for c, v in xo.get("measurement_excluded", {}).items() if is_terminal(v.get("class"))}
    allx = {c: v.get("class") for c, v in xo.get("measurement_excluded", {}).items()}
    sz = xo.get("structural_zero_runs", {})
    # 09 has not ruled on these zero-row runs yet (exact label): never structural zeros; TEST passes them, --final STOPs on them
    szp = {k for k, v in (sz.items() if isinstance(sz, dict) else ()) if isinstance(v, dict) and v.get("class") in PENDING_LABELS}
    szero = set(sz) - szp
    p09 = {c for c, v in allx.items() if v in PENDING_LABELS} | {k.split("|", 1)[1] for k in szp if k.split("|", 1)[0] not in EXTRA_WORLDS}
    # extra arms (report-only by default, 0d/09): a pending label there is reported; it gates only with --extra-arms-gate
    p09x = {k for k, v in xo.get("extra_arm_excluded", {}).items() if v.get("class") in PENDING_LABELS} | {k for k in szp if k.split("|", 1)[0] in EXTRA_WORLDS}
    if extra_gate:
        p09 |= {k.split("|", 1)[1] for k in p09x}
    xarm = {k: v.get("class") for k, v in xo.get("extra_arm_excluded", {}).items()}  # "<sim2|run3>|<cell>" -> class (a failed IL extra run)
    rows = census_rows(census, O) if census is not None else {}
    tc = W["terminal_cells"]
    unclassed = sorted(c for c in tc if not is_terminal(allx.get(c)))
    if unclassed:
        raise SystemExit(f"STOP: terminal cells not classed terminal in the exclusions file: {unclassed}")
    t09 = sorted(c for c in tc if c in p09)  # a pending-09 mark on a terminal cell: conflicting records; never pass a final
    if t09 and final:
        raise SystemExit(f"STOP: terminal cells also marked pending ({list(PENDING_LABELS)}): {t09}")
    res, pending, xpend = {}, [], {}
    xc = {w: {"admitted": 0, "routed": 0, "scored": 0, "terminal": 0, "pending_label": 0, "not_scored": 0} for w in EXTRA_WORLDS}
    for x in cells:
        c = x["cell"]
        xm = []  # extra arms first: counted and reported for every waitlisted cell, before any primary-arm early return (Codex r19)
        for w in x.get("extra_arms", []):
            k = f"{w}|{c}"
            xc[w]["admitted"] += 1
            xc[w]["routed"] += x.get("extra_state", {}).get(w, {}).get("state") == "routed"
            if k in p09x:
                xc[w]["pending_label"] += 1
                xm.append(f"{k}: class in {list(PENDING_LABELS)}")
            elif is_terminal(xarm.get(k)):
                xc[w]["terminal"] += 1
            else:
                m_ = scored_missing(c, [w], rows, szero, labels) if rows else [f"no census given ({w})"]
                xc[w]["scored" if not m_ else "not_scored"] += 1
                xm += m_
        if xm:
            xpend[c] = xm
        if c in p09:
            res[c] = {"resolved": False if final else "pending_09_class (TEST pass)", "missing": [f"class in {list(PENDING_LABELS)}"]}
            if final:
                pending.append(c)
            continue
        if c in term:
            res[c] = {"resolved": "terminal", "class": term[c]}
            continue
        miss = scored_missing(c, x["arms"], rows, szero, labels) if rows else [f"no census given ({w})" for w in x["arms"]]
        if miss or (xm and extra_gate):
            pending.append(c)
            res[c] = {"resolved": False, "missing": miss + (xm if extra_gate else [])}
        else:
            res[c] = {"resolved": "scored"}
    return {"origin": O, "n_waitlist": len(cells), "resolved_scored": sum(v["resolved"] == "scored" for v in res.values()),
            "resolved_terminal": sum(v["resolved"] == "terminal" for v in res.values()), "terminal_cells_classed": len(tc),
            "pending_09_class": sorted(c for c in p09 if c in res or c in tc), "extra_arm_pending": sorted(xpend),
            "extra_arms": {w: v | {"in_scope_not_built": v["admitted"] - v["routed"]} for w, v in xc.items()}, "pending": pending,
            "unresolved": {c: res[c]["missing"][:12] + ([f"... {len(res[c]['missing']) - 12} more"] if len(res[c]["missing"]) > 12 else []) for c in pending}}


def check(a):
    W = load_ref(a.waitlist)
    if W.get("rule") != RULE:
        raise SystemExit(f"STOP: waitlist rule {W.get('rule')!r} != {RULE!r}")
    X = load_ref(a.exclusions)
    xl = " ".join(str(X.get(k, "")) for k in ("label", "status"))
    if a.final == ("TEST" in xl):  # the mode must agree with the exclusions file's own label: a TEST file never feeds a final, and vice versa
        raise SystemExit(f"STOP: exclusions label/status {xl.strip()!r} vs --final={a.final}: run TEST files without --final, final files with it")
    origins = sorted({x.get("origin") for x in W["waitlist"]}, key=str)
    if not origins:  # an empty waitlist still validates its terminal cells: name the origin
        if a.origin is None:
            raise SystemExit("STOP: the waitlist is empty; give --origin so its terminal cells are still checked")
        origins = [a.origin]
    if a.origin is not None and origins != [a.origin]:
        raise SystemExit(f"STOP: waitlist cells of origin {origins}, check runs for {a.origin}")
    if len(origins) > 1 and W["terminal_cells"]:
        raise SystemExit("STOP: a multi-origin waitlist carries terminal_cells without an origin")
    lists = {}
    for nm in ("probes", "readouts"):  # default ONLY when the option is omitted; an explicitly empty list STOPs
        v = getattr(a, nm)
        if v is not None:
            lists[nm] = [x.strip() for x in v.split(",")]
            if not lists[nm] or any(not x for x in lists[nm]):
                raise SystemExit(f"STOP: --{nm} {v!r} has an empty item")
    wl_p = list(dict.fromkeys(lb.rsplit(" ", 1)[0] for lb in W.get("labels", []))) or None  # omitted dimensions come from the waitlist's labels
    wl_r = list(dict.fromkeys(lb.rsplit(" ", 1)[1] for lb in W.get("labels", []))) or None
    labels = labels_of(lists.get("probes", wl_p), lists.get("readouts", wl_r))
    if W.get("labels") and labels != list(W["labels"]):
        raise SystemExit(f"STOP: --probes/--readouts give {labels}, the waitlist was built for {W['labels']}")
    census = load_ref(a.census) if a.census else None
    if W.get("mode") == "full-length" and census is not None and not (census.get("meta") or {}).get("full_length"):
        raise SystemExit("STOP: a full-length waitlist needs the uncut census (meta.full_length); a 32k census never resolves a full-length cell")
    per = [check_origin(W, X, O, labels, census, a.final, a.extra_arms_gate) for O in origins]
    pending = [p_ if len(origins) == 1 else f"{r['origin']}|{p_}" for r in per for p_ in r["pending"]]
    summ = {"waitlist": a.waitlist, "exclusions": a.exclusions, "census": a.census, "rule": RULE, "mode": "final" if a.final else "TEST",
            "labels": labels, "extra_arms_gate": a.extra_arms_gate, "origins": per,
            "pending_09_class": sorted(c for r in per for c in r["pending_09_class"]), "pending": pending}
    if len(origins) == 1:  # the 32k layout: one origin's fields at the top level
        summ |= {k: v for k, v in per[0].items() if k not in ("pending", "pending_09_class")}
    print(json.dumps(summ, indent=1))
    if pending:
        raise SystemExit(f"STOP: {len(pending)} waitlisted cells unresolved")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("generate")
    g.add_argument("--out", required=True)
    g.add_argument("--chain-ref", default=CHAIN_BRANCH)
    g.add_argument("--census")
    g.add_argument("--authority", help="<commit>:<path>@<sha256> of d9's routing authority; its S_paper493 staged sets must equal the records'")
    f = sp.add_parser("generate-full")
    f.add_argument("--out", required=True)
    f.add_argument("--cohort", required=True, help="<commit>:<path>@<sha256> with origins.<O>.headline_cells (no-filter list or the exclusions file)")
    f.add_argument("--inputs-full", required=True, help="<commit>:<path>@<sha256> of INPUTS_FULL (overrides and inventories pins)")
    f.add_argument("--il-scope", required=True, help="<commit>:<path>@<sha256> of il_five_arm_cells v2 (the IL (B) cells per origin)")
    f.add_argument("--ilarms", help="<commit>:<path>@<sha256> of the LOADER's UNCUT ilarms/v1 block (routed extra arms; a 32k block STOPs)")
    f.add_argument("--sp493-waitlist", help="<path>@<sha256> of the committed 32k waitlist (S_paper493 cohort when --cohort lacks it)")
    k = sp.add_parser("check")
    k.add_argument("--waitlist", required=True)
    k.add_argument("--exclusions", required=True)
    k.add_argument("--census")
    k.add_argument("--origin", help="check only this origin (the waitlist must hold no other); default: every origin in the waitlist")
    k.add_argument("--probes", help="comma-separated probe names (default: the six headline probes); e.g. the full-length six")
    k.add_argument("--readouts", help="comma-separated readouts (default C,R)")
    k.add_argument("--final", action="store_true", help="final build: 'zero-row, class pending 09' cells STOP (TEST: reported, pass)")
    k.add_argument("--extra-arms-gate", action="store_true", help="IL (B) extra arms gate too (default: report-only, 0d/09 03:3x; the cohort = primary arms)")
    a = ap.parse_args()
    {"generate": generate, "generate-full": generate_full, "check": check}[a.cmd](a)


if __name__ == "__main__":
    main()
