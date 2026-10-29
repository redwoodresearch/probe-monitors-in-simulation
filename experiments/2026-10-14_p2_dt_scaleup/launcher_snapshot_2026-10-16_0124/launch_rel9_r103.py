"""Launcher rel9: DT scoring of the sim-honest batches (09 Append 25h section 11 @ fd163c4c7; attestation gate @ 5aa9f32bf; RULE Amendment 31).
One command per batch: python3 -I launch_rel9.py <batch> [<root target> ...], batch in simh311_q1..q4 or simh_nm360_<g>. For every d9-routed root of
the batch (routing staged_set <batch>_sim[_s<j>]_v1[_try<n>]) not yet scored under /simh/<batch>/dt/, SERIALLY (never two launches at once):
verify by key (verify_roots.py: attestation failing {} with every check passing, the writer gate, repro, format, stage record = routing,
manifest counts, cells in the pinned list), cite every staging exclusion by key (09 @ 0f9927f3e item 6), check that no staged key repeats another
root of the batch or a scored folder, then one `run_dt_q4b.py --job arm` launch with its preflight.
Stops: (1) the scale-up line ($100, enforced again in main()), counted conservatively: every DT-app dollar billed in complete hours is
settled, every other DT app is unsettled at $2.70; (2) the rel9+ cumulative cap (4f / Aswin's day plan): over every app this launcher has
created, billed cost (complete hours) + $2.70 x (apps not yet fully billed + this launch) must stay <= $15, refused otherwise.
DRY=1: everything up to the launch, then main()'s local checks through dryrun_main.py (no app, nothing in results/, no ledger line).
PREPARE ONLY until 4f confirms Aswin's day-plan approval covers the batch."""
import collections
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import zoneinfo
import contextlib
import fcntl
from pathlib import Path

P2 = Path(__file__).resolve().parent
DRY = os.environ.get("DRY") == "1"
DTS = Path(os.environ.get("DT_WORKTREE", P2 / "dts"))  # the DT worktree whose HEAD (pushed, clean) is launched
LINE, STREAM = os.environ.get("DT_LINE", "rel9"), os.environ.get("DT_STREAM", "")  # Amendment 42: "full" = the full-length (cutfull) line
SHARDS = int(os.environ.get("DT_SCORE_SHARDS", "1"))  # Amendment 42: score_layer containers per layer (job --score-shards); 1 = unchanged
assert (LINE, STREAM) == ("rel9", "") or (LINE in ("full", "sp493") and STREAM in ("1", "2", "3")), "DT_LINE=rel9 (no DT_STREAM), or DT_LINE=full|sp493 with DT_STREAM in 1..3"
assert 1 <= SHARDS <= 8 and (SHARDS == 1 or LINE == "full"), "DT_SCORE_SHARDS: 1..8, and only on the full line"
L = P2 / (f"launch_rel9_dryrun_{datetime.datetime.now():%H%M%S}" if DRY else {"rel9": "launch_rel9", "full": "launch_full", "sp493": "launch_sp493"}[LINE])
LEDGER = P2 / "dt4f" / {"rel9": "rel9plus_ledger.jsonl", "full": "full_ledger.jsonl", "sp493": "sp493_ledger.jsonl"}[LINE]  # one line per launch try (append-only)
CAP, ARM_WORST, SU_STOP = {"rel9": 15.0, "full": 60.0, "sp493": 150.0}[LINE], 2.7, 100.0  # 42: full $60; 45: sp493 $150 RESERVED (Aswin in-session, 4f ~23:1x)
MAX_INFLIGHT, BILLED_TARGET = {"rel9": (1, None), "full": (3, 12.0), "sp493": (3, 6.5)}[LINE]  # 42/45: 3 parallel streams; billed target = quote + 30%, a flag only
SP493_LAST_DISPATCH = datetime.datetime.fromisoformat("2026-10-16T02:50:00-07:00")  # 45: Aswin's 03:00 PT hard stop, less a 10-minute job margin
CUTFULL = None  # 42: 09's cutfull staged-set grammar, OFF until 09 issues its tested regex (then re.compile(<verbatim>))
B, CF = "experiments/2026-10-14_q3_overnight_chain/results", "experiments/2026-10-14_q3_overnight_chain/config"
RT = Path("/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad/chain_run_state/routing")
MODAL, WVOL = "/Users/aswinsurya/.local/bin/modal", "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"
DT_APP = "probe-shift_aswin-surya_sub-p2probes_dtq4b_kill-2026-11-13"
BATCHES = {**{f"simh311_q{k}": "su311" for k in range(1, 5)}, **{f"simh_nm360_{g}": "nm360" for g in ("s1", "s2", "s3", "s4d", "s4r", "s5")}}
CELLS = {"nm360": ("ef171c813:experiments/2026-10-14_q3_new_mains_mode/config/new_mains_cells_v1.json", "98bfb97e26e7ec580b08643a3a38a5e66a0f6b63333f4013aa57559383402e8e"),
         "su311": ("0caadbfb5:experiments/2026-10-14_q2_scaleup_cohort/results/scaleup_tasks_plus311_2026-10-14_125135.json", "69f1f75313c9610f7d16b18225c1166268c0b6e084859c641d0213f293896a8b")}
# no RULED table: 09 @ 0f9927f3e item 6 (sim-honest staging exclusions are evidence-classified, no mandatory per-cell ruling)
FILL = json.loads((P2 / "dt4f" / "launch_rel5_fill.json").read_text())  # rule31 / spend_rel9 are added after their commits
git = lambda *a: subprocess.run(["git", "-C", str(DTS), *a], capture_output=True, check=True).stdout  # noqa: E731
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
now = lambda: datetime.datetime.now(zoneinfo.ZoneInfo("America/Los_Angeles"))  # noqa: E731
def sh(*a, tries: int = 10, timeout: int = 120) -> str:
    """A Modal CLI call with bounded retries (Codex 75: the laptop network drops today): each try has a timeout; after `tries` failures,
    stop for a human. Never returns partial output as success."""
    err = ""
    for k in range(tries):
        try:
            return subprocess.run(list(a), capture_output=True, text=True, check=True, timeout=timeout).stdout
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            err = (getattr(e, "stderr", "") or "")[-300:]
            time.sleep(min(30, 5 * (k + 1)))
    raise SystemExit(f"Modal CLI failed {tries} times: {' '.join(a[:4])} …: {err}")


UTC = datetime.timezone.utc
HOUR = datetime.timedelta(hours=1)
FINAL_AFTER = datetime.timedelta(hours=3)  # Codex 68: a billed hour is used only once it ended 3 h ago (billing posts 1-2 h late; no guarantee)


def _bill(start: str) -> list:
    return [x | {"_t": datetime.datetime.fromisoformat(x["interval_start"]).replace(tzinfo=UTC)}
            for x in json.loads(sh(MODAL, "billing", "report", "--start", start, "--resolution", "h", "--json"))]


RATE_CORE, RATE_GIB, PAD, WALL_PAD = 0.0000131, 0.00000222, 130, 60  # DT RULE "Cost" basis: $/core-s, $/GiB-s; +130 s start-up/scale-down per container
KINDS = {"simh_arm": (("arm", 1), ("gate_a", 1), ("score_layer", 16)),  # an honest-only arm job: no labelling parts; <= 16 per-layer containers (RULE Cost)
         "simh_primary": (("simh_primary", 1), ("gate_a", 1), ("score_layer", 16)),  # primary_v3 (Amendment 31): the same containers
         "relaunch": (("relaunch", 1), ("gate_a", 1), ("score_layer", 16)),  # honest only (Amendment 29)
         "arm_honest": (("arm", 1), ("gate_a", 1), ("score_layer", 16)),  # Amendment 39: a real honest-only arm job (ha25f): no labelling parts
         "arm_attack": (("arm", 1), ("gate_a", 1), ("score_layer", 16), ("label_part", 24)),  # an arm with attack epochs (RULE Cost table)
         "fold_check": (("check", 1),)}  # check_dt_unseen.py


def specs() -> dict:
    """Each Modal function's (cores, GiB, timeout s, retries) from the launched worktree's decorators (hard limits), both DT scripts."""
    out = {}
    for f in ("run_dt_q4b.py", "check_dt_unseen.py"):
        src = (DTS / "experiments/2026-10-14_p2_dt_scaleup" / f).read_text()
        for m in re.finditer(r"@app\.function\(([^)]*(?:\([^)]*\)[^)]*)*)\)\s*\ndef (\w+)\(", src):
            c, g, t, r = (re.search(x, m[1]) for x in (r"cpu=\(([0-9.]+), ([0-9.]+)\)", r"memory=\(([0-9]+), ([0-9]+)\)", r"timeout=([0-9]+)", r"retries=([0-9]+)"))
            if c and g and t and r:
                out[m[2]] = (float(c[2]), int(g[2]) / 1024, int(t[1]), int(r[1]))
    return out


def ceilings(kind: str) -> tuple:
    """(full ceiling $, $ per second with every container alive) of one app of this kind. Full: sum over its containers of count x
    (2 + retries) x (timeout + 130 s) x (cores x 0.0000131 + GiB x 0.00000222); the 2 allows one infrastructure restart of every container
    (Codex 69: Modal restarts outside `retries`). Rate: sum of count x price per second; a stopped app's work fits in (wall time) x rate,
    because a restarted container replaces the one it restarts (they never run side by side), so at most `count` run at once. An
    unknown kind or function: the flat $2.70, no rate."""
    sp = specs()
    if kind not in KINDS or any(f not in sp for f, _ in KINDS[kind]):
        return ARM_WORST, None
    mult = lambda f: SHARDS if f == "score_layer" else 1  # noqa: E731  42: --score-shards multiplies the per-layer containers
    full = sum(n * mult(f) * (2 + sp[f][3]) * (sp[f][2] + PAD) * (sp[f][0] * RATE_CORE + sp[f][1] * RATE_GIB) for f, n in KINDS[kind])
    rate = sum(n * mult(f) * (sp[f][0] * RATE_CORE + sp[f][1] * RATE_GIB) for f, n in KINDS[kind])
    return round(full, 4), rate


def _hours(t0: datetime.datetime, t1: datetime.datetime) -> list:
    """Every UTC billing hour an app's life touches, from its created hour to its stopped hour."""
    h, out = t0.astimezone(UTC).replace(minute=0, second=0, microsecond=0), []
    while h <= t1.astimezone(UTC):
        out.append(h)
        h += HOUR
    return out


def _complete(interval_start: datetime.datetime, t: datetime.datetime) -> bool:
    """A billed hour is final 3 h after it ended (Codex 68: billing posts 1-2 h late, with no guarantee)."""
    return interval_start + HOUR + FINAL_AFTER <= t




def _ledger() -> list:
    return [json.loads(ln) for ln in LEDGER.read_text().splitlines() if ln.strip()] if LEDGER.exists() else []


def _append(rec: dict) -> None:
    if not DRY:
        with LEDGER.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")


def observe(ids: list, wait_s: int = 90) -> None:
    """Record each app's created/stopped times while `modal app list` still shows it (stopped apps drop out after about 15 minutes).
    Called right after a launch try returns (main() waits for the job) and at every cap check."""
    t_end = time.time() + wait_s
    todo = [a for a in ids if a not in {x.get("observe") for x in _ledger()}]
    while todo:
        listed = {a["app_id"]: a for a in json.loads(sh(MODAL, "app", "list", "--json"))}
        for a in list(todo):
            st = listed.get(a)
            if st is not None and st.get("state") == "stopped" and st.get("stopped_at"):
                _append({"observe": a, "created_at": st["created_at"], "stopped_at": st["stopped_at"]})
                todo.remove(a)
        if not todo or time.time() >= t_end:
            break
        time.sleep(10)


def cap_check(kind: str = "simh_arm") -> dict:
    """The rel9+ cumulative cap (4f, 12:2x PT: each app's bound is its OWN ceiling; flat $2.70 only as the fallback; Codex 68-70):
    sum over every earlier rel9+ reservation of its bound + this launch's planning ceiling <= $15. An earlier reservation's bound is:
    $0 for the 100-ephemeral-app refusal (no app is created); otherwise the sum over its apps of each app's billed cost once it was
    observed stopped and its stop hour and all its billing rows ended at least 3 h ago, else (observed wall time + 60 s) x the rate
    recorded at launch (valid under any number of infrastructure restarts: a restart replaces its container, so at most `count` of each
    function run at once). The launch is REFUSED while any earlier reservation is unreconciled, has no captured app ID, or has an app not
    observed stopped (still running, or gone from `modal app list` unseen): such an app has no bound. This launch's planning ceiling
    allows one infrastructure restart per container (Modal does not cap restarts), and it is the only app ever in flight, because a launch
    waits for its job and the next launch requires every earlier app observed stopped.
    sp493 line (Amendment 45, Aswin in-session via 4f ~23:1x): every try ever reserved on the line keeps its FULL ceiling, whatever happens
    (no app, a failure, a retry, billing posted): bound = sum of every earlier try's ceiling + this try's ceiling <= $150. Posted billing is
    only an alert (billed_flag), never a release. Tries whose job has not returned are in flight (at most 3)."""
    led = _ledger()
    if LINE == "sp493":
        allr = [x for x in led if "reserve" in x and "app" not in x and "reconciled" not in x]
        rec_of = {x["reconciled"]: x for x in led if "reconciled" in x}
        observe(sorted({a for x in rec_of.values() for a in x.get("apps", [])}), wait_s=0)
        stopped = {x["observe"] for x in _ledger() if "observe" in x}
        # Codex round 103: in flight = not reconciled, or reconciled with an app not observed stopped (a detached app after a client failure)
        # Codex round 104: a reconciled try with no app ID that did not prove "no app created" is unknown, so it stays in flight (a human reconciles it)
        inflight = sorted(x["reserve"] for x in allr if x["reserve"] not in rec_of or any(a not in stopped for a in rec_of[x["reserve"]].get("apps", []))
                          or (not rec_of[x["reserve"]].get("apps") and rec_of[x["reserve"]].get("no_app_created") is not True))
        this = ceilings(kind)[0]
        bound = round(sum(float(x["ceiling_usd"]) for x in allr) + this, 2)
        if len(inflight) >= MAX_INFLIGHT and not DRY:
            raise SystemExit(f"sp493 line: {len(inflight)} tries in flight (max {MAX_INFLIGHT}); wait for one to return")
        rec = {"rule": "sp493 $150 RESERVED cap (Amendment 45): every earlier try's full ceiling (never released) + this try's ceiling", "ledger": str(LEDGER),
               "reservations": len(allr), "reserved_usd": round(sum(float(x["ceiling_usd"]) for x in allr), 4), "this_launch_ceiling_usd": this,
               "bound_usd": bound, "cap_usd": CAP, "line": LINE, "in_flight": inflight}
        if DRY:
            return rec | {"dry_run_only": {"over_cap": bound > CAP, "in_flight_at_max": len(inflight) >= MAX_INFLIGHT}}
        if bound > CAP:
            raise SystemExit(f"sp493 cap: reserved ${rec['reserved_usd']} + this ${this} = ${bound} > ${CAP}; stop (Aswin's cap; no billing escape)")
        if now() >= SP493_LAST_DISPATCH:
            raise SystemExit(f"sp493 line: past the last dispatch time {SP493_LAST_DISPATCH:%H:%M} PT (Aswin's 03:00 PT hard stop)")
        return rec
    res = {x["reserve"]: x for x in led if "reserve" in x and "app" not in x and "reconciled" not in x}
    rec_of = {x["reconciled"]: x for x in led if "reconciled" in x}
    apps = {a: rid for rid, x in rec_of.items() for a in x.get("apps", [])}
    observe(sorted(apps), wait_s=0)
    obs = {x["observe"]: x for x in _ledger() if "observe" in x}
    t = datetime.datetime.now(UTC)
    start = (min((datetime.datetime.fromisoformat(x["at"]).astimezone(UTC) for x in res.values()), default=t) - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    rows, billed_hours = collections.defaultdict(list), set()
    for x in (_bill(start) if apps else []):
        billed_hours.add(x["_t"])  # Amendment 38 (4f 16:0x): an hour the report covers (any app has a row in it)
        if x["object_id"] in apps:
            rows[x["object_id"]].append(x)
    bound_of, blocked, inflight = {}, [], []
    for rid, r_ in sorted(res.items()):
        rate = r_.get("rate_usd_s")
        rc = rec_of.get(rid)
        if rc is None or (not rc.get("apps") and rc.get("no_app_created") is not True):
            if MAX_INFLIGHT > 1:  # 42: another stream's try in flight: bounded by its recorded ceiling
                inflight.append(rid)
                bound_of[rid] = float(r_["ceiling_usd"])
                continue
            blocked.append(f"{rid}: unreconciled, or no app ID captured")
            continue
        if rc.get("no_app_created") is True:
            bound_of[rid] = 0.0
            continue
        b = 0.0
        for a in rc["apps"]:
            o = obs.get(a)
            if o is None and rate is not None and MAX_INFLIGHT > 1:  # 42: an app not yet observed stopped: its ceiling (conservative)
                inflight.append(rid)
                b += float(r_["ceiling_usd"])
                continue
            if o is None or rate is None:
                blocked.append(f"{rid}: {a} not observed stopped (or no recorded rate)")
                continue
            t0, t1 = (datetime.datetime.fromisoformat(o[k]).astimezone(UTC) for k in ("created_at", "stopped_at"))
            stop_hour_end = t1.replace(minute=0, second=0, microsecond=0) + HOUR
            if rows[a] and stop_hour_end + FINAL_AFTER <= t and all(_complete(x["_t"], t) for x in rows[a]):
                b += sum(float(x["cost"]) for x in rows[a])
            elif not rows[a] and stop_hour_end + FINAL_AFTER <= t and all(h in billed_hours for h in _hours(t0, t1)):
                b += 0.0  # Amendment 38 (4f): a stalled app (no container ran) settles at $0 only on FINAL billing that has no row for it
            else:
                b += ((t1 - t0).total_seconds() + WALL_PAD) * rate
        bound_of[rid] = round(b, 4)
    if blocked and not DRY:  # 37d: a DRY run reports instead of stopping, so dry runs can run while a launch is in flight; it never launches
        raise SystemExit(f"{LINE} cap: cannot bound {len(blocked)} earlier reservation(s): {blocked[:3]}; wait for them to stop, or reconcile by hand")
    if len(set(inflight)) >= MAX_INFLIGHT > 1 and not DRY:  # 42: at most 3 apps in flight on the full line
        raise SystemExit(f"{LINE} line: {len(set(inflight))} tries in flight (max {MAX_INFLIGHT}); wait for one to stop")
    this = ceilings(kind)[0]
    bound = round(sum(bound_of.values()) + this, 2)
    rec = {"rule": "rel9+ cap $15: earlier reservations' bounds (billed when final, else observed wall x recorded rate) + this launch's planning ceiling "
                   "(RULE Amendment 31; 4f 12:2x PT per-app ceilings)", "ledger": str(LEDGER), "reservations": len(res), "bounds_usd": bound_of,
           "this_launch_ceiling_usd": this, "bound_usd": bound, "cap_usd": CAP, "line": LINE, "in_flight": sorted(set(inflight))}
    if DRY:
        rec |= {"dry_run_only": {"blocked": blocked, "over_cap": bound > CAP}}  # a real launch would stop on either
    elif bound > CAP:
        raise SystemExit(f"{LINE} cap: bound ${bound} > ${CAP} over {len(res)} reservations; wait for billing to become final or ask 4f")
    return rec


PT = zoneinfo.ZoneInfo("America/Los_Angeles")


def remaining_for_line(cap: dict) -> int:
    """main()'s remaining_launches term for this line ($2.70 per launch to come). Every DT launch from rel9 on, this one included, is inside
    the $15 rel9+ cap, so their total cost is at most $15, whatever earlier rel9+ apps turn out to cost (Codex 73: subtracting their bounds
    is not safe, since bounds shrink as billing becomes final). In $2.70 units: ceil(15 / 2.7) = 6, a constant. Earlier rel9+ apps are
    also counted in su_line(), so they are counted twice (conservative). No DT launch outside rel9 is planned (4f's day plan)."""
    import math
    return math.ceil(CAP / ARM_WORST)


def _known() -> dict:
    """{app: (kind, wall s or None, stop time UTC or None)} for every DT app this fork's launchers, fold checks and the rel9 ledger know.
    Stop evidence (Codex 71: rc 0 or an observed stop, nothing else): a launcher return with rc 0 (main() saw the job finish; wall = return
    - logged start); a fold check whose log shows "App completed" (a non-detached run: the app stops with its client; wall = log mtime -
    the result stamp + 300 s for app start-up before main()); a ledger observation. Any other known app (rc != 0, unobserved) has none."""
    out = {}
    for f in sorted(P2.glob("dtlaunch_g*.log")) + sorted(P2.glob("rel[6-9]_*.log")):
        start = {}
        for ln in f.read_text().splitlines():
            m = re.match(r"^(\d\d:\d\d:\d\d) (\S+) try \d+ ", ln)
            if m:
                start[m[2]] = m[1]
            m = re.match(r"^(\S+) rc (-?\d+) app \['(ap-[A-Za-z0-9]{22})'\] returned (\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d)", ln)
            if not m:
                continue
            kind = "relaunch" if m[1].startswith("rh1e4") else "simh_primary" if m[1] == "primary_v3" else "simh_arm" if m[1].startswith("simh") else "arm_attack"
            if m[2] != "0" or m[1] not in start:
                out[m[3]] = (kind, None, None)  # known, no stop evidence: never dropped (Codex 71)
                continue
            t0 = datetime.datetime.fromisoformat(f"{m[4]}T{start[m[1]]}").replace(tzinfo=PT)
            t1 = datetime.datetime.fromisoformat(f"{m[4]}T{m[5]}").replace(tzinfo=PT)
            out[m[3]] = (kind, (t1 - t0).total_seconds(), t1.astimezone(UTC)) if t1 >= t0 else (kind, None, None)
    for f in sorted(x for x in P2.glob("launch_rel*/launch_*_try*.log") if "dryrun" not in x.parent.name):  # Codex 72: every app ID in any real per-try log is known
        root = re.sub(r"_try\d+(\..*)?\.log$", "", f.name[len("launch_"):])
        kind = "relaunch" if root.startswith("rh1e4") else "simh_primary" if root == "primary_v3" else "simh_arm" if root.startswith("simh") else "arm_attack"
        for a in set(re.findall(r"ap-[A-Za-z0-9]{22}", f.read_text())):
            out.setdefault(a, (kind, None, None))
    for f in sorted(P2.glob("fold_check_*.log")):
        txt = f.read_text()
        st = re.search(r"dt_unseen_check_(\d{4}-\d\d-\d\d)_(\d\d)(\d\d)(\d\d)\.json", txt)
        for a in set(re.findall(r"ap-[A-Za-z0-9]{22}", txt)):
            if st and "App completed" in txt:
                t0 = datetime.datetime.fromisoformat(f"{st[1]}T{st[2]}:{st[3]}:{st[4]}").replace(tzinfo=PT)
                t1 = datetime.datetime.fromtimestamp(f.stat().st_mtime, tz=UTC)
                out[a] = ("fold_check", (t1 - t0).total_seconds() + 300, t1)
            else:
                out.setdefault(a, ("fold_check", None, None))
    led = _ledger()
    kind_of = {x["reserve"]: x.get("kind", "") for x in led if "reserve" in x and "reconciled" not in x}
    for x in led:  # every rel9+ app is known; without an observation it has no stop evidence
        for a in x.get("apps", []) if "reconciled" in x else []:
            out[a] = (kind_of.get(x["reconciled"], ""), None, None)
    for x in led:
        if "observe" in x:
            t0, t1 = (datetime.datetime.fromisoformat(x[k]).astimezone(UTC) for k in ("created_at", "stopped_at"))
            out[x["observe"]] = (out.get(x["observe"], ("simh_arm", None, None))[0], (t1 - t0).total_seconds(), t1)
    return out


def su_line() -> tuple:
    """Scale-up line ($100), per app, for main()'s preflight (settled + 2.7 x (remaining + unsettled)). settled_usd = every DT-app dollar
    billed in FINAL hours (3 h after the hour) + an upper bound for every DT app not yet settled. An app is settled (its billed dollars only)
    when it has stop evidence, billing rows, all rows final and its stop hour final. Otherwise: with stop evidence and a wall, + (wall + 60 s)
    x its rate (valid under any number of restarts); running, known without stop evidence, or unknown: one unsettled $2.70 job, until it
    is not running and its last billing row ended 6 h ago, when it settles at billed (Codex 71). An app with no billing row stays unsettled."""
    t = datetime.datetime.now(UTC)
    b = [x for x in _bill("2026-10-15") if x.get("description") == DT_APP]
    billed_final = sum(float(x["cost"]) for x in b if _complete(x["_t"], t))
    rows = collections.defaultdict(list)
    for x in b:
        rows[x["object_id"]].append(x)
    known = _known()
    listed = {a["app_id"]: a for a in json.loads(sh(MODAL, "app", "list", "--json")) if a.get("description") == DT_APP}
    bound, uns = 0.0, set()
    for a in sorted(set(rows) | set(known) | set(listed)):
        kind, wall, stop = known.get(a, ("", None, None))
        st = listed.get(a)
        if st is not None and st.get("state") != "stopped":
            uns.add(a)
            continue
        if st is not None and st.get("stopped_at") and stop is None:
            t0, t1 = (datetime.datetime.fromisoformat(st[k]).astimezone(UTC) for k in ("created_at", "stopped_at"))
            wall, stop = (t1 - t0).total_seconds(), t1
        r_ = rows.get(a, [])
        if stop is not None and r_ and all(_complete(x["_t"], t) for x in r_) and _complete(stop.replace(minute=0, second=0, microsecond=0), t):
            continue  # settled: its dollars are in billed_final
        if stop is None and st is None and r_ and all(x["_t"] + HOUR + datetime.timedelta(hours=6) <= t for x in r_):
            continue  # no stop evidence, but not running and its last billed hour ended 6 h ago: settled at billed (in billed_final)
        if a not in known and st is None:
            uns.add(a)
            continue
        full, rate = ceilings(kind)
        if stop is not None and wall is not None and rate is not None:
            bound += (wall + WALL_PAD) * rate
        else:
            uns.add(a)
    return round(billed_final + bound, 4), uns


batch = sys.argv[1] if len(sys.argv) > 1 else ""
if LINE == "full":  # 42: the full line runs only its own (a'') paths, added with 09's regex and d9's pins
    raise SystemExit("full line: no launch path yet (Amendment 42 builds the infrastructure; the (a'') paths come with 09's regex and d9's pins)")
if batch not in BATCHES and batch not in ("primary_v3", "il311_q4c", "ug", "ha25f", "oldsel", "t196fill", "s4rgap", "sp493"):
    raise SystemExit(f"usage: launch_rel9.py <batch> [targets...]; batch in {sorted(BATCHES)}")
assert all(FILL.get(k) for k in ("rule31", "spend_rel9")) or DRY, "RULE Amendment 31 and the rel9 SPEND row must be committed first"
head = git("rev-parse", "HEAD").decode().strip()
assert not git("status", "--porcelain", "--untracked-files=no").strip(), f"{DTS} has uncommitted tracked changes"
assert git("branch", "-r", "--contains", head).strip(), "the DT worktree HEAD is not pushed"
for c in ("fd163c4c7", "5aa9f32bf", "0f9927f3e"):  # 25h, 0d's attestation and 09's rulings must be reachable (pinned in Amendment 31)
    assert git("cat-file", "-t", c).strip() == b"commit", c
L.mkdir(exist_ok=True)
LOCK = P2 / "dt4f" / ("rel9.lock" if LINE == "rel9" else f"full.lock.s{STREAM}")  # 42: one lock per full-line stream
if not DRY:
    try:
        LOCK.open("x").write(json.dumps({"pid": os.getpid(), "batch": batch, "at": now().isoformat(timespec="seconds")}))
    except FileExistsError:
        raise SystemExit(f"another rel9 invocation holds {LOCK} (the budget is shared); wait for it, or rename a stale lock after checking") from None
    import atexit
    atexit.register(lambda: LOCK.rename(LOCK.with_name(f"{LOCK.name}.released_{now().strftime('%H%M%S')}_{os.getpid()}")))


def run_primary_v3() -> None:
    """primary_v3 (Amendment 31): one `--job simh_primary` launch; its gates are in main() (counted keys at their commit, each dump's extraction
    evidence from its committed records); the job refuses until 09's folder is pinned (PV3["out"])."""
    cap = cap_check("simh_primary")
    s_su, uns_su = su_line()
    rem = remaining_for_line(cap)  # every later DT launch is inside the rel9+ cap (Amendment 31)
    bound = round(s_su + ARM_WORST * (rem + len(uns_su)), 2)
    assert bound <= SU_STOP, ("scale-up line", bound)
    man = L / "simh_counted_keys_primary_v3.json"
    man.write_bytes(git("show", "572f67dbb:experiments/2026-10-14_q3_q35_simhonest_purchase/results/simh_counted_keys_primary_v3.json"))
    msha = sha(man.read_bytes())
    assert msha == "80753f44aa66e78445f233e43e18c96fd01913246afcc03420b9322c447f4082"
    cells = DTS / "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"
    csha = sha(cells.read_bytes())
    assert csha == "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb"
    pf = {"message": "primary_v3", "manifest_sha256": msha, "checked_at": now().isoformat(timespec="seconds"),
          "d9_gates": {"h200": True, "longest_epoch_gate": True, "corrected_render": True},
          "d9_gate_evidence": {"all": "checked in main() from the committed extract and format records of simh_full_b1 / b2 (famexp's pins @ 01c1d09bb): "
                                      "H200 on every shard, device_count 1, a reference reproduction and one longest-context read-twice per shard within tolerance, "
                                      "epochs 145 / 4, format check pass"},
          "settled_usd": s_su, "remaining_launches": rem, "unsettled_jobs": len(uns_su), "budget_line": "su311/nm360 DT scale-up line ($100 stop), conservative DT-app count",
          "rel9_cap": cap, "citations": {"grammar": "09 Append 25h section 11 @ fd163c4c7", "attestation_gate": "25h section 10 @ 5aa9f32bf",
                                         "folders": "09 @ 0f9927f3e item 7: /simh/primary_v3/dt/simh_full_b1/ and /simh/primary_v3/dt/simh_full_b2/, create-only",
                                         "rule": f"DT RULE Amendment 31 @ {FILL.get('rule31', 'PENDING')}", "spend": f"SPEND @ {FILL.get('spend_rel9', 'PENDING')}"},
          "folds": "T196 cells: designated held-out fold (S_old); Amendment 30's never-in-cohort tasks listed as unmapped",
          "firewall": "SIM arm: sealed, no thresholds, rates, catches or comparisons", "stratum": "S_old"}
    assert "PENDING" not in json.dumps(pf) or DRY, "a RULE or SPEND reference is still PENDING"
    pff = L / "preflight_primary_v3.json"
    pff.write_text(json.dumps(pf, indent=1))
    args = ["--job", "simh_primary", "--manifest", str(man), "--manifest-sha", msha, "--cells", str(cells), "--cells-sha", csha, "--name", "primary_v3", "--preflight", str(pff)]
    cmd = ["env", "PYTHONPATH=src", "uv", "run", "--no-project", "--python", "3.12", "--with", "modal==1.6.1", "--with", "numpy",
           *(["python", str(P2 / "dryrun_main.py")] if DRY else ["modal", "run", "--detach", "experiments/2026-10-14_p2_dt_scaleup/run_dt_q4b.py"]), *args]
    if not DRY:
        try:
            (L / "claim_primary_v3").open("x").close()
        except FileExistsError:
            raise SystemExit("primary_v3: already submitted") from None
    launch_one("primary_v3", "simh_primary", cmd, bound, cap, ["simh/primary_v3/dt/simh_full_b1", "simh/primary_v3/dt/simh_full_b2"])
    raise SystemExit(0)


Q4C_RULED = {"sim2|t411_media_processing": "unknown (1-message run, terminal OutputContractError); tier 3; no IL pair (09 @ 0f9927f3e item 2)",
             "sim2|t501_user_tracking": "unknown (1-message run, terminal EmptyPolicyCompletionError); tier 3; no IL pair (09 @ 0f9927f3e item 2)"}


def vol_present(path: str) -> bool:
    """True/False only from SUCCESSFUL listings (sh retries with timeouts, then stops): walk up while the parent itself is absent."""
    parent, name = path.rsplit("/", 1) if "/" in path else ("", path)
    if parent and not vol_present(parent):
        return False
    return name in {x.strip("/").split("/")[-1] for x in sh(MODAL, "volume", "ls", WVOL, *([parent] if parent else [])).split()}


def vol_names(path: str) -> set:
    """Names in an EXISTING Volume folder from ONE `volume ls --json` call (sh retries with timeouts, then stops; never an empty guess):
    the inventory's per-folder listing. Its parent's listing named the folder, so a missing name here is confirmed by this successful
    listing (Amendment 43b, 4f: 1 call per folder instead of vol_ls's parent walk)."""
    return {x["filename"].strip("/").split("/")[-1] for x in json.loads(sh(MODAL, "volume", "ls", "--json", WVOL, path))}


def vol_ls(path: str) -> set:
    """Names in a Volume folder; empty only when the folder is confirmed absent; a failing listing stops (Codex 76)."""
    return {x.strip("/").split("/")[-1] for x in sh(MODAL, "volume", "ls", WVOL, path).split()} if vol_present(path) else set()


def recover(t: str, ids: list, folders: list, rc: int, head: str, started: datetime.datetime, pf_sha: str) -> bool:
    """rc != 0 with an app ID (e.g. the laptop lost the network while main() waited for .remote; the detached job runs on): wait until every
    app is stopped (bounded: 50 min, every Modal call retried with timeouts), then read each expected create-only folder on the work Volume.
    - every folder has manifest_dt.json of THIS attempt: its preflight_sha256 equals this try's preflight file (written once per root, checked_at
      to the second), its git equals the HEAD captured before the try, its stamp is not before the try; and every npz the manifest lists is in
      the folder -> write a record RECOVERED from the manifests into results/ and return True (the chain may continue);
    - every folder CONFIRMED absent (its parent listing succeeds and does not hold it) -> return False (a retry is a new launch);
    - anything else (a listing error, npz without a manifest, another attempt's manifest) -> stop for a human; nothing deleted (Codex 75)."""
    if not ids or DRY:
        return False
    t_end = time.time() + 3000
    while True:
        listed = {a["app_id"]: a for a in json.loads(sh(MODAL, "app", "list", "--json"))}
        if all(listed.get(a, {}).get("state") == "stopped" for a in ids):
            break
        if time.time() > t_end:
            raise SystemExit(f"{t}: rc {rc} and app(s) {ids} not stopped after 50 min; stop for a human")
        time.sleep(30)
    observe(ids, wait_s=0)
    got = {}
    for fo in folders:
        if not vol_present(fo):
            got[fo] = None  # confirmed absent: a successful listing does not hold it
            continue
        names = {x.strip("/").split("/")[-1] for x in sh(MODAL, "volume", "ls", WVOL, fo).split()}
        dst = L / f"recovered_{fo.replace('/', '__')}.json"
        if "manifest_dt.json" not in names:
            raise SystemExit(f"{t}: {fo} holds {sorted(names)} without manifest_dt.json (partial); stop for a human, nothing deleted")
        sh(MODAL, "volume", "get", "--force", WVOL, f"{fo}/manifest_dt.json", str(dst))
        m = json.loads(dst.read_text())
        rel = [v["path"].split(f"/work/{fo}/", 1)[1] if f"/work/{fo}/" in v["path"] else "?" for v in m.get("files", {}).values()]
        missing = sorted(r_ for r_ in rel if r_ == "?" or Path(r_).name not in (names if "/" not in r_ else vol_ls(f"{fo}/{r_.rsplit('/', 1)[0]}")))
        if missing:
            raise SystemExit(f"{t}: {fo}'s manifest lists files the folder lacks: {missing}; stop for a human")
        got[fo] = m
    if all(m is None for m in got.values()):  # 37c: say so, so a failed try's log shows that the guard ran (Q2 s5, 15:32)
        print(t, f"recover: rc {rc}; app(s) {ids} stopped; every expected folder confirmed absent on the Volume (no output): a retry is a new launch",
              flush=True)
        return False
    st = started.strftime("%Y-%m-%d_%H%M%S")
    if any(m is None or m.get("git") != head or m.get("stamp (PT)", "") < st or m.get("preflight_sha256") != pf_sha for m in got.values()):
        raise SystemExit(f"{t}: the folders are not all complete manifests of THIS attempt (git {head[:9]}, preflight {pf_sha[:12]}, stamp >= {st}); stop for a human")
    per = ("d9_arm_to_world", "rows", "no_fold_cells", "runs", "non_ended_staged", "launch_task", "root_read", "files", "folder", "extraction_run",
           "zero_call_runs", "unmapped_runs", "firewall")
    first = next(iter(got.values()))
    rec = {k: v for k, v in first.items() if k not in per} | {
        "folders": {fo.rsplit("/", 1)[-1]: {"world": (list(m.get("d9_arm_to_world", {}).values()) or [m.get("extraction_run")])[0], "rows": m["rows"],
                                           "runs": len(m["runs"]), "files": m["files"]} for fo, m in got.items()},
        "recovered_from_volume": {"why": f"client rc {rc}: main() wrote no local record; the detached job finished server-side", "apps": ids,
                                  "attempt": {"head": head, "preflight_sha256": pf_sha, "started": st},
                                  "sources": [f"/{fo}/manifest_dt.json (written last by the job), fetched read-only" for fo in got]}}
    dest = DTS / "experiments/2026-10-14_p2_dt_scaleup/results" / f"dt_q4b_{first.get('job', 'arm')}_{t}_{first['stamp (PT)']}_recovered.json"
    dest.open("x").write(json.dumps(rec, indent=1))
    print(t, "RECOVERED from the Volume:", {fo: (m["rows"], len(m["runs"])) for fo, m in got.items()}, "->", dest.name, flush=True)
    return True


@contextlib.contextmanager
def _caplock():
    """Amendment 42: the cap check and the reservation are one step across parallel streams (an exclusive flock on dt4f/<line>.caplock)."""
    with open(P2 / "dt4f" / f"{LINE}.caplock", "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def billed_flag() -> None:
    """Amendment 42 (full line): posted billing of this line's apps by app ID against the billed target (quote + 30%); a flag, not a stop."""
    if BILLED_TARGET is None or DRY:
        return
    apps = {a for x in _ledger() if "reconciled" in x for a in x.get("apps", [])}
    led0 = min((x["at"] for x in _ledger() if "reserve" in x), default=None)
    if not apps or led0 is None:
        return
    usd = sum(float(x["cost"]) for x in _bill(led0[:10]) if x["object_id"] in apps)
    print(now().strftime("%H:%M:%S"), f"{LINE} line: billed posted ${usd:.4f} over {len(apps)} apps (target ${BILLED_TARGET})",
          "WARNING: billed target exceeded" if usd > BILLED_TARGET else "", flush=True)


def launch_one(t: str, kind: str, cmd: list, bound: float, cap: dict, folders: list = ()) -> list:
    """One root: a reservation BEFORE each try, reconciled after with every app ID; the app's times observed at once. On the full line
    the cap check runs again for every try, inside the line's flock together with the reservation (parallel streams, Amendment 42)."""
    for k in range(1, 16):
        with _caplock():
            if k > 1 or LINE in ("full", "sp493"):
                cap = cap_check(kind)
            rid = f"{t}#try{k}@{now().strftime('%Y-%m-%dT%H:%M:%S')}"
            _c = ceilings(kind)
            _append({"reserve": rid, "kind": kind, "ceiling_usd": _c[0], "rate_usd_s": _c[1], "root": t, "try": k, "at": now().isoformat(timespec="seconds"),
                     **({"stream": STREAM} if STREAM else {})})
        print(now().strftime("%H:%M:%S"), t, "try", k, "line bound", bound, "rel9+ bound", cap["bound_usd"], *(["DRY-ONLY", cap.get("dry_run_only")] if DRY else []), flush=True)
        t_start = now()
        head_now = git("rev-parse", "HEAD").decode().strip()  # captured BEFORE the try (Codex 75)
        pf_arg = cmd[cmd.index("--preflight") + 1] if "--preflight" in cmd else ""
        pf_sha = sha(Path(pf_arg).read_bytes()) if pf_arg else ""
        try:  # Codex 76: the client has a timeout too; a detached job runs on, and recover() decides from the app state and the Volume
            pr = subprocess.run(cmd, cwd=DTS, capture_output=True, text=True, timeout=3600)
        except subprocess.TimeoutExpired as e:
            dec = lambda b: b.decode() if isinstance(b, bytes) else (b or "")  # noqa: E731
            pr = subprocess.CompletedProcess(cmd, -9, dec(e.stdout), dec(e.stderr) + "\nlauncher: client timed out after 3600 s")
        (L / f"launch_{t}_try{k}.log").write_text(" ".join(cmd) + "\n" + pr.stdout + pr.stderr)
        ids = sorted(set(re.findall(r"ap-[A-Za-z0-9]{22}", pr.stdout + pr.stderr)))
        limit = "reached limit of 100 ephemeral apps" in pr.stdout + pr.stderr
        _append({"reconciled": rid, "apps": ids, "rc": pr.returncode, "no_app_created": bool(limit and not ids), "at": now().isoformat(timespec="seconds")})
        if ids and not DRY:
            observe(ids)
        if limit:
            time.sleep(120)
            continue
        print(t, "rc", pr.returncode, "app", ids[:1], "returned", now().strftime("%Y-%m-%d %H:%M:%S %Z"), flush=True)
        if pr.returncode and not (folders and pf_sha and recover(t, ids, list(folders), pr.returncode, head_now, t_start, pf_sha)):
            raise SystemExit(f"{t}: launch failed (rc {pr.returncode}); stopping before the next root")
        billed_flag()
        return ids[:1]
    raise SystemExit(f"{t}: app limit never cleared")


def run_q4c(targets: list) -> None:
    """il311_q4c (Amendment 32): the +311 Q4 completion, run3 and sim2, as q2c (Amendments 25/27) with 09's rulings @ 0f9927f3e: sim2 t411/t501
    staged out, class unknown (item 2); run3|t505 and sim2|t380 not attempted but staged, so scored as zero-call runs if they have no rows (item 3).
    By-key inputs: dt4f/xcheck_q4c.json (record v2 @ 78a5b7ebd vs the pinned q4 per-arm map, tiers, staged = record minus ruled)."""
    x = json.loads((P2 / "dt4f" / "xcheck_q4c.json").read_text())
    assert x["record"]["pass"] and x["record"]["map_sha_ok"] and x["record_keys_equal_perarm_map"] and x["record_launch_runkey_equal_map"] \
        and x["all_replacement_no_fallback"] and not x["cells_failing_tier_checks"] and x["staged_run3_equal_record_run3"] \
        and x["staged_sim2_equal_record_sim2_minus_ruled"] and x["staged_sha_equal_replacement"] and all(x["ruled_excluded_sha_equal_record"].values()) \
        and all(x["ruled_staged_false_in_record"].values()) and sorted(x["ruled_in_staged_out_rows"]) == sorted(Q4C_RULED) and all(x["zero_unit_staged"].values()), "q4c by key"
    rt = {"il311_q4c_run3": "il-q4c__il311_q4c_run3.json", "il311_q4c_sim2": "il-q4c__il311_q4c_sim2.json"}
    subprocess.run(["git", "-C", str(DTS), "fetch", "-q", "origin", "aswin/q3-overnight-chain"], check=True)
    rec_commit = git("rev-parse", "--short=9", "origin/aswin/q3-overnight-chain").decode().strip()
    spec = {t: {"routing": rt[t], "attest": "il-q4c_attest.json", "records": rec_commit} for t in targets}
    vout = L / "verify_il311_q4c.json"
    subprocess.run(["python3", "-I", str(P2 / "verify_roots.py"), str(RT), str(DTS), str(vout), json.dumps(spec)], check=True, capture_output=True)
    V = json.loads(vout.read_text())["roots"]
    _ls = sh(MODAL, "volume", "ls", WVOL, "scaleup")  # fresh scored-folder inventory (retried, timed out, never silently empty)
    for nm in sorted({y.strip("/").split("/")[-1] for y in _ls.split() if "__" in y}):
        dst = P2 / "manifests_dt" / f"{nm}.json"
        if dst.exists():
            continue
        has = {y.strip("/").split("/")[-1] for y in sh(MODAL, "volume", "ls", WVOL, f"scaleup/{nm}").split()}
        if "manifest_dt.json" in has:
            sh(MODAL, "volume", "get", WVOL, f"scaleup/{nm}/manifest_dt.json", str(dst))
        elif "dt" in has:
            raise SystemExit(f"{nm}: DT output without manifest_dt.json; the scored-folder inventory is incomplete")
    scored = {}
    for f in (P2 / "manifests_dt").glob("il311_*.json"):
        for j in json.loads(f.read_text())["runs"]:
            scored.setdefault(j, []).append(f.stem)
    allow = {r["key"] for r in json.loads(git("show", "78a5b7ebd:experiments/2026-10-14_q3_overnight_chain/results/supersession/supersession_il311_q4c_2026-10-15_v2.json"))["records"]}
    cspec, csha = CELLS["su311"]
    cells = L / "cells_su311.json"
    cells.write_bytes(git("show", cspec))
    assert sha(cells.read_bytes()) == csha
    for t in targets:
        v, r = V[t], json.loads((RT / rt[t]).read_text())
        assert v["attestation"]["pass"] and v["repro"]["ok"] and v["repro"]["bit_exact"] == v["repro"]["epochs"] and not v["needs_21g"] and all(v["format"].values()), (t, "gates")
        assert v["labels"]["match"] and v["stage"]["epochs_match"] and v["stage"]["shas_match"] and v["stage"]["exclusions_match_routing"], (t, "labels/stage")
        assert v["manifest"]["counts_match_routing"] and not v["manifest"]["tasks_not_in_pinned_list"] and v["pool_key"] == "il311", (t, "manifest")
        ptr = json.loads((RT / v["attestation"]["pointer_chain"][-1]).read_text())
        att = json.loads(git("show", f"{ptr['commit']}:{ptr['path']}"))["attestations"][r["arms"][0]["staged_sha"]]
        h = next(c for c in att if c["check"] == "longest_epoch_h200_gate")
        assert h["pass"] and h["detail"]["chunk_writers_recorded_h200"] is True and h["detail"]["unrecorded_chunks"] == [], (t, "writer gate")
        p_, c_ = (y.strip() for y in r["manifest"].split(" @ "))
        man_raw = git("show", f"{c_}:{p_}")
        files = json.loads(man_raw)["files"]
        excl = r.get("staging_exclusions", {})
        arms = {a["meta_arm"] for a in r["arms"]}
        staged = {f"{f['world']}|{f['task']}" for f in files if f["arm_dir"] in arms and f"{f['launch']}/{f['rel']}#e0" not in excl}
        ex_j = {f"{f['world']}|{f['task']}": excl[f"{f['launch']}/{f['rel']}#e0"] for f in files if f"{f['launch']}/{f['rel']}#e0" in excl}
        assert set(ex_j) == {k for k in Q4C_RULED if k.split("|")[0] == t.split("_")[-1]}, (t, "exclusions = 09's ruled rows")
        rep = {j for j in staged if j in scored} - allow
        if rep:
            raise SystemExit(f"{t}: {len(rep)} staged keys repeat earlier scored il311 folders outside the record's keys")
        found = [y for y in git("ls-tree", "-r", "--name-only", rec_commit, f"{B}/").decode().split() if y.split("/")[-1].startswith(f"extract_chain_{v['root']}_a")]
        assert len(found) == 1, (t, found)
        cap = cap_check("arm_attack")
        s_su, uns_su = su_line()
        rem = remaining_for_line(cap)  # every later DT launch is inside the rel9+ cap (Amendment 31)
        bound = round(s_su + ARM_WORST * (rem + len(uns_su)), 2)
        assert bound <= 200.0, (t, "IL line", bound)
        man = L / f"manifest_{t}.json"
        man.write_bytes(man_raw)
        msha = sha(man_raw)
        assert msha == v["manifest"]["sha256"], t
        pf = {"message": t, "manifest_sha256": msha, "checked_at": now().isoformat(timespec="seconds"),
              "d9_gates": {"h200": True, "longest_epoch_gate": True, "corrected_render": True},
              "d9_gate_evidence": {"routing": rt[t], "routing_sha256": v["routing_sha256"], "attestation_commit": v["attestation"]["commit"],
                                   "h200": f"attestation failing {{}}; longest_epoch_h200_gate pass incl. the writer gate; extraction {v['extraction_records']}",
                                   "longest_epoch_gate": f"repro {v['repro']['bit_exact']}/{v['repro']['epochs']} bit-exact, failing []",
                                   "corrected_render": f"format check pass {v['format']}; stage record {v['stage']['record']} @ {rec_commit} = routing"},
              "settled_usd": s_su, "remaining_launches": rem, "unsettled_jobs": len(uns_su), "budget_line": "IL DT line ($200 stop), conservative DT-app count",
              "rel9_cap": cap, "citations": {"rule": f"DT RULE Amendment 32 @ {FILL.get('rule32', 'PENDING')}", "spend": f"SPEND @ {FILL.get('spend_q4c', 'PENDING')}",
                                             "rulings": "09 @ 0f9927f3e items 1-3 (q4c sim2 t411/t501 unknown; run3|t505 and sim2|t380 not attempted)",
                                             "record": "supersession_il311_q4c_2026-10-15_v2.json @ 78a5b7ebd (pinned in the job); attestation il-q4c @ 605b7e1e7f"},
              "exclusions": {j: {"stager_string": s_, "class": Q4C_RULED[j]} for j, s_ in ex_j.items()},
              "exclusions_never_scored_0": "declared exclusions have no staged epoch and no row; the job never scores them",
              "zero_unit_runs": "run3|t505_tiktok and sim2|t380_model_registry: not attempted (09 item 3), staged; listed with their call count, never scored 0",
              "uniqueness": f"by key: {len(staged)} staged joins; repeats of earlier scored il311 folders only within the record's {len(allow)} keys "
                            "(the 21 tier 3 originals in il311_q4_run3_s2, kept as provenance)",
              "folds": "S_new: fold -1", "stratum": r["stratum"], "staged_set": r["staged_set"]}
        assert "PENDING" not in json.dumps(pf) or DRY, (t, "a RULE or SPEND reference is still PENDING")
        pff = L / f"preflight_{t}.json"
        pff.write_text(json.dumps(pf, indent=1))
        args = ["--job", "arm", "--manifest", str(man), "--manifest-sha", msha, "--runs", v["root"], "--name", t, "--cells", str(cells), "--cells-sha", csha,
                "--eval-root", "evals_v1", "--stage-records", f"{rec_commit}:{B}/{v['stage']['record']}", "--root-records", f"{rec_commit}:{found[0]}", "--preflight", str(pff)]
        cmd = ["env", "PYTHONPATH=src", "uv", "run", "--no-project", "--python", "3.12", "--with", "modal==1.6.1", "--with", "numpy",
               *(["python", str(P2 / "dryrun_main.py")] if DRY else ["modal", "run", "--detach", "experiments/2026-10-14_p2_dt_scaleup/run_dt_q4b.py"]), *args]
        if not DRY:
            try:
                (L / f"claim_{t}").open("x").close()
            except FileExistsError:
                raise SystemExit(f"{t}: already submitted") from None
        launch_one(t, "arm_attack", cmd, bound, cap, [f"scaleup/{r['staged_set']}__{r['arms'][0]['meta_arm'].split('@')[0]}"])
    raise SystemExit(0)


UG_FOLDER = re.compile(r"^ug_sim(?:_s[0-9]+)?_v1(?:_try[0-9]+)?__sim_attack_ug$")  # 09's ug scorer-folder regex (same as run_dt_q4b's UG)
UG_UNKNOWN = ("t470", "t424", "t389", "t215", "t426")  # 09 @ fb3d4649a §1: class unknown x5, staged out
HA25F_FOLDER = re.compile(r"^su311_q[124]_main_ha25f_v1(_try[0-9]+)?__honest_a$")  # 09 @ 1bc718e98, verbatim (same as run_dt_q4b's HA25F)
HA25F_KEYS = {"q1": ("t389_model_registry", "d160ffdc956ffc89180043b3575b002d4b0f1cc3f7600082f38decc81f28dae0"),
              "q2": ("t309_slack_clone", "69c964700620933d0fbfa771fc83068d208a3e20c48dfad47d443a2ec88f5028"),
              "q4": ("t325_model_registry", "e1cdf8d1594bac78b6e64377557bf63d008f2af0248912f21c7bd60c523b7247")}
HA25F_ATT_V2 = ("c6ec4212d", "experiments/2026-10-14_q3_overnight_chain/results/hygiene/attestations_ha25f_ha25f_v2.json")  # 39b: d9's re-attestation
T196_CELLS = ("HEAD:experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json", "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb")


def _verified(t: str, rtf: str, attest: str, pool: str, labels: str, branch: str = "aswin/q3-overnight-chain", att_pin: tuple = ()) -> tuple:
    """d9's gates for one routed root by key (verify_roots.py), as for the other rel9 roots; returns (v, routing, rec_commit, man_raw, files, ex_j, staged, found).
    branch: where d9 committed the root's records (the 25f H_A roots are on aswin/q3-ha25f, attestation @ 06a50cc19)."""
    fetch(branch) if LINE == "sp493" else subprocess.run(["git", "-C", str(DTS), "fetch", "-q", "origin", branch], check=True)  # 45f: retried on sp493
    rec_commit = git("rev-parse", "--short=9", f"origin/{branch}").decode().strip()
    vout = L / f"verify_{t}.json"
    subprocess.run(["python3", "-I", str(P2 / ("verify_roots_r103.py" if LINE == "sp493" else "verify_roots.py")), str(RT), str(DTS), str(vout),
                    json.dumps({t: {"routing": rtf, "attest": attest, "records": rec_commit}})], check=True, capture_output=True)  # 45: the sp493 cell list
    rb_ = (RT / rtf).read_bytes()
    v, r = json.loads(vout.read_text())["roots"][t], json.loads(rb_)
    if LINE == "sp493" and sha(rb_) != v["routing_sha256"]:  # Codex 109: this read and verify_roots' read are the same bytes
        raise SystemExit(f"{t}: the routing changed between verify_roots and the launcher's read; stop")
    if "_cutfull" in r["staged_set"] and (CUTFULL is None or "_cutfull_gate_" in r["staged_set"] or not CUTFULL.fullmatch(r["staged_set"])):
        raise SystemExit(f"{t}: {r['staged_set']}: cutfull names wait for 09's regex (Amendment 42); gate sets are never scored")
    assert v["attestation"]["pass"] and v["repro"]["ok"] and v["repro"]["bit_exact"] == v["repro"]["epochs"] and not v["needs_21g"] and all(v["format"].values()), (t, "gates")
    assert (v["labels"].get("match") is True) if labels == "match" else (v["labels"].get("absent") is True), (t, "labels", v["labels"])
    assert v["stage"]["epochs_match"] and v["stage"]["shas_match"] and v["stage"]["exclusions_match_routing"], (t, "stage")
    assert v["manifest"]["counts_match_routing"] and not v["manifest"]["tasks_not_in_pinned_list"] and v["pool_key"] == pool, (t, "manifest/pool")
    ptr = json.loads((RT / v["attestation"]["pointer_chain"][-1]).read_text())
    att = json.loads(git("show", f"{att_pin[0]}:{att_pin[1]}" if att_pin else f"{ptr['commit']}:{ptr['path']}"))["attestations"][r["arms"][0]["staged_sha"]]
    assert not att_pin or all(c["pass"] is True for c in att), (t, "pinned re-attestation: every check passes")  # 39b: d9's v2 for ha25f
    h = next(c for c in att if c["check"] == "longest_epoch_h200_gate")
    assert h["pass"] and h["detail"]["chunk_writers_recorded_h200"] is True and h["detail"]["unrecorded_chunks"] == [], (t, "writer gate")
    p_, c_ = (y.strip() for y in r["manifest"].split(" @ "))
    man_raw = git("show", f"{c_}:{p_}")
    files, excl, arms = json.loads(man_raw)["files"], r.get("staging_exclusions", {}), {a["meta_arm"] for a in r["arms"]}
    staged = {f"{f['world']}|{f['task']}": f for f in files if f["arm_dir"] in arms and f"{f['launch']}/{f['rel']}#e0" not in excl
              and not (LINE == "sp493" and f.get("attempt_kind") == "reference")}  # 45e: sp493 donor reference rows are never staged or scored here
    ex_j = {f"{f['world']}|{f['task']}": excl[f"{f['launch']}/{f['rel']}#e0"] for f in files if f"{f['launch']}/{f['rel']}#e0" in excl}
    found = [y for y in git("ls-tree", "-r", "--name-only", rec_commit, f"{B}/").decode().split() if y.split("/")[-1].startswith(f"extract_chain_{v['root']}_a")]
    assert len(found) == 1, (t, found)
    return v, r, rec_commit, man_raw, files, ex_j, staged, found[0]


def _no_dt_yet(folder: str) -> None:
    """A shared /scaleup folder may hold other teams' outputs (p2/, famexp_r2/, sol/); it must hold no DT output yet."""
    has = vol_ls(f"scaleup/{folder}")
    if {"manifest_dt.json", "dt"} & has:
        raise SystemExit(f"scaleup/{folder} already holds DT output {sorted({'manifest_dt.json', 'dt'} & has)}; never overwritten")


def _launch_arm(t: str, kind: str, rtf: str, v: dict, r: dict, rec_commit: str, man_raw: bytes, found: str, cells_specs: list, pf_extra: dict, folder: str,
                out_rel: str = "", outs: tuple = ()) -> None:
    cap = cap_check(kind)
    if LINE in ("full", "sp493"):  # 42: the full line's own real-time bound ($60) replaces the scale-up/IL lines; main() checks full_bound_usd; 45: sp493
        s_su, uns_su, rem, bound = 0.0, set(), 1, cap["bound_usd"]
    else:
        s_su, uns_su = su_line()
        rem = remaining_for_line(cap)  # every later DT launch is inside the rel9+ cap (Amendment 31)
        bound = round(s_su + ARM_WORST * (rem + len(uns_su)), 2)
        assert bound <= SU_STOP, (t, "scale-up line", bound)  # ug and ha25f runs are not IL: main() applies the scale-up stop
    man = L / f"manifest_{t}.json"
    man.write_bytes(man_raw)
    msha = sha(man_raw)
    assert msha == v["manifest"]["sha256"], t
    cfiles, cshas = [], []
    for nm, (cspec, csha) in cells_specs:
        cf = L / f"cells_{nm}.json"
        cf.write_bytes(git("show", cspec))
        assert sha(cf.read_bytes()) == csha, (t, nm)
        cfiles.append(str(cf))
        cshas.append(csha)
    pf = {"message": t, "manifest_sha256": msha, "checked_at": now().isoformat(timespec="seconds"),
          "d9_gates": {"h200": True, "longest_epoch_gate": True, "corrected_render": True},
          "d9_gate_evidence": {"routing": rtf, "routing_sha256": v["routing_sha256"], "attestation_commit": v["attestation"]["commit"],
                               "h200": f"attestation failing {{}}; longest_epoch_h200_gate pass incl. the writer gate; extraction {v['extraction_records']}",
                               "longest_epoch_gate": f"repro {v['repro']['bit_exact']}/{v['repro']['epochs']} bit-exact, failing []",
                               "corrected_render": f"format check pass {v['format']}; stage record {v['stage']['record']} @ {rec_commit} = routing"},
          "settled_usd": s_su, "remaining_launches": rem, "unsettled_jobs": len(uns_su), "budget_line": "scale-up DT line ($100 stop), conservative DT-app count",
          "rel9_cap": cap, "stratum": r["stratum"], "staged_set": r["staged_set"], "folder": folder} | pf_extra \
        | ({"line": "full", "full_bound_usd": cap["bound_usd"], "budget_line": f"full-length line (${CAP} real-time bound; Amendment 42)", "stream": STREAM,
            "score_shards": SHARDS} if LINE == "full" else {}) \
        | ({"line": "sp493", "sp493_reserved_usd": cap["bound_usd"], "budget_line": f"sp493 line (${CAP} RESERVED cap; Amendment 45)", "stream": STREAM}
           if LINE == "sp493" else {})
    assert "PENDING" not in json.dumps(pf) or DRY, (t, "a RULE or SPEND reference is still PENDING")
    pff = L / f"preflight_{t}.json"
    pff.write_text(json.dumps(pf, indent=1))
    args = ["--job", "arm", "--manifest", str(man), "--manifest-sha", msha, "--runs", v["root"], "--name", t, "--cells", ",".join(cfiles), "--cells-sha", ",".join(cshas),
            "--eval-root", "evals_v1", "--stage-records", f"{rec_commit}:{B}/{v['stage']['record']}", "--root-records", f"{rec_commit}:{found}", "--preflight", str(pff),
            *(["--score-shards", str(SHARDS)] if SHARDS != 1 else [])]
    cmd = ["env", "PYTHONPATH=src", "uv", "run", "--no-project", "--python", "3.12", "--with", "modal==1.6.1", "--with", "numpy",
           *(["python", str(P2 / "dryrun_main.py")] if DRY else ["modal", "run", "--detach", "experiments/2026-10-14_p2_dt_scaleup/run_dt_q4b.py"]), *args]
    if LINE == "sp493" and not DRY:  # 45: main()'s local checks first, with no app and no reservation (a local refusal never uses the allowance)
        pre = subprocess.run([*cmd[:cmd.index("modal")], "python", str(P2 / "dryrun_main.py"), *args], cwd=DTS, capture_output=True, text=True, timeout=900)
        (L / f"predispatch_{t}.log").write_text(pre.stdout + pre.stderr)
        if pre.returncode or "DRYRUN-OK" not in pre.stdout:
            raise SystemExit(f"{t}: main()'s local checks refuse this root before dispatch (no reservation): {(pre.stdout + pre.stderr).strip().splitlines()[-1:]}")
    if not DRY:
        try:
            (L / f"claim_{t}").open("x").close()
        except FileExistsError:
            raise SystemExit(f"{t}: already submitted") from None
    launch_one(t, kind, cmd, bound, cap, list(outs) or [out_rel or f"scaleup/{folder}"])


def run_ug() -> None:
    """ug (Amendment 38; Aswin in 4f's session ~15:53 PT 10-15, item a): ONE attack-arm job over ug_sim_v1_try2 (143 staged sim_ug runs),
    folder ug_sim_v1_try2__sim_attack_ug. 09: the 5 staging exclusions are class unknown (@ fb3d4649a §1); t83 is staged and scored but
    analysis-excluded (@ 6a36b4a27): a status row; t88 never started (no epoch); the pre-scoring list is confirmed by key @ 915b45b87.
    Cells: the T196 list (S_old: designated held-out folds) and the +311 list (S_new: fold -1). Sealed: no thresholds, rates or catches."""
    t, rtf = "ug_sim", "ug__ug_sim.json"
    v, r, rec_commit, man_raw, files, ex_j, staged, found = _verified(t, rtf, "ug_attest.json", "ug", "match")
    folder = f"{r['staged_set']}__sim_attack_ug"
    assert UG_FOLDER.fullmatch(folder) and r["stratum"] == "sim_ug" and [a["meta_arm"] for a in r["arms"]] == ["sim_attack@september_ungrounded"], (t, folder)
    assert len(staged) == 143 and all(k.startswith("sim_ug|") for k in staged), (t, "staged", len(staged))
    assert sorted(j.split("|")[1].split("_")[0] for j in ex_j) == sorted(UG_UNKNOWN) and all(j.startswith("sim_ug|") for j in ex_j), (t, "exclusions = 09's unknown x5")
    assert "sim_ug|t83_auto_workflow" in staged, (t, "t83 is staged (scored, analysis-excluded)")
    _no_dt_yet(folder)
    pf_extra = {"citations": {"rule": f"DT RULE Amendment 38 @ {FILL.get('rule38', 'PENDING')}", "spend": f"SPEND @ {FILL.get('spend_ug', 'PENDING')}",
                              "approval": "Aswin in 4f's session, about 15:53 PT 10-15 (item a), relayed by 4f",
                              "rulings": "09 @ fb3d4649a (class unknown x5; t88 never started), @ 6a36b4a27 (t83 analysis-excluded), @ 915b45b87 (pre-scoring list confirmed by key)",
                              "format": "09 @ fd40f4db4 §2 (ug folder regex; Append 34)", "run_table": "run_table_ug.json @ 676e7527d", "comparability_map": "@ e1601ac31"},
                "exclusions": {j: {"stager_string": s_, "class": "unknown (09 @ fb3d4649a §1); staged out, never scored 0"} for j, s_ in ex_j.items()},
                "status_rows": {"sim_ug|t83_auto_workflow": "staged and scored here (sealed); analysis-excluded by 09 @ 6a36b4a27 (counted-vs-pinned reference mismatch, no fallback)",
                                "sim_ug|t88": "never started: no epoch, no row (09 @ fb3d4649a)"},
                "folds": "T196 cells (S_old): designated held-out fold; +311 cells (S_new): fold -1",
                "firewall": "SIM arm: sealed, no thresholds, rates, catches or comparisons (the arm job computes none)"}
    _launch_arm(t, "arm_attack", rtf, v, r, rec_commit, man_raw, found, [("t196", T196_CELLS), ("su311", CELLS["su311"])], pf_extra, folder)
    raise SystemExit(0)


def run_ha25f(targets: list) -> None:
    """ha25f (Amendment 39; Aswin in 4f's session ~15:53 PT 10-15, item c): one honest-only arm job per 25f H_A folder, in series. Each folder
    holds exactly one honest_a run, its key and .eval sha256 as 09 lists them (@ 46dda27f7, ef5654c0e, 1bc718e98); S_new, fold -1. 0d: t389's
    and t309's H_A enter the analysis only after 09's admission or extension ruling; scoring them now is sealed."""
    rts = {f"su311_{q}_main_ha25f": f"ha25f__su311_{q}_main_ha25f.json" for q in HA25F_KEYS}
    assert targets and set(targets) <= set(rts), (targets, sorted(rts))
    for t in targets:
        v, r, rec_commit, man_raw, files, ex_j, staged, found = _verified(t, rts[t], "ha25f_attest.json", "ha25f", "absent", "aswin/q3-ha25f", HA25F_ATT_V2)
        folder, q = f"{r['staged_set']}__honest_a", t.split("_")[1]
        assert HA25F_FOLDER.fullmatch(folder) and r["stratum"] == "S_new" and [a["meta_arm"] for a in r["arms"]] == ["honest_a"], (t, folder)
        key, want = HA25F_KEYS[q]
        assert not ex_j and list(staged) == [f"honest_a|{key}"] and staged[f"honest_a|{key}"]["sha256"] == want, (t, "one key with 09's listed .eval sha256")
        _no_dt_yet(folder)
        pf_extra = {"citations": {"rule": f"DT RULE Amendment 39 @ {FILL.get('rule39', 'PENDING')}", "spend": f"SPEND @ {FILL.get('spend_ha25f', 'PENDING')}",
                                  "approval": "Aswin in 4f's session, about 15:53 PT 10-15 (item c), relayed by 4f",
                                  "rulings": "09 @ 46dda27f7 (25f H_A note), ef5654c0e (folder names), 1bc718e98 (t309 addition; regex q[124])",
                                  "condition": "0d: t389's and t309's H_A enter the analysis only after 09's admission or extension ruling; scored sealed now",
                                  "writer_gate": "the normal gate, on d9's re-attestation attestations_ha25f_ha25f_v2.json @ c6ec4212d (revision 2; supersedes @ 06a50cc19, kept); "
                                                 "context: 09's READ_LOG @ 07bca2d6b (a single-shard first-attempt H200 run meets the gate)"},
                    "key": {f"honest_a|{key}": {"eval_sha256": want, "check": "equals 09's listed reference .eval sha256"}},
                    "readers": "09 @ ef5654c0e: only the 25f analysis reads these folders, for H_A; every other reader must exclude them",
                    "use": ("not used: replacement staging-excluded (t389's sim-honest replacement in simh311_q1_sim_rlr was staging-excluded, so t389 never "
                            "enters H_sim and its H_A is never read; scored as approved; 4f 16:3x PT)") if key == "t389_model_registry" else "25f H_A (sealed)",
                    "folds": "S_new: fold -1"}
        _launch_arm(t, "arm_honest", rts[t], v, r, rec_commit, man_raw, found, [("su311", CELLS["su311"])], pf_extra, folder)
    raise SystemExit(0)


OLDSEL_FOLDER = re.compile(r"^ilold_oldsel_(?:run3_v1(?:_try[0-9]+)?__attack|sim2_v1(?:_try[0-9]+)?__sim_attack)$")  # 09 @ 273470963, verbatim
OLDSEL_STATUS = ("89595bff5", "experiments/2026-10-15_q3_interleaved_run3_sim2/config/addon/sel97_status_rows.json")  # t151, t96: status rows only
IL_STAGE_MAPS_L = ("0a52fa131", "experiments/2026-10-14_q3_overnight_chain/config/il_stage_maps.json")  # the job's pinned stage maps (old: sel 97)
CELLS_T196 = ("HEAD:experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json", "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb")


def run_oldsel(targets: list) -> None:
    """sel97 (Amendment 40; Aswin in 4f's session ~15:53 PT 10-15, item b): one attack-arm job per folder, in series (run3 first). 09's
    Append 32 note @ 273470963: a descriptive add-on, never a registered IL launch or replacement; stage "sel"; outcomes embargoed until the
    registered IL result is read. 95 il_old selection cells (t151, t96 status rows @ 89595bff5); IL line ($200 stop); sealed."""
    rts = {"ilold_oldsel_run3": "il-oldsel__ilold_oldsel_run3.json", "ilold_oldsel_sim2": "il-oldsel__ilold_oldsel_sim2.json"}
    assert targets and set(targets) <= set(rts), (targets, sorted(rts))
    sel = set(json.loads(git("show", f"{IL_STAGE_MAPS_L[0]}:{IL_STAGE_MAPS_L[1]}"))["old"]["sel"]["cells"])
    status = json.loads(git("show", f"{OLDSEL_STATUS[0]}:{OLDSEL_STATUS[1]}"))["rows"]
    assert len(sel) == 97 and {r["key"].split("|")[1] for r in status} == {"t151_model_registry", "t96_file_indexing"}, "sel97 cells / status rows"
    ilold_scored = {}
    for nm in sorted({y.strip("/").split("/")[-1] for y in sh(MODAL, "volume", "ls", WVOL, "scaleup").split() if y.strip("/").split("/")[-1].startswith("ilold_")}):
        if "manifest_dt.json" in vol_names(f"scaleup/{nm}"):  # 43b: one listing call per folder
            dst = L / f"scored_{nm}.json"
            sh(MODAL, "volume", "get", "--force", WVOL, f"scaleup/{nm}/manifest_dt.json", str(dst))
            for j in json.loads(dst.read_text())["runs"]:
                ilold_scored.setdefault(j, []).append(nm)
    for t in targets:
        v, r, rec_commit, man_raw, files, ex_j, staged, found = _verified(t, rts[t], "il-oldsel_attest.json", "ilold", "match")
        world = "run3" if t.endswith("_run3") else "sim2"
        folder = f"{r['staged_set']}__{'attack' if world == 'run3' else 'sim_attack'}"
        assert OLDSEL_FOLDER.fullmatch(folder) and r["stratum"] == "S_old", (t, folder)
        tasks = {k.split("|")[1] for k in staged}
        assert all(k.startswith(f"{world}|") for k in staged) and tasks <= sel and all(f["stage"] == "sel" for f in staged.values()), (t, "sel cells, stage sel")
        want_ex = set() if world == "run3" else {"t188", "t126"}  # d9's routing (attested @ 2cb4ff968): sim2 t188, t126 staged out
        assert {k.split("|")[1].split("_")[0] for k in ex_j} == want_ex and len(staged) == (95 if world == "run3" else 93), (t, "staged / exclusions", len(staged), sorted(ex_j))
        rep = sorted(j for j in staged if j in ilold_scored)
        if rep:
            raise SystemExit(f"{t}: {len(rep)} staged keys repeat earlier scored ilold folders: {rep[:3]}")
        _no_dt_yet(folder)
        pf_extra = {"citations": {"rule": f"DT RULE Amendment 40 @ {FILL.get('rule40', 'PENDING')}", "spend": f"SPEND @ {FILL.get('spend_sel97', 'PENDING')}",
                                  "approval": "Aswin in 4f's session, about 15:53 PT 10-15 (item b), relayed by 4f",
                                  "registration": "09's Append 32 note @ 273470963 (descriptive add-on; folder regex; stage sel)",
                                  "attestation": "attestations_il_il-oldsel.json @ 2cb4ff968 (failing {})", "status_rows": "sel97_status_rows.json @ 89595bff5"},
                    "label": "descriptive add-on; not a registered IL launch or replacement; cannot amend the registered IL result; embargoed until the registered IL result is read",
                    "exclusions": {j: {"stager_string": s_, "class": "staged out by d9 (no 09 class cited at scoring); never scored 0"} for j, s_ in ex_j.items()},
                    "status_rows": {f"{world}|t151_model_registry": "not run: no pinned reference (09; @ 89595bff5)", f"{world}|t96_file_indexing": "not run (09 ruling ~13:30, option b; @ 89595bff5)"},
                    "uniqueness": f"by key: {len(staged)} staged joins; none in an earlier scored ilold folder ({len(ilold_scored)} keys read)",
                    "folds": "T196 (S_old) selection cells: designated held-out fold", "budget_line": "IL DT line ($200 stop; main() applies it to IL runs), conservative DT-app count; checked here against $100, which is stricter"}
        _launch_arm(t, "arm_attack", rts[t], v, r, rec_commit, man_raw, found, [("t196", CELLS_T196)], pf_extra, folder)
    raise SystemExit(0)


T196FILL_FOLDER = re.compile(r"^simh_t196fill(?:_s[0-9]+)?_v1(?:_try[0-9]+)?__sim_honest$")  # Amendment 43: 09's ACK regex, verbatim
S4RGAP_FOLDER = re.compile(r"^ilnm_s4rgap_sim2(_s[0-9]+)?_v1(_try[0-9]+)?__sim_attack$")  # Amendment 43: 09's sim2 gap-fill §3 regex, verbatim


SUVOL_L = "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14"
CELLS_SP493 = ("HEAD:experiments/2026-10-14_p2_dt_scaleup/config/sp493_cells_v1.json", "b095416cff2b510b2b05414a2e5ec4db93f4cf253468b3e47d7d2327bdc90f03")
SP493_STRATA = ("S_paper493", "S_paper493_sim", "S_paper493_simh")
SP493_SUPERSEDED_FINALS = {  # 4f ~23:3x: the chain's 23:31 combined finals, built from the stopped batches; SUPERSEDED, never read or used (sha256 at 23:3x)
    "final_hygiene_sp493.json": "61f3989c5110ed98d66b901ee24f9e8955f915105bb58fff2abd7d1cfd6b798b",
    "final_hygiene_sp493_g1_sim.json": "7e2330a86030c1f9a8c441302b843b212318532a29f2d36fd1db7609bdb40a74",
    "final_hygiene_sp493_g2_sim.json": "f1ae361ed7ceed2e33accf0acbd061cc0ee258220701a4e43116d13876702d75"}
SP493_RELEASES = ((P2 / "sp493_release_2026-10-16_r1.json", "3d6610c96d4ae9a9e021b8770617377afda510d5c51815cf905ce7ff66d6de41"),  # 4f's release r1 (00:3x)
                  (P2 / "sp493_release_2026-10-16_r2.json", "2228272ce99933cc77dfffe09153674617d60a918b61f3e13ffcf98c14403ff1"))  # r2 (01:3x): box C real arms


def fetch(branch: str) -> None:
    """git fetch with retries: parallel sp493 streams can race on the remote-tracking ref lock (01:22 dry runs); fails loudly after 6 tries."""
    for k in range(6):
        pr = subprocess.run(["git", "-C", str(DTS), "fetch", "-q", "origin", branch], capture_output=True, text=True)
        if pr.returncode == 0:
            return
        time.sleep(2 + 3 * k)
    raise SystemExit(f"git fetch origin {branch} failed 6 times: {pr.stderr.strip()[-300:]}")
SP493_SPLIT = ("59f333d03", "experiments/2026-10-15_q3_interleaved_run3_sim2/results/sp493_sim_split_2026-10-15.json",
               "5444cc11b4a6d6520fdb453819a5ffde145e430b9f8727ee5c9ce82089583fe9")  # 45f: d9's L1/L2 split by key (g1 101 / g6 137, g2 96 / g7 121), "forever"
SP493_L3 = ("627400b22", "experiments/2026-10-15_q3_interleaved_run3_sim2/config/sp493sim/refindex_L3.json",
            "7f98a7de46d2748c31333427191bd3cc3a55a3cdcb07670c0a6a3638f1de064a")  # 45f: L3's 37 cell keys (g3)
JOINS = P2 / "dt4f" / "sp493_joins.jsonl"  # Codex round 103: every sp493 root's scored joins, claimed inside the cap flock before dispatch; never released


def claim_joins(t: str, joins: set) -> None:
    """Atomic across streams (the line's flock): refuse if another root already claimed any of these joins; else append this root's claim."""
    with _caplock():
        prior = [json.loads(x) for x in JOINS.read_text().splitlines() if x.strip()] if JOINS.exists() else []
        clash = sorted(j for x in prior if x["root"] != t for j in set(x["joins"]) & joins)
        if clash:
            raise SystemExit(f"{t}: joins already claimed by another sp493 root: {clash[:3]} (each (world, cell) is scored once)")
        if any(x["root"] == t and set(x["joins"]) != joins for x in prior):  # Codex round 104: a retry of a root must claim exactly the same joins
            raise SystemExit(f"{t}: its staged joins differ from its earlier claim; stop for a human (the routing changed)")
        if not DRY and not any(x["root"] == t for x in prior):
            with JOINS.open("a") as fh:
                fh.write(json.dumps({"root": t, "joins": sorted(joins), "at": now().isoformat(timespec="seconds")}) + "\n")


def _job_grammar() -> dict:
    """Amendment 45: the job's own sp493 grammar (SP493 regexes, refusals, cohort-out and never-staged keys), lifted from the launched HEAD's run_dt_q4b.py."""
    import ast
    src = git("show", "HEAD:experiments/2026-10-14_p2_dt_scaleup/run_dt_q4b.py").decode()
    want = {"SET_WORLD", "SP493", "SP493_SET", "SP493_POOLS", "SP493_MIN_TRY", "SP493_STRATUM", "SP493_WORLDS", "SP493_WORLD_MAP", "SP493_SIMH_WORLD", "SP493_SUPERSEDED",
            "SP493_REFUSE_KEYS", "SP493_NEVER", "SP493_NAMES", "SP493_G12_REMAINDER_OK", "SP493_TAIL_POOLS", "SP493_GROUP", "sp493_fam"}
    tree, g = ast.parse(src), {"re": re}
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in want) or (isinstance(n, ast.Assign) and any(getattr(x, "id", "") in want for x in n.targets))]
    exec(compile(ast.Module(body=keep, type_ignores=[]), "run_dt_q4b.py@HEAD", "exec"), g)
    assert want <= set(g), sorted(want - set(g))
    return g


def run_sp493(targets: list) -> None:
    """sp493 (Amendment 45; Aswin in-session via 4f ~23:1x: to 03:00 PT, $150 reserved cap): ONE arm job per d9-routed S_paper493 root, named by
    its routing file <batch>__<target>.json. By key: d9's gates (verify_roots), the job's grammar and refusals on every folder (superseded _try2 and
    earlier, g1/g2 sim pending 09, any _rlr), stratum S_paper493*, the scale-up Volume, no never-staged key staged, t987 counted (the job skips it),
    no staged join in an earlier scored sp493 folder, no DT output yet. 09 Q1: real and sim attack /scaleup/<folder>/dt/, sim-honest
    /simh/sp493_g<k>/dt/<folder>/ (create-only); 09 Q2 (a): a staged t987 key stops the root; SEALED until 09's analysis note. Before any
    reservation, main()'s local checks run once through dryrun_main.py (no app), so a local refusal never consumes the $150 allowance."""
    if LINE != "sp493":
        raise SystemExit("sp493 roots run only with DT_LINE=sp493 (Amendment 45)")
    if not targets:
        raise SystemExit("sp493: name the routing file(s) <batch>__<target>.json (4f relays d9's routed roots)")
    bad = [x for x in targets if x in SP493_SUPERSEDED_FINALS or x.startswith("final_") or not re.fullmatch(r"sp493-[a-z0-9_]+__sp493_[a-z0-9_]+\.json", x)]
    if bad:
        raise SystemExit(f"sp493: refused targets {bad}: not a per-root routing file, or a SUPERSEDED combined final (4f ~23:3x: rebuilt by the chain)")
    G = _job_grammar()
    released = {}  # 4f (00:3x): d9 replaced routing files in place, so each record is pinned by PATH AND SHA256 from 4f's release lists
    for rp, rs in SP493_RELEASES:
        rraw = rp.read_bytes()
        if sha(rraw) != rs:
            raise SystemExit(f"{rp.name}: sha256 differs from its pin; stop")
        for x in json.loads(rraw)["released"]:
            if x["routing_file"] in released and released[x["routing_file"]] != x:
                raise SystemExit(f"{x['routing_file']}: two release lists disagree on it; stop")
            released[x["routing_file"]] = x
    scored, mfs = {}, []
    for nm in sorted(x for x in vol_ls("scaleup") if x.startswith("sp493_")):  # /scaleup/<folder> is shared: only DT's own files matter
        names = vol_names(f"scaleup/{nm}")
        if "dt" in names and "manifest_dt.json" not in names:
            raise SystemExit(f"scaleup/{nm}: DT npz without manifest_dt.json (a partial write); stop for a human")
        mfs += [f"scaleup/{nm}/manifest_dt.json"] if "manifest_dt.json" in names else []
    for gk in sorted(x for x in vol_ls("simh") if x.startswith("sp493_g")):
        for nm in sorted(vol_ls(f"simh/{gk}/dt")):
            if "manifest_dt.json" not in vol_names(f"simh/{gk}/dt/{nm}"):
                raise SystemExit(f"simh/{gk}/dt/{nm}: a folder without manifest_dt.json (a partial write); stop for a human")
            mfs.append(f"simh/{gk}/dt/{nm}/manifest_dt.json")
    for mf in mfs:
        dst = L / f"scored_{mf.replace('/', '__')}"
        sh(MODAL, "volume", "get", "--force", WVOL, mf, str(dst))
        for j in json.loads(dst.read_text())["runs"]:
            scored.setdefault(j, []).append(mf)
    for rtf in targets:
        rb = (RT / rtf).read_bytes() if rtf in released and (RT / rtf).exists() else b""
        if rtf not in released or sha(rb) != released[rtf]["routing_sha256"]:  # Codex 109: hash and parse the same bytes
            raise SystemExit(f"{rtf}: not in 4f's release list, or its sha256 differs from the release pin; stop (fail closed)")
        r = r0 = json.loads(rb)
        t, batch = r["staged_set"], rtf.split("__")[0]
        arms = [a["meta_arm"].split("@")[0] for a in r["arms"]]
        folders = [f"{r['staged_set']}__{a}" for a in arms]
        try:
            fams = {G["sp493_fam"](x) for x in [r["staged_set"], *folders]}
        except ValueError as e:
            raise SystemExit(f"{rtf}: refused by the job's grammar: {e}") from None
        if len(fams) != 1 or None in fams or r["stratum"] != G["SP493_STRATUM"][next(iter(fams))[0]] or r.get("volume") != SUVOL_L or not arms:
            raise SystemExit(f"{rtf}: not one sp493 family, a stratum other than its family's, another Volume, or no arms ({sorted(map(str, fams))}, {r['stratum']}, {r.get('volume')})")
        fam_, pool_ = next(iter(fams))
        labels = "match" if any(a in ("attack", "sim_attack") for a in arms) else "absent"
        v, r, rec_commit, man_raw, files, ex_j, staged, found = _verified(t, rtf, f"{batch}_attest.json", r["pool"], labels, os.environ.get("DT_SP493_BRANCH", "aswin/q3-overnight-chain"))
        if v["routing_sha256"] != released[rtf]["routing_sha256"] or r != r0 or r["manifest"] != released[rtf]["manifest"]:
            raise SystemExit(f"{rtf}: the routing read by verify_roots, or its manifest field, differs from the release pin; stop")
        never = sorted(j for j in staged if j in G["SP493_NEVER"])
        se = sorted(j for j, f in staged.items() if f.get("sample_error", "missing") is not None)  # 45c (09 ~00:0x): staged rows need sample_error == null
        if se or any("sample_error" not in f for f in files):
            raise SystemExit(f"{t}: staged rows whose sample_error is not null {se[:3]} ({len(se)}), or manifest rows without the field; stop (09: fail closed)")
        refused = {j: G["SP493_REFUSE_KEYS"][j.split("|")[1]] for j in staged if j.split("|")[1] in G["SP493_REFUSE_KEYS"]}
        joins = {f"{G['SP493_WORLD_MAP'].get(j.split('|')[0], j.split('|')[0])}|{j.split('|')[1]}" for j in staged}  # the scored joins (09 Q1 world map)
        rep = sorted(j for j in joins if j in scored)
        if never or refused or rep:
            raise SystemExit(f"{t}: staged keys d9 excludes by key {never}, keys 09 refuses by name {refused}, or joins already scored in sp493 folders {rep[:3]}; stop")
        if G["SP493_SUPERSEDED"] is None and not DRY:
            raise SystemExit(f"{t}: d9's explicit SUPERSEDED list is not pinned in the job yet (09 Q3); no launch, no reservation")
        grp = set(G["SP493_GROUP"].get(pool_, ()))  # 45f: the recipe group (g1+g6, g2+g7, g3) by each set's own g token, across x and y

        def in_grp(st):  # a sim-attack staged set the grammar admits whose g is in this group; refused or other sets: False
            try:
                f_ = G["sp493_fam"](st)
            except ValueError:
                return False
            return f_ is not None and f_[0] == "sim_attack" and f_[1] in grp
        if fam_ == "sim_attack":  # 45f (09 @ b9e0b6eda): each staged cell belongs to this set's g by d9's split (g1/g2/g6/g7) or L3's 37 (g3)
            fetch("aswin/q3-interleaved-run3-sim2")
            sraw, lraw = git("show", f"{SP493_SPLIT[0]}:{SP493_SPLIT[1]}"), git("show", f"{SP493_L3[0]}:{SP493_L3[1]}")
            if sha(sraw) != SP493_SPLIT[2] or sha(lraw) != SP493_L3[2]:
                raise SystemExit(f"{t}: d9's split list or L3 key file differs from its pin; stop")
            grp_of = {x["key"]: x["group"] for x in json.loads(sraw)["rows"]} | {k: "g3" for k in json.loads(lraw)["references"]}
            off = sorted(j for j in staged if grp_of.get(j.split("|")[1]) != pool_)
            if off:
                raise SystemExit(f"{t}: staged cells outside group {pool_} by d9's split / L3 keys: {off[:3]} ({len(off)}); stop (09)")
        if fam_ == "sim_attack":  # 45c/45f (09): each cell is in exactly ONE admitted set of its group; a cell in two = STOP
            mine, other, peers = {j.split("|")[1] for j in staged}, {}, []
            for nm2, x2 in sorted(released.items()):  # Codex 109: every released routing is checked against its pin first (same bytes hashed and parsed)
                if nm2 == rtf:
                    continue
                b2 = (RT / nm2).read_bytes() if (RT / nm2).exists() else b""
                if sha(b2) != x2["routing_sha256"]:
                    raise SystemExit(f"{t}: released routing {nm2} differs from its pin; one-set-per-cell cannot be checked; stop")
                peers.append(json.loads(b2))
            # 09 (b) (~01:2x): partial tail releases must not block; an unreleased routing is not admitted, so it is not scanned here. A later
            # release is checked when it launches (this set is then a released peer), and claim_joins refuses any repeated join.
            for r2 in peers:
                if not in_grp(r2.get("staged_set", "")) or not r2.get("arms"):
                    continue  # another group, another family, or refused (superseded, burned, outside the grammar): not an admitted set of this group
                p2, c2 = (y.strip() for y in r2["manifest"].split(" @ "))
                a2, x2 = {a["meta_arm"] for a in r2["arms"]}, r2.get("staging_exclusions") or {}
                for f in json.loads(git("show", f"{c2}:{p2}"))["files"]:
                    if f["arm_dir"] in a2 and f"{f['launch']}/{f['rel']}#e0" not in x2 and f.get("attempt_kind") != "reference" and f["task"] in mine:
                        other.setdefault(f["task"], []).append(r2["staged_set"])
            if other:
                raise SystemExit(f"{t}: cells also in another admitted routed set of group {sorted(grp)}: {sorted(other.items())[:3]}; STOP (09)")
        claim_joins(t, joins)
        prov = RT / f"{batch}_provenance.json"
        if fam_ == "sim_attack" and not prov.exists():
            raise SystemExit(f"{t}: a sim-attack root without d9's provenance file {prov.name}; stop")
        outs = tuple(f"simh/sp493_{pool_}/dt/{fo}" if fam_ == "sim_honest" else f"scaleup/{fo}" for fo in folders)
        for fo in folders:
            if fam_ == "sim_honest" and fo in vol_ls(f"simh/sp493_{pool_}/dt"):
                raise SystemExit(f"simh/sp493_{pool_}/dt/{fo} already exists; never overwritten")
            if fam_ != "sim_honest":
                _no_dt_yet(fo)
        cells = [("sp493", CELLS_SP493)] + ([("t196", CELLS_T196)] if pool_ == "g4" else [])  # 09 Q2; g4: T196 keys on sp493 T196 mains (09 ~23:5x)
        pf_extra = {"citations": {"rule": f"DT RULE Amendment 45 @ {FILL.get('rule45', 'PENDING')}", "spend": f"SPEND @ {FILL.get('spend_sp493', 'PENDING')}",
                                  "approval": FILL.get("approval_sp493", "PENDING (Aswin's OK via 4f)"), "registration": "09 Append 39 + amendment 1 @ 922cbb75b; regexes @ 9cb3d0291 (ack @ baa36605d); _try3 @ 1243f1e27",
                                  "heldout_gate": FILL.get("heldout_sp493", "PENDING"), "routing_records": f"d9 records @ {rec_commit}"},
                    "label": "S_paper493: SEALED until 09's analysis note; PRELIMINARY", "family": fam_, "pool": pool_, "folders": folders,
                    "routing_pin": {"releases": {rp.name: rs for rp, rs in SP493_RELEASES}, "routing_file": rtf, "routing_sha256": released[rtf]["routing_sha256"]},
                    "exclusions": {j: {"stager_string": s_, "class": "staged out by d9; never scored 0"} for j, s_ in ex_j.items()},
                    "refused_by_name": {"rule": "09 Q2 (a): t987 refused by name, fail closed", "staged": sorted(refused)},
                    "uniqueness": f"by key: {len(staged)} staged joins; none in an earlier scored sp493 folder ({len(scored)} keys read)",
                    **({"provenance": {"file": prov.name, "sha256": sha(prov.read_bytes())}} if prov.exists() else {}),
                    "folds": "DT's own fold of the main where its fit saw the main (rows.json, by key), else fold -1 (am.1 section 2; the job asserts it)",
                    **({"group_check": {"split": f"{SP493_SPLIT[0]}:{SP493_SPLIT[1]}", "split_sha256": SP493_SPLIT[2], "l3": f"{SP493_L3[0]}:{SP493_L3[1]}",
                                        "l3_sha256": SP493_L3[2], "group": pool_, "recipe_group": sorted(G["SP493_GROUP"].get(pool_, ()))}} if fam_ == "sim_attack" else {})}
        kind = "arm_attack" if labels == "match" else "simh_arm" if fam_ == "sim_honest" else "arm_honest"
        _launch_arm(t, kind, rtf, v, r, rec_commit, man_raw, found, cells, pf_extra, folders[0], "", outs)
    raise SystemExit(0)


def run_t196fill() -> None:
    """t196fill (Amendment 43; PREP: a launch needs Aswin's OK via 4f): ONE sim-honest job over simh_t196fill_v1 (8 T196 cells, world
    t196fill), folder simh_t196fill_v1__sim_honest under /work/simh/simh_t196fill/dt/ (create-only). 09's A25 fill-in (@ 286b2f518 +
    addendum 6): LABELLED, POST HOC, sealed, second refresh only. T196 cells on their held-out folds."""
    t, rtf = "simh_t196fill", "simh-t196fill__simh_t196fill.json"
    v, r, rec_commit, man_raw, files, ex_j, staged, found = _verified(t, rtf, "simh-t196fill_attest.json", "simh_t196fill", "absent")
    folder = f"{r['staged_set']}__sim_honest"
    assert T196FILL_FOLDER.fullmatch(folder) and r["stratum"] == "simh_t196fill" and [a["meta_arm"] for a in r["arms"]] == ["sim_honest@september"], (t, folder)
    assert len(staged) == 8 and not ex_j and all(k.startswith("t196fill|") for k in staged), (t, "8 staged t196fill runs, no exclusions", len(staged), sorted(ex_j))
    if folder in vol_ls("simh/simh_t196fill/dt"):
        raise SystemExit(f"simh/simh_t196fill/dt/{folder} already exists; never overwritten")
    pf_extra = {"citations": {"rule": f"DT RULE Amendment 43 @ {FILL.get('rule43', 'PENDING')}", "spend": f"SPEND @ {FILL.get('spend_t196fill', 'PENDING')}",
                              "approval": FILL.get("approval_t196fill", "PENDING (Aswin's OK via 4f)"), "registration": "09 ACK (@ 286b2f518 + addendum 6; regex)",
                              "records": "manifest @ 61482c1c0; repro @ dbce7de48; attestation @ 1549de878"},
                "label": "LABELLED, POST HOC, sealed, second refresh only (09)", "folds": "T196 cells (S_old): designated held-out fold",
                "firewall": "SIM arm: sealed, no thresholds, rates, catches or comparisons (the arm job computes none)"}
    _launch_arm(t, "simh_arm", rtf, v, r, rec_commit, man_raw, found, [("t196", CELLS_T196)], pf_extra, folder, f"simh/simh_t196fill/dt/{folder}")
    raise SystemExit(0)


def run_s4rgap() -> None:
    """s4rgap (Amendment 43; PREP: a launch needs Aswin's OK via 4f): ONE sim-attack job over ilnm_s4rgap_sim2_v1 (09's sim2 gap-fill part
    s4r §3): 59 routed = 59 registered keys, 57 staged, nm145 and nm242 staged out (nm145's re-stage needs 09's ruling); stratum S_sim2gap,
    world sim2, nm360 (S_newmains) cells on fold -1. DESCRIPTIVE, sealed; no outcome read except under 09's analysis note @ 9e17531cc."""
    t, rtf = "ilnm_s4rgap_sim2", "il-s4rgap__ilnm_s4rgap_sim2.json"
    v, r, rec_commit, man_raw, files, ex_j, staged, found = _verified(t, rtf, "il-s4rgap_attest.json", "ilnm", "match")
    folder = f"{r['staged_set']}__sim_attack"
    assert S4RGAP_FOLDER.fullmatch(folder) and r["stratum"] == "S_sim2gap" and [a["meta_arm"] for a in r["arms"]] == ["sim_attack@september"], (t, folder)
    assert len(staged) == 57 and all(k.startswith("sim2|") for k in staged) and sorted(k.split("|")[1].split("_")[0] for k in ex_j) == ["nm145", "nm242"], \
        (t, "57 staged sim2 runs; exactly nm145 and nm242 staged out", len(staged), sorted(ex_j))
    scored = {}
    for nm in sorted({y.strip("/").split("/")[-1] for y in sh(MODAL, "volume", "ls", WVOL, "scaleup").split() if y.strip("/").split("/")[-1].startswith("ilnm_")}):
        if "manifest_dt.json" in vol_names(f"scaleup/{nm}"):  # 43b: one listing call per folder
            dst = L / f"scored_{nm}.json"
            sh(MODAL, "volume", "get", "--force", WVOL, f"scaleup/{nm}/manifest_dt.json", str(dst))
            for j in json.loads(dst.read_text())["runs"]:
                scored.setdefault(j, []).append(nm)
    rep = sorted(j for j in staged if j in scored)
    if rep:
        raise SystemExit(f"{t}: {len(rep)} staged keys repeat earlier scored ilnm folders: {rep[:3]}")
    _no_dt_yet(folder)
    pf_extra = {"citations": {"rule": f"DT RULE Amendment 43 @ {FILL.get('rule43', 'PENDING')}", "spend": f"SPEND @ {FILL.get('spend_s4rgap', 'PENDING')}",
                              "approval": FILL.get("approval_s4rgap", "PENDING (Aswin's OK via 4f)"), "registration": "09's sim2 gap-fill part s4r §3 (regex)",
                              "records": "manifest @ 5cbd3354c; repro + labels @ 46afaeebf; attestation @ 0452b1c3d; coverage @ 75e640282"},
                "label": "DESCRIPTIVE, sealed; no outcome read except under 09's analysis note @ 9e17531cc", "stratum_note": "S_sim2gap (its own stratum)",
                "exclusions": {j: {"stager_string": s_, "class": "staged out by d9 (nm145's re-stage needs 09's ruling); never scored 0"} for j, s_ in ex_j.items()},
                "uniqueness": f"by key: {len(staged)} staged joins; none in an earlier scored ilnm folder ({len(scored)} keys read)",
                "folds": "nm360 (S_newmains) cells: fold -1"}
    _launch_arm(t, "arm_attack", rtf, v, r, rec_commit, man_raw, found, [("nm360", CELLS["nm360"])], pf_extra, folder)
    raise SystemExit(0)


if batch == "primary_v3":
    run_primary_v3()
if batch == "t196fill":
    run_t196fill()
if batch == "sp493":
    run_sp493(sys.argv[2:])
if batch == "s4rgap":
    run_s4rgap()
if batch == "oldsel":
    run_oldsel(sys.argv[2:])
if batch == "ug":
    run_ug()
if batch == "ha25f":
    run_ha25f(sys.argv[2:])
if batch == "il311_q4c":
    run_q4c(sys.argv[2:] or ["il311_q4c_run3", "il311_q4c_sim2"])
rx = re.compile(rf"^{re.escape(batch)}_sim(?:_s[0-9]+|_rlr(?:[2-9]|[1-9][0-9]+)?)?_v1(?:_try[0-9]+)?$")  # Amendment 34: _rlr roots (09 @ 91485ce80); 41: _rlr<k> (@ 502c1b05e)
routes = {}
for f in sorted(RT.glob("*__*.json")):
    r = json.loads(f.read_text())
    if rx.fullmatch(r.get("staged_set", "")):
        routes[r["target"]] = (f.name, r)
want = sys.argv[2:] or sorted(routes)
if not want or set(want) - set(routes):
    raise SystemExit(f"{batch}: no routed root, or an unrouted target {sorted(set(want) - set(routes))}")
subprocess.run(["git", "-C", str(DTS), "fetch", "-q", "origin", "aswin/q3-overnight-chain"], check=True)
rec_commit = git("rev-parse", "--short=9", "origin/aswin/q3-overnight-chain").decode().strip()
spec = {t: {"routing": routes[t][0], "attest": f"{routes[t][0].split('__')[0]}_attest.json", "records": rec_commit} for t in want}
vout = L / f"verify_{batch}.json"
subprocess.run(["python3", "-I", str(P2 / "verify_roots.py"), str(RT), str(DTS), str(vout), json.dumps(spec)], check=True, capture_output=True)
V = json.loads(vout.read_text())["roots"]
scored_folders = vol_ls(f"simh/{batch}/dt")  # create-only folders already under /simh/<batch>/dt (DT's); empty only if confirmed absent
scored = {}
for fo in sorted(scored_folders):
    dst = L / f"scored_{fo}.json"
    if "manifest_dt.json" not in vol_ls(f"simh/{batch}/dt/{fo}"):
        raise SystemExit(f"/simh/{batch}/dt/{fo}: a folder without manifest_dt.json (a partial write); stop")
    sh(MODAL, "volume", "get", "--force", WVOL, f"simh/{batch}/dt/{fo}/manifest_dt.json", str(dst))
    scored |= {j: fo for j in json.loads(dst.read_text())["runs"]}
seen, plan = {}, []
for t in want:
    v, (rname, r) = V[t], routes[t]
    folder = f"{r['staged_set']}__sim_honest"
    assert folder not in scored_folders, (t, "already scored")
    a = v["attestation"]
    assert a["pass"] and v["repro"]["ok"] and v["repro"]["bit_exact"] == v["repro"]["epochs"] and not v["needs_21g"] and all(v["format"].values()), (t, "gates")
    assert v["labels"].get("absent") is True or v["labels"]["match"], (t, "labels")
    assert v["stage"]["epochs_match"] and v["stage"]["shas_match"] and v["stage"]["exclusions_match_routing"], (t, "stage")
    assert v["manifest"]["counts_match_routing"] and not v["manifest"]["tasks_not_in_pinned_list"] and v["manifest"].get("nm_built_agrees", v["pool_key"] != "simh_nm360"), (t, "manifest")
    assert v["pool_key"] == ("simh311" if batch.startswith("simh311_") else "simh_nm360") and [x["meta_arm"] for x in r["arms"]] == ["sim_honest@september"], (t, "pool/arm")
    ptr = json.loads((RT / a["pointer_chain"][-1]).read_text())
    att = json.loads(git("show", f"{ptr['commit']}:{ptr['path']}"))["attestations"][r["arms"][0]["staged_sha"]]
    h = next(c for c in att if c["check"] == "longest_epoch_h200_gate")
    assert h["pass"] and h["detail"]["chunk_writers_recorded_h200"] is True and h["detail"]["unrecorded_chunks"] == [], (t, "writer gate")
    p, c = (x.strip() for x in r["manifest"].split(" @ "))
    man_raw = git("show", f"{c}:{p}")
    files = json.loads(man_raw)["files"]
    excl = r.get("staging_exclusions", {})
    staged = {f"{f['world']}|{f['task']}" for f in files if f["arm_dir"] == "sim_honest@september" and f"{f['launch']}/{f['rel']}#e0" not in excl}
    ex_j = {f"{f['world']}|{f['task']}": excl[f"{f['launch']}/{f['rel']}#e0"] for f in files if f"{f['launch']}/{f['rel']}#e0" in excl}
    # 09 @ 0f9927f3e item 6: sim-honest staging exclusions need no per-cell 09 ruling; each is cited by key with its stager string and source,
    # and classified by d9 from original-run evidence (25e / Append 35) in a committed evidence file before any outcome of its batch is read
    RL_RULED = {"13be06982": "infrastructure (provider rate limit, mid-trajectory); one sim-only replacement in its batch's _rlr root (09 @ 13be06982; grammar @ 91485ce80)",
                "c5420f2f5": "infrastructure (provider rate limit); one sim-only replacement in its batch's _rlr root (09 @ c5420f2f5; gate cleared by d9 @ 98ffa9e40)",
                "c10a14948": "terminal error, class unknown; one sim-only replacement in its batch's _rlr root (09 @ c10a14948)",
                "dd93a41cd": "terminal error, class unknown; one sim-only replacement in its batch's _rlr root (09 @ dd93a41cd: t293 §1 / standing rule §2, READ_LOG @ b474b5d35)",
                "81f75b763": "infrastructure (provider rate limit); one sim-only replacement in its batch's _rlr root (09 @ 81f75b763 §1)",
                "ed3b4cdff": "terminal error, class unknown; one sim-only replacement in its batch's _rlr root (09's standing rule, READ_LOG STANDING_RULE_ADD @ ed3b4cdff; evidence @ 9aef2c54a; root check @ bf28250cb)",
                "9843c6d7a": "terminal error, class unknown; one sim-only replacement in its batch's _rlr root (09's standing rule, READ_LOG STANDING_RULE_ADD @ 9843c6d7a; evidence @ fc362d007; root check @ 47c1ae3ef)"}
    RL_KEYS = {("simh_nm360_s3", "nm103_file_indexing"): "13be06982", ("simh_nm360_s3", "nm134_media_processing"): "c10a14948",
               ("simh311_q3", "t298_user_tracking"): "13be06982", ("simh311_q3", "t310_file_indexing"): "c5420f2f5",
               ("simh311_q2", "t271_slack_clone"): "c5420f2f5", ("simh311_q2", "t410_file_indexing"): "c5420f2f5",
               **{("simh311_q1", c_): "c5420f2f5" for c_ in ("t462_file_indexing", "t242_model_registry")},
               **{("simh311_q1", c_): "c10a14948" for c_ in ("t237_media_processing", "t300_user_tracking", "t389_model_registry")},
               **{("simh311_q4", c_): "c10a14948" for c_ in ("t361_file_indexing", "t411_media_processing", "t413_media_processing")},
               **{("simh_nm360_s4d", c_): "c10a14948" for c_ in ("nm107_media_processing", "nm258_tiktok")},
               ("simh311_q2", "t293_media_processing"): "dd93a41cd", ("simh_nm360_s2", "nm150_model_registry"): "dd93a41cd",
               **{("simh311_q2", c_): "81f75b763" for c_ in ("t240_model_registry", "t275_shared_dotfiles", "t457_media_processing")},
               ("simh_nm360_s4r", "nm353_web_scraping"): "ed3b4cdff", ("simh311_q1", "t320_auto_workflow"): "9843c6d7a"}
    ex_cite = {j: {"stager_string": s_, "source": f"{v['stage']['record']} @ {rec_commit} (c_gate.excluded)",
                   "class": RL_RULED[RL_KEYS[(batch, j.split("|")[1])]] if (batch, j.split("|")[1]) in RL_KEYS
                   else "classified from original-run evidence before the batch's first read (09 @ 0f9927f3e item 6; 25e / Append 35)"}
               for j, s_ in ex_j.items()}  # Amendment 34: 09's by-key rulings where they exist
    is_rlr = re.fullmatch(rf"{re.escape(batch)}_sim_rlr(?:[2-9]|[1-9][0-9]+)?_v1(?:_try[0-9]+)?", r["staged_set"]) is not None  # staged set decides (Codex 85); 41: k
    assert ("_rlr" in t) == is_rlr, (t, r["staged_set"], "target and staged set disagree on _rlr")
    if is_rlr:  # Amendment 37c: in an `_rlr` root the excluded run IS the replacement; the original stays a superseded status row
        assert all((batch, j.split("|")[1]) in RL_KEYS for j in ex_cite), (t, "an _rlr exclusion outside the replacement rulings")
        for j, c_ in ex_cite.items():
            c_["class"] = ("replacement run staging-excluded: the key has no scored rows and no fallback to its original (09 @ 91485ce80 §3: "
                           f"the replacement is the authoritative artifact; the original is a superseded status row). Original's ruling: {c_['class']}")
    assert all(j.startswith("simh_") for j in staged) and all(f["world"] in ("simh_311", "simh_nm360") for f in files if f["arm_dir"] == "sim_honest@september"), (t, "world")
    rep = {j: scored[j] for j in staged if j in scored} | {j: seen[j] for j in staged if j in seen}
    if rep:
        raise SystemExit(f"{t}: {len(rep)} staged keys repeat a scored folder or another root of this batch (25h section 11: each key exactly once)")
    seen |= {j: t for j in staged}
    found = [x for x in git("ls-tree", "-r", "--name-only", rec_commit, f"{B}/").decode().split() if x.split("/")[-1].startswith(f"extract_chain_{v['root']}_a")]
    assert len(found) == 1 and git("cat-file", "-e", f"{rec_commit}:{B}/{v['stage']['record']}") == b"", (t, found)
    plan.append((t, v, r, folder, man_raw, staged, ex_cite, found[0]))
print(now().strftime("%H:%M:%S"), batch, "verified by key (staged joins per root):", {p_[0]: len(p_[5]) for p_ in plan}, flush=True)
done = {}
for t, v, r, folder, man_raw, staged, ex_cite, exrec in plan:  # serial: one launch at a time (rel8's build race)
    claim = L / f"claim_{t}"
    if not DRY:
        try:  # a failed try's claim is archived by RENAMING it, never deleted
            claim.open("x").close()
        except FileExistsError:
            raise SystemExit(f"{t}: already submitted by another rel9 invocation") from None
    cap = cap_check()
    s_su, uns_su = su_line()
    rem = remaining_for_line(cap)  # every later DT launch is inside the rel9+ cap (Amendment 31)
    bound = round(s_su + ARM_WORST * (rem + len(uns_su)), 2)
    assert bound <= SU_STOP, (t, "scale-up line", bound)
    man = L / f"manifest_{t}.json"
    man.write_bytes(man_raw)
    msha = sha(man_raw)
    assert msha == v["manifest"]["sha256"], t
    cspec, csha = CELLS[BATCHES[batch]]
    cells = L / f"cells_{BATCHES[batch]}.json"
    cells.write_bytes(git("show", cspec))
    assert sha(cells.read_bytes()) == csha
    pf = {"message": t, "manifest_sha256": msha, "checked_at": now().isoformat(timespec="seconds"),
          "d9_gates": {"h200": True, "longest_epoch_gate": True, "corrected_render": True},
          "d9_gate_evidence": {"routing": rname, "routing_sha256": v["routing_sha256"], "attestation_commit": v["attestation"]["commit"],
                               "h200": f"attestation failing {{}}; longest_epoch_h200_gate pass (every_shard_h200, last_attempt_no_errors, chunk_writers_recorded_h200, "
                                       f"unrecorded_chunks []); extraction {v['extraction_records']} (single attempt, done_before 0)",
                               "longest_epoch_gate": f"repro {v['repro']['bit_exact']}/{v['repro']['epochs']} bit-exact, failing [], every shard one of each",
                               "corrected_render": f"format check pass {v['format']}; stage record {v['stage']['record']} @ {rec_commit} epochs/shas/exclusions match routing"},
          "settled_usd": s_su, "remaining_launches": rem, "unsettled_jobs": len(uns_su), "budget_line": "su311/nm360 DT scale-up line ($100 stop), conservative DT-app count",
          "rel9_cap": cap, "citations": {"grammar": "09 Append 25h section 11 @ fd163c4c7 (folders /simh/<batch>/dt/<d9 routed folder>/, create-only)",
                                         "rulings": "09 @ 0f9927f3e: item 5(a) DT schema as on the real side, no in-job run score; item 6 exclusions evidence-classified",
                                         "attestation_gate": "25h section 10 @ 5aa9f32bf (0d's attestation completes the pre-scoring gate)",
                                         "names": "09 format note @ fd40f4db4 (simh311) / 25g section 3 @ e6ded1811 (simh_nm360); d9 chain config @ 68be1d2bf",
                                         "rule": f"DT RULE Amendment 31 @ {FILL.get('rule31', 'PENDING')}", "spend": f"SPEND @ {FILL.get('spend_rel9', 'PENDING')}"},
          "exclusions": ex_cite,
          "uniqueness": f"by key: {len(staged)} staged joins, unique within {batch}; 0 repeat its scored folders {sorted(scored_folders)} or its other roots",
          "folds": "+311 / nm360 mains: fold -1 (verified unseen by the DT fold check); routing exactly as on the real side (25h section 11)",
          "firewall": "SIM arm: sealed, no thresholds, rates, catches or comparisons (the arm job computes none)", "stratum": r["stratum"], "staged_set": r["staged_set"]}
    assert "PENDING" not in json.dumps(pf) or DRY, (t, "a RULE or SPEND reference is still PENDING")
    pff = L / f"preflight_{t}.json"
    pff.write_text(json.dumps(pf, indent=1))
    args = ["--job", "arm", "--manifest", str(man), "--manifest-sha", msha, "--runs", v["root"], "--name", t, "--cells", str(cells), "--cells-sha", csha,
            "--eval-root", "evals_v1", "--stage-records", f"{rec_commit}:{B}/{v['stage']['record']}", "--root-records", f"{rec_commit}:{exrec}", "--preflight", str(pff)]
    cmd = ["env", "PYTHONPATH=src", "uv", "run", "--no-project", "--python", "3.12", "--with", "modal==1.6.1", "--with", "numpy",
           *(["python", str(P2 / "dryrun_main.py")] if DRY else ["modal", "run", "--detach", "experiments/2026-10-14_p2_dt_scaleup/run_dt_q4b.py"]), *args]
    batch_of = re.match(r"^(simh311_q[1-4]|simh_nm360_(?:s1|s2|s3|s4d|s4r|s5))_sim", r["staged_set"])[1]
    done[t] = launch_one(t, "simh_arm", cmd, bound, cap, [f"simh/{batch_of}/dt/{folder}"])
print("launched", done)
