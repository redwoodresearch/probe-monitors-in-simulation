"""Overnight chain (RULE.md here): one resumable, idempotent orchestrator, run on the laptop under caffeinate + nohup with the pinned
Modal 1.6.1 environment. Per scale-up batch (4 main quarters, then each quarter's simulated follow-up), as each launch ENDS:
  ended  -> box_files (on the box: exact files + B's counted attempts, hashes only) -> relay (box -> laptop cache -> Volume;
  sha256 checked on the Volume by a Modal CPU function; laptop copy deleted; per-file log) -> manifests + targets committed ->
  staging per target (stage_chain.py; its own format gate + coverage gate) -> the format check on 5 seeded staged runs per arm
  -> the per-shard reference arm (repro-c) -> H200 extraction (extract_chain.py, one shard per container) -> compare_repro
  (every shard's repro-c rows, incl. batch C's longest epoch, vs q4b_extract_c_v1) -> attempt labels (labels_chain.py)
  -> ACTIVATIONS/SPEND rows committed -> a routing file per arm (the agent relays it).
Never stops a run or a job on a failed read: every read retries with backoff and the loop goes on; a failed GATE stops that
batch loudly (status STOPPED with the reason) and the others continue. Reads launch.json and hashes; never opens a transcript.
State: <STATE>/state.json (outside git). Polls every >= 120 s. AWS: the redwood-sandbox profile, explicitly.

  cd <worktree> && PYTHONPATH=src AWS_PROFILE=redwood-sandbox nohup caffeinate -i \\
     /Users/aswinsurya/.cache/probe-shift-modal161/bin/python experiments/2026-10-14_q3_overnight_chain/chain.py <STATE> &
"""
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess as sp
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEVBOX_SH = REPO / "experiments/2026-10-07_phase2_attack_check/devbox.sh"
MODAL = "/Users/aswinsurya/.cache/probe-shift-modal161/bin/modal"
VOL = "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14"
REL = "experiments/2026-10-14_q3_overnight_chain"
BOXES = {"q1": ("aswin@i-002bbb22c185c0f07", "2026-10-14_154401_scaleup311_q1"), "q2": ("aswin@i-0c02a38abcde32bfb", "2026-10-14_154401_scaleup311_q2"),
         "q3": ("aswin@i-0cc7555acb95c694b", "2026-10-14_155838_scaleup311_q3"), "q4": ("aswin@i-07bce857a818b6f69", "2026-10-14_160246_scaleup311_q4")}
BATCH_CFG = HERE / "config" / "chain_batches.json"  # optional: more launches (e.g. the 360 new mains), each {q, box, run, pool}; a config change, not code
POOLS = {"su311": 300.0}  # each pool's own night reservation (d9/0d: the scale-up $300; the new mains get their own)
SNAP_S = 3600  # a running follow-up is snapshotted about every hour (d9/0d/Aswin: results asap)
# a running MAIN launch too (d9 10-14 18:50): its first snapshot at once, then about hourly; whole finished cells only (box_files.py)
SNAP_OF = {"main": "ms", "ms": "ms", "fu": "fus", "fus": "fus", "il": "ils", "ils": "ils"}  # each batch kind's snapshot kind
FU_KINDS, IL_KINDS, LIVE_SNAPS = ("fu", "fus", "il", "ils"), ("il", "ils"), ("fus", "ms", "ils")  # follow-up-like families; interleaved; live snapshots
POLL_S, FMT_N, EP_PER_SHARD, MAX_SHARDS, MIN_SHARDS = 150, 5, 6, 64, 4  # 6 epochs per shard (d9/Aswin 10-14 ~20:30: more H200s, same gates)
STEPS = ["ended", "boxfiles", "relay", "commit_targets", "stage", "format", "repro", "extract", "compare", "labels", "record", "ready", "hygiene",
         "provenance", "attest"]  # provenance and attest (09's, 10-14 ~19:45): batches done under an earlier version run them too
TERMINAL = ("attest", "STOPPED", "EMPTY")
os.environ["AWS_PROFILE"] = "redwood-sandbox"  # explicitly, never an inherited profile
os.environ["GIT_OPTIONAL_LOCKS"] = "0"  # read-only git (ours and the Modal clients' status/rev-parse) never takes the index lock
H200_USD_H = 5.3


# Concurrency (d9/0d/Aswin 10-14 ~20:45): one worker thread per batch. STATE is held by whoever runs chain code and is released only
# around waits that touch no state (subprocesses, sleeps, Modal file I/O); every state read, mutation and save happens under it.
# GIT serializes the worktree's git commands; the order is always GIT before STATE, and a push never holds STATE.
STATE, GIT, LOGLOCK, RELAY_SEM = threading.Lock(), threading.Lock(), threading.Lock(), threading.Semaphore(2)
DRAIN = threading.Event()  # SIGTERM: finish the steps in progress, start no new step, then exit (a restart between steps)
_tl = threading.local()


@contextmanager
def locked():
    STATE.acquire()
    _tl.held = True
    try:
        yield
    finally:
        _tl.held = False
        STATE.release()


@contextmanager
def unlocked():
    """Release STATE (if this thread holds it) for a wait that touches no state."""
    held = getattr(_tl, "held", False)
    if held:
        _tl.held = False
        STATE.release()
    try:
        yield
    finally:
        if held:
            STATE.acquire()
            _tl.held = True


def now() -> str:
    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S PT")


class Gate(Exception):
    """A failed gate: that batch stops loudly; the chain goes on with the others."""


class Wait(Exception):
    """Not now (e.g. the reservation is full): the batch waits, says so in the status, and retries next pass."""


class State:
    def __init__(self, d: Path):
        self.d, self.p = d, d / "state.json"
        d.mkdir(parents=True, exist_ok=True)
        self.s = json.loads(self.p.read_text()) if self.p.exists() else {"batches": {}, "log": []}

    def save(self):
        tmp = self.p.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.s, indent=1))
        tmp.replace(self.p)

    def log(self, msg: str):
        line = f"{now()}  {msg}"
        with LOGLOCK:
            print(line, flush=True)
            with (self.d / "chain.log").open("a") as f:
                f.write(line + "\n")


def run(cmd: list, *, input_: bytes | None = None, timeout: int = 3600, env: dict | None = None, cwd: Path = REPO, release: bool = False) -> sp.CompletedProcess:
    with unlocked() if release else nullcontext():  # release: a subprocess that touches no state (modal run, ssh, rsync)
        # its own process group: on a timeout the WHOLE group dies (an SSM proxy grandchild holding the pipe hung sp.run on 10-14 21:38)
        p = sp.Popen(cmd, stdin=sp.PIPE if input_ is not None else sp.DEVNULL, stdout=sp.PIPE, stderr=sp.PIPE,
                     env={**os.environ, **(env or {})}, cwd=cwd, start_new_session=True)
        try:
            out, err = p.communicate(input_, timeout=timeout)
        except sp.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                p.communicate(timeout=30)
            except sp.TimeoutExpired:  # a process outside the group still holds a pipe: close our ends and move on
                for f in (p.stdout, p.stderr):
                    f.close()
                p.wait(timeout=30)
            raise RuntimeError(f"{os.path.basename(str(cmd[0]))}: timed out after {timeout} s; its process group was killed (a retryable transport failure)") from None
        return sp.CompletedProcess(cmd, p.returncode, out, err)


def retry(f, what: str, st: State, tries: int = 8, base: int = 30):
    """A read with retries and backoff; after the last try it raises, and the main loop retries the step on its next pass."""
    for i in range(tries):
        try:
            return f()
        except Gate:
            raise
        except Exception as e:  # a failed read: log, back off, retry; never act on it
            st.log(f"retry {what} ({i + 1}/{tries}): {type(e).__name__}: {str(e)[:200]}")
            with unlocked():
                time.sleep(min(base * 2 ** i, 600))
    raise RuntimeError(f"{what}: failed after {tries} tries")


def root_of(b: dict) -> str:
    return b.get("root", "phase2/runs")  # relative to the box user's home


def box(box_: str, cmd: str, input_: bytes | None = None, timeout: int = 900) -> bytes:
    r = run(["bash", str(DEVBOX_SH), cmd], input_=input_, timeout=timeout, env={"DEVBOX": box_}, release=True)
    if r.returncode == 3:  # box_files.py's validation failure (not a read failure)
        raise Gate(f"box {box_}: {r.stderr.decode()[-400:]}")
    if r.returncode != 0:
        raise RuntimeError(f"box {box_}: rc {r.returncode}: {r.stderr.decode()[-300:]}")
    return r.stdout


def git(*a: str, check: bool = True) -> str:
    r = run(["git", *a], timeout=300)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(a)}: {r.stderr.decode()[-300:]}")
    return r.stdout.decode().strip()


BRANCH, HTTPS = "aswin/q3-overnight-chain", ("-c", "url.https://github.com/.insteadOf=git@github.com:")


def git_locked(*a: str) -> str:
    """A git read that must not interleave with a commit or a push's merge (GIT before STATE)."""
    with unlocked():
        GIT.acquire()
    try:
        return git(*a)
    finally:
        GIT.release()


def commit(paths: list[str], msg: str, st: State, prepare=None):
    """Commit locally (GIT, then STATE), then push WITHOUT STATE (its own GIT per attempt, backoff with no lock held). `prepare`
    writes the tracked files being committed inside the same GIT section, so no git read ever sees them half-done."""
    with unlocked():
        GIT.acquire()  # order: GIT before STATE (unlocked() re-takes STATE while GIT is held)
    try:
        if prepare is not None:
            prepare()
        other = set(git("diff", "--cached", "--name-only").split()) - set(paths)
        if other:  # never commit what something else staged
            raise Gate(f"refusing to commit: unrelated staged files {sorted(other)[:5]}")
        try:
            git("add", "--", *paths)
            if git("diff", "--cached", "--name-only"):
                git("commit", "-q", "--only", "-m", msg + "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>", "--", *paths)
        except Exception:  # never leave this commit's paths staged for another batch's commit to trip on
            git("reset", "-q", "--", *paths, check=False)
            raise
        head = git("rev-parse", "HEAD")
    finally:
        GIT.release()
    with unlocked():
        push_verified(head, st)
    return head


def push_verified(head: str, st: State):
    """Push HEAD and confirm the remote branch contains `head`; on a rejected push, fetch and MERGE (never rebase: recorded shas stay
    valid) and retry. Holds only GIT, and only during each attempt."""
    for i in range(8):
        with GIT:
            r = run(["git", *HTTPS, "push", "-q", "origin", f"HEAD:{BRANCH}"], timeout=300)
            if r.returncode != 0 and ("rejected" in r.stderr.decode() or "fetch first" in r.stderr.decode()):
                f = run(["git", *HTTPS, "fetch", "-q", "origin", BRANCH], timeout=300)
                if f.returncode == 0:
                    run(["git", "merge", "-q", "--no-edit", "FETCH_HEAD"], timeout=300)
            ls = run(["git", *HTTPS, "ls-remote", "origin", f"refs/heads/{BRANCH}"], timeout=300)
            remote = ls.stdout.decode().split()[0] if ls.returncode == 0 and ls.stdout.split() else None
            if r.returncode == 0 and remote and run(["git", "merge-base", "--is-ancestor", head, remote], timeout=60).returncode == 0:
                return
        st.log(f"retry git push ({i + 1}/8): rc {r.returncode}: {r.stderr.decode()[-200:]}")
        time.sleep(min(30 * 2 ** i, 600))
    raise RuntimeError(f"push of {head[:10]} did not land after 8 tries")


GATE_MARKS = ("format gate failed", "format check FAILED", "positive control FAILED", "staging stopped", "c gate:", "fail the bar",
              "negative control PASSED", "differs from", "several counted attempts", "missing from the Volume")


SHARD_WORST_H = (30 + 10 + 1) / 60  # extract_chain.py: 30-min timeout + 10-min startup bound + 1-min scaledown, all billable
CPU_WORST_USD = {"stage": 4.0, "format": 0.5, "repro": 0.5, "compare": 0.5, "labels": 0.5, "hygiene": 1.0, "provenance": 1.0, "stats": 0.5}
SIM_ARM = "sim_attack@september"
CORE_POOLS = ("su311", "nm360", "il")  # a follow-up with another simulated arm lives in its own pool (no file name shared with these)


def sim_arm_of(b: dict) -> str:
    """A follow-up family's simulated arm folder (config `sim_arm`; d9 10-15: the sim-honest +311 launches' sim_honest@september, the
    ungrounded arm's sim_attack@september_ungrounded); every other batch: sim_attack@september."""
    return b.get("sim_arm") or SIM_ARM


def sim_world_of(b: dict) -> str:
    return b.get("sim_world") or "sim"


def sim_is_honest(b: dict) -> bool:
    """A simulated HONEST arm: no attack labels and no approach provenance (honest arms are exempt, Append 21 section 5)."""
    return sim_arm_of(b).startswith("sim_honest")
LIVE_LOGS = (".launcher.log", ".docker_events.jsonl", ".resources.log")  # sibling logs a box may still append to after the launch ends
DROPPED = [{"cell": "t278_model_registry", "stage": "preflight", "reason": "sample construction failed at preflight (the 196's t88/t169 type)"}]  # 48 x 2-CPU parts x 30-min timeout + assemble


def reserve(st: State, b: dict, what: str, usd: float):
    """The batch's pool reservation (scale-up $300; the new mains their own): booked spend (CPU steps at their worst case;
    extractions at their settlement) plus in-flight extraction worst cases. A step that would pass it waits and says so; it never
    starts over budget."""
    pool, cap = b["pool"], st.s["pools"][b["pool"]]
    booked, inflight = st.s.setdefault("booked_usd", {}).setdefault(pool, 0.0), st.s.setdefault("inflight_usd", {}).setdefault(pool, {})
    if booked + sum(inflight.values()) + usd > cap:
        b["waiting_budget"] = f"{pool} {what}: booked ${booked:.0f} + in flight ${sum(inflight.values()):.0f} + this ${usd:.0f} > ${cap:.0f}"
        raise Wait(b["waiting_budget"])
    b.pop("waiting_budget", None)
    if what.startswith("extract"):
        inflight[f"{b['id']}:{what}"] = usd
    else:
        st.s["booked_usd"][pool] = booked + usd
    st.save()


def settle_extract(st: State, b: dict, t: str, job: dict, force_worst: bool = False):
    """An extraction attempt ended: release its worst case and book its cost. Per shard with a record: its runtime plus the full
    startup and scaledown bounds (unmeasured, so kept at their worst); a failed shard, or no record at all: the full worst case."""
    new = new_records(job)
    if new and not force_worst:
        shards = valid_record(new[-1])["shards"]
        cost = sum(((s["seconds"] / 3600 + (10 + 1) / 60) if isinstance(s, dict) and "seconds" in s else SHARD_WORST_H) for s in shards) * H200_USD_H
    else:
        cost = job["worst"]
    pool = b["pool"]
    st.s["inflight_usd"][pool].pop(f"{b['id']}:extract {t} a{job['att']}", None)
    st.s["booked_usd"][pool] = st.s["booked_usd"].get(pool, 0.0) + cost
    st.log(f"{b['id']}: extract {t} attempt {job['att']} ended; booked ${cost:.2f} ({pool} total ${st.s['booked_usd'][pool]:.2f})")


def valid_record(p: Path) -> dict | None:
    """An extraction record that parses and lists its shards (a write cut short by a kill or a sleep is not one)."""
    try:
        r = json.loads(p.read_text())
        return r if isinstance(r.get("shards"), list) else None
    except Exception:
        return None


def new_records(job: dict) -> list:
    return [p for p in sorted((HERE / "results").glob(f"extract_chain_{job['run']}_a{job['att']}_20*.json"))
            if str(p) not in set(job["before"]) and valid_record(p) is not None]


def attempt_over(st: State, b: dict, t: str, job: dict) -> bool:
    """The attempt's laptop client has exited, or never had its pid saved. Its shard fan-out runs in a Modal orchestrator
    (extract_chain.py), which a laptop sleep or a chain restart does not cancel: re-attach instead of relaunching over it.
    True only when the attempt is KNOWN to be over: a valid record is here, or is fetched from the Volume (result_a<n>.json) or from
    the finished call; or the call ended (timed out, cancelled, failed remotely, output expired); or no orchestrator started within
    15 min of the launch. False while it still runs. An unknown status (a failed read) raises: the job is kept and retried, and
    after 10 such passes in a row the batch stops loudly rather than risk two attempts writing one root."""
    import modal

    if new_records(job):
        return True
    stamp = lambda: datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")  # noqa: E731

    def publish(raw: bytes, how: str):  # write-once and atomic: a temporary file, then a rename to a name that does not exist
        dest = HERE / "results" / f"extract_chain_{job['run']}_a{job['att']}_{stamp()}.json"
        while dest.exists():
            time.sleep(1)
            dest = HERE / "results" / f"extract_chain_{job['run']}_a{job['att']}_{stamp()}.json"
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(raw)
        tmp.rename(dest)
        st.log(f"{b['id']}: extract {t} a{job['att']}: record {how} ({dest.name})")

    vol = modal.Volume.from_name(VOL, version=2)

    def vol_json(path: str):
        try:
            with unlocked():
                return json.loads(b"".join(vol.read_file(path)))
        except (FileNotFoundError, modal.exception.NotFoundError):
            return None

    res = vol_json(f"{job['run']}/result_a{job['att']}.json")
    if res is not None and isinstance(res.get("shards"), list):
        publish(json.dumps(res, indent=1).encode(), "fetched from the Volume")
        return True
    log = Path(job["log"]).read_text() if Path(job["log"]).exists() else ""
    cid = next((json.loads(x)["orchestrator_call_id"] for x in log.splitlines() if x.startswith('{"orchestrator_call_id"')), None)
    if cid is None:
        marker = vol_json(f"{job['run']}/orchestrator_a{job['att']}.json")
        cid = marker.get("orchestrator_call_id") if marker else None
    if cid is None:
        if "t" not in job:  # a client from before the orchestrator (its fan-out lived in the client, which has exited): over
            return True
        if time.time() - job["t"] > 30 * 60:  # never infer that an unacknowledged orchestrator ended: stop loudly instead
            raise Gate(f"extract {t} a{job['att']}: no orchestrator call id (client log or Volume marker) 30 min after launch; "
                       "stopping rather than risk a second attempt over a pending one")
        return False
    try:
        with unlocked():
            out = modal.FunctionCall.from_id(cid).get(timeout=0)
    except (modal.exception.OutputExpiredError, modal.exception.FunctionTimeoutError) as e:  # terminal (both subclass TimeoutError)
        st.log(f"{b['id']}: extract {t} a{job['att']}: orchestrator {cid} over ({type(e).__name__})")
        return True
    except (modal.exception.TimeoutError, TimeoutError):
        job.pop("unknown", None)
        return False  # still running on Modal: wait, never launch over it
    except (modal.exception.InputCancellation, modal.exception.RemoteError) as e:
        st.log(f"{b['id']}: extract {t} a{job['att']}: orchestrator {cid} ended ({type(e).__name__})")
        return True
    except Exception as e:  # unknown: keep the job; never assume it ended
        job["unknown"] = job.get("unknown", 0) + 1
        if job["unknown"] >= 10:
            raise Gate(f"extract {t} a{job['att']}: orchestrator {cid} status unknown 10 times ({type(e).__name__}); stopping rather than risk overlap")
        raise RuntimeError(f"extract {t} a{job['att']}: orchestrator status unknown ({type(e).__name__}); retrying") from None
    publish(json.dumps(out, indent=1).encode(), "taken from the finished orchestrator call")
    return True


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # a finished child of this process is a zombie until reaped
        return os.waitpid(pid, os.WNOHANG) == (0, 0)
    except ChildProcessError:
        return True  # not our child (a chain restart): alive as far as kill(0) says


def modal_run(args: list, st: State, what: str, timeout: int = 4 * 3600, b: dict | None = None, detach: bool = False) -> str:
    """A failed gate (its own message in the output) stops the batch; any other failure (client, network, Modal) is retried on
    the next pass, and stops the batch only after 4 failures of the same step. `detach`: the entrypoint spawns a remote driver
    that a disconnect must not cancel."""
    r = run([MODAL, "run", *(["--detach"] if detach else []), *args], timeout=timeout, env={"PYTHONPATH": str(REPO / "src")}, release=True)
    out = (r.stdout.decode() + r.stderr.decode())
    (st.d / "modal_logs").mkdir(exist_ok=True)
    (st.d / "modal_logs" / f"{what}_{datetime.now().strftime('%H%M%S')}.log").write_text(out)
    if r.returncode != 0:
        if any(m in out for m in GATE_MARKS):
            raise Gate(f"{what}: gate failed: {out[-800:]}")
        if b is not None:  # counted and shown in the status file; never turned into a stop
            b.setdefault("fails", {})[what] = b.get("fails", {}).get(what, 0) + 1
        raise RuntimeError(f"{what}: modal run exited {r.returncode} (retrying): {out[-300:]}")
    return out


# ---- the steps ------------------------------------------------------------------------------------------------------------
EP2_SEALED = "SEALED: 09 Append 38 §4 pool-key artifact pending (no epoch-2 outcome is read before 09's pool-key artifact is committed and verified by key; @ d9013b2e7)"
RLR_TAG = re.compile(r"sim_rlr(?:[2-9]|[1-9][0-9]+)?")  # 09 @ 91485ce80, @ 502c1b05e: k = 1 is _rlr; k >= 2 is _rlr<k>; _rlr0 / _rlr1 invalid
RLR_TABLE = re.compile(r"run_table_rlr(?:[2-9]|[1-9][0-9]+)?")


def cell_of(task: str) -> str:
    return task[:-3] if task.endswith("_r2") else task


def fam(x: dict) -> str:
    """A follow-up family: the follow-up batch and its snapshots ("<fu id>_s<k>"). A quarter can have two (the Q2 rerun)."""
    return x.get("family") or (x["id"] if x["kind"] in ("fu", "il") else x["id"].rsplit("_s", 1)[0])


def bare_id(run: str) -> str:
    """A launch's id as every record names it: the run folder's name (the T196 layout's runs are relayed as runs/<id>; d9 10-15)."""
    return run.rsplit("/", 1)[-1]


def qtok(b: dict) -> str:
    """The quarter token in a follow-up family's target and file names ("q2r" for the Q2 rerun; else the quarter)."""
    return b.get("qtok", b["q"])


def snaps_of(st: State, b: dict) -> list:
    """The snapshot batches of b's launch (b's own family: a main launch's, or a follow-up's), b itself excluded."""
    if b["kind"] == "ep2":  # final only: no snapshots
        return []
    if b["kind"] in FU_KINDS:
        return [x for x in st.s["batches"].values() if x["kind"] == SNAP_OF[b["kind"]] and x["pool"] == b["pool"] and fam(x) == fam(b) and x["id"] != b["id"]]
    return [x for x in st.s["batches"].values() if x["kind"] == SNAP_OF[b["kind"]] and x["q"] == b["q"] and x["pool"] == b["pool"] and x["id"] != b["id"]]


def exclusions_of(b: dict) -> dict:
    """Cells of b's follow-up launch recorded as EXCLUDED (results/hygiene/exclusions_<pool>_<batch>.json): cell -> record."""
    p = HERE / "results" / "hygiene" / f"exclusions_{b['pool']}_{b['id']}.json"
    return {r["cell"]: r for r in json.loads(p.read_text())["excluded"]} if p.exists() else {}


def main_of(st: State, b: dict) -> dict | None:
    """The main batch whose launch holds b's honest donors (an interleaved batch names it: it sits in another pool). None for the
    T196 interleaved launch, whose donors are its own reference copies (donor_index; relayed with the batch)."""
    if b.get("donor_index") or b["kind"] == "ep2":  # T196 IL / ug: own reference copies; epoch-2 real honest: no main at all
        return None
    if b.get("main_batch"):
        return st.s["batches"][b["main_batch"]]
    return next(x for x in st.s["batches"].values() if x["kind"] == "main" and x["q"] == b["q"] and x["pool"] == b["pool"])


def main_relayed(st: State, b: dict) -> bool:
    """A follow-up's honest donors come from its main launch: the remainder's relay and every main snapshot's must be done."""
    mb = main_of(st, b)
    if mb is None:  # donors relayed by the batch itself
        return True
    return all("relay" in x["done"] for x in [mb, *snaps_of(st, mb)] if x["status"] != "EMPTY")  # a STOPPED unrelayed main blocks (its donors are missing)


def step_ended(b: dict, st: State):
    """The launch has ENDED: its launch.json says finished, seen twice >= POLL_S apart. (For a follow-up, first find its run.) A
    snapshot of a running launch needs no end."""
    if b["kind"] in LIVE_SNAPS:
        return True
    if b["kind"] == "ep2":  # epoch-2 real honest (09 Append 38): ended = its key list is committed (built from the launch's relayed, verified files)
        return sp.run(["git", "cat-file", "-e", f"HEAD:{REL}/{b['keylist']}"], cwd=HERE, capture_output=True).returncode == 0
    if b["kind"] == "fu" and not b.get("run"):
        out = box(b["box"], f"cd ~/{root_of(b)} && for f in *_fu.meta.json; do [ -f \"$f\" ] && echo \"$f\" && cat \"$f\" && echo; done").decode()
        found = None
        for line in out.splitlines():
            if line.startswith("{"):
                try:
                    m = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if m.get("run1_launch") == b["main_run"] and m.get("sim_launch"):
                    found = m["sim_launch"]
        if not found:
            return False
        b["run"] = found
        st.log(f"{b['id']}: follow-up run is {found}")
    # launch.json "finished" is written by the frozen launcher as it exits; required on two passes >= POLL_S apart (settled). No pgrep:
    # a pgrep -f over ssh matches its own command line (the waiter self-match trap).
    probe = ("python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(\"finished\" if d.get(\"finished\") else \"running\")' "
             f"~/{root_of(b)}/{b['run']}/launch.json")
    o = box(b["box"], probe).decode().split()
    if o != ["finished"]:
        b.pop("finished_seen", None)
        b.pop("finished_seen_t", None)
        return False
    if not b.get("finished_seen_t"):
        b["finished_seen"], b["finished_seen_t"] = now(), time.time()
        st.save()
        return False
    return time.time() - b["finished_seen_t"] >= POLL_S


def ep2_boxfiles(b: dict, st: State) -> bytes:
    """An epoch-2 launch's 'box read' without the box: the counted rows of its committed key list (folders only; status rows never
    stage), each .eval's bytes from the committed relay allowlist the key list names. Keys and hashes only."""
    kl = json.loads(git("show", f"HEAD:{REL}/{b['keylist']}"))
    if kl["launch"] != b["run"]:
        raise Gate(f"{b['id']}: key list is for {kl['launch']}, not {b['run']}")
    apath = kl["inputs"]["allowlist"].split(" @ ")[0]
    allow = json.loads(git("show", f"HEAD:{apath}"))
    size = {f["rel"]: f["bytes"] for f in allow["files"]}
    rows = []
    for folder, keys in kl["folders"].items():
        arm = folder.rsplit("__", 1)[1]
        for r in keys:
            a_, cell = r["key"].split("|", 1)
            if a_ != arm or r["eval_sha256"] != next(f["sha256"] for f in allow["files"] if f["rel"] == r["eval"]):
                raise Gate(f"{b['id']}: {r['key']} disagrees with its folder or the allowlist")
            rows.append({"launch": b["run"], "rel": r["eval"], "path": f"{b['run']}/{r['eval']}", "sha256": r["eval_sha256"], "bytes": size[r["eval"]],
                         "arm_dir": arm, "task": cell, "launch_task": r["launch_task"], "attempt_index": r["attempt_index"], "attempt_kind": r["attempt_kind"],
                         "policy_called": True, "state": "done", "world": arm, "epoch": "ep2", "launch_id": b["run"], "stratum": "ep2"})
    man = {"note": f"epoch-2 real honest {b['run']} (09 Append 38): the counted rows of {b['keylist']}", "counts": {"files": len(rows),
           "by_arm": {a: sum(f["arm_dir"] == a for f in rows) for a in ("honest_a", "honest_b")}}, "counted_attempts_without_eval": {"main": []},
           "status_rows": kl["status_rows"], "files": sorted(rows, key=lambda f: f["path"])}
    d = {"kind": "ep2", "run": b["run"], "facts": {"roster": sorted({f["task"] for f in rows}), "not_counted": [], "policy": None},
         "manifests": {"main": man}, "excluded_earlier_snapshots": [], "members": sorted({f["task"] for f in rows}), "main_run": None, "references": [],
         "markers": [], "launch_subset": "{}", "files": [{"path": f["path"], "bytes": f["bytes"], "sha256": f["sha256"]} for f in man["files"]],
         "counted_without_eval": []}
    return json.dumps(d).encode()


def step_boxfiles(b: dict, st: State):
    if b["kind"] == "ep2":
        raw = ep2_boxfiles(b, st)
        d = json.loads(raw)
        b["members"] = d["members"]
        (st.d / f"boxfiles_{b['id']}.json").write_bytes(raw)
        b["files"], b["bytes"], b["counted_without_eval"] = len(d["files"]), sum(f["bytes"] for f in d["files"]), 0
        st.log(f"{b['id']}: {b['files']} counted epoch-2 rows from {b['keylist']}")
        return
    snaps = snaps_of(st, b)
    if b["kind"] in ("main", "fu", "il") and any("boxfiles" not in x["done"] and x["status"] != "STOPPED" for x in snaps):
        raise Wait("the final batch waits for every snapshot's membership")
    prior = sorted({m for x in snaps if "boxfiles" in x["done"] for m in x.get("members", [])})  # cells (never taken twice)
    args = [b["kind"], b["run"], b.get("main_run") or "-", ",".join(prior) or "-", str(b.get("snap", "final"))]
    env = f"CHAIN_ROOT=$HOME/{root_of(b)} CHAIN_STRATUM={b.get('stratum', 'S_new')} "
    if b["kind"] in ("fu", "fus") and (b.get("sim_arm") or b.get("sim_world")):  # a follow-up family's own simulated arm and world
        env += f"CHAIN_SIM_ARM={shlex.quote(sim_arm_of(b))} CHAIN_SIM_WORLD={shlex.quote(sim_world_of(b))} "
    raw = box(b["box"], env + "python3 - " + " ".join(args), input_=(HERE / "box_files.py").read_bytes(), timeout=1800)
    d = json.loads(raw)
    if b.get("only_cells"):  # a staging retry of named cells (il-s4rgap-t2), or one group of a shared sim launch (X/Y): the full read kept beside it
        (st.d / f"boxfiles_{b['id']}.full.json").write_bytes(raw)
        il = b["kind"] in IL_KINDS
        d = only_cells_view(d, set(b["only_cells"]), il)
        raw = json.dumps(d).encode()
        st.log(f"{b['id']}: box read narrowed to only_cells {sorted(b['only_cells'])} (members {d['members']}; box donor rows dropped)" if il else
               f"{b['id']}: box read narrowed to group {b.get('group')} ({len(b['only_cells'])} cells; {len(d['members'])} members; their index donor rows kept)")
    b["members"] = d["members"]  # cells; persisted with this step, so the next snapshot and the remainder exclude them
    if b["kind"] in LIVE_SNAPS and not d["members"]:
        b["empty"] = True
    (st.d / f"boxfiles_{b['id']}.json").write_bytes(raw)
    b["files"] = len(d["files"])
    b["bytes"] = sum(f["bytes"] for f in d["files"])
    b["counted_without_eval"] = len(d["counted_without_eval"])
    st.log(f"{b['id']}: {b['files']} files, {b['bytes']} bytes; counts {json.dumps({k: v['counts'] for k, v in d['manifests'].items()})}")


def routing_dir(st: State, b: dict) -> Path:
    """Where a batch's routing files go: the routing dir, or routing/pending/ for a re-stage that stays non-routable until its by-key byte diff
    passes (route_pending, set by a state edit; 09 via d9 10-16 ~01:45): the live routed files are never overwritten before the gate (Codex r109)."""
    d = st.d / "routing" / ("pending" if b.get("route_pending") else "")
    d.mkdir(parents=True, exist_ok=True)
    return d


def only_cells_view(d: dict, cells: set, il: bool = True) -> dict:
    """An interleaved box read narrowed to `cells`: its run rows, members, roster, launch records and .eval files for those cells only, and
    NO donor rows (a staging retry takes its donors from the main's pinned manifests, option (b)). With il False (one group of a shared sim
    launch, S_paper493 X/Y): every row of those cells, its index donor rows included. A cell the launch does not list stops."""
    if not cells <= set(d["facts"]["roster"]):
        raise Gate(f"only_cells {sorted(cells - set(d['facts']['roster']))} are not in the launch's roster")
    keep = {f["path"] for m in d["manifests"].values() for f in m["files"] if f["task"] in cells and (not il or f["world"] in ("run3", "sim2"))}
    ne = {x["path"]: x["task"] in cells for x in d["counted_without_eval"]}  # a counted attempt without an .eval: kept only for its own cell (Codex r104)
    for m in d["manifests"].values():
        if any(p not in ne for v in m["counted_attempts_without_eval"].values() for p in v):
            raise Gate("a manifest's counted attempt without an .eval is not in the box read's counted_without_eval")
        m["counted_attempts_without_eval"] = {k: [p for p in v if ne[p]] for k, v in m["counted_attempts_without_eval"].items()}
        m["files"] = [f for f in m["files"] if f["path"] in keep]
        m["counts"] = {"files": len(m["files"]), "by_arm": {a: sum(f["arm_dir"] == a for f in m["files"]) for a in sorted({f["arm_dir"] for f in m["files"]})}}
        m["only_cells"] = sorted(cells)
    d["files"] = [f for f in d["files"] if not f["path"].endswith(".eval") or f["path"] in keep]
    if not il:  # the group's own reference names only
        d["references"] = [x for x in d.get("references", []) if x.removesuffix(".eval") in cells]
    d["members"] = [x for x in d["members"] if x.split("/", 1)[-1] in cells]
    d["counted_without_eval"] = [x for x in d["counted_without_eval"] if x["task"] in cells]
    d["facts"]["roster"] = sorted(cells)
    d["facts"]["not_counted"] = [x for x in d["facts"]["not_counted"] if x["task"] in cells]
    sub = json.loads(d["launch_subset"])
    sub["runs"] = [r for r in sub["runs"] if (r["task"][:-3] if r["task"].endswith("_r2") else r["task"]) in cells]
    d["launch_subset"] = json.dumps(sub, sort_keys=True)
    d["only_cells"] = sorted(cells)
    return d


def group_cells(F: dict) -> list:
    """The cells of a sim follow-up family that runs ONE group of a shared launch (S_paper493 X/Y: g3/g6/g7; 09 via d9 10-16 ~01:00): the pinned
    task file's `groups` entries equal to the family's group (commit + sha256 asserted). The launch's other groups are its other families."""
    g = F["group_cells"]
    raw = sp.run(["git", "show", f"{g['commit']}:{g['path']}"], capture_output=True, check=True, cwd=REPO).stdout
    if hashlib.sha256(raw).hexdigest() != g["sha256"]:
        raise SystemExit(f"{F['id']}: the groups map {g['path']} @ {g['commit']} differs from its pinned sha256")
    groups = json.loads(raw)["groups"]
    cells = sorted(c for c, x in groups.items() if x == g["group"])
    if not cells:
        raise SystemExit(f"{F['id']}: group {g['group']} has no cells in {g['path']}")
    return cells


def check_group_families(fus: list):
    """Every family on a launch that has a group family is a group family pinning the same task file; the launch's groups map is
    partitioned exactly (families disjoint by cell, their union = every cell of the map)."""
    by_run = {}
    for F in fus:
        by_run.setdefault((F["box"], F["run"]), []).append(F)
    for (bx, run), Fs in by_run.items():
        if not any(F.get("group_cells") for F in Fs):
            continue
        pins = {json.dumps({k: F.get("group_cells", {}).get(k) for k in ("commit", "path", "sha256")}, sort_keys=True) for F in Fs}
        if len(pins) != 1 or not all(F.get("group_cells") for F in Fs):
            raise SystemExit(f"{run}: every family of a grouped launch must pin the same groups map ({[F['id'] for F in Fs]})")
        g = Fs[0]["group_cells"]
        raw = sp.run(["git", "show", f"{g['commit']}:{g['path']}"], capture_output=True, check=True, cwd=REPO).stdout
        allc = set(json.loads(raw)["groups"])
        seen = [c for F in Fs for c in group_cells(F)]
        if len(seen) != len(set(seen)) or set(seen) != allc:
            raise SystemExit(f"{run}: its group families do not partition the groups map ({len(seen)} cells, {len(set(seen))} distinct, map {len(allc)})")


RELAY_META = ("launch.json", "gate1_final.json")  # the metadata the chain's Modal steps read from the Volume (labels, provenance) or a launch's gate


def relay_allowed(src: str, named: set, run: str, generated: bool = False) -> bool:
    """The relay ALLOWLIST (d9 10-15 16:40): an .eval file the batch's own manifests name, or one of the launch's own metadata files
    directly in its run folder (<run>/launch.json, <run>/gate1_final.json, <run>/upload_manifest_*.tsv), or a snapshot's generated
    launch.json. Nothing else, ever: in particular no cells.json or approach file, and no path with '..' or a leading '/'."""
    parts = src.split("/")
    if src.startswith("/") or any(x in ("", ".", "..") for x in parts):
        return False
    name = parts[-1]
    if name == "cells.json" or "approach" in name.lower():
        return False
    if generated:
        return name == "launch.json"
    if name.endswith(".eval"):
        return src in named
    return src.rsplit("/", 1)[0] == run and (name in RELAY_META or (name.startswith("upload_manifest_") and name.endswith(".tsv")))


def relay_plan(b: dict, d: dict, cache: Path) -> tuple[list, list]:
    """The relay items of a batch (box read d) and the refused files (names): every candidate passes relay_allowed or is refused."""
    named = {f["path"] for m in d["manifests"].values() for f in m["files"]}  # the .eval files the batch's own manifests name
    cand = [{"src": r["path"], "dest": f"evals_v1/{r['path']}", "sha256": r["sha256"], "bytes": r["bytes"]} for r in d["files"]]
    if b["kind"] in LIVE_SNAPS:
        cand += [{"src": r["path"], "dest": f"snaps/{b['id']}/{r['path']}", "sha256": r["sha256"], "bytes": r["bytes"]}
                 for r in d["files"] if r["path"].startswith(b["run"] + "/") and r["path"].endswith(".eval")]
        gen = d["launch_subset"].encode()
        g = cache / "_gen" / "launch.json"
        g.parent.mkdir(parents=True, exist_ok=True)
        g.write_bytes(gen)
        cand.append({"src": str(g.relative_to(cache)), "dest": f"snaps/{b['id']}/{b['run']}/launch.json",
                     "sha256": hashlib.sha256(gen).hexdigest(), "bytes": len(gen), "generated": True})
    items = [x for x in cand if relay_allowed(x["src"], named, b["run"], x.get("generated", False))]
    refused = sorted({x["src"] for x in cand if x not in items})
    return items, refused


def step_relay(b: dict, st: State):
    """box -> laptop cache -> Volume, every destination sha256-checked on the Volume before any laptop copy is deleted. A snapshot
    also gets its own labeler folder snaps/<batch>/<run>/: its members' .evals and launch records (box_files' launch_subset), so
    its labels read only its own runs."""
    import modal

    d = json.loads((st.d / f"boxfiles_{b['id']}.json").read_text())
    cache = st.d / "cache" / b["id"]  # per batch: no two relays share a cache folder
    cache.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(cache).free < 50 * 1024**3:
        raise RuntimeError("laptop free disk < 50 GB; waiting")
    items, refused = relay_plan(b, d, cache)
    if refused:  # d9 10-15 16:40: an ALLOWLIST; everything else stays on the box (approach/attack files move only via relay_approaches.sh)
        st.log(f"{b['id']}: relay refuses {len(refused)} files not on the allowlist (names only): {', '.join(refused)}")
    vol = modal.Volume.from_name(VOL, create_if_missing=True, version=2)
    hasher = modal.Function.from_name("probe-shift_aswin-surya_sub-q3chain_volhash_kill-2026-11-14", "hash_on_volume")
    with unlocked():
        on_vol = hasher.remote([x["dest"] for x in items])  # resumable: what is already there and right is skipped
    todo = [x for x in items if on_vol.get(x["dest"]) != x["sha256"]]
    own = (f"evals_v1/{b['run']}/", f"evals_v1/{b['run']}.", f"snaps/{b['id']}/")
    foreign = [x["dest"] for x in todo if on_vol.get(x["dest"]) is not None and not x["dest"].startswith(own)]
    if foreign:  # never overwrite another launch's file (a donor or reference copy) with other bytes; only this launch's own partial uploads (Codex 10-15 19:28)
        raise Gate(f"{b['id']}: {len(foreign)} relay destinations outside this launch already hold other bytes on the Volume (e.g. {foreign[0]})")
    if b["kind"] == "ep2" and todo:  # epoch-2 files are relayed only by relay_ep2.py (allowlist); this step only verifies them
        raise Gate(f"{b['id']}: {len(todo)} epoch-2 files are not on the Volume with their sha256 (e.g. {todo[0]['dest']}); relay them with relay_ep2.py")
    st.log(f"{b['id']}: relay {len(todo)} of {len(items)} destinations (the rest already verified on the Volume)")
    fetch = sorted({x["src"] for x in todo if not x.get("generated")})
    if fetch:
        lst = st.d / f"relay_{b['id']}.lst"
        lst.write_text("".join(s + "\n" for s in fetch))
        with unlocked(), RELAY_SEM:  # at most 2 relays move bytes at once
            r = run(["bash", str(DEVBOX_SH), "rsync", "-a", f"--files-from={lst}", f"box:{root_of(b)}/", str(cache) + "/"], timeout=3600,
                    env={"DEVBOX": b["box"]})
        if r.returncode != 0:
            raise RuntimeError(f"rsync rc {r.returncode}: {r.stderr.decode()[-300:]}")
    for x in todo:  # every check before anything uploads (hashes only; the laptop never opens a file's contents)
        f = cache / x["src"]
        h = hashlib.sha256()
        with f.open("rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        if (f.stat().st_size != x["bytes"] or h.hexdigest() != x["sha256"]) and x["src"].endswith(LIVE_LOGS) and f.stat().st_size >= x["bytes"]:
            # a sibling LOG the box still appends to (e.g. the new mains' resource monitor outlives the launch): relay the copy taken now,
            # its own hash checked on the Volume (a log, not data; logged)
            st.log(f"{b['id']}: {x['src']} grew on the box since its hash ({x['bytes']} -> {f.stat().st_size} bytes); relaying the copy taken now")
            x["bytes"], x["sha256"], x["log_snapshot"] = f.stat().st_size, h.hexdigest(), True
            continue
        if f.stat().st_size != x["bytes"] or h.hexdigest() != x["sha256"]:
            f.unlink()
            b["relay_mismatch"] = b.get("relay_mismatch", 0) + 1
            if b["relay_mismatch"] >= 3:  # a finished run's files do not change: a persistent mismatch is a real error
                raise Gate(f"{x['src']}: laptop copy differs from the box manifest ({b['relay_mismatch']} times)")
            raise RuntimeError(f"{x['src']}: laptop copy differs from the box manifest; re-copying next pass")
    if todo:
        with unlocked(), RELAY_SEM:
            with vol.batch_upload(force=True) as batch:  # force: a partial earlier upload of a wrong file is replaced, then checked
                for x in todo:
                    batch.put_file(str(cache / x["src"]), "/" + x["dest"])
            on_vol = hasher.remote([x["dest"] for x in todo])
        bad = [x["dest"] for x in todo if on_vol.get(x["dest"]) != x["sha256"]]
        with (st.d / "relay_log.tsv").open("a") as log:
            for x in todo:
                log.write("\t".join([b["id"], x["dest"], str(x["bytes"]), x["sha256"], str(on_vol.get(x["dest"])), now(),
                                     ("verified" if x["dest"] not in bad else "MISMATCH") + ("\tlog_snapshot: true" if x.get("log_snapshot") else "")]) + "\n")
        if bad:
            raise RuntimeError(f"{len(bad)} destinations differ on the Volume; nothing deleted; retrying")
    with (st.d / "relay_log.tsv").open("a") as log:  # every destination verified: delete every laptop copy (and any left by a crash)
        for s in sorted({x["src"] for x in items}):
            f = cache / s
            if f.exists():
                f.unlink()
                log.write("\t".join([b["id"], s, "", "", "", now(), "deleted"]) + "\n")
    b["volume_files"] = len(items)


def il_worlds(b: dict) -> tuple:
    """An interleaved family's worlds: run3 and sim2, or sim2 only for a sim-only launch (the sim2 gap-fill, real_arms []; 09 @ b9081cd57)."""
    w = tuple(b.get("worlds", ("run3", "sim2")))
    if not w or not set(w) <= {"run3", "sim2"}:
        raise RuntimeError(f"{b['id']}: worlds {w!r} must be a non-empty subset of run3, sim2")
    return w


def targets_for(b: dict) -> dict:
    q, pool = b["q"], b["pool"]
    if b["kind"] == "ep2":  # one staged set per launch, both honest arms: folders <name>_v1__honest_a / __honest_b (09's ep2 ruling)
        return {b["name"]: {"launch": [b["run"]], "manifest": f"manifest_{b['name']}.json", "cache": "/ev/evals_v1", "set": "main",
                            "out": f"{b['name']}_v1", "expect_sha256": None, "stage_arms": ["honest_a", "honest_b"]}}
    sfx = f"_s{b['snap']}" if b["kind"] in LIVE_SNAPS else ""  # a snapshot of the running launch, or (no suffix) its final remainder
    if b["kind"] in IL_KINDS:  # interleaved: two targets, run3 (attack) and sim2 ({SIM_ARM}); names <prefix>_<q>_<world> (09's grammar)
        pfx = b["name_prefix"]
        rev = b.get("rev", "")  # a re-stage from committed corrected manifests (il-s4d's _v2: donors its box read could not see; 09 via d9 10-15)
        tag = b.get("target_tag", "")  # a staging retry family's own target / manifest names (il-s4rgap-t2: 09 via d9 10-15 ~19:35); same staged-set stem
        return {f"{pfx}_{q}_{w}{sfx}{tag}": {"launch": [*b.get("donor_launches", [b["main_run"]]), b["run"]], "manifest": f"manifest_{pfx}_{q}_{w}{sfx}{tag}{rev}.json", "manifest_set": w, "cache": "/ev/evals_v1",
                                        "set": "main", "out": f"{pfx}_{q}_{w}{sfx}_v1", "expect_sha256": None, "stage_arms": [arm]}
                for w, arm in (("run3", "attack"), ("sim2", SIM_ARM)) if w in il_worlds(b)}
    if b["kind"] in ("fu", "fus"):
        q = qtok(b)  # a second follow-up family of the quarter (the Q2 rerun) gets its own names
    if b["kind"] in ("main", "ms"):
        return {f"{pool}_{q}_main{sfx}": {"launch": [b["run"]], "manifest": f"manifest_{pool}_{q}_main{sfx}.json", "cache": "/ev/evals_v1", "set": "main",
                                          "out": f"{pool}_{q}_main{sfx}_v1", "expect_sha256": None, "stage_arms": ["honest_a", "honest_b", "attack"], **({"donor_launches": b["stage_donor_launches"]} if b.get("stage_donor_launches") else {})},
                f"{pool}_{q}_run2{sfx}": {"launch": [b["run"]], "run2": b["run"], "manifest": f"manifest_{pool}_{q}_run2{sfx}.json", "cache": "/ev/evals_v1",
                                          "set": "run2", "out": f"{pool}_{q}_run2{sfx}_v1", "expect_sha256": None, "stage_arms": ["attack"], **({"donor_launches": b["stage_donor_launches"]} if b.get("stage_donor_launches") else {})}}
    tag = b.get("sim_tag", "sim")  # a follow-up family's own name token (config `sim_tag`; default sim; "" = none: simh_t196fill, 09's root)
    stem = "_".join(x for x in (pool, q, tag) if x) + sfx  # an empty qtok (the ug family: ug_sim[_s<j>]) drops its token
    return {stem: {"launch": [*b.get("donor_launches", [b["main_run"]]), b["run"]], "manifest": f"manifest_{stem}.json",
                   "cache": "/ev/evals_v1", "set": "main", "out": f"{stem}_v1", "expect_sha256": None, "stage_arms": [sim_arm_of(b)], **({"donor_launches": b["stage_donor_launches"]} if b.get("stage_donor_launches") else {})}}


def donor_supply(b: dict, name: str, man: dict, st: State):
    """Option (b) (09 via d9 10-15 ~07:20): an interleaved family that ran off its main's box, where the main's copy holds only
    launch.json and references/, gets EXACTLY the main's counted honest_a/honest_b rows for its own cells, by key, from pinned main
    manifests (commit + file sha256), each row naming its source. The stager sha-checks every .eval on the Volume against the row.
    A box read that already has SOME donor row is a mixed state: the family stops for a ruling."""
    if any(f["arm_dir"] in ("honest_a", "honest_b") for f in man["files"]):
        raise Gate(f"{name}: the box read has donor rows and donors_from_main is set (a mixed state stops for a ruling)")
    cells = {f["task"] for f in man["files"] if f["world"] in ("run3", "sim2")}
    sup = {}
    for src in b["donor_manifests"]:
        raw = sp.run(["git", "show", f"{src['commit']}:{REL}/{src['path']}"], capture_output=True, check=True, cwd=REPO).stdout
        if hashlib.sha256(raw).hexdigest() != src["sha256"]:
            raise Gate(f"{name}: pinned donor manifest {src['path']} @ {src['commit']} differs from its sha256")
        for f in json.loads(raw)["files"]:
            if f["arm_dir"] in ("honest_a", "honest_b") and f["task"] in cells:
                k = (f["arm_dir"], f["task"])
                if k in sup:
                    raise Gate(f"{name}: two counted donor rows for {k}")
                sup[k] = {x: v for x, v in f.items() if x != "snapshot_id"} | {"world": f["arm_dir"],
                         "donor_supplied_from": f"{src['path']} @ {src['commit']} (sha256 {src['sha256']})"}
    man["files"] = sorted(man["files"] + list(sup.values()), key=lambda f: f["path"])
    man["counts"] = {"files": len(man["files"]), "by_arm": {a: sum(f["arm_dir"] == a for f in man["files"]) for a in sorted({f["arm_dir"] for f in man["files"]})}}
    man["donors_supplied"] = {"rule": "09 @ 65ef3a9ff section 2 (option b): the main's counted honest rows for this family's cells, by key", "sources": b["donor_manifests"],
                              "rows": len(sup), "cells": len(cells), "cells_without_a_donor_row": sorted(c for c in cells if ("honest_a", c) not in sup and ("honest_b", c) not in sup)}
    st.log(f"{b['id']}: {name}: {len(sup)} donor rows supplied from the main's pinned manifests for {len(cells)} cells")


def sample_error_admit(man: dict, errs: dict, structural: set) -> list:
    """09's admission rule (via d9 10-16 ~00:05 / ~00:45): every manifest row, of every arm (attack, simulated and honest, donors included), carries
    sample_error (null | class, from the .eval's own sample metadata, sample_errors_chain.py). A row with a non-null class that is not one of 09's
    structural classes is refused by the stager (and an errored donor is never a render donor). Returns those rows (key, class, path)."""
    missing = [f["path"] for f in man["files"] if f["path"] not in errs]
    if missing:
        raise Gate(f"sample_error unknown for {len(missing)} manifest rows (e.g. {missing[:2]})")
    for f in man["files"]:
        f["sample_error"] = errs[f["path"]]
    # the rows stay in the manifest: the stager refuses an errored row ("sample error: <class>") and never takes an errored run as a donor
    # ("no usable donor: no clean donor"; 09 10-16 ~00:45), so every refusal is a staging exclusion by key, counted by c_gate
    return [{"key": f"{f['arm_dir']}|{f['task']}", "class": f["sample_error"], "path": f["path"]} for f in man["files"]
            if f["sample_error"] is not None and f["sample_error"] not in structural]


def scan_sample_errors(b: dict, name: str, man: dict, st: State) -> dict:
    """sample_error of every manifest row's .eval (paths + sha256s only; the scan reads the Volume copies, sha-asserted)."""
    rows = sorted({(f["path"], f["sha256"]) for f in man["files"]})
    (st.d / "hygiene").mkdir(exist_ok=True)
    spec = st.d / "hygiene" / f"sample_error_spec_{name}.json"
    spec.write_text(json.dumps([{"path": p, "sha256": h} for p, h in rows]))
    out = st.d / "hygiene" / f"sample_errors_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    reserve(st, b, f"sample_errors {name}", CPU_WORST_USD["labels"])
    modal_run([f"{REL}/sample_errors_chain.py", "--spec", str(spec), "--out", str(out)], st, f"sample_errors_{name}", b=b)
    res = json.loads(out.read_text())["rows"]
    bad = sorted({(p, v.get("sha256")) for p, v in res.items()} ^ set(rows))  # exact equality: no missing, extra or other-sha path
    if bad:
        raise Gate(f"{name}: the sample_error scan does not match the manifest's (path, sha256) for {len(bad)} rows")
    return {p: v["sample_error"] for p, v in res.items()}


def add_index_donors(b: dict, name: str, man: dict, st: State, pre: dict | None = None):
    """S_paper493 real arms (09 via d9 10-15 ~23:15): an attack cell with no honest donor row of its own launch gets ONE honest_a donor row, keyed by
    the cell, from the pinned reference indexes (config index_donors: refindex L1/L2/L3 by commit + sha256): the copy of its origin main's reference.
    A copy under refs_old/ or runs/ is read where the relays put it (evals_v1/<path>); a copy inside a real-arm launch's references/ resolves to that
    launch's relayed honest_a row of the origin key. By key, the donor's sha256 must equal the index pin; a path seen in any committed manifest with
    another sha256 stops. Rows carry the index commit and sha256. A cell no index lists is left for the stager (its exclusion names it).
    With `pre` (path -> sample_error of the launch rows; 09 via d9 10-16 ~01:05): a cell whose launch donor rows ALL carry a sample error gets the
    index row too, as a FALLBACK (index_fallback: true; the copy itself, never the launch's errored row): the stager keys it apart from the
    counted rows and takes it only when no launch donor is usable. (A sample-clean launch donor whose prompt is empty is not seen here: the
    chain reads no prompt. Such a cell keeps the stager's "every listed donor has an empty system prompt" exclusion, which the re-stage
    dry runs list by key against the indexes for 09; every case tonight is errored.)"""
    idx = {}
    for src in b["index_donors"]:
        raw = sp.run(["git", "show", f"{src['commit']}:{src['path']}"], capture_output=True, check=True, cwd=REPO).stdout
        if hashlib.sha256(raw).hexdigest() != src["sha256"]:
            raise Gate(f"{name}: pinned reference index {src['path']} @ {src['commit']} differs from its sha256")
        for cell, e in json.loads(raw)["references"].items():
            if cell in idx:
                raise Gate(f"{name}: {cell} is in two pinned reference indexes")
            idx[cell] = e | {"index": f"{src['path']} @ {src['commit'][:12]} (sha256 {src['sha256']})"}
    seen = {}
    for p in sorted((HERE / "config").glob("manifest_*.json")):
        for f in json.loads(p.read_text())["files"]:
            seen.setdefault(f["path"], set()).add(f["sha256"])
    have = {f["task"] for f in man["files"] if f["arm_dir"] in ("honest_a", "honest_b")}
    clean = {f["task"] for f in man["files"] if f["arm_dir"] in ("honest_a", "honest_b") and (pre is None or pre[f["path"]] is None)}
    need = sorted({cell_of(f["task"]) for f in man["files"] if f["arm_dir"] == "attack"} - clean)
    rows, unlisted = [], []
    for c in need:
        e = idx.get(c)
        if e is None:
            unlisted.append(c)
            continue
        rel = e["copy"].split("/phase2/", 1)[1]
        fallback = c in have  # every launch donor of the cell is errored: the index copy as a fallback candidate
        if rel.startswith("runs_sp493/") and not fallback:  # another (or this) real-arm launch's references/<origin>.eval: its relayed honest_a row
            launch = rel.split("/")[1]
            cands = [f for f in man["files"] if f["launch"] == launch and f["arm_dir"] == "honest_a" and f["task"] == e["origin_key"]]
            if not cands:
                src_all = [x for x in st.s["batches"].values() if x["kind"] in ("main", "ms") and bare_id(x["run"]) == launch]
                if src_all and not any(x.get("commit_targets") for x in src_all):  # the source launch's manifests are not committed yet (c before b)
                    raise Wait(f"{name}: {c}: the donor launch {launch} has not committed its manifests yet")
                src_b = [x for x in src_all if x.get("commit_targets")]
                cands = [f for x in src_b for tt, tv in targets_for(x).items() if tv.get("set") == "main"
                         for f in json.loads((HERE / "config" / tv["manifest"]).read_text())["files"]
                         if f["launch"] == launch and f["arm_dir"] == "honest_a" and f["task"] == e["origin_key"] and not f.get("reference_index")]
            if len(cands) != 1 or cands[0]["sha256"] != e["sha256"]:
                raise Gate(f"{name}: {c}: reference {e['origin_key']} of {launch} is {len(cands)} honest_a rows, sha256 "
                           f"{[x['sha256'][:12] for x in cands]} vs the index pin {e['sha256'][:12]}")
            row = {k: v for k, v in cands[0].items() if k not in ("snapshot_id", "stage")}
        else:
            launch, tail = rel.split("/references/", 1)
            row = {"launch": launch, "rel": f"references/{tail}", "path": rel, "sha256": e["sha256"], "bytes": None, "attempt_index": None,
                   "policy_called": True, "state": "done", "launch_id": launch.rsplit("/", 1)[-1]}
        if seen.get(row["path"], {e["sha256"]}) != {e["sha256"]}:
            raise Gate(f"{name}: {c}: donor path {row['path']} has another sha256 in a committed manifest")
        rows.append(row | {"arm_dir": "honest_a", "world": "honest_a", "task": c, "launch_task": c, "attempt_kind": "reference", "stratum": b.get("stratum"),
                           "reference_origin_key": e["origin_key"], "reference_sha256_pin": e["sha256"], "reference_index": e["index"],
                           **({"index_fallback": True} if fallback else {})})
    man["files"] = sorted(man["files"] + rows, key=lambda f: f["path"] + "|" + f["task"])
    man["counts"] = {"files": len(man["files"]), "by_arm": {a: sum(f["arm_dir"] == a for f in man["files"]) for a in sorted({f["arm_dir"] for f in man["files"]})}}
    man["index_donors"] = {"rule": "09 via d9 10-15 ~23:15: one honest_a donor row per attack cell lacking one, from the pinned reference indexes (never staged)",
                           "indexes": b["index_donors"], "rows": len(rows), "cells_needing_donors": len(need), "cells_not_in_any_index": unlisted}
    fb = sorted(r["task"] for r in rows if r.get("index_fallback"))
    if fb:  # only when some cell needed one: every other manifest stays byte-identical
        man["index_donors"]["fallback"] = {"rule": "09 via d9 10-16 ~01:05: a cell whose launch donors all carry a sample error gets the index copy as a "
                                                    "fallback candidate (index_fallback), taken by the stager only if no launch donor is usable", "cells": fb}
    st.log(f"{b['id']}: {name}: {len(rows)} index donor rows for {len(need)} attack cells without a usable launch donor (fallback: {fb or 'none'}); not in any index: {unlisted or 'none'}")


def step_commit_targets(b: dict, st: State):
    d = json.loads((st.d / f"boxfiles_{b['id']}.json").read_text())
    tg = targets_for(b)
    paths, keep, mans = [], [], {}
    for name, t in tg.items():
        set_key = t.get("manifest_set") or ("run2" if t["set"] == "run2" else "main")
        man = d["manifests"][set_key]
        if b["kind"] in IL_KINDS:  # 09's IL fields: the pilot slice by launcher task from the IL branch's committed stage task files
            missing = sorted({f["task"] for f in man["files"] if f["world"] in ("run3", "sim2") and f["task"] not in b.get("stage_of", {})})
            if missing:  # never write an unknown stage: the registered pilot slice depends on it
                raise Gate(f"{name}: {len(missing)} interleaved tasks have no stage in config/il_stage_maps.json (e.g. {missing[:3]})")
            for f in man["files"]:
                if f["world"] in ("run3", "sim2"):
                    f["stage"] = b["stage_of"][f["task"]]
            if b.get("donors_from_main"):  # 09 via d9 10-15 ~07:20 (option b): an off-box family whose box copy of the main holds only references
                donor_supply(b, name, man, st)
            if b.get("donor_cells_only"):  # sel97 (2e/d9 10-15 ~16:30): honest donor rows only for this launch's own run3/sim2 cells
                own = {f["task"] for f in man["files"] if f["world"] in ("run3", "sim2")}
                drop = sorted({f["task"] for f in man["files"] if f["arm_dir"] in ("honest_a", "honest_b") and f["task"] not in own})
                man["files"] = [f for f in man["files"] if f["arm_dir"] not in ("honest_a", "honest_b") or f["task"] in own]
                man["counts"] = {"files": len(man["files"]), "by_arm": {a: sum(f["arm_dir"] == a for f in man["files"]) for a in sorted({f["arm_dir"] for f in man["files"]})}}
                man["donor_cells_only"] = {"rule": "honest donor rows only for the launch's own cells (sel97; t151/t96 status rows only)", "dropped_donor_cells": drop}
                st.log(f"{b['id']}: {name}: donor rows kept for the launch's {len(own)} cells only; dropped donors of {len(drop)} other cells")
        if b.get("exclude_keys") and b["kind"] in ("main", "ms"):  # "<arm folder>|<task>" keys that must enter no staged set (registered exclusions);
            # "<target>:<arm folder>|<task>" for one target only (09 10-16 ~00:00: run2|t615_media_processing_r2, an unlabelable call; its run1 stays)
            # (before the index donors: an excluded honest row is no donor; 10-16 ~01:05)
            ex = {k for k in b["exclude_keys"] if ":" not in k} | {k.split(":", 1)[1] for k in b["exclude_keys"] if k.startswith(f"{name}:")}
            hit = sorted(f"{f['arm_dir']}|{f['task']}" for f in man["files"] if f"{f['arm_dir']}|{f['task']}" in ex)
            man["files"] = [f for f in man["files"] if f"{f['arm_dir']}|{f['task']}" not in ex]
            man["counts"] = {"files": len(man["files"]), "by_arm": {a: sum(f["arm_dir"] == a for f in man["files"]) for a in sorted({f["arm_dir"] for f in man["files"]})}}
            man["excluded_keys"] = {"keys": hit, "rule": "registered exclusions (config exclude_keys): status rows, never staged"}
            if hit:
                st.log(f"{b['id']}: {name}: excluded keys never staged: {', '.join(hit)}")
        pre = None  # the launch rows' sample_error, read before the index donors (their fallback needs it)
        if b.get("index_donors") and b["kind"] in ("main", "ms"):  # S_paper493 real arms: the pinned reference of each attack cell's origin main
            if b.get("sample_error_rule") and not b.get("rev"):
                pre = scan_sample_errors(b, name, man, st)
            add_index_donors(b, name, man, st, pre)
            if "excluded_keys" in man:  # the key order of before (index_donors, then excluded_keys): a set without a fallback keeps its bytes
                man["excluded_keys"] = man.pop("excluded_keys")
        mans[name] = man
        if b.get("sample_error_rule"):  # 09 (via d9 10-16 ~00:05): admission needs sample_error == null on every row of every arm
            if b.get("rev"):  # a committed corrected manifest is staged as is: the rule would scan rows it does not stage
                raise Gate(f"{name}: sample_error_rule with a rev (corrected) manifest is not supported")
            if pre is None:
                errs = scan_sample_errors(b, name, man, st)
            else:  # the index rows only (the launch rows were read above): the same values as one scan of every row
                new = [f for f in man["files"] if f["path"] not in pre]
                errs = pre | (scan_sample_errors(b, f"{name}_index", {"files": new}, st) if new else {})
            refused = sample_error_admit(man, errs, set(b.get("sample_error_structural", [])))
            man["sample_error_rule"] = {"rule": "09 via d9 10-16 ~00:05: every row carries sample_error (sample metadata, the B fork's class rule); "
                                                "a non-null class refuses the row by key (never staged)", "refused": refused}
            st.log(f"{b['id']}: {name}: sample_error on {len(man['files'])} rows; refused by key: {[x['key'] + ' (' + x['class'] + ')' for x in refused] or 'none'}")
        p = HERE / "config" / t["manifest"]
        if b.get("rev"):  # the committed corrected manifest is used as is: it must hold every row of this box read unchanged, and only honest
            # donor rows of the main launch besides, each byte-identical (path, sha256) to the main family's committed manifests
            fixed = json.loads(p.read_text())
            have = {f["path"]: f for f in fixed["files"]}
            bad = [f["path"] for f in man["files"] if f["path"] not in have or any(have[f["path"]].get(k) != v for k, v in f.items())]
            mb = main_of(st, b)
            main_rows = {f["path"]: f for x in [mb, *snaps_of(st, mb)] for tt, tv in targets_for(x).items() if tt in x.get("targets", [])
                         for f in json.loads((HERE / "config" / tv["manifest"]).read_text())["files"] if f["arm_dir"] in ("honest_a", "honest_b")}
            box_paths = {f["path"] for f in man["files"]}
            bad += [k for k, f in have.items() if k not in box_paths and (f["arm_dir"] not in ("honest_a", "honest_b") or main_rows.get(k, {}).get("sha256") != f["sha256"])]
            if bad or fixed.get("supersedes") != t["manifest"].replace(b["rev"], ""):
                raise Gate(f"{name}: corrected manifest {t['manifest']} does not extend the box read with the main's donors only ({len(bad)} bad rows, e.g. {bad[:2]})")
            st.log(f"{b['id']}: {t['manifest']} verified: {len(box_paths)} box-read rows unchanged, {len(have) - len(box_paths)} main donors")
        else:
            p.write_text(json.dumps(man, indent=1) + "\n")
            paths.append(f"{REL}/config/{t['manifest']}")
        if any(f["arm_dir"] in t["stage_arms"] for f in man["files"]):
            keep.append(name)
        else:  # e.g. a remainder whose every cell an earlier snapshot took: nothing to stage, extract or route
            st.log(f"{b['id']}: target {name} has no runs to stage; skipped")
    if b.get("index_donors") or b.get("stage_donors_from_rows"):  # honest donor rows of other launches: read by the stager, never staged
        b["stage_donor_launches"] = sorted({f["launch"] for name, t in tg.items() for f in mans[name]["files"]
                                            if f["arm_dir"] in ("honest_a", "honest_b") and f["launch"] not in t["launch"]})
        tg = targets_for(b)
        st.log(f"{b['id']}: donor-only launches (never staged): {', '.join(b['stage_donor_launches']) or 'none'}")
    ct = HERE / "config" / "chain_targets.json"

    def prepare():  # read-modify-write of the shared targets file, atomically, inside the commit's GIT section
        allt = json.loads(ct.read_text()) if ct.exists() else {}
        allt |= tg
        tmp = ct.with_suffix(".tmp")
        tmp.write_text(json.dumps(allt, indent=1) + "\n")
        tmp.replace(ct)

    paths.append(f"{REL}/config/chain_targets.json")
    held = sorted(n for n in keep if n in b.get("hold_targets", []))  # a target 09 holds (sp493 b_run2, t615 labeler gap): manifest written, never staged
    if held:
        st.log(f"{b['id']}: held targets (not staged until a ruling): {', '.join(held)}")
    b["targets"] = sorted(n for n in keep if n not in held)
    b["commit_targets"] = commit(paths, f"Overnight chain: {b['id']} counted-attempt manifests and staging targets ({b['run']})", st, prepare=prepare)


def stage_reattach(st: State, b: dict, t: str, suffix: str):
    """An earlier staging try of `t` whose client did not record a result. Its fan-out runs in a Modal driver (stage_chain.py
    stage_driver), which a laptop sleep or a restart does not cancel: "running" while it runs (wait; never start a new try over
    it), the path of a record published from its result once it finished (from the Volume or the finished call), or None when it
    is over without a result or never started (a new try then stages under a fresh write-once name, so nothing is overwritten).
    An unknown status raises (the try is kept and retried)."""
    import modal

    name = targets_for(b)[t]["out"] + suffix
    vol = modal.Volume.from_name(VOL, version=2)

    def vol_json(path: str):
        try:
            with unlocked():
                return json.loads(b"".join(vol.read_file(path)))
        except (FileNotFoundError, modal.exception.NotFoundError):
            return None

    def publish(rec: dict) -> Path:
        set_name = next(k for k in rec if k not in ("git", "stamp"))
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        dest = HERE / "results" / f"chain_stage_{t}_{stamp}.json"
        while dest.exists():
            time.sleep(1)
            dest = HERE / "results" / f"chain_stage_{t}_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}.json"
        tmp = dest.with_suffix(".part")
        tmp.write_text(json.dumps({"git": rec["git"], set_name: rec[set_name]}, indent=1, default=str))
        tmp.rename(dest)
        st.log(f"{b['id']}: stage {t}{suffix}: record re-attached from Modal ({dest.name})")
        return dest

    for loc in sorted((HERE / "results").glob(f"chain_stage_{t}_20*.json"), reverse=True):  # this try's record already here
        try:
            r = json.loads(loc.read_text())
            if r[next(k for k in r if k not in ("git", "stamp"))]["out"] == name:
                return loc
        except Exception:
            continue  # a cut-short file is not a record
    rec = vol_json(f"stage_results/{name}.json")
    if rec is not None:
        return publish(rec)
    failed = vol_json(f"stage_results/{name}.failed.json")
    if failed is not None:  # the driver ended with an exception: a failed gate stops the batch; anything else is a new try
        st.log(f"{b['id']}: stage {t}{suffix}: driver failed on Modal ({failed.get('failed')})")
        if any(m in failed.get("message", "") for m in GATE_MARKS):
            raise Gate(f"{t}: staging driver gate failed: {failed.get('message', '')[:400]}")
        return None
    marker = vol_json(f"stage_results/{name}.started.json")
    cid = marker.get("stage_driver_call_id") if marker else None
    if cid is None:
        for log in sorted((st.d / "modal_logs").glob(f"stage_{t}_*.log"), reverse=True):
            for line in log.read_text().splitlines():
                if line.startswith('{"stage_driver_call_id"') and json.loads(line).get("name") == name:
                    cid = json.loads(line)["stage_driver_call_id"]
                    break
            if cid:
                break
    if cid is None:
        return None  # no driver was started for this try
    try:
        with unlocked():
            out = modal.FunctionCall.from_id(cid).get(timeout=0)
    except (modal.exception.OutputExpiredError, modal.exception.FunctionTimeoutError, modal.exception.InputCancellation, modal.exception.RemoteError):
        b.get("stage_unknown", {}).pop(t, None)
        return None  # over without a result
    except (modal.exception.TimeoutError, TimeoutError):
        b.get("stage_unknown", {}).pop(t, None)  # a confirmed status resets the count (consecutive unknowns only)
        return "running"
    except Exception as e:  # its failure marker is written before it re-raises; without one, the status is unknown: keep, retry, bound
        if vol_json(f"stage_results/{name}.failed.json") is not None:
            return stage_reattach(st, b, t, suffix)
        b.setdefault("stage_unknown", {})[t] = b.get("stage_unknown", {}).get(t, 0) + 1
        if b["stage_unknown"][t] >= 10:
            raise Gate(f"stage {t}{suffix}: driver {cid} status unknown 10 times ({type(e).__name__})")
        raise RuntimeError(f"stage {t}{suffix}: driver status unknown ({type(e).__name__}); retrying") from None
    b.get("stage_unknown", {}).pop(t, None)
    return publish(out)


def step_stage(b: dict, st: State):
    b.setdefault("stage", {})
    for t in b["targets"]:
        if t in b["stage"]:
            continue
        prev = b.setdefault("stage_tries", {}).get(t, 0)
        rec, suffix = None, None
        if prev and b.get("stage_abandoned", {}).get(t) != prev:  # an earlier try left no record here: re-attach to its Modal driver before any new try
            psuffix = "" if prev == 1 else f"_try{prev}"
            r_ = stage_reattach(st, b, t, psuffix)
            if r_ == "running":
                raise Wait(f"stage {t} try {prev} still running on Modal (re-attached)")
            if r_ is not None:
                rec, suffix = r_, psuffix
        if rec is None:
            n = prev + 1
            b["stage_tries"][t] = n
            b.get("stage_unknown", {}).pop(t, None)  # a fresh try starts a fresh count
            st.save()
            suffix = "" if n == 1 else f"_try{n}"  # a rerun after a lost client stages under a fresh, write-once name
            before = set((HERE / "results").glob(f"chain_stage_{t}_20*.json"))
            reserve(st, b, f"stage {t} try {n}", CPU_WORST_USD["stage"])
            t_call = time.time()
            try:
                modal_run([f"{REL}/stage_chain.py", "--target", t, *(["--suffix", suffix] if suffix else [])], st, f"stage_{t}", timeout=3 * 3600, b=b, detach=True)
            except RuntimeError as e:  # a refusal before any spawn (the stager's clean-tree check lost a race with another batch's commit) writes
                # nothing, so the name is not consumed: only when THIS call's process exited and THIS call's own log holds the refusal (a timeout
                # or a transport failure never rolls back; Codex 10-15 23:36)
                logs = [q for q in (st.d / "modal_logs").glob(f"stage_{t}_*.log") if q.stat().st_mtime >= t_call]
                if str(e).startswith(f"stage_{t}: modal run exited") and len(logs) == 1 and "refusing to run from a dirty tree" in logs[0].read_text():
                    b["stage_tries"][t] = prev
                    st.save()
                    st.log(f"{b['id']}: stage {t} try {n} refused before any spawn (dirty tree); the try counter stays at {prev}")
                raise
            rec = sorted(set((HERE / "results").glob(f"chain_stage_{t}_20*.json")) - before)[-1]
        r = json.loads(rec.read_text())
        set_name = next(k for k in r if k not in ("git", "stamp"))
        a = r[set_name]["assembled"]
        if a.get("STOP") or "c_gate" not in r[set_name]:
            raise Gate(f"{t}: staging stopped: {a.get('STOP')}")
        b["stage"][t] = {"record": rec.name, "out": r[set_name]["out"], "suffix": suffix, "epochs": a["epochs"], "staged_inputs_sha256": a["staged_inputs_sha256"],
                         "excluded": len(r[set_name]["c_gate"]["excluded"]), "fmt_gate_sha256": a.get("fmt_gate_sha256"),
                         "think_open_truncated_at_cut_runs": {arm: g.get("think_open_truncated_at_cut_runs") for arm, g in a["format_gate"].items()}}
        st.log(f"{b['id']}: staged {t}: {a['epochs']}")


def step_format(b: dict, st: State):
    b.setdefault("format", {})
    for t in b["targets"]:
        for arm in b["stage"][t]["epochs"]:
            k = f"{t}:{arm}"
            if k in b["format"]:
                continue
            n = min(FMT_N, b["stage"][t]["epochs"][arm])
            before = set((HERE / "results").glob(f"format_check_{t}_{arm}_20*.json"))
            reserve(st, b, f"format {k}", CPU_WORST_USD["format"])
            modal_run([f"{REL}/stage_chain.py", "--target", t, "--format-n", str(n), "--arm", arm, *(["--suffix", b["stage"][t]["suffix"]] if b["stage"][t]["suffix"] else [])], st,
                      f"format_{t}_{arm}", b=b)
            rec = sorted(set((HERE / "results").glob(f"format_check_{t}_{arm}_20*.json")) - before)[-1]
            r = json.loads(rec.read_text())
            if not r["passes"]:
                raise Gate(f"{k}: format check FAILED ({rec.name})")
            b["format"][k] = rec.name


def step_repro(b: dict, st: State):
    b.setdefault("shards", {})
    b.setdefault("repro", {})
    for t in b["targets"]:
        if t in b["repro"]:
            continue
        ep = sum(b["stage"][t]["epochs"].values())
        shards = max(MIN_SHARDS, min(MAX_SHARDS, math.ceil(ep / EP_PER_SHARD)))
        reserve(st, b, f"repro {t}", CPU_WORST_USD["repro"])
        out = modal_run([f"{REL}/stage_chain.py", "--target", t, "--repro-shards", str(shards), *(["--suffix", b["stage"][t]["suffix"]] if b["stage"][t]["suffix"] else [])], st,
                        f"repro_{t}", b=b)
        r = json.loads(next(line for line in out.splitlines() if line.startswith("{") and "repro_sources" in line))
        b["shards"][t], b["repro"][t] = shards, r["sha256"]


def step_extract(b: dict, st: State):
    b.setdefault("extract", {})
    paths = []
    for t in b["targets"]:
        if t in b["extract"] or t in b.setdefault("cfg_committed", []):
            continue
        run_name = b["stage"][t]["out"].replace("_v1", "_extract_v1")
        cfg = HERE / "config" / f"chain_extract_{t}.yaml"
        sh = dict(b["stage"][t]["staged_inputs_sha256"]) | {"repro-c.jsonl": b["repro"][t]}
        cfg.write_text(f"# Overnight chain: {t} staged by stage_chain.py (record {b['stage'][t]['record']}), plus the per-shard reference arm repro-c;\n"
                       f"# every layer, last-token rows and output_end span means, as q4b_extract_c.yaml (chunk 8, pad 4096), H200.\n"
                       f"run: {run_name}\ninputs: {b['stage'][t]['out']}\ninputs_sha256:\n"
                       + "".join(f"  {k}: {v}\n" for k, v in sorted(sh.items()))
                       + f"model: Qwen/Qwen3.5-27B\nshards: {b['shards'][t]}\nchunk_size: 8\npad_to: 4096\n")
        paths.append(f"{REL}/config/chain_extract_{t}.yaml")
    if paths:
        commit(paths, f"Overnight chain: {b['id']} extraction configs", st)
        b["cfg_committed"] += [t for t in b["targets"] if f"{REL}/config/chain_extract_{t}.yaml" in paths]
        st.save()
    import modal

    vol = modal.Volume.from_name(VOL, version=2)
    pending = []  # a batch's targets extract in parallel; the step completes when all have
    for t in b["targets"]:
        if t in b["extract"]:
            continue
        run_name = b["stage"][t]["out"].replace("_v1", "_extract_v1")
        want = dict(b["stage"][t]["staged_inputs_sha256"]) | {"repro-c.jsonl": b["repro"][t]}
        job = b.setdefault("extract_job", {}).get(t)
        if job:
            if job["pid"] is not None and pid_alive(job["pid"]):
                pending.append(f"{t} a{job['att']}")
                continue
            if not attempt_over(st, b, t, job):  # the client is gone (or its pid was never saved) but the Modal orchestrator still runs
                pending.append(f"{t} a{job['att']} (re-attached; running on Modal)")
                continue
            # pid None and no record: whether it ran is unknown, so the full worst case
            settle_extract(st, b, t, job, force_worst=job["pid"] is None and not new_records(job))  # books its cost, releases its reservation
            b["extract_job"].pop(t)
            st.save()
        try:  # recover a completed extraction whose acknowledgement was lost: its complete.json names these exact inputs
            done = json.loads(b"".join(vol.read_file(f"{run_name}/complete.json")))
        except (FileNotFoundError, modal.exception.NotFoundError):
            done = None  # confirmed absent: go on to launch
        except Exception as e:  # unreadable is not absent: retry next pass, without advancing the attempt or applying any gate
            raise RuntimeError(f"extract {t}: complete.json unreadable ({type(e).__name__}); retrying") from None
        if done is not None:
            if done.get("inputs_sha256") != want:
                raise Gate(f"extract {t}: {run_name}/complete.json names other inputs")
            b["extract"][t] = run_name
            st.log(f"{b['id']}: extract {t} already complete (attempt {done.get('attempt')}); recovered")
            continue
        if git_locked("status", "--porcelain", "--untracked-files=no", "--", "*.py", "src"):  # 2e: every real extraction pins clean CODE (records and configs the chain commits are not code)
            raise Gate(f"extract {t}: the run worktree has modified tracked files; refusing to extract")
        att = b.setdefault("extract_attempt", {}).get(t, 0) + 1  # monotonic and persisted BEFORE the launch (the lease is per attempt)
        if att > 4:
            raise Gate(f"extract {t}: 4 attempts failed")
        worst = b["shards"][t] * SHARD_WORST_H * H200_USD_H
        reserve(st, b, f"extract {t} a{att}", worst)  # raises Wait when the night's reservation is full
        log = st.d / "modal_logs" / f"extract_{t}_a{att}.log"
        log.parent.mkdir(exist_ok=True)
        before = sorted(str(p) for p in (HERE / "results").glob(f"extract_chain_{run_name}_a{att}_*.json"))
        b["extract_attempt"][t] = att  # attempt, job and reservation persisted BEFORE the launch (a crash in between keeps them)
        b.setdefault("extract_job", {})[t] = {"pid": None, "att": att, "worst": worst, "before": before, "run": run_name, "log": str(log), "t": time.time()}
        st.save()
        proc = sp.Popen([MODAL, "run", "--detach", f"{REL}/extract_chain.py", "--config", f"chain_extract_{t}.yaml", "--attempt", str(att)],
                        stdout=log.open("w"), stderr=sp.STDOUT, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO / "src")},
                        start_new_session=True)  # survives a chain restart; the chain polls it instead of blocking
        b["extract_job"][t]["pid"] = proc.pid
        st.save()
        st.log(f"{b['id']}: extract {t} attempt {att} launched (pid {proc.pid}, {b['shards'][t]} shards, worst ${worst:.0f})")
        pending.append(f"{t} a{att}")
    if pending:
        raise Wait(f"extracting {', '.join(pending)}")


def step_compare(b: dict, st: State):
    b.setdefault("compare", {})
    for t in b["targets"]:
        if t in b["compare"]:
            continue
        before = set((HERE / "results").glob(f"compare_repro_{b['extract'][t]}_repro-c_*.json"))
        reserve(st, b, f"compare {t}", CPU_WORST_USD["compare"])
        modal_run([f"{REL}/compare_repro.py", "--run", b["extract"][t], "--arm", "repro-c"], st, f"compare_{t}", b=b)
        b["compare"][t] = sorted(set((HERE / "results").glob(f"compare_repro_{b['extract'][t]}_repro-c_*.json")) - before)[-1].name


def step_labels(b: dict, st: State):
    if b["kind"] == "ep2":
        st.log(f"{b['id']}: labels not applicable (real honest arms)")
        return
    # an interleaved batch: two label files, its attack (run3, `--kind main`) and its simulated attack (sim2, `--kind fu`); one per world it has
    slots = ((("labels", "main", "run3"), ("labels_sim2", "fu", "sim2")) if b["kind"] in IL_KINDS else (("labels", "main" if b["kind"] in ("main", "ms") else "fu", None),))
    for slot, kind in ((s, k) for s, k, w in slots if w is None or w in il_worlds(b)):
        if b.get(slot):
            continue
        if kind == "fu" and b["kind"] in ("fu", "fus") and sim_is_honest(b):  # a simulated honest arm: no attack labels
            st.log(f"{b['id']}: labels not applicable ({sim_arm_of(b)} is an honest arm)")
            continue
        reserve(st, b, f"labels {b['id']} {kind}", CPU_WORST_USD["labels"])
        args = ["--run", b["run"], "--kind", kind]
        if kind == "fu" and b["kind"] in ("fu", "fus") and sim_arm_of(b) != SIM_ARM:
            args += ["--arm", sim_arm_of(b)]
        if b["kind"] in LIVE_SNAPS:
            args += ["--root", f"snaps/{b['id']}"]  # its members' records and .evals only
        out = modal_run([f"{REL}/labels_chain.py", *args], st, f"labels_{b['id']}_{kind}", b=b)
        b[slot] = json.loads(next(line for line in out.splitlines() if line.startswith("{") and '"file"' in line))
        st.save()


def step_record(b: dict, st: State):
    paths = [f"{REL}/results/{b['stage'][t]['record']}" for t in b["targets"]]
    paths += [f"{REL}/results/{v}" for v in b["format"].values()] + [f"{REL}/results/{v}" for v in b["compare"].values()]
    paths += [f"{REL}/{b[k]['file']}" for k in ("labels", "labels_sim2") if b.get(k)]
    paths += [f"{REL}/results/{p.name}" for t in b["targets"] for p in (HERE / "results").glob(f"extract_chain_{b['extract'][t]}_a*.json")
              if valid_record(p) is not None]
    rows = "".join(f"| {now()[:16]} | `{b['extract'][t]}` | Qwen3.5-27B, every layer, H200 | {VOL} | stage_chain.py {t} ({b['stage'][t]['record']}); "
                   f"extract_chain.py; repro-c gate {b['compare'][t]} | overnight chain, {b['run']} |\n" for t in b["targets"])
    act = REPO / "ACTIVATIONS.md"

    def prepare():  # inside the commit's GIT section (no git read sees the file half-appended)
        if not all(f"`{b['extract'][t]}`" in act.read_text() for t in b["targets"]):  # a rerun after a failed push appends nothing twice
            with act.open("a") as f:
                f.write(rows)

    paths.append("ACTIVATIONS.md")
    b["record_commit"] = commit(paths, f"Overnight chain: {b['id']} staged, extracted and labelled ({b['run']}); records", st, prepare=prepare)


def honest_manifests(st: State, b: dict) -> list:
    """09's registry: the main-target manifests that hold b's honest arms (a main batch: its own; a follow-up: every main-family
    batch of its quarter, the remainder and each snapshot), each with its path @ commit and file sha256."""
    mb = main_of(st, b)
    fam = [b] if b["kind"] in ("main", "ms") or mb is None else [*snaps_of(st, mb), mb]
    out = []
    for x in fam:
        if not x.get("commit_targets"):  # only members whose targets were committed (a commit that stopped leaves targets unrecorded)
            continue
        for t, tv in targets_for(x).items():
            if (mb is None and (tv.get("manifest_set") == "run3" or x["kind"] in ("fu", "fus"))) or ("honest_a" in tv["stage_arms"] and t in x.get("targets", [])):  # T196 IL / an index-donor follow-up: donors in its own committed manifest (staged or not)
                p = HERE / "config" / tv["manifest"]
                out.append({"target": t, "batch": x["id"], "manifest": f"{REL}/config/{tv['manifest']} @ {x['commit_targets'][:10]}",
                            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "extraction_root": x.get("extract", {}).get(t)})
    return out


def staging_exclusions(b: dict, t: str) -> dict:
    """The staging record's per-key exclusions with stage_q4b's reasons (c_gate.excluded: key -> reason; keys and reasons only)."""
    r = json.loads((HERE / "results" / b["stage"][t]["record"]).read_text())
    return r[next(k for k in r if k != "git")]["c_gate"]["excluded"]


def step_ready(b: dict, st: State):
    """One routing file per EXTRACTION ROOT (per target: a quarter's main, its run2, each sim snapshot, the sim remainder); the agent
    sends each to 4f, 09, 2e and df (cc d9, 0d)."""
    n_snaps = sum(1 for x in snaps_of(st, b) if not x.get("empty"))
    Q = b["q"].upper()
    tag0 = {"main": f"final remainder of the {Q} main (after {n_snaps} snapshots)" if n_snaps else f"{Q} main (real arms)",
            "ms": f"snapshot {b.get('snap')} of the {Q} main", "fus": f"snapshot {b.get('snap')} of the {b.get('label', Q + ' sim follow-up')}",
            "fu": f"final remainder of the {b.get('label', Q + ' sim follow-up')} (after {n_snaps} snapshots)",
            "ils": f"snapshot {b.get('snap')} of the {b.get('label', 'interleaved ' + Q)}",
            "il": f"final remainder of the {b.get('label', 'interleaved ' + Q)} (after {n_snaps} snapshots)",
            "ep2": f"epoch-2 real honest {b.get('label', Q)}"}[b["kind"]]
    (st.d / "routing").mkdir(exist_ok=True)
    for t in b["targets"]:
        tv = targets_for(b)[t]
        nm = b["stage"][t]["out"] + b["stage"][t].get("suffix", "") + "|" + t
        if any(tok in nm for tok in ("_dry", "_regress")):  # a dry-run or regression staging is never routed (d9 10-15 ~23:15 / ~23:45)
            raise Gate(f"{b['id']}: {t}: refusing to write a routing file for a dry-run or regression name ({nm})")
        man = json.loads((HERE / "config" / tv["manifest"]).read_text())
        if b.get("sample_error_rule"):  # the manifest-level assert: every row carries the field; no ADMITTED (staged) row carries a sample error (09)
            ex_ = {k.split("#", 1)[0] for k in staging_exclusions(b, t)}
            own_ = set(tv["launch"]) if tv.get("donor_launches") else None
            bad = [f"{f['arm_dir']}|{f['task']}" for f in man["files"] if "sample_error" not in f]
            bad += [f"{f['arm_dir']}|{f['task']}" for f in man["files"] if f["arm_dir"] in tv["stage_arms"] and f.get("attempt_kind") != "reference"
                    and (own_ is None or f["launch"] in own_) and f["path"] not in ex_ and f.get("sample_error") is not None]  # no exemption at admission
            if bad:
                raise Gate(f"{b['id']}: {t}: {len(bad)} admitted manifest rows without a null sample_error (e.g. {bad[:3]})")
        arms = []
        for arm, n in b["stage"][t]["epochs"].items():
            if arm == "repro-c":
                continue
            worlds = sorted({f["world"] for f in man["files"] if f["arm_dir"] == arm})
            arms.append({"meta_arm": arm, "staged_inputs_file": f"{b['stage'][t]['out']}/inputs/{arm}.jsonl", "world": worlds, "epochs": n,
                         "staged_sha": b["stage"][t]["staged_inputs_sha256"].get(f"{arm}.jsonl"),
                         "think_open_truncated_at_cut_runs": b["stage"][t]["think_open_truncated_at_cut_runs"].get(arm),
                         "format_check": f"{REL}/results/{b['format'].get(f'{t}:{arm}')} @ {b['record_commit'][:10]}",
                         "sim_launch_ne_run1_launch": sum(bool(f.get("sim_launch_ne_run1_launch")) for f in man["files"] if f["arm_dir"] == arm) if b["kind"] in FU_KINDS else None,
                         **({"stage_counts": {sg: sum(1 for f in man["files"] if f["arm_dir"] == arm and f.get("stage") == sg) for sg in sorted({f.get("stage") for f in man["files"] if f["arm_dir"] == arm})}}
                            if b["kind"] in IL_KINDS else {})})
        msg = {"batch": b["id"], "target": t, "stratum": b.get("stratum", "S_new"), "tag": tag0 + (" — run2 (second attack, same launch)" if t.endswith("_run2") else ""), "pool": b["pool"],
               "members": b.get("members"), "runs": [b["main_run"], b["run"]] if b["kind"] in FU_KINDS else [b["run"]], "volume": VOL,
               "extraction_root": b["extract"][t], "staged_set": b["stage"][t]["out"], "fmt_gate_sha256": b["stage"][t]["fmt_gate_sha256"],
               "manifest": f"{REL}/config/{tv['manifest']} @ {b['commit_targets'][:10]}",
               "repro_gate": f"{REL}/results/{b['compare'][t]} @ {b['record_commit'][:10]}",
               **({"sealed": EP2_SEALED} if b["kind"] == "ep2" else {}),
               "labels": (lambda L: f"{REL}/{L['file']} (sha256 {L['sha256']}) @ {b['record_commit'][:10]}" if L else ("not applicable: real honest arms" if b["kind"] == "ep2" else f"not applicable: {sim_arm_of(b)} is an honest arm"))(
                   b["labels_sim2"] if t in targets_for(b) and targets_for(b)[t].get("manifest_set") == "sim2" else b.get("labels")), "arms": arms,
               "staging_exclusions": staging_exclusions(b, t)}
        (routing_dir(st, b) / f"{b['id']}__{t}.json").write_text(json.dumps(msg, indent=1))
    write_registry(st, b.get("stratum", "S_new"), routing=b)


def write_registry(st: State, stratum: str, routing: dict | None = None) -> Path:
    """09's registry, one file per stratum beside routing/ (spec.registry reads it): every routed target of the stratum, ALL entries
    recomputed from the state on every write (so main_family_complete and the honest manifests never go stale)."""
    entries = {}
    for x in st.s["batches"].values():
        if x.get("stratum", "S_new") != stratum or ("ready" not in x["done"] and x is not routing):  # routing: the batch being routed now
            continue
        mfam = main_of(st, x) or x
        for t in x["targets"]:
            entries[f"{x['id']}__{t}.json"] = {"batch": x["id"], "target": t, "extraction_root": x["extract"][t], "honest_manifests": honest_manifests(st, x),
                                               "main_family_complete": all(y.get("commit_targets") for y in [*snaps_of(st, mfam), mfam] if y["status"] not in ("STOPPED", "EMPTY"))}
    reg_p = st.d / f"registry_{stratum}.json"
    tmp = reg_p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"stratum": stratum, "written": now(), "entries": entries}, indent=1))
    tmp.replace(reg_p)
    return reg_p


def il_no_eval(st, fam_b: list) -> dict:
    """{(arm folder, task): launcher state} of every counted attempt without an .eval, over an IL family's batches (each batch's own
    boxfiles record: its counted_without_eval list and the launcher state from its own launch_subset)."""
    out = {}
    for x in fam_b:
        p = st.d / f"boxfiles_{x['id']}.json"
        if p.exists():
            d = json.loads(p.read_text())
            state_of = {(r.get("arm") or (r.get("key") or "").split("/")[0], r["task"]): r.get("state") for r in json.loads(d["launch_subset"])["runs"]}
            for y in d.get("counted_without_eval", []):
                out[(y["arm"], y["task"])] = state_of.get((y["arm"], y["task"]))
    return out


def il_final_rows(b: dict, fam_b: list, box_: dict, starts: dict, pause_at, common: dict, no_eval: dict) -> list:
    """An IL launch's final run table rows in the snapshot tables' layout: key `<world>|<cell>`, the run's own first_start and the pause
    gate at that start, its status (ran / the launcher's state for a run never counted / not_run) and the batch whose staging took it."""
    where = {}  # run key (arm folder/cell) -> the batch that took it (snapshot k, or the final remainder)
    for x in fam_b:
        for m in x.get("members", []):
            where[m] = x["snap"] if x["kind"] == "ils" else "final"
    order = box_["facts"].get("il_order", {})
    not_counted = {(x["arm"], x["task"]): x["state"] for x in box_["facts"]["not_counted"]}
    # a counted attempt whose folder holds no .eval is never "ran" (a universe count would take "ran" as a scored run): every batch's list
    rows = []
    for c in box_["facts"]["roster"]:
        for w, arms in ((w, a) for w, a in (("run3", ("attack",)), ("sim2", (SIM_ARM, "sim_attack"))) if w in il_worlds(b)):
            took = next((where[f"{a}/{c}"] for a in arms if f"{a}/{c}" in where), None)
            ne = next(((a, c) for a in arms if (a, c) in no_eval), None)
            status = (f"counted, no .eval ({no_eval[ne] or 'state unknown'})" if ne is not None else "ran" if took is not None
                      else next((not_counted[(a, c)] for a in arms if (a, c) in not_counted), "not_run"))
            t0 = starts.get((w, c))
            rows.append({"key": f"{w}|{c}", "cell": c, "world": w, "stage": b.get("stage_of", {}).get(c, "unknown"), "il_order": order.get(c),
                         "snapshot": took, "status": status, "first_start": t0, "pause_gate_at_start": pause_at(t0), "il_batch": b["id"]} | common)
    return rows


def step_hygiene(b: dict, st: State):
    """09's hygiene files for this batch (after its routing; never blocks it): per-arm tool-error tables, attempt labels in 09's enum,
    and for a follow-up the quarter's run table; each names the arm's staged_sha. Then a follow-up routing file."""
    worlds, attack_rows = {}, []
    for t, tv in targets_for(b).items():  # from the manifests, so an arm that staging excluded entirely still has its rows
        man = json.loads((HERE / "config" / tv["manifest"]).read_text())
        for arm in tv["stage_arms"]:
            rows = [f for f in man["files"] if f["arm_dir"] == arm and not f.get("reference_sha256_pin")]  # index donors are never staged
            if not rows or t not in b["targets"]:
                continue
            world = rows[0]["world"]
            worlds[world] = {"arm": arm, "staged_sha": b["stage"][t]["staged_inputs_sha256"].get(f"{arm}.jsonl"),
                             "staged_set": f"{b['stage'][t]['out']}/inputs/{arm}.jsonl",
                             "rows": [{"key": f"{f['world']}|{f['task']}", "path": f["path"], "sha256": f["sha256"]} for f in rows]}
            if arm == "attack" or (arm == sim_arm_of(b) and not sim_is_honest(b)):  # IL batches: sim_arm_of = sim_attack@september
                attack_rows += rows
    lab_files = [b[k] for k in ("labels", "labels_sim2") if b.get(k)]
    label_rows = [json.loads(x) for L in lab_files for x in (HERE / L["file"]).read_text().splitlines()]
    # this batch's cells (an interleaved batch: its RUNS) only: a remainder's labeler reads the whole launch, so its snapshots' are dropped here
    if b["kind"] in IL_KINDS:
        label_rows = [r for r in label_rows if f"{r['arm']}/{r['task']}" in set(b["members"])]
    else:
        label_rows = [r for r in label_rows if cell_of(r["task"]) in set(b["members"])]
    spec = {"batch": f"{b['pool']}_{b['id']}", "kind": b["kind"], "sim_world": sim_world_of(b), "worlds": worlds, "attack_rows": attack_rows, "label_rows": label_rows,
            "labeler": "; ".join(f"{REL}/{L['file']} (sha256 {L['sha256']})" for L in lab_files), "staged_sha": {w: v["staged_sha"] for w, v in worlds.items()},
            "staged_sets": {v["staged_sha"]: {"world": w, "staged_set": v["staged_set"], "batch": f"{b['pool']}_{b['id']}"} for w, v in worlds.items() if v["staged_sha"]}}
    external_rt = b["kind"] in ("fu", "fus") and main_of(st, b) is None  # index donors (no main launch): the run table is built outside the chain
    if external_rt and not b.get("run_table_external"):
        raise Gate(f"{b['id']}: the run table of a follow-up without a main batch (index donors) waits for 09's registration")
    if external_rt:  # ug (09 @ 48363962a Q2): make_run_table_ug.py writes run_table_ug.json after this family's box read
        st.log(f"{b['id']}: run table built outside the chain ({b['run_table_external']})")
    elif b["kind"] == "ep2":  # the key list (counted rows and status rows) as the launch's run table
        kl = json.loads(git("show", f"HEAD:{REL}/{b['keylist']}"))
        rows = [{"key": r["key"], "cell": r["key"].split("|", 1)[1], "status": "counted", "folder": f} for f, rs in kl["folders"].items() for r in rs] + \
               [{"key": r["key"], "cell": r["key"].split("|", 1)[1], "status": r["class"]} for r in kl["status_rows"]]
        spec["run_table"] = {"launch": b["run"], "epoch": "ep2", "keylist": f"{b['keylist']} @ {git('log', '-1', '--format=%h', '--', b['keylist']).strip()}",
                             "sealed": EP2_SEALED, "staged_sha": spec["staged_sha"], "cells": sorted(rows, key=lambda x: x["key"])}
        spec["run_table_name"] = f"run_table_ep2_{b['q']}.json"
    elif b["kind"] == "fu" and b.get("rlr_of"):  # a sim-honest replacement launch (09 @ 91485ce80): a root of its parent batch; its own table, its launch's cells only
        fu_box = json.loads((st.d / f"boxfiles_{b['id']}.json").read_text())
        par = st.s["batches"][b["rlr_of"]]
        par_fam = [par, *snaps_of(st, par)]
        arm = sim_arm_of(b)
        excl = {}  # cell -> staging exclusion reason (keys <launch>/<arm>/<cell>/<file>.eval#e<k>)
        for t in b["targets"]:
            for k, why in staging_exclusions(b, t).items():
                parts = k.split("#", 1)[0].split("/")
                excl[parts[parts.index(arm) + 1]] = why
        not_run = {x["task"]: x["state"] for x in fu_box["facts"]["not_counted"] if x["arm"] == arm}
        if any("boxfiles" not in x["done"] for x in par_fam if x["status"] != "STOPPED"):
            raise Wait(f"the replacement table waits for every batch of {par['id']} to be read")
        ruled = b["rlr_keys"]  # cell -> {class, ruling, root_check}: only keys a committed ruling names for this parent batch
        if set(fu_box["facts"]["roster"]) != set(ruled):
            raise Gate(f"launch cells {sorted(fu_box['facts']['roster'])} != ruled replacement keys {sorted(ruled)}")
        rows = []
        for c in fu_box["facts"]["roster"]:
            orig = [x["id"] for x in par_fam if c in x.get("members", [])]
            if len(orig) != 1:
                raise Gate(f"{c}: in {len(orig)} batches of {par['id']} (a replacement supersedes exactly one original run)")
            status = "excluded" if c in excl else "ran" if c in b.get("members", []) else "replacement_not_counted"  # no fallback to the original
            rows.append({"key": f"{sim_world_of(b)}|{c}", "cell": c, "sim_launch": b["run"], "sim_status": status,
                         **({"staging_exclusion_reason": excl[c]} if c in excl else {}),
                         **({"sim_status_detail": {"launcher_state": not_run.get(c, "absent")}} if status == "replacement_not_counted" else {}),
                         "supersedes": {"root": orig[0], "launch": par["run"], "run_key": f"{arm}/{c}"}, "replacement_launch_id": bare_id(b["run"]),
                         "replacement_batch": b["id"], "replacement_kind": "rlr", **{k: ruled[c][k] for k in ("class", "ruling", "root_check")}})
        spec["run_table"] = {"pool": b["pool"], "rlr_of": par["id"], "grammar": "09 @ 91485ce80", "staged_sha": spec["staged_sha"], "cells": rows}
        spec["run_table_name"] = f"run_table_{b['sim_tag'].removeprefix('sim_')}_{b['pool']}_{qtok(b)}.json"  # rlr / rlr<k>; not matched by run_table_<pool>_*
    elif b["kind"] == "fu":  # the quarter's run table: every cell of its main launch; sims = every snapshot's and this remainder's
        mb = main_of(st, b)
        main_box = json.loads((st.d / f"boxfiles_{mb['id']}.json").read_text())
        fu_box = json.loads((st.d / f"boxfiles_{b['id']}.json").read_text())
        snaps = snaps_of(st, b)
        if any("boxfiles" not in x["done"] for x in snaps if x["status"] != "STOPPED"):
            raise Wait("the run table waits for every snapshot's membership")
        sims = set(b.get("members", [])) | {m for x in snaps for m in x.get("members", [])}  # every simulated run that started
        refs = fu_box.get("references", [])
        cap = "SIM_CAP_PAUSE.txt" in fu_box.get("markers", [])
        not_run = {x["task"]: x["state"] for x in fu_box["facts"]["not_counted"] if x["arm"] == sim_arm_of(b)}
        cells = main_box["facts"]["roster"]  # every cell the main launch lists (never-started and failed cells included)
        if any("boxfiles" not in x["done"] for x in snaps_of(st, mb) if x["status"] != "STOPPED"):
            raise Wait("the run table waits for every main snapshot's membership")
        real_batch = {c: x["id"] for x in [mb, *snaps_of(st, mb)] for c in x.get("members", [])}  # which batch staged each cell's real arms
        first = st.s["batches"][b["rerun_of"]] if b.get("rerun_of") else None  # a rerun: the quarter's table merges both follow-ups
        if first is not None:
            if "boxfiles" not in first["done"] or any("boxfiles" not in x["done"] for x in snaps_of(st, first) if x["status"] != "STOPPED"):
                raise Wait("the merged run table waits for the first follow-up's membership")
            first_sims = set(first.get("members", [])) | {m for x in snaps_of(st, first) for m in x.get("members", [])}
            excl = exclusions_of(first)
            rerun_tasks = {x["task"] for x in fu_box["facts"]["not_counted"]} | sims  # every cell the rerun launch lists (started or not)
        rows = []
        for c in cells:
            launches = None
            if first is not None:
                f_ = {"launch": first["run"], "status": "excluded" if c in excl else ("ran" if c in first_sims else "not_run")}
                if c in excl:
                    f_ |= {"excluded": True, "exclusion_cause": excl[c]["exclusion_cause"], "manifest_key": excl[c]["manifest_key"]}
                launches = [f_] + ([{"launch": b["run"], "status": "ran" if c in sims else "not_run"}] if c in rerun_tasks else [])
                if b.get("rerun_reason") and c in rerun_tasks and c not in sims:  # a listed replacement never counted: no fallback (09 @ bffe22e3e)
                    rows.append({"cell": c, "attack_pair_launch": "same", "run1_launch": mb["run"], "run2_launch": mb["run"], "real_batch": real_batch.get(c),
                                 "sim_launch": b["run"], "sim_status": "replacement_not_counted", "sim_status_detail": {"launcher_state": not_run.get(c, "absent")},
                                 "sim_launches": launches, "rerun_of": {"launch": first["run"], "run_key": f"{sim_arm_of(b)}/{c}"},
                                 "rerun_reason": b["rerun_reason"], "replacement_launch_id": bare_id(b["run"])})
                    continue
                if c not in sims and c in first_sims and c not in excl:  # the first follow-up's ended run counts
                    rows.append({"cell": c, "attack_pair_launch": "same", "run1_launch": mb["run"], "run2_launch": mb["run"], "real_batch": real_batch.get(c),
                                 "sim_launch": first["run"], "sim_status": "ran", "sim_launches": launches})
                    continue
            if c in sims:
                rerun = ({"rerun_of": {"launch": first["run"], "run_key": f"{sim_arm_of(b)}/{c}"}, "rerun_reason": b.get("rerun_reason"),
                          "replacement_launch_id": bare_id(b["run"])} if first is not None and b.get("rerun_reason") else {})  # 09 @ bffe22e3e (S5r)
                rows.append({"cell": c, "attack_pair_launch": "same", "run1_launch": mb["run"], "run2_launch": mb["run"], "real_batch": real_batch.get(c),
                             "sim_launch": b["run"], "sim_status": "ran", **({"sim_launches": launches} if launches else {}), **rerun})
                continue
            has_ref = any(re.search(rf"(^|[/_@]){re.escape(c)}\.eval$", r) for r in refs)  # the follow-up's pinned honest_a copies, by exact name
            rows.append({"cell": c, "attack_pair_launch": "same", "run1_launch": mb["run"], "run2_launch": mb["run"], "real_batch": real_batch.get(c), "sim_launch": "NOT_RUN",
                         "sim_status": "no_honest_A_reference" if not has_ref else "cap_stopped",
                         "sim_status_detail": {"pinned_reference": has_ref, "SIM_CAP_PAUSE": cap, "launcher_state": not_run.get(c, "absent")},
                         **({"sim_launches": launches} if launches else {})})
        missing = set(cells) - {r["cell"] for r in rows}
        if missing:
            raise Gate(f"run table misses {len(missing)} cells of the main launch's roster")
        spec["run_table"] = {"quarter": b["q"], "pool": b["pool"], "staged_sha": spec["staged_sha"], "cells": rows}
        spec["run_table_name"] = f"run_table_{b['pool']}_{b['q']}.json"
        if first is not None and b.get("rerun_reason"):  # a rerun family with a stated reason (S5r): its merged table is a NEW file naming the one it replaces
            spec["run_table"]["supersedes"] = spec["run_table_name"]
            spec["run_table_name"] = f"run_table_{b['pool']}_{qtok(b)}.json"
    if b["kind"] == "fus" and not external_rt:  # 09: this snapshot's run-table rows as it lands (own file, append-only; the quarter's final table supersedes them)
        mb = main_of(st, b)
        real_batch = {c: x["id"] for x in [mb, *snaps_of(st, mb)] for c in x.get("members", [])}
        excl = exclusions_of(b)
        rows = [{"key": f"{sim_world_of(b)}|{c}", "cell": c, "attack_pair_launch": "same", "run1_launch": mb["run"], "run2_launch": mb["run"], "real_batch": real_batch.get(c),
                 "sim_launch": b["run"], "sim_status": "ran", "snapshot": b["snap"]} for c in b["members"] if c not in excl]
        spec["run_table"] = {"quarter": b["q"], "pool": b["pool"], "snapshot": b["snap"], "superseded_by": f"run_table_{b['pool']}_{b['q']}.json",
                             "staged_sha": spec["staged_sha"], "cells": rows}
        spec["run_table_name"] = f"run_table_snap_{b['pool']}_{qtok(b)}_s{b['snap']}.json"  # not matched by combine_final's run_table_<pool>_*
    if b["kind"] in IL_KINDS:  # 09's IL run table: per cell (final) or per run (snapshot), with stage, il_order and the pause gate at its start
        fam_b = [b, *snaps_of(st, b)] if b["kind"] == "il" else [b]
        if b["kind"] == "il" and any("boxfiles" not in x["done"] for x in fam_b if x["status"] != "STOPPED"):
            raise Wait("the IL run table waits for every snapshot's membership")
        box_ = json.loads((st.d / f"boxfiles_{b['id']}.json").read_text())
        pauses = [(x[:19], x) for x in box_.get("pause_events", [])]
        starts = {}
        for x in fam_b:
            if not x.get("commit_targets"):
                continue
            for t, tv in targets_for(x).items():
                for f in json.loads((HERE / "config" / tv["manifest"]).read_text())["files"]:
                    if f["world"] in ("run3", "sim2") and f.get("first_start"):
                        starts[(f["world"], f["task"])] = f["first_start"]

        def pause_at(t0):
            if not t0:
                return None
            t0 = t0.replace("T", " ")[:19]
            before = [line for ts, line in pauses if ts <= t0]
            return before[-1] if before else "none"

        order = box_["facts"].get("il_order", {})
        common = {"launch_id": bare_id(b["run"]), "main_launch": b["main_run"], "interleaved_launch": True, "sim_attempt_timeout_s": 300}
        if b["kind"] == "il":  # the final table: the snapshot tables' layout (Amendment 20), one row per run (world|cell), every roster cell
            spec["run_table"] = {"launch": b["run"], "pool": b["pool"], "staged_sha": spec["staged_sha"], "layout": "world-keyed (as the snapshot tables)",
                                 "cells": il_final_rows(b, fam_b, box_, starts, pause_at, common, il_no_eval(st, fam_b))}
            spec["run_table_name"] = f"run_table_{b['name_prefix']}_{b['q']}{b.get('run_table_suffix', '')}.json"
        else:
            rows = []
            for m in sorted(b.get("members", [])):
                arm, c = m.split("/", 1)
                w = "run3" if arm == "attack" else "sim2"
                rows.append({"key": f"{w}|{c}", "cell": c, "world": w, "stage": b.get("stage_of", {}).get(c, "unknown"), "il_order": order.get(c), "snapshot": b["snap"],
                             "first_start": starts.get((w, c)), "pause_gate_at_start": pause_at(starts.get((w, c)))} | common)
            spec["run_table"] = {"launch": b["run"], "pool": b["pool"], "snapshot": b["snap"], "superseded_by": f"run_table_{b['name_prefix']}_{b['q']}.json",
                                 "staged_sha": spec["staged_sha"], "cells": rows}
            spec["run_table_name"] = f"run_table_snap_{b['name_prefix']}_{b['q']}_s{b['snap']}.json"
    (st.d / "hygiene").mkdir(exist_ok=True)
    sp_path = st.d / "hygiene" / f"spec_{b['id']}.json"
    sp_path.write_text(json.dumps(spec))
    reserve(st, b, f"hygiene {b['id']}", CPU_WORST_USD["hygiene"])
    out_dir = HERE / "results" / "hygiene"
    out = modal_run([f"{REL}/hygiene_chain.py", "--spec", str(sp_path), "--out", str(out_dir)], st, f"hygiene_{b['id']}", b=b)
    # (run tables are named run_table_<quarter>.json by hygiene_chain.py from the batch id's first two characters; batch ids keep q first)
    written = json.loads(next(line for line in out.splitlines() if line.startswith('{"written"')))["written"]
    paths = [f"{REL}/results/hygiene/{n}" for n in written]
    b["hygiene_commit"] = commit(paths, f"Overnight chain: {b['id']} hygiene files (tool errors, attempt labels{', run table' if b['kind'] in FU_KINDS else ''})", st)
    (st.d / "routing").mkdir(exist_ok=True)
    (routing_dir(st, b) / f"{b['id']}_hygiene.json").write_text(json.dumps({"batch": b["id"], "commit": b["hygiene_commit"], **({"sealed": EP2_SEALED} if b["kind"] == "ep2" else {}), "files": written,
                                                                          "paths": paths}, indent=1))
    # the final combined files are written by the main loop once every batch is terminal (done or STOPPED)


def pool_files(found: list, kind: str, pf: str, others: set, d: Path) -> list:
    """The files <kind>_<pf>_*.json of name prefix pf, less those of a LONGER configured pool or name prefix in `others` (the glob for
    sp493 also matches sp493_g4_simh's files; d9 10-15 19:1x): a file named <kind>_<o>_* or <kind>_<o>.json for o = <pf>_... is o's."""
    longer = [o for o in others if o.startswith(f"{pf}_")]
    return [d / n for n in found if not any(n.startswith(f"{kind}_{o}_") or n == f"{kind}_{o}.json" for o in longer)]


def ambiguous_names(batches: list, pools: set) -> list:
    """pool_files' guard assumes ownership by name: a batch's file-name token (id, q, qtok) after its pool or IL prefix must never spell a
    LONGER configured pool name (e.g. a sp493 batch named g4_simh would write labels_sp493_g4_simh.json, which the guard gives to pool
    sp493_g4_simh). Returns the offending "<prefix>_<token> ~ <longer name>" strings; the loader stops on any (Codex @ 19:18)."""
    names = set(pools) | {x["name_prefix"] for x in batches if x.get("name_prefix")}
    bad = []
    for x in batches:
        own = {x["pool"]} | ({x["name_prefix"]} if x.get("name_prefix") else set())
        for pf in own:
            for t in {x["id"], x["q"], x.get("qtok", x["q"])} - {""}:
                n = f"{pf}_{t}"
                bad += [f"{n} ~ {o}" for o in names - own if o.startswith(f"{pf}_") and (n == o or n.startswith(f"{o}_"))]
    return sorted(set(bad))


def combine_final(st: State, pool: str):
    """After a pool's last batch: its combined run table and attempt labels, from the committed per-batch files (keys only).
    The scale-up's names are run_table_311.json and attempt_labels_311.json (09's)."""
    if pool == "ep2":  # epoch-2: the per-launch run tables (sealed; 09's §4 pool-key note decides pooling) are the final files; nothing combined
        st.log("final combine (ep2): not combined; the per-launch run tables run_table_ep2_<q>.json stand (sealed until 09's §4 note)")
        return
    d = HERE / "results" / "hygiene"
    own = {pool} | {x["name_prefix"] for x in st.s["batches"].values() if x["pool"] == pool and x.get("name_prefix")}
    names = set(st.s["pools"]) | {x["name_prefix"] for x in st.s["batches"].values() if x.get("name_prefix")}

    def mine(kind: str, pf: str) -> list:
        return pool_files(sorted(q.name for q in d.glob(f"{kind}_{pf}_*.json")), kind, pf, names - own, d)

    cells, labels, rules, counts, by_rule, staged = [], {}, {}, {}, {}, {}
    excl_runs = {}  # (launch, cell) -> exclusion record
    for p in mine("exclusions", pool):
        for r in json.loads(p.read_text())["excluded"]:
            excl_runs[(r["launch"], r["cell"])] = r
    prefixes = sorted({x.get("name_prefix", pool) for x in st.s["batches"].values() if x["pool"] == pool} | {pool})
    tables = sorted({q for pf in prefixes for q in mine("run_table", pf)})
    gone = {json.loads(q.read_text()).get("supersedes") for q in tables} - {None}  # a corrected table replaces the file it names
    for p in [q for q in tables if q.name not in gone]:
        rt = json.loads(p.read_text())
        if rt.get("pool", pool) != pool:  # another pool's table under a shared name prefix (il_retry's run_table_ilnm_*_try2)
            continue
        for c in rt["cells"]:  # an excluded run never counts as the cell's sim run, whichever table carries it
            r = excl_runs.get((c.get("sim_launch"), c["cell"]))
            if r is not None:
                c |= {"sim_launch": "NOT_RUN", "sim_status": "excluded", "excluded_run": {"launch": r["launch"], "excluded": True,
                      "exclusion_cause": r["exclusion_cause"], "manifest_key": r["manifest_key"]}}
        cells += rt["cells"]
        staged |= rt.get("staged_sets", {})
    for p in sorted(q for q in d.glob(f"run_table_rlr*_{pool}_*.json") if RLR_TABLE.fullmatch(q.name.split(f"_{pool}_")[0])):
        # sim-honest replacements (09 @ 91485ce80, _rlr<k> @ 502c1b05e; d9 10-15 ~14:35): both runs stay in the accounting
        rt = json.loads(p.read_text())
        if rt.get("pool") != pool or not rt.get("rlr_of") or st.s["batches"].get(rt["rlr_of"], {}).get("pool") != pool:
            continue  # a name collision with another pool (e.g. pool <x>_extra): only this pool's replacement tables, by their own metadata
        for r in rt["cells"]:  # the original keeps its row, marked superseded with the ruling and root check; the replacement's row is authoritative
            orig = [c for c in cells if c["cell"] == r["cell"] and r["supersedes"]["launch"] in (c.get("sim_launch"), c.get("excluded_run", {}).get("launch"))]
            if len(orig) != 1:
                raise RuntimeError(f"{p.name}: {r['cell']} matches {len(orig)} original rows of launch {r['supersedes']['launch']} in the {pool} tables")
            if "superseded_by" in orig[0]:  # one replacement per original run: a second root for the same cell needs a ruling
                raise RuntimeError(f"{p.name}: {r['cell']}'s original row is already superseded by {orig[0]['superseded_by']}")
            orig[0] |= {"sim_status": f"superseded by {r['replacement_launch_id']}", "superseded_by": {"launch": r["sim_launch"], "batch": r["replacement_batch"]},
                        "supersession": {k: r[k] for k in ("class", "ruling", "root_check")}}
            cells.append(r | {"authoritative": True})
        staged |= rt.get("staged_sets", {})
    excluded = {}  # batch -> {sim|cell: record}
    for p in mine("exclusions", pool):
        ex = json.loads(p.read_text())
        excluded[f"{pool}_{ex['batch']}"] = {r["key"]: r for r in ex["excluded"]}
    for rb in [x for x in st.s["batches"].values() if x["pool"] == pool and x["kind"] == "fu" and x.get("rerun_reason") and x.get("rerun_of")]:
        # a rerun family with a stated reason (S5r): every cell its launch lists supersedes the first family's sim run of that cell, counted
        # replacement or not (no fallback; 09 @ bffe22e3e), so the first family's labels for those cells leave the combined labels
        fb = st.s["batches"][rb["rerun_of"]]
        rbox = json.loads((st.d / f"boxfiles_{rb['id']}.json").read_text())
        listed = {x["task"] for x in rbox["facts"]["not_counted"]} | set(rb.get("members", []))
        for x in [fb, *snaps_of(st, fb)]:
            for c in listed & set(x.get("members", [])):  # only where the first family actually took the cell's sim run
                excluded.setdefault(f"{pool}_{x['id']}", {})[f"sim|{c}"] = {"exclusion_cause": f"superseded by the rerun launch {rb['run']} ({rb['rerun_reason']})",
                                                                           "launch": fb["run"], "superseded_by": rb["run"]}
    for p in mine("labels", pool):
        lab = json.loads(p.read_text())
        drop = excluded.get(lab["batch"], {})
        if drop:  # the excluded runs never count: their labels and counts leave this batch's contribution
            for k in drop:
                if k in lab["labels"]:
                    v, rule, w = lab["labels"].pop(k), lab.get("rules", {}).pop(k, None), k.split("|")[0]
                    lab["counts"][w][v] -= 1
                    lab.setdefault("counts_by_label_rule", {}).setdefault(w, {})[f"{v}|{rule}"] -= 1
        labels |= lab["labels"]
        rules |= lab.get("rules", {})
        for src, dst in ((lab["counts"], counts), (lab.get("counts_by_label_rule", {}), by_rule)):
            for w, c in src.items():
                for k, v in c.items():
                    dst.setdefault(w, {}).setdefault(k, 0)
                    dst[w][k] += v
        staged |= lab.get("staged_sets", {})
    for c in cells:
        if "run1_launch" in c:
            assert c["run1_launch"] == c["run2_launch"], c["cell"]
    suffix = "311" if pool == "su311" else pool
    rt_p, lab_p = d / f"run_table_{suffix}.json", d / f"attempt_labels_{suffix}.json"
    ext = any(x.get("run_table_external") for x in st.s["batches"].values() if x["pool"] == pool)  # ug: its run table is built outside the chain
    if not ext:
        rt_p.write_text(json.dumps({"cells": cells, "dropped": DROPPED if pool == "su311" else [], "staged_sets": staged}, indent=1) + "\n")
    lab_p.write_text(json.dumps({"labels": labels, "rules": rules, "counts": counts, "counts_by_label_rule": by_rule, "staged_sets": staged,
                                 "excluded": {k: {"batch": bt, "exclusion_cause": r["exclusion_cause"], "launch": r["launch"]} for bt, d_ in excluded.items() for k, r in d_.items()}},
                                indent=1) + "\n")  # staged_sets: ONE entry per staged set, keyed by its staged sha (09 matches by key)
    finals = [f"{REL}/results/hygiene/{lab_p.name}"] + ([] if ext else [f"{REL}/results/hygiene/{rt_p.name}"])
    c = commit(finals, f"Overnight chain: final combined {'attempt labels' if ext else 'run table and attempt labels'} ({pool})", st)
    (st.d / "routing").mkdir(exist_ok=True)
    (st.d / "routing" / f"final_hygiene_{pool}.json").write_text(json.dumps({"pool": pool, "commit": c, "cells": len(cells), "labels": len(labels), "counts": counts,
                                                                           "paths": finals}, indent=1))
    st.log(f"final hygiene ({pool}) combined @ {c[:10]}: {len(cells)} cells, {len(labels)} labels")


def provenance_il(b: dict, st: State):
    """09's amendment A for an interleaved batch: per run3/sim2 run of this batch (key <world>|<cell>), every attack arm of the cell
    the chain has relayed (run1, run2 from the main family; sim1 from the quarter's follow-up families, excluded runs left out; run3,
    sim2 from the interleaved family) hashed and compared pairwise (provenance_chain.py, mode pairwise)."""
    if not b.get("targets"):
        return True
    mb = main_of(st, b)

    def rows_of(x, worlds):
        out = []
        for t in x.get("targets", []):
            for f in json.loads((HERE / "config" / targets_for(x)[t]["manifest"]).read_text())["files"]:
                if f["world"] in worlds:
                    out.append((t, f))
        return out

    arms = {}  # (world, cell) -> manifest row (the T196 launch: its run1/run2/sim1 are not relayed by this chain; only run3 and sim2)
    for x in ([*snaps_of(st, mb), mb] if mb is not None else []):
        if x.get("commit_targets"):
            for _, f in rows_of(x, ("run1", "run2")):
                arms[(f["world"], f["task"])] = f
    for x in st.s["batches"].values():
        if mb is not None and x["kind"] in ("fu", "fus") and x.get("commit_targets") and (main_of(st, x) or {}).get("id") == mb["id"]:  # index-donor families: no main
            ex = exclusions_of(x)
            for _, f in rows_of(x, ("sim",)):
                if f["task"] not in ex:
                    arms[("sim1", f["task"])] = f
    head = st.s["batches"].get(fam(b), b)
    for x in [head, *snaps_of(st, head)]:
        if x.get("commit_targets"):
            for _, f in rows_of(x, ("run3", "sim2")):
                arms.setdefault((f["world"], f["task"]), f)
    rows = []
    for t, f in rows_of(b, ("run3", "sim2")):
        arm = "attack" if f["world"] == "run3" else SIM_ARM
        if f"{arm}/{f['task']}" not in set(b.get("members", [])):
            continue
        c = f["task"]
        worlds = {w: {"path": g["path"], "sha256": g["sha256"], "launch_task": g["launch_task"], "launch": g["launch"]}
                  for w in ("run1", "run2", "sim1", "run3", "sim2") if (g := (f if w == f["world"] else arms.get((w, c))))}
        rows.append({"key": f"{f['world']}|{c}", "cell": c, "subject": f["world"], "worlds": worlds,
                     "staged_shas": {f["world"]: b["stage"][t]["staged_inputs_sha256"].get(f"{arm}.jsonl")}})
    name = f"approach_provenance_{b['name_prefix']}_{b['q']}{'_s' + str(b['snap']) if b['kind'] == 'ils' else ''}{b.get('target_tag', '')}{b.get('rev', '')}.json"  # a re-stage: a new file
    out = HERE / "results" / "hygiene" / name
    if not out.exists():
        box_ = json.loads((st.d / f"boxfiles_{b['id']}.json").read_text())
        spec = {"mode": "pairwise", "batch": b["id"], "pool": b["pool"], "q": b["q"], "snap": b.get("snap"), "main_run": b["main_run"], "sim_run": b["run"],
                "policy": box_["facts"].get("policy"), "rows": rows}
        sp_path = st.d / "hygiene" / f"prov_spec_{b['id']}.json"
        sp_path.parent.mkdir(exist_ok=True)
        sp_path.write_text(json.dumps(spec))
        reserve(st, b, f"provenance {b['id']}", CPU_WORST_USD["provenance"])
        modal_run([f"{REL}/provenance_chain.py", "--spec", str(sp_path), "--out", str(out)], st, f"provenance_{b['id']}", b=b)
    d = json.loads(out.read_text())
    b["provenance"] = {"file": name, "pass": d["pass"], "counts": d["counts"]}
    b["provenance_commit"] = commit([f"{REL}/results/hygiene/{name}"], f"Overnight chain: {b['id']} approach provenance, pairwise (pass {d['pass']})", st)
    (routing_dir(st, b) / f"{b['id']}_provenance.json").write_text(json.dumps({"batch": b["id"], "commit": b["provenance_commit"], "path": f"{REL}/results/hygiene/{name}",
                                                                            "pass": d["pass"], "counts": d["counts"]}, indent=1))
    return True


def provenance_ug(b: dict, st: State):
    """09's ug note @ 48363962a Q1 (Append 34 §7): per staged cell of this family, run1 / run2 / sim1 from the committed comparability
    map (FINAL's artifacts; S_old .evals on 2e's Volume) and sim_ug from this family's own manifest, hashed by provenance_chain.py's ug
    mode. A cell staging excluded is not a row (its exclusion stands in the staging record)."""
    cm = b["comparability_map"]
    raw = sp.run(["git", "show", f"{cm['commit']}:{cm['path']}"], capture_output=True, check=True, cwd=REPO).stdout
    if hashlib.sha256(raw).hexdigest() != cm["sha256"]:
        raise Gate(f"{b['id']}: the comparability map differs from its pinned sha256")
    cmap = json.loads(raw)["cells"]
    (t,) = b["targets"]
    man = json.loads((HERE / "config" / targets_for(b)[t]["manifest"]).read_text())
    excl = {k.split("#", 1)[0] for k in staging_exclusions(b, t)}
    ssha = b["stage"][t]["staged_inputs_sha256"].get(f"{sim_arm_of(b)}.jsonl")
    rows = []
    for f in man["files"]:
        if f["world"] != sim_world_of(b) or f["path"] in excl:
            continue
        m = cmap.get(f"{sim_world_of(b)}|{f['task']}")
        if m is None:
            raise Gate(f"{b['id']}: {f['task']} is not in the comparability map")
        w = {"sim_ug": {"root": "/ev", "path": f"evals_v1/{f['path']}", "sha256": f["sha256"], "key": f["path"]}}
        for x in ("run1", "run2", "sim1"):
            a = m[x]
            w[x] = None if a is None else {"root": "/src" if a["volume"].endswith("q4own_evals_kill-2026-11-13") else "/ev", "path": a["volume_path"],
                                           "sha256": a["eval_sha256"], "key": a["row_key"]}
        rows.append({"cell": f["task"], "origin": m["origin"], "worlds": w, "staged_shas": [None, None, ssha],
                     **{f"{x}_null_reason": m.get(f"{x}_null_reason") for x in ("run1", "run2", "sim1") if m[x] is None}})
    box_ = json.loads((st.d / f"boxfiles_{b['id']}.json").read_text())
    name = f"approach_provenance_{b['pool']}_{b['id']}.json"
    out = HERE / "results" / "hygiene" / name
    if not out.exists():
        spec = {"mode": "ug", "batch": b["id"], "pool": b["pool"], "sim_run": b["run"], "comparability_map": cm, "policy": box_["facts"].get("policy"), "rows": rows}
        sp_path = st.d / "hygiene" / f"prov_spec_{b['id']}.json"
        sp_path.parent.mkdir(exist_ok=True)
        sp_path.write_text(json.dumps(spec))
        reserve(st, b, f"provenance {b['id']}", CPU_WORST_USD["provenance"])
        modal_run([f"{REL}/provenance_chain.py", "--spec", str(sp_path), "--out", str(out)], st, f"provenance_{b['id']}", b=b)
    d = json.loads(out.read_text())
    b["provenance"] = {"file": name, "pass": d["pass"], "counts": d["counts"]}
    b["provenance_commit"] = commit([f"{REL}/results/hygiene/{name}"], f"Overnight chain: {b['id']} ug comparability (pass {d['pass']})", st)
    (routing_dir(st, b) / f"{b['id']}_provenance.json").write_text(json.dumps({"batch": b["id"], "commit": b["provenance_commit"], "path": f"{REL}/results/hygiene/{name}",
                                                                            "pass": d["pass"], "counts": d["counts"]}, indent=1))
    return True


def provenance_sp493_map(b: dict, st: State):
    """S_paper493 sim attack (09's ruling via d9 10-15 ~19:3x; map spot-checked by 09 @ c3931a963): per staged cell of this batch, the pinned
    comparability map's entry <sim_world>|<cell> must exist, be admitted, and carry equal, non-empty registered (facts v2 q2_approach_sha256),
    sim on-box and real-arm on-box approach sha256s. Keys and hashes only. A failing cell fails the attestation loudly (09 rules its status row)."""
    cm = b["comparability_map"]
    raw = sp.run(["git", "show", f"{cm['commit']}:{cm['path']}"], capture_output=True, check=True, cwd=REPO).stdout
    if hashlib.sha256(raw).hexdigest() != cm["sha256"]:
        raise Gate(f"{b['id']}: the comparability map differs from its pinned sha256")
    cmap = json.loads(raw)["map"]
    (t,) = b["targets"]
    man = json.loads((HERE / "config" / targets_for(b)[t]["manifest"]).read_text())
    excl = {k.split("#", 1)[0] for k in staging_exclusions(b, t)}
    ssha = b["stage"][t]["staged_inputs_sha256"].get(f"{sim_arm_of(b)}.jsonl")
    cells = []
    for f in man["files"]:
        if f["world"] != sim_world_of(b) or f["path"] in excl:
            continue
        key = f"{sim_world_of(b)}|{f['task']}"
        m = cmap.get(key) or {}
        shas = [m.get(k) for k in ("q2_approach_sha256", "sim_onbox_approach_sha256", "real_onbox_approach_sha256")]
        ok = m.get("key") == f["task"] and m.get("admitted") is True and all(shas) and len(set(shas)) == 1  # outer key <sim_world>|<cell>; inner key the bare cell
        cells.append({"cell": f["task"], "key": key, "staged_shas": [None, None, ssha], "pass": ok, "admitted": m.get("admitted"),
                      "real_roots": [m.get("real_r1_root"), m.get("real_r2_root")],
                      **({} if ok else {"reason": "not in the map" if not m else (m.get("reasons") or "approach sha256s unequal or empty")})})
    d = {"mode": "sp493_approach_map", "batch": b["id"], "pool": b["pool"], "sim_run": b["run"], "comparability_map": cm, "cells": cells,
         "pass": bool(cells) and all(c["pass"] for c in cells), "counts": {"cells": len(cells), "pass": sum(c["pass"] for c in cells)}}
    name = f"approach_provenance_{b['pool']}_{b['id']}.json"
    out = HERE / "results" / "hygiene" / name

    def prepare():  # inside the commit's GIT section
        out.write_text(json.dumps(d, indent=1) + "\n")

    b["provenance"] = {"file": name, "pass": d["pass"], "counts": d["counts"]}
    b["provenance_commit"] = commit([f"{REL}/results/hygiene/{name}"], f"Overnight chain: {b['id']} S_paper493 approach map check (pass {d['pass']})", st, prepare=prepare)
    (routing_dir(st, b) / f"{b['id']}_provenance.json").write_text(json.dumps({"batch": b["id"], "commit": b["provenance_commit"], "path": f"{REL}/results/hygiene/{name}",
                                                                            "pass": d["pass"], "counts": d["counts"]}, indent=1))
    return True


def step_provenance(b: dict, st: State):
    """09's approach provenance (Append 21 section 5) for a simulated batch's cells: run1, run2 and sim, attack arms only (honest arms
    are exempt). Main batches: nothing here; their attack sets are covered cell by cell by the simulated batches."""
    if b["kind"] in IL_KINDS:
        return provenance_il(b, st)
    if b["kind"] == "ep2":  # real honest arms: exempt
        return True
    if b["kind"] not in ("fu", "fus") or not b.get("targets"):
        return True
    if sim_is_honest(b):  # a simulated honest arm: exempt, as the honest arms
        return True
    mb = main_of(st, b)
    if mb is None and not b.get("comparability_map") and b.get("provenance_pending"):  # a registered map is coming (S_paper493 g1-g3, 09 10-15 ~19:3x)
        raise Wait(f"{b['id']}: approach provenance waits for {b['provenance_pending']}")
    if mb is None and not b.get("comparability_map"):  # index donors (no main batch) and no registered map: stop loudly
        raise Gate(f"{b['id']}: approach provenance for a follow-up without a main batch (index donors) waits for 09's registration")
    if mb is None and b.get("provenance_mode") == "sp493_approach_map":
        return provenance_sp493_map(b, st)
    if mb is None and b.get("provenance_mode", "ug") != "ug":  # any other mode: no check written for it
        raise Gate(f"{b['id']}: provenance mode {b['provenance_mode']} is not implemented yet (a map alone is not enough; Codex 10-15 19:28)")
    if mb is None:
        return provenance_ug(b, st)
    real = {"run1": {}, "run2": {}}  # world -> cell -> (manifest row, the attack staged set's sha)
    for x in [*snaps_of(st, mb), mb]:
        if not x.get("commit_targets"):
            continue
        for t in x.get("targets", []):
            man = json.loads((HERE / "config" / targets_for(x)[t]["manifest"]).read_text())
            ssha = x.get("stage", {}).get(t, {}).get("staged_inputs_sha256", {}).get("attack.jsonl")
            for f in man["files"]:
                if f["world"] in real:
                    real[f["world"]][f["task"]] = (f, ssha)
    (t,) = b["targets"]
    man = json.loads((HERE / "config" / targets_for(b)[t]["manifest"]).read_text())
    fu_ = st.s["batches"].get(fam(b), b)
    if fu_.get("exclude_first"):  # a rerun family publishes only after the first follow-up's provenance no longer holds the re-run cells
        first = st.s["batches"][fu_["exclude_first"]["batch"]]
        if not first.get("exclusions_applied") or first.get("provenance_supersede") or "attest" not in first["done"]:
            raise Wait(f"waits for {first['id']}'s provenance and attestations without its excluded cells")
    excl = exclusions_of(b)  # EXCLUDED cells (e.g. the Q2 first follow-up's Ctrl+C-stopped runs) are never provenance rows
    sims = {f["task"]: (f, b["stage"][t]["staged_inputs_sha256"].get(f"{sim_arm_of(b)}.jsonl")) for f in man["files"] if f["world"] == sim_world_of(b) and f["task"] not in excl}
    cells = []
    for c in sorted(sims):
        row = {"cell": c, "staged_shas": []}
        for w, src in (("run1", real["run1"].get(c)), ("run2", real["run2"].get(c)), ("sim", sims[c])):
            row[w] = None if src is None else {"path": src[0]["path"], "sha256": src[0]["sha256"], "launch_task": src[0]["launch_task"]}
            row["staged_shas"].append(None if src is None else src[1])
        cells.append(row)
    box_ = json.loads((st.d / f"boxfiles_{b['id']}.json").read_text())
    name = f"approach_provenance_{b['pool']}_{qtok(b)}{'_s' + str(b['snap']) if b['kind'] == 'fus' else ''}.json"
    out = HERE / "results" / "hygiene" / name
    if b.pop("provenance_supersede", False) and out.exists():  # exclusions recorded after it was written: a new version, same name
        out.unlink()
    if not out.exists():  # write-once: a file left by a run whose commit failed is reused
        spec = {"batch": b["id"], "pool": b["pool"], "q": b["q"], "snap": b.get("snap"), "main_run": b["main_run"], "sim_run": b["run"],
                "sim_policy": box_["facts"].get("policy"), "cells": cells}
        sp_path = st.d / "hygiene" / f"prov_spec_{b['id']}.json"
        sp_path.parent.mkdir(exist_ok=True)
        sp_path.write_text(json.dumps(spec))
        reserve(st, b, f"provenance {b['id']}", CPU_WORST_USD["provenance"])
        modal_run([f"{REL}/provenance_chain.py", "--spec", str(sp_path), "--out", str(out)], st, f"provenance_{b['id']}", b=b)
    d = json.loads(out.read_text())
    b["provenance"] = {"file": name, "pass": d["pass"], "counts": d["counts"]}
    b["provenance_commit"] = commit([f"{REL}/results/hygiene/{name}"], f"Overnight chain: {b['id']} approach provenance (pass {d['pass']})", st)
    (routing_dir(st, b) / f"{b['id']}_provenance.json").write_text(json.dumps({"batch": b["id"], "commit": b["provenance_commit"], "path": f"{REL}/results/hygiene/{name}",
                                                                            "pass": d["pass"], "counts": d["counts"]}, indent=1))
    return True


def evidence(path: Path) -> dict:
    return {"path": f"{REL}/{path.relative_to(HERE)}", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def writer_coverage(root: str, recs: list) -> dict:
    """09 via d9 10-15 (2e's finding on ilnm_s5_sim2): every chunk the root holds on the Volume must come from a recorded shard (the
    writer's own record, or the record the orchestrator recovered from it) with no error that ran on one H200."""
    import modal

    vol = modal.Volume.from_name(VOL, version=2)
    with unlocked():
        chunks = sorted(e.path.split("/")[-1][: -len(".json")] for e in vol.listdir(f"{root}/chunks") if e.path.endswith(".json"))
    covered = {c["chunk"] for p in recs for s in json.loads(p.read_text())["shards"]
               if "error" not in s and s.get("gpu_name") == "NVIDIA H200" and s.get("gpu_count") == 1 for c in s.get("chunks") or []}
    unrecorded = [c for c in chunks if c not in covered]
    return {"chunks": len(chunks), "unrecorded": unrecorded, "pass": bool(chunks) and not unrecorded}


def step_attest(b: dict, st: State):
    """09's attestations (agreed with 0d): per staged set (keyed by its staged sha), one entry per check, each `pass` computed here
    from the gate records, its evidence file (which must contain the staged sha string) and this script @ the run commit."""
    head = git("rev-parse", "HEAD")
    checker, utc = f"{REL}/chain.py@{head}", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    att = {}

    def add(ssha: str, check: str, ok: bool, covers: dict, ev: Path | None, extra: list | None = None, why: dict | None = None):
        has = ev is not None and ev.exists() and ssha in ev.read_text()
        e = {"check": check, "pass": bool(ok) and has, "covers": {"staged_sha": ssha, **covers}, "evidence": evidence(ev) if ev is not None and ev.exists() else None,
             "checker": checker, "computed_utc": utc}
        if extra:
            e["evidence_extra"] = [evidence(x) for x in extra if x.exists()]
        if why:
            e["detail"] = why | {"evidence_contains_staged_sha": has}
        att.setdefault(ssha, []).append(e)

    for t in b.get("targets", []):
        tv = targets_for(b)[t]
        man = json.loads((HERE / "config" / tv["manifest"]).read_text())
        root, suffix = b["extract"][t], b["stage"][t]["suffix"]
        cov0 = {"manifest": f"{REL}/config/{tv['manifest']}@{b['commit_targets']}", "extraction_root": root, "target": t}
        arms = [a for a in b["stage"][t]["epochs"] if a != "repro-c"]
        if not arms:  # a target whose every run was excluded at staging (only repro-c staged): no staged data set to attest
            continue
        if not b.setdefault("stats", {}).get(t):  # the staging code's token counts per staged set (0d's condition on repro-c)
            before = set((HERE / "results").glob(f"staged_stats_{t}_20*.json"))
            reserve(st, b, f"stats {t}", CPU_WORST_USD["stats"])
            modal_run([f"{REL}/stage_chain.py", "--target", t, "--stats-arms", ",".join(arms), *(["--suffix", suffix] if suffix else [])], st, f"stats_{t}", b=b)
            b["stats"][t] = sorted(set((HERE / "results").glob(f"staged_stats_{t}_20*.json")) - before)[-1].name
            st.save()
        stats_p = HERE / "results" / b["stats"][t]
        stats = {x["arm"]: x for x in json.loads(stats_p.read_text())["sets"]}
        recs = sorted((p for p in (HERE / "results").glob(f"extract_chain_{root}_a*_*.json") if valid_record(p) is not None),
                      key=lambda p: (int(re.match(r"_a(\d+)_", p.name[len(f"extract_chain_{root}"):]).group(1)), p.name))
        shards = [(p, s) for p in recs for s in json.loads(p.read_text())["shards"]]
        last_ok = bool(recs) and all("error" not in s for s in json.loads(recs[-1].read_text())["shards"])
        h200 = bool(shards) and all("error" in s or (s.get("gpu_name") == "NVIDIA H200" and s.get("gpu_count") == 1) for _, s in shards)
        wcov = writer_coverage(root, recs)  # the writers of the actual chunks, not a later attempt that wrote nothing
        cmp_p = HERE / "results" / b["compare"][t]
        cmp = json.loads(cmp_p.read_text())
        cmp_ok = not cmp["failing_shards"] and all(cmp["shards"].values()) and cmp["coverage"]["every_shard_one_of_each"] and cmp["coverage"]["shards_seen"] == cmp["coverage"]["shards_expected"] and cmp["passing"] == cmp["epochs"]
        stage_p = HERE / "results" / b["stage"][t]["record"]
        srec = json.loads(stage_p.read_text())
        fgate = srec[next(k for k in srec if k != "git")]["assembled"]["format_gate"]
        for arm in arms:
            ssha = b["stage"][t]["staged_inputs_sha256"][f"{arm}.jsonl"]
            rows = [f for f in man["files"] if f["arm_dir"] == arm and not f.get("reference_sha256_pin")]  # index donors are never staged
            world = rows[0]["world"] if rows else None
            cov = cov0 | {"arm": arm, "world": world}
            sx = stats.get(arm, {})
            long_ok = sx.get("staged_sha") == ssha and sx.get("n_epochs_over_32768") == 0
            add(ssha, "longest_epoch_h200_gate", h200 and last_ok and cmp_ok and long_ok and wcov["pass"], cov, recs[-1] if recs else None, [cmp_p, stats_p, *recs[:-1]],
                {"every_shard_h200": h200, "last_attempt_no_errors": last_ok, "repro_c_compare_pass": cmp_ok, "max_staged_tokens": sx.get("max_staged_tokens"),
                 "chunk_writers_recorded_h200": wcov["pass"], "volume_chunks": wcov["chunks"], "unrecorded_chunks": wcov["unrecorded"][:20],
                 "n_epochs_over_32768": sx.get("n_epochs_over_32768"), "reference": "repro-c: batch C's 32,768-token longest staged epoch + one seeded, per shard"})
            fc_p = HERE / "results" / b["format"][f"{t}:{arm}"]
            fc = json.loads(fc_p.read_text())
            add(ssha, "corrected_render", fc["passes"] and fc["staged_sha"] == ssha and fgate.get(arm, {}).get("passes") is True, cov, fc_p, [stage_p],
                {"format_check_passes": fc["passes"], "assemble_format_gate_passes": fgate.get(arm, {}).get("passes")})
            te_p = HERE / "results" / "hygiene" / f"toolerr_{b['pool']}_{b['id']}_{world}.json"
            te = json.loads(te_p.read_text()) if te_p.exists() else {}
            add(ssha, "tool_error_table", te.get("staged_sha") == ssha and te.get("totals", {}).get("runs") == len(rows), cov, te_p if te_p.exists() else None, None,
                {"rows": te.get("totals", {}).get("runs"), "manifest_rows": len(rows)})
    if b["kind"] in IL_KINDS and b.get("provenance"):  # pairwise approach provenance: each of this batch's run3 / sim2 staged sets
        pv_p = HERE / "results" / "hygiene" / b["provenance"]["file"]
        pv = json.loads(pv_p.read_text())
        for t in b.get("targets", []):
            w = targets_for(b)[t]["manifest_set"]
            ssha = b["stage"][t]["staged_inputs_sha256"].get(f"{'attack' if w == 'run3' else SIM_ARM}.jsonl")
            rows = [r for r in pv["rows"] if r["subject"] == w]
            if ssha and rows:
                add(ssha, "approach_provenance", all(r["pass"] for r in rows), {"world": w, "cells": sorted(r["cell"] for r in rows), "target": t,
                    "manifest": f"{REL}/config/{targets_for(b)[t]['manifest']}@{b['commit_targets']}", "extraction_root": b.get("extract", {}).get(t)}, pv_p, None,
                    {"rows_pass": sum(r["pass"] for r in rows), "rows": len(rows), "pairs": sum(len(r["pairs"]) for r in rows)})
    if b["kind"] in ("fu", "fus") and b.get("provenance"):  # approach provenance: this sim set, and the run1/run2 attack sets cell by cell
        pv_p = HERE / "results" / "hygiene" / b["provenance"]["file"]
        pv = json.loads(pv_p.read_text())
        mb = main_of(st, b)
        where = {}  # attack staged sha -> (batch, target)
        for x in ([b, *snaps_of(st, mb), mb] if mb is not None else [b]):  # an index-donor family (ug): its own staged set only
            for t in x.get("targets", []):
                s_ = x.get("stage", {}).get(t, {}).get("staged_inputs_sha256", {})
                for k in ("attack.jsonl", f"{sim_arm_of(b)}.jsonl"):
                    if s_.get(k):
                        where[s_[k]] = (x, t)
        by_sha = {}
        for c in pv["cells"]:
            for w, ssha in zip(("run1", "run2", sim_world_of(b)), c["staged_shas"]):
                if ssha:
                    by_sha.setdefault(ssha, {"world": w, "cells": []})["cells"].append(c)
        for ssha, g in by_sha.items():
            x, t = where.get(ssha, (None, None))
            cov = {"world": g["world"], "cells": sorted(c["cell"] for c in g["cells"])}
            if x is not None:
                cov |= {"manifest": f"{REL}/config/{targets_for(x)[t]['manifest']}@{x['commit_targets']}", "extraction_root": x.get("extract", {}).get(t), "target": t}
            add(ssha, "approach_provenance", all(c["pass"] for c in g["cells"]), cov, pv_p, None,
                {"cells_pass": sum(c["pass"] for c in g["cells"]), "cells": len(g["cells"])})
    name = f"attestations_{b['pool']}_{b['id']}{b.get('rev', '')}.json"  # a re-stage (rev): a new file; the first version stays
    out = HERE / "results" / "hygiene" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    reg = write_registry(st, b.get("stratum", "S_new"))
    reg_copy = HERE / "results" / "hygiene" / reg.name

    def prepare():  # the tracked files, inside the commit's GIT section
        out.write_text(json.dumps({"batch": b["id"], "pool": b["pool"], "stratum": b.get("stratum", "S_new"), **({"sealed": EP2_SEALED} if b["kind"] == "ep2" else {}),
                                   "attestations": att}, indent=1) + "\n")
        reg_copy.write_bytes(reg.read_bytes())

    b["attest_commit"] = commit([f"{REL}/results/hygiene/{name}", f"{REL}/results/hygiene/{reg.name}", *[f"{REL}/results/{v}" for v in b.get("stats", {}).values()]],
                                f"Overnight chain: {b['id']} attestations", st, prepare=prepare)
    fails = {s: [e["check"] for e in es if not e["pass"]] for s, es in att.items()}
    (routing_dir(st, b) / f"{b['id']}_attest.json").write_text(json.dumps({"batch": b["id"], "commit": b["attest_commit"], "path": f"{REL}/results/hygiene/{name}",
                                                                        "sets": len(att), "failing": {s: f for s, f in fails.items() if f}}, indent=1))
    return True


STEP_FN = {"ended": step_ended, "boxfiles": step_boxfiles, "relay": step_relay, "commit_targets": step_commit_targets, "stage": step_stage,
           "format": step_format, "repro": step_repro, "extract": step_extract, "compare": step_compare, "labels": step_labels,
           "record": step_record, "ready": step_ready, "hygiene": step_hygiene, "provenance": step_provenance, "attest": step_attest}


def recover_known_stops(st: State):
    """Two infrastructure-caused stops of 10-14 ~23:20, re-opened with a log line (never a data or gate failure):
    (1) a main remainder STOPPED at relay because a sibling log the box still appends to never matched its hash (relay now takes
    the copy as of the relay); (2) a follow-up STOPPED at stage because its main's donors were never relayed (follow-ups now wait
    for a STOPPED main): its failed try is abandoned and a fresh try stages under a new write-once name."""
    for b in st.s["batches"].values():
        why = b.get("reason") or ""
        if b["status"] != "STOPPED":
            continue
        if b.get("stopped_at") == "relay" and "laptop copy differs from the box manifest" in why and any(why.split(":")[0].endswith(x) for x in LIVE_LOGS):
            b["status"], b["relay_mismatch"] = "waiting", 0
        elif (b.get("stopped_at") == "stage" and "missing from the Volume or differs from the upload manifest" in why and b["kind"] in ("fu", "fus")
              and not main_relayed(st, b)):  # only while its main's donors are verifiably not relayed
            b["status"] = "waiting"
            for t in b.get("targets", []):
                if t not in b.get("stage", {}) and b.get("stage_tries", {}).get(t):
                    b.setdefault("stage_abandoned", {})[t] = b["stage_tries"][t]
        else:
            continue
        b["recovered_stop"] = {"stopped_at": b.pop("stopped_at"), "reason": b.pop("reason")[-300:], "at": now()}
        st.s.setdefault("final_combined", {}).pop(b["pool"], None)  # its pool's combined finals are recomputed
        st.log(f"{b['id']}: re-opened an infrastructure stop at {b['recovered_stop']['stopped_at']} ({b['recovered_stop']['reason'][-120:]})")
    st.save()


def write_exclusions(st: State, busy: frozenset = frozenset()):
    """A rerun follow-up's `exclude_first`: the first follow-up's runs in the given launcher state are EXCLUDED records (key
    sim|<cell>, manifest key, launch, cause) in results/hygiene/exclusions_<pool>_<first batch>.json. Idempotent and reconciled
    on every pass: the file is written once, committed until a commit is recorded, and whenever its sha256 differs from the one
    the first batch last applied, that batch's provenance and attestations are re-queued (new versions, same names) and the
    pool's combined finals are invalidated."""
    for b in [x for x in st.s["batches"].values() if x["kind"] == "fu" and x.get("exclude_first")]:
        ef = b["exclude_first"]
        first = st.s["batches"][ef["batch"]]
        if "commit_targets" not in first["done"] or first["id"] in busy:  # never while a worker runs that batch's steps
            continue
        p = HERE / "results" / "hygiene" / f"exclusions_{first['pool']}_{first['id']}.json"
        if not p.exists():
            tgt = targets_for(first)[next(iter(targets_for(first)))]
            man = json.loads((HERE / "config" / tgt["manifest"]).read_text())
            w_ = sim_world_of(first)
            rows = [{"key": f"{w_}|{f['task']}", "cell": f["task"], "manifest_key": f["path"], "sha256": f["sha256"], "launch": first["run"], "state": f.get("state"),
                     "excluded": True, "exclusion_cause": ef["cause"], "rerun_in": b["run"]} for f in man["files"] if f["world"] == w_ and f.get("state") == ef["state"]]
            doc = {"batch": first["id"], "launch": first["run"], "manifest": f"{REL}/config/{tgt['manifest']}@{first.get('commit_targets')}",
                   "selected_on": {"state": ef["state"]}, "excluded": rows}
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".part")
            tmp.write_text(json.dumps(doc, indent=1) + "\n")
            tmp.rename(p)
            st.log(f"{first['id']}: {len(rows)} runs recorded as EXCLUDED ({p.name})")
        if not first.get("exclusions_commit"):
            first["exclusions_commit"] = commit([f"{REL}/results/hygiene/{p.name}"], f"Overnight chain: {first['id']} EXCLUDED records ({ef['cause'][:40]})", st)
            st.save()
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if first.get("exclusions_applied") != h:  # recompute its provenance and attestations without the excluded cells
            first["provenance_supersede"] = True  # whether or not its provenance step completed (a written-but-uncommitted file too)
            if "provenance" in first["done"] or "attest" in first["done"]:
                first["done"] = [x for x in first["done"] if x not in ("provenance", "attest")]
                first["status"] = "hygiene"
            first["exclusions_applied"] = h
            st.s.setdefault("final_combined", {}).pop(first["pool"], None)
            st.save()


def orphan_check(st: State) -> dict:
    """After a restart that stopped the previous process between passes (RULE.md, 10-14 ~20:20, d9's safeguards): log anything the
    old process may have left. Reads only, except a stale git index lock with no git process running."""
    out = {"at": now(), "found": []}
    gitdir = Path(git("rev-parse", "--git-dir"))
    gitdir = gitdir if gitdir.is_absolute() else REPO / gitdir
    ps_ = run(["ps", "-axo", "pid=,ppid=,command="], timeout=60)
    procs = ps_.stdout.decode().splitlines()
    for lock in sorted(gitdir.glob("*.lock")):
        busy = [x for x in procs if re.search(r"(^|/)git( |$)", x.split(None, 2)[-1])]
        if ps_.returncode != 0 or not procs:  # no reliable process list: never remove a lock
            out["found"].append({"git_lock": str(lock), "action": "left (process listing failed)"})
        elif busy:
            out["found"].append({"git_lock": str(lock), "action": "left (a git process is running)", "git_processes": len(busy)})
        else:
            lock.unlink()
            out["found"].append({"git_lock": str(lock), "action": "removed (no git process running)"})
    def ours(cmd: str) -> bool:  # a `modal run [--detach] <REL>/<script>.py` client of THIS chain (not another session's job that names our files)
        toks = cmd.split()
        i = next((k for k, t in enumerate(toks) if t == "run" and k and toks[k - 1].endswith("modal")), None)
        # the script is the first token after `run` that names a Python file (option values such as `--env main` never do)
        script = next((t for t in toks[i + 1:] if t.split("::")[0].endswith(".py")), "") if i is not None else ""
        return script.startswith(f"{REL}/")

    def clients() -> list:  # Modal clients of the old process's blocking steps (detached extractions are polled, not waited for)
        r = run(["ps", "-axo", "pid=,ppid=,command="], timeout=60)
        if r.returncode != 0:
            raise RuntimeError("process listing failed")
        return [x.strip() for x in r.stdout.decode().splitlines() if ours(x) and "extract_chain.py" not in x]

    for x in procs:
        if ours(x):
            out["found"].append({"modal_client": x.strip()[:300]})
    t0 = time.time()
    while (left := clients()) and time.time() - t0 < 3600:  # a surviving blocking-step client finishes before any worker starts
        st.log(f"ORPHAN CHECK: waiting for {len(left)} surviving Modal client(s) of the old process to finish")
        time.sleep(30)
    if left:
        raise SystemExit(f"{len(left)} Modal clients of the old process still running after 60 min; not starting")
    for b in st.s["batches"].values():
        for t, job in (b.get("extract_job") or {}).items():
            out["found"].append({"extraction": f"{b['id']} {t} a{job['att']}", "pid": job["pid"],
                                 "alive": job["pid"] is not None and pid_alive(job["pid"]),
                                 "action": "polled; on exit settled and recovered from complete.json, else a new attempt (never overwriting)"})
        nxt = next((x for x in STEPS if x not in b["done"]), None)
        if b["status"] not in TERMINAL and nxt not in (None, "ended"):
            out["found"].append({"batch": b["id"], "next_step": nxt, "action": "rerun from state (stage: write-once _try<n>; records timestamped, open('x'))"})
    out["relay"] = ("every relay destination is re-hashed on the Volume before it is skipped; a partial upload hashes differently, is "
                    "re-uploaded (force=True) and re-verified before any laptop copy is deleted")
    (st.d / f"orphan_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(out, indent=1))
    for f in out["found"]:
        st.log(f"ORPHAN CHECK: {json.dumps(f)[:400]}")
    st.log(f"ORPHAN CHECK: {len(out['found'])} items; relay: {out['relay']}")
    return out


def main(state_dir: str, wait_pid: int | None = None):
    with locked():  # the main thread holds STATE except while it sleeps; workers take it in turn
        _main(state_dir, wait_pid)


def _main(state_dir: str, wait_pid: int | None = None):
    if wait_pid:  # started under caffeinate BEFORE the old process is stopped: touch nothing until it has exited
        t0 = time.time()
        while pid_alive(wait_pid):
            if time.time() - t0 > 3600:
                raise SystemExit(f"pid {wait_pid} still alive after 60 min; not starting")
            time.sleep(1)
    st = State(Path(state_dir))
    if wait_pid:
        orphan_check(st)
    if "gpu_actual_usd" in st.s or "gpu_inflight" in st.s:  # an earlier version's accounting: refuse rather than lose it
        raise SystemExit("legacy state (gpu_actual_usd / gpu_inflight): start with a fresh state directory")
    launches = [{"q": q, "box": bx, "run": rn, "pool": "su311"} for q, (bx, rn) in BOXES.items()]
    cfg = json.loads(BATCH_CFG.read_text()) if BATCH_CFG.exists() else {}
    launches += cfg.get("launches", [])  # e.g. the 360 new mains: {q, box, run, pool}
    st.s["pools"] = POOLS | {k: float(v) for k, v in cfg.get("pools", {}).items()}
    for L in launches:
        bid = L["q"] if L["pool"] == "su311" else f"{L['pool']}-{L['q']}"
        base = {"q": L["q"], "pool": L["pool"], "box": L["box"], "status": "waiting", "root": L.get("root", "phase2/runs"),
                "stratum": L.get("stratum", "S_new")}
        fo = {"final_only": True} if L.get("final_only") else {}  # a launch that ended before its family existed (S4's 4r): no snapshots
        if bid not in st.s["batches"]:
            st.s.setdefault("final_combined", {}).pop(L["pool"], None)  # a new family: the pool's combined finals are recomputed
        st.s["batches"].setdefault(bid, base | fo | {"id": bid, "kind": "main", "run": L["run"], "done": []})  # each batch its own list
        if "commit_targets" not in st.s["batches"][bid]["done"] and L.get("exclude_keys"):  # keys never staged (e.g. S_paper493's extra t619 honest_b)
            st.s["batches"][bid]["exclude_keys"] = L["exclude_keys"]
        if "commit_targets" not in st.s["batches"][bid]["done"] and L.get("index_donors"):  # S_paper493 real arms: pinned reference indexes
            st.s["batches"][bid]["index_donors"] = L["index_donors"]
        for x in [st.s["batches"][bid], *snaps_of(st, st.s["batches"][bid])]:  # 09's admission rule: the config's current values before commit_targets,
            if "commit_targets" not in x["done"]:  # for the main and every one of its snapshots (ms) already in the state
                for k_ in ("sample_error_rule", "sample_error_structural"):
                    if k_ in L:
                        x[k_] = L[k_]
                    else:
                        x.pop(k_, None)
        if L.get("first_try") and not st.s["batches"][bid].get("stage"):  # S_paper493 v2 (09): folders <set>_v1_try<n> mean "relaunch n superseding
            # the earlier launch wholesale" (not a staging retry): the first staging of each target is try n, under its write-once _try<n> name
            n0 = int(L["first_try"])
            for t in targets_for(st.s["batches"][bid]):
                st.s["batches"][bid].setdefault("stage_tries", {}).setdefault(t, n0 - 1)
                st.s["batches"][bid].setdefault("stage_abandoned", {}).setdefault(t, n0 - 1)  # no earlier try to re-attach to
        if not L.get("no_followup"):  # a real-arms-only launch (S_paper493) has no simulated follow-up
            st.s["batches"].setdefault(f"{bid}fu", base | fo | {"id": f"{bid}fu", "kind": "fu", "main_run": L["run"], "run": L.get("fu_run"), "done": []})
    stage_maps = json.loads((HERE / "config" / "il_stage_maps.json").read_text()) if (HERE / "config" / "il_stage_maps.json").exists() else {}
    for L in cfg.get("il_launches", []):  # interleaved launches (run3 + sim2 in one launch; donors from the main): their own families
        smap = {c: stg for stg, v in stage_maps.get(L["stage_key"], {}).items() for c in v["cells"]}
        if L["id"] not in st.s["batches"]:
            st.s.setdefault("final_combined", {}).pop(L["pool"], None)
        st.s["batches"].setdefault(L["id"], {"id": L["id"], "kind": "il", "q": L["q"], "pool": L["pool"], "box": L["box"], "status": "waiting", "done": [],
                                             "root": L.get("root", "phase2/runs"), "stratum": L["stratum"], "main_run": L["main_run"], "main_batch": L.get("main_batch"),
                                             "run": L["run"], "family": L["id"], "name_prefix": L["name_prefix"], "label": L["label"],
                                             **{k: L[k] for k in ("donor_index", "donor_launches", "worlds", "only_cells", "target_tag", "run_table_suffix") if k in L},
                                             "final_only": bool(L.get("final_only"))})
        if L.get("first_try") and not st.s["batches"][L["id"]].get("stage"):  # a staging retry family: its first staging is try n (<set>_v1_try<n>)
            for t in targets_for(st.s["batches"][L["id"]]):
                st.s["batches"][L["id"]].setdefault("stage_tries", {}).setdefault(t, int(L["first_try"]) - 1)
                st.s["batches"][L["id"]].setdefault("stage_abandoned", {}).setdefault(t, int(L["first_try"]) - 1)
        st.s["batches"][L["id"]]["stage_of"] = smap  # from the IL branch's committed stage task files (config/il_stage_maps.json)
        if "commit_targets" not in st.s["batches"][L["id"]]["done"]:  # option (b): read at commit_targets, so a later edit applies only before it
            for k in ("donors_from_main", "donor_manifests", "donor_cells_only"):
                if k in L:
                    st.s["batches"][L["id"]][k] = L[k]
        for x in [st.s["batches"][L["id"]], *snaps_of(st, st.s["batches"][L["id"]])]:  # 09's sample_error rule: the config's current values,
            if "commit_targets" not in x["done"]:  # for the family and each of its snapshots, before commit_targets
                for k_ in ("sample_error_rule", "sample_error_structural"):
                    if k_ in L:
                        x[k_] = L[k_]
                    else:
                        x.pop(k_, None)
        for x in snaps_of(st, st.s["batches"][L["id"]]):
            x["stage_of"] = smap
    for E in cfg.get("ep2_launches", []):  # epoch-2 real honest launches (09 Append 38): final only, from their committed key lists
        if E["id"] not in st.s["batches"]:
            st.s.setdefault("final_combined", {}).pop("ep2", None)
        st.s["batches"].setdefault(E["id"], {"id": E["id"], "kind": "ep2", "q": E["q"], "pool": "ep2", "box": E.get("box"), "status": "waiting", "done": [],
                                             "root": "phase2/runs_ep2", "stratum": "ep2", "main_run": None, "run": E["run"], "family": E["id"], "name": E["name"],
                                             "keylist": E["keylist"], "label": E["label"], "final_only": True})
    check_group_families(cfg.get("followups", []))
    for F in cfg.get("followups", []):  # an extra follow-up launch of a quarter (the Q2 rerun): its own family, run preset
        if F.get("sim_arm", SIM_ARM) != SIM_ARM and F["pool"] in CORE_POOLS:
            raise SystemExit(f"{F['id']}: a follow-up with simulated arm {F['sim_arm']} needs its own pool (not {F['pool']})")
        if F.get("rlr_of"):  # 09 @ 91485ce80: <pool>_<q>_sim_rlr_v1, final only, a root of its parent batch (same pool, quarter token, arm and world)
            P = next((x for x in cfg.get("followups", []) if x["id"] == F["rlr_of"]), None)
            if (P is None or not RLR_TAG.fullmatch(F.get("sim_tag") or "") or not F.get("final_only")
                    or any(F.get(k) != P.get(k) for k in ("pool", "qtok", "stratum", "sim_arm", "sim_world", "main_batch"))):
                raise SystemExit(f"{F['id']}: a replacement family needs sim_tag sim_rlr or sim_rlr<k> (k >= 2), final_only and its parent's pool, qtok, stratum, arm, world and main batch")
            K = F.get("rlr_keys")
            if not isinstance(K, dict) or not K or any(set(v) != {"class", "ruling", "root_check"} or v["class"] not in ("infrastructure", "unknown") for v in K.values()):
                raise SystemExit(f"{F['id']}: rlr_keys must map each ruled cell to its class (infrastructure|unknown), ruling and root_check")
        if F["id"] not in st.s["batches"]:
            st.s.setdefault("final_combined", {}).pop(F["pool"], None)  # a new family: the pool's combined finals are recomputed
        st.s["batches"].setdefault(F["id"], {"id": F["id"], "kind": "fu", "q": F["q"], "pool": F["pool"], "box": F["box"], "status": "waiting",
                                             "root": F.get("root", "phase2/runs"), "stratum": F.get("stratum", "S_new"), "main_run": F["main_run"],
                                             "run": F["run"], "done": [], "family": F["id"], "qtok": F["qtok"], "label": F["label"],
                                             "rerun_of": F.get("rerun_of"), "exclude_first": F.get("exclude_first"), "rerun_reason": F.get("rerun_reason"),
                                             "final_only": bool(F.get("final_only")),
                                             **{k: F[k] for k in ("sim_arm", "sim_world", "sim_tag", "main_batch", "donor_index", "donor_launches", "snap_at_once",
                                                                  "run_table_external", "comparability_map", "provenance_pending", "provenance_mode", "stage_donors_from_rows", "rlr_of", "rlr_keys") if k in F}})
        for x in [st.s["batches"][F["id"]], *snaps_of(st, st.s["batches"][F["id"]])]:  # provenance inputs: the config's current value until the batch's provenance ran
            if "provenance" not in x["done"] and any(k_ in F for k_ in ("comparability_map", "provenance_pending", "provenance_mode")):  # (a map committed after a family started)
                for k_ in ("comparability_map", "provenance_pending", "provenance_mode"):
                    if k_ in F:
                        x[k_] = F[k_]
                    else:
                        x.pop(k_, None)
        for x in [st.s["batches"][F["id"]], *snaps_of(st, st.s["batches"][F["id"]])]:  # donor-only launches and 09's sample_error rule:
            if "commit_targets" not in x["done"]:  # the config's current values before commit_targets
                for k_ in ("stage_donors_from_rows", "sample_error_rule", "sample_error_structural"):
                    if k_ in F:
                        x[k_] = F[k_]
                    else:
                        x.pop(k_, None)
        if "rlr_keys" in F and "hygiene" not in st.s["batches"][F["id"]]["done"]:  # the ruled keys: the config's current pins until the table is written
            st.s["batches"][F["id"]]["rlr_keys"] = F["rlr_keys"]
        for k_ in ("snap_at_once", "snap_s"):  # scheduling options (not identity): a family already in the state takes the config's current value
            if k_ in F:
                st.s["batches"][F["id"]][k_] = bool(F[k_]) if k_ == "snap_at_once" else int(F[k_])
        if F.get("group_cells"):  # one group of a shared sim launch: its cells, for the family and each snapshot, before its box read
            gc = group_cells(F)
            for x in [st.s["batches"][F["id"]], *snaps_of(st, st.s["batches"][F["id"]])]:
                if "boxfiles" not in x["done"]:
                    x["only_cells"], x["group"] = gc, F["group_cells"]["group"]
    for b in st.s["batches"].values():  # a main batch whose box files an earlier version read (no snapshots then): every roster cell
        if b["kind"] == "main" and "boxfiles" in b["done"] and "members" not in b:
            b["members"] = json.loads((st.d / f"boxfiles_{b['id']}.json").read_text())["facts"]["roster"]
    bad = ambiguous_names(list(st.s["batches"].values()), set(st.s["pools"]))
    if bad:
        raise SystemExit(f"batch names that spell a longer pool name (combine_final's guard would misfile them): {bad}")
    st.save()
    recover_known_stops(st)
    st.log(f"chain start; git {git('rev-parse', 'HEAD')[:10]}")

    def advance(b: dict, steps: list, one: bool) -> bool:
        """Run b's next pending steps among `steps` (only one if `one`); True if any step completed."""
        moved = False
        for step in STEPS:
            if step in b["done"]:
                continue
            if DRAIN.is_set():  # draining for a restart: no new step
                return moved
            if step not in steps:
                return moved
            try:
                r = STEP_FN[step](b, st)
                if step == "ended" and not r:
                    return moved  # not ended yet; look again next pass
                b["done"].append(step)
                b["status"] = "ready" if step == "ready" else step
                if step == "boxfiles" and b.get("empty"):  # a snapshot with no newly finished run: nothing to do
                    b["status"] = "EMPTY"
                    st.log(f"{b['id']}: empty snapshot")
                    st.save()
                    return moved
                st.log(f"{b['id']}: {step} done")
                st.save()
                moved = True
                if one:
                    return moved
            except Wait as e:
                b["status"] = f"waiting: {e}"
                st.save()
                return moved
            except Gate as e:
                b["status"], b["stopped_at"], b["reason"] = "STOPPED", step, str(e)[-1500:]
                st.log(f"{b['id']}: GATE STOP at {step}: {str(e)[-400:]}")
                st.save()
                return moved
            except Exception as e:  # a failed read or call: retried on the next pass, never acted on
                b["retries"] = b.get("retries", 0) + 1
                st.log(f"{b['id']}: {step} will retry: {type(e).__name__}: {str(e)[-300:]} | {traceback.format_exc(limit=2)[-300:]}")
                st.save()
                return moved
        return moved

    def work(b: dict) -> bool:
        with locked():
            return advance(b, STEPS, one=False)

    running, next_t = {}, {}
    pool_ = ThreadPoolExecutor(max_workers=24)
    if threading.current_thread() is threading.main_thread():  # (a test may run main() in a thread)
        signal.signal(signal.SIGTERM, lambda *_: DRAIN.set())
    while True:
        terminal = TERMINAL
        if DRAIN.is_set():
            if not running:
                st.save()
                st.log("drained on SIGTERM: no step in progress; exiting")
                return
            st.log(f"draining: waiting for {len(running)} batch step(s) in progress")
            for bid, fut in list(running.items()):
                if fut.done():
                    running.pop(bid)
            with unlocked():
                time.sleep(5)
            continue
        try:
            write_exclusions(st, frozenset(running))
        except Exception as e:  # a failed commit or read: retried next pass
            st.log(f"exclusions will retry: {type(e).__name__}: {str(e)[-300:]}")
        for pool in st.s["pools"]:  # each pool's final files once all of its batches are terminal
            mine = [x for x in st.s["batches"].values() if x["pool"] == pool]
            if mine and all(x["status"] in terminal for x in mine) and not st.s.setdefault("final_combined", {}).get(pool):
                try:
                    combine_final(st, pool)
                    st.s["final_combined"][pool] = True
                    st.save()
                except Exception as e:  # a failed push or read: retried next pass
                    st.log(f"final combine ({pool}) will retry: {type(e).__name__}: {str(e)[-300:]}")
        # snapshots: a running follow-up whose main launch is relayed gets a new snapshot batch about every SNAP_S
        for fu in [x for x in list(st.s["batches"].values()) if x["kind"] in ("fu", "il")]:
            if not fu.get("run") or fu.get("final_only") or not main_relayed(st, fu) or "ended" in fu["done"] or fu.get("finished_seen_t"):
                continue
            snaps = snaps_of(st, fu)
            at_once = fu["kind"] == "il" or fu.get("snap_at_once")  # an IL launch, or a family configured so (the sim-honest launches; d9 10-15 ~11:55): snapshot 1 at once
            last = max([x["created_t"] for x in snaps] + ([] if at_once else [fu.setdefault("found_t", time.time())]), default=0)
            if time.time() - last >= (fu.get("snap_s") or SNAP_S) and all("boxfiles" in x["done"] or x["status"] == "STOPPED" for x in snaps):  # snap_s: a family's own cadence
                k = len(snaps) + 1
                sid = f"{fu['id']}_s{k}"
                st.s["batches"][sid] = {"id": sid, "kind": SNAP_OF[fu["kind"]], "q": fu["q"], "pool": fu["pool"], "box": fu["box"], "main_run": fu["main_run"],
                                        "run": fu["run"], "snap": k, "created_t": time.time(), "done": [], "status": "waiting",
                                        "root": fu.get("root", "phase2/runs"), "stratum": fu.get("stratum", "S_new"), "family": fam(fu),
                                        **{k_: fu[k_] for k_ in ("qtok", "label", "name_prefix", "main_batch", "stage_of", "donor_index", "donor_launches", "donors_from_main", "donor_manifests",
                                                                "sim_arm", "sim_world", "sim_tag", "run_table_external", "comparability_map", "provenance_pending", "provenance_mode", "stage_donors_from_rows",
                                                                "sample_error_rule", "sample_error_structural", "worlds", "only_cells", "group") if k_ in fu}}
                st.log(f"{sid}: snapshot {k} of the {fu['q']} follow-up {fu['run']} created")
                st.save()
        # main snapshots: a running main launch gets its first snapshot at once, then one about every SNAP_S, until it is seen finished
        for mb in [x for x in list(st.s["batches"].values()) if x["kind"] == "main"]:
            if "ended" in mb["done"] or mb.get("finished_seen_t") or mb["status"] == "STOPPED" or mb.get("final_only"):
                continue
            snaps = snaps_of(st, mb)
            if time.time() - max([x["created_t"] for x in snaps], default=0) >= SNAP_S and all("boxfiles" in x["done"] or x["status"] == "STOPPED" for x in snaps):
                k = len(snaps) + 1
                sid = f"{mb['id']}_m{k}"
                st.s["batches"][sid] = {"id": sid, "kind": "ms", "q": mb["q"], "pool": mb["pool"], "box": mb["box"], "run": mb["run"], "snap": k,
                                        "created_t": time.time(), "done": [], "status": "waiting", "root": mb.get("root", "phase2/runs"),
                                        "stratum": mb.get("stratum", "S_new"),
                                        **{k_: mb[k_] for k_ in ("sample_error_rule", "sample_error_structural", "index_donors", "exclude_keys") if k_ in mb}}
                st.log(f"{sid}: snapshot {k} of the {mb['q']} main {mb['run']} created")
                st.save()
        live = [b for b in st.s["batches"].values() if b["status"] not in terminal]
        if not live and not running and all(st.s.get("final_combined", {}).get(pool) for pool in {x["pool"] for x in st.s["batches"].values()}):
            st.log("all batches done, empty or stopped; chain ends")
            return
        for bid, fut in list(running.items()):  # collect finished workers
            if fut.done():
                running.pop(bid)
                moved = False
                try:
                    moved = fut.result()
                except Exception as e:  # advance() catches step errors; anything here is a worker bug: logged, retried next pass
                    st.log(f"{bid}: worker error {type(e).__name__}: {str(e)[-300:]}")
                next_t[bid] = time.time() + (5 if moved else POLL_S)  # each batch polls its box or job at most every POLL_S when idle
        for b in live:  # one worker per batch: it runs the batch's steps back to back until it waits, stops or ends
            if b["id"] in running or time.time() < next_t.get(b["id"], 0):
                continue
            if b["kind"] in FU_KINDS and not main_relayed(st, b):
                continue  # a follow-up's donors come from its main launch's relayed files (the remainder's and every snapshot's)
            running[b["id"]] = pool_.submit(work, b)
        st.save()
        with unlocked():
            time.sleep(5)

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == "--wait-pid" else None)
