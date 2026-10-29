"""Start the pilot's vLLM servers (run_vllm_tunnel.py) from the laptop, each under its own app name, and record their
app IDs and tunnel URLs for run_pilot.py (--base-urls) and watch_and_stop.sh (the app IDs):

  uv run --no-project --python 3.12 --with modal==1.2.6 --with pyyaml python \\
      experiments/2026-10-07_phase2_attack_check/launch_servers.py --count 2 [--server-mode deployed] [--stand-in]

--server-mode ephemeral (the default, the tested path): one `modal run --detach` per server, as before.
--server-mode deployed: only if Modal's limit of 100 live ephemeral apps per workspace blocks a launch. Each server is
deployed as its own app (`modal deploy`, under a unique per-launch name: `serve` has max_containers=1, so two servers
can't share an app) and its `serve` function is spawned. A deployed app outlives this process: the server still ends
itself (idle exit, hard timeout), and watch_and_stop.sh stops every app at the end and alarms unless each is confirmed
stopped. --stand-in runs stand_in_server.py, a trivial CPU function, instead of vLLM, to dry-run either path for cents.

Writes results/servers_<stamp>_<tag>.json from the first app on: per server its name, mode, app ID, tunnel URL, API
base (the tunnel URL plus /v1, what run_pilot.py --base-urls takes) and spawned call ID (deployed). If anything fails
(Ctrl+C included), it stops every app the launch created, confirms each is stopped (an ALARM otherwise), and records
that.
"""

import argparse
import contextlib
import json
import os
import queue
import re
import secrets
import signal
import string
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from modal import exception as modal_exception

HERE = Path(__file__).resolve().parent
MODAL = ["uv", "run", "--no-project", "--python", "3.12", "--with", "modal==1.2.6", "--with", "pyyaml", "modal"]
# The URL prints once vLLM and the proxy answer, so allow vLLM's own startup allowance (servers.yaml) and a margin.
URL_WAIT_MIN = yaml.safe_load((HERE / "config" / "servers.yaml").read_text())["startup_wait_min"] + 10


def modal_json(*cmd: str) -> list[dict]:
    return json.loads(subprocess.run([*MODAL, *cmd, "--json"], capture_output=True, text=True, check=True,
                                     timeout=120).stdout)


def live_apps(name: str) -> list[str]:
    """Live apps whose name contains `name`: one server's full name, or a launch's tag (srv<tag>) for all of its."""
    return [a["App ID"] for a in modal_json("app", "list") if name in a["Description"] and a["State"] != "stopped"]


def containers_left(name: str, minutes: int = 10) -> list[str]:
    """Wait until no container runs for an app whose name contains `name`; returns the IDs still running at the end."""
    for _ in range(minutes * 2):
        if not (ids := [c["Container ID"] for c in modal_json("container", "list") if name in c["App Name"]]):
            return []
        time.sleep(30)
    return ids


def stop_and_confirm(name: str, quiet_polls: int = 4, max_polls: int = 20) -> list[str]:
    """Stop every live app with this name until none is listed on `quiet_polls` polls in a row, 15 s apart: an app a
    launch was still creating can appear after a first empty listing. Returns what is still not confirmed stopped
    (empty when all are)."""
    quiet = 0
    for _ in range(max_polls):
        ids = live_apps(name)
        quiet = 0 if ids else quiet + 1
        if quiet >= quiet_polls:
            return []
        for app_id in ids:
            subprocess.run([*MODAL, "app", "stop", app_id], capture_output=True, timeout=120, check=False)
        time.sleep(15)
    return live_apps(name) or [f"{name} kept reappearing for {max_polls} polls"]


def stop_or_report(name: str) -> list[str]:
    """stop_and_confirm(), but a failing CLI call (an error, a timeout) is reported as unconfirmed instead of raised, so
    the other apps are still stopped."""
    try:
        return stop_and_confirm(name)
    except BaseException as e:  # noqa: BLE001 (recorded as unconfirmed, and alarmed)
        return [f"unconfirmed: {e!r}"[:300]]


def containers_or_report(name: str) -> list[str]:
    try:
        return containers_left(name)
    except BaseException as e:  # noqa: BLE001 (recorded as unconfirmed, and alarmed)
        return [f"unconfirmed: {e!r}"[:300]]


def end_launcher(proc: subprocess.Popen) -> None:
    """End a `modal run` launcher with its process group (uv and the modal CLI under it). uv exits on its own only after
    its child, so a launcher that has exited needs nothing. A running one gets SIGTERM, then SIGKILL, sent to its group
    only while it is unreaped, so the group ID can't have been reused. macOS refuses (EPERM) to signal a group whose
    leader has just become a zombie: that means it exited, which is fine."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, sig)
        except (PermissionError, ProcessLookupError):
            if proc.poll() is not None:  # it exited meanwhile
                return
            raise
        try:
            proc.wait(timeout=60)
            return
        except subprocess.TimeoutExpired:
            continue


@contextlib.contextmanager
def sigint_deferred():
    """Ctrl+C waits until the block ends, so a launcher is never started without being registered for cleanup. (The
    launcher inherits the blocked SIGINT; it runs in its own session and is ended with SIGTERM/SIGKILL.)"""
    old = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old)


class NoAnswer(TimeoutError):
    """within()'s own timeout, told apart from a TimeoutError the called function itself raised."""


def within(seconds: float, fn):
    """fn() on a daemon thread, given at most `seconds`: a stalled Modal RPC can't hold off the deadline and cleanup."""
    done: queue.Queue = queue.Queue()

    def run() -> None:
        try:
            done.put((True, fn()))
        except BaseException as e:  # noqa: BLE001 (re-raised in the caller)
            done.put((False, e))

    threading.Thread(target=run, daemon=True).start()
    try:
        ok, value = done.get(timeout=max(1, seconds))
    except queue.Empty:
        raise NoAnswer(f"no answer within {seconds:.0f} s") from None
    if not ok:
        raise value
    return value


def wait_for_url(app_id: str, deadline: float, check) -> str:
    """The TUNNEL_URL line from the app's log, by `deadline`. The log stream is read on a thread (a silent stream can't
    block the deadline) and reopened if it ends; `check()` runs every few seconds and raises if a launch has failed."""
    while time.time() < deadline:
        lines: queue.Queue = queue.Queue()
        logs = subprocess.Popen([*MODAL, "app", "logs", app_id], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)
        threading.Thread(target=lambda q=lines, out=logs.stdout: [q.put(x) for x in out] + [q.put(None)],
                         daemon=True).start()
        try:
            while time.time() < deadline:
                check()
                try:
                    line = lines.get(timeout=5)
                except queue.Empty:
                    continue
                if line is None:  # the stream ended: reopen it
                    break
                if (m := re.search(r"TUNNEL_URL=(\S+)", line)) and time.time() < deadline:
                    return m.group(1)
        finally:
            logs.kill()
            logs.wait()
    raise TimeoutError(f"{app_id}: no TUNNEL_URL by the deadline")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=2)
    ap.add_argument("--server-mode", choices=["ephemeral", "deployed"], default="ephemeral")
    ap.add_argument("--stand-in", action="store_true", help="a trivial CPU function instead of vLLM (dry run)")
    ap.add_argument("--server-config", default="servers.yaml",
                    help="the server's config in config/ (servers_qwen3_32b.yaml for the Qwen3-32B arm)")
    ap.add_argument("--url-wait-min", type=float, default=URL_WAIT_MIN,
                    help="give up (and stop every app) if a server has no URL this long after the launch starts; "
                         "a capped check sets its own (default: servers.yaml's startup allowance plus 10)")
    args = ap.parse_args()
    assert (HERE / "config" / args.server_config).is_file(), f"no config/{args.server_config}"
    script = HERE / ("stand_in_server.py" if args.stand_in else "run_vllm_tunnel.py")
    # probe-shift_aswin-surya_sub-phase2_srv<6 hex><letter>_kill-2026-11-06: random per launch (two launches in one
    # minute can't collide), at most 64 characters
    tag = secrets.token_hex(3)
    names = [f"probe-shift_aswin-surya_sub-phase2_srv{tag}{string.ascii_lowercase[i]}_kill-2026-11-06"
             for i in range(args.count)]
    for name in names:
        assert not live_apps(name), f"{name} already exists"
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = HERE / "results" / f"servers_{stamp}_{tag}.json"  # the tag too: two launches in one second can't collide
    servers, launchers, calls = [], {}, {}  # calls: a deployed server's spawned FunctionCall, by app name
    polled = [0.0]  # when the deployed calls were last polled

    def save(status: str, **extra) -> None:  # written from the first app on, so a failure still leaves a record
        out.write_text(json.dumps({"status": status, "mode": args.server_mode, "stand_in": args.stand_in,
                                   "server_config": args.server_config, "servers": servers, **extra}, indent=1))

    def check() -> None:  # an ephemeral launch refused (e.g. by Modal's app limit) or failing fails the whole start
        for name, (proc, log) in launchers.items():
            if proc.poll() not in (None, 0):
                tail = Path(log).read_text()[-1500:]
                raise RuntimeError(f"{name}: `modal run` failed (exit {proc.returncode}). If it is Modal's "
                                   f"ephemeral-app limit, retry with --server-mode deployed.\n{tail}")
        # A deployed server's call that has finished, either way, ended before its URL (vLLM or CUDA failed at
        # startup): fail now rather than wait out the deadline while the other GPUs bill. Polled every 15 s.
        if calls and time.time() - polled[0] >= 15:
            polled[0] = time.time()
            for name, call in calls.items():
                try:
                    result = within(30, lambda c=call: c.get(timeout=0))
                except (modal_exception.FunctionTimeoutError, modal_exception.OutputExpiredError) as e:
                    raise RuntimeError(f"{name}: the deployed server ended before its URL: {e!r}"[:1500]) from e
                except TimeoutError:  # still running: modal 1.2.6 raises the builtin TimeoutError for an unfinished
                    continue  # call (_functions.py poll_function); NoAnswer (the poll got no answer) is one too
                except Exception as e:  # noqa: BLE001 (the server's own failure, re-raised as the launch's)
                    raise RuntimeError(f"{name}: the deployed server ended before its URL: {e!r}"[:1500]) from e
                raise RuntimeError(f"{name}: the deployed server returned before its URL: {result!r}"[:500])

    deadline = time.time() + args.url_wait_min * 60  # from the start: GPU time counts from the first app
    try:
        for name in names:
            env = os.environ | {"PHASE2_SERVER_APP": name, "PHASE2_SERVER_CONFIG": args.server_config}
            record = {"name": name, "mode": args.server_mode}
            servers.append(record)
            save("starting")
            if args.server_mode == "ephemeral":
                log = HERE / "results" / f"server_launch_{name}.log"
                with sigint_deferred():
                    launchers[name] = (subprocess.Popen([*MODAL, "run", "--detach", str(script)], env=env,
                                                        cwd=HERE.parents[1], stdout=open(log, "w"),
                                                        stderr=subprocess.STDOUT, start_new_session=True), str(log))
                record["launch_log"] = str(log.relative_to(HERE.parents[1]))  # repo-relative: no local paths
            else:
                subprocess.run([*MODAL, "deploy", str(script)], env=env, cwd=HERE.parents[1], check=True,
                               timeout=max(1, min(900, deadline - time.time())))  # earlier servers' GPUs are running
                import modal

                calls[name] = within(deadline - time.time(),
                                     lambda n=name: modal.Function.from_name(n, "serve").spawn())
                record["call_id"] = calls[name].object_id
            print(f"started {name} ({args.server_mode})", flush=True)
            if time.time() > deadline:
                raise TimeoutError(f"{name}: started after the deadline")
        for record in servers:
            while not (ids := live_apps(record["name"])):  # an ephemeral app appears once `modal run` creates it
                check()
                if time.time() > deadline:
                    raise TimeoutError(f"{record['name']}: no app by the deadline")
                time.sleep(10)
            assert len(ids) == 1, f"{record['name']}: {len(ids)} live apps"
            record["app_id"] = ids[0]
            save("starting")
            record["tunnel_url"] = wait_for_url(record["app_id"], deadline, check)
            record["base_url"] = record["tunnel_url"].rstrip("/") + "/v1"  # the server's OpenAI-compatible base
            print(f"{record['name']}: {record['app_id']} {record['base_url']}", flush=True)
        save("up")
    except BaseException as e:  # anything, Ctrl+C included: stop every app this launch created, and confirm
        # A second Ctrl+C must not cut the stops short: ignored until cleanup ends (the modal CLI calls inherit that).
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print(f"start failed ({e!r}); stopping the apps it created (Ctrl+C is ignored until that ends)", flush=True)
        alive = {}  # a launcher that won't end could still create its app later: no listing can rule that out
        for name, (proc, _) in launchers.items():  # a `modal run` still creating its app: end it before confirming
            try:
                end_launcher(proc)
            except OSError as err:
                print(f"{name}: could not end its launcher ({err!r})", flush=True)
            if proc.poll() is None:
                alive[f"launcher of {name}"] = [f"pid {proc.pid} still running"]
        left = alive | {s["name"]: stop_or_report(s["name"]) for s in servers}
        if any(left.values()):  # then every app with this launch's tag, and wait until none of them has a container
            print(f"not confirmed: {left}; stopping every app tagged srv{tag}", flush=True)
            left = alive | {f"srv{tag}": stop_or_report(f"srv{tag}"), "containers": containers_or_report(f"srv{tag}")}
        for s in servers:
            s["stopped_after_failure"] = not any(left.values())
        save(f"failed: {e!r}"[:500], unconfirmed=left)
        if any(left.values()):
            print(f"ALARM: not confirmed stopped, stop by hand: {left}", flush=True)
        raise
    for name, (proc, _) in launchers.items():  # detached: the apps keep running without these local clients
        try:
            end_launcher(proc)
        except OSError as err:  # the servers are up and recorded; only a local client may linger
            print(f"{name}: could not end its local `modal run` ({err!r}); the server is unaffected", flush=True)
    print("base URLs (run_pilot.py --base-urls):", ",".join(s["base_url"] for s in servers))
    print("app IDs (watch_and_stop.sh):", ",".join(s["app_id"] for s in servers))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
