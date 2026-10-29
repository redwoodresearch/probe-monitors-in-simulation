"""Serve Qwen3.5-27B with vLLM on one Modal H200, reachable through a Modal tunnel rather than a web endpoint.

The tunnel is a TLS passthrough to the container's port, so requests skip the web proxy behind which the pilot's long
runs lost model calls. Every request needs `Authorization: Bearer <key>` (vLLM reads VLLM_API_KEY from the pilot's Modal
Secret; the client reads the same key from rr-env). A tunnel has no scale-to-zero, so the function ends itself: after
`idle_exit_min` with no request running once one has been served, after `startup_idle_min` if none ever arrives, and
at `hard_timeout_h` regardless. The tunnel URL and per-minute request counts, generated tokens/s and GPU utilization go
to the app's log.

The tunnel reaches vLLM through request_log_proxy.py, which logs every chat request's prompt and completion token IDs
and sampled-token logprobs to the volume REQUEST_LOG (one file per server: start time and Modal call ID in its name,
committed every minute); vLLM itself
listens on loopback only.

  M="uv run --no-project --python 3.12 --with modal==1.2.6 --with pyyaml modal"
  $M run --detach experiments/2026-10-07_phase2_attack_check/run_vllm_tunnel.py      # prints TUNNEL_URL=... when up
  (or launch_servers.py, which starts several and records their app IDs and tunnel URLs; its --server-mode deployed
  deploys each server as its own app under a unique name, for when Modal's limit of 100 ephemeral apps blocks a run)
  $M app logs <app id>                                                                 # URL, request counts
  $M app stop <app id>                                                                 # stop early
"""

import http.client
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
# The app's name: launch_servers.py gives each server of a launch its own (deployed servers need distinct names).
APP = os.environ.get("PHASE2_SERVER_APP", "probe-shift_aswin-surya_sub-phase2_tunnel_kill-2026-11-06")
assert APP.startswith("probe-shift_aswin-surya_sub-phase2_") and len(APP) <= 64, APP
SECRET = "probe-shift_aswin-surya_sub-phase2_vllm-key_kill-2026-11-06"  # holds VLLM_API_KEY (run_setup_key.py, PR #50)
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
REQUEST_LOG = "probe-shift_aswin-surya_sub-phase2_request-log_kill-2026-11-06"
CONFIG_IN_IMAGE = "/root/servers.yaml"
# The server's config in config/: servers.yaml (Qwen3.5-27B, the pilot's policy), unless launch_servers.py
# --server-config names another (servers_qwen3_32b.yaml). The chosen file is baked into the image for the container.
CONFIG_FILE = os.environ.get("PHASE2_SERVER_CONFIG", "servers.yaml")
assert "/" not in CONFIG_FILE and CONFIG_FILE.endswith(".yaml"), CONFIG_FILE
PROXY_IN_IMAGE = "/root/request_log_proxy.py"
PORT = 8000  # the proxy, which the tunnel forwards
VLLM_PORT = 8001  # vLLM, on loopback


def _config() -> dict:
    import yaml

    return yaml.safe_load((HERE / "config" / CONFIG_FILE if modal.is_local() else Path(CONFIG_IN_IMAGE)).read_text())


CFG = _config()
# The weights' Volume and where it mounts. Default: HF_CACHE at /hf (an HF_HOME tree). A config may name another
# Volume whose root is a hub folder, mounted at /hf/hub (the Llama-3.3-70B check: the Apollo sweep's Llama cache).
WEIGHTS = CFG.get("hf_cache") or {"volume": HF_CACHE, "mount": "/hf"}
assert set(WEIGHTS) == {"volume", "mount"} and WEIGHTS["mount"] in ("/hf", "/hf/hub"), WEIGHTS
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(f"vllm=={CFG['vllm_version']}", "pyyaml")
    .uv_pip_install("tzdata==2025.2")  # zoneinfo's data for the PT progress lines; its own layer, so vLLM's stays cached
    .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
    .add_local_file(HERE / "config" / CONFIG_FILE, CONFIG_IN_IMAGE)
    .add_local_file(HERE / "request_log_proxy.py", PROXY_IN_IMAGE)
)
app = modal.App(APP)
log_volume = modal.Volume.from_name(REQUEST_LOG, create_if_missing=True)


def _metric(text: str, name: str) -> float:
    """Sum of one vLLM Prometheus metric over its label sets (0 when absent)."""
    return sum(float(m) for m in re.findall(rf"^{name}(?:{{[^}}]*}})? ([0-9.e+-]+)$", text, re.MULTILINE))


def _gpu() -> str:
    """GPU utilization and memory, for the progress line (utilization is checked in the first minutes of a run). Never
    raises: a monitoring failure must not end the server and every run with it."""
    query = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
    try:
        out = subprocess.run(query, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as e:
        return f"gpu=? ({e!r})"
    if out.returncode:
        return f"gpu=? (nvidia-smi exit {out.returncode})"
    try:
        util, used, total = (x.strip() for x in out.stdout.splitlines()[0].split(","))
        return f"gpu={util}% mem={int(used) / 1024:.0f}/{int(total) / 1024:.0f}GiB"
    except (IndexError, ValueError):
        return f"gpu=? (unparsed: {out.stdout[:80]!r})"


def _wait_for_models(url: str, headers: dict, procs: list, minutes: float, what: str) -> None:
    """Return once `url` answers 200. It is authenticated, so a wrong key fails here and not on the first run: any HTTP
    error status means the server is listening and something is wrong, so it ends the function at once (HTTPError is an
    OSError, so it is caught first). Only "not listening yet" is retried, for at most `minutes`."""
    deadline = time.time() + minutes * 60
    while True:
        assert all(p.poll() is None for p in procs), f"{what}: a process exited during startup"
        try:
            urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=5)
            return
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{what}: /v1/models answered HTTP {e.code}; check the key (Secret {SECRET})") from e
        except OSError:
            if time.time() > deadline:
                raise RuntimeError(f"{what}: not listening {minutes} min after launch") from None
            time.sleep(5)


@app.function(image=image, gpu=CFG["gpu"],
              volumes={WEIGHTS["mount"]: modal.Volume.from_name(WEIGHTS["volume"]).read_only(), "/logs": log_volume},
              secrets=[modal.Secret.from_name(SECRET)], timeout=int(CFG["hard_timeout_h"] * 3600), max_containers=1)
def serve() -> str:
    model = CFG["model"]
    cmd = ["vllm", "serve", model["hf_id"], f"--revision={model['revision']}", f"--served-model-name={model['hf_id']}",
           "--host=127.0.0.1", f"--port={VLLM_PORT}", *model["args"]]
    print("launching:", cmd, flush=True)
    proc = subprocess.Popen(cmd)
    proxy = None
    headers = {"Authorization": f"Bearer {os.environ['VLLM_API_KEY']}"}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    # One file per server: servers started in the same second (the 10-08 pilot's three) shared one name, and Volume
    # commits are last-write-wins. The function call ID makes it unique.
    log = f"/logs/requests_{stamp}_{modal.current_function_call_id()}.jsonl.gz"
    # Per-minute, server-wide load. Arms of a task share a server, so per-arm figures are each run's exposure (this
    # file averaged over the run's minutes on its server) plus per-request latency from the request log.
    load_log = log.replace("/requests_", "/load_").removesuffix(".gz")
    try:
        _wait_for_models(f"http://127.0.0.1:{VLLM_PORT}/v1/models", headers, [proc], CFG["startup_wait_min"], "vLLM")
        inject = {"repetition_detection": CFG["repetition_settings"][CFG["repetition_setting"]]}
        header = {"model": model["hf_id"], "revision": model["revision"], "vllm_version": CFG["vllm_version"],
                  "vllm_args": model["args"], "modal_call_id": modal.current_function_call_id(),
                  "repetition_setting": CFG["repetition_setting"], "injected_into_every_chat_request": inject,
                  "logprobs": "vLLM's default logprobs mode, raw_logprobs: log-softmax of the model's logits before "
                              "temperature, top-k and top-p, for the sampled token only"}
        proxy = subprocess.Popen(["python", PROXY_IN_IMAGE, f"--port={PORT}", f"--log={log}",
                                  f"--upstream=http://127.0.0.1:{VLLM_PORT}", f"--header={json.dumps(header)}",
                                  f"--inject={json.dumps(inject)}"])
        _wait_for_models(f"http://127.0.0.1:{PORT}/v1/models", headers, [proc, proxy], 2, "request-log proxy")
        print(f"REQUEST_LOG={REQUEST_LOG}:{log.removeprefix('/logs/')}", flush=True)

        def metrics() -> str:
            request = urllib.request.Request(f"http://127.0.0.1:{VLLM_PORT}/metrics", headers=headers)
            return urllib.request.urlopen(request, timeout=10).read().decode()

        # The idle exit reads this gauge; a renamed metric would read as "always idle" and end a live run, so fail now.
        assert "vllm:num_requests_running" in metrics(), "vLLM's /metrics has no vllm:num_requests_running"
        with modal.forward(PORT) as tunnel:
            print(f"TUNNEL_URL={tunnel.url}", flush=True)
            up, last_busy, served_any, done_before, failed_scrapes = time.time(), time.time(), False, 0.0, 0
            tokens_before, t_before, queue_before = 0.0, time.time(), (0.0, 0.0)
            while proc.poll() is None and proxy.poll() is None:
                time.sleep(60)
                try:  # the request log so far, readable from other containers; a failure is retried next minute
                    log_volume.commit()
                except Exception as e:  # noqa: BLE001 — a monitoring step must not end the server
                    print(f"volume commit failed (retried next minute): {e!r}", flush=True)
                now = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%H:%M:%S PT")
                try:
                    text = metrics()
                # A slow scrape under load must not end the server and every run's tunnel with it.
                except (OSError, http.client.HTTPException) as e:
                    failed_scrapes += 1
                    print(f"{now} /metrics failed ({failed_scrapes} in a row): {e!r}", flush=True)
                    last_busy = time.time()  # unknown counts as busy, so a scrape problem never triggers the idle exit
                    if failed_scrapes >= 10:
                        reason = "vLLM's /metrics failed 10 minutes in a row"
                        break
                    continue
                failed_scrapes = 0
                running = _metric(text, "vllm:num_requests_running")
                waiting = _metric(text, "vllm:num_requests_waiting")
                done = _metric(text, "vllm:request_success_total")
                tokens = _metric(text, "vllm:generation_tokens_total")
                rate = (tokens - tokens_before) / max(time.time() - t_before, 1)
                tokens_before, t_before = tokens, time.time()
                kv_name = next((n for n in ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc") if n in text), None)
                queue = (_metric(text, "vllm:request_queue_time_seconds_sum"),
                         _metric(text, "vllm:request_queue_time_seconds_count"))
                finished = queue[1] - queue_before[1]  # mean queue time of the requests that finished this minute
                load = {"time": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds"),
                        "running": running, "waiting": waiting, "kv_cache_usage": _metric(text, kv_name) if kv_name
                        else None, "queue_s_mean": (queue[0] - queue_before[0]) / finished if finished else None,
                        "requests_finished": finished, "generated_tok_s": round(rate, 1), "gpu": _gpu()}
                queue_before = queue
                try:  # committed now, so a stop by the watcher loses no sample
                    with open(load_log, "a") as f:
                        f.write(json.dumps(load) + "\n")
                    log_volume.commit()
                except Exception as e:  # noqa: BLE001 — a monitoring step must not end the server
                    print(f"{now} load log write or commit failed: {e!r}", flush=True)
                print(f"{now} running={running:.0f} waiting={waiting:.0f} succeeded={done:.0f} "
                      f"generated={rate:.0f} tok/s kv={load['kv_cache_usage']} {load['gpu']}", flush=True)
                served_any |= done > 0 or running > 0
                # Busy if a request is in flight now, or one finished since the last poll (it ran between two polls).
                if running or waiting or done > done_before:
                    last_busy = time.time()
                done_before = done
                if served_any and time.time() - last_busy > CFG["idle_exit_min"] * 60:
                    reason = f"idle {CFG['idle_exit_min']} min after serving {done:.0f} requests"
                    break
                if not served_any and time.time() - up > CFG["startup_idle_min"] * 60:
                    reason = f"no request within {CFG['startup_idle_min']} min of startup"
                    break
            else:
                reason = f"vllm exited with {proc.returncode}" if proc.poll() is not None else (
                    f"request-log proxy exited with {proxy.returncode}")
        print(f"stopping: {reason}", flush=True)
        return reason
    finally:
        for p in (proxy, proc):  # the proxy first, so its last record is written before the final commit
            if p is not None and p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=60)
                except subprocess.TimeoutExpired:  # killed rather than left running; the cleanup still finishes
                    p.kill()
                    p.wait()
        log_volume.commit()


@app.local_entrypoint()
def main():
    print(serve.remote())
