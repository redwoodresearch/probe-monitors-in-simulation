"""A trivial CPU stand-in for run_vllm_tunnel.py's `serve`, to dry-run launch_servers.py (either server mode) and the
watcher's stop check for cents: same app naming (PHASE2_SERVER_APP), same `serve` name, prints a TUNNEL_URL line after
STARTUP_SECONDS (a window to test Ctrl+C mid-launch, as the real server's startup gives), then ends on its own after
STAND_IN_SECONDS in all, as the real server's idle exit would.
"""

import os
import time

import modal

APP = os.environ.get("PHASE2_SERVER_APP", "probe-shift_aswin-surya_sub-phase2_stand-in_kill-2026-11-06")
assert APP.startswith("probe-shift_aswin-surya_sub-phase2_") and len(APP) <= 64, APP
STARTUP_SECONDS, STAND_IN_SECONDS = 30, 90
app = modal.App(APP)


@app.function(cpu=0.125, memory=256, timeout=300, max_containers=1)
def serve() -> str:
    time.sleep(STARTUP_SECONDS)
    print("TUNNEL_URL=https://stand-in.invalid", flush=True)  # bare, as Modal's tunnel.url is
    time.sleep(STAND_IN_SECONDS - STARTUP_SECONDS)
    return "stand-in ended"


@app.local_entrypoint()
def main():
    print(serve.remote())
