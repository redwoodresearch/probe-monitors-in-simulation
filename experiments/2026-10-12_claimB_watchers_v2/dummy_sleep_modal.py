"""A dummy CPU app for live watcher tests (v2.3, 10-13): one 0.125-CPU container that sleeps, so a stop can be tried on
an ephemeral `modal run` app. Costs fractions of a cent per minute.

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' modal run --detach dummy_sleep_modal.py
"""

import time

import modal

app = modal.App("probe-shift_aswin-surya_sub-claimb-wtest_kill-2026-11-12")


@app.function(cpu=0.125, memory=256, timeout=420, retries=0)
def sleep_for(seconds: int = 300):
    time.sleep(seconds)
    return seconds


@app.local_entrypoint()
def main():
    sleep_for.remote(300)
