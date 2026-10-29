"""Make the vLLM endpoint key once: a random token stored as a Modal Secret (the server side) and in the General
Sandbox secret store via rr-env (the client side, readable on the laptop with SSO and on the devbox with its role).
The value never reaches stdout, argv or a file. Rotate by deleting both and running this again.

  uv run --no-project --with modal==1.2.6 python experiments/2026-10-06_phase2_feasibility/run_setup_key.py
"""

import secrets
import subprocess

import modal

MODAL_SECRET = "probe-shift_aswin-surya_sub-phase2_vllm-key_kill-2026-11-06"
RR_ENV_LABEL = "probe-shift-aswin-surya-sub-phase2-vllm-key-kill-2026-11-06"  # rr-env labels: [a-z0-9-], max 63

# Refuse to run twice: rr-env put would overwrite the client key while the Modal create below fails, splitting them.
existing = [s.name for s in modal.Secret.objects.list()]
assert MODAL_SECRET not in existing, f"{MODAL_SECRET} exists; delete it and the rr-env secret to rotate"
token = secrets.token_urlsafe(32)
# Client side first, so a failure here leaves no orphan server key behind.
subprocess.run(["rr-env", "put", RR_ENV_LABEL, "--from-stdin", "--profile", "redwood-sandbox"],
               input=token, text=True, check=True)
modal.Secret.objects.create(MODAL_SECRET, {"VLLM_API_KEY": token})  # raises if it already exists
print("created the Modal Secret and the rr-env secret named at the top of this file")  # names only; never the token
