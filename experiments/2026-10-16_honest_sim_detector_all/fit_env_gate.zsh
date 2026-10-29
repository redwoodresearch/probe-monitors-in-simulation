# Environment gate for the honest-sim detector rerun (route C, 2026-10-18; a37cc R1 before FIT). Run as its own process from the GATE:
#   zsh -f fit_env_gate.zsh   -> prints exactly FIT_ENV_OK (rc 0), or one STOP line (rc 1).
# PYTHONPATH / PYTHONHOME would decide which tree add_local_python_source("probe_shift") mounts even when the venv's .pth check passes;
# UV_PROJECT_ENVIRONMENT would point `uv sync --check` at another venv. "Set" counts even when empty. uv is pinned to the version the
# package was built and checked with.
[ -z "${PYTHONPATH+x}" ] || { echo "STOP: PYTHONPATH is set"; exit 1; }
[ -z "${PYTHONHOME+x}" ] || { echo "STOP: PYTHONHOME is set"; exit 1; }
[ -z "${UV_PROJECT_ENVIRONMENT+x}" ] || { echo "STOP: UV_PROJECT_ENVIRONMENT is set"; exit 1; }
[ "$(uv --version 2>/dev/null | cut -d' ' -f1-2)" = "uv 0.12.19" ] || { echo "STOP: uv is not 0.12.19"; exit 1; }
echo FIT_ENV_OK
