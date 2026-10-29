"""Detach-safe run outputs (Aswin 10-16 via 2e: every Modal launch runs with `modal run --detach`, and a client drop must not lose work): the remote function
writes its return value, with the launch context and the config it ran, to the Volume under an exclusive name and commits; the local entrypoint writes
results/ from the return value as before, and fetch_panel_output.py re-materialises the same results/ files from the Volume copy by app id."""

import base64
import json
from pathlib import Path

OUT_DIR = "xfer/panel_out"


def _enc(x):
    import numpy as np

    if isinstance(x, (bytes, bytearray)):
        return {"__bytes_b64__": base64.b64encode(bytes(x)).decode()}
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    raise TypeError(f"panel_out: cannot store {type(x).__name__}")


def _dec(d):
    return base64.b64decode(d["__bytes_b64__"]) if set(d) == {"__bytes_b64__"} else d


def save_out(vol, mount, name, payload):
    """Write payload as JSON to <mount>/xfer/panel_out/<name>.json (exclusive create) and commit the Volume; returns the Volume path."""
    p = Path(mount) / OUT_DIR / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as fh:
        fh.write(json.dumps(payload, default=_enc))
    vol.commit()
    return f"{OUT_DIR}/{name}.json"


def write_once(path, data):
    """Create path with data (bytes) exclusively; if it already exists with the SAME bytes (a partial earlier materialisation), keep it; different bytes STOP
    (Codex detach r1: a client drop between files must be recoverable without ever overwriting)."""
    p = Path(path)
    try:
        with p.open("xb") as fh:
            fh.write(data)
    except FileExistsError:
        if p.read_bytes() != data:
            raise SystemExit(f"STOP: {p.name} exists with different content (never overwritten)") from None
        print(f"{p.name}: already present with identical bytes (kept)")


def load_out(path):
    return json.loads(Path(path).read_text(), object_hook=_dec)
