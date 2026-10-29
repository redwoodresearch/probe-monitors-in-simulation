"""Resolve each probe's training length from its FIT spec, by key (git object reads only; no compute, no score values).

For every sheet entry: 'fit_spec_field' reads the fit record at commit:path, checks its sha256 and reads the named cut field per readout;
'table_row' reads a committed training-length table row per readout (its class text verbatim, class key checked); 'pending' stays pending and is listed.
Writes results/training_lengths_<stamp>.json (exclusive create); the table generator takes it by sha256 and stops on any pending entry it uses.

  uv run python experiments/2026-10-16_fl_vs_32k_appendix/resolve_training_lengths.py [--sheet config/training_length_sheet.json]
"""

import argparse
import datetime
import json
import re
import subprocess
from pathlib import Path
from zoneinfo import ZoneInfo

from flio import sha_bytes, stop, write_new

HERE = Path(__file__).resolve().parent


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout


def blob(e, name):
    b = git("show", f"{e['commit']}:{e['path']}")
    if sha_bytes(b) != e["sha256"]:
        stop(f"{name}: {e['path']} @ {e['commit']} sha256 {sha_bytes(b)} != {e['sha256']}")
    return json.loads(b)


def dotted(d, path, name):
    for p in path.split("."):
        if not isinstance(d, dict) or p not in d:
            stop(f"{name}: no field {path}")
        d = d[p]
    return d


def cut_text(field, v):
    if v is None:
        return f"trained on full transcripts (fit spec {field} = none)"
    if isinstance(v, int) and not isinstance(v, bool):
        return f"trained within the first {v:,} tokens (fit spec {field} = {v})"
    stop(f"unexpected cut value type {type(v).__name__} at {field}")


def rel(p):
    """The sheet's path relative to the repo when it is inside it, else absolute (a sheet outside the repo is never a plan input)."""
    p, root = p.resolve(), HERE.parent.parent.resolve()
    return str(p.relative_to(root)) if p.is_relative_to(root) else str(p)


def table_row_checks(name, e):
    """No readout borrows another's row (each row is named for its own readout, optionally with a fit tag), and an unsigned table never resolves."""
    if len(set(e["rows"].values())) != len(e["rows"]) or any(not re.search(rf"\({ro}(; [^)]+)?\)$", row) for ro, row in e["rows"].items()):
        stop(f"{name}: each readout needs its own table row, named '<probe> (<readout>)' or '<probe> (<readout>; <fit>)': {e['rows']}")
    if "<" in str(e.get("signed", "")):
        stop(f"{name}: the table's signature is not filled in yet ({e['signed']}); resolve with the signed sheet only")


def check_signature(name, sig, table):
    """A structured signature {commit, file}: the signing commit's change to that file names the table's commit and the first 12 hex digits
    of its sha256 (09's READ_LOG form), both taken from the table entry itself so a changed table can never ride on an old signature."""
    if not isinstance(sig, dict):
        return None
    text = git("show", sig["commit"], "--", f":(top){sig['file']}").decode()  # pathspec from the repo root (git runs in this folder)
    need = [table["commit"], table["sha256"][:12]]
    miss = [m for m in need if m not in text]
    if miss:
        stop(f"{name}: signature commit {sig['commit']} does not name {miss} in {sig['file']}")
    return {"commit": sig["commit"], "file": sig["file"], "names": need}


def canon(row):
    return json.dumps(row, sort_keys=True)


def resolve(name, e):
    src = {k: e[k] for k in ("path", "commit", "sha256") if k in e}
    if e["kind"] == "pending":
        return {"status": "pending", "probe": e["probe"], "why": e["why"]}
    if e["kind"] == "fit_spec_field":
        rec = blob(e, name)
        out = {}
        for ro, field in e["fields"].items():
            v = dotted(rec, field, name)
            if v != e["expect"]:
                stop(f"{name}: {field} = {v!r}, sheet expects {e['expect']!r}")
            out[ro] = {"text": cut_text(field, v), "field": field, "value": v, "source": src}
        return {"status": "resolved", "probe": e["probe"], "kind": "fit_spec_field", "by_readout": out}
    if e["kind"] == "table_row":  # one signed-table row PER READOUT (r3 has explicit C rows; no readout borrows another's row)
        table_row_checks(name, e)
        signature = check_signature(name, e.get("signed"), e)
        tab = blob(e, name)
        signed_in = None
        if e.get("row_signed_in"):  # the row as 09 signed it in an earlier revision: same signature check there, and the rows identical
            rv = e["row_signed_in"]
            signed_in = {**{k: rv[k] for k in ("path", "commit", "sha256")}, "signature": check_signature(f"{name} (signed revision)", rv["signed"], rv)}
            rtab = blob(rv, f"{name} (signed revision)")
        out = {}
        for ro, row in e["rows"].items():
            rows = [r for r in tab["rows"] if r.get("probe") == row]
            if len(rows) != 1:
                stop(f"{name}: {len(rows)} rows named {row!r} in {e['path']}")
            r = rows[0]
            if r.get("class_key") != e["expect_class_key"]:
                stop(f"{name} {ro}: class_key {r.get('class_key')!r} != {e['expect_class_key']!r}")
            if signed_in is not None:
                r0 = [x for x in rtab["rows"] if x.get("probe") == row]
                if len(r0) != 1 or canon(r0[0]) != canon(r):
                    stop(f"{name} {ro}: row {row!r} differs from the revision 09 signed it in ({e['row_signed_in']['commit']})")
            out[ro] = {"text": r["class"], "class_key": r["class_key"], "row": row, "class_source": r.get("class_source"), "source": src, "signature": signature,
                       "row_signed_in": signed_in}
        return {"status": "resolved", "probe": e["probe"], "kind": "table_row", "by_readout": out}
    stop(f"{name}: unknown kind {e['kind']!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sheet", default=str(HERE / "config" / "training_length_sheet_r8.json"))  # r8, complete; every row signed by 09
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    a = ap.parse_args()
    sb = Path(a.sheet).read_bytes()
    sheet = json.loads(sb)
    probes = {n: resolve(n, e) for n, e in sheet["probes"].items()}
    now = datetime.datetime.now(ZoneInfo("America/Los_Angeles"))
    out = {"what": "training length per probe from its FIT spec (by key); 'pending' entries have no committed source yet",
           "sheet": {"path": rel(Path(a.sheet)), "sha256": sha_bytes(sb)},
           "git": git("rev-parse", "HEAD").decode().strip(), "stamp (PT)": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
           "pending": sorted(n for n, v in probes.items() if v["status"] == "pending"), "probes": probes}
    write_new(Path(a.out_dir) / f"training_lengths_{now.strftime('%Y-%m-%d_%H%M%S')}.json", json.dumps(out, indent=1))
    print("pending:", out["pending"])


if __name__ == "__main__":
    main()
