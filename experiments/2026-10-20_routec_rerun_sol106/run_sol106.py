"""sol106: independent re-derivation of release v2.2 row 106 (attack-side Sol fallback share).

F rerun (POST HOC, disclosed): Addendum F as amended by F1, F2, F3 and F4. It runs only after DIAG_AMEND3/4/5 outcome
A (results commit 0a0bae859) and 09's ruling (i). Changes from the stopped run (bf8ae996): the two non-writer ids join
the allowed set of the unexpected-id STOP gate (F); the STOP-only gate G_EVAL_MODEL_NOT_JUDGE (F2, restating F1.2);
labels; toy units; the decorator (F1.3, F2.3); explicit zstd support with gate G_ZSTD_ZIP (F3); a pinned real-copy
control before the scan and an empty-R rule, both STOP "PLUMBING" (F4). Verdicts, counts and the comparison are
unchanged.

Definitions: PRESTATED_<stamp>.md in this item folder (spec lines quoted there). Values stay in the container: the
results JSON goes to the output Volume, and the laptop sees one SOL106_RESULT line of booleans, key names, counts of
mismatching keys, the Volume path and shas. No numbers from data. Errors carry only gate names or exception type names.

Modal (named binary, from the committed blob in the worktree):
  "$M" run --detach run_sol106.py --mode real       # real run (toy controls run first, in-container)
  "$M" run run_sol106.py --mode test                # toy, predicted PASS (exit 0)
  "$M" run run_sol106.py --mode test-neg            # toy with one Sol bit flipped, predicted FAIL (exit 1)
The toys run only inside Modal calls: the test and test-neg calls, and in-container before the real run (09: no local
Python). The `toy` / `toy-neg` command-line entry at the end of this file is not used.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
import zlib
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

# Addendum F3 (COMMON_RULES, 09's lesson after DIAG_AMEND5): Inspect writes .eval members with zstd (ZIP method 93);
# stdlib zipfile < 3.14 cannot read them. raw_header used to work only because Inspect, imported earlier in the same
# process, patches zipfile: a hidden dependency. Import the patches explicitly and gate on them (G_ZSTD_ZIP).
if sys.version_info < (3, 14):
    try:
        import zipfile_zstd  # noqa: F401  (an inspect-ai dependency on Python < 3.14)
    except ImportError:  # only the laptop's local entrypoint can lack it, and it never opens a zip
        pass
try:  # Inspect's own zip helper: imports zipfile_zstd and installs its multi-frame zstd decoder at import
    import inspect_ai._util.zipfile  # noqa: F401
except ImportError:  # laptop only, as above; G_ZSTD_ZIP requires the patch in the worker
    pass
ZIP_ZSTD = getattr(zipfile, "ZIP_ZSTANDARD", None)

_WORKER = os.environ.get("SOL106_WORKER") == "1"  # the isolated data worker never imports modal (Codex r7)

APP_NAME = "probe-shift_aswin-surya_sub-rerun_sol106_kill-2026-11-19"
OUT_DIR = "/out/sol106"
# label -> (Volume name, mount point, root of run paths under the mount); handoff section 3
VOLS = {
    "chain": ("probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14", "/chain", "/chain/evals_v1"),
    "q4own": ("probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13", "/q4own", "/q4own/evals_v1"),
    "pilots": ("probe-shift_aswin-surya_sub-q3pilots_runs_kill-2026-11-13", "/pilots", "/pilots"),
    "q2fresh": ("probe-shift_aswin-surya_sub-q2-fresh_work_kill-2026-11-13", "/q2fresh", "/q2fresh"),
}
CELLS = ("command|L48", "command|L63", "reply|L48", "reply|L63")
SIM_YAML_L19 = "  PT_SIM_MODEL: openai/gpt-6-astra"
SIM_YAML_L20 = "  PT_SIM_FALLBACK_MODEL: openai/gpt-5.6-sol"
POLICY_ID = "vllm/Qwen/Qwen3.5-27B"  # NOTE_SF L16 @ d1db55766 (quoted in the handoff)
# Addendum F / F1.1: ids that are not simulator writers. They only widen the unexpected-id gate's allowed set; no
# verdict function reads them. Role texts are 09's rulings (GATES @ be2fbc529, relayed by 2e ~16:3x) after the event
# check's outcome A (DIAG_AMEND3/4/5): ruling (i) holds for both ids. RESULTS_COMMIT is the commit that holds that
# check's results (record 806b99fc...), filled before freezing; unit F_labels fails on a placeholder. 09's final
# text uses the short sha 0a0bae859 (full: 0a0bae859a276a4372957a85bb6cd26047715ca3).
RESULTS_COMMIT = "0a0bae859"
TERRA_ROLE = "simulator refusal / router judge (sim.yaml @ 8d7d6a192 L21-22)"
TERRA_CLAUSE = ("span placement (simulated tool calls) consistent; non-writer status rests on the config, not on the "
                "span data")
NANO_ROLE = f"side-task scorer (nano), by data (DIAG_AMEND3/4/5 outcome A @ {RESULTS_COMMIT})"  # 09's OK'd text
NON_WRITER_IDS = {"openai/gpt-5.6-terra": f"{TERRA_ROLE} (terra); {TERRA_CLAUSE}",
                  "openai/gpt-5-nano-2025-08-07": NANO_ROLE}
NON_WRITER_LABEL = (f"{TERRA_ROLE} (terra); {TERRA_CLAUSE}; {NANO_ROLE}; non-writer; never counted as Sol, "
                    "never decide SOL / FREE / INCOMPLETE")
POST_HOC_LABEL = ("POST HOC, disclosed: NON_WRITER_IDS added after the 13:4x STOP (record bf8ae996) under 09's "
                  "ruling (i) on the sol106 diagnostic")  # Addendum F, verbatim (09: keep)
POST_HOC_F2_LABEL = ("POST HOC, disclosed (Addendum F2): G_EVAL_MODEL_NOT_JUDGE (STOP EVAL_MODEL_NOT_ALLOWED) "
                     "restates F1.2's gate after the diagnostic (record 1e9942e8) showed eval.model none/none in every "
                     "read copy; allowed eval.model: none/none or the policy; 09's ruling")
EVAL_MODEL_NONE = "none/none"  # F2: no writer recorded in eval.model
EM_MISSING, EM_NONSTR = "<missing>", "<non-string>"  # F2: printed names for a missing / non-string eval.model
_ABSENT = object()
TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens", "input_tokens_cache_write",
                "input_tokens_cache_read", "reasoning_tokens")
TARGET_KEYS = ("n_sol_using", "n_all", "n_sol_free", "n_unknown")
PRINTED_PCT = 42  # public sentence, provenance v17 row 106: ratio_pct_round_eq, 0 dp
LABEL = "independent value check (definitions: sim.yaml + NOTE_SF; population / join / copies as ruled by 09)"
BLINDING_LABEL = "independent derivation; the author was NOT blind to the target count (disclosed §0)"  # 09, exact
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"[A-Za-z0-9._/:-]{1,80}")  # 09 (addendum E): id-shaped strings only, used with fullmatch
NON_ID = "<non-id string>"
MAX_IDS_PRINTED = 10
PINS = {  # name -> (commit, path, sha256 of the blob bytes)
    "refit3": ("80d75b27e", "experiments/2026-10-16_attack_refit3/results/attack_all_refit3_2026-10-16_141813.json",
               "df7171a98d82d28ae86653480be922d978fc36104c63f622b3a69fb43a074eac"),
    "target": ("db6f66287", "experiments/2026-10-20_routec_simpool_numbers/results/attack_sol_share_2026-10-20_023418.json",
               "522897ef4fddcbefef46e396361ae2e22564231484a0d0a3162a28181d544b2d"),
    "scan_rows": ("ff8ea35a7", "experiments/2026-10-16_attack_prompt_provenance/results/openai_usage_2026-10-16_185845_scan_rows.json",
                  "0d0392fb773b1804a3ded3b4f3fd2f4dc8e7ae33b1ee660dd7200d0550410197"),
    "sim_yaml": ("8d7d6a192", "experiments/2026-10-07_phase2_attack_check/config/sim.yaml",
                 "03340860a688a04ea6cb017b46067e037748f5864108b79a4ff84d02245f4612"),
}
# Addendum F4: the real-copy positive control pinned by DIAG_AMEND5 §3 (accepted by 09); it read cleanly there.
CONTROL_KEY = "sim|nm000_auto_workflow"
CONTROL_PATH = ("2026-10-14_newmains_s5_fu/sim_attack@september/nm000_auto_workflow/"
                "2026-10-15T07-54-10-00-00_la-auto-workflow-sim-attack-1_FnjkqEukeqBUUiFU74Wfwi.eval")
CONTROL_SHA = "b84d7d9b4a85cabe85153e1d6c5e742620cce4ea64007f53f8e2f87c5c4ba369"
CONTROL_NANO = "openai/gpt-5-nano-2025-08-07"
MANIFEST_COMMIT = "1df288e61"
MANIFEST_DIR = "experiments/2026-10-14_q3_overnight_chain/config/"
MANIFEST_N = 293
MANIFEST_DIGEST = "bce4051be4bf26b9bb855b300604732799df88fc8c03c86f23dcbbb0c7425ab1"
# Gate predictions for the real run (PRESTATED section 8). Informative gates do not block the figure.
PREDICTED = {"n_paths_without_sha_ok_copy": 0, "n_sha_mismatch_files": 0, "n_sha_read_errors": 0,
             "n_paths_multi_copy": 0, "n_readers_disagree": 0, "n_copies_unexpected_model_id": 0,
             "n_token_field_inconsistent": 0, "n_sol_incomplete_overlap": 0, "n_prod_xcheck_disagree": 0}


class GateError(RuntimeError):
    """Raised with a gate name only; the only exception text allowed to leave the container."""


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def manifest_digest(items: list[tuple[str, bytes]]) -> str:
    return sha256("".join(f"{p}\t{sha256(b)}\n" for p, b in sorted(items)).encode())


def has_tokens(u: dict | None) -> bool:  # NOTE_SF L23: tokens greater than 0, not name presence
    return bool(u) and any((u.get(f) or 0) > 0 for f in TOKEN_FIELDS)


def copy_verdict(bits: dict[str, bool] | None, ids: dict[str, str]) -> str:
    """One source copy (NOTE_SF L13-L18). bits: model id -> any token count > 0; None = read error / no header."""
    if bits is None:
        return "INCOMPLETE"  # L18
    if bits.get(ids["sol"]):
        return "SOL"  # L14, listed first: wins over L15 (PRESTATED gap 5)
    if not bits or not bits.get(ids["astra"]) or not bits.get(ids["policy"]):
        return "INCOMPLETE"  # L15-L16
    return "FREE"  # L17


def copy_verdict_gap5_alt(bits: dict[str, bool] | None, ids: dict[str, str]) -> str:
    """09's gap-5 sensitivity (addendum A): a copy with Sol tokens but missing simulator/policy tokens is INCOMPLETE."""
    if bits and bits.get(ids["sol"]) and not (bits.get(ids["astra"]) and bits.get(ids["policy"])):
        return "INCOMPLETE"
    return copy_verdict(bits, ids)


def pre_status(gates: dict) -> tuple[str | None, str | None]:
    """Before the target is read (addendum A; F1.2 order as restated by F2): STOP on SOL+FREE copies, then on an
    eval.model outside {none/none, policy}, then on unexpected model ids; GATE_FAIL otherwise. Missing keys fail closed."""
    if not gates.get("G_NO_STOP", False):
        return "STOP", "SOL_AND_FREE_COPIES"
    if not gates.get("G_EVAL_MODEL_NOT_JUDGE", False):
        return "STOP", "EVAL_MODEL_NOT_ALLOWED"
    if not gates.get("G_NO_UNEXPECTED_MODEL_IDS", False):
        return "STOP", "UNEXPECTED_MODEL_IDS"
    if not all(gates.values()):
        return "GATE_FAIL", None
    return None, None


def post_status(prim: dict, gap5_pct_eq_printed: bool) -> str:
    """After comparison: any primary mismatch -> MISMATCH; else a gap-5 change of the rounded 42 -> UNRESOLVED."""
    if not all(prim.values()):
        return "MISMATCH"
    return "PASS" if gap5_pct_eq_printed else "UNRESOLVED"


def path_verdict(vs: list[str]) -> str:
    """Per path over its copies (NOTE_SF L19). No copy at all is UNKNOWN (PRESTATED gap 4)."""
    s = set(vs)
    if "SOL" in s and "FREE" in s:
        return "STOP"
    if "SOL" in s:
        return "SOL"
    if "INCOMPLETE" in s or not s:
        return "UNKNOWN"
    return "FREE"


def pct_half_up(num: int, den: int) -> int | None:
    if not den:
        return None
    return int((Decimal(100 * num) / Decimal(den)).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def as_int(x) -> int | None:
    """Integral record count or None (bools, non-numbers and non-integral floats are not counts)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return int(x) if float(x).is_integer() else None


def zstd_selftest() -> bool:
    """G_ZSTD_ZIP (Addendum F3): Inspect's zip patches are installed (zstd plus its multi-frame decoder) and zipfile can
    write AND read back a zstd member (Inspect's member format) in this process."""
    if ZIP_ZSTD is None or getattr(zipfile, "_inspect_ai_multiframe_installed", False) is not True:
        return False
    try:
        with tempfile.TemporaryDirectory() as tmp:
            p = f"{tmp}/z.zip"
            with zipfile.ZipFile(p, "w", compression=ZIP_ZSTD) as z:
                z.writestr("m.json", b'{"k": [1, 2]}')
            with zipfile.ZipFile(p) as z:
                return z.getinfo("m.json").compress_type == ZIP_ZSTD and z.read("m.json") == b'{"k": [1, 2]}'
    except Exception:
        return False


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def em_kind(v) -> tuple[str, str | None]:
    """F2: classify a projected eval.model without raising: ("str", v), ("missing", None) or ("nonstr", None)."""
    if v is _ABSENT:
        return "missing", None
    if isinstance(v, str):
        return "str", v
    return "nonstr", None


def raw_header(path: str) -> dict:
    """Second reader: header.json member only; keeps per-model bits and eval.model (F2), never content.
    has_header: True / False (zip opened, no header.json) / None (this reader failed: unknown)."""
    try:
        with zipfile.ZipFile(path) as z:
            if "header.json" not in z.namelist():
                return {"has_header": False, "bits": None, "tot_bits": None, "err": None, "em": None}
            h = json.loads(z.read("header.json"))
        mu = (h.get("stats") or {}).get("model_usage") or {}
        ev = h.get("eval")  # F2: projected without raising, so it never changes the reader's other fields
        em = em_kind(ev.get("model", _ABSENT) if isinstance(ev, dict) else _ABSENT)
        return {"has_header": True, "bits": {m: has_tokens(u) for m, u in mu.items()},
                "tot_bits": {m: ((u or {}).get("total_tokens") or 0) > 0 for m, u in mu.items()}, "err": None,
                "em": em}
    except Exception as e:  # keep the type name only
        return {"has_header": None, "bits": None, "tot_bits": None, "err": type(e).__name__, "em": None}


async def inspect_header(path: str) -> dict:
    """Primary reader: Inspect's API, header_only=True (no samples are read). Also projects log.eval.model (F2)."""
    from inspect_ai.log import read_eval_log_async

    try:
        log = await read_eval_log_async(path, header_only=True)
        # F1.2: the eval.model projection sits inside this error boundary (Codex r28); it cannot raise on the pinned
        # EvalLog / EvalSpec types (plain getattr with defaults)
        spec = getattr(log, "eval", None)
        em = em_kind(getattr(spec, "model", _ABSENT) if spec is not None else _ABSENT)
    except Exception as e:  # a read error is a defined verdict (L18); keep the type name only
        return {"bits": None, "err": type(e).__name__, "status": None, "em": None}
    mu = log.stats.model_usage if log.stats else {}
    return {"bits": {m: has_tokens(u.model_dump()) for m, u in mu.items()}, "err": None, "status": log.status,
            "em": em}


async def scan_path(key: str, path: str, exp_sha: str, roots: dict[str, str], ids: dict, sem) -> dict:
    copies = []
    for vol, root in roots.items():  # every Volume root is searched; sha-matched files are the copies
        full = os.path.join(root, path)
        async with sem:
            if not await asyncio.to_thread(os.path.isfile, full):
                continue
            try:
                got = await asyncio.to_thread(file_sha256, full)
            except Exception as e:  # unverifiable file: not a copy; counted
                copies.append({"vol": vol, "sha_ok": False, "sha_err": type(e).__name__})
                continue
            if got != exp_sha:
                copies.append({"vol": vol, "sha_ok": False, "sha_err": None})
                continue
            ins = await inspect_header(full)
            raw = await asyncio.to_thread(raw_header, full)
        # L18: primary read error, or a zip that definitely has no header.json -> no model list
        bits = None if (ins["bits"] is None or raw["has_header"] is False) else ins["bits"]
        # F1.2 / F2: eval.model from the raw reader if it read the header, else Inspect's; neither -> unread (None)
        em = raw["em"] if raw["has_header"] is True else (ins["em"] if ins["err"] is None else None)
        copies.append({"vol": vol, "sha_ok": True, "inspect_err": ins["err"], "raw_err": raw["err"],
                       "has_header": raw["has_header"], "status": ins["status"], "bits": bits,
                       "raw_bits": raw["bits"], "tot_bits": raw["tot_bits"], "verdict": copy_verdict(bits, ids),
                       "eval_model_kind": em[0] if em else None, "eval_model": em[1] if em else None})
    return {"key": key, "path": path, "sha256": exp_sha, "copies": copies,
            "verdict": path_verdict([c["verdict"] for c in copies if c["sha_ok"]])}


async def scan_all(jobs, roots, ids, conc=32):
    sem = asyncio.Semaphore(conc)
    return await asyncio.gather(*(scan_path(k, p, s, roots, ids, sem) for k, p, s in jobs))


def _run_worker(cmd: list[str], sink, timeout: int = 1700) -> int:
    """Run cmd as an isolated process: stdin closed; stdout AND stderr go to sink for its whole life, including its
    exit-time flushes and any child process. Data is only ever parsed in such a process (Codex r6/r7)."""
    env = {**os.environ, "SOL106_WORKER": "1"}
    return subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=sink, stderr=sink, env=env, check=False,
                          timeout=timeout).returncode


def _noise_main() -> None:
    """For _worker_selftest only: writes markers every way a dependency could, incl. a buffered flush at exit."""
    print("s1", end="")  # stays buffered until interpreter exit
    sys.stderr.write("s2")
    os.write(1, b" s3\n")
    os.write(2, b" s4\n")
    subprocess.run(["sh", "-c", "printf ' s5'; printf ' s6' 1>&2"], check=False)
    raise SystemExit(3)


def _worker_selftest() -> bool:
    """The launcher must route every write of the worker, its child, and its exit flush to the sink."""
    with tempfile.TemporaryFile() as sink:
        rc = _run_worker([sys.executable, os.path.abspath(__file__), "--noise"], sink, timeout=120)
        sink.seek(0)
        text = sink.read().decode(errors="replace")
    return rc == 3 and all(f"s{i}" in text for i in range(1, 7))


def _seen_ids(c: dict) -> set[str]:
    """Model ids observed in a copy by either reader (names only)."""
    return set(c.get("bits") or {}) | set(c.get("raw_bits") or {})


def known_ids(ids: dict) -> set[str]:
    """Addendum F's one logic change: the allowed set of the unexpected-id STOP gate gains the two non-writers."""
    return {ids["sol"], ids["astra"], ids["policy"]} | set(NON_WRITER_IDS)


def _em_name(c: dict) -> str:
    k = c["eval_model_kind"]
    if k == "missing":
        return EM_MISSING
    if k == "nonstr":
        return EM_NONSTR
    v = c["eval_model"]
    return v if isinstance(v, str) and ID_RE.fullmatch(v) else NON_ID


def eval_model_check(copies: list[dict], ids: dict) -> dict:
    """Addendum F2, G_EVAL_MODEL_NOT_JUDGE. R = the given sha-matched copies that either reader read; X = those whose
    eval.model is not "none/none" or the policy (a missing or non-string value counts as not allowed). The gate is
    "X is empty" (vacuously true for empty R). Names (guarded, distinct, sorted, at most 10) and the {gt0, eq_all}
    pair are information only; the counts stay on the Volume."""
    allowed = {EVAL_MODEL_NONE, ids["policy"]}
    r = [c for c in copies if c.get("eval_model_kind") is not None]
    x = [c for c in r if not (c["eval_model_kind"] == "str" and c["eval_model"] in allowed)]
    return {"gate": not x, "n_read": len(r), "n_not_allowed": len(x),
            "names": sorted({_em_name(c) for c in x})[:MAX_IDS_PRINTED],
            "pair": {"gt0": len(x) > 0, "eq_all": len(x) == len(r)}}


def pre_decision(gates: dict, emc: dict) -> tuple[str | None, str | None]:
    """Addendum F4 (09-approved): an empty R, i.e. no sha-matched copy read by either reader, STOPs as PLUMBING before
    every other rule, so F2's gate can never pass vacuously into a target read. Otherwise pre_status decides."""
    if emc.get("n_read", 0) == 0:
        return "STOP", "PLUMBING"
    return pre_status(gates)


def _control_pinned(refit3: dict, manifests: list, key: str, path: str, sha: str) -> bool:
    """F4: the pinned key is a population key (a sim| key with rows in some cell), its run key is path + '#e0', and
    every sim_attack manifest row for the path has exactly that sha and the key's task (derive's join, for one key)."""
    rk = refit3.get("run_keys") or {}
    in_pop = key.startswith("sim|") and any(key in (refit3["cells"][c]["rows_per_run"] or {}) for c in CELLS)
    rows = [r for _, m in manifests for r in ((m.get("files") or []) if isinstance(m, dict) else [])
            if isinstance(r, dict) and r.get("path") == path and str(r.get("arm_dir") or "").startswith("sim_attack")]
    return (in_pop and rk.get(key) == path + "#e0" and bool(rows) and {r.get("sha256") for r in rows} == {sha}
            and all(r.get("task") == key.split("|", 1)[1] for r in rows))


def control_check(refit3: dict, manifests: list, roots: dict, key: str, path: str, sha: str) -> dict:
    """Addendum F4 real-copy positive control (DIAG_AMEND5 §3, ported): before any scan, the pinned copy must exist as
    exactly one sha-matched file, Inspect's header read must succeed, the raw reader must read header.json, nano must
    be among BOTH readers' model_usage keys, and its eval.model (F1.2 selection) must be a string. Header-only reads.
    Booleans and one guarded exception type name only."""
    pinned = _control_pinned(refit3, manifests, key, path, sha)
    matched, sha_err = [], None
    for r in roots.values():
        full = os.path.join(r, path)
        try:
            if os.path.isfile(full) and file_sha256(full) == sha:
                matched.append(full)
        except Exception as e:  # a filesystem error is a PLUMBING result, not a worker crash
            sha_err = type(e).__name__
    out = {"pinned": pinned, "one_copy": len(matched) == 1 and sha_err is None, "inspect_ok": False,
           "raw_header": False, "nano_listed": False, "eval_model_str": False, "err": sha_err}
    if out["one_copy"]:
        ins = asyncio.run(inspect_header(matched[0]))
        raw = raw_header(matched[0])
        em = raw["em"] if raw["has_header"] is True else (ins["em"] if ins["err"] is None else None)
        out.update({"inspect_ok": ins["err"] is None, "raw_header": raw["has_header"] is True,
                    "nano_listed": CONTROL_NANO in (ins["bits"] or {}) and CONTROL_NANO in (raw["bits"] or {}),
                    "eval_model_str": em is not None and em[0] == "str", "err": ins["err"] or raw["err"]})
    if out["err"] is not None and not ID_RE.fullmatch(out["err"]):
        out["err"] = NON_ID
    out["ok"] = all(out[k] for k in ("pinned", "one_copy", "inspect_ok", "raw_header", "nano_listed", "eval_model_str"))
    return out


def safe_ids(ids: list) -> dict:
    """For the SOL106_RESULT line only (09, addendum E): each id printed only if it is id-shaped, else NON_ID (counted);
    at most MAX_IDS_PRINTED names, plus the total count. The full list stays on the Volume."""
    shown = [i if isinstance(i, str) and ID_RE.fullmatch(i) else NON_ID for i in ids]
    return {"unexpected_model_ids": shown[:MAX_IDS_PRINTED], "n_unexpected_model_ids": len(ids),
            "n_unexpected_non_id_strings": shown.count(NON_ID)}


def counts(keys, verdict: dict[str, str]) -> dict:
    vs = [verdict[k] for k in keys]
    return {"n_all": len(vs), "n_sol_using": vs.count("SOL"), "n_sol_free": vs.count("FREE"),
            "n_unknown": vs.count("UNKNOWN"), "n_stop": vs.count("STOP")}


def derive(refit3: dict, manifests: list[tuple[str, dict]], scan_rows: dict, ids: dict, roots: dict) -> dict:
    """Population, join, copies, verdicts, counts. Raises GateError(name) on a blocking gate."""
    rk = refit3["run_keys"]
    sim = {k: v for k, v in rk.items() if k.startswith("sim|")}
    g = {"G_RUNKEYS_SUFFIX_E0": all(isinstance(v, str) and v.endswith("#e0") for v in sim.values())}
    part = True
    for c in CELLS:
        rpr, rwr = refit3["cells"][c]["rows_per_run"], refit3["cells"][c]["runs_without_rows"]
        zr = [r.get("name") for r in refit3["zero_row_runs"][c]["runs"]]
        part &= (sorted(list(rpr) + list(rwr)) == sorted(rk) and not set(rpr) & set(rwr)
                 and all(as_int(n) is not None and n > 0 for n in rpr.values()) and set(zr) == set(rwr))
    g["G_CELLS_PARTITION"] = part
    # Gap 1: a cell's simulated population = sim| keys with rows in that cell; pooled = union over the 4 cells.
    cell_pop = {c: sorted(k for k in refit3["cells"][c]["rows_per_run"] if k in sim) for c in CELLS}
    pop = sorted(set().union(*cell_pop.values()))
    # Gap 2: run -> path = refit3 .run_keys value without the '#e0' suffix; sha from the launch manifests.
    path_of = {k: sim[k].rsplit("#", 1)[0] for k in pop}
    g["G_PATHS_INJECTIVE"] = len(set(path_of.values())) == len(pop)
    by_path: dict[str, list[dict]] = {}
    for _, m in manifests:
        for r in ((m.get("files") or []) if isinstance(m, dict) else []):
            if isinstance(r, dict) and str(r.get("arm_dir") or "").startswith("sim_attack"):
                by_path.setdefault(r.get("path"), []).append(r)
    exp_sha, join_ok = {}, True
    for k in pop:
        rs = by_path.get(path_of[k], [])
        shas = {r.get("sha256") for r in rs}
        ok = (bool(rs) and len(shas) == 1 and all(isinstance(s, str) and SHA_RE.fullmatch(s) for s in shas)
              and all(r.get("task") == k.split("|", 1)[1] for r in rs))
        join_ok &= ok
        if ok:
            exp_sha[k] = next(iter(shas))
    g["G_MANIFEST_JOIN"] = join_ok
    for name in ("G_RUNKEYS_SUFFIX_E0", "G_CELLS_PARTITION", "G_PATHS_INJECTIVE", "G_MANIFEST_JOIN"):
        if not g[name]:
            raise GateError(name)
    # Gaps 3 and 4: fresh header reads of every sha-matched copy on the 4 Volumes.
    per_path = asyncio.run(scan_all([(k, path_of[k], exp_sha[k]) for k in pop], roots, ids))
    verdict = {r["key"]: r["verdict"] for r in per_path}
    for r in per_path:  # 09's gap-5 sensitivity: overlap copies as INCOMPLETE (addendum A)
        r["verdict_gap5_alt"] = path_verdict([copy_verdict_gap5_alt(c["bits"], ids) for c in r["copies"] if c["sha_ok"]])
    verdict_alt = {r["key"]: r["verdict_gap5_alt"] for r in per_path}
    known = known_ids(ids)  # Addendum F
    ok_copies = [c for r in per_path for c in r["copies"] if c["sha_ok"]]
    diag = {
        "n_paths_without_sha_ok_copy": sum(1 for r in per_path if not any(c["sha_ok"] for c in r["copies"])),
        "n_sha_mismatch_files": sum(1 for r in per_path for c in r["copies"] if not c["sha_ok"] and not c["sha_err"]),
        "n_sha_read_errors": sum(1 for r in per_path for c in r["copies"] if not c["sha_ok"] and c["sha_err"]),
        "n_paths_multi_copy": sum(1 for r in per_path if sum(c["sha_ok"] for c in r["copies"]) > 1),
        "n_readers_disagree": sum(1 for c in ok_copies if c["bits"] != c["raw_bits"]),
        # ids seen by EITHER reader (Codex r5): an Inspect failure must not hide an unexpected id the raw read saw
        "n_copies_unexpected_model_id": sum(1 for c in ok_copies if _seen_ids(c) - known),
        "n_token_field_inconsistent": sum(1 for c in ok_copies if c["raw_bits"] and c["raw_bits"] != c["tot_bits"]),
        "n_sol_incomplete_overlap": sum(1 for c in ok_copies if c["bits"] and c["bits"].get(ids["sol"]) and not (
            c["bits"].get(ids["astra"]) and c["bits"].get(ids["policy"]))),
    }
    unexpected_ids = sorted({m for c in ok_copies for m in _seen_ids(c) - known})
    # Cross-check only (never enters the figure): production scan rows, same rule, joined by path. The row schema
    # has no read-error or header field, so L18 reduces to: missing or empty model_usage -> INCOMPLETE.
    prod: dict[str, list[dict]] = {}
    for r in scan_rows.get("rows") or []:
        prod.setdefault("/".join(r.get("parts") or []), []).append(r)
    prod_dis = []
    for k in pop:
        vs = [copy_verdict({m: has_tokens(u) for m, u in (r.get("model_usage") or {}).items()}, ids)
              for r in prod.get(path_of[k], [])]
        if path_verdict(vs) != verdict[k]:
            prod_dis.append(k)
    diag["n_prod_xcheck_disagree"] = len(prod_dis)
    return {"gates_blocking": g, "diag": diag, "unexpected_model_ids": unexpected_ids,
            "eval_model_check": eval_model_check(ok_copies, ids),  # F2
            "prod_xcheck_disagree_keys": prod_dis, "pooled": counts(pop, verdict),
            "pooled_gap5_alt": counts(pop, verdict_alt),
            "n_all_gap1_alt_all_sim_keys": len(sim),  # 09's gap-1 diagnostic: all .run_keys sim| keys
            "per_cell": {c: counts(cell_pop[c], verdict) for c in CELLS},
            "per_path": per_path, "population_keys": pop}


def compare(mine: dict, target: dict, printed: int = PRINTED_PCT, bump: int = 0) -> tuple[dict, dict]:
    """Primary: pooled counts (exact) and the 0-dp percent. Secondary: per-cell n_class1_* (gap 1 note)."""
    rec = target["a_reduce_counts"]
    rp = {k: as_int(rec["pooled_union_of_cells"].get(k)) for k in TARGET_KEYS}
    if rp["n_sol_using"] is not None:
        rp["n_sol_using"] += bump
    p = mine["pooled"]
    prim = {f"pooled.{k}": rp[k] is not None and p[k] == rp[k] for k in TARGET_KEYS}
    mine_pct = pct_half_up(p["n_sol_using"], p["n_all"])
    prim["pooled.pct_0dp_eq_printed"] = mine_pct is not None and mine_pct == printed
    rec_pct = pct_half_up(rp["n_sol_using"], rp["n_all"]) if None not in (rp["n_sol_using"], rp["n_all"]) else None
    prim["pooled.pct_0dp_eq_record"] = rec_pct is not None and mine_pct == rec_pct
    sec = {}
    for c in CELLS:
        rc = rec.get("per_cell", {}).get(c, {})
        for k in TARGET_KEYS:
            rkey = "n_class1_" + k[2:]
            v = as_int(rc.get(rkey))
            sec[f"per_cell.{c}.{rkey}"] = v is not None and mine["per_cell"][c][k] == v
    return prim, sec


def comparator_selftest(mine: dict) -> bool:
    """Negative control on real-shaped data without the record: a target built from MY OWN counts must match on every
    key, and the same target with n_sol_using + 1 must mismatch on n_sol_using. Never vacuous."""
    p = mine["pooled"]
    selft = {"a_reduce_counts": {"pooled_union_of_cells": {k: p[k] for k in TARGET_KEYS}, "per_cell": {
        c: {"n_class1_" + k[2:]: mine["per_cell"][c][k] for k in TARGET_KEYS} for c in CELLS}}}
    my_pct = pct_half_up(p["n_sol_using"], p["n_all"])
    s_prim, s_sec = compare(mine, selft, printed=my_pct)
    s_neg, _ = compare(mine, selft, printed=my_pct, bump=1)
    return my_pct is not None and all(s_prim.values()) and all(s_sec.values()) and not s_neg["pooled.n_sol_using"]


# ---------------------------------------------------------------- toy positive / negative controls
TOY_IDS = {"sol": "openai/gpt-5.6-sol", "astra": "openai/gpt-6-astra", "policy": POLICY_ID}
_A, _Q, _S = TOY_IDS["astra"], TOY_IDS["policy"], TOY_IDS["sol"]
_TE, _NA = "openai/gpt-5.6-terra", "openai/gpt-5-nano-2025-08-07"  # the two non-writers (Addendum F)
_T = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
_Z = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
# task -> list of (volume label, kind, usage); kind: eval | garbage | noheader | wrongsha | same. Expected verdict after '#'.
TOY = {
    "t01": [("chain", "eval", {_A: _T, _Q: _T, _TE: _T})],                 # FREE (terra tokens: F's F2 toy change)
    "t02": [("chain", "eval", {_A: _T, _Q: _T, _S: _T, _NA: _T})],         # SOL (nano tokens: F's F2 toy change)
    "t03": [("chain", "eval", {_A: _T, _Q: _T, _S: _Z})],                  # FREE (zero Sol tokens, L23)
    "t04": [("chain", "eval", {_Q: _T})],                                  # UNKNOWN (no simulator)
    "t05": [("pilots", "eval", {})],                                       # UNKNOWN (empty list)
    "t06": [("chain", "eval", {_Q: _T, _S: _T})],                          # SOL (precedence; overlap 1)
    "t07": [],                                                             # UNKNOWN (no copy)
    "t08": [("chain", "garbage", None)],                                   # UNKNOWN (read error)
    "t09": [("chain", "eval", {_A: _T, _Q: _T}), ("q4own", "same", None)],  # FREE (2 copies)
    "t10": [("chain", "wrongsha", {_A: _T, _Q: _T}), ("q2fresh", "eval", {_A: _T, _Q: _T, _S: _T})],  # SOL
    "t11": [("chain", "eval", {_A: _T, _Q: _T, _S: _T})],                  # no rows in any cell: excluded
    "t12": [("chain", "eval", {_A: _T, _Q: _T, "openai/gpt-5.6-sol-mini": _T})],  # FREE (exact id rule)
    "t13": [("chain", "noheader", None)],                                  # UNKNOWN (no header.json)
    "t14": [("chain", "eval", {_A: _T, _Q: _T, _S: {"input_tokens": 5, "output_tokens": 0, "total_tokens": 0}})],  # SOL
    "t15": [("chain", "eval", {_A: _T, _Q: _T})],                          # FREE, rows only in reply|L48
    "t16": [("chain", "eval", {_A: _T})],                                  # UNKNOWN (no policy)
    # Inspect cannot parse this header (no EvalSpec) but the raw reader sees an unexpected id: UNKNOWN, and it must
    # count in n_copies_unexpected_model_id (Codex r5)
    "t17": [("chain", "rawonly", {_A: _T, _Q: _T, "openai/unexpected-x": _T})],
}
TOY_ROWS_ONLY_IN = {"t11": (), "t12": ("command|L48", "reply|L63"), "t15": ("reply|L48",)}
# Hand-derived expectations (by construction, not computed by the code under test).
TOY_EXPECT_POOLED = {"n_all": 16, "n_sol_using": 4, "n_sol_free": 5, "n_unknown": 7}
TOY_EXPECT_CELLS = {"command|L48": (15, 4, 4, 7), "command|L63": (14, 4, 3, 7),
                    "reply|L48": (15, 4, 4, 7), "reply|L63": (15, 4, 4, 7)}
TOY_PRINTED_PCT = 25  # 400/16 = 25.0
TOY_EXPECT_DIAG = {"n_paths_without_sha_ok_copy": 1, "n_sha_mismatch_files": 1, "n_sha_read_errors": 0,
                   "n_paths_multi_copy": 1, "n_readers_disagree": 1, "n_copies_unexpected_model_id": 2,
                   "n_token_field_inconsistent": 1, "n_sol_incomplete_overlap": 1, "n_prod_xcheck_disagree": 0}
TOY_NEG_EXPECT = {"pooled.n_sol_using": False, "pooled.n_sol_free": False, "pooled.n_all": True,
                  "pooled.n_unknown": True, "pooled.pct_0dp_eq_printed": False, "pooled.pct_0dp_eq_record": False}
# Addendum A: gap-5 alternative (t06 -> INCOMPLETE -> UNKNOWN) and gap-1 alternative n_all (all 16 sim| keys).
TOY_EXPECT_GAP5_ALT = {"n_all": 16, "n_sol_using": 3, "n_sol_free": 5, "n_unknown": 8, "n_stop": 0}  # 300/16 -> 19%
TOY_EXPECT_GAP1_ALT_N_ALL = 17
TOY_EXPECT_UNEXPECTED_IDS = ["openai/gpt-5.6-sol-mini", "openai/unexpected-x"]
# F2 main-toy flow (hand-derived; disclosed as an extra): the read sha-matched copies are t01-t06, t09 x2, t10's
# q2fresh copy, t12, t14, t15, t16 (Inspect-written, eval.model "none/none") and t17 (raw-only header, no "eval" ->
# missing): 14 read, 1 not allowed. t08 (garbage) and t13 (no header.json) are read by neither reader; t07 has no
# copy; t10's chain copy is a sha mismatch; t11 is outside the population.
TOY_EXPECT_EM_MAIN = {"gate": False, "n_read": 14, "n_not_allowed": 1, "names": [EM_MISSING],
                      "pair": {"gt0": True, "eq_all": False}}
TOY_EXPECT_EM_NO_T17 = {"gate": True, "n_read": 13, "n_not_allowed": 0, "names": [],
                        "pair": {"gt0": False, "eq_all": False}}
# F2.2 fixture table: unit -> (the three copies, gate overrides, gate, pre_status, names, (gt0, eq_all), |R|)
_EMN, _EMU = "openai/unknown-writer", "unread"
F2_UNITS = {
    "none": (("none/none",) * 3, {}, True, (None, None), [], (False, False), 3),
    "policy": ((POLICY_ID,) * 3, {}, True, (None, None), [], (False, False), 3),
    "mixed": (("none/none", "none/none", POLICY_ID), {}, True, (None, None), [], (False, False), 3),
    "terra": ((_TE, "none/none", "none/none"), {}, False, ("STOP", "EVAL_MODEL_NOT_ALLOWED"), [_TE], (True, False), 3),
    "nano": ((_NA, "none/none", "none/none"), {}, False, ("STOP", "EVAL_MODEL_NOT_ALLOWED"), [_NA], (True, False), 3),
    "unknown": ((_EMN, "none/none", "none/none"), {}, False, ("STOP", "EVAL_MODEL_NOT_ALLOWED"), [_EMN], (True, False),
                3),
    "missing": ((_ABSENT, "none/none", "none/none"), {}, False, ("STOP", "EVAL_MODEL_NOT_ALLOWED"), [EM_MISSING],
                (True, False), 3),
    "non-string": ((7, "none/none", "none/none"), {}, False, ("STOP", "EVAL_MODEL_NOT_ALLOWED"), [EM_NONSTR],
                   (True, False), 3),
    "all terra": ((_TE,) * 3, {}, False, ("STOP", "EVAL_MODEL_NOT_ALLOWED"), [_TE], (True, True), 3),
    "unread": ((_EMU, "none/none", "none/none"), {}, True, (None, None), [], (False, False), 2),
    "order": ((_TE, "none/none", "none/none"), {"G_NO_STOP": False}, False, ("STOP", "SOL_AND_FREE_COPIES"), [_TE],
              (True, False), 3),
}


def _em_copy(v) -> dict:
    """A toy copy record for the F2 units: 'unread' -> neither reader read it; anything else through em_kind."""
    if isinstance(v, str) and v == _EMU:
        return {"eval_model_kind": None, "eval_model": None}
    k, val = em_kind(v)
    return {"eval_model_kind": k, "eval_model": val}


def _write_eval(path: str, usage: dict) -> None:
    from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSpec, EvalStats, write_eval_log
    from inspect_ai.model import ModelUsage

    os.makedirs(os.path.dirname(path), exist_ok=True)
    log = EvalLog(status="success", eval=EvalSpec(created="2026-10-20T00:00:00+00:00", task="toy",
                  dataset=EvalDataset(), model="none/none", config=EvalConfig()),
                  stats=EvalStats(model_usage={m: ModelUsage(**u) for m, u in usage.items()}))
    write_eval_log(log, path, format="eval")


# F4 control units on the main toy: (key run, path run, sha mode) -> hand-derived
# (pinned, one_copy, inspect_ok, raw_header, nano_listed, eval_model_str, err is None, ok)
F4_CONTROL_UNITS = (("t02", "t02", "manifest"), ("t02", "t02", "wrong"), ("t01", "t01", "manifest"),
                    ("t17", "t17", "manifest"), ("t13", "t13", "manifest"), ("t08", "t08", "manifest"),
                    ("t09", "t09", "manifest"), ("t02", "t01", "manifest"))
F4_CONTROL_EXPECT = (
    (True, True, True, True, True, True, True, True),          # t02: nano tokens, Inspect-written, eval.model none/none
    (False, False, False, False, False, False, True, False),   # wrong sha: no copy, nothing read
    (True, True, True, True, False, True, True, False),        # t01: terra, no nano
    (True, True, False, True, False, False, False, False),     # t17: Inspect fails, raw reads, no nano, eval missing
    (True, True, False, False, False, False, False, False),    # t13: zip without header.json
    (True, True, False, False, False, False, False, False),    # t08: not a zip
    (True, False, False, False, False, False, True, False),    # t09: two sha-matched copies
    (False, True, True, True, False, True, True, False),       # key t02 with t01's path and sha: not pinned
)


def _toy_all_unreadable() -> bool:
    """09's required F4 fixture: every sha-matched copy unreadable by both readers. The pre-F4 rules would read the
    target (every gate true, F2's gate vacuous); F4's empty-R rule must STOP as PLUMBING instead."""
    with tempfile.TemporaryDirectory() as tmp:
        roots = {"chain": f"{tmp}/chain/evals_v1", "q4own": f"{tmp}/q4own/evals_v1",
                 "pilots": f"{tmp}/pilots", "q2fresh": f"{tmp}/q2fresh"}
        run_keys, mrows = {}, []
        for t in ("u1", "u2"):
            rel = f"L/sim_attack@september/{t}/x_{t}.eval"
            run_keys[f"sim|{t}"] = rel + "#e0"
            full = os.path.join(roots["chain"], rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            Path(full).write_bytes(b"not a zip file")
            mrows.append({"path": rel, "sha256": file_sha256(full), "task": t, "arm_dir": "sim_attack@september"})
        cells = {c: {"rows_per_run": {k: 3 for k in run_keys}, "runs_without_rows": []} for c in CELLS}
        zero = {c: {"n": 0, "runs": []} for c in CELLS}
        mine = derive({"run_keys": run_keys, "cells": cells, "zero_row_runs": zero}, [("m.json", {"files": mrows})],
                      {"rows": []}, TOY_IDS, roots)
    p, alt, emc = mine["pooled"], mine["pooled_gap5_alt"], mine["eval_model_check"]
    gates = {"G_NO_STOP": p["n_stop"] == 0 and alt["n_stop"] == 0 and all(v["n_stop"] == 0
                                                                          for v in mine["per_cell"].values()),
             "G_EVAL_MODEL_NOT_JUDGE": emc["gate"],
             "G_NO_UNEXPECTED_MODEL_IDS": mine["diag"]["n_copies_unexpected_model_id"] == 0,
             "G_SUM": p["n_all"] == p["n_sol_using"] + p["n_sol_free"] + p["n_unknown"] + p["n_stop"]}
    return (p == {"n_all": 2, "n_sol_using": 0, "n_sol_free": 0, "n_unknown": 2, "n_stop": 0}
            and emc["n_read"] == 0 and emc["gate"] is True and all(gates.values())
            and pre_status(gates) == (None, None)            # the pre-F4 path would have read the target
            and pre_decision(gates, emc) == ("STOP", "PLUMBING"))


def _zstd_forced_false_control() -> bool:
    """F4 (reviewer): G_ZSTD_ZIP's predicted-FAIL control. With Inspect's multi-frame flag forced to False the self
    test must fail; the flag is restored in a finally block, and the self test must pass again afterwards."""
    had = hasattr(zipfile, "_inspect_ai_multiframe_installed")
    prev = getattr(zipfile, "_inspect_ai_multiframe_installed", None)
    try:
        zipfile._inspect_ai_multiframe_installed = False  # type: ignore[attr-defined]
        neg = zstd_selftest() is False
    finally:
        if had:
            zipfile._inspect_ai_multiframe_installed = prev  # type: ignore[attr-defined]
        else:
            del zipfile._inspect_ai_multiframe_installed  # type: ignore[attr-defined]
    return neg and zstd_selftest() is True


def _raw_em_units() -> bool:
    """F2: the raw reader's eval.model projection on hand-made headers; it never changes the reader's other fields."""
    cases = [({"eval": {"model": "none/none"}, "stats": {"model_usage": {}}}, ("str", "none/none")),
             ({"eval": {"model": 7}, "stats": {"model_usage": {}}}, ("nonstr", None)),
             ({"eval": {"model": None}}, ("nonstr", None)),
             ({"eval": {}}, ("missing", None)),
             ({"eval": "x"}, ("missing", None)),
             ({"stats": {"model_usage": {_A: _T}}}, ("missing", None))]
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for i, (hdr, exp) in enumerate(cases):
            p = f"{tmp}/h{i}.eval"
            with zipfile.ZipFile(p, "w") as z:
                z.writestr("header.json", json.dumps(hdr))
            r = raw_header(p)
            ok &= r["has_header"] is True and r["err"] is None and r["em"] == exp
        ok &= raw_header(f"{tmp}/h5.eval")["bits"] == {_A: True}  # bits unaffected by the projection
        p = f"{tmp}/nohdr.eval"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("dummy.txt", "x")
        ok &= raw_header(p)["em"] is None
    return ok


def _gate_raised(fn, name: str) -> bool:
    try:
        fn()
    except GateError as e:
        return str(e) == name
    return False


def run_toy(perturb: bool) -> dict:
    """Builds toy Volumes, manifests, refit3-like record and target; runs derive() and compare()."""
    with tempfile.TemporaryDirectory() as tmp:
        roots = {"chain": f"{tmp}/chain/evals_v1", "q4own": f"{tmp}/q4own/evals_v1",
                 "pilots": f"{tmp}/pilots", "q2fresh": f"{tmp}/q2fresh"}
        run_keys, mrows, prow = {}, [], []
        for t, copies in TOY.items():
            rel = f"L/sim_attack@september/{t}/x_{t}.eval"
            run_keys[f"sim|{t}"] = rel + "#e0"
            run_keys[f"run1|{t}"] = f"L/attack/{t}/y_{t}.eval#e0"
            good = None
            for vol, kind, usage in copies:
                full = os.path.join(roots[vol], rel)
                os.makedirs(os.path.dirname(full), exist_ok=True)
                if kind in ("eval", "wrongsha"):
                    u = dict(usage)
                    if perturb and t == "t01":
                        u[_S] = _T  # negative control: flip one Sol bit
                    _write_eval(full, u)
                elif kind == "garbage":
                    Path(full).write_bytes(b"not a zip file")
                elif kind == "noheader":
                    with zipfile.ZipFile(full, "w") as z:
                        z.writestr("dummy.txt", "x")
                elif kind == "rawonly":  # header.json without an EvalSpec: Inspect fails, the raw reader succeeds
                    with zipfile.ZipFile(full, "w") as z:
                        z.writestr("header.json", json.dumps({"stats": {"model_usage": usage}}))
                elif kind == "same":
                    Path(full).write_bytes(Path(os.path.join(roots[copies[0][0]], rel)).read_bytes())
                if kind != "wrongsha":
                    good = good or full
                    pu = copies[0][2] if kind == "same" else ({} if kind == "rawonly" else usage)
                    prow.append({"vol": vol, "parts": rel.split("/"), "model_usage": pu or {}})
            sha = file_sha256(good) if good else "0" * 64
            mrows.append({"path": rel, "sha256": sha, "task": t, "arm_dir": "sim_attack@september"})
        cells, zero = {}, {}
        for c in CELLS:
            rpr = {k: 3 for k in run_keys
                   if not (k.startswith("sim|") and c not in TOY_ROWS_ONLY_IN.get(k[4:], CELLS))}
            rwr = [k for k in run_keys if k not in rpr]
            cells[c] = {"rows_per_run": rpr, "runs_without_rows": rwr}
            zero[c] = {"n": len(rwr), "runs": [{"name": k, "key": "z", "set": "S"} for k in rwr]}
        refit3 = {"run_keys": run_keys, "cells": cells, "zero_row_runs": zero}
        t09 = next(r for r in mrows if r["task"] == "t09")
        manifests = [("m1.json", {"files": mrows}), ("m2.json", {"files": [dict(t09)]})]  # t09 listed twice
        target = {"a_reduce_counts": {"pooled_union_of_cells": dict(TOY_EXPECT_POOLED), "per_cell": {
            c: dict(zip(("n_class1_all", "n_class1_sol_using", "n_class1_sol_free", "n_class1_unknown"), v))
            for c, v in TOY_EXPECT_CELLS.items()}}}
        mine = derive(refit3, manifests, {"rows": prow}, TOY_IDS, roots)
        # Blocking-gate negatives (each must raise before any read): conflicting sha, null sha, bad zero-row list.
        conflict = manifests + [("m3.json", {"files": [dict(mrows[0], sha256="f" * 64)]})]
        nullsha = [("m1.json", {"files": [dict(r, sha256=None) if r["task"] == "t02" else r for r in mrows]})]
        nlsha = [("m1.json", {"files": [dict(r, sha256=r["sha256"] + "\n") if r["task"] == "t02" else r
                                        for r in mrows]})]
        badzero = {**refit3, "zero_row_runs": {**zero, CELLS[0]: {"n": 0, "runs": []}}}
        dupzero = {**refit3, "zero_row_runs": {**zero, CELLS[0]: {"n": 0, "runs": zero[CELLS[0]]["runs"] * 2}}}
        gate_negs = {
            "conflicting_sha": _gate_raised(lambda: derive(refit3, conflict, {}, TOY_IDS, roots), "G_MANIFEST_JOIN"),
            "null_sha": _gate_raised(lambda: derive(refit3, nullsha, {}, TOY_IDS, roots), "G_MANIFEST_JOIN"),
            "newline_sha": _gate_raised(lambda: derive(refit3, nlsha, {}, TOY_IDS, roots), "G_MANIFEST_JOIN"),
            "zero_rows": _gate_raised(lambda: derive(badzero, manifests, {}, TOY_IDS, roots), "G_CELLS_PARTITION"),
            # set semantics: a repeated name in zero_row_runs must NOT fail the partition gate
            "zero_dup_ok": derive(dupzero, manifests, {"rows": prow}, TOY_IDS, roots)["pooled"] == mine["pooled"],
        }
        # F2: Inspect's own eval.model projection on a toy file Inspect wrote (the raw reader takes precedence in derive)
        ins_em = asyncio.run(inspect_header(os.path.join(roots["chain"], "L/sim_attack@september/t01/x_t01.eval")))["em"]
        # F4: the real-copy control's decision on toy copies (key run, path run, sha mode)
        sha_of = {r["task"]: r["sha256"] for r in mrows}
        ctl_units = [control_check(refit3, manifests, roots, f"sim|{k}", f"L/sim_attack@september/{pt}/x_{pt}.eval",
                                   sha_of[pt] if mode == "manifest" else "0" * 64) for k, pt, mode in F4_CONTROL_UNITS]
    prim, sec = compare(mine, target, printed=TOY_PRINTED_PCT)
    units = {"SOL+FREE": path_verdict(["SOL", "FREE"]) == "STOP", "SOL+INC": path_verdict(["SOL", "INCOMPLETE"]) == "SOL",
             "FREE+INC": path_verdict(["FREE", "INCOMPLETE"]) == "UNKNOWN", "none": path_verdict([]) == "UNKNOWN",
             "FREE+FREE": path_verdict(["FREE", "FREE"]) == "FREE",
             "half_up_12.5": pct_half_up(1, 8) == 13, "half_up_37.5": pct_half_up(3, 8) == 38,
             "half_up_0.5": pct_half_up(1, 200) == 1, "zero_den": pct_half_up(1, 0) is None,
             "as_int": as_int(7.0) == 7 and as_int(7.5) is None and as_int(True) is None and as_int("1") is None,
             "comparator_selftest": comparator_selftest(mine), "worker_output_isolated": _worker_selftest(),
             "F3_zstd_roundtrip": zstd_selftest()}
    ids = TOY_IDS
    allg = {"G_NO_STOP": True, "G_EVAL_MODEL_NOT_JUDGE": True, "G_NO_UNEXPECTED_MODEL_IDS": True, "G_X": True}
    units.update({  # addendum A: status logic and the gap-5 copy rule
        "pre_ok": pre_status(allg) == (None, None),
        "pre_stop_solfree": pre_status({**allg, "G_NO_STOP": False}) == ("STOP", "SOL_AND_FREE_COPIES"),
        "pre_stop_ids": pre_status({**allg, "G_NO_UNEXPECTED_MODEL_IDS": False}) == ("STOP", "UNEXPECTED_MODEL_IDS"),
        "pre_stop_ids_missing": pre_status({"G_NO_STOP": True, "G_EVAL_MODEL_NOT_JUDGE": True, "G_X": True}) == (
            "STOP", "UNEXPECTED_MODEL_IDS"),
        "pre_gate_fail": pre_status({**allg, "G_X": False}) == ("GATE_FAIL", None),
        "post_pass": post_status({"a": True, "b": True}, True) == "PASS",
        "post_unresolved": post_status({"a": True, "b": True}, False) == "UNRESOLVED",
        "post_mismatch": post_status({"a": False, "b": True}, True) == "MISMATCH" == post_status({"a": False}, False),
        "gap5_rule": (copy_verdict_gap5_alt({_S: True, _Q: True}, ids) == "INCOMPLETE"
                      and copy_verdict_gap5_alt({_S: True, _A: True, _Q: True}, ids) == "SOL"
                      and copy_verdict_gap5_alt({_A: True, _Q: True}, ids) == "FREE"
                      and copy_verdict_gap5_alt(None, ids) == "INCOMPLETE"),
        # toy flows: the toy's unexpected id (t12) must STOP; its gap-5 change of the rounded pct must be UNRESOLVED
        "flow_stop_ids": pre_status({**allg, "G_NO_UNEXPECTED_MODEL_IDS":
                                     mine["diag"]["n_copies_unexpected_model_id"] == 0}) == ("STOP", "UNEXPECTED_MODEL_IDS"),
        "flow_gap5_unresolved": post_status(prim, pct_half_up(mine["pooled_gap5_alt"]["n_sol_using"],
                                                              mine["pooled_gap5_alt"]["n_all"]) == TOY_PRINTED_PCT) == "UNRESOLVED",
    })
    # Addendum E (09): the printed-id guards. Hand-derived: sorted by code point, the 3 non-id strings come first.
    many = sorted([f"openai/m{i:02d}" for i in range(12)] + ["bad id", "a" * 81, "a\nb", "z" * 80])
    r = safe_ids(many)
    units.update({
        "ids_cap_10": len(r["unexpected_model_ids"]) == 10 and r["n_unexpected_model_ids"] == 16,
        "ids_non_id_counted": (r["n_unexpected_non_id_strings"] == 3
                               and r["unexpected_model_ids"] == [NON_ID] * 3 + [f"openai/m{i:02d}" for i in range(7)]),
        "ids_len_80_ok_81_not": (safe_ids(["z" * 80])["n_unexpected_non_id_strings"] == 0
                                 and safe_ids(["z" * 81])["unexpected_model_ids"] == [NON_ID]),
        "ids_empty": safe_ids([]) == {"unexpected_model_ids": [], "n_unexpected_model_ids": 0,
                                      "n_unexpected_non_id_strings": 0},
        "ids_toy_flow": safe_ids(mine["unexpected_model_ids"]) == {
            "unexpected_model_ids": TOY_EXPECT_UNEXPECTED_IDS, "n_unexpected_model_ids": 2,
            "n_unexpected_non_id_strings": 0},
    })
    # Addendum F (its "F2. Toy changes" units): non-writers never decide a verdict and never trip the id gate
    ok_by_key = {r["key"]: [c for c in r["copies"] if c["sha_ok"]] for r in mine["per_path"]}
    unexp = {k: [bool(_seen_ids(c) - known_ids(ids)) for c in cs] for k, cs in ok_by_key.items()}
    units.update({
        "F_terra_policy_incomplete": copy_verdict({_TE: True, _Q: True}, ids) == "INCOMPLETE",
        "F_free_with_nonwriters": copy_verdict({_A: True, _Q: True, _TE: True, _NA: True}, ids) == "FREE",
        "F_sol_with_nano": copy_verdict({_A: True, _Q: True, _S: True, _NA: True}, ids) == "SOL",
        "F_flow_nonwriters_seen_not_unexpected": (_TE in _seen_ids(ok_by_key["sim|t01"][0])
                                                  and _NA in _seen_ids(ok_by_key["sim|t02"][0])
                                                  and unexp["sim|t01"] == [False] and unexp["sim|t02"] == [False]),
        "F_flow_unexpected_x_still_trips": unexp["sim|t17"] == [True],
        "F_flow_only_t12_t17_trip": sorted(k for k, v in unexp.items() if any(v)) == ["sim|t12", "sim|t17"],
    })
    # Addendum F2: G_EVAL_MODEL_NOT_JUDGE. The fixture table (F2.2), the no-key unit, both readers' projections, and
    # the main-toy flow (the t17 extra).
    for name, (vals, over, gate, ps, names, (gt0, eq_all), n_r) in F2_UNITS.items():
        chk = eval_model_check([_em_copy(v) for v in vals], ids)
        units[f"F2_{name}"] = (chk["gate"] is gate and chk["n_read"] == n_r and chk["names"] == names
                               and chk["pair"] == {"gt0": gt0, "eq_all": eq_all}
                               and pre_status({**allg, **over, "G_EVAL_MODEL_NOT_JUDGE": chk["gate"]}) == ps)
    units["F2_no_key"] = pre_status({"G_NO_STOP": True, "G_NO_UNEXPECTED_MODEL_IDS": True, "G_X": True}) == (
        "STOP", "EVAL_MODEL_NOT_ALLOWED")
    units["F2_empty_R"] = eval_model_check([], ids) == {"gate": True, "n_read": 0, "n_not_allowed": 0, "names": [],
                                                        "pair": {"gt0": False, "eq_all": True}}
    units["F2_em_kind"] = (em_kind("none/none") == ("str", "none/none") and em_kind(_ABSENT) == ("missing", None)
                           and em_kind(None) == ("nonstr", None) and em_kind(7) == ("nonstr", None))
    units["F2_raw_projection"] = _raw_em_units()
    units["F2_inspect_projection"] = ins_em == ("str", EVAL_MODEL_NONE)
    units["F2_flow_main"] = mine["eval_model_check"] == TOY_EXPECT_EM_MAIN
    units["F2_flow_without_t17"] = eval_model_check(
        [c for k, cs in ok_by_key.items() if k != "sim|t17" for c in cs], ids) == TOY_EXPECT_EM_NO_T17
    # F4 (reviewer): every label is checked against a literal, not against itself
    units["F_labels"] = (
        re.fullmatch(r"[0-9a-f]{9,40}", RESULTS_COMMIT) is not None  # placeholder filled
        and NON_WRITER_IDS == {
            "openai/gpt-5.6-terra": "simulator refusal / router judge (sim.yaml @ 8d7d6a192 L21-22) (terra); span "
                                    "placement (simulated tool calls) consistent; non-writer status rests on the config, "
                                    "not on the span data",
            "openai/gpt-5-nano-2025-08-07": "side-task scorer (nano), by data (DIAG_AMEND3/4/5 outcome A @ 0a0bae859)"}
        and NON_WRITER_LABEL == (
            "simulator refusal / router judge (sim.yaml @ 8d7d6a192 L21-22) (terra); span placement (simulated tool "
            "calls) consistent; non-writer status rests on the config, not on the span data; side-task scorer (nano), by "
            "data (DIAG_AMEND3/4/5 outcome A @ 0a0bae859); non-writer; never counted as Sol, never decide SOL / FREE / "
            "INCOMPLETE")
        and POST_HOC_LABEL == ("POST HOC, disclosed: NON_WRITER_IDS added after the 13:4x STOP (record bf8ae996) under "
                               "09's ruling (i) on the sol106 diagnostic")
        and POST_HOC_F2_LABEL == ("POST HOC, disclosed (Addendum F2): G_EVAL_MODEL_NOT_JUDGE (STOP EVAL_MODEL_NOT_ALLOWED) "
                                  "restates F1.2's gate after the diagnostic (record 1e9942e8) showed eval.model none/none "
                                  "in every read copy; allowed eval.model: none/none or the policy; 09's ruling"))
    # F4: the real-copy control on toy copies, the empty-R PLUMBING rule (09's required fixture: all copies unreadable
    # -> STOP PLUMBING, where the pre-F4 rules would have read the target), and G_ZSTD_ZIP's forced-false control
    for i, (c, e) in enumerate(zip(ctl_units, F4_CONTROL_EXPECT, strict=True)):
        units[f"F4_control_{i}"] = (tuple(c[k] for k in ("pinned", "one_copy", "inspect_ok", "raw_header", "nano_listed",
                                                         "eval_model_str")) + (c["err"] is None, c["ok"])) == e
    units["F4_all_unreadable_plumbing"] = _toy_all_unreadable()
    units["F4_empty_R_rule"] = (pre_decision({**allg}, eval_model_check([], ids)) == ("STOP", "PLUMBING")
                                and pre_decision({**allg}, mine["eval_model_check"]) == (None, None)
                                and pre_decision({**allg, "G_NO_STOP": False}, eval_model_check([], ids))
                                == ("STOP", "PLUMBING"))
    units["F4_zstd_forced_false"] = _zstd_forced_false_control()
    alt_ok = {"gap5_alt_counts": mine["pooled_gap5_alt"] == TOY_EXPECT_GAP5_ALT,
              "gap1_alt_n_all": mine["n_all_gap1_alt_all_sim_keys"] == TOY_EXPECT_GAP1_ALT_N_ALL,
              "unexpected_ids_either_reader": mine["unexpected_model_ids"] == TOY_EXPECT_UNEXPECTED_IDS}
    neg_prim, _ = compare(mine, target, printed=TOY_PRINTED_PCT, bump=1)
    diag_ok = {k: mine["diag"][k] == v for k, v in TOY_EXPECT_DIAG.items()}
    ok = (all(prim.values()) and all(sec.values()) and all(units.values()) and all(diag_ok.values())
          and all(gate_negs.values()) and all(alt_ok.values()) and not neg_prim["pooled.n_sol_using"])
    return {"pass": ok, "match": prim, "secondary_all": all(sec.values()), "units": units, "gate_negs": gate_negs,
            "alt_ok": alt_ok, "diag_as_expected": diag_ok, "neg_compare_mismatches": not neg_prim["pooled.n_sol_using"]}


# ---------------------------------------------------------------- run (worker side, then the Modal driver)
def _summary_line(d: dict) -> str:
    return "SOL106_RESULT " + json.dumps(d, sort_keys=True)


def _write_out(name: str, record: dict) -> tuple[str, str]:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = f"{OUT_DIR}/{name}"
    data = json.dumps(record, sort_keys=True, indent=1).encode()
    with open(path, "xb") as f:  # append-only: never overwrite; the driver commits the Volume after the worker exits
        f.write(data)
    return path, sha256(data)


def _drive(mode: str, blobs: dict, manifests: list, meta: dict, stage: list) -> dict:
    stage[0] = "zstd"  # Addendum F3: without explicit zstd support the raw reader cannot read any real member
    if not zstd_selftest():
        raise GateError("G_ZSTD_ZIP")
    stage[0] = "toy"
    toy_pos, toy_neg = run_toy(False), run_toy(True)
    gates = {"G_ZSTD_ZIP": True, "G_TOY_POS_PASS": toy_pos["pass"],
             "G_TOY_NEG_FAILS": (not toy_neg["pass"]) and toy_neg["match"] == TOY_NEG_EXPECT}
    if mode in ("test", "test-neg"):
        res = toy_neg if mode == "test-neg" else toy_pos
        stage[0] = "write"
        path, rsha = _write_out(f"sol106_{mode}_{meta['stamp']}.json", {"mode": mode, "meta": meta, "toy": res,
                                                                         "gates": gates})
        return {"mode": mode, "status": "PASS" if res["pass"] else "MISMATCH", "gates": gates,
                "match": res["match"], "results_path": path, "results_sha256": rsha}
    for name in ("G_TOY_POS_PASS", "G_TOY_NEG_FAILS"):
        if not gates[name]:
            raise GateError(name)
    stage[0] = "inputs"
    raw = {k: zlib.decompress(v) for k, v in blobs.items()}
    gates["G_INPUT_SHAS"] = set(raw) == set(PINS) and all(sha256(raw[k]) == PINS[k][2] for k in PINS)
    mf = [(p, zlib.decompress(b)) for p, b in manifests]
    gates["G_MANIFEST_DIGEST"] = len(mf) == MANIFEST_N and manifest_digest(mf) == MANIFEST_DIGEST
    lines = raw["sim_yaml"].decode().splitlines() if "sim_yaml" in raw else []
    gates["G_SIM_YAML_L19_L20"] = len(lines) >= 20 and lines[18] == SIM_YAML_L19 and lines[19] == SIM_YAML_L20
    roots = {lab: root for lab, (_, _, root) in VOLS.items()}
    gates["G_ROOTS_EXIST"] = all(os.path.isdir(r) for r in roots.values())
    for name in ("G_INPUT_SHAS", "G_MANIFEST_DIGEST", "G_SIM_YAML_L19_L20", "G_ROOTS_EXIST"):
        if not gates[name]:
            raise GateError(name)
    ids = {"astra": lines[18].split(":", 1)[1].strip(), "sol": lines[19].split(":", 1)[1].strip(), "policy": POLICY_ID}
    refit3, mfj = json.loads(raw["refit3"]), [(p, json.loads(b)) for p, b in mf]
    stage[0] = "control"  # Addendum F4: the pinned real copy must read through BOTH readers before any scan
    ctl = control_check(refit3, mfj, roots, CONTROL_KEY, CONTROL_PATH, CONTROL_SHA)
    gates["G_CONTROL_PINNED"] = ctl["pinned"]
    if not gates["G_CONTROL_PINNED"]:
        raise GateError("G_CONTROL_PINNED")
    if not ctl["ok"]:  # PLUMBING: the target is never read, nothing is scanned
        stage[0] = "write"
        path, rsha = _write_out(f"sol106_results_{meta['stamp']}.json", {
            "label": LABEL, "blinding_label": BLINDING_LABEL, "mode": mode, "meta": meta, "pins": PINS, "ids": ids,
            "status": "STOP", "stop_reason": "PLUMBING", "gates": gates, "control": ctl,
            "toy": {"pos_pass": toy_pos["pass"], "neg_pass": toy_neg["pass"]}})
        return {"label": LABEL, "blinding_label": BLINDING_LABEL, "mode": mode, "status": "STOP", "stop_reason": "PLUMBING",
                "gates": gates, "control": ctl, "non_writer_label": NON_WRITER_LABEL, "post_hoc_label": POST_HOC_LABEL,
                "post_hoc_f2_label": POST_HOC_F2_LABEL, "results_path": path, "results_sha256": rsha}
    stage[0] = "derive"
    mine = derive(refit3, mfj, json.loads(raw["scan_rows"]), ids, roots)
    gates.update(mine["gates_blocking"])
    stage[0] = "compare"
    p = mine["pooled"]
    alt = mine["pooled_gap5_alt"]
    gates["G_NO_STOP"] = (p["n_stop"] == 0 and alt["n_stop"] == 0
                          and all(v["n_stop"] == 0 for v in mine["per_cell"].values()))
    # F2 (restating F1.2): any read copy whose eval.model is not none/none or the policy STOPs before any verdict.
    emc = mine["eval_model_check"]
    gates["G_EVAL_MODEL_NOT_JUDGE"] = emc["gate"]
    # Addendum A (09), as amended by F: any copy with a model id outside the three expected ids and the two non-writers
    # STOPs before any verdict.
    gates["G_NO_UNEXPECTED_MODEL_IDS"] = mine["diag"]["n_copies_unexpected_model_id"] == 0
    gates["G_SUM"] = p["n_all"] == p["n_sol_using"] + p["n_sol_free"] + p["n_unknown"] + p["n_stop"]
    gates["G_COMPARATOR_SELFTEST"] = comparator_selftest(mine)
    as_pred = {k: mine["diag"][k] == v for k, v in PREDICTED.items()}
    base = {"label": LABEL, "blinding_label": BLINDING_LABEL, "mode": mode, "meta": meta, "pins": PINS, "manifest_pin": [MANIFEST_COMMIT, MANIFEST_DIR,
            MANIFEST_N, MANIFEST_DIGEST], "ids": ids, "vols": VOLS, "diag": mine["diag"], "diag_predicted": PREDICTED,
            "diag_as_predicted": as_pred, "unexpected_model_ids": mine["unexpected_model_ids"],
            "prod_xcheck_disagree_keys": mine["prod_xcheck_disagree_keys"],
            "population_keys": mine["population_keys"], "per_path": mine["per_path"],
            "non_writer_ids": NON_WRITER_IDS, "non_writer_label": NON_WRITER_LABEL, "post_hoc_label": POST_HOC_LABEL,
            "post_hoc_f2_label": POST_HOC_F2_LABEL, "eval_model_check": emc, "control": ctl}
    # F / F2: printed in both real-run summaries; names and booleans only (the counts in emc stay on the Volume)
    f_print = {"non_writer_label": NON_WRITER_LABEL, "post_hoc_label": POST_HOC_LABEL,
               "post_hoc_f2_label": POST_HOC_F2_LABEL, "eval_model_not_allowed": {"names": emc["names"]},
               "copies_eval_model_not_allowed": emc["pair"], "control": ctl}

    pre, stop_reason = pre_decision(gates, emc)  # F4: an empty R STOPs as PLUMBING first
    if pre is not None:
        # STOP / GATE_FAIL: the target is never read; no counts, percent or match fields (sections 5-6; addendum A)
        stage[0] = "write"
        path, rsha = _write_out(f"sol106_results_{meta['stamp']}.json", {
            **base, "status": pre, "stop_reason": stop_reason, "gates": gates,
            "toy": {"pos_pass": toy_pos["pass"], "neg_pass": toy_neg["pass"]}})
        return {"label": LABEL, "blinding_label": BLINDING_LABEL, "mode": mode, "status": pre, "stop_reason": stop_reason, "gates": gates,
                "diag_as_predicted": as_pred, **safe_ids(mine["unexpected_model_ids"]), **f_print,
                "results_path": path, "results_sha256": rsha}
    stage[0] = "compare"
    target = json.loads(raw["target"])  # first and only read of the target's values
    prim, sec = compare(mine, target)
    rec = target["a_reduce_counts"]
    my_pct, alt_pct = pct_half_up(p["n_sol_using"], p["n_all"]), pct_half_up(alt["n_sol_using"], alt["n_all"])
    # Addendum A diagnostics (booleans to the laptop; never change the status, except gap 5 -> UNRESOLVED):
    extra = {"diag_gap1_alt_n_all_eq_record": mine["n_all_gap1_alt_all_sim_keys"] == as_int(
                 rec["pooled_union_of_cells"].get("n_all")),
             "gap5_alt_pct_eq_printed": alt_pct is not None and alt_pct == PRINTED_PCT}
    status = post_status(prim, extra["gap5_alt_pct_eq_printed"])
    record = {**base, "status": status, "gates": gates, "toy": {"pos": toy_pos, "neg": toy_neg},
              "mine": {"pooled": p, "per_cell": mine["per_cell"], "pct_0dp": my_pct,
                       "gap5_alt": {"pooled": alt, "pct_0dp": alt_pct},
                       "gap1_alt_n_all_all_sim_keys": mine["n_all_gap1_alt_all_sim_keys"]},
              "record": {"pooled_union_of_cells": rec.get("pooled_union_of_cells"), "per_cell": rec.get("per_cell")},
              "match_primary": prim, "match_secondary": sec, "addendum_a": extra}
    stage[0] = "write"
    path, rsha = _write_out(f"sol106_results_{meta['stamp']}.json", record)
    return {"label": LABEL, "blinding_label": BLINDING_LABEL, "mode": mode, "status": status, "gates": gates, "match": prim, "match_secondary": sec,
            "addendum_a": extra, "diag_as_predicted": as_pred, **safe_ids(mine["unexpected_model_ids"]), **f_print,
            "n_keys_mismatch": sum(not v for v in prim.values()),
            "n_secondary_mismatch": sum(not v for v in sec.values()), "results_path": path, "results_sha256": rsha}


def _worker_main(td: str) -> None:
    """Isolated data worker (started by _run_worker, so its stdout/stderr are /dev/null for its whole life).
    Writes only a sanitized summary: the returned summary dict, or a gate name / exception type name with a stage."""
    import logging
    import pickle
    import warnings

    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    stage = ["start"]
    try:
        with open(f"{td}/in.pkl", "rb") as f:
            a = pickle.load(f)
        res = {"out": _drive(a["mode"], a["blobs"], a["manifests"], a["meta"], stage)}
    except GateError as e:
        res = {"err": f"gate {e} at stage {stage[0]}"}
    except Exception as e:
        res = {"err": f"{type(e).__name__} at stage {stage[0]}"}
    with open(f"{td}/summary.json", "w") as f:
        json.dump(res, f)


def _git(*args: str, cwd: Path) -> bytes:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True).stdout


if not _WORKER:  # ------------------------------------------------ Modal (driver and local entrypoint)
    import modal

    image = modal.Image.debian_slim(python_version="3.12").pip_install("inspect-ai==0.3.276")
    app = modal.App(APP_NAME)
    _vols = {mnt: modal.Volume.from_name(name).read_only() for name, mnt, _ in VOLS.values()}
    out_vol = modal.Volume.from_name(APP_NAME, create_if_missing=True)

    # F1.3 + F2.3 (accepted by 09): request = limit; bound 3 x (120 + 1800 + 60) s x $0.00003508/s = $0.2083752
    @app.function(image=image, volumes={**_vols, "/out": out_vol}, cpu=(2.0, 2.0), memory=(4096, 4096), timeout=1800,
                  startup_timeout=120, scaledown_window=60, retries=0, max_containers=1)
    def drive(mode: str, blobs: dict, manifests: list, meta: dict) -> dict:
        """Driver: never parses data. Runs the isolated worker (stdout/stderr at /dev/null), reads its sanitized
        summary, commits the output Volume. Only gate names, exception type names or an exit code leave."""
        import pickle

        res, err = None, None
        try:
            try:
                with tempfile.TemporaryDirectory() as td:
                    with open(f"{td}/in.pkl", "wb") as f:
                        pickle.dump({"mode": mode, "blobs": blobs, "manifests": manifests, "meta": meta}, f)
                    rc = _run_worker([sys.executable, os.path.abspath(__file__), "--worker", td], subprocess.DEVNULL)
                    sp = f"{td}/summary.json"
                    res = (json.loads(Path(sp).read_text()) if os.path.exists(sp)
                           else {"err": f"worker exit {rc}, no summary"})
            finally:  # reviewer: a timeout or a bad summary must not lose a results file the worker already wrote
                out_vol.commit()
        except Exception as e:
            err = f"{type(e).__name__} in driver"
        if err is None and "err" in res:
            err = str(res["err"])
        if err is not None:
            raise RuntimeError("sol106 failed: " + err)  # raised outside any except block: no chained context
        return res["out"]

    @app.local_entrypoint()
    def main(mode: str = "real"):
        from datetime import datetime
        from zoneinfo import ZoneInfo

        if mode not in ("real", "test", "test-neg"):
            raise SystemExit("mode must be real, test or test-neg")
        here = Path(__file__).resolve()
        meta = {"script_sha256": sha256(here.read_bytes()), "tz": "America/Los_Angeles",
                "stamp": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")}
        blobs, manifests = {}, []
        if mode == "real":  # git provenance required only for the real run; input bytes are never parsed or printed
            top = Path(_git("rev-parse", "--show-toplevel", cwd=here.parent).decode().strip())
            rel = here.relative_to(top).as_posix()
            blob_here = _git("hash-object", str(here), cwd=top).decode().strip()
            committed = subprocess.run(["git", "rev-parse", f"HEAD:{rel}"], cwd=top,
                                       capture_output=True).stdout.decode().strip()
            meta.update({"git_head": _git("rev-parse", "HEAD", cwd=top).decode().strip(), "script_rel": rel,
                         "script_blob": blob_here})
            if _git("status", "--porcelain", cwd=top).strip():
                raise SystemExit("refusing: worktree not clean")
            if blob_here != committed:
                raise SystemExit("refusing: script is not the committed blob at HEAD")
            for name, (commit, path, pin) in PINS.items():
                b = _git("show", f"{commit}:{path}", cwd=top)
                if sha256(b) != pin:
                    raise SystemExit(f"refusing: input sha differs: {name}")
                blobs[name] = zlib.compress(b, 6)
            ls = _git("ls-tree", "-r", "--full-tree", MANIFEST_COMMIT, "--", MANIFEST_DIR, cwd=top).decode().splitlines()
            items = []
            for line in ls:
                info, path = line.split("\t", 1)
                if (path.rsplit("/", 1)[-1].startswith("manifest_") and path.endswith(".json")
                        and "/" not in path[len(MANIFEST_DIR):]):
                    items.append((path, _git("cat-file", "blob", info.split()[2], cwd=top)))
            if len(items) != MANIFEST_N or manifest_digest(items) != MANIFEST_DIGEST:
                raise SystemExit("refusing: manifest digest differs")
            manifests = [(p, zlib.compress(b, 6)) for p, b in items]
        out = drive.remote(mode, blobs, manifests, meta)
        print(_summary_line(out))
        if out["status"] != "PASS":
            raise SystemExit(1)


if __name__ == "__main__":  # worker / noise (internal), or the local toy: no Modal call, no data
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        _worker_main(sys.argv[2])
        raise SystemExit(0)
    if len(sys.argv) == 2 and sys.argv[1] == "--noise":
        _noise_main()
    if len(sys.argv) != 2 or sys.argv[1] not in ("toy", "toy-neg"):
        raise SystemExit("usage: run_sol106.py toy|toy-neg")
    res = run_toy(sys.argv[1] == "toy-neg")
    print(_summary_line({"mode": sys.argv[1], "status": "PASS" if res["pass"] else "MISMATCH", "match": res["match"],
                         "units": res["units"], "gate_negs": res["gate_negs"], "secondary_all": res["secondary_all"],
                         "alt_ok": res["alt_ok"],
                         "diag_as_expected": res["diag_as_expected"],
                         "neg_compare_mismatches": res["neg_compare_mismatches"]}))
    raise SystemExit(0 if res["pass"] else 1)
