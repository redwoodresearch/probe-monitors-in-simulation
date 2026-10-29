"""sol106 diagnostic: where do the two unexpected model ids sit in the attack-run headers? (09's request, 13:5x PDT)

From .eval HEADERS (header.json; Inspect's header read as a cross-check). The target record is never read. For copies
with nano usage only, 09's check (b) also opens the samples/*.json members to test the PRESENCE of a
scores.*.metadata.judge_messages key; contents are never kept or printed (PRESTATED_DIAG_AMEND1). nano is the
post-episode side-task judge (a scorer); terra is the simulator refusal/router judge. The laptop gets the reduced
print set only (booleans for (a), usage-only and (c); full counts on the Volume). Same population (union over the 4 cells of sim| keys with rows), join (refit3 .run_keys path, manifest
sha) and copies (sha-matched files on the 4 Volumes) as run_sol106.py, in the same worker-isolated pattern.
Spec: PRESTATED_DIAG_<stamp>.md in this folder.

Modal (named binary, from the committed blob in the worktree):
  "$M" run diag_sol106_ids.py --mode test-neg      # toy, predicted FAIL (exit 1)
  "$M" run diag_sol106_ids.py --mode test          # toy, predicted PASS (exit 0)
  "$M" run --detach diag_sol106_ids.py --mode real # the diagnostic
Local toy only (no Modal, no data): uv run --no-project --python 3.12 --with modal==1.6.1
  --with inspect-ai==0.3.276 python diag_sol106_ids.py toy | toy-neg
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
from collections import Counter
from pathlib import Path

_WORKER = os.environ.get("SOL106_WORKER") == "1"  # the isolated data worker never imports modal

APP_NAME = "probe-shift_aswin-surya_sub-rerun_sol106diag_kill-2026-11-19"
OUT_VOL_NAME = "probe-shift_aswin-surya_sub-rerun_sol106_kill-2026-11-19"  # the item's output Volume
OUT_DIR = "/out/sol106_diag"
VOLS = {
    "chain": ("probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14", "/chain", "/chain/evals_v1"),
    "q4own": ("probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13", "/q4own", "/q4own/evals_v1"),
    "pilots": ("probe-shift_aswin-surya_sub-q3pilots_runs_kill-2026-11-13", "/pilots", "/pilots"),
    "q2fresh": ("probe-shift_aswin-surya_sub-q2-fresh_work_kill-2026-11-13", "/q2fresh", "/q2fresh"),
}
CELLS = ("command|L48", "command|L63", "reply|L48", "reply|L63")
SIM_YAML_L19 = "  PT_SIM_MODEL: openai/gpt-6-astra"
SIM_YAML_L20 = "  PT_SIM_FALLBACK_MODEL: openai/gpt-5.6-sol"
SIM_YAML_L21 = "  PT_SIM_REFUSAL_JUDGE_MODEL: openai/gpt-5.6-terra"  # terra's role cite (DIAG_AMEND1, 1a)
SIM_YAML_L22 = "  PT_SIM_ROUTER_JUDGE_MODEL: openai/gpt-5.6-terra"
POLICY_ID = "vllm/Qwen/Qwen3.5-27B"
DIAG_IDS = {"nano": "openai/gpt-5-nano-2025-08-07", "terra": "openai/gpt-5.6-terra"}  # from the 13:4x STOP (bf8ae996)
# 09 (relayed ~14:2x): terra = simulator refusal/router judge (sim.yaml @ 8d7d6a192 L21-L22); nano = post-episode
# side-task judge, a scorer (sims src/runner/arenas/sandbox.py @ e561f9365 L51, L65, L72). Neither is a writer.
SIMS_COMMIT = "e561f9365ef2f9c5f8990491927b5559ae94d473"  # 09 check (a): the sims checkout HEAD the cites were read at
TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens", "input_tokens_cache_write",
                "input_tokens_cache_read", "reasoning_tokens")
SHA_RE = re.compile(r"[0-9a-f]{64}")
ID_RE = re.compile(r"[A-Za-z0-9._/:-]{1,80}")  # printed strings (ids, role names) must fully match, else NON_ID
NON_ID = "<non-id string>"
MAX_PRINTED = 10
PINS = {
    "refit3": ("80d75b27e", "experiments/2026-10-16_attack_refit3/results/attack_all_refit3_2026-10-16_141813.json",
               "df7171a98d82d28ae86653480be922d978fc36104c63f622b3a69fb43a074eac"),
    "sim_yaml": ("8d7d6a192", "experiments/2026-10-07_phase2_attack_check/config/sim.yaml",
                 "03340860a688a04ea6cb017b46067e037748f5864108b79a4ff84d02245f4612"),
}
MANIFEST_COMMIT = "1df288e61"
MANIFEST_DIR = "experiments/2026-10-14_q3_overnight_chain/config/"
MANIFEST_N = 293
MANIFEST_DIGEST = "bce4051be4bf26b9bb855b300604732799df88fc8c03c86f23dcbbb0c7425ab1"


class GateError(RuntimeError):
    """Raised with a gate name only."""


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def manifest_digest(items: list[tuple[str, bytes]]) -> str:
    return sha256("".join(f"{p}\t{sha256(b)}\n" for p, b in sorted(items)).encode())


def has_tokens(u) -> bool:
    """Any token count > 0 (null = 0). Strict: a non-object entry or a non-numeric count raises (-> unreadable)."""
    if not isinstance(u, dict):
        raise TypeError("usage entry is not an object")
    vals = [u.get(f) for f in TOKEN_FIELDS]
    if any(v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))) for v in vals):
        raise TypeError("non-numeric token count")
    return any((v or 0) > 0 for v in vals)


def as_int(x) -> int | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return int(x) if float(x).is_integer() else None


def guard(s) -> str:
    return s if isinstance(s, str) and ID_RE.fullmatch(s) else NON_ID


def guard_counts(c: Counter) -> dict:
    """Printed {string: count}: guarded strings, at most MAX_PRINTED entries (largest first), plus totals."""
    g = Counter()
    for k, v in c.items():
        g[guard(k)] += v
    top = sorted(g.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_PRINTED]
    return {"items": dict(top), "n_distinct": len(c), "n_non_id": sum(1 for k in c if guard(k) == NON_ID)}


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _roles_map(mr) -> dict[str, list[str]]:
    """eval.model_roles -> {role: sorted model ids}; ModelConfig dicts or lists of them; only the 'model' field.
    A non-object model_roles raises (-> unreadable)."""
    if mr is not None and not isinstance(mr, dict):
        raise TypeError("model_roles is not an object")
    out = {}
    for r, cfg in (mr or {}).items():
        cfgs = cfg if isinstance(cfg, list) else [cfg]
        out[str(r)] = sorted({c.get("model") for c in cfgs if isinstance(c, dict) and isinstance(c.get("model"), str)})
    return out


def _revision_fields(commit, origin, packages) -> dict:
    """09's check (a) inputs, projected only: the recorded git commit, the repo NAME of the recorded origin (basename
    without .git; never the full URL), and the recorded control-tower package version string."""
    if packages is not None and not isinstance(packages, dict):
        raise TypeError("packages is not an object")
    repo = None
    if isinstance(origin, str) and origin:
        repo = origin.rstrip("/").rsplit("/", 1)[-1].rsplit(":", 1)[-1].removesuffix(".git")
    ct = (packages or {}).get("control-tower")  # the specified key only (Codex r12 #2)
    ct = ct if isinstance(ct, str) else None
    return {"revision_commit": commit if isinstance(commit, str) else None, "revision_repo": repo,
            "control_tower_version": ct}


def judge_messages_presence(path: str) -> dict:
    """09's check (b), presence only: for each samples/*.json member, does some score carry metadata.judge_messages?
    The member is parsed in the worker's memory; only one boolean per sample is kept; contents are never kept or
    printed. Any read/parse/type error -> err (type name only)."""
    try:
        n, n_with = 0, 0
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():  # every entry, read directly: duplicate names cannot hide one (Codex r12 #3)
                if not (info.filename.startswith("samples/") and info.filename.endswith(".json")):
                    continue
                s = json.loads(z.read(info))
                scores = s.get("scores") or {}
                if not isinstance(scores, dict):
                    raise TypeError("scores is not an object")
                n += 1
                n_with += any(isinstance(v, dict) and isinstance(v.get("metadata"), dict)
                              and "judge_messages" in v["metadata"] for v in scores.values())
                del s, scores
        return {"n_samples": n, "n_with_key": n_with, "err": None}
    except Exception as e:
        return {"n_samples": None, "n_with_key": None, "err": type(e).__name__}


def raw_header(path: str) -> dict:
    """Primary for this diagnostic: the header.json member only. It is parsed in memory (as any JSON parse must); only
    eval.model, model_roles[*].model, model_usage token bits and role_usage token bits are projected, inside the
    error boundary, and the parsed object is dropped at once. Any read/parse/projection error -> not ok (type only)."""
    try:
        with zipfile.ZipFile(path) as z:
            if "header.json" not in z.namelist():
                return {"ok": False, "has_header": False, "err": None}
            h = json.loads(z.read("header.json"))
        ev, st = h.get("eval") or {}, h.get("stats") or {}
        em = ev.get("model")
        rev = ev.get("revision") or {}
        if not isinstance(rev, dict):
            raise TypeError("revision is not an object")
        out = {"ok": True, "has_header": True, "err": None, "eval_model": em if isinstance(em, str) else None,
               "roles": _roles_map(ev.get("model_roles")),
               "bits": {str(m): has_tokens(u) for m, u in (st.get("model_usage") or {}).items()},
               "role_bits": {str(r): has_tokens(u) for r, u in (st.get("role_usage") or {}).items()},
               **_revision_fields(rev.get("commit"), rev.get("origin"), ev.get("packages"))}
        del h, ev, st, rev
        return out
    except Exception as e:
        return {"ok": False, "has_header": None, "err": type(e).__name__}


async def inspect_header(path: str) -> dict:
    """Cross-check: Inspect's header_only read; projects .model of each role config (no model_dump of configs) and the
    token fields only, inside the error boundary."""
    from inspect_ai.log import read_eval_log_async

    try:
        log = await read_eval_log_async(path, header_only=True)
        roles = {}
        for r, v in (log.eval.model_roles or {}).items():
            cfgs = v if isinstance(v, list) else [v]
            roles[str(r)] = sorted({c.model for c in cfgs if isinstance(getattr(c, "model", None), str)})
        st, tf, rev = log.stats, set(TOKEN_FIELDS), log.eval.revision
        out = {"ok": True, "err": None, "eval_model": log.eval.model, "roles": roles,
               "bits": {str(m): has_tokens(u.model_dump(include=tf)) for m, u in ((st.model_usage if st else None)
                                                                                  or {}).items()},
               "role_bits": {str(r): has_tokens(u.model_dump(include=tf)) for r, u in ((getattr(st, "role_usage", None)
                                                                                          if st else None) or {}).items()},
               **_revision_fields(getattr(rev, "commit", None), getattr(rev, "origin", None), log.eval.packages)}
        del log, st, rev
        return out
    except Exception as e:
        return {"ok": False, "err": type(e).__name__}


async def scan_path(key, path, exp_sha, roots, sem) -> dict:
    copies = []
    for vol, root in roots.items():
        full = os.path.join(root, path)
        async with sem:
            if not await asyncio.to_thread(os.path.isfile, full):
                continue
            try:
                got = await asyncio.to_thread(file_sha256, full)
            except Exception as e:
                copies.append({"vol": vol, "sha_ok": False, "sha_err": type(e).__name__})
                continue
            if got != exp_sha:
                copies.append({"vol": vol, "sha_ok": False, "sha_err": None})
                continue
            raw = await asyncio.to_thread(raw_header, full)
            ins = await inspect_header(full)
            nano_seen = DIAG_IDS["nano"] in (set(raw.get("bits") or {}) | set(ins.get("bits") or {}))
            jm = await asyncio.to_thread(judge_messages_presence, full) if nano_seen else None
        copies.append({"vol": vol, "sha_ok": True, "raw": raw, "ins": ins, "jm": jm})
    return {"key": key, "path": path, "copies": copies}


async def scan_all(jobs, roots, conc=32):
    sem = asyncio.Semaphore(conc)
    return await asyncio.gather(*(scan_path(k, p, s, roots, sem) for k, p, s in jobs))


def population_and_join(refit3: dict, manifests: list[tuple[str, dict]]):
    """Exactly run_sol106's gap-1 population and gap-2 join, with its blocking gates."""
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
    cell_pop = {c: sorted(k for k in refit3["cells"][c]["rows_per_run"] if k in sim) for c in CELLS}
    pop = sorted(set().union(*cell_pop.values()))
    path_of = {k: sim[k].rsplit("#", 1)[0] for k in pop}
    g["G_PATHS_INJECTIVE"] = len(set(path_of.values())) == len(pop)
    by_path: dict = {}
    for _, m in manifests:
        for r in ((m.get("files") or []) if isinstance(m, dict) else []):
            if isinstance(r, dict) and str(r.get("arm_dir") or "").startswith("sim_attack"):
                by_path.setdefault(r.get("path"), []).append(r)
    exp_sha, ok_all = {}, True
    for k in pop:
        rs = by_path.get(path_of[k], [])
        shas = {r.get("sha256") for r in rs}
        ok = (bool(rs) and len(shas) == 1 and all(isinstance(s, str) and SHA_RE.fullmatch(s) for s in shas)
              and all(r.get("task") == k.split("|", 1)[1] for r in rs))
        ok_all &= ok
        if ok:
            exp_sha[k] = next(iter(shas))
    g["G_MANIFEST_JOIN"] = ok_all
    for name, v in g.items():
        if not v:
            raise GateError(name)
    return pop, cell_pop, path_of, exp_sha, len(sim)


def _hdr(raw: dict, ins: dict) -> dict | None:
    """The header view used for a copy: the raw reader if it read the header, else Inspect, else None (unreadable)."""
    return raw if raw.get("ok") else (ins if ins.get("ok") else None)


def diagnose(refit3: dict, manifests: list, ids: dict, roots: dict) -> dict:
    """09's (a)-(d) per diagnostic id, plus context counts and the pre-stated decision booleans."""
    pop, cell_pop, path_of, exp_sha, _ = population_and_join(refit3, manifests)
    per_path = asyncio.run(scan_all([(k, path_of[k], exp_sha[k]) for k in pop], roots))
    known = {ids["sol"], ids["astra"], ids["policy"]}
    diag_ids = set(DIAG_IDS.values())
    rows = []  # one per sha-ok copy: (key, hdr view or None, raw, ins, ids present in model_usage by either reader)
    for r in per_path:
        for c in r["copies"]:
            if c["sha_ok"]:
                raw, ins = c["raw"], c["ins"]
                seen = set(raw.get("bits") or {}) | set(ins.get("bits") or {})
                rows.append((r["key"], _hdr(raw, ins), raw, ins, seen))
    readable = [(k, h, s) for k, h, _, _, s in rows if h is not None]
    tok = lambda h, m: bool(h["bits"].get(m))  # noqa: E731
    in_role = lambda h, m: any(m in ms for ms in h["roles"].values())  # noqa: E731
    readable_keys = {k for k, _, _ in readable}
    out = {"per_id": {}, "n_readable_copies": len(readable), "n_readable_paths": len(readable_keys),
           "n_readable_paths_per_cell": {c: sum(1 for k in cell_pop[c] if k in readable_keys) for c in CELLS}}
    for name, mid in DIAG_IDS.items():
        pres = [(k, h) for k, h, s in readable if mid in s]  # present = a stats.model_usage key
        keys_pres = {k for k, _ in pres}
        declared = [h for _, h, _ in readable if in_role(h, mid)]  # over ALL readable copies (Codex r10 #1)
        out["per_id"][name] = {
            "id": mid,
            "a_n_copies_present": len(pres),
            "a_n_copies_tokens": sum(1 for _, h in pres if tok(h, mid)),
            "a_n_paths_present": len(keys_pres),
            "a_n_paths_present_per_cell": {c: sum(1 for k in cell_pop[c] if k in keys_pres) for c in CELLS},
            "b_n_copies_eval_model": sum(1 for _, h, _ in readable if h["eval_model"] == mid),
            "b_n_copies_declared_role": len(declared),
            "b_role_names": guard_counts(Counter(rn for h in declared for rn, ms in h["roles"].items() if mid in ms)),
            "b_n_copies_usage_only": sum(1 for _, h in pres if h["eval_model"] != mid and not in_role(h, mid)),
            "b_n_copies_declared_role_usage_tokens": sum(1 for h in declared if any(
                h["role_bits"].get(rn) for rn, ms in h["roles"].items() if mid in ms)),
            "d_n_copies_no_astra_no_sol": sum(1 for _, h in pres if not tok(h, ids["astra"]) and not tok(h, ids["sol"])),
            # DIAG_AMEND1 1b: a diag id present in a copy where the policy has 0 tokens
            "d2_n_copies_no_policy_tokens": sum(1 for _, h in pres if not tok(h, ids["policy"])),
        }
    t = DIAG_IDS["terra"]
    out["c_terra_x_sol_copies"] = {
        "terra_and_sol": sum(1 for _, h, s in readable if t in s and tok(h, ids["sol"])),
        "terra_no_sol": sum(1 for _, h, s in readable if t in s and not tok(h, ids["sol"])),
        "no_terra_sol": sum(1 for _, h, s in readable if t not in s and tok(h, ids["sol"])),
        "no_terra_no_sol": sum(1 for _, h, s in readable if t not in s and not tok(h, ids["sol"])),
        "unreadable": sum(1 for _, h, _, _, _ in rows if h is None)}
    out["d_n_copies_any_diag_id_no_astra_no_sol"] = sum(
        1 for _, h, s in readable if s & diag_ids and not tok(h, ids["astra"]) and not tok(h, ids["sol"]))
    other = Counter(m for _, _, _, _, s in rows for m in s - known - diag_ids)
    fields = ("eval_model", "roles", "bits", "role_bits", "revision_commit", "revision_repo", "control_tower_version")
    out["ctx"] = {
        "n_paths_without_sha_ok_copy": sum(1 for r in per_path if not any(c["sha_ok"] for c in r["copies"])),
        "n_paths_multi_copy": sum(1 for r in per_path if sum(c["sha_ok"] for c in r["copies"]) > 1),
        "n_copies_raw_not_ok": sum(1 for _, _, raw, _, _ in rows if not raw.get("ok")),
        "n_copies_inspect_not_ok": sum(1 for _, _, _, ins, _ in rows if not ins.get("ok")),
        "n_copies_readers_disagree": sum(1 for _, _, raw, ins, _ in rows if raw.get("ok") and ins.get("ok")
                                         and any(raw.get(f) != ins.get(f) for f in fields)),  # incl. role bits (#2)
        # over readable copies only (Codex r10 #5): an unreadable header is counted above, not as "not policy"
        "n_copies_eval_model_not_policy": sum(1 for _, h, _ in readable if h["eval_model"] != ids["policy"]),
        "eval_model_not_policy": guard_counts(Counter(h["eval_model"] for _, h, _ in readable
                                                      if h["eval_model"] != ids["policy"])),
        "role_names_all": guard_counts(Counter(rn for _, h, _ in readable for rn in h["roles"])),
        "other_unexpected_ids": guard_counts(other),
    }
    pi = out["per_id"]
    dec = {"D_eval_model_is_diag_id": any(v["b_n_copies_eval_model"] > 0 for v in pi.values()),
           "D_diag_id_declared_in_model_roles": any(v["b_n_copies_declared_role"] > 0 for v in pi.values()),
           "D_diag_id_without_astra_and_sol": out["d_n_copies_any_diag_id_no_astra_no_sol"] > 0,
           "D_other_unexpected_ids": len(other) > 0,
           "D_readers_disagree_or_unreadable": (out["ctx"]["n_copies_readers_disagree"] > 0
                                                or out["ctx"]["n_copies_raw_not_ok"] > 0
                                                or out["ctx"]["n_copies_inspect_not_ok"] > 0),
           # DIAG_AMEND1 1b: two more (iii) triggers
           "D_diag_id_without_policy_tokens": any(v["d2_n_copies_no_policy_tokens"] > 0 for v in pi.values()),
           "D_eval_model_not_policy": out["ctx"]["n_copies_eval_model_not_policy"] > 0}
    # Pre-stated mapping (PRESTATED_DIAG section 5, as amended by DIAG_AMEND1): (ii) if a diag id is ever eval.model;
    # "(i)-eligible" only if every D_* is false: NECESSARY, not sufficient for (i), which also needs 09's role cites;
    # anything else goes to 09 as (iii). 09 rules; this is a hint, not a decision.
    dec["hint"] = ("(ii)" if dec["D_eval_model_is_diag_id"]
                   else ("(i)-eligible" if not any(dec.values()) else "(iii)"))
    out["decision_booleans"] = dec
    # 09's checks (a) and (b) (DIAG_AMEND1): the run STOPs if either fails
    nano_copies = [c for r in per_path for c in r["copies"] if c["sha_ok"] and c.get("jm") is not None]
    jm_ok = [c for c in nano_copies if c["jm"]["err"] is None and c["jm"]["n_samples"] > 0
             and c["jm"]["n_with_key"] == c["jm"]["n_samples"]]
    ck = {"n_copies_revision_eq_sims": sum(1 for _, h, _ in readable if h.get("revision_commit") == SIMS_COMMIT),
          "n_copies_revision_missing": sum(1 for _, h, _ in readable if not h.get("revision_commit")),
          "recorded_commits": guard_counts(Counter(h["revision_commit"] for _, h, _ in readable
                                                   if h.get("revision_commit"))),
          "revision_repos": guard_counts(Counter(h["revision_repo"] for _, h, _ in readable if h.get("revision_repo"))),
          "control_tower_versions": guard_counts(Counter(h["control_tower_version"] for _, h, _ in readable
                                                         if h.get("control_tower_version"))),
          "n_copies_nano_present": len(nano_copies),
          "n_copies_nano_judge_messages_ok": len(jm_ok),
          "n_copies_nano_no_samples": sum(1 for c in nano_copies if c["jm"]["err"] is None and c["jm"]["n_samples"] == 0),
          "n_copies_nano_sample_read_error": sum(1 for c in nano_copies if c["jm"]["err"] is not None)}
    ck["G_SIMS_COMMIT"] = len(readable) > 0 and ck["n_copies_revision_eq_sims"] == len(readable)
    ck["G_JUDGE_MESSAGES"] = ck["n_copies_nano_judge_messages_ok"] == ck["n_copies_nano_present"]
    out["checks_09"] = ck
    out["_per_path"] = per_path  # Volume record only
    return out


def diag_status(ck: dict) -> tuple[str, list]:
    """Real-mode status: DONE only if both of 09's checks pass; else STOP with the failing check names."""
    reasons = [n for n, g in (("SIMS_COMMIT", "G_SIMS_COMMIT"), ("JUDGE_MESSAGES", "G_JUDGE_MESSAGES")) if not ck.get(g)]
    return ("STOP" if reasons else "DONE"), reasons


def _pair(n: int, total: int) -> dict:
    return {"gt0": n > 0, "eq_all": n == total}


def _names(g: dict) -> dict:
    """A guard_counts field for the laptop: guarded NAMES only; its counts stay on the Volume."""
    return {"names": sorted(g["items"])}


def print_set(out: dict) -> dict:
    """DIAG_AMEND1 2 + DIAG_AMEND2 (blinding): the ONLY data that reaches the laptop. Every count is a {gt0, eq_all}
    pair (eq_all: equal to all readable copies; paths and per-cell paths against readable paths); every guard_counts
    field is names only; the decision booleans and the gates as given. No integer is printed. Full counts: Volume."""
    nr, nrp, nrc = out["n_readable_copies"], out["n_readable_paths"], out["n_readable_paths_per_cell"]
    per = {}
    for name, v in out["per_id"].items():
        per[name] = {
            "id": v["id"],
            "a_copies_present": _pair(v["a_n_copies_present"], nr),
            "a_copies_tokens": _pair(v["a_n_copies_tokens"], nr),
            "a_paths_present": _pair(v["a_n_paths_present"], nrp),
            "a_paths_present_per_cell": {c: _pair(v["a_n_paths_present_per_cell"][c], nrc[c]) for c in CELLS},
            "b_eval_model": _pair(v["b_n_copies_eval_model"], nr),
            "b_declared_role": _pair(v["b_n_copies_declared_role"], nr),
            "b_role_names": _names(v["b_role_names"]),
            "b_declared_role_usage_tokens": _pair(v["b_n_copies_declared_role_usage_tokens"], nr),
            "b_usage_only": _pair(v["b_n_copies_usage_only"], nr),
            "d_no_astra_no_sol": _pair(v["d_n_copies_no_astra_no_sol"], nr),
            "d2_no_policy_tokens": _pair(v["d2_n_copies_no_policy_tokens"], nr),
        }
    c = out["c_terra_x_sol_copies"]
    # path counts against readable paths, copy counts against readable copies (Codex r15)
    ctx = {k: (_names(v) if isinstance(v, dict) else _pair(v, nrp if k.startswith("n_paths_") else nr))
           for k, v in out["ctx"].items()}
    return {"per_id": per,
            "c_terra_x_sol": {k: _pair(c[k], nr) for k in ("terra_and_sol", "terra_no_sol", "no_terra_sol",
                                                           "no_terra_no_sol")},
            "c_unreadable": _pair(c["unreadable"], nr + c["unreadable"]),  # against all sha-matched copies (r16)
            "d_any_diag_id_no_astra_no_sol": _pair(out["d_n_copies_any_diag_id_no_astra_no_sol"], nr),
            "ctx": ctx, "decision_booleans": out["decision_booleans"],
            "checks_09": _print_checks_09(out["checks_09"], nr)}


def _print_checks_09(ck: dict, nr: int) -> dict:
    """Checks (a)/(b) for the laptop: gates, {gt0, eq_all} pairs, guarded NAMES (no counts). No integer."""
    return {"G_SIMS_COMMIT": ck["G_SIMS_COMMIT"], "G_JUDGE_MESSAGES": ck["G_JUDGE_MESSAGES"],
            "revision_eq_sims": _pair(ck["n_copies_revision_eq_sims"], nr),
            "revision_missing": _pair(ck["n_copies_revision_missing"], nr),
            "recorded_commits": _names(ck["recorded_commits"]), "revision_repos": _names(ck["revision_repos"]),
            "control_tower_versions": _names(ck["control_tower_versions"]),
            "nano_present": _pair(ck["n_copies_nano_present"], nr),
            "nano_judge_messages_ok": _pair(ck["n_copies_nano_judge_messages_ok"], ck["n_copies_nano_present"]),
            "nano_no_samples": _pair(ck["n_copies_nano_no_samples"], nr),
            "nano_sample_read_error": _pair(ck["n_copies_nano_sample_read_error"], nr)}


def _int_leaves(x) -> int:
    """Count integer or float leaves in a printed structure (bools are not counted). Used by the toy unit."""
    if isinstance(x, dict):
        return sum(_int_leaves(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return sum(_int_leaves(v) for v in x)
    return int(isinstance(x, (int, float)) and not isinstance(x, bool))


# ---------------------------------------------------------------- toy (hand-derived expectations)
_A, _Q, _S = "openai/gpt-6-astra", POLICY_ID, "openai/gpt-5.6-sol"
_N, _R = DIAG_IDS["nano"], DIAG_IDS["terra"]
TOY_IDS = {"astra": _A, "sol": _S, "policy": _Q}
_T = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
# run -> (eval.model, model_usage ids, model_roles {role: id}, role_usage roles with tokens, cells with rows | None=all,
#         copy kind: "eval" | "none" | "malformed")
TOY = {
    "d01": (_Q, [_A, _Q, _R], {}, [], None, "eval"),
    "d02": (_Q, [_A, _Q, _S, _R], {}, [], None, "eval"),
    "d03": (_Q, [_A, _Q, _N], {"scorer": _N}, ["scorer"], None, "eval", {"jm": True}),  # nano: side-task scorer
    "d04": (_Q, [_Q, _R], {}, [], ("command|L63",), "eval"),
    "d05": (_R, [_A, _R], {}, [], None, "eval"),
    "d06": (_Q, [_A, _Q], {}, [], None, "none"),
    "d07": (_Q, [_A, _Q, _S], {}, [], ("reply|L48",), "eval"),
    "d08": (_Q, [_A, _Q, _N, _R], {}, [], (), "eval"),  # no rows in any cell: excluded
    "d09": (_N, [_A, _Q, _S], {}, [], None, "eval"),  # nano is eval.model but NOT in model_usage (Codex r10 #1)
    "d10": (_Q, [_A, _Q], {"judge": _R}, ["judge"], None, "eval"),  # terra declared, NOT in model_usage (#1)
    "d11": (_Q, [], {}, [], None, "malformed"),  # model_usage is a list: both readers fail (#4, #5)
}
TOY_EXPECT = {
    "nano": {"a_n_copies_present": 1, "a_n_copies_tokens": 1, "a_n_paths_present": 1,
             "a_n_paths_present_per_cell": {"command|L48": 1, "command|L63": 1, "reply|L48": 1, "reply|L63": 1},
             "b_n_copies_eval_model": 1, "b_n_copies_declared_role": 1,
             "b_role_names": {"items": {"scorer": 1}, "n_distinct": 1, "n_non_id": 0},
             "b_n_copies_usage_only": 0, "b_n_copies_declared_role_usage_tokens": 1, "d_n_copies_no_astra_no_sol": 0,
             "d2_n_copies_no_policy_tokens": 0},
    "terra": {"a_n_copies_present": 4, "a_n_copies_tokens": 4, "a_n_paths_present": 4,
              "a_n_paths_present_per_cell": {"command|L48": 3, "command|L63": 4, "reply|L48": 3, "reply|L63": 3},
              "b_n_copies_eval_model": 1, "b_n_copies_declared_role": 1,
              "b_role_names": {"items": {"judge": 1}, "n_distinct": 1, "n_non_id": 0},
              "b_n_copies_usage_only": 3, "b_n_copies_declared_role_usage_tokens": 1, "d_n_copies_no_astra_no_sol": 1,
              "d2_n_copies_no_policy_tokens": 1},  # d05: terra present, no policy tokens
}
# readable = sha-ok copies some reader read: d01-d05, d07, d09, d10 (d06 no copy, d08 excluded, d11 unreadable)
TOY_EXPECT_READABLE = {"n_readable_copies": 8, "n_readable_paths": 8,
                       "n_readable_paths_per_cell": {"command|L48": 6, "command|L63": 7, "reply|L48": 7, "reply|L63": 6}}
# DIAG_AMEND1 1b: extra runs used only in the hint subsets below (the main TOY above is unchanged)
TOY_EXTRA = {
    "d12": (_Q, [_A, _Q, _N, _R], {}, [], None, "eval", {"jm": True}),  # nano and terra usage-only, with simulator and policy
    "d13": (_Q, [_A, _S, _R], {}, [], None, "eval"),  # terra present, policy 0 tokens
    "d14": ("vllm/Other/Model-1", [_A, _Q], {}, [], None, "eval"),  # eval.model is neither the policy nor a diag id
    "d15": (_Q, [_A, _Q, _N], {}, [], None, "eval", {"jm": False}),  # nano present, sample WITHOUT judge_messages
    "d16": (_Q, [_A, _Q], {}, [], None, "eval", {"copy2": True}),  # a second identical copy on q4own: 2 copies, 1 path
}
_OTHER_COMMIT = "f" * 40
_F = {"D_eval_model_is_diag_id": False, "D_diag_id_declared_in_model_roles": False,
      "D_diag_id_without_astra_and_sol": False, "D_other_unexpected_ids": False,
      "D_readers_disagree_or_unreadable": False, "D_diag_id_without_policy_tokens": False,
      "D_eval_model_not_policy": False}
TOY_SUBSETS = {  # name -> (runs, hand-derived decision booleans, recorded commit for every copy, expected 09 checks)
    "i_eligible": (("d01", "d02", "d12"), {**_F, "hint": "(i)-eligible"}, SIMS_COMMIT,
                   {"G_SIMS_COMMIT": True, "G_JUDGE_MESSAGES": True, "status": "DONE"}),
    "iii_no_policy": (("d01", "d13"), {**_F, "D_diag_id_without_policy_tokens": True, "hint": "(iii)"}, None, None),
    "iii_eval_model": (("d01", "d14"), {**_F, "D_eval_model_not_policy": True, "hint": "(iii)"}, None, None),
    "iii_no_astra_no_sol": (("d01", "d04"), {**_F, "D_diag_id_without_astra_and_sol": True, "hint": "(iii)"}, None,
                            None),
    # 09's checks: a nano copy whose sample lacks judge_messages -> STOP; a different recorded commit -> STOP
    "stop_judge_messages": (("d01", "d15"), {**_F, "hint": "(i)-eligible"}, SIMS_COMMIT,
                            {"G_SIMS_COMMIT": True, "G_JUDGE_MESSAGES": False, "status": "STOP"}),
    "stop_sims_commit": (("d01", "d12"), {**_F, "hint": "(i)-eligible"}, _OTHER_COMMIT,
                         {"G_SIMS_COMMIT": False, "G_JUDGE_MESSAGES": True, "status": "STOP",
                          "recorded_commits": [_OTHER_COMMIT]}),
}
TOY_EXPECT_C = {"terra_and_sol": 1, "terra_no_sol": 3, "no_terra_sol": 2, "no_terra_no_sol": 2, "unreadable": 1}
TOY_EXPECT_D_ANY = 1
TOY_EXPECT_CTX = {"n_paths_without_sha_ok_copy": 1, "n_paths_multi_copy": 0, "n_copies_raw_not_ok": 1,
                  "n_copies_inspect_not_ok": 1, "n_copies_readers_disagree": 0, "n_copies_eval_model_not_policy": 2,
                  "eval_model_not_policy": {"items": {_R: 1, _N: 1}, "n_distinct": 2, "n_non_id": 0},
                  "role_names_all": {"items": {"scorer": 1, "judge": 1}, "n_distinct": 2, "n_non_id": 0},
                  "other_unexpected_ids": {"items": {}, "n_distinct": 0, "n_non_id": 0}}
TOY_EXPECT_DEC = {"D_eval_model_is_diag_id": True, "D_diag_id_declared_in_model_roles": True,
                  "D_diag_id_without_astra_and_sol": True, "D_other_unexpected_ids": False,
                  "D_readers_disagree_or_unreadable": True, "D_diag_id_without_policy_tokens": True,
                  "D_eval_model_not_policy": True, "hint": "(ii)"}


def _write_eval(path: str, eval_model: str, ids: list, roles: dict, role_usage: list, rev: str | None = None,
                jm: bool | None = None) -> None:
    from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalRevision, EvalSample, EvalSpec, EvalStats, \
        write_eval_log
    from inspect_ai.model import ModelConfig, ModelUsage
    from inspect_ai.scorer import Score

    os.makedirs(os.path.dirname(path), exist_ok=True)
    samples = None
    if jm is not None:  # one sample whose side-task score does / does not carry metadata.judge_messages
        samples = [EvalSample(id=1, epoch=1, input="x", target="", scores={
            "side_task": Score(value=1, metadata={"judge_messages": []} if jm else {"other": 1})})]
    log = EvalLog(status="success", eval=EvalSpec(
        created="2026-10-20T00:00:00+00:00", task="toy", dataset=EvalDataset(), model=eval_model, config=EvalConfig(),
        model_roles={r: ModelConfig(model=m) for r, m in roles.items()} or None,
        revision=EvalRevision(type="git", origin="https://github.com/redwoodresearch/simulated-control-evals.git",
                              commit=rev) if rev else None),
        stats=EvalStats(model_usage={m: ModelUsage(**_T) for m in ids},
                        role_usage={r: ModelUsage(**_T) for r in role_usage}), samples=samples)
    write_eval_log(log, path, format="eval")


def _run_worker(cmd: list[str], sink, timeout: int = 1700) -> int:
    env = {**os.environ, "SOL106_WORKER": "1"}
    return subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=sink, stderr=sink, env=env, check=False,
                          timeout=timeout).returncode


def _noise_main() -> None:
    print("s1", end="")
    sys.stderr.write("s2")
    os.write(1, b" s3\n")
    os.write(2, b" s4\n")
    subprocess.run(["sh", "-c", "printf ' s5'; printf ' s6' 1>&2"], check=False)
    raise SystemExit(3)


def _worker_selftest() -> bool:
    with tempfile.TemporaryFile() as sink:
        rc = _run_worker([sys.executable, os.path.abspath(__file__), "--noise"], sink, timeout=120)
        sink.seek(0)
        text = sink.read().decode(errors="replace")
    return rc == 3 and all(f"s{i}" in text for i in range(1, 7))


def _toy_out(toy: dict, perturb: bool, rev: str | None = None) -> dict:
    """Builds toy Volumes, manifests and a refit3-like record for the given runs, and runs diagnose(). rev: the
    recorded commit written into every toy header (None: no revision recorded)."""
    with tempfile.TemporaryDirectory() as tmp:
        roots = {"chain": f"{tmp}/chain/evals_v1", "q4own": f"{tmp}/q4own/evals_v1",
                 "pilots": f"{tmp}/pilots", "q2fresh": f"{tmp}/q2fresh"}
        run_keys, mrows = {}, []
        for t, (em, ids, roles, ru, _, kind, *extra) in toy.items():
            ex = extra[0] if extra else {}
            rel = f"L/sim_attack@september/{t}/x_{t}.eval"
            run_keys[f"sim|{t}"] = rel + "#e0"
            run_keys[f"run1|{t}"] = f"L/attack/{t}/y_{t}.eval#e0"
            sha = "0" * 64
            full = os.path.join(roots["chain"], rel)
            if kind == "eval":
                _write_eval(full, em, ids + ([_N] if perturb and t == "d01" else []), roles, ru, rev, ex.get("jm"))
                if ex.get("copy2"):  # the same bytes on a second root (a multi-copy path)
                    full2 = os.path.join(roots["q4own"], rel)
                    os.makedirs(os.path.dirname(full2), exist_ok=True)
                    Path(full2).write_bytes(Path(full).read_bytes())
            elif kind == "malformed":  # valid JSON header whose model_usage is a list
                os.makedirs(os.path.dirname(full), exist_ok=True)
                with zipfile.ZipFile(full, "w") as z:
                    z.writestr("header.json", json.dumps({"eval": {"model": 5}, "stats": {"model_usage": [1, 2]}}))
            if kind != "none":
                sha = file_sha256(full)
            mrows.append({"path": rel, "sha256": sha, "task": t, "arm_dir": "sim_attack@september"})
        cells, zero = {}, {}
        for c in CELLS:
            rpr = {k: 3 for k in run_keys if not (k.startswith("sim|") and toy[k[4:]][4] is not None
                                                   and c not in toy[k[4:]][4])}
            rwr = [k for k in run_keys if k not in rpr]
            cells[c] = {"rows_per_run": rpr, "runs_without_rows": rwr}
            zero[c] = {"n": len(rwr), "runs": [{"name": k, "key": "z", "set": "S"} for k in rwr]}
        return diagnose({"run_keys": run_keys, "cells": cells, "zero_row_runs": zero},
                        [("m1.json", {"files": mrows})], TOY_IDS, roots)


def run_toy(perturb: bool) -> dict:
    out = _toy_out(TOY, perturb)
    checks = {f"{n}.{k}": out["per_id"][n][k] == v for n, e in TOY_EXPECT.items() for k, v in e.items()}
    checks["readable_totals"] = {k: out[k] for k in TOY_EXPECT_READABLE} == TOY_EXPECT_READABLE
    # DIAG_AMEND1 2: the laptop print set carries no raw (a) / usage-only / (c) counts
    ps = print_set(out)
    checks.update({
        "print_terra_present": ps["per_id"]["terra"]["a_copies_present"] == {"gt0": True, "eq_all": False},
        "print_nano_cell": ps["per_id"]["nano"]["a_paths_present_per_cell"]["command|L48"] == {"gt0": True,
                                                                                               "eq_all": False},
        "print_c": ps["c_terra_x_sol"]["no_terra_no_sol"] == {"gt0": True, "eq_all": False}
                   and ps["c_unreadable"] == {"gt0": True, "eq_all": False},
        "print_no_raw_counts": ("c_terra_x_sol_copies" not in ps and not any(
            k.startswith("a_n_") or k == "b_n_copies_usage_only" for v in ps["per_id"].values() for k in v)),
    })
    # DIAG_AMEND1 1b: hint subsets other than (ii)
    allt = {**TOY, **TOY_EXTRA}
    no_ints = _int_leaves(ps) == 0  # DIAG_AMEND2: no integer anywhere in the printed set (gates are booleans)
    for name, (sub, dec, rev, ck_exp) in TOY_SUBSETS.items():
        o = _toy_out({k: allt[k] for k in sub}, False, rev)
        no_ints &= _int_leaves(print_set(o)) == 0
        checks[f"sub_{name}"] = o["decision_booleans"] == dec
        if ck_exp is not None:  # 09's checks (a)/(b) and the resulting real-mode status
            ck, pc = o["checks_09"], print_set(o)["checks_09"]
            checks[f"sub_{name}_checks09"] = (
                ck["G_SIMS_COMMIT"] == ck_exp["G_SIMS_COMMIT"] and ck["G_JUDGE_MESSAGES"] == ck_exp["G_JUDGE_MESSAGES"]
                and diag_status(ck)[0] == ck_exp["status"]
                and ("recorded_commits" not in ck_exp or pc["recorded_commits"]["names"] == ck_exp["recorded_commits"]))
        if name == "i_eligible":  # terra is present in all 3 readable copies
            checks["print_eq_all_true"] = print_set(o)["per_id"]["terra"]["a_copies_present"] == {"gt0": True,
                                                                                                  "eq_all": True}
    # Codex r15: ctx path counts are paired against readable PATHS. d16 alone has 2 readable copies and 1 readable path,
    # so n_paths_multi_copy = 1 must print eq_all true (against copies it would print false).
    o16 = _toy_out({"d16": TOY_EXTRA["d16"]}, False)
    p16 = print_set(o16)
    no_ints &= _int_leaves(p16) == 0
    checks["print_ctx_paths_denominator"] = (o16["n_readable_copies"] == 2 and o16["n_readable_paths"] == 1
                                             and p16["ctx"]["n_paths_multi_copy"] == {"gt0": True, "eq_all": True}
                                             and p16["ctx"]["n_copies_raw_not_ok"] == {"gt0": False, "eq_all": False})
    # the walker must see integers (a discriminating self-test), and the print sets must hold none
    checks["print_no_integers"] = (no_ints and _int_leaves({"a": {"b": [1, True]}, "c": 2.5}) == 2
                                   and _int_leaves({"x": True, "y": "3", "z": [False]}) == 0)
    # main toy: no revision recorded -> check (a) fails; d03 (nano) carries judge_messages -> check (b) passes
    ck = out["checks_09"]
    checks["main_checks09"] = (ck["G_SIMS_COMMIT"] is False and ck["n_copies_revision_missing"] == 8
                               and ck["n_copies_nano_present"] == 1 and ck["n_copies_nano_judge_messages_ok"] == 1
                               and ck["G_JUDGE_MESSAGES"] is True and diag_status(ck) == ("STOP", ["SIMS_COMMIT"]))
    checks["diag_status_units"] = (diag_status({"G_SIMS_COMMIT": True, "G_JUDGE_MESSAGES": True}) == ("DONE", [])
                                   and diag_status({"G_SIMS_COMMIT": True, "G_JUDGE_MESSAGES": False})
                                   == ("STOP", ["JUDGE_MESSAGES"]) and diag_status({}) == ("STOP", ["SIMS_COMMIT",
                                                                                                    "JUDGE_MESSAGES"]))
    # Codex r12 #3: two members with the same sample name, the FIRST lacking judge_messages -> must not pass
    import warnings

    with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
        warnings.simplefilter("ignore")  # zipfile warns on duplicate names
        p = f"{tmp}/dup.eval"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("samples/1_epoch_1.json", json.dumps({"scores": {"s": {"value": 1, "metadata": {"x": 1}}}}))
            z.writestr("samples/1_epoch_1.json", json.dumps({"scores": {"s": {"value": 1,
                                                                             "metadata": {"judge_messages": []}}}}))
        r = judge_messages_presence(p)
    checks["jm_duplicate_member"] = r == {"n_samples": 2, "n_with_key": 1, "err": None}
    # Codex r12 #1: printed commit / repo / version entries are names only
    pc = print_set(out)["checks_09"]
    checks["print_names_only"] = all(set(pc[k]) == {"names"} for k in ("recorded_commits", "revision_repos",
                                                                         "control_tower_versions"))
    # Codex r12 #2: only the "control-tower" key is read
    checks["ct_key_only"] = (_revision_fields("c", "o", {"control_tower": "9.9"})["control_tower_version"] is None
                             and _revision_fields("c", "o", {"control-tower": "3.0.0", "control_tower": "9.9"})[
                                 "control_tower_version"] == "3.0.0")
    checks.update({"c_terra_x_sol": out["c_terra_x_sol_copies"] == TOY_EXPECT_C,
                   "d_any": out["d_n_copies_any_diag_id_no_astra_no_sol"] == TOY_EXPECT_D_ANY,
                   "decision_booleans": out["decision_booleans"] == TOY_EXPECT_DEC,
                   "guard_non_id": guard("a b") == NON_ID and guard("x" * 81) == NON_ID and guard("a\n") == NON_ID
                   and guard(_R) == _R,
                   "guard_cap": len(guard_counts(Counter({f"r{i:02d}": 1 for i in range(12)}))["items"]) == MAX_PRINTED,
                   "worker_output_isolated": _worker_selftest()})
    checks.update({f"ctx.{k}": out["ctx"][k] == v for k, v in TOY_EXPECT_CTX.items()})
    return {"pass": all(checks.values()), "checks": checks}


# ---------------------------------------------------------------- run (worker side, then the Modal driver)
def _summary_line(d: dict) -> str:
    return "DIAG106_RESULT " + json.dumps(d, sort_keys=True)


def _write_out(name: str, record: dict) -> tuple[str, str]:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = f"{OUT_DIR}/{name}"
    data = json.dumps(record, sort_keys=True, indent=1).encode()
    with open(path, "xb") as f:  # append-only; the driver commits after the worker exits
        f.write(data)
    return path, sha256(data)


def _drive(mode: str, blobs: dict, manifests: list, meta: dict, stage: list) -> dict:
    stage[0] = "toy"
    toy_pos, toy_neg = run_toy(False), run_toy(True)
    gates = {"G_TOY_POS_PASS": toy_pos["pass"],
             "G_TOY_NEG_FAILS": (not toy_neg["pass"]) and not toy_neg["checks"]["nano.a_n_copies_present"]}
    if mode in ("test", "test-neg"):
        res = toy_neg if mode == "test-neg" else toy_pos
        stage[0] = "write"
        path, rsha = _write_out(f"diag_{mode}_{meta['stamp']}.json", {"mode": mode, "meta": meta, "toy": res,
                                                                       "gates": gates})
        return {"mode": mode, "status": "PASS" if res["pass"] else "MISMATCH", "gates": gates,
                "n_checks_false": sum(not v for v in res["checks"].values()), "results_path": path,
                "results_sha256": rsha}
    for name in ("G_TOY_POS_PASS", "G_TOY_NEG_FAILS"):
        if not gates[name]:
            raise GateError(name)
    stage[0] = "inputs"
    raw = {k: zlib.decompress(v) for k, v in blobs.items()}
    gates["G_INPUT_SHAS"] = set(raw) == set(PINS) and all(sha256(raw[k]) == PINS[k][2] for k in PINS)
    mf = [(p, zlib.decompress(b)) for p, b in manifests]
    gates["G_MANIFEST_DIGEST"] = len(mf) == MANIFEST_N and manifest_digest(mf) == MANIFEST_DIGEST
    lines = raw["sim_yaml"].decode().splitlines() if "sim_yaml" in raw else []
    # DIAG_AMEND1 1a: L21-L22 (terra's role cite) are byte-checked too
    gates["G_SIM_YAML_L19_L22"] = (len(lines) >= 22 and lines[18] == SIM_YAML_L19 and lines[19] == SIM_YAML_L20
                                   and lines[20] == SIM_YAML_L21 and lines[21] == SIM_YAML_L22)
    roots = {lab: root for lab, (_, _, root) in VOLS.items()}
    gates["G_ROOTS_EXIST"] = all(os.path.isdir(r) for r in roots.values())
    for name in ("G_INPUT_SHAS", "G_MANIFEST_DIGEST", "G_SIM_YAML_L19_L22", "G_ROOTS_EXIST"):
        if not gates[name]:
            raise GateError(name)
    ids = {"astra": lines[18].split(":", 1)[1].strip(), "sol": lines[19].split(":", 1)[1].strip(), "policy": POLICY_ID}
    stage[0] = "diagnose"
    out = diagnose(json.loads(raw["refit3"]), [(p, json.loads(b)) for p, b in mf], ids, roots)
    status, stop_reasons = diag_status(out["checks_09"])  # 09: STOP if check (a) or (b) fails
    full = {k: v for k, v in out.items() if not k.startswith("_")}  # all counts: Volume record only
    stage[0] = "write"
    path, rsha = _write_out(f"diag_results_{meta['stamp']}.json", {
        "mode": mode, "status": status, "stop_reasons": stop_reasons, "meta": meta, "pins": PINS, "gates": gates,
        "toy": {"pos": toy_pos, "neg": toy_neg}, "full_counts": full, "per_path": out["_per_path"]})
    # DIAG_AMEND1 2: the laptop gets only the reduced print set
    return {"mode": mode, "status": status, "stop_reasons": stop_reasons, "gates": gates, **print_set(out),
            "results_path": path, "results_sha256": rsha}


def _worker_main(td: str) -> None:
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
    out_vol = modal.Volume.from_name(OUT_VOL_NAME, create_if_missing=True)

    @app.function(image=image, volumes={**_vols, "/out": out_vol}, cpu=(2.0, 2.0), memory=(4096, 4096), timeout=1800,
                  startup_timeout=120, scaledown_window=60, retries=0, max_containers=1)  # DIAG_AMEND2
    def drive(mode: str, blobs: dict, manifests: list, meta: dict) -> dict:
        """Driver: never parses data; isolated worker; commit in finally; sanitized errors only."""
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
            finally:
                out_vol.commit()
        except Exception as e:
            err = f"{type(e).__name__} in driver"
        if err is None and "err" in res:
            err = str(res["err"])
        if err is not None:
            raise RuntimeError("diag106 failed: " + err)
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
        if mode == "real":
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
        if out["status"] not in ("PASS", "DONE"):
            raise SystemExit(1)


if __name__ == "__main__":  # worker / noise (internal), or the local toy
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        _worker_main(sys.argv[2])
        raise SystemExit(0)
    if len(sys.argv) == 2 and sys.argv[1] == "--noise":
        _noise_main()
    if len(sys.argv) != 2 or sys.argv[1] not in ("toy", "toy-neg"):
        raise SystemExit("usage: diag_sol106_ids.py toy|toy-neg")
    res = run_toy(sys.argv[1] == "toy-neg")
    print(_summary_line({"mode": sys.argv[1], "status": "PASS" if res["pass"] else "MISMATCH",
                         "checks_false": sorted(k for k, v in res["checks"].items() if not v)}))
    raise SystemExit(0 if res["pass"] else 1)
