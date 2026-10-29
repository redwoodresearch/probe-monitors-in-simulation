"""POST HOC (the source audit's k = 1 question; 0d's approval under Aswin's standing under-$5 rule, before 19:31 PT 10-11):
where do k = 1's identical simulated replies come from? For each of the 400 sites, is the real reply at t found verbatim
in what the simulator had: its honest reference run's outputs (rendered as September pastes them, `str(output)[:cap]`),
its stored system prompt, the history as its session held it (earlier calls and outputs, outputs capped at 6,000 chars;
memory[:-2]), the agent's call at t as shown to it (memory[-2]), its grounding tools' responses (RECONSTRUCTED: re-served
from the same static tree with the sample's path mappings, as runner/simulate/config.py configures codebase_read; grep's
60-hit cap follows filesystem order, so its replay may differ), or any file of the environment's codebase (per file,
inside the root)? The uncapped real history outputs are a separate diagnostic. A match shows the text was AVAILABLE to
the simulator, not that it copied it.

Grouping is the committed identical flag (`sim_text_equals_self_text` in claimB/k1_v1/scores_4be7958c7157.json: the
simulated reply after fixed-sim v2's R1/R2/R5 fixes, the donor clock shift and the level-1 scrub equals the scrubbed real
reply; run_k1_modal.py build_jobs). `sim_in` and `sim_equals_real_here` use the UNCORRECTED published reader reply, so
they are diagnostics only. Two normalisations on both sides: `raw` (text as stored; files read with errors="replace")
and `scrub1` (probe_shift.scrub level 1). The cache is bound to what was scored (record digests, the published replies'
sha256 against the scores' inputs) and the simulator build to the local September tree (commit, src diff, codebases).

$0 on the laptop, from text already on it (the k = 1 cache the laptop simulator run wrote) and the static codebase.
Output: numbers, site ids, task/environment/tool names, lengths and hashes; no reply, transcript, command or file text
is printed or saved, and failures name site ids only.

  cd $TMPDIR && PYTHONPATH=<repo>/src uv run --project ~/simulated-control-evals-sept --no-sync python \\
      <repo>/experiments/2026-10-11_claimB_k_replies/analyze_POSTHOC_k1_identical_source.py \\
      --cache <k1 cache dir> --scores <local copy of scores_4be7958c7157.json>
"""

import argparse
import asyncio
import contextlib
import hashlib
import io
import json
import logging
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from probe_shift.scrub import scrub
import sim.affordances.codebase_read as CR
from sim.affordances.codebase_read import codebase_read_tools

HERE = Path(__file__).parent
SEPT = Path.home() / "simulated-control-evals-sept"
SCORES_SHA = "b9e4eeac0d82ffef762fdf83d4dfbf7ff6242607b983abbfd11da03fd207a8af"
MODES = ("raw", "scrub1")
BINS = [(1, 20, "1-20"), (21, 200, "21-200"), (201, 2000, "201-2000"), (2001, float("inf"), ">2000")]
SOURCES = ("reference_outputs", "system_prompt", "history_as_seen", "call_at_t_as_seen", "grounding_responses",
           "env_codebase_files")
DIAG = ("history_outputs_uncapped",)


class Stop(Exception):
    """Raised with site ids and counts only."""


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tree_sha256(root: Path) -> str:
    """Copied from 2026-10-06_matched_step_pairs/run_simulate_pairs.py:94 (the hash k = 1 recorded per environment)."""
    h = hashlib.sha256()
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        h.update(str(p.relative_to(root)).encode() + b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def norm(t: str, mode: str) -> str:
    return t if mode == "raw" else scrub(t, 1)


def bin_of(n: int) -> str:
    return next(name for lo, hi, name in BINS if lo <= n <= hi)


def codebase_files(root: Path) -> tuple[list[str], dict]:
    base, texts, cov = root.resolve(), [], Counter()
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        if not p.resolve().is_relative_to(base):
            cov["outside_root"] += 1
            continue
        try:
            texts.append(p.read_text(errors="replace"))
            cov["read"] += 1
        except OSError:
            cov["unreadable"] += 1
    return texts, dict(cov)


async def grounding_responses(rec: dict, calls: list, root: Path) -> tuple[list[str], int]:
    """Re-serve each grounding call as codebase_read does; every response is kept (errors too, as the simulator saw them);
    calls whose arguments don't fit the tool are counted (the simulator answered those with an error message)."""
    _, fns = codebase_read_tools(str(root), path_mappings=rec["sim_metadata"].get("container_path_mappings"))
    outs, bad = [], 0
    for c in calls:
        try:
            outs.append(await fns[c["tool"]](**c["arguments"]))
        except (TypeError, KeyError):
            bad += 1
    return outs, bad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--scores", required=True, type=Path)
    a = p.parse_args()
    logging.disable(logging.CRITICAL)
    raw_scores = a.scores.read_bytes()
    if sha(raw_scores) != SCORES_SHA:
        raise Stop("scores file is not the committed one")
    scores = json.loads(raw_scores)
    flags = {r["site"]: r for r in scores["sites"]}
    recs = [json.loads(x) for x in open(a.cache / "records.jsonl")]
    sim_rows = [json.loads(x) for x in open(a.cache / "sim_results.jsonl")]
    replies_raw = (a.cache / "replies.jsonl").read_bytes()
    if sha(replies_raw) != scores["inputs_sha256"]["replies"]:
        raise Stop("published replies differ from the ones scored")
    replies = {r["site"]: r for r in (json.loads(x) for x in replies_raw.decode().splitlines())}
    sims = {r["site"]: r for r in sim_rows}
    if len(sim_rows) != len(sims) or len(recs) != 400 or len(replies) != 400:
        raise Stop(f"row counts: records {len(recs)}, sim rows {len(sim_rows)} for {len(sims)} sites, replies {len(replies)}")
    if not (set(flags) == set(sims) == set(replies) == {r["site"] for r in recs}):
        raise Stop("sites don't line up across records, sim results, replies and scores")
    bad = [r["site"] for r in recs if sha(json.dumps(r, sort_keys=True).encode()) != sims[r["site"]]["record_sha256"]
           or replies[r["site"]]["record_sha256"] != sims[r["site"]]["record_sha256"]]
    bad += [s for s in sims if sims[s]["status"] != "ok" or replies[s]["status"] != "ok"
            or replies[s]["reader_output"] != sims[s]["result"]["reader_output"]]
    if bad:
        raise Stop(f"sites not bound to the scored replies: {sorted(set(bad))}")
    builds = {json.dumps({k: v for k, v in s["build"].items() if k != "script_sha256"}, sort_keys=True) for s in sims.values()}
    if len(builds) != 1:
        raise Stop(f"{len(builds)} simulator builds across sites")
    build = sims[recs[0]["site"]]["build"]
    git_sept = lambda *x: subprocess.run(["git", "-C", str(SEPT), *x], capture_output=True, text=True, check=True).stdout  # noqa: E731
    if not Path(CR.__file__).resolve().is_relative_to(SEPT.resolve()):
        raise Stop("the grounding code is not imported from the September tree")
    if git_sept("rev-parse", "HEAD").strip() != build["sims_commit"] or sha(git_sept("diff", "HEAD", "--", "src").encode()) != build["sims_src_diff_sha256"]:
        raise Stop("the local September tree is not the build the simulator ran")
    envs = sorted({r["environment"] for r in recs})
    roots = {e: SEPT / ".settings" / "linuxarena" / e / "codebase" for e in envs}
    if any(tree_sha256(roots[e]) != build["codebase_sha256"][e] for e in envs):
        raise Stop("a local codebase differs from the one the simulator read")
    cbf, coverage = {}, {}
    for e in envs:
        texts, coverage[e] = codebase_files(roots[e])
        for m in MODES:
            cbf[(e, m)] = [norm(t, m) for t in texts]

    rows, grounding = [], Counter()
    for rec in sorted(recs, key=lambda r: r["site"]):
        s, sim, md = rec["site"], sims[rec["site"]]["result"], rec["sim_metadata"]
        mem = sim["memory"]
        if md["orig_trajectory_mode"] != "dump" or len(mem) != 2 * (len(rec["prefix"]) + 1) or any(
                m["role"] != ("user" if i % 2 == 0 else "assistant") for i, m in enumerate(mem)):
            raise Stop(f"site {s}: unexpected reference mode or session layout")
        texts = [rec["real_output"], replies[s]["reader_output"], rec["stored_system_prompt"]] + [m["content"] for m in mem] + \
                [c.get("output") for c in rec["prefix"]]
        if any(not isinstance(t, str) for t in texts):
            raise Stop(f"site {s}: a non-text value")
        cap = md["orig_trajectory_out_cap"]
        ref = [str(o.get("output", ""))[:cap] for o in md["orig_trajectory"]]  # original_trajectory.py:67's rendering
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            g_outs, g_bad = asyncio.run(grounding_responses(rec, sim["grounding_calls"], roots[rec["environment"]]))
        grounding["calls"] += len(sim["grounding_calls"])
        grounding["calls_with_unfit_arguments"] += g_bad
        row = {"site": s, "environment": rec["environment"], "main_task": rec["main_task"], "tool": flags[s]["tool"],
               "identical": bool(flags[s]["sim_text_equals_self_text"]), "grounding_calls": len(sim["grounding_calls"]),
               "grounding_calls_unfit_not_searched": g_bad}
        for m in MODES:
            real, simr = norm(rec["real_output"], m), norm(replies[s]["reader_output"], m)
            hay = {"reference_outputs": [norm(x, m) for x in ref],
                   "system_prompt": [norm(rec["stored_system_prompt"], m)],
                   "history_as_seen": [norm(x["content"], m) for x in mem[:-2]],
                   "call_at_t_as_seen": [norm(mem[-2]["content"], m)],
                   "grounding_responses": [norm(x, m) for x in g_outs],
                   "env_codebase_files": cbf[(rec["environment"], m)],
                   "history_outputs_uncapped": [norm(c["output"], m) for c in rec["prefix"]]}
            row[m] = {"len_real": len(real), "len_sim_uncorrected": len(simr), "real_sha256_16": sha(real.encode())[:16],
                      "sim_uncorrected_equals_real": simr == real,
                      "real_in": {k: bool(real) and any(real in h for h in v) for k, v in hay.items()},
                      "sim_uncorrected_in": {k: bool(simr) and any(simr in h for h in v) for k, v in hay.items()}}
        rows.append(row)

    def counts(rr, m, who):
        rr = [r for r in rr if r[m]["len_real" if who == "real_in" else "len_sim_uncorrected"] > 0]
        c = {"nonempty_sites": len(rr)}
        for k in SOURCES + DIAG:
            c[k] = sum(r[m][who][k] for r in rr)
        c["any_source"] = sum(any(r[m][who][k] for k in SOURCES) for r in rr)
        c["any_source_except_codebase"] = sum(any(r[m][who][k] for k in SOURCES if k != "env_codebase_files") for r in rr)
        c["none"] = c["nonempty_sites"] - c["any_source"]
        return c

    summary = {}
    for m in MODES:
        x = {}
        for name, grp in (("identical", [r for r in rows if r["identical"]]), ("non_identical", [r for r in rows if not r["identical"]])):
            x[name] = {"empty_real_replies": sum(r[m]["len_real"] == 0 for r in grp),
                       "real_reply_found_in": counts(grp, m, "real_in"),
                       "uncorrected_sim_reply_found_in": counts(grp, m, "sim_uncorrected_in"),
                       "real_reply_found_in_by_length": {b: counts([r for r in grp if r[m]["len_real"] and bin_of(r[m]["len_real"]) == b], m, "real_in")
                                                         for _, _, b in BINS}}
        x["uncorrected_equality_vs_committed_flag"] = dict(Counter(f"flag={r['identical']}|uncorrected_equal={r[m]['sim_uncorrected_equals_real']}" for r in rows))
        summary[m] = x
    out = {"label": "POST HOC: k = 1 identical replies vs what the simulator had (substring counts; availability, not copying; no text)",
           "sites": len(rows), "identical": sum(r["identical"] for r in rows), "summary": summary, "grounding": dict(grounding),
           "codebase_coverage": coverage, "per_site": rows,
           "inputs": {"records.jsonl": sha((a.cache / "records.jsonl").read_bytes()), "sim_results.jsonl": sha((a.cache / "sim_results.jsonl").read_bytes()),
                      "replies.jsonl": sha(replies_raw), "scores_4be7958c7157.json": SCORES_SHA, "sims_commit": build["sims_commit"],
                      "codebase_sha256": build["codebase_sha256"]},
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
           "script_sha256": sha(Path(__file__).read_bytes())}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"k1_identical_source_POSTHOC_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    for m in MODES:
        for g in ("identical", "non_identical"):
            print(m, g, "real reply found in:", json.dumps(summary[m][g]["real_reply_found_in"]))
            print(m, g, "uncorrected sim reply found in:", json.dumps(summary[m][g]["uncorrected_sim_reply_found_in"]))
            print(m, g, "real reply by length:", json.dumps({b: {k: v[k] for k in ("nonempty_sites", "reference_outputs", "history_as_seen", "call_at_t_as_seen", "grounding_responses", "env_codebase_files", "any_source", "none")}
                                                            for b, v in summary[m][g]["real_reply_found_in_by_length"].items()}))
        print(m, "uncorrected equality vs flag:", summary[m]["uncorrected_equality_vs_committed_flag"])
    print("grounding:", out["grounding"], "codebase coverage:", coverage)
    print("->", dst)


if __name__ == "__main__":
    try:
        main()
    except Stop as e:
        raise SystemExit(f"STOP: {e}") from None
    except Exception as e:  # no message or traceback: either could carry input text
        raise SystemExit(f"FAILED: {type(e).__name__}") from None
