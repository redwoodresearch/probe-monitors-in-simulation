"""Parameterised .eval pair check (09's GO item 2, the nm305 group, via 2e 17:5x; NOT RUN until CUT's glance): the CUT-passed logic of
check_nm172_eval_pair.py @ 0361c17d0 (its helpers canon, parts, allow, no_ids, chars and pair copied verbatim), over every pair of a named group of runs
whose probe units are byte-identical. Are two runs input-identical (cause (i)), or do their messages differ (cause (ii), an activation file
attached to the wrong run)? HASHES, LENGTHS AND KEYS ONLY: no message, tool or metadata text is printed or written.

Every file's sha256 must equal its full expected digest before any sample is read, else STOP. Per run and per pair: as check_nm172_eval_pair.py.
Controls in the same run, on the first run of EVERY pair: against itself must give cause (i); against a copy with ONE tool-call argument changed must
give cause (ii) with exactly one mismatch at that index; content parts that are neither text nor reasoning STOP. 09's rule per pair; a non-zero exit
on any STOP or any cause (ii) pair (CUT). Two inputs: --group (a named group, every pair; nm305 ran @ c87067ff0) or --pairs-file (09 via 2e 19:0x:
the byte-identical declared E12 pairs, keys plus full .eval shas from the frozen inventory / E12 block; a SEEDED sample of up to --k of them, the
seed fixed below, the pairs sorted before sampling, the selection recorded in --out).
Usage: modal run --detach check_eval_pairs.py (--group nm305 | --pairs-file <json> [--k 10]) --out <new local json> (the client stays up for --out)"""
import hashlib
import itertools
import json

import modal

VOL = "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14"  # the .eval store refresh2's eval_meta.py reads (/ev/evals_v1)
app = modal.App("probe-shift_aswin-surya_sub-e2esim_evalpairs_kill-2026-11-16")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("zipfile-zstd==0.0.4")
vol = modal.Volume.from_name(VOL)
SEED_STRING = "E2E sampled declared-E12-pair message-hash check, 09 via 2e, 2026-10-16"  # the sample's seed: int(sha256(SEED_STRING)) mod 2**32
SEED = int(hashlib.sha256(SEED_STRING.encode()).hexdigest(), 16) % 2**32
GROUPS = {  # group -> run key -> (path under evals_v1, full expected sha256); nm305 from 2e 17:5x (E12 block 683d4de1... / inventory 4a5d0fb3...)
    "nm305": {
        "honest_b|nm305_user_tracking": (
            "2026-10-14_newmains_s1/honest_b/nm305_user_tracking/2026-10-15T03-12-28-00-00_la-user-tracking-real-honest-1_CtLkbkiGRHthy4rebscHWc.eval",
            "74bf7712a2d433ccf9b53031c9af60df29f4c88a2d521edf265e69c20a3ab9cf"),
        "honest_a_e2|nm305_user_tracking": (
            "2026-10-15_123630_ep2_nm/honest_a/nm305_user_tracking/2026-10-15T19-56-07-00-00_la-user-tracking-real-honest-1_crX9Ne8kQ4iG23reVT3FuV.eval",
            "1149c9c2839761fa90efb7d101b3b3f66beda90fda42daa0b62a792e323b7118"),
        "honest_b_e2|nm305_user_tracking": (
            "2026-10-15_123630_ep2_nm/honest_b/nm305_user_tracking/2026-10-15T19-57-38-00-00_la-user-tracking-real-honest-1_bcsuxEYJFsR5sJnSASzfAy.eval",
            "e8a8edee4424cc2d2486d4803f47055c316d0d0dc1ab76cb78e413d1c6f0ae33"),
    },
}


def canon(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def parts(c):  # content as a string, or each part's type and its text (text parts) or reasoning (reasoning parts); `internal` never enters
    if isinstance(c, str) or c is None:
        return c
    return [{"type": p.get("type"), "text": p.get("text") if p.get("type") == "text" else p.get("reasoning") if p.get("type") == "reasoning" else None}
            if isinstance(p, dict) else {"type": type(p).__name__} for p in c]


def allow(m):  # the content hash's allowlist
    d = {"role": m.get("role"), "content": parts(m.get("content")),
         "tool_calls": [{"function": t.get("function"), "arguments": t.get("arguments")} for t in (m.get("tool_calls") or []) if isinstance(t, dict)]}
    if m.get("role") == "tool":
        d |= {"function": m.get("function"), "error": m.get("error")}
    return d


def no_ids(m):  # ids dropped at fixed positions only: message.id, message.tool_call_id, tool_calls[].id (an argument named "id" survives)
    d = {k: v for k, v in m.items() if k not in ("id", "tool_call_id")}
    if isinstance(d.get("tool_calls"), list):
        d["tool_calls"] = [{k: v for k, v in t.items() if k != "id"} if isinstance(t, dict) else t for t in d["tool_calls"]]
    return d


def chars(c):  # characters of text and reasoning
    if isinstance(c, str):
        return len(c)
    return sum(len(p.get("text") or p.get("reasoning") or "") for p in c if isinstance(p, dict)) if isinstance(c, list) else 0


def pair(a, s, sha_ok):  # 09's decision rule on two message lists
    import hashlib

    h = lambda m: hashlib.sha256(canon(allow(m)).encode()).hexdigest()  # noqa: E731
    n = max(len(a), len(s))
    rows = [{"i": i, "same_role": i < len(a) and i < len(s) and a[i].get("role") == s[i].get("role"),
             "content_sha256_match": i < len(a) and i < len(s) and h(a[i]) == h(s[i]),
             "full_sha256_match": i < len(a) and i < len(s) and canon(a[i]) == canon(s[i]),
             "fields_differing": (sorted(k for k in set(a[i]) | set(s[i]) if canon(a[i].get(k)) != canon(s[i].get(k))) if i < len(a) and i < len(s)
                                  else "missing in one run")} for i in range(n)]
    mm = [r["i"] for r in rows if not r["content_sha256_match"]]
    if not sha_ok:
        verdict = "STOP: a file digest differs from the expected one"
    elif not a or not s:
        verdict = "STOP: an empty message list"
    elif len(a) != len(s):
        verdict = "cause (ii)"  # different messages (different counts) with identical probe units
    else:
        verdict = "cause (i)" if not mm else "cause (ii)"
    return {"verdict": verdict, "n_messages": [len(a), len(s)], "n_content_mismatches": len(mm), "first_mismatch_index": mm[0] if mm else None,
            "pairwise": rows}


@app.function(image=image, cpu=1, memory=4096, timeout=900, retries=0, volumes={"/ev": vol})
def compare(runs: dict, pairs: list, label: str) -> dict:
    """runs: key -> (path under evals_v1, full expected sha256); pairs: [[a, b]]; the controls run on the first run of every pair."""
    import copy
    import hashlib
    import zipfile
    from pathlib import Path

    import zipfile_zstd  # noqa: F401  (.eval members are zstd-compressed)

    sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
    msgs, out = {}, {"group": label, "runs": {}}
    for key, (rel, want) in runs.items():  # every digest first: nothing is read from a sample unless every file is the expected one
        b = (Path("/ev/evals_v1") / rel).read_bytes()
        out["runs"][key] = {"path": f"evals_v1/{rel}", "sha256": sha(b), "sha_matches_expected": sha(b) == want}
    if not all(r["sha_matches_expected"] for r in out["runs"].values()):
        return out | {"verdict": "STOP: a file digest differs from the expected one (no sample read)"}
    for key, (rel, _) in runs.items():
        r = out["runs"][key]
        with zipfile.ZipFile(Path("/ev/evals_v1") / rel) as z:
            sams = sorted(n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json"))
            r["samples"] = len(sams)
            ms = json.loads(z.read(sams[0])).get("messages") or [] if len(sams) == 1 else []
        msgs[key] = ms
        r |= {"n_messages": len(ms), "n_tool_calls": sum(len(m.get("tool_calls") or []) for m in ms), "n_tool_messages": sum(1 for m in ms if m.get("role") == "tool"),
              "messages_sha256_as_stored": sha(canon(ms).encode()), "messages_sha256_ids_removed": sha(canon([no_ids(m) for m in ms]).encode()),
              "per_message": [{"i": i, "role": m.get("role"), "chars": chars(m.get("content")), "n_tool_calls": len(m.get("tool_calls") or []),
                               "content_sha256": sha(canon(allow(m)).encode()), "full_sha256": sha(canon(m).encode())} for i, m in enumerate(ms)],
              "tool_output_chars": [chars(m.get("content")) for m in ms if m.get("role") == "tool"]}
    if any(r["samples"] != 1 for r in out["runs"].values()):
        return out | {"verdict": "STOP: a file does not hold exactly one sample"}
    for key, ms in msgs.items():  # a content part that is neither text nor reasoning is outside the hash's allowlist
        out["runs"][key]["other_content_parts"] = sum(1 for m in ms if isinstance(m.get("content"), list) for p_ in m["content"]
                                                      if not (isinstance(p_, dict) and p_.get("type") in ("text", "reasoning")))
    def controls(first):  # self must give cause (i); one tool-call argument changed must give cause (ii) with one mismatch at that index
        cs_ = {k: v for k, v in pair(msgs[first], msgs[first], True).items() if k != "pairwise"} | {"run": first}
        det_, at_ = copy.deepcopy(msgs[first]), None
        for i, m in enumerate(det_):
            tc_ = next((t for t in (m.get("tool_calls") or []) if isinstance(t, dict) and isinstance(t.get("arguments"), dict) and t["arguments"]), None)
            if tc_ is not None:
                a0_ = sorted(tc_["arguments"])[0]
                tc_["arguments"][a0_] = canon(tc_["arguments"][a0_]) + "#changed"  # the changed value is never printed or written
                at_ = i
                break
        dc_ = pair(msgs[first], det_, True) if at_ is not None else {"verdict": "no tool-call argument to change"}
        cd_ = {"run": first, "changed_message_index": at_, "verdict": dc_["verdict"], "first_mismatch_index": dc_.get("first_mismatch_index"),
               "n_content_mismatches": dc_.get("n_content_mismatches")}
        ok_ = cs_["verdict"] == "cause (i)" and dc_["verdict"] == "cause (ii)" and dc_.get("first_mismatch_index") == at_ and dc_.get("n_content_mismatches") == 1
        return cs_, cd_, ok_

    ctl = {a: controls(a) for a in sorted({a for a, _ in pairs})}
    first = sorted(ctl)[0]
    out["control_self"], out["control_detection"] = ctl[first][0], ctl[first][1]
    out["controls_by_run"] = {a: {"self": c[0]["verdict"], "detection": c[1], "ok": c[2]} for a, c in ctl.items()}
    out["pairs"] = {}
    for a, b in pairs:
        p = pair(msgs[a], msgs[b], True)
        out["pairs"][f"{a} vs {b}"] = p | {"whole_list_match": {x: out["runs"][a][f"messages_sha256_{x}"] == out["runs"][b][f"messages_sha256_{x}"]
                                                                for x in ("as_stored", "ids_removed")}}
    bad_ = sorted(a for a, c in ctl.items() if not c[2])
    out["verdict"] = (f"STOP: a control (self cause (i), or one changed argument cause (ii) at its index) failed on {bad_}" if bad_
                      else "STOP: content parts that are neither text nor reasoning" if any(r["other_content_parts"] for r in out["runs"].values())
                      else None)
    if out["verdict"] is None:  # CUT: a STOP pair makes the whole check STOP; a cause (ii) pair is a STOP under 09's rule too (non-zero exit)
        stops = sorted(k for k, p in out["pairs"].items() if p["verdict"].startswith("STOP"))
        two = sorted(k for k, p in out["pairs"].items() if p["verdict"] == "cause (ii)")
        out["verdict"] = (f"STOP: pair verdicts STOP in {stops}" if stops else f"cause (ii) (a STOP under 09's rule) in {two}" if two
                          else "cause (i) for every pair")
    return out


def select_pairs(doc, k):
    """A pairs file -> (runs, the sampled pairs, the selection record). doc: {"pairs": [{"a", "a_eval", "a_sha256", "b", "b_eval", "b_sha256"}]}, every
    eval a path under evals_v1 and every sha256 the full digest from the frozen inventory / E12 block. Selection (09 via 2e 19:0x): each pair as
    (a, b) with a < b, the pairs sorted, then random.Random(SEED).sample of min(k, n). One key must name one (eval, sha256) everywhere."""
    import random

    runs, ps = {}, set()
    for p in doc["pairs"]:
        a, b = sorted(((p["a"], p["a_eval"], p["a_sha256"]), (p["b"], p["b_eval"], p["b_sha256"])))
        assert a[0] != b[0], ("a pair of one key", a[0])
        for key, ev, sh in (a, b):
            assert len(sh) == 64 and runs.setdefault(key, (ev, sh)) == (ev, sh), ("one key, two files", key)
        ps.add((a[0], b[0]))
    ps = sorted(ps)
    sel = sorted(random.Random(SEED).sample(ps, min(k, len(ps))))
    return {x: runs[x] for p in sel for x in p}, [list(p) for p in sel], {"seed_string": SEED_STRING, "seed": SEED, "n_pairs": len(ps), "k": k,
                                                                          "selected": [list(p) for p in sel]}


@app.local_entrypoint()
def main(out: str, group: str = "", pairs_file: str = "", k: int = 10):
    assert bool(group) != bool(pairs_file), "give --group or --pairs-file"
    if group:
        runs, pairs, sel = GROUPS[group], [list(p) for p in itertools.combinations(GROUPS[group], 2)], None
    else:
        import hashlib

        b_ = open(pairs_file, "rb").read()
        runs, pairs, sel = select_pairs(json.loads(b_), k)
        sel |= {"pairs_file": pairs_file, "pairs_file_sha256": hashlib.sha256(b_).hexdigest()}
    r = compare.remote(runs, pairs, group or "sampled pairs") | ({"selection": sel} if sel else {})
    with open(out, "x") as fh:
        json.dump(r, fh, indent=1)
    print(json.dumps({"verdict": r["verdict"], "sha_ok": {k: v["sha_matches_expected"] for k, v in r["runs"].items()},
                      "control_self": (r.get("control_self") or {}).get("verdict"), "detection_control": r.get("control_detection"),
                      "other_content_parts": {k: v.get("other_content_parts") for k, v in r["runs"].items()},
                      "pairs": {k: {x: p.get(x) for x in ("verdict", "n_messages", "n_content_mismatches", "first_mismatch_index", "whole_list_match")}
                                for k, p in (r.get("pairs") or {}).items()}}))
    if r["verdict"] != "cause (i) for every pair":  # any STOP, any cause (ii) pair: non-zero exit; the per-pair table is in --out
        raise SystemExit(1)
