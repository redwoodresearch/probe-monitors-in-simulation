"""nm172 pair check for 0d / 09 (2e 17:2x): the real honest run honest_a_e2|nm172_shared_dotfiles and the sim honest run
simh_nm360|nm172_shared_dotfiles carry byte-identical probe units. Are the two runs input-identical (cause (i)), or do their messages differ
(cause (ii), an activation file attached to the wrong run)? HASHES, LENGTHS AND KEYS ONLY: no message, tool or metadata text is printed or written.

Each file's sha256 must equal its full expected digest (refresh2's gate2_eval_meta.json, sha256 fb4f2f28...) before any sample is read, else STOP.
Per run: sample count (one expected), message count, tool calls, tool messages, the sha256 of the whole messages list as stored and with ids removed
at fixed positions only (message.id, message.tool_call_id, tool_calls[].id). Per message: role, characters of text and reasoning parts, number of
tool calls, the content sha256 over an explicit allowlist (role; content as a string, or each part's type and its text or reasoning, never `internal`;
each tool call's function and arguments, arguments untouched; a tool message's function and error), the full message sha256 as stored, and the
NAMES of the fields where the two runs' messages differ. Tool output lengths. 09's rule, applied here: cause (i) only if both digests match, the
message counts are equal and above zero, and every content hash matches; otherwise "cause (ii)" or "STOP: <reason>", and a non-zero exit. Positive
control in the same run: the real file against itself must give cause (i).
Usage: modal run --detach check_nm172_eval_pair.py --out <new local json> (the client stays up to write --out)"""
import json

import modal

VOL = "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14"  # the .eval store refresh2's eval_meta.py reads (/ev/evals_v1)
app = modal.App("probe-shift_aswin-surya_sub-e2esim_nm172pair_kill-2026-11-16")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("zipfile-zstd==0.0.4")
vol = modal.Volume.from_name(VOL)
REAL, SIM = "honest_a_e2|nm172_shared_dotfiles", "simh_nm360|nm172_shared_dotfiles"
RUNS = {  # run key -> (path under evals_v1, full expected sha256 from gate2_eval_meta.json)
    REAL: ("2026-10-15_123630_ep2_nm/honest_a/nm172_shared_dotfiles/2026-10-15T19-50-05-00-00_la-shared-dotfiles-real-honest-1_6tdbiHJWkAsZSWLmGcTfHh.eval",
           "941fedbb06c36c2cf1ff798e0ebb44c52e8d643bc63d7fe5c4d1d3ab08a154fe"),
    SIM: ("2026-10-15_simh_nm_s1/sim_honest@september/nm172_shared_dotfiles/2026-10-15T19-11-26-00-00_la-shared-dotfiles-sim-honest-1_kqJxHhCUmqLXYxy5NAvzE8.eval",
          "ca1f248f29f8009eb197bc2a6486a5400a61ee70391ca7bcebc470434941aec1"),
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
def compare() -> dict:
    import hashlib
    import zipfile
    from pathlib import Path

    import zipfile_zstd  # noqa: F401  (.eval members are zstd-compressed)

    sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
    msgs, out = {}, {"runs": {}}
    for key, (rel, want) in RUNS.items():  # every digest first: nothing is read from a sample unless both files are the expected ones
        b = (Path("/ev/evals_v1") / rel).read_bytes()
        out["runs"][key] = {"path": f"evals_v1/{rel}", "sha256": sha(b), "sha_matches_expected": sha(b) == want}
    if not all(r["sha_matches_expected"] for r in out["runs"].values()):
        return out | {"verdict": "STOP: a file digest differs from the expected one (no sample read)"}
    for key, (rel, _) in RUNS.items():
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
    for key, ms in msgs.items():  # CUT optional (b): a content part that is neither text nor reasoning is outside the hash's allowlist
        out["runs"][key]["other_content_parts"] = sum(1 for m in ms if isinstance(m.get("content"), list) for p_ in m["content"]
                                                      if not (isinstance(p_, dict) and p_.get("type") in ("text", "reasoning")))
    out["control_real_vs_real"] = {k: v for k, v in pair(msgs[REAL], msgs[REAL], True).items() if k != "pairwise"}
    import copy  # CUT optional (a): the DETECTION control, real against a copy with ONE tool-call argument changed, must give cause (ii) at that index

    det_, at_ = copy.deepcopy(msgs[REAL]), None
    for i, m in enumerate(det_):
        tc_ = next((t for t in (m.get("tool_calls") or []) if isinstance(t, dict) and isinstance(t.get("arguments"), dict) and t["arguments"]), None)
        if tc_ is not None:
            a0_ = sorted(tc_["arguments"])[0]
            tc_["arguments"][a0_] = canon(tc_["arguments"][a0_]) + "#changed"  # the changed value is never printed or written
            at_ = i
            break
    dc_ = pair(msgs[REAL], det_, True) if at_ is not None else {"verdict": "no tool-call argument to change"}
    out["control_detection"] = {"changed_message_index": at_, "verdict": dc_["verdict"], "first_mismatch_index": dc_.get("first_mismatch_index"),
                                "n_content_mismatches": dc_.get("n_content_mismatches")}
    p = pair(msgs[REAL], msgs[SIM], True)
    out["pair"] = p
    out["whole_list_match"] = {x: out["runs"][REAL][f"messages_sha256_{x}"] == out["runs"][SIM][f"messages_sha256_{x}"] for x in ("as_stored", "ids_removed")}
    out["verdict"] = ("STOP: the positive control (real against itself) did not give cause (i)" if out["control_real_vs_real"]["verdict"] != "cause (i)"
                      else "STOP: the detection control (one tool-call argument changed) did not give cause (ii) at its index"
                      if not (dc_["verdict"] == "cause (ii)" and dc_.get("first_mismatch_index") == at_ and dc_.get("n_content_mismatches") == 1)
                      else "STOP: content parts that are neither text nor reasoning" if any(r["other_content_parts"] for r in out["runs"].values())
                      else p["verdict"])
    return out


@app.local_entrypoint()
def main(out: str):
    r = compare.remote()
    with open(out, "x") as fh:
        json.dump(r, fh, indent=1)
    p = r.get("pair") or {}
    print(json.dumps({"verdict": r["verdict"], "control": (r.get("control_real_vs_real") or {}).get("verdict"), "n_messages": p.get("n_messages"),
                      "n_content_mismatches": p.get("n_content_mismatches"), "first_mismatch_index": p.get("first_mismatch_index"),
                      "whole_list_match": r.get("whole_list_match"), "sha_ok": {k: v["sha_matches_expected"] for k, v in r["runs"].items()},
                      "detection_control": r.get("control_detection"), "other_content_parts": {k: v.get("other_content_parts") for k, v in r["runs"].items()}}))
    if r["verdict"] != "cause (i)":
        raise SystemExit(1)
