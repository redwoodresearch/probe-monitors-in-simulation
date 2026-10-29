"""Schema probe for the served-context recheck: key names, types, lengths and array shapes only. Modal CPU, read-only.

Prints NO string values except short identifier-like ones (keys, file names, world names), and no token ids or scores:
lists show their length and the structure of their first element. Nothing is written.

  uv run --no-project --with 'modal>=1.2' modal run experiments/2026-10-14_p2_served_recheck/run_schema_probe.py
"""

import modal

APP = "probe-shift_aswin-surya_sub-p2probes_servedrecheck_kill-2026-11-13"
VOLS = {
    "/acts": "probe-shift_aswin-surya_sub-p2served_acts_kill-2026-11-14",
    "/cw": "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14",
    "/q3": "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08",
    "/q4": "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13",
    "/rl": "probe-shift_aswin-surya_sub-phase2_request-log_kill-2026-11-06",
}
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.3.3")
volumes = {k: modal.Volume.from_name(v).read_only() for k, v in VOLS.items()}


def _ident(s: str) -> bool:
    import re

    return len(s) <= 60 and re.fullmatch(r"[A-Za-z0-9_.:/@|=+\- ]*", s) is not None


def show(o, path="", depth=0, maxd=5, out=None):
    out = [] if out is None else out
    if depth > maxd:
        return out
    if isinstance(o, dict):
        ks = list(o.keys())
        if len(ks) > 30:
            out.append(f"{path}: dict[{len(ks)}] keys e.g. {[k if _ident(k) else '<k>' for k in ks[:2]]}")
            show(o[ks[0]], path + "[<k>]", depth + 1, maxd, out)
            return out
        for k in ks:
            kk = k if _ident(k) else f"<key len {len(k)}>"
            v = o[k]
            if isinstance(v, bool) or v is None:
                out.append(f"{path}.{kk}: {v!r}")
            elif isinstance(v, (int, float)):
                out.append(f"{path}.{kk}: {type(v).__name__}")
            elif isinstance(v, str):
                out.append(f"{path}.{kk}: str[{len(v)}]" + (f" = {v!r}" if _ident(v) else ""))
            elif isinstance(v, (list, dict)):
                out.append(f"{path}.{kk}: {type(v).__name__}[{len(v)}]")
                show(v, f"{path}.{kk}", depth + 1, maxd, out)
    elif isinstance(o, list) and o:
        v = o[0]
        if isinstance(v, (dict, list)):
            show(v, path + "[0]", depth + 1, maxd, out)
        else:
            out.append(f"{path}[0]: {type(v).__name__}")
    return out


@app.function(image=image, volumes=volumes, cpu=2, memory=16384, timeout=600, retries=0)
def probe() -> str:
    import gzip
    import json
    import os

    import numpy as np

    lines = []
    P = lines.append

    def first_jsonl(path):
        with open(path, "rb") as f:
            return json.loads(f.readline())

    def ls(path, n=8):
        xs = sorted(os.listdir(path))
        P(f"LS {path}: {len(xs)} entries, e.g. {xs[:n]}")
        return xs

    S = "/acts/served_2026-10-14_200651"
    for L in ("A", "A0", "B"):
        ls(f"{S}/{L}/inputs")
    for w in ("run1", "honest_a"):
        P(f"== {S}/A/inputs/{w}.jsonl record 0")
        lines.extend(show(first_jsonl(f"{S}/A/inputs/{w}.jsonl")))
    P(f"== {S}/B/inputs/run1.jsonl record 0")
    lines.extend(show(first_jsonl(f"{S}/B/inputs/run1.jsonl")))
    R = "/acts/served_2026-10-14_200651_extract_A_v1"
    ls(R)
    for f in ("config.json", "complete.json", "attempt_1.json"):
        P(f"== {R}/{f}")
        lines.extend(show(json.load(open(f"{R}/{f}")), maxd=3))
    ch = ls(f"{R}/chunks")
    P(f"== {R}/chunks/{ch[0]}")
    lines.extend(show(json.load(open(f"{R}/chunks/{ch[0]}"))))
    m = ls(f"{R}/meta/run1", 2)
    P(f"== {R}/meta/run1/{m[0]}")
    lines.extend(show(json.load(open(f"{R}/meta/run1/{m[0]}"))))
    m = ls(f"{R}/meta/honest_a", 2)
    P(f"== {R}/meta/honest_a/{m[0]}")
    lines.extend(show(json.load(open(f"{R}/meta/honest_a/{m[0]}"))))
    bl = ls(f"{R}/by_layer/L34", 3)
    a = np.load(f"{R}/by_layer/L34/{bl[0]}", mmap_mode="r")
    P(f"NPY {bl[0]}: {a.dtype} {a.shape}")

    C = "/cw/served"
    ls(C)
    ls(f"{C}/A/p2", 40)
    z = np.load(f"{C}/A/p2/fibs__reply_mean.npz", allow_pickle=False)
    P("NPZ fibs__reply_mean: " + "; ".join(f"{k} {z[k].dtype} {z[k].shape}" for k in z.files))
    z = np.load(f"{C}/A/p2/fibs__command.npz", allow_pickle=False)
    P("NPZ fibs__command: " + "; ".join(f"{k} {z[k].dtype} {z[k].shape}" for k in z.files))
    z = np.load(f"{C}/A/p2/refusal__call_end_content.npz", allow_pickle=False)
    P("NPZ refusal__call_end_content: " + "; ".join(f"{k} {z[k].dtype} {z[k].shape}" for k in z.files))
    for f in ("fibs__run_means.json", "calibration.json", "manifest_served.json"):
        P(f"== {C}/A/p2/{f}")
        lines.extend(show(json.load(open(f"{C}/A/p2/{f}")), maxd=4))
    ua = [x for x in os.listdir(C) if x.startswith("unit_assignment")]
    P(f"== {C}/{ua[0]}")
    lines.extend(show(json.load(open(f"{C}/{ua[0]}")), maxd=5))

    for st in ("rerun_strict_segends_v2", "scaleup_l2_segends_v1", "scaleup_canary_segends_v1", "scaleup_topup_segends_v1",
               "second_attack_segends_v1", "honest_c_segends_v1"):
        xs = ls(f"/q3/{st}/inputs", 10)
    P("== /q3/rerun_strict_segends_v2/inputs/attack.jsonl record 0")
    lines.extend(show(first_jsonl("/q3/rerun_strict_segends_v2/inputs/attack.jsonl")))
    for x in ("2026-10-09_rerun_strict_segends_v1", "2026-10-10_scaleup_l2_segends_v1", "2026-10-10_second_attack_segends_v1",
              "2026-10-11_honest_c_segends_v1"):
        if os.path.exists(f"/q3/{x}/config.json"):
            P(f"== /q3/{x}/config.json")
            lines.extend(show(json.load(open(f"/q3/{x}/config.json")), maxd=3))

    ls("/q4/q4b_extract_main_v1")
    ls("/q4/q4b_extract_main_v1/meta")
    mm = ls("/q4/q4b_extract_main_v1/meta/honest_b", 2)
    P(f"== /q4/q4b_extract_main_v1/meta/honest_b/{mm[0]}")
    lines.extend(show(json.load(open(f"/q4/q4b_extract_main_v1/meta/honest_b/{mm[0]}"))))
    cc = ls("/q4/q4b_extract_main_v1/chunks", 3)
    P(f"== /q4/q4b_extract_main_v1/chunks/{cc[0]}")
    lines.extend(show(json.load(open(f"/q4/q4b_extract_main_v1/chunks/{cc[0]}"))))
    ls("/q4/q4b_main_v1/inputs")
    P("== /q4/q4b_main_v1/inputs/honest_b.jsonl record 0")
    lines.extend(show(first_jsonl("/q4/q4b_main_v1/inputs/honest_b.jsonl")))
    ls("/q4/q4b_extract_c_v1")
    ls("/q4/q4f_c_v1/inputs")
    bl = ls("/q4/q4b_extract_main_v1/by_layer/L34", 3)
    a = np.load(f"/q4/q4b_extract_main_v1/by_layer/L34/{bl[0]}", mmap_mode="r")
    P(f"NPY q4b {bl[0]}: {a.dtype} {a.shape}")

    logs = sorted(x for x in os.listdir("/rl") if x.startswith("requests_2026-10-11"))
    P(f"LOGS 10-11: {logs}")
    with gzip.open(f"/rl/{logs[0]}", "rt") as f:
        recs = [json.loads(f.readline()) for _ in range(3)]
    for i, r in enumerate(recs):
        P(f"== log record {i}")
        lines.extend(show(r, maxd=4))
    return "\n".join(lines)


@app.function(image=image, volumes=volumes, cpu=2, memory=16384, timeout=600, retries=0)
def probe2() -> str:
    import json
    import os
    from collections import defaultdict

    import numpy as np

    lines = []
    P = lines.append

    def first_jsonl(path):
        with open(path, "rb") as f:
            return json.loads(f.readline())

    def call_keys(calls, label):
        by = defaultdict(set)
        cnt = defaultdict(int)
        for c in calls:
            by[c.get("kind")].add(tuple(sorted(c.keys())))
            cnt[c.get("kind")] += 1
        for k, v in by.items():
            P(f"CALLKEYS {label} kind={k} n={cnt[k]}: {sorted(set().union(*v))}; variants={len(v)}")

    C = "/cw/served"
    m = json.load(open(f"{C}/A/p2/manifest_served.json"))
    for k, v in m.items():
        kk = k if _ident(k) else f"<key len {len(k)}>"
        P(f"MANIFEST .{kk}: {type(v).__name__}" + (f"[{len(v)}]" if isinstance(v, (list, dict, str)) else ""))
        if isinstance(v, dict) and len(v) <= 30:
            P("   subkeys: " + str([x if _ident(x) else '<k>' for x in v.keys()]))
    ua = json.load(open(f"{C}/unit_assignment_2026-10-14_204921.json"))
    P("UA label: " + ua["label"])
    P("UA columns: " + str(ua["columns"]))
    P("UA row0 types: " + str([type(x).__name__ for x in ua["rows"][0]]))
    P("UA other keys: " + str([k for k in ua.keys()]))
    for f in sorted(os.listdir(f"{C}/A/p2")):
        if f.endswith(".npz") and (f.startswith("fibs") or f.startswith("apollo")):
            z = np.load(f"{C}/A/p2/{f}", allow_pickle=False)
            P(f"NPZ {f}: " + "; ".join(f"{k} {z[k].dtype} {z[k].shape}" for k in z.files))
    for f in ("fibs__run_means.json", "calibration.json"):
        P(f"LABEL {f}: " + json.load(open(f"{C}/A/p2/{f}"))["label"])
    S = "/acts/served_2026-10-14_200651"
    r = first_jsonl(f"{S}/A/inputs/run1.jsonl")
    call_keys(r["calls"], "A inputs run1 rec0")
    R = "/acts/served_2026-10-14_200651_extract_A_v1"
    mf = sorted(os.listdir(f"{R}/meta/run1"))[0]
    mm = json.load(open(f"{R}/meta/run1/{mf}"))
    call_keys(mm["calls"], "A meta run1 file0")
    P(f"A meta run1 file0: n_rows={mm['n_rows']} n_calls={len(mm['calls'])} ids_len_vs_tokens: n_tokens={mm['n_tokens']}")
    r = first_jsonl("/q3/rerun_strict_segends_v2/inputs/attack.jsonl")
    call_keys(r["calls"], "segends attack rec0")
    r = first_jsonl("/q4/q4b_main_v1/inputs/honest_b.jsonl")
    call_keys(r["calls"], "reader q4b_main inputs honest_b rec0")
    rm = sorted(os.listdir("/q4/q4b_extract_main_v1/meta/honest_b"))[0]
    mm = json.load(open(f"/q4/q4b_extract_main_v1/meta/honest_b/{rm}"))
    call_keys(mm["calls"], "reader meta honest_b file0")
    P(f"reader meta file0: n_rows={mm['n_rows']}")
    P("LS q4b_extract_c_v1/meta: " + str(sorted(os.listdir("/q4/q4b_extract_c_v1/meta"))))
    cf = sorted(os.listdir("/q4/q4b_extract_c_v1/meta/honest_b"))
    P(f"q4b_extract_c_v1/meta/honest_b: {len(cf)} files")
    mm = json.load(open(f"/q4/q4b_extract_c_v1/meta/honest_b/{cf[0]}"))
    call_keys(mm["calls"], "reader C meta file0")
    r = first_jsonl("/q4/q4f_c_v1/inputs/honest_b.jsonl")
    call_keys(r["calls"], "reader C inputs rec0")
    P("reader C inputs rec0 keys: " + str(sorted(r.keys())))
    return "\n".join(lines)


HF = "/hf/hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654"


@app.function(image=image.pip_install("tokenizers==0.22.1"), volumes={"/hf": modal.Volume.from_name("probe-shift_aswin-surya_hf-cache").read_only()},
              cpu=1, memory=4096, timeout=300, retries=0)
def probe3() -> str:
    """The pinned chat template (a config file, not transcript text) and marker token ids."""
    from tokenizers import Tokenizer

    t = Tokenizer.from_file(f"{HF}/tokenizer.json")
    out = ["TEMPLATE:", open(f"{HF}/chat_template.jinja").read(), "MARKERS:"]
    for s in ("<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>",
              "</tool_response>", "<tools>", "</tools>", "\n</parameter>", "\n</tool_response>", "<tool_response>\n",
              "<think>\n\n</think>\n\n", "\n\n", "<|im_start|>assistant\n", "<|im_start|>system\n", "<|endoftext|>"):
        out.append(f"{s!r}: {t.encode(s, add_special_tokens=False).ids}")
    for i in (15, 77, 27):
        out.append(f"id {i}: {t.id_to_token(i)!r}")
    return "\n".join(out)


@app.function(image=image, volumes=volumes, cpu=2, memory=16384, timeout=600, retries=0)
def probe4() -> str:
    """Types (not values) of each call field, per kind, in the reader / served / layout records."""
    import json
    import os
    from collections import Counter

    out = []

    def types(calls, label):
        c = Counter()
        for x in calls:
            for k, v in x.items():
                t = type(v).__name__ + (f"[{len(v)}]" if isinstance(v, list) else "")
                c[(x.get("kind"), k, t)] += 1
        for (kind, k, t), n in sorted(c.items(), key=lambda z: (str(z[0][0]), z[0][1])):
            if k in ("readout", "row", "row_mean", "span", "n", "call", "call_index", "reader_row", "reader_row_mean"):
                out.append(f"{label} kind={kind} {k}: {t} x{n}")

    def first(path):
        with open(path, "rb") as f:
            return json.loads(f.readline())

    types(first("/q4/q4b_main_v1/inputs/honest_b.jsonl")["calls"], "reader inputs")
    m = sorted(os.listdir("/q4/q4b_extract_main_v1/meta/honest_b"))[0]
    types(json.load(open(f"/q4/q4b_extract_main_v1/meta/honest_b/{m}"))["calls"], "reader meta")
    types(first("/q3/rerun_strict_segends_v2/inputs/attack.jsonl")["calls"], "served staging")
    types(first("/acts/served_2026-10-14_200651/A/inputs/run1.jsonl")["calls"], "A inputs")
    R = "/acts/served_2026-10-14_200651_extract_A_v1/meta/run1"
    types(json.load(open(f"{R}/{sorted(os.listdir(R))[0]}"))["calls"], "A meta")
    return "\n".join(out)


@app.function(image=image, volumes=volumes, cpu=2, memory=16384, timeout=600, retries=0)
def probe5() -> str:
    """Which served-staging / reader / layout call records lack 'kind', 'src_irid' or 'call_index' (key names only)."""
    import json
    import os
    from collections import Counter

    out = []
    files = [f"/q3/{st}/inputs/{f}" for st in ("rerun_strict_segends_v2", "scaleup_l2_segends_v1", "scaleup_canary_segends_v1",
             "scaleup_topup_segends_v1", "second_attack_segends_v1", "honest_c_segends_v1") for f in sorted(os.listdir(f"/q3/{st}/inputs"))]
    files += [f"/acts/served_2026-10-14_200651/{L}/inputs/{w}.jsonl" for L in ("A", "B") for w in ("run1", "honest_c")]
    files += ["/q4/q4b_main_v1/inputs/honest_b.jsonl", "/q4/q4f_c_v1/inputs/honest_b.jsonl", "/q4/q4f_main_v1_changed/inputs/honest_b.jsonl"]
    for path in files:
        c = Counter()
        ks = Counter()
        with open(path, "rb") as f:
            for line in f:
                r = json.loads(line)
                for x in r["calls"]:
                    miss = tuple(k for k in ("kind", "src_irid", "call_index", "irid", "readout") if k not in x)
                    c[miss] += 1
                    if miss:
                        ks[tuple(sorted(x.keys()))] += 1
        out.append(f"{path}: {dict(c)}; key sets of incomplete calls: {list(ks.items())[:3]}")
    return "\n".join(out)


@app.function(image=image, volumes={"/p2": modal.Volume.from_name("probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13").read_only()},
              cpu=1, memory=4096, timeout=300, retries=0)
def probe6() -> str:
    """Field names, dtypes and shapes of the reader-layout per-row npz files (no values)."""
    import numpy as np

    out = []
    for f in ("fibs/2026-10-14_124948/command.npz", "fibs/2026-10-14_124948/call_end_content.npz",
              "2026-10-14_p2_highstakes/2026-10-14_133519/highstakes_reply_mean.npz"):
        z = np.load(f"/p2/{f}", allow_pickle=False)
        out.append(f + ": " + "; ".join(f"{k} {z[k].dtype} {z[k].shape}" for k in z.files))
    return "\n".join(out)


@app.local_entrypoint()
def main(which: str = "1"):
    print({"1": probe, "2": probe2, "3": probe3, "4": probe4, "5": probe5, "6": probe6}[which].remote())
