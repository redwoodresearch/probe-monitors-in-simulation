"""Diagnose the one output_end unit that failed "in the final user turn of its request" (recheck_2026-10-14_230545,
honest_a|t16_file_indexing). Token ids and counts only; nothing is decoded. Modal CPU, read-only.

  uv run --no-project --with 'modal>=1.2' modal run experiments/2026-10-14_p2_served_recheck/run_diag_userturn.py
"""

import json

import modal

app = modal.App("probe-shift_aswin-surya_sub-p2probes_servedrecheck_kill-2026-11-13")
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.3.3")
vols = {"/acts": modal.Volume.from_name("probe-shift_aswin-surya_sub-p2served_acts_kill-2026-11-14").read_only(),
        "/q3": modal.Volume.from_name("probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08").read_only()}
IM_START, IM_END, TR_O, TR_C, ASSISTANT = 248045, 248046, 248066, 248067, 74455


@app.function(image=image, volumes=vols, cpu=2, memory=16384, timeout=300, retries=0)
def diag() -> dict:
    import os

    import numpy as np

    out = []
    A = [json.loads(x) for x in open("/acts/served_2026-10-14_200651/A/inputs/honest_a.jsonl") if '"t16_file_indexing"' in x[:3000]]
    for r in A:
        st = r["served_staging"]
        S = None
        for f in sorted(os.listdir(f"/q3/{st}/inputs")):
            for line in open(f"/q3/{st}/inputs/{f}"):
                if r["key"] in line[:3000]:
                    x = json.loads(line)
                    if x["key"] == r["key"] and x["epoch"] == r["epoch"]:
                        S = x
        s = np.asarray(S["ids"])
        sc = {(c["irid"], c.get("kind"), c.get("src_irid"), c.get("call_index")): c for c in S["calls"]}
        for c in r["calls"]:
            if c["kind"] != "output_end":
                continue
            v = sc[(c["irid"], c["kind"], c["src_irid"], c["call_index"])]
            rs, n = v["readout"], v["n"]
            starts = np.flatnonzero(s[:n] == IM_START)
            h = int(starts[-1])
            ok = len(starts) >= 2 and s[h + 1] == ASSISTANT and int(starts[-2]) < rs < h
            if ok:
                continue
            tro = int(np.flatnonzero(s[:rs] == TR_O)[-1])
            trc = rs + int(np.flatnonzero(s[rs:] == TR_C)[0])
            span = s[tro + 1:trc]
            prev_im = [int(p) for p in starts if p < rs][-3:]
            out.append({"epoch": r["epoch"], "call_index": c["call_index"],
                        "<|im_start|> inside this tool_response (count)": int((span == IM_START).sum()),
                        "<|im_end|> inside this tool_response (count)": int((span == IM_END).sum()),
                        "token after each in-span <|im_start|> is 'assistant'/'user'/'system' id": [int(s[p + 1]) for p in np.flatnonzero(s[tro + 1:trc] == IM_START) + tro + 1][:5],
                        "last 3 <|im_start|> before readout, offsets from readout": [p - rs for p in prev_im],
                        "<tool_response> offset": tro - rs, "</tool_response> offset": trc - rs, "request end offset": n - rs,
                        "final <|im_start|> (generation prompt) followed by assistant": bool(s[h + 1] == ASSISTANT),
                        "role ids after each <|im_start|> from the readout to the request end (offset, id)":
                            [(int(p) - rs, int(s[p + 1])) for p in starts if p > rs],
                        "role id of the turn holding the readout (token after the last <|im_start|> before it)": int(s[prev_im[-1] + 1]),
                        "role ids: user, assistant, system": [882, ASSISTANT, 8678]})
    return {"units": out}


@app.function(image=image, volumes=vols, cpu=2, memory=16384, timeout=300, retries=0)
def diag_think() -> dict:
    """Think markers that are not part of an assistant turn's opening think block, in A: by enclosing turn's role id."""
    from collections import Counter

    import numpy as np

    THINK_O, THINK_C = 248068, 248069
    res = {}
    for w in ("run1", "run2"):
        c = Counter()
        for line in open(f"/acts/served_2026-10-14_200651/A/inputs/{w}.jsonl"):
            a = np.asarray(json.loads(line)["ids"])
            hdr = set((np.flatnonzero((a[:-3] == IM_START) & (a[1:-2] == ASSISTANT) & (a[2:-1] == 198) & (a[3:] == THINK_O)) + 3).tolist())
            opening_close = set()
            for j in sorted(hdr):
                k = np.flatnonzero(a[j + 1:] == THINK_C)
                m = np.flatnonzero((a[j + 1:] == IM_START) | (a[j + 1:] == IM_END))
                if len(k) and (not len(m) or k[0] < m[0]):
                    opening_close.add(j + 1 + int(k[0]))
            starts = np.flatnonzero(a == IM_START)
            for p in np.flatnonzero((a == THINK_O) | (a == THINK_C)):
                if int(p) in hdr or int(p) in opening_close:
                    continue
                st = starts[starts < p]
                role = int(a[st[-1] + 1]) if len(st) else -1
                c[("<think>" if a[p] == THINK_O else "</think>", role)] += 1
        res[w] = {f"{k[0]} in turn with role id {k[1]}": v for k, v in c.items()}
    return res


@app.local_entrypoint()
def main():
    print(json.dumps(diag.remote(), indent=1))
    print(json.dumps(diag_think.remote(), indent=1))
