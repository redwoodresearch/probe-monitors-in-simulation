"""General staged-row resolver (2e 19:3x: for refresh2's generalized gate 2 and the duplicate-content gate; CUT reviews). Input: a score file's path
and an npz row key "<world>|<cell>". Output: that run's STAGED row (the token ids the extractor read): the staged file (its sha256 checked), the
line index, the eval reference, and the ids (handed to the caller inside the container, never printed). Route (CUT's map):
  score path -> its batch (".../cutfull/<batch>/..." or ".../cutfull_ep2/<batch>/...")
  -> the batch's routing records (experiments/2026-10-14_q3_overnight_chain/results/cutfull/routing/<batch>__<target>[_r<k>].json; _attest skipped;
     every revision of a batch must name the same staged set and the same arms)
  -> the one arm whose world list holds the row key's world -> its staged_inputs_file on the record's volume (the file's sha256 must start with
     the record's staged_sha)
  -> the one line of that file whose key's task dir (<launch>/<arm_dir>/<task_dir>/<file>#e<k>) is the row key's cell.
Anything else is returned as an error for that item ("no batch in path", "no routing record", "routing revisions disagree", "world in no arm",
"world in two arms", "staged sha mismatch", "cell not found", "cell on two lines"), never guessed. The uncut (full-length) score files resolve to
the q3cutfull staged sets; that is the pooled build's line. Use: plan() on the laptop or in a container (keys only), then rows() in a container
that mounts the volumes. A self-test (main) resolves the token-prefix check's runs from their uncut files and compares with that check's export."""
import hashlib
import json
import re
import subprocess

import modal

ROUTING = ("a5cbbd766", "experiments/2026-10-14_q3_overnight_chain/results/cutfull/routing")
BATCH = re.compile(r"/cutfull(?:_ep2)?/([^/]+)/")
VOLS = {"/conv": ("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", None), "/aU": ("probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14", None)}
MOUNT_OF = {v[0]: m for m, v in VOLS.items()}
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.1.3")
app = modal.App("probe-shift_aswin-surya_sub-e2esim_resolver_kill-2026-11-16")
mounts = {m: modal.Volume.from_name(n).read_only() for m, (n, _) in VOLS.items()}


def load_routing(commit=ROUTING[0], d=ROUTING[1]):
    """Laptop, keys only: batch -> [routing records], each cut to batch, revision, volume, staged_set and arms."""
    names = subprocess.run(["git", "ls-tree", "--name-only", commit, d + "/"], capture_output=True, check=True, text=True).stdout.split()
    out = {}
    for n in names:
        if n.endswith(".json") and "_attest" not in n:
            r = json.loads(subprocess.run(["git", "show", f"{commit}:{n}"], capture_output=True, check=True).stdout)
            out.setdefault(r["batch"], []).append({"file": n, "revision": r.get("revision"), "volume": r["volume"], "staged_set": r["staged_set"],
                                                   "arms": [{k: a.get(k) for k in ("meta_arm", "world", "staged_inputs_file", "staged_sha")} for a in r["arms"]]})
    return out


def plan(items, routing):
    """items: [{"path": score file, "row": "<world>|<cell>"}] -> per item {batch, mount, staged_file, staged_sha, cell} or {"error": ...}. Pure."""
    out = []
    for it in items:
        m = BATCH.search(it["path"])
        if not m:
            out.append({"error": "no batch in path"})
            continue
        recs = routing.get(m.group(1))
        # a batch can hold several targets (e.g. main and run2): the run folder in the path (<staged set>__<arm>) names the staged set
        ss = next((x.split("__", 1)[0] for x in it["path"].split("/") if "__" in x and not x.endswith(".npz")), None)
        if recs and ss:
            recs = [r for r in recs if r["staged_set"] == ss] or recs
        if not recs:
            out.append({"batch": m.group(1), "error": "no routing record"})
            continue
        if len({json.dumps([r["volume"], r["staged_set"], r["arms"]], sort_keys=True) for r in recs}) != 1:
            out.append({"batch": m.group(1), "error": "routing revisions disagree"})
            continue
        w, c = it["row"].split("|", 1)
        # an arm without a world list (some _r<k> revisions carry "keys" instead) answers to its meta_arm name
        arms = [a for a in recs[0]["arms"] if w in (a["world"] if isinstance(a["world"], list) else [a["world"] or a["meta_arm"]])]
        if len(arms) != 1:
            out.append({"batch": m.group(1), "error": "world in no arm" if not arms else "world in two arms"})
            continue
        out.append({"batch": m.group(1), "mount": MOUNT_OF.get(recs[0]["volume"]), "staged_file": arms[0]["staged_inputs_file"],
                    "staged_sha": arms[0]["staged_sha"], "cell": c, "routing_files": sorted(r["file"] for r in recs)})
    return out


def rows(plans):
    """In a container that mounts the staged volume: -> per plan {staged_file, staged_sha256, line_index, eval_ref, ids (numpy int64)} or an error.
    Each staged file is hashed and scanned once."""
    import numpy as np

    want, res, seen = {}, [dict(p) for p in plans], {}
    for i, p in enumerate(plans):
        if "error" not in p:
            want.setdefault((p["mount"], p["staged_file"]), []).append(i)
    for (mnt, f), idx in want.items():
        h, found = hashlib.sha256(), {}
        cells = {plans[i]["cell"] for i in idx}
        with open(f"{mnt}/{f}", "rb") as fh:
            for li, ln in enumerate(fh):
                h.update(ln)
                for c in cells:
                    if f"/{c}/".encode() in ln:
                        d = json.loads(ln)
                        if d.get("key", "").split("/")[2:3] == [c]:
                            found.setdefault(c, []).append((li, d["key"], np.asarray(d["ids"], np.int64)))
        sha_ = h.hexdigest()
        seen[(mnt, f)] = sha_
        for i in idx:
            r, c = res[i], plans[i]["cell"]
            if not sha_.startswith(plans[i]["staged_sha"]):
                r["error"] = "staged sha mismatch"
            elif len(found.get(c, [])) != 1:
                r["error"] = "cell not found" if not found.get(c) else "cell on two lines"
            else:
                li, ek, ids = found[c][0]
                r |= {"staged_sha256": sha_, "line_index": li, "eval_ref": ek, "ids": ids}
    return res


@app.function(image=image, cpu=2, memory=8192, timeout=1800, retries=0, volumes=mounts)
def selftest(items: list, routing: dict) -> list:
    out = []
    for r in rows(plan(items, routing)):
        ids = r.pop("ids", None)
        out.append(r | ({"n_tokens": int(len(ids)), "ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest()} if ids is not None else {}))
    return out


@app.local_entrypoint()
def main(token_prefix_result: str, out: str):
    """Positive control: every run of the token-prefix check, resolved from one of its uncut score files, must give the same staged file, line,
    eval reference and ids sha256 as that check's export; negative control: a world that no arm holds must give "world in no arm"."""
    tp = json.loads(open(token_prefix_result, "rb").read())
    items, want = [], []
    for lab, r in sorted(tp["runs_export"].items()):
        f = next((x for x in r["files"] if x["line"] == "uncut"), None)
        if f and r["staged"].get("uncut"):
            items.append({"path": f["path"], "row": r["npz_row"]})
            want.append((lab, r["staged"]["uncut"], r["eval_ref"]))
    items.append({"path": items[0]["path"], "row": "no_such_world|" + items[0]["row"].split("|", 1)[1]})
    got = selftest.remote(items, load_routing())
    checks = [{"run": lab, "same_file": g.get("staged_file") == s["file"].split(":", 1)[1], "same_line": g.get("line_index") == s["line_index"],
               "same_eval_ref": g.get("eval_ref") == ek, "same_ids": g.get("ids_sha256") == s["ids_sha256"], "error": g.get("error")}
              for (lab, s, ek), g in zip(want, got)]
    neg = got[-1].get("error")
    res = {"routing": list(ROUTING), "token_prefix_result": token_prefix_result,
           "token_prefix_result_sha256": hashlib.sha256(open(token_prefix_result, "rb").read()).hexdigest(), "checks": checks, "negative_control": neg,
           "pass": all(all(c[k] for k in ("same_file", "same_line", "same_eval_ref", "same_ids")) for c in checks) and neg == "world in no arm"}
    with open(out, "x") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({"pass": res["pass"], "n": len(checks), "failed": [c for c in checks if not all(c[k] for k in ("same_file", "same_line", "same_eval_ref", "same_ids"))],
                      "negative_control": neg}))
    if not res["pass"]:
        raise SystemExit("STOP: the resolver's self-test failed")
