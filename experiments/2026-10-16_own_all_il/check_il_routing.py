"""Positive pre-inventory dry run (ownall-il, 10-16): the routing side of run_own_all.py's --il job build on the real IL routing.
Mirrors _cut_jobs_b's per-record checks and _il_bad's structural checks (world token, cut form, staged set); counts and names only."""
import collections, hashlib, json, re, subprocess, sys
from pathlib import Path

wt, rc = sys.argv[1], sys.argv[2]
sys.path.insert(0, f"{wt}/experiments/2026-10-14_p2_family_expansion")
import run_own_all as r  # noqa: E402

git = lambda *a: subprocess.run(["git", "-C", wt, *a], capture_output=True, text=True, check=True).stdout  # noqa: E731
show = lambda p: subprocess.run(["git", "-C", wt, "show", f"{rc}:{p}"], capture_output=True, check=True).stdout  # noqa: E731
best = {}
for p in git("ls-tree", "-r", "--name-only", rc, r.CUT_ROUTING).split():
    m = re.fullmatch(r".*/(.+?__.+?_cutfull)(?:_r([0-9]+))?\.json", p)
    if m and m[1].startswith("il-") and int(m[2] or 1) > best.get(m[1], (0, ""))[0]:
        best[m[1]] = (int(m[2] or 1), p)
fail, per_world, epochs_w, fams, revs, arm_world, vols, batches = [], collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
for stem_, (k, p) in sorted(best.items()):
    rec = json.loads(show(p))
    revs[k] += 1
    folders = {a["meta_arm"].split("@", 1)[0]: re.sub(r"_cutfull_v1_try[0-9]+$", "_cutfull_v1", rec["staged_set"]) + "__" + a["meta_arm"].split("@", 1)[0] for a in rec["arms"]}
    try:
        at = json.loads(show(p[:-5] + "_attest.json"))
    except subprocess.CalledProcessError:
        fail.append(f"{stem_}: no attestation pointer"); continue
    if at.get("failing") != {} or at.get("revision") != k or rec.get("cut") != "full" or rec.get("revision") != k:
        fail.append(f"{stem_}: attestation / cut / revision")
    if any(Path(a["staged_inputs_file"]).parent.parent.name != rec["staged_set"] for a in rec["arms"]):
        fail.append(f"{stem_}: staged inputs outside the staged set")
    if any(Path(a["staged_inputs_file"]).name not in {Path(b["staged_inputs_file"]).name for b in rec["arms"]} for a in rec["arms"]):
        fail.append(f"{stem_}: staged file names")
    vols[rec["volume"] == r.cf.VOLUME] += 1
    batches[rec["batch"]] += 1
    for a in rec["arms"]:
        tok = a["meta_arm"].split("@", 1)[0]
        fo = folders[tok]
        m_ = r.IL_FOLDER.fullmatch(fo)
        w = (m_["w"] or m_["w2"]) if m_ else None
        e = {"folder": fo, "world": w or "?", "key": "_", "staged": f"{rec['staged_set']}/inputs/{a['meta_arm']}.jsonl"}
        why = r._il_bad(e, {"_"}, {m_["fam"]} if m_ else set())  # structural only: the family / key sets come from the scope file later
        if why:
            fail.append(f"{fo}: {why}")
            continue
        fams[m_["fam"]] += 1
        per_world[w] += 1
        epochs_w[w] += a["epochs"]
        aw = a.get("world")
        arm_world[f"{w}|{aw}"] += 1
print(json.dumps({"routing_commit": rc, "il_routing_stems": len(best), "revisions": dict(revs), "arms_by_world": dict(per_world), "epochs_by_world": dict(epochs_w),
                  "epochs_total": sum(epochs_w.values()), "families": dict(sorted(fams.items())), "folder_world|arm_world": dict(arm_world),
                  "on_cutfull_volume": dict(vols), "batches": len(batches), "failures": len(fail), "failure_examples": fail[:8]}, indent=1))
