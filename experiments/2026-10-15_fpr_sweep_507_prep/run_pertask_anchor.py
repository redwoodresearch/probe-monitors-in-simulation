"""PERTASK addendum, per-probe ANCHOR (0d via 2e 17:5x; narrowed by 09 @ 635b14564 item 1, 2e 19:3x; POST HOC): per-task agreement of
max_minus_tau. LEAD (every probe): the registered run1 anchor (main = run1 vs sim, yardstick = run1 vs run2). BESIDE IT, for on-policy
probes (own_frac{100,25,10}_L48, own_all*_L48; cross-fitted on run 1's launch): the run2 anchor (main = run2 vs sim, yardstick =
run2 vs run1), block sensitivity_run2_anchor, labelled "sensitivity: run1 shares the training launch".
Per probe and set (S_old, S_new, pooled): Spearman and Pearson r^2 for main and yardstick, and the PAIRED difference
delta = main - yardstick = r(anchor, sim) - r(anchor, other real run), one sign for every row (negative: the sim tracks the anchor less
well than the other real run does). Intervals: the SAME main-level task-bootstrap draws and seed as the Spearman and r^2 addenda (one
fresh default_rng(seed), multinomial per origin over that origin's sorted mains, S_old then S_new; pooled = both origins' draw b
combined). Cells: those with finite run1, run2 and sim values.
Gates: (1) every point equals an independent direct computation on the named arms (r^2: np.corrcoef; Spearman: double-argsort ranks,
when the arm has no ties), and against literal per-block arm names. (2) With --paired-ref/--r2-ref, every lead row reproduces the
earlier Spearman and r^2 addenda exactly by key (point, ci95, finite draws, n_cells), and the sensitivity yardsticks reproduce the
addenda's run1-vs-run2 values (both correlations are symmetric). (3) Where the source record carries Spearman panels (a PERTASK record),
the lead's Spearman main and yardstick equal its run1_vs_sim and run1_vs_run2 panels exactly (same draws, seed and cells). gate.complete
is true when (2) or (3) ran.
UNION STREAM (09 RULE item 30 @ 8c5a39038; ruling via 2e 10-16: union stream + fixed tau, labelled, is enough for the body): when the PERTASK
record carries meta.merge (an --all-origins job), union_stream repeats every multi-origin set (and pooled_all_origins / _without_S4 /
paper_cells where all members exist) with each cell weighted by its canonical main's multiplicity on the final record's union stream
(draws_main over meta.merge's mains, strata and seed: one draw per main across sets); points equal the per-origin stream's (gated).
--final <commit>:<path>@<sha> names that final record: its meta.merge (read in the driver, final_keys) must equal the PERTASK record's.
The client ships sha256-checked bytes only (git show; no jq since a3fd1612 R1, 10-18); every record VALUE and KEY is parsed inside the Modal function
(0d 10-16), which writes a NEW stamped record to the solavg Volume (xfer/pertask/, exclusive create; a remote driver, so it lands under
--detach even if the client dies); the client keeps a results copy.
FAN-OUT (2e 10-16, speed): --fanout runs one `part` container per (label, block); each reads the whole source, so the draws are the same,
and the driver merges the parts in the one-container loop's order. Gates: --repro (two labels computed by one container in one loop vs
the fan-out, both on the whole source's draws: each side's own gates pass, the panel gate restricted to the two labels, AND byte-identical
JSON; writes a keys/counts/shas/booleans record) and --same-as <commit>:<path>@<sha> (every non-meta key byte-identical to a committed
one-container record on the same inputs).
GROUPS (0d via 2e 10-16, Appendix H): --groups (with --fanout --same-as <record> --final) adds groups_r2_delta: the paired R^2 delta
(sim R^2 - rerun R^2) averaged per draw over the six-probe headline group and the four-probe subset (registered @ 19ba8f563), equal weight,
withholding (and counting) draws where a member is not finite; gated by --same-as and by each member's regenerated per-draw array
reproducing that record's stored delta.

  uv run --no-project --with modal==1.6.1 --with numpy==2.3.3 modal run --detach experiments/2026-10-15_fpr_sweep_507_prep/run_pertask_anchor.py \
      --src <commit>:<path> --sha <sha256> [--final <commit>:<path>@<sha256>]
      [--paired-ref <commit>:<path> --paired-sha <sha256> --r2-ref <commit>:<path> --r2-sha <sha256>]
      [--fanout [--repro | --same-as <commit>:<path>@<sha256> [--groups]]]
      [--label <text> (171-pool) | --a-sidecar git:<40-hex>:<path>.json@<sha256> --sd-git <40-hex> --sd-script-sha256 <64-hex> (route C)]
ROUTE C (09 @ READ_LOG 8a4e2bf84 L2213; 2e 10-17): no typed label. The container checks A's label sidecar and re-asserts the Stage D source
by key (routec_label), puts 09's via-Stage-D label into meta.label (and meta.routec_label), and writes the record's label sidecar
<record stem>_LABEL.json {record, record_sha256, label} beside it (label_sidecar); the client keeps a results copy of both.
"""

import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("probe-shift_aswin-surya_sub-pertask_anchor_kill-2026-11-14")
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
         .add_local_file(HERE / "run_fprsweep_507.py", "/root/run_fprsweep_507.py"))  # draws_main: the union stream (09 RULE item 30)
VOL = modal.Volume.from_name("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", version=2)
ARMS = ("run1", "run2", "sim")
ON_POLICY = re.compile(r"^own_(frac(?:100|25|10)|all\w*)_L48$")
SENS = "sensitivity: run1 shares the training launch"  # own_all (trained on run 1 of every origin)
SENS_T196 = "sensitivity: real run 1 AND the sim run share the probe's training launch (S_old only)"  # 09 @ b0a3e0df9: T196-run1-trained (own_frac*)
CAPTION = ("Every probe's lead is anchored on run 1 (registered; 09 @ 635b14564 item 1; READ_LOG @ 93ef051b2). On-policy probes also show a run 2 "
           "anchor beside it: own_frac*, own_a and DT labelled 'sensitivity: real run 1 AND the sim run share the probe's training launch (S_old only)' "
           "(09 @ b0a3e0df9), own_all* labelled 'sensitivity: run1 shares the training launch'.")


ALLO = {"pooled_all_origins": ["S_old", "S_new", "S_newmains", "S_paper493"],  # run_peraction311.COMBOS_ALLO's members, by base origin
        "pooled_all_origins_without_S4": ["S_old", "S_new", "S_newmains without S4", "S_paper493"], "paper_cells": ["S_old", "S_new", "S_paper493"]}
UNION_CAPTION = ("interval: bootstrap over main tasks (one draw per main across sets); thresholds fixed at the point estimate (threshold uncertainty not "
                 "included, so the interval is likely somewhat narrow)")  # 09's body caption, verbatim (via 2e, 10-16)
NOT_READABLE = "not readable (09 item 45(2))"  # run_peraction311.NOT_READABLE: a pooled set below the readability gate at 1e-2 (labelled, not dropped)
T196_RUN1 = re.compile(r"^(own_a|DT_\w+)$")  # 09 @ b0a3e0df9: also trained on T196 run 1 (own_a, DT): the same S_old sensitivity label


SHARE = "config/kappa_launch_share.json"  # the launch-share table (derived by key by check_kappa_launch_share.py; DT's PERTASK-only entry per 09 item 16)
REL = "experiments/2026-10-15_fpr_sweep_507_prep"  # the results copy's repo folder (the label sidecar names the record by this path)
D1_TAIL = "(read, never recomputed; identity-guarded on the bytes read)"  # 09's two templates' shared tail (READ_LOG @ 8a4e2bf84 L2213)


def routec_label(rec: dict, side: tuple, args: dict) -> str:
    """Route C (09 @ READ_LOG 8a4e2bf84 L2213; 2e 10-17): no typed label. side = (A's label sidecar bytes, its pinned sha256), checked
    before the parse; it must be exactly {record, record_sha256, label}, name the A that --final pins and route C, and be PRELIMINARY (no
    FINAL-release path exists). The Stage D source record is re-asserted by key (09 GATES: record sha == pin is checked by prep): meta.git
    == --sd-git, meta.script_sha256 == --sd-script-sha256, meta.pertask.reference_record == --final's A, and its D1 label equal to 09's
    template on A's sidecar label. A commits are full 40-hex (2e). Returns 09's via-Stage-D label:
    "<A sidecar label>; D1 read from route C pooled A @ <A commit> via Stage D @ <Stage D record's commit> (read, never recomputed; ...)"."""
    b_, want_ = side
    if hashlib.sha256(b_).hexdigest() != want_:
        raise SystemExit("route C: A's label sidecar sha256 differs from its pin (STOP)")
    sc = json.loads(b_)
    if not isinstance(sc, dict) or set(sc) != {"record", "record_sha256", "label"} or not all(isinstance(v, str) and v.strip() for v in sc.values()):
        raise SystemExit("route C: A's label sidecar is not {record, record_sha256, label} as non-empty strings (STOP)")
    a_ref, a_sha = (args.get("final") or "").rsplit("@", 1) if "@" in (args.get("final") or "") else ("", "")
    a_commit, sd_commit = a_ref.split(":", 1)[0], args["src"].split(":", 1)[0]
    if not (re.fullmatch(r"[0-9a-f]{40}", a_commit) and re.fullmatch(r"[0-9a-f]{40}", sd_commit)):
        raise SystemExit("route C: --final and --src must cite full 40-hex commits (STOP)")
    lab = sc["label"]
    if sc["record_sha256"] != a_sha or "route C" not in lab or "PRELIMINARY" not in lab or "171-pool" in lab:
        raise SystemExit("route C: A's label sidecar does not name --final's A, or its label is not route C and PRELIMINARY (STOP)")
    m_ = rec.get("meta") or {}
    if m_.get("git") != args["sd_git"] or m_.get("script_sha256") != args["sd_script_sha256"]:
        raise SystemExit("route C: the Stage D record's meta.git or meta.script_sha256 differs from its pin (STOP)")
    if (m_.get("pertask") or {}).get("reference_record") != f"{a_ref} ({a_sha})":
        raise SystemExit("route C: the Stage D record's reference record is not --final's A (STOP)")
    d1_ = rec.get("d1") or {}
    if d1_.get("mode") != "read" or d1_.get("labels") != [f"{lab}; D1 read from route C pooled A @ {a_commit} {D1_TAIL}"]:
        raise SystemExit("route C: the Stage D record's D1 label is not 09's template on A's sidecar label (STOP)")
    return f"{lab}; D1 read from route C pooled A @ {a_commit} via Stage D @ {sd_commit} {D1_TAIL}"


def final_keys(blobs: dict, args: dict, meta: dict) -> tuple:
    """a3fd1612 R1 (10-18): what the client used to read with jq on the laptop from A (a record holding measured values), read here from
    sha-checked bytes: A's meta.merge, A's (cell, canonical main) pairs from its all-origins rows (as the jq program '[.rows[] | to_entries[]
    | select(.key | contains(" (all-origins draws)")) | .value | select(type == "object" and (.cell_hits_draw0 | type) == "object") |
    [.cohort_cell_keys, .cell_hits_draw0.mains] | transpose[]] | unique' did, except that unequal lengths or a non-string STOP here instead
    of padding with null), and with --groups A's meta.auroc06_subset. The client ships A's committed bytes (gzip when --final's path ends
    .gz) under blobs["final_raw"] with --final's sha (the decompressed JSON's) and parses nothing; they are dropped here, so the parts never
    receive them. The Stage D record's meta.merge (the client's old '.meta.merge != null') is prep's check: it STOPs on --final without
    meta.merge and on meta.merge without --final."""
    import gzip

    fr_ = blobs.get("final_raw")
    blobs = {k_: v_ for k_, v_ in blobs.items() if k_ != "final_raw"}
    if fr_ is None:
        if args.get("final") or args.get("groups"):
            raise SystemExit("--final (or --groups) given, but A's bytes were not shipped (STOP)")
        return blobs, args, meta
    ref_, want_ = args["final"].rsplit("@", 1)
    b_ = gzip.decompress(fr_) if ref_.endswith(".gz") else fr_
    if hashlib.sha256(b_).hexdigest() != want_:
        raise SystemExit("--final: sha256 of A's shipped bytes (container; STOP)")
    a_ = json.loads(b_)
    del b_, fr_
    pairs_ = set()
    for by_ in (a_.get("rows") or {}).values():
        for k_, v_ in by_.items():
            if " (all-origins draws)" in k_ and isinstance(v_, dict) and isinstance(v_.get("cell_hits_draw0"), dict):
                cs_, ms_ = v_.get("cohort_cell_keys"), v_["cell_hits_draw0"].get("mains")
                if not (isinstance(cs_, list) and isinstance(ms_, list) and len(cs_) == len(ms_) and all(isinstance(x_, str) for x_ in cs_ + ms_)):
                    raise SystemExit(f"--final: {k_}: cohort_cell_keys and cell_hits_draw0.mains are not equal-length lists of strings (STOP)")
                pairs_ |= set(zip(cs_, ms_))
    blobs |= {"final_merge": json.dumps((a_.get("meta") or {}).get("merge")).encode(), "final_cell_mains": json.dumps(sorted(map(list, pairs_))).encode()}
    if args.get("groups"):  # memberships by key: six = the client's full_override.HEADLINE_SIX (code); four = A's meta.auroc06_subset
        au_ = (a_.get("meta") or {}).get("auroc06_subset") or {}
        if au_.get("membership_pin") != "19ba8f563" or len(au_.get("members") or []) != 4:
            raise SystemExit("--groups: the final record's meta.auroc06_subset is not the four-probe subset registered @ 19ba8f563")
        gr_ = {"headline_six": {"members": list(args["groups"]["six"]), "label": "six-probe headline group",
                                "source": "full_override.HEADLINE_SIX (Stage A's headline_six_full, run_peraction311.FULL_GROUPS)"},
               "four_probe_subset": {"members": list(au_["members"]), "label": "four-probe subset (registered @ 19ba8f563)",
                                     "source": f"Stage A record {args['final']}: .meta.auroc06_subset (membership_pin 19ba8f563; 09 RULE 45(3) keeps it fixed)",
                                     "registered": au_}}
        args = args | {"groups": gr_, "groups_keep": sorted({f"{f} {ro}" for g in gr_.values() for f in g["members"] for ro in ("C", "R")})}
        meta = meta | {"groups_r2_delta": {"rule": "0d via 2e 10-16 (Appendix H): paired R^2 delta group means per draw, equal weight; a re-draw of the "
                                                   "per-draw values (none are stored), byte-identical to --same-as's record (0d OK on the deviation)",
                                           "memberships": {k: {"members": v["members"], "label": v["label"], "source": v["source"]} for k, v in gr_.items()}}}
    return blobs, args, meta


def label_sidecar(nm: str, sha: str, out: dict):
    """Route C: the record's label sidecar {record, record_sha256, label} for the slot readout and post_pertask (bytes), from the label the
    container composed (meta.routec_label); None for a 171-pool run or a failed gate (no sidecar for a _GATEFAIL record)."""
    lab = (out["meta"].get("routec_label") or {}).get("label")
    if not lab or not out["gate"]["pass"]:
        return None
    return (json.dumps({"record": f"{REL}/results/{nm}", "record_sha256": sha, "label": lab}, indent=1) + "\n").encode()


def share_of(label: str, table: dict) -> tuple:
    """(run1_shares, sim_shares) for the label's family, by the launch-share table (09 @ 4c095dec9 item 16: the run2 block is by launch share)."""
    name = label.rsplit(" ", 1)[0]
    e = table["families"].get(name) or table["families"].get(name.removesuffix("|uncut"))
    if e is None:
        raise SystemExit(f"{label}: no launch-share entry in {SHARE}")
    return (set(e.get("run1_shares", e.get("pertask_run1_shares", []))), set(e.get("sim_shares", e.get("pertask_sim_shares", []))))


def anchor_of(label: str, table: dict) -> list:
    r1, _ = share_of(label, table)
    return [("lead", "run1", "run2")] + ([("sensitivity_run2_anchor", "run2", "run1")] if r1 else [])


@app.function(image=image, cpu=(2, 2), memory=(4096, 4096), timeout=4 * 3600, retries=0, volumes={"/sol": VOL})  # union stream: ~2x the draws work
def anchored(blobs: dict, args: dict, meta: dict) -> dict:
    """Reads the sha-checked records shipped as bytes (the client never parses a value; 0d 10-16), computes the blocks, runs every gate (finish)
    and writes the record to the solavg Volume (remote driver: it lands even if the client dies). --fanout: one `part` container per
    (label, block), merged here; --repro: the fan-out vs one container on a two-label subset, and nothing else."""
    blobs, args, meta = final_keys(blobs, args, meta)  # a3fd1612 R1: A's keys read here, not on the laptop; A's bytes go no further
    pb_ = {k_: v_ for k_, v_ in blobs.items() if k_ != "checked"} | {"checked": {"src": blobs["checked"]["src"]}}  # what a part needs
    fan_ = (lambda us: list(part.map(us, kwargs={"blobs": pb_, "args": args, "meta": {"label": ""}}))) if args.get("fanout") else None
    if args.get("repro"):
        out = repro(pb_, args, meta)
    else:
        out = build(blobs, args, meta, fan_)
    dr_ = out.pop("_draws", None)
    if args.get("same_as"):  # a committed one-container record on the same inputs: every non-meta key must be byte-identical (JSON)
        ks_, ref_ = ("gate", "lead", "sensitivity_run2_anchor", "union_stream"), json.loads(blobs["checked"]["same_as"][0])
        a_, b_ = json.dumps({k: out[k] for k in ks_}, default=float).encode(), json.dumps({k: ref_[k] for k in ks_}).encode()
        out["gate"]["same_as"] = {"record": args["same_as"], "equal": a_ == b_, "sha256_mine": hashlib.sha256(a_).hexdigest(),
                                  "sha256_ref": hashlib.sha256(b_).hexdigest(), "keys": list(ks_)}
        out["gate"]["pass"] = out["gate"]["pass"] and a_ == b_
    if args.get("groups"):  # after --same-as, which compares the four keys above only
        out["groups_r2_delta"] = groups_reduce(dr_, ref_, args["groups"], out["meta"]["interval_labels"]["thresholds"])
        out["gate"]["groups"] = out["groups_r2_delta"]["gate"]
        out["gate"]["pass"] = out["gate"]["pass"] and out["groups_r2_delta"]["gate"]["pass"]
    blob = json.dumps(out, default=float).encode()
    nm = f"{args['name']}{'' if out['gate']['pass'] else '_GATEFAIL'}.json"
    p_ = Path("/sol/xfer/pertask") / nm
    p_.parent.mkdir(parents=True, exist_ok=True)
    with p_.open("xb") as fh:
        fh.write(blob)
    sha_ = hashlib.sha256(blob).hexdigest()
    sc_ = label_sidecar(nm, sha_, out)
    if sc_:  # route C: written after the record, exclusive
        with (p_.parent / f"{p_.stem}_LABEL.json").open("xb") as fh:
            fh.write(sc_)
    VOL.commit()
    return {"name": nm, "volume_path": str(p_).removeprefix("/sol/"), "sha256": sha_, "gate": out["gate"], "blob": blob, "sidecar": sc_}


@app.function(image=image, cpu=(2, 2), memory=(4096, 4096), timeout=2 * 3600, retries=0)
def part(only, blobs: dict, args: dict, meta: dict) -> dict:
    """Fan-out: _anchored for the given (label, block) pairs only (None: every pair, i.e. the one-container loop). Each part reads the whole
    source, so the draws (over every label's mains) and each pair's numbers equal the one-container run's."""
    data, anchors, seed, D, post = prep(blobs, args, meta)
    return _anchored(data, anchors, seed, D, post.get("bootstrap"), post.get("shares"), post.get("union"),
                     only=None if only is None else {tuple(u_) for u_ in only}, keep=set(args["groups_keep"]) if args.get("groups_keep") else None)


def units(labels, anchors: dict) -> list:
    """One fan-out unit per (label, block), in the one-container loop's order (sorted labels, then each label's anchors)."""
    return [[(lb, b_)] for lb in sorted(labels) for b_, _, _ in anchors[lb]]


def merge_parts(parts: list, us: list) -> dict:
    """The parts (one per unit of `us`, in units() order) as one _anchored result: blocks in label order, and each gate list with the
    per-origin entries first and the union entries ('union|') after, as the one-container loop appends them."""
    if not us or len(parts) != len(us):
        raise SystemExit(f"fan-out: {len(parts)} parts for {len(us)} units (none, or not one per unit)")
    out, uni = {"lead": {}, "sensitivity_run2_anchor": {}}, None
    gate, tail = {"compared": 0, "mismatch": [], "spearman_direct_skipped_ties": []}, {"mismatch": [], "spearman_direct_skipped_ties": []}
    for p in parts:
        if ("union_stream" in p) != ("union_stream" in parts[0]):
            raise SystemExit("fan-out: parts disagree on whether there is a union stream")
        gate["compared"] += p["gate"]["compared"]
        for k in tail:
            gate[k] += [e for e in p["gate"][k] if not e.startswith("union|")]
            tail[k] += [e for e in p["gate"][k] if e.startswith("union|")]
        if "union_stream" in p:
            u_ = p["union_stream"]
            if uni is None:
                uni = {b: {} for b in out} | {k: v for k, v in u_.items() if k not in out}
            elif any(uni[k] != v for k, v in u_.items() if k not in out):
                raise SystemExit("fan-out: parts disagree on the union stream's draws, seed or mains")
        for dst_, src_ in [(out, p)] + ([(uni, p["union_stream"])] if "union_stream" in p else []):
            for b_ in out:
                if set(src_[b_]) & set(dst_[b_]):
                    raise SystemExit(f"fan-out: two parts returned the same {b_} label")
                dst_[b_] |= src_[b_]
    for k in tail:
        gate[k] += tail[k]
    got_, want_ = {(lb, b_) for b_ in out for lb in out[b_]}, {u_ for us_ in us for u_ in map(tuple, us_)}
    if got_ != want_:
        raise SystemExit(f"fan-out: merged (label, block) pairs differ from the units ({len(got_ ^ want_)} differ)")
    return {"gate": gate | {"pass": not gate["mismatch"]}} | out | ({"union_stream": uni} if uni is not None else {})


def repro(blobs: dict, args: dict, meta: dict) -> dict:
    """--repro: on two labels (the first with only a lead block, and the first with a run2 block), one `part` computes all their (label,
    block) pairs in ONE loop (the one-container loop, filtered) and one `part` per pair runs the fan-out; both read the whole source, so the
    draws are the whole source's (Codex r2: a label subset would change the mains and the draws). PASS needs (i) each side's own gates to
    pass: the _anchored gate (direct computation + union points) and the finished record's gate, with the panel gate (3) restricted to the
    two labels so it applies, and (ii) the merged result and the finished record byte-identical (JSON). The repro record carries keys,
    counts, shas and booleans only."""
    data, anchors, seed, D, post = prep(blobs, args, json.loads(json.dumps(meta)))
    sub_ = sorted([lb for lb in sorted(data) if len(anchors[lb]) == 1][:1] + [lb for lb in sorted(data) if len(anchors[lb]) == 2][:1])
    us_ = units(sub_, anchors)
    one_ = part.spawn([u_ for uu_ in us_ for u_ in uu_], blobs, args, {"label": ""})
    fan_ = merge_parts(list(part.map(us_, kwargs={"blobs": blobs, "args": args, "meta": {"label": ""}})), us_)
    one_ = one_.get()
    sd_ = {lb: data[lb] for lb in sub_}
    ps_ = post | {"panels": None if post.get("panels") is None else {lb: v for lb, v in post["panels"].items() if lb in sub_}}
    sides_ = {"one": one_, "fan": fan_}
    fins_ = {k: finish(v, sd_, anchors, seed, D, ps_) for k, v in sides_.items()}
    raw_ = {k: json.dumps(v, default=float).encode() for k, v in sides_.items()}
    fin_ = {k: json.dumps(v, default=float).encode() for k, v in fins_.items()}
    gates_ = {k: {"anchored_pass": sides_[k]["gate"]["pass"], "anchored_compared": sides_[k]["gate"]["compared"],
                  "anchored_mismatch_n": len(sides_[k]["gate"]["mismatch"]), "record_pass": g_["pass"], "record_complete": g_["complete"],
                  "panels_compared": (g_["record_panels"] or {}).get("compared"), "panels_mismatch_n": len((g_["record_panels"] or {}).get("mismatch") or [])}
              for k, g_ in ((k, fins_[k]["gate"]) for k in sides_)}
    side_ok_ = {k: g_["anchored_pass"] and g_["record_pass"] and (g_["record_complete"] or ps_["panels"] is None) for k, g_ in gates_.items()}
    eq_ = {"anchored_equal": raw_["one"] == raw_["fan"], "record_equal": fin_["one"] == fin_["fan"]}
    ok_ = all(side_ok_.values()) and all(eq_.values())
    why_ = [f"{k}: own gates did not pass" for k, v in side_ok_.items() if not v] + [f"not {k}" for k, v in eq_.items() if not v]
    return {"meta": meta | {"seed": seed, "draws": D}, "gate": {"pass": ok_, "mismatch": why_},
            "repro": {"labels": sub_, "units": [u_[0] for u_ in us_], "union_stream": "union_stream" in one_, "side_gates": gates_, "side_ok": side_ok_}
            | eq_ | {"sha256": {k: hashlib.sha256(v).hexdigest() for k, v in raw_.items()}, "sha256_record": {k: hashlib.sha256(v).hexdigest() for k, v in fin_.items()}}}


def build(blobs: dict, args: dict, meta: dict, fan=None) -> dict:
    """The record from the shipped bytes (no Volume write; the synthetic check calls this directly). fan: units -> parts (the fan-out)."""
    data, anchors, seed, D, post = prep(blobs, args, meta)
    if fan is None:
        res = _anchored(data, anchors, seed, D, post.get("bootstrap"), post.get("shares"), post.get("union"),
                        keep=set(args["groups_keep"]) if args.get("groups_keep") else None)
    else:
        us_ = units(data, anchors)
        res = merge_parts(fan(us_), us_)
    dr_ = {}  # --groups: the kept per-draw arrays, out of the result before finish (the record's bytes are unchanged)
    for stream_, blk_ in (("per_origin_streams", res["lead"]), ("union_stream", (res.get("union_stream") or {}).get("lead", {}))):
        for lb, o_ in blk_.items():
            for s_, r_ in o_["sets"].items():
                if "_draws" in r_:
                    dr_[(stream_, lb, s_)] = r_.pop("_draws")
    out = finish(res, data, anchors, seed, D, post)
    return out | ({"_draws": dr_} if args.get("groups_keep") else {})


def groups_reduce(dr: dict, ref: dict, groups: dict, thresholds: str) -> dict:
    """--groups (0d via 2e 10-16): the paired R^2 delta (sim R^2 - rerun R^2, run1-anchored lead) averaged over a probe group per draw, equal
    weight, from the regenerated per-draw arrays (same draws: the record is byte-identical to `ref` by --same-as). A draw where any member's
    delta is not finite is WITHHELD (counted). A set where a member lacks the row is 'unavailable'. Gate: each member's per-draw array
    reproduces `ref`'s stored delta (finite_draws, ci95 and draws exactly; the point is the full-sample value, equal by --same-as)."""
    import numpy as np

    gate = {"compared": 0, "mismatch": []}
    blk = {"per_origin_streams": ref["lead"], "union_stream": ref["union_stream"]["lead"]}

    def ci(x, n):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) >= 0.95 * n else None

    want = {(stream, m, s) for stream, by in blk.items() for g in groups.values() for ro in ("C", "R") for f in g["members"]
            for m in [f"{f} {ro}"] for s in (by.get(m) or {}).get("sets", {})}
    if want - set(dr):
        raise SystemExit(f"--groups: {len(want - set(dr))} member rows of the reference have no regenerated per-draw array")
    for (stream, lb, s), bt in sorted(dr.items()):
        st = blk[stream][lb]["sets"][s]["r2"]["delta_main_minus_yardstick"]
        fin = bt[np.isfinite(bt)]
        gate["compared"] += 1
        if len(bt) != st["draws"] or int(len(fin)) != st["finite_draws"] or ci(fin, len(bt)) != st["ci95"]:
            gate["mismatch"].append(f"{stream}|{lb}|{s}: the regenerated per-draw delta does not reproduce the stored delta")
    out = {"interval_label": f"fixed-threshold ({thresholds})", "groups": groups,
           "withhold_rule": "a draw is withheld from a group when any member's delta is not finite on it (counted in withheld_draws)"}
    for stream, by in blk.items():
        for gname, g in groups.items():
            for ro in ("C", "R"):
                mem = [f"{f} {ro}" for f in g["members"]]
                for s in sorted({s_ for m in mem for s_ in (by.get(m) or {}).get("sets", {})}):
                    miss = [m for m in mem if s not in (by.get(m) or {}).get("sets", {})]
                    nr_ = [f"{m} | {s}" for m in mem if m not in miss and by[m]["sets"][s].get("readability")]
                    if nr_:  # 09 (via 2e 10-16): a not-readable member withholds the group mean as a whole (never averaged over the rest);
                        e = {"withheld": f"{NOT_READABLE}: {nr_}"} | ({"members_missing": miss} if miss else {})  # first (CUT r1 #3)
                    elif miss:
                        e = {"unavailable": "a member has no row for this set", "members_missing": miss}
                    else:
                        M = np.vstack([dr[(stream, m, s)] for m in mem])
                        ok = np.isfinite(M).all(axis=0)
                        pts = [by[m]["sets"][s]["r2"]["delta_main_minus_yardstick"]["point"] for m in mem]
                        c_ = ci(M[:, ok].mean(axis=0), M.shape[1])
                        e = {"point": float(np.mean(pts)) if all(p is not None and np.isfinite(p) for p in pts) else None, "draws": int(M.shape[1]),
                             "finite_draws": int(ok.sum()), "withheld_draws": int((~ok).sum()), "ci95": c_,
                             "excludes_0": (c_[0] > 0 or c_[1] < 0) if c_ else None}
                    out.setdefault(stream, {}).setdefault(gname, {}).setdefault(ro, {})[s] = e | {"members": mem}
    return out | {"gate": gate | {"pass": not gate["mismatch"] and gate["compared"] > 0}}


def prep(blobs: dict, args: dict, meta: dict) -> tuple:
    """Parses the shipped bytes: (data, anchors, seed, D, post)."""
    for k_, (b_, want_) in blobs["checked"].items():  # the client's sha256 checks, repeated here on the bytes received
        if hashlib.sha256(b_).hexdigest() != want_:
            raise SystemExit(f"{k_}: sha256 of the shipped bytes")
    rec = json.loads(blobs["checked"]["src"][0])
    if "gate" in rec and rec["gate"].get("pass") is not True:  # a source PERTASK record whose own gate failed (panels None) is never used
        raise SystemExit("source record's gate did not pass")
    lab_c = routec_label(rec, blobs["a_sidecar"], args) if args.get("a_sidecar") else None  # route C: composed here from A's sidecar
    if (rec.get("d1") or {}).get("mode") == "read" and not (args.get("label") or lab_c):  # 09 / 2e 10-17: the record names the D1 read in meta.label
        raise SystemExit("the source record read D1 from pooled A: pass --label <the full label 09 gives> (STOP)")
    data = {}
    for lb, by in rec["rows"].items():
        for og, o in by.items():
            if "per_task" not in o:  # a withheld row of the source record: no tasks here
                continue
            d = data.setdefault(lb, {}).setdefault(og, {})
            for r in o["per_task"]:
                d.setdefault(r["cell"], {"main": r["main"]})[r["arm"]] = r["max_minus_tau"]
    table_ = json.loads(blobs["share"])
    anchors = {lb: anchor_of(lb, table_) for lb in data}
    shares_ = {lb: [sorted(x) for x in share_of(lb, table_)] for lb in data}
    seed, D = rec["bootstrap"]["seed"], rec["bootstrap"]["draws"]
    if lab_c:  # where --label's text goes on the 171-pool path (main), then the Stage D record's own label as before
        meta["label"] += f"; {lab_c}"
        meta["routec_label"] = {"label": lab_c, "a_sidecar": args["a_sidecar"], "rule": "09 @ READ_LOG 8a4e2bf84 L2213 (via Stage D template)",
                                "stage_d_reasserted": {"git": args["sd_git"], "script_sha256": args["sd_script_sha256"], "d1_label": "09's template, exact"}}
    meta["label"] += ("; " + rec["meta"]["pertask"]["label"] if "pertask" in rec["meta"] and "label" in rec["meta"]["pertask"] else "")
    meta["interval_labels"] = {  # 09 RULE item 30 @ 8c5a39038: (3) these intervals hold each pipeline's draw-0 tau fixed; (1) a combo set's draws are per origin
        "thresholds": "thresholds fixed at the point estimate (each pipeline's draw-0 tau; 09 RULE item 30(3))",
        "sets": {set_: ("per-origin draw streams (registered; a shared main drawn independently per origin): appendix unless re-run on the union "
                        "stream (09 RULE item 30(1))") if len(ogs) > 1 else "one origin's draws" for set_, ogs in (rec["bootstrap"].get("sets") or {}).items()}}
    uni_, uni_why_ = None, None
    mg_ = rec.get("meta", {}).get("merge")
    if mg_ is None:
        if args.get("final"):
            raise SystemExit("--final given, but the PERTASK record carries no meta.merge: no union stream to compute")
        uni_why_ = "the PERTASK record carries no meta.merge (not an --all-origins job): no union stream"
    else:  # the final record (--final) whose union stream this is: its meta.merge (final_keys, driver) must equal the PERTASK record's
        fm_ = json.loads(blobs["final_merge"]) if blobs.get("final_merge") else None
        fsha_ = args["final"].rsplit("@", 1)[1] if args.get("final") else None
        if fm_ is None or any(fm_.get(k_) != mg_.get(k_) for k_ in ("mains", "env_of_main", "seed")) or fsha_ not in rec["meta"].get("pertask", {}).get("reference_record", ""):
            raise SystemExit("--final: missing, or its meta.merge (mains, strata, seed) differs from the PERTASK record's, or it is not the PERTASK record's reference record")
        cmp_ = json.loads(blobs["final_cell_mains"]) if blobs.get("final_cell_mains") else []  # [cell, canonical main] (final_keys, driver)
        uni_ = {k_: mg_[k_] for k_ in ("mains", "env_of_main", "seed")} | {"cell_main_pairs": cmp_}
        meta["union_stream"] = {"final_record": args["final"], "mains": len(mg_["mains"]), "seed": mg_["seed"], "caption": UNION_CAPTION,
                                "cell_mains": f"{len(cmp_)} (cell, canonical main) pairs from the final record's all-origins rows (per-set S_old rows key "
                                              "their mains by cell)"}
    post = {"meta": meta, "name": args["name"], "panels": rec.get("panels"), "bootstrap": rec.get("bootstrap"), "union": uni_, "union_unavailable": uni_why_,
            "shares": shares_, "readability": rec.get("readability"),
            "refs": ({"paired": json.loads(blobs["checked"]["paired"][0]), "r2": json.loads(blobs["checked"]["r2"][0]), "paired_name": args["paired_name"],
                      "r2_name": args["r2_name"]} if "paired" in blobs["checked"] else None)}
    return data, anchors, seed, D, post


def _anchored(data: dict, anchors: dict, seed: list, D: int, boot: dict = None, shares: dict = None, uni: dict = None, only: set = None,
              keep: set = None) -> dict:
    """only: the (label, block) pairs to compute (the fan-out); None computes every pair. The draws are the same either way. keep (--groups):
    lead labels whose per-draw R^2 delta (main - yardstick) each set also returns under '_draws' (popped before the record is built)."""
    import numpy as np

    sel = lambda lb, b_: only is None or (lb, b_) in only  # noqa: E731

    def rank_avg(a):
        o = np.argsort(a, kind="mergesort")
        s_ = a[o]
        st = np.flatnonzero(np.r_[True, s_[1:] != s_[:-1]])
        cnt = np.diff(np.r_[st, len(a)])
        r = np.empty(len(a))
        r[o] = np.repeat(st + (cnt - 1) / 2.0, cnt) + 1.0
        return r

    def spear(x, y):  # as run_pertask_paired.py
        if len(x) < 3:
            return np.nan
        rx, ry = rank_avg(x), rank_avg(y)
        rx, ry = rx - rx.mean(), ry - ry.mean()
        den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
        return float((rx * ry).sum() / den) if den > 0 else np.nan

    def r2(x, y):  # as run_pertask_r2.py
        if len(x) < 3:
            return np.nan
        xc, yc = x - x.mean(), y - y.mean()
        den = (xc ** 2).sum() * (yc ** 2).sum()
        return float((xc * yc).sum() ** 2 / den) if den > 0 else np.nan

    def summ(point, bt, Dn=None):
        Dn = D if Dn is None else Dn
        fin = bt[np.isfinite(bt)]
        return {"point": point, "draws": Dn, "finite_draws": int(len(fin)),
                "ci95": [float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))] if len(fin) >= 0.95 * Dn else None}

    def direct(m, x, y):  # independent of rank_avg/spear/r2
        if m == "r2":
            return float(np.corrcoef(x, y)[0, 1] ** 2)
        if len(np.unique(x)) < len(x) or len(np.unique(y)) < len(y):
            return None
        return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])

    METRICS = {"spearman": spear, "r2": r2}

    def one_set(lb, block, a0, a1, set_, ogs, X, cols, mult, Dn, tag=""):  # one set's points, draws and gate (1); mult(og, b): draw b's main multiplicities
        v = {a: np.concatenate([X[og][a] for og in ogs]) for a in ARMS}
        off = np.cumsum([0] + [len(cols[og]) for og in ogs])[:-1]
        bt = {(m, k): np.empty(Dn) for m in METRICS for k in ("main", "yardstick")}
        for b in range(Dn):
            ii = np.concatenate([np.repeat(np.arange(len(cols[og])), mult(og, b)[cols[og]]) + of for og, of in zip(ogs, off)])
            for m, f in METRICS.items():
                bt[m, "main"][b], bt[m, "yardstick"][b] = f(v[a0][ii], v["sim"][ii]), f(v[a0][ii], v[a1][ii])
        res = {"n_cells": int(len(v[a0]))}
        for m, f in METRICS.items():
            pm, py = f(v[a0], v["sim"]), f(v[a0], v[a1])
            d_ = summ(pm - py, bt[m, "main"] - bt[m, "yardstick"], Dn) | {"excludes_0": None}
            d_["excludes_0"] = (d_["ci95"][0] > 0 or d_["ci95"][1] < 0) if d_["ci95"] else None
            res[m] = {"main": summ(pm, bt[m, "main"], Dn), "yardstick": summ(py, bt[m, "yardstick"], Dn), "delta_main_minus_yardstick": d_}
            for k, other in (("main", "sim"), ("yardstick", a1)):  # gate (1): direct computation on the named arms
                dv, mine = direct(m, v[a0], v[other]), res[m][k]["point"]
                if dv is None:
                    gate["spearman_direct_skipped_ties"].append(f"{tag}{block}|{lb}|{set_}|{k}")
                    continue
                gate["compared"] += 1
                if not (abs(dv - mine) <= 1e-12 or (np.isnan(dv) and np.isnan(mine))):
                    gate["mismatch"].append(f"{tag}{block}|{lb}|{set_}|{m}|{k}: {mine!r} vs direct {dv!r}")
            # the block's pairing: literal arm names per block, independent of a0/a1
            x_, o_ = {"lead": ("run1", "run2"), "sensitivity_run2_anchor": ("run2", "run1")}[block]
            for k, y_ in (("main", "sim"), ("yardstick", o_)):
                dv, mine = direct(m, v[x_], v[y_]), res[m][k]["point"]
                if dv is None:
                    gate["spearman_direct_skipped_ties"].append(f"{tag}{block}|{lb}|{set_}|{k}|literal")
                    continue
                gate["compared"] += 1
                if not (abs(dv - mine) <= 1e-12 or (np.isnan(dv) and np.isnan(mine))):
                    gate["mismatch"].append(f"{tag}{block}|{lb}|{set_}|{m}|{k}: {mine!r} vs literal {x_} vs {y_} {dv!r}")
        if keep is not None and block == "lead" and lb in keep:  # --groups: the per-draw paired R^2 delta, the same array summ() saw
            res["_draws"] = bt["r2", "main"] - bt["r2", "yardstick"]
        return res

    if boot and "origins" in boot:  # the source record's own origins and sets, by key (pertask_reduce's draws: origins in this order)
        origins, sets = tuple(boot["origins"]), {k_: list(v_) for k_, v_ in boot["sets"].items()}
    else:  # a two-origin record from before the generalization
        origins, sets = ("S_old", "S_new"), {"S_old": ["S_old"], "S_new": ["S_new"], "pooled": ["S_old", "S_new"]}
    rng = np.random.default_rng(seed)
    mains = {og: sorted({c_["main"] for lb in data for c_ in data[lb].get(og, {}).values()}) for og in origins}
    counts = {og: rng.multinomial(len(mains[og]), np.full(len(mains[og]), 1 / len(mains[og])), size=D) for og in origins if mains[og]}  # as pertask_reduce
    out, gate = {"lead": {}, "sensitivity_run2_anchor": {}}, {"compared": 0, "mismatch": [], "spearman_direct_skipped_ties": []}
    for lb in sorted(data):
        if not any(sel(lb, b_) for b_, _, _ in anchors[lb]):
            continue
        X, mcol = {}, {}
        for og in [o_ for o_ in origins if o_ in data[lb]]:
            cells = sorted(c for c, v in data[lb][og].items() if all(v.get(a) is not None and np.isfinite(v[a]) for a in ARMS))
            X[og] = {a: np.array([data[lb][og][c][a] for c in cells], float) for a in ARMS}
            mi = {m: i for i, m in enumerate(mains[og])}
            mcol[og] = np.array([mi[data[lb][og][c]["main"]] for c in cells], int)
        r1s_, sms_ = (set(shares[lb][0]), set(shares[lb][1])) if shares else (set(), set())
        for block, a0, a1 in anchors[lb]:
            if not sel(lb, block):
                continue
            out[block][lb] = {"anchor": a0, "other_real": a1, "main_pair": f"{a0}_vs_sim", "yardstick_pair": f"{a0}_vs_{a1}",
                              "anchor_reason": "registered run1 anchor" if block == "lead" else "by launch share, per set (labels)", "sets": {}}
            if block != "lead":
                out[block][lb]["labels_by_set"] = {}
            for set_, ogs in sets.items():
                if not all(og in X for og in ogs):
                    continue  # a set whose member row this probe lacks (withheld in the source) is absent for it
                bases_ = {og.split(" ")[0] for og in ogs}
                if block != "lead":  # the run2 block only where real run 1 shares the scoring model's training launch in some member set
                    if not bases_ & r1s_:
                        continue
                    out[block][lb]["labels_by_set"][set_] = SENS_T196 if ("S_old" in bases_ & sms_) else SENS
                out[block][lb]["sets"][set_] = one_set(lb, block, a0, a1, set_, ogs, X, mcol, lambda og, b: counts[og][b], D)
    if uni is not None:  # 09 RULE item 30: the same per-task points on the final record's UNION stream (one draw per canonical main across sets)
        import sys

        sys.path.insert(0, "/root")
        from run_fprsweep_507 import draws_main  # the rows' own draw function (the all-origins rows build W from it, same mains / strata / seed)

        um_, Wu = draws_main(uni["mains"], {m: (uni["env_of_main"][m],) for m in uni["mains"]}, seed=uni["seed"])
        ucol = {m: i for i, m in enumerate(um_)}
        c2m = {}  # cell -> canonical main from the final record's all-origins rows (cohort_cell_keys x cell_hits_draw0.mains); one main per cell
        for c, m in uni.get("cell_main_pairs") or []:
            if c2m.setdefault(c, m) != m:
                raise SystemExit(f"union stream: cell {c} has two canonical mains in the final record's all-origins rows")
        Wi = Wu.astype(np.int64)
        if not ((Wi == Wu).all() and (Wu[0] == 1).all()):
            raise SystemExit("union stream: multiplicities not integral, or draw 0 not all ones")
        Du = int(Wu.shape[0] - 1)
        usets = {f"{k} (union stream)": v for k, v in sets.items() if len(v) > 1} | {
            f"{k} (union stream)": v for k, v in ALLO.items() if k not in sets and all(o_ in origins for o_ in v)}
        ublk = {"lead": {}, "sensitivity_run2_anchor": {}}
        for lb in sorted(data):
            if not any(sel(lb, b_) for b_, _, _ in anchors[lb]):
                continue
            X, ucols = {}, {}
            for og in [o_ for o_ in origins if o_ in data[lb]]:
                cells = sorted(c for c, v in data[lb][og].items() if all(v.get(a) is not None and np.isfinite(v[a]) for a in ARMS))
                X[og] = {a: np.array([data[lb][og][c][a] for c in cells], float) for a in ARMS}
                cm_ = []  # each cell's canonical main: the final record's all-origins map (per-set S_old rows key their mains by cell)
                for c in cells:
                    own_, m_ = data[lb][og][c]["main"], c2m.get(c)
                    if m_ is None and own_ in ucol:  # a cell no all-origins row carries, whose own main is canonical
                        m_ = own_
                    if m_ is None or m_ not in ucol or (own_ in ucol and own_ != m_):
                        raise SystemExit(f"union stream: {lb} {og} cell {c}: no consistent canonical main")
                    cm_.append(m_)
                ucols[og] = np.array([ucol[m_] for m_ in cm_], int)
            r1s_, sms_ = (set(shares[lb][0]), set(shares[lb][1])) if shares else (set(), set())
            for block, a0, a1 in anchors[lb]:
                if not sel(lb, block):
                    continue
                for set_, ogs in usets.items():
                    if not all(og in X for og in ogs):
                        continue
                    bases_ = {og.split(" ")[0] for og in ogs}
                    if block != "lead" and not bases_ & r1s_:
                        continue
                    e_ = ublk[block].setdefault(lb, {"anchor": a0, "other_real": a1, "sets": {}} | ({"labels_by_set": {}} if block != "lead" else {}))
                    if block != "lead":
                        e_["labels_by_set"][set_] = SENS_T196 if ("S_old" in bases_ & sms_) else SENS
                    r_ = one_set(lb, block, a0, a1, set_, ogs, X, ucols, lambda og, b: Wi[b + 1], Du, tag="union|")
                    base_set = set_.removesuffix(" (union stream)")
                    po_ = out[block].get(lb, {}).get("sets", {}).get(base_set)
                    if po_ is not None:  # same cells, so the points equal the per-origin stream's exactly (only the resampling differs)
                        for m in METRICS:
                            for k in ("main", "yardstick"):
                                gate["compared"] += 1
                                a_, b_ = r_[m][k]["point"], po_[m][k]["point"]
                                if not (a_ == b_ or (a_ != a_ and b_ != b_)) or r_["n_cells"] != po_["n_cells"]:
                                    gate["mismatch"].append(f"union|{block}|{lb}|{set_}|{m}|{k}: point differs from the per-origin stream's")
                    e_["sets"][set_] = r_
        out["union_stream"] = ublk | {"draws": Du, "seed": uni["seed"], "mains": len(um_), "caption": UNION_CAPTION,
                                      "rule": "09 RULE item 30(1)/(3) @ 8c5a39038; ruling via 2e 10-16: union stream + fixed tau, labelled, is enough for the body"}
    return {"gate": gate | {"pass": not gate["mismatch"]}} | out


def finish(res, data, anchors, seed, D, post):
    """Gates (2) earlier addenda and (3) the source record's own Spearman panels, then the record (built inside the Modal function)."""
    gate = res["gate"] | {"refs": None, "record_panels": None}
    same = lambda a, b: a == b or (a != a and b != b)  # noqa: E731 (NaN equals NaN)

    def eq(mine, ref, n_mine, n_ref):
        return (n_mine == n_ref and same(mine["point"], ref["point"]) and mine["finite_draws"] == ref["finite_draws"]
                and (mine["ci95"] == ref["ci95"] if mine["ci95"] is None or ref["ci95"] is None
                     else len(mine["ci95"]) == len(ref["ci95"]) == 2 and all(same(x, y) for x, y in zip(mine["ci95"], ref["ci95"]))))

    if post.get("refs"):  # gate (2): exact reproduction of the earlier addenda, by key
        pr, rr = post["refs"]["paired"], post["refs"]["r2"]
        if [pr["meta"]["seed"], pr["meta"]["draws"], rr["meta"]["seed"], rr["meta"]["draws"]] != [seed, D, seed, D]:
            raise SystemExit("addenda seed/draws differ from the record's")
        g = {"compared": 0, "mismatch": []}
        if not (set(res["lead"]) == set(pr["paired"]) == set(rr["r2"]) == set(data)
                and all(set(res["lead"][lb]["sets"]) == set(pr["paired"][lb]) == set(rr["r2"][lb]) for lb in data)
                and set(res["sensitivity_run2_anchor"]) == {lb for lb in data if len(anchors[lb]) == 2}):
            g["mismatch"].append("probe/set membership differs between this record and the addenda")
        for block, lb, o in [(b_, lb, o) for b_ in ("lead", "sensitivity_run2_anchor") for lb, o in res[b_].items()]:
            for set_, s in o["sets"].items():
                P, R = pr["paired"][lb][set_], rr["r2"][lb][set_]
                pairs = [("spearman", "yardstick", P["spearman_run1_vs_run2"], P["n_cells"]), ("r2", "yardstick", R["r2_run1_vs_run2"], R["n_cells"])]
                if o["anchor"] == "run1":
                    pairs += [("spearman", "main", P["spearman_run1_vs_sim"], P["n_cells"]), ("r2", "main", R["r2_run1_vs_sim"], R["n_cells"]),
                              ("spearman", "delta_main_minus_yardstick", P["delta_spearman_run1sim_minus_run1run2"], P["n_cells"]),
                              ("r2", "delta_main_minus_yardstick", R["delta_r2_run1sim_minus_run1run2"], R["n_cells"])]
                for m, k, ref, n_ref in pairs:
                    mine = s[m][k]
                    g["compared"] += 1
                    ok = eq(mine, ref, s["n_cells"], n_ref)
                    if ok and k.startswith("delta"):
                        ok = mine["excludes_0"] == ref["excludes_0"]
                    if not ok:
                        g["mismatch"].append(f"{block}|{lb}|{set_}|{m}|{k}")
        gate["refs"] = g | {"paired_ref": post["refs"]["paired_name"], "r2_ref": post["refs"]["r2_name"]}
    if post.get("panels") is not None:  # gate (3): the lead's Spearman = the source record's own panels (same draws, seed and cells), exactly
        pn = post["panels"]
        g = {"compared": 0, "mismatch": []}
        if not pn or set(res["lead"]) != set(pn) or any(set(res["lead"][lb]["sets"]) != set(pn[lb].get(k_, {})) for lb in pn
                                                        for k_ in ("run1_vs_sim", "run1_vs_run2")):
            g["mismatch"].append("probe/set membership differs between this record and the source record's panels (or the panels are empty)")
        for lb, o in res["lead"].items():
            for set_, s in o["sets"].items():
                for k, pname in (("main", "run1_vs_sim"), ("yardstick", "run1_vs_run2")):
                    ref = pn.get(lb, {}).get(pname, {}).get(set_, {}).get("spearman_max_minus_tau")
                    g["compared"] += 1
                    if ref is None or not eq(s["spearman"][k], ref, s["n_cells"], ref["n_cells"]):
                        g["mismatch"].append(f"lead|{lb}|{set_}|spearman|{k}")
        gate["record_panels"] = g
    gate["complete"] = bool(gate["refs"] or gate["record_panels"])
    gate["pass"] = gate["pass"] and not (gate["refs"] or {}).get("mismatch") and not (gate["record_panels"] or {}).get("mismatch")
    rdb_ = post.get("readability")  # a pooled PERTASK record's not-readable sets (09 item 45(2)): LABELLED here too, never dropped
    if rdb_:
        nr_ = rdb_.get("not_readable_sets") or {}
        for blk_ in (res["lead"], res["sensitivity_run2_anchor"]) + (((res.get("union_stream") or {}).get("lead", {}),
                                                                    (res.get("union_stream") or {}).get("sensitivity_run2_anchor", {})) if res.get("union_stream") else ()):
            for lb, o_ in blk_.items():
                for s_, e_ in o_["sets"].items():
                    base_ = s_.removesuffix(" (union stream)")
                    if s_ in nr_.get(lb, ()) or base_ in nr_.get(lb, ()) or any(og_ in nr_.get(lb, ()) for og_ in (post.get("bootstrap") or {}).get("sets", {}).get(base_, ())):
                        e_["readability"] = NOT_READABLE
    return {"meta": post["meta"] | {"seed": seed, "draws": D}, "gate": gate, "lead": res["lead"], "sensitivity_run2_anchor": res["sensitivity_run2_anchor"],
            "union_stream": res.get("union_stream") or {"unavailable": post.get("union_unavailable")}} | (
           {"readability": {"rule": NOT_READABLE + ": a set with a not-readable member row is LABELLED (sets[...].readability), not dropped; group means "
                                                   "with a not-readable member are withheld as a whole (--groups)", "source": "the PERTASK record's readability block",
                            "not_readable_sets": rdb_.get("not_readable_sets")}} if rdb_ else {})


@app.local_entrypoint()
def main(src: str, sha: str, paired_ref: str = "", paired_sha: str = "", r2_ref: str = "", r2_sha: str = "", final: str = "", fanout: bool = False,
         repro: bool = False, same_as: str = "", groups: bool = False, label: str = "", a_sidecar: str = "", sd_git: str = "",
         sd_script_sha256: str = ""):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("uncommitted tracked changes")
    side_ = None
    if a_sidecar or sd_git or sd_script_sha256:  # route C (09 @ READ_LOG 8a4e2bf84 L2213): the label is composed in the container, never typed
        m_ = re.fullmatch(r"git:([0-9a-f]{40}):([^ @]+\.json)@([0-9a-f]{64})", a_sidecar)
        if not (m_ and re.fullmatch(r"[0-9a-f]{40}", sd_git) and re.fullmatch(r"[0-9a-f]{64}", sd_script_sha256)) or label or not final or repro:
            raise SystemExit("route C: --a-sidecar git:<40-hex>:<path>.json@<sha256>, --sd-git <40-hex> and --sd-script-sha256 <64-hex> go "
                             "together, with --final and without --label or --repro (STOP)")
        side_ = (git("show", f"{m_[1]}:{m_[2]}"), m_[3])  # bytes only: checked against the pin and parsed in the container (routec_label)
        if hashlib.sha256(side_[0]).hexdigest() != side_[1]:
            raise SystemExit("route C: sha256 of A's label sidecar (STOP)")

    def blob(ref, want):  # bytes only, sha256-checked here and again in the container; this client never parses a record (0d 10-16)
        import gzip

        b = git("show", ref)
        if ref.endswith(".gz"):  # a record over GitHub's 100 MB limit is committed gzipped; its sha256 is the decompressed JSON's
            b = gzip.decompress(b)
        if hashlib.sha256(b).hexdigest() != want:
            raise SystemExit(f"sha256 of {ref}")
        return b

    if bool(paired_ref) != bool(r2_ref):
        raise SystemExit("pass both --paired-ref and --r2-ref, or neither")
    if (repro or same_as) and not fanout or repro and same_as or groups and not (same_as and final):
        raise SystemExit("--repro and --same-as each need --fanout, and not both; --groups needs --same-as and --final")
    checked = {"src": (blob(src, sha), sha)}
    if paired_ref:
        checked |= {"paired": (blob(paired_ref, paired_sha), paired_sha), "r2": (blob(r2_ref, r2_sha), r2_sha)}
    if same_as:
        checked["same_as"] = (blob(*same_as.rsplit("@", 1)), same_as.rsplit("@", 1)[1])
    fr_ = None
    if final:  # bytes only (a3fd1612 R1): the decompressed sha is checked here; A's keys are read in the driver (final_keys)
        import gzip

        fref_, fsha_ = final.rsplit("@", 1)
        fr_ = git("show", fref_)
        if hashlib.sha256(gzip.decompress(fr_) if fref_.endswith(".gz") else fr_).hexdigest() != fsha_:
            raise SystemExit(f"sha256 of {fref_}")
    gr_ = None
    if groups:  # six = Stage A's headline_six_full (full_override.HEADLINE_SIX, code); the driver adds the four from A's meta (final_keys)
        import sys

        sys.path.insert(0, str(HERE))
        import full_override as FO

        gr_ = {"six": list(FO.HEADLINE_SIX)}
    tb_ = git("show", f"HEAD:./{SHARE}")
    if (HERE / SHARE).read_bytes() != tb_:
        raise SystemExit(f"{SHARE} differs from HEAD")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    meta = {"git": git("rev-parse", "HEAD").decode().strip(), "stamp": stamp, "app_id": app.app_id, "source_record": f"{src} ({sha})",
            "label": "POST HOC; PRELIMINARY; run1-anchored lead with the on-policy run2 anchor beside (09 @ 635b14564); addendum to the PERTASK record (the record itself is unchanged)",
            "caption": CAPTION, "on_policy_rule": ON_POLICY.pattern,
            "definition": "lead: every probe anchored on run1; sensitivity_run2_anchor (on-policy probes only): anchored on run2; main = r(anchor, sim), "
                          "yardstick = r(anchor, other real run), r = Spearman or squared Pearson of max_minus_tau across tasks (per action 1%, "
                          "separate lines; the source record's readout); delta_main_minus_yardstick = main - yardstick, the same sign for every row "
                          "(sensitivity rows: [run2 vs sim] - [run2 vs run1]); all from one joint main-level resample per draw "
                          "(the record's draws); cells with all three arms finite"}
    if label:  # 09 / 2e 10-17: the run's label (e.g. 171-pool / D1 read from pooled A) goes INTO the record's meta.label, not only a NOTE
        meta["label"] += f"; {label}"
    meta["launch_share"] = {"path": SHARE, "sha256": hashlib.sha256(tb_).hexdigest(), "rule": "09 @ 4c095dec9 item 16: the run2 block by launch share, per set"}
    if fanout:
        meta["compute"] = ("fan-out: one container per (label, block), each reading the whole source (same draws), merged in the one-container "
                           "loop's order" + ("; --repro: fan-out vs one container on a subset" if repro else "")
                           + (f"; --same-as {same_as}: every non-meta key byte-identical to that record" if same_as else ""))
    args = {"name": f"{Path(src.split(':', 1)[1]).stem}_anchor_{stamp}_{app.app_id}{'_fanout_repro' if repro else ''}", "final": final or None,
            "paired_name": f"{paired_ref} ({paired_sha})" if paired_ref else None, "r2_name": f"{r2_ref} ({r2_sha})" if r2_ref else None,
            "fanout": fanout, "repro": repro, "same_as": same_as or None, "groups": gr_, "label": label or None,
            "groups_keep": None,  # the driver sets it from the memberships it builds (final_keys)
            "src": src, "a_sidecar": a_sidecar or None, "sd_git": sd_git or None, "sd_script_sha256": sd_script_sha256 or None}
    call_ = anchored.spawn({"checked": checked, "share": tb_} | ({"final_raw": fr_} if fr_ else {}) | ({"a_sidecar": side_} if side_ else {}), args, meta)
    print(json.dumps({"anchored_call": call_.object_id, "volume_dir": "xfer/pertask/"}), flush=True)
    r_ = call_.get()
    if hashlib.sha256(r_["blob"]).hexdigest() != r_["sha256"]:
        raise SystemExit("record sha256 (driver vs local)")
    (HERE / "results" / r_["name"]).open("xb").write(r_["blob"])
    if r_.get("sidecar"):  # route C: the record's label sidecar, as the driver wrote it beside the record (bytes as returned; never parsed here)
        sn_ = f"{Path(r_['name']).stem}_LABEL.json"
        (HERE / "results" / sn_).open("xb").write(r_["sidecar"])
        print(json.dumps({"label_sidecar": sn_, "sha256": hashlib.sha256(r_["sidecar"]).hexdigest(), "record_sha256": r_["sha256"]}), flush=True)
    elif side_ and r_["gate"]["pass"]:
        raise SystemExit("route C: the driver returned no label sidecar for a passing record (STOP)")
    g = r_["gate"]
    if repro:  # keys, shas and booleans only
        print(json.dumps({"record": r_["name"], "volume_path": r_["volume_path"], "sha256": r_["sha256"], "gate_pass": g["pass"]}), flush=True)
        if not g["pass"]:
            raise SystemExit(json.dumps({"REPRO_FAIL": g["mismatch"]}))
        return
    print(json.dumps({"record": r_["name"], "volume_path": r_["volume_path"], "sha256": r_["sha256"], "gate_pass": g["pass"], "gate_complete": g["complete"],
                      "compared": [g["compared"], (g["refs"] or {}).get("compared"), (g["record_panels"] or {}).get("compared")],
                      "ties_skipped": len(g["spearman_direct_skipped_ties"]), "same_as_equal": (g.get("same_as") or {}).get("equal"),
                      "groups_gate": {k: (g.get("groups") or {}).get(k) for k in ("pass", "compared")} if groups else None}), flush=True)
    if not g["pass"]:
        raise SystemExit(json.dumps({"GATE_FAIL": ((["same_as: not byte-identical to the named record"] if (g.get("same_as") or {}).get("equal") is False else [])
                                                   + (g.get("groups") or {}).get("mismatch", []) + g["mismatch"] + ((g["refs"] or {}).get("mismatch") or [])
                                                   + ((g["record_panels"] or {}).get("mismatch") or []))[:20]}))
