"""Shared I/O for the post figures ported from the writing session's figs_simple templates (2e 10-16, Aswin via 0d).

Each figure writes three files: the PNG (no baked-in title, footer or caption), <name>_source.json (every input record by path@sha and every
plotted value with its key path) and <name>_text.md (title and caption, drafted from the record; the writing session edits it). Inputs are
named on the command line as FILE SHA256 [SOURCE], where SOURCE is the record's path@commit for the citation; figio.load checks the sha.
"""

import argparse
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path

from figio import BANNER, BUILD, RES, TRAIN, has_ci, load, nice, no_stale, wmark, withheld_foot  # noqa: F401


def sidecar_label(sidecar, record_sha):
    """A route C label sidecar = (FILE, SHA256, SOURCE), shipped as a plan record and read here, in the render container: the bytes hash to
    SHA256 before the parse; exactly {record, record_sha256, label}; `record` non-empty; record_sha256 = record_sha (the record drawn).
    -> the label (a string). STOP on any miss."""
    if not (isinstance(record_sha, str) and re.fullmatch(r"[0-9a-f]{64}", record_sha)):
        sys.exit("STOP: the drawn record's sha256 is not 64 hex")
    raw = (RES / Path(sidecar[0]).name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sidecar[1]:
        sys.exit(f"STOP: sha mismatch for the label sidecar {sidecar[0]}")
    sc = json.loads(raw)
    if not isinstance(sc, dict) or set(sc) != {"record", "record_sha256", "label"}:
        sys.exit("STOP: the label sidecar is not exactly {record, record_sha256, label}")
    if not (isinstance(sc["record"], str) and sc["record"]) or sc["record_sha256"] != record_sha:
        sys.exit("STOP: the label sidecar names another record (record_sha256 differs from the record drawn)")
    if not isinstance(sc["label"], str):
        sys.exit("STOP: the sidecar label is not a string")
    return sc["label"]


def routec_label(sidecar, record_sha, what, label_sha256):
    """09's route C label rule for a REGISTERED label (A, B, R37; READ_LOG L2172, amended L2213; each its own sidecar, 09 via 2e 10-18):
    sidecar_label's checks, then the sha256 of the label's UTF-8 bytes = label_sha256 (the plan pins it from 09's READ_LOG line; the text is
    never written in code), and PRELIMINARY. -> "<label> | <what>". STOP on any miss."""
    if not (isinstance(label_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", label_sha256)):
        sys.exit("STOP: the registered label's sha256 is not 64 hex")
    lb = sidecar_label(sidecar, record_sha)
    if not (hashlib.sha256(lb.encode("utf-8")).hexdigest() == label_sha256 and "PRELIMINARY" in lb):
        sys.exit("STOP: the sidecar label is not 09's registered string (sha256) or lacks PRELIMINARY")
    return f"{lb} | {what}"


INPUT_SIDECAR = {"nargs": 5, "action": "append", "metavar": ("ROLE", "FILE", "SHA256", "SOURCE", "STEP"), "default": None,
                 "help": "route C: the label sidecar of a DERIVED drawn input (role: its Out key, e.g. group, pw, four; STEP: the step its label "
                         "names, e.g. run_group_catch), one per drawn input besides --record"}


def routec_inputs(inputs, sidecars, a_label, a_label_sha256, a_sha):
    """09 (via 2e, 10-18): a figure checks the sidecar of EVERY record it draws, and a DERIVED record's label is COMPOSED, not registered
    (09's byte rule, verbatim in effect): (iii) its sidecar's record_sha256 = that input's own pinned sha256 (sidecar_label); (ii) its label's
    UTF-8 bytes START WITH exactly A's registered label bytes (that prefix hashes to a_label_sha256) and continue with exactly
    " | <STEP> on route C pooled A @ <a_sha>", where a_sha is the A record this figure draws; (iv) PRELIMINARY is present. inputs = {role:
    (FILE, SHA256, SOURCE)} of the drawn inputs besides --record; sidecars = [[ROLE, FILE, SHA256, SOURCE, STEP], ...]: exactly one per role.
    a_label = A's label, already checked by routec_label. -> {role: label}. STOP on any miss, or a missing, extra or repeated role."""
    ab = a_label.encode("utf-8") if isinstance(a_label, str) else b""
    if not ab or hashlib.sha256(ab).hexdigest() != a_label_sha256 or not re.fullmatch(r"[0-9a-f]{64}", str(a_sha)):
        sys.exit("STOP: A's label or the A record's sha256 is not the checked one")
    got = {}
    for role, f, s, src, step in sidecars or []:
        if role in got or role not in inputs:
            sys.exit(f"STOP: --input-sidecar role {role!r} is repeated or names no drawn input ({sorted(inputs)})")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", step):
            sys.exit(f"STOP: --input-sidecar {role}: the step name is not a plain identifier")
        lbb = sidecar_label((f, s, src), inputs[role][1]).encode("utf-8")
        if hashlib.sha256(lbb[:len(ab)]).hexdigest() != a_label_sha256:
            sys.exit(f"STOP: {role}: the label does not start with A's registered label bytes")
        if lbb[len(ab):] != f" | {step} on route C pooled A @ {a_sha}".encode("utf-8"):
            sys.exit(f"STOP: {role}: the label's suffix is not ' | {step} on route C pooled A @ <the A record drawn>'")
        if b"PRELIMINARY" not in lbb:
            sys.exit(f"STOP: {role}: the label lacks PRELIMINARY")
        got[role] = lbb.decode("utf-8")
    if set(got) != set(inputs):
        sys.exit(f"STOP: drawn inputs without their label sidecar: {sorted(set(inputs) - set(got))}")
    return got


def check_record_meta(meta, git, script_sha256, inputs_full_sha256):
    """Route C (09 S2 item 10, re-asserted by key in the container before any gate or drawing; run_tables_v81's check_record_meta): the
    record's meta.git is the plan's launch commit, meta.script_sha256 the plan's runner sha256 (blob 71633875 -> dcd265d7...), and
    meta.full_length.inputs_full the FROZEN file (sha256 = the plan's, status FROZEN..., test false). Any miss STOPs, naming keys only."""
    fin = ((meta or {}).get("full_length") or {}).get("inputs_full") or {}
    bad = [k for k, ok in (("git", (meta or {}).get("git") == git), ("script_sha256", (meta or {}).get("script_sha256") == script_sha256),
                           ("full_length.inputs_full.sha256", fin.get("sha256") == inputs_full_sha256),
                           ("full_length.inputs_full.status", str(fin.get("status", "")).startswith("FROZEN")),
                           ("full_length.inputs_full.test", fin.get("test") is False)) if not ok]
    if bad:
        sys.exit(f"STOP: the record's meta differs from the plan's --record-meta at {bad}")
    return {"git": git, "script_sha256": script_sha256, "inputs_full_sha256": inputs_full_sha256, "checked": "by key, in the container"}

INK, MUTED, SIM, REAL = "#222222", "#6b6b68", "#eb6834", "#2a78d6"
GRAY = "#8d8d88"  # the ONE real-vs-real (rerun) colour everywhere (make_sweep_final's; make_rms used #9a9a96, make_pertask blue)
BAR, SUM = "#8a8fc4", "#3b3f7a"
# the comparator is ONE plan field (render_all passes it as FIG_COMPARATOR): run_mean = 09's lead, sim − mean(run 1, run 2), the real bar = the
# run mean, the rerun run 2 − run 1 in gray (pairs, sweep, Fig 3); run1 = anchored on real run 1. The per-task figures always anchor on run 1.
COMPARATOR = os.environ.get("FIG_COMPARATOR", "").strip()


def comparator():
    if COMPARATOR not in ("run_mean", "run1"):
        sys.exit(f"STOP: the render plan's comparator must be run_mean or run1 (got {COMPARATOR!r})")
    return COMPARATOR


# the run-mean estimator is ONE plan field too (FIG_ESTIMATOR): A = the registered per-arm run mean, B = the all-runs mean (09 @ 3aaf425ef); the
# excess is always on (A). Required whenever the comparator is run_mean (Codex ports r5: no silent A/B mix across figures).
ESTIMATOR = os.environ.get("FIG_ESTIMATOR", "").strip()


def estimator():
    """A or B. The plan gives it directly, or (2e 10-16) as {"record": basename, "sha256": ..., "key": ["meta", "estimator", "lead"]}: the
    full-length build's selection record (chosen there by width), shipped with the plan's records and read here by key."""
    if comparator() == "run1":
        return None
    est = ESTIMATOR
    if est.startswith("{"):
        spec = json.loads(est)
        x = load(spec["record"], spec["sha256"])
        for k in spec["key"]:
            if not isinstance(x, dict) or k not in x:
                sys.exit(f"STOP: no {spec['key']} in {spec['record']}")
            x = x[k]
        est = x
    if est not in ("A", "B"):
        sys.exit(f"STOP: comparator run_mean needs the estimator, A or B (got {est!r})")
    return est


RC = {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED}
F = "0.01"


def args(desc, records, opts=None):
    """records: [(flag, help)] each taking FILE SHA256 SOURCE; plus --name (output stem), --stamp, --git, --rows/--labels when given in opts."""
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--git", default="")
    ap.add_argument("--fig-number", default="", help="the post's figure number (0d FINAL: 1..8 by first appearance), heading the caption title as Figure N:; the filename stays")
    for flag, help_ in records:
        ap.add_argument(flag, nargs=3, metavar=("FILE", "SHA256", "SOURCE"), required=True, help=help_ + " (SOURCE = path@commit, cited)")
    for name, kw in (opts or {}).items():
        ap.add_argument(name, **kw)
    return ap.parse_args()


# 09 @ 2d61af835: c_asis_uncut is a positive control on the shared-threshold rows only; never in S or the headline figures.
NOT_HEADLINE = ("c_asis_uncut",) + tuple(BUILD.get("not_headline") or ())


def check_rows(name, rows):
    bad = [r for r in rows if r.rsplit(" ", 1)[0] in NOT_HEADLINE or r in NOT_HEADLINE]
    if bad:
        sys.exit(f"STOP: {name}: {bad} may not appear in a headline figure (09 @ 2d61af835)")


class Out:
    """Collects the source JSON (inputs, values by key) and writes PNG + source + text with exclusive create."""

    def __init__(self, name, a, inputs):
        self.name, self.a = name, a
        check_rows(name, getattr(a, "rows", None) or [])
        self.src = {"figure": name, "git": a.git, "inputs": {k: {"file": v[0], "sha256": v[1], "source": v[2]} for k, v in inputs.items()},
                    "values": [], "build_profile": BUILD or None}
        self.recs = {k: load(v[0], v[1]) for k, v in inputs.items()}

    def val(self, rec_key, path, node=None):
        """Read rec[path...] by key and log it; path is a list of keys. Returns the node."""
        x = self.recs[rec_key]
        for k in path:
            if not isinstance(x, dict) or k not in x:
                sys.exit(f"STOP: {self.name}: no key {path} in {self.src['inputs'][rec_key]['file']}")
            x = x[k]
        if isinstance(x, dict) and "point" in x:
            self.src["values"].append({"input": rec_key, "key": path, "point": x["point"], "ci95": x.get("ci95")})
        elif isinstance(x, (int, float, list)) or x is None:  # scalars and intervals (Codex ports r1)
            self.src["values"].append({"input": rec_key, "key": path, "value": x})
        return x

    def get(self, rec_key, path):
        """Like val, but a missing key returns None (logged as missing) instead of stopping: for status fields that may be absent."""
        x = self.recs[rec_key]
        for k in path:
            if not isinstance(x, dict) or k not in x:
                self.src["values"].append({"input": rec_key, "key": path, "missing": True})
                return None
            x = x[k]
        self.src["values"].append({"input": rec_key, "key": path, "value": x if isinstance(x, (str, int, float, bool)) or x is None else "<non-scalar>"})
        return x

    def text(self, title, caption, probes=(), n=None):
        """Title and caption for the text file; the banner heads the title (its {n} = this figure's task count n, from a record), the build's
        length label and training lengths head the caption."""
        if BUILD:
            miss = [p for p in probes if p not in TRAIN]
            if miss:
                sys.exit(f"STOP: {self.name}: drawn probes without a training length in the build profile: {miss}")
            # the figure's readouts: its --readouts, else its rows' C/R suffixes; a probe's per-readout text (train_by_readout) when the profile
            # has one for every readout drawn (0d via 2e, 10-16: the R figures quote the R row), else its single train text
            ros = list(getattr(self.a, "readouts", None) or sorted({r.rsplit(" ", 1)[-1] for r in (getattr(self.a, "rows", None) or [])
                                                                     if r.endswith((" C", " R"))}))

            def ttxt(p):
                tb = TRAIN[p].get("train_by_readout") or {}
                tx = [tb.get(r) for r in ros]
                if not ros or not all(isinstance(t, str) and t.strip() for t in tx):
                    return TRAIN[p]["train"]
                u = list(dict.fromkeys(tx))
                return u[0] if len(u) == 1 else "; ".join(f"{r} readout: {t}" for r, t in zip(ros, tx))
            self.src["train_readouts"] = ros
            caption = (f"{BUILD['length_label'].strip()}. Training length (each probe's FIT spec): "
                       + "; ".join(f"{nice(p)}: {ttxt(p)}" + (f" ({TRAIN[p]['note']})" if TRAIN[p].get("note") else "")
                                   for p in dict.fromkeys(probes)) + ". " + caption)
        b = BANNER
        if "{n}" in b:
            if not isinstance(n, int) or n <= 0:
                sys.exit(f"STOP: {self.name}: the title banner needs this figure's task count (got {n!r})")
            b = b.replace("{n}", f"{n:,}")
        self.src["title_banner"] = b or None
        self.title = (b + "\n" if b else "") + (f"Figure {self.a.fig_number}: " if getattr(self.a, "fig_number", "") else "") + title
        self.src["fig_number"] = getattr(self.a, "fig_number", "") or None
        self.caption = caption + withheld_foot()  # every ‡ drawn: the record's reason (point drawn without its interval)

    def save(self, fig, dpi=200):
        from matplotlib.text import Text
        if fig._suptitle is not None or any(t.get_gid() != "panel-label" for t in fig.texts):  # a figure-level panel label is allowed (Fig 6 v2)
            sys.exit(f"STOP: {self.name}: titles and captions go in the text file, not the PNG")
        for t in fig.findobj(Text):
            no_stale(t.get_text(), self.name)
            if "{model}" in t.get_text():  # an unfilled model placeholder never reaches the PNG
                sys.exit(f"STOP: {self.name}: a label still holds {{model}}")
        if "{n}" in self.title:
            sys.exit(f"STOP: {self.name}: banner {{n}} not filled")
        for s in (self.title, self.caption, json.dumps(self.src)):
            no_stale(s, self.name)
        stem = f"{self.name}{self.a.stamp}"
        buf = io.BytesIO()
        fig.savefig(buf, dpi=dpi, format="png")
        with open(RES / f"{stem}.png", "xb") as f:  # exclusive create
            f.write(buf.getvalue())
        with open(RES / f"{stem}_source.json", "x") as f:
            f.write(json.dumps({**self.src, "text_file": f"{stem}_text.md"}, indent=1))  # title and caption live only in the text file
        with open(RES / f"{stem}_text.md", "x") as f:
            f.write(f"# {self.title}\n\n{self.caption}\n\nSource: {stem}_source.json (inputs by path@sha; every plotted value with its key).\n")
        print(RES / f"{stem}.png")


# The off-policy probe's source model, as displayed (the writing session's choice via 2e, 10-16; Aswin can override). Source: #36 runs.json @
# dea0a4e79 (sha256 76b11486…), real-honest run key base200_000_000_2026-09-17T06-26-24-00-00_la-auto-workflow-real-honest-20_kKMJeKKJiUCApdD647d9QX
# __abstract_audit_logger_interface:252ea9__1, log sha256 fa0fffbe8d696ddae200418276a27bd9aaed01316eb9e8f27b3ccaed683e781e, header
# plan.steps.2.params.model_name = 'openai/gpt-5.6-sol' (eval.model there is 'none/none').
OFF_POLICY_MODEL = "GPT-5.6-Sol"


def label(p, labels):
    """Row label: the plan's post label for this probe key (its {model} = OFF_POLICY_MODEL), else the build profile's display name. Under a
    build profile every drawn probe must be mapped explicitly by label key (the writing session's plain names via 2e, 10-16), else STOP."""
    if BUILD and not labels.get(p):
        sys.exit(f"STOP: no plain name in the plan's labels for probe key {p!r} (every post figure maps its probes explicitly)")
    return (labels.get(p) or nice(p)).replace("{model}", OFF_POLICY_MODEL)  # the fallback name too (Codex r39)
