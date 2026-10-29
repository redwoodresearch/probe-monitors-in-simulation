"""Shared I/O for the final-build figure scripts (render only; no recomputation).

Records are named on the command line with their sha256 and loaded from results/ by basename; every output is
written with exclusive create, so a render never overwrites an earlier one. Caption constants for the final build live here.
"""

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"

DISCLOSURE = "Registered headline Z moved to the appendix by the author's choice; all-tasks separate-line Δ_full leads."
NM360_LABEL = "fold −1-only pool: five-model mean with all-unit per-model moments (25h §1 extension; POST HOC)"
NM360_PROBES = ("Sol (a)", "c_asis", "own_frac100_L48", "own_frac25_L48", "own_frac10_L48")
# 25h §5 caption items (RULE.md @ 0f9927f3e), shortened; used only when the record carries no meta.captions
S5_DEFAULT = {
    "a": "intervals resample tasks (mains with all their runs and units together), never units",
    "b": "run-length confound: one per-action line can map to different per-run alarm shares in sim and real (simulated runs can be shorter, "
         "fewer cut at the 32,768-token limit); per-pool per-run alarm shares and units per run are in the record",
    "c": "τ_sim and τ_real are each re-estimated inside every draw",
}


def args_with(desc, *specs, opts=None):
    """Common CLI: --stamp (required; output suffix), --git (script commit; default: this checkout's HEAD), record specs, extra options."""
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--stamp", required=True, help="output-name suffix, e.g. _final or _dryrun")
    ap.add_argument("--git", default="", help="script commit for provenance (the render wrapper passes it)")
    for name, help_ in specs:
        ap.add_argument(name, nargs=2, metavar=("FILE", "SHA256"), required=True, help=help_)
    ap.add_argument("--commit", default="", help="commit holding the records (caption)")
    for name, kw in (opts or {}).items():
        ap.add_argument(name, **kw)
    a = ap.parse_args()
    if not a.git:
        a.git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=RES.parent).stdout.strip()
    return a


def load(name, sha):
    raw = (RES / Path(name).name).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != sha:
        sys.exit(f"sha mismatch for {name}: {got} != {sha}")
    return json.loads(raw)


# 0d recheck 10-16: the union-Sol records carry a stale row-level gate label ("gate pending … not relayed until 09 verifies"), contradicting
# their own meta.fold_extension_passed ("gate passed (09 @ c6023b805 …)"). No output may echo it: every figure text and numbers file is scanned.
STALE_GATE = re.compile(r"(?i)gate pending|not relayed until")


def no_stale(text, where):
    m = STALE_GATE.search(text or "")
    if m:
        sys.exit(f"STOP: stale gate label {m.group(0)!r} in {where}; captions and legends use the passed-gate label")


# 0d 10-16: the plan's title_banner heads every figure title. Its task count is PER FIGURE: the plan writes "{n}" (e.g. "DRAFT: {n} tasks,
# attempt-filtered; …") and each script fills its own count from its record; a literal "<number> tasks" in the plan stops the render.
BANNER = os.environ.get("FIG_TITLE_BANNER", "").strip()
if re.search(r"(?i)\d[\d,]*\s*tasks?\b", BANNER):
    sys.exit(f"STOP: title_banner {BANNER!r} hard-codes a task count; write {{n}} so each figure fills its own")
_FILLED = {"banner": ""}  # the banner as last drawn (save_fig checks it, save_json records it)


def suptitle(fig, text, n=None, **kw):
    """Figure title with the plan's banner (if any) as its own first line; n = this figure's task count, from its record."""
    b = BANNER
    if "{n}" in b:
        if not isinstance(n, int) or n <= 0:
            sys.exit(f"STOP: the title banner needs this figure's task count (got {n!r})")
        b = b.replace("{n}", f"{n:,}")
    _FILLED["banner"] = b
    return fig.suptitle(f"{b}\n{text}" if b else text, **kw)


def save_fig(fig, name, dpi=150):
    from matplotlib.text import Text
    texts = [t.get_text() for t in fig.findobj(Text)]  # titles, axis labels, ticks, legends, panel notes, caption
    st = fig._suptitle.get_text() if fig._suptitle is not None else None
    for t in texts:
        no_stale(t, name)
    if BANNER and not (_FILLED["banner"] and any(_FILLED["banner"] in t for t in texts)):
        sys.exit(f"STOP: {name} lacks the plan's title banner {BANNER!r}")
    if BUILD and not any(BUILD["length_label"].strip() in t.replace("\n", " ") for t in texts if t != st):  # a caption, not the title
        sys.exit(f"STOP: {name} caption lacks the build's length label {BUILD['length_label']!r}")
    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi)
    with open(RES / name, "xb") as f:  # exclusive create: a collision fails loudly
        f.write(buf.getvalue())
    print(RES / name)


def save_json(obj, name):
    obj = {**obj, "title_banner": _FILLED["banner"] or None, "title_banner_template": BANNER or None, **({"build_profile": BUILD} if BUILD else {})}
    text = json.dumps(obj, indent=1)
    no_stale(text, name)
    with open(RES / name, "x") as f:
        f.write(text)
    print(RES / name)


def s5_caption(rec):
    """25h §5 (a)–(c): the record's own meta.captions text when present, else the RULE text above. Returns (text, source)."""
    lst = (rec.get("meta") or {}).get("captions_25h_s5")  # 507-prep's Stage A shape: a list of 3 strings "(a) …", "(b) …", "(c) …"
    if lst is not None:
        if not (isinstance(lst, list) and len(lst) == 3 and all(isinstance(t, str) and t.strip().startswith(f"({k})") for t, k in zip(lst, "abc"))):
            sys.exit(f"meta.captions_25h_s5 has an unexpected shape: {lst!r}")
        return "25h §5: " + " ".join(t.strip() for t in lst), "record meta.captions_25h_s5"
    cap = (rec.get("meta") or {}).get("captions")
    if isinstance(cap, str) and cap.strip():
        return cap.strip(), "record meta.captions"
    if isinstance(cap, dict):  # keys such as "a", "(a)", "25h_5a", "25h §5 (a)": the item letter is the last letter before an optional ")"
        s5 = {}
        for k, v in cap.items():
            kk = k.lower().strip().rstrip(")")
            if isinstance(v, str) and kk and kk[-1] in "abc" and ("5" in kk or len(kk) <= 2):
                s5[kk[-1]] = v.strip()
        if set(s5) == {"a", "b", "c"}:
            return "25h §5: " + "; ".join(f"({k}) {s5[k]}" for k in "abc") + ".", "record meta.captions"
    return "25h §5: " + "; ".join(f"({k}) {v}" for k, v in S5_DEFAULT.items()) + ".", "RULE @ 0f9927f3e (record has no meta.captions)"


def wrap(text, width=210):
    import textwrap

    return "\n".join(textwrap.fill(p, width) for p in text.split("\n"))


# Probe display order (fixed); the probes drawn are the ones present in the record. own_all (Append 40) sits after own_frac when present.
DISPLAY_ORDER = ["Sol (a)", "FIBS", "harmfulness", "refusal", "high-stakes", "own_frac100_L48", "own_frac25_L48", "own_frac10_L48",
                 "own_all_L48", "own_all25_L48", "own_all10_L48", "c_asis"]
NOT_SHOWN = {"own_a": "S_old only (no mapped units on the other sets); in the record, not drawn"}
OUTSIDE_FAMILY = ("own_all_L48", "own_all25_L48", "own_all10_L48")
OWN_ALL_NOTE = ("† own_all (Append 40; 4f's famexp): new recipe; POST HOC. Per-probe rows only, outside the family statistics and never in "
                "any group mean or excess (not in X_err / U, LORO, D_e2e or PANEL).")


def probes_in(labels, ro, ok=lambda lab: True):
    """Probes of readout ro present in the record (labels like 'FIBS C'), in DISPLAY_ORDER; unknown probes go before c_asis (without a build
    profile; with one, probes outside it are reported in the skipped list, never drawn).
    A label that fails ok() (e.g. 'not in this build') is simply absent. Returns (shown, skipped_with_reason)."""
    names = {lab[: -len(ro) - 1] for lab in labels if lab.endswith(f" {ro}")}
    skipped = {n: "DT design: drawn in the DT figure" for n in sorted(names) if n.startswith("DT_")}  # Codex r1: reported, not silent
    names = {n for n in names if not n.startswith("DT_")}
    shown_extra = draw_extra()
    skipped.update({n: "absent: fails this figure's presence check (e.g. 'not in this build')" for n in sorted(names) if not ok(f"{n} {ro}")})
    names = {n for n in names if ok(f"{n} {ro}")}
    shown = [p for p in DISPLAY_ORDER if p in names]
    if BUILD:  # a build profile names every drawn probe; the record's other labels are reported, never drawn (no training length for them)
        skipped.update({n: NOT_SHOWN.get(n, "not in the build profile") for n in sorted(names) if n not in DISPLAY_ORDER})
        return shown, skipped
    extra = sorted(n for n in names if n not in DISPLAY_ORDER and (n not in NOT_SHOWN or n in shown_extra))
    if extra:
        i = shown.index("c_asis") if "c_asis" in shown else len(shown)
        shown[i:i] = extra
    skipped.update({n: NOT_SHOWN[n] for n in sorted(names) if n in NOT_SHOWN and n not in shown_extra})
    return shown, skipped


# 2e 10-16 ~02:00 (Aswin): the full-length (no 32k cut) build. The render plan's "build" profile (env FIG_BUILD) names the probes by record
# key, in display order, each with its training length from its FIT spec; it also sets the length label every caption carries and the
# probe groups that captions name. Without a profile the scripts keep the 32k build's lists above.
BUILD = json.loads(os.environ.get("FIG_BUILD") or "{}")
DISPLAY, TRAIN = {}, {}
if BUILD:
    if not (isinstance(BUILD.get("length_label"), str) and BUILD["length_label"].strip() and isinstance(BUILD.get("probes"), list) and BUILD["probes"]):
        sys.exit("STOP: the build profile needs length_label and a non-empty probes list")
    for e in BUILD["probes"]:
        if not all(isinstance(e.get(k), str) and e[k].strip() for k in ("key", "train", "train_source")):
            sys.exit(f"STOP: build probe {e!r} needs key, train (training length from its FIT spec) and train_source (fit spec path@sha)")
        DISPLAY[e["key"]] = e.get("display") or e["key"].replace("_L48", "")
        TRAIN[e["key"]] = e
    DISPLAY_ORDER = [e["key"] for e in BUILD["probes"]]
    OUTSIDE_FAMILY = tuple(BUILD.get("outside_family", OUTSIDE_FAMILY))
    NM360_PROBES = tuple(BUILD.get("nm360_probes", NM360_PROBES))
    NOT_SHOWN = {**NOT_SHOWN, **{k: v for k, v in (BUILD.get("not_shown") or {}).items()}}


# 09 RULE item 30 @ 8c5a39038: headline intervals come from the UNION stream (pooled_all_origins: one draw per main across origins); the
# build profile names the pooled combination every pooled figure reads (points and intervals), and its label goes into each caption.
POOL = BUILD.get("pool", "pooled_all")
POOL_NO4 = BUILD.get("pool_no4", "pooled_all_without_S4")
LEAD_COMBO = "pooled_all_origins"  # 09 (via 2e, 10-16): the lead combination of every headline figure, checked BY NAME (records' role tags are not trusted)
if BUILD and (POOL, POOL_NO4) != (LEAD_COMBO, LEAD_COMBO + "_without_S4"):
    sys.exit(f"STOP: the build's pooled combination is {POOL!r}/{POOL_NO4!r}; 09: the lead is {LEAD_COMBO!r} (pooled_all = three origins, not the headline)")
STREAM_TXT = ("Intervals and points: the union stream (pooled_all_origins: one draw per main across origins; 09 RULE item 30); intervals are pointwise, "
              "not simultaneous across probes. " if POOL == "pooled_all_origins"
              else f"Intervals and points: the {POOL} combination (per-origin streams). " if BUILD else "")
PER_ORIGIN_TXT = "Intervals: per-origin streams (each task set resampled on its own; 09 RULE item 30: appendix). " if BUILD else ""


def nice(p):
    base = DISPLAY.get(p) or p.replace("_L48", "")
    return base + (" (stress test)" if p == "c_asis" else "") + (" †" if p in OUTSIDE_FAMILY else "")


def build_txt(probes, in_labels=False):
    """First caption sentence of a build-profile render: the length label, then each drawn probe's training length from its FIT spec
    (and its note, e.g. 'registered probe scored on full transcripts (OOD past 32k)'); in_labels=True when the row labels carry them
    (the post figure's short caption). '' without a profile."""
    if not BUILD:
        return ""
    miss = [p for p in probes if p not in TRAIN]
    if miss:
        sys.exit(f"STOP: drawn probes without a training length in the build profile: {miss}")
    if in_labels:
        return f"{BUILD['length_label'].strip()}. Each row label gives the probe's training length (its FIT spec). "
    items = "; ".join(f"{nice(p)}: {TRAIN[p]['train']}" + (f" ({TRAIN[p]['note']})" if TRAIN[p].get("note") else "") for p in dict.fromkeys(probes))
    return f"{BUILD['length_label'].strip()}. Training length (each probe's FIT spec): {items}. "


COMPARATOR = {"sim_minus_run1": "sim − run 1", "sim_minus_run2": "sim − run 2", "delta_full": "Δ_full (sim − mean real)"}


def registered_txt(rows, labels):
    """'sim − run 1 for Sol (a), FIBS; sim − run 2 for own_frac100' from the rows' own 'registered' fields (rows[label][set])."""
    g = {}
    for lab in labels:
        for x in (rows.get(lab) or {}).values():
            if isinstance(x, dict) and x.get("registered"):
                names = g.setdefault(x["registered"], [])
                n = nice(lab.rsplit(" ", 1)[0])
                if n not in names:
                    names.append(n)
    return "; ".join(f"{COMPARATOR.get(k, k)} for {', '.join(v)}" for k, v in sorted(g.items())) or "not stated in the record"


# On-policy probes. 09 @ 635b14564 (0d GO), replacing @ 2bb4f5146 for own_frac/own_all:
# - own_frac10/25/100_L48 and own_all{,25,10}_L48: LEAD = the registered run-mean Δ_full; Δ_full,run2 (gap.sim_minus_run2) is drawn BESIDE it
#   as a labelled sensitivity (own_frac: real run 1 AND the sim share the training launch on S_old; own_all: run1 on every origin).
# - own_a, R7, I1: LEAD = Δ_full,run2 = TPR_sim(τ_sim) − TPR_run2(τ_real), separate thresholds (gap.sim_minus_run2); Δ_full beside in the numbers file.
# The record's own per-probe "primary_comparator" wins when present; else this fixed rule.
ON_POLICY_PREFIX = ("own_frac", "own_all")  # prefix match (own_frac100_L48, own_all25_L48, ...): lead Δ_full, run-2 sensitivity beside
ON_POLICY_EXACT = ("own_a", "R7", "I1")  # exact match only (Codex r3: not R70, I10, own_abc): lead Δ_full,run2
GAP_OF = {"sim_minus_run2": "sim_minus_run2", "delta_full_run2": "sim_minus_run2", "delta_full": "delta_full"}
# 09 RULE @ b0a3e0df9 (label correction, no number changes): for probes trained on T196 run1 (own_frac*, own_a, DT) the S_old shared-launch
# sensitivity reads "real run 1 AND the sim run share the probe's training launch (S_old only)": on T196 the sims ran inside run1's sub-launches.
T196_SENS_NOTE = "sensitivity: real run 1 AND the sim run share the probe's training launch (S_old only)"
OWN_ALL_SENS_NOTE = "sensitivity: run1 shares the training launch (own_all: every origin)"  # own_all is not T196-only (not in 09's correction)
RUN2_SENS_NOTE = T196_SENS_NOTE  # kept name: the generic note now reads per probe via sens_label()
OWN_A_RUN2_SENS_NOTE = T196_SENS_NOTE


def overrides():
    """Render-time comparator overrides {probe: gap} from the render plan (env FIG_COMPARATOR_OVERRIDES); they win over the record."""
    o = json.loads(os.environ.get("FIG_COMPARATOR_OVERRIDES") or "{}")
    bad = {k: v for k, v in o.items() if not isinstance(v, str) or v not in GAP_OF}
    if bad:
        sys.exit(f"unknown comparator override(s): {bad}")
    return o


def draw_extra():
    """Probes normally not drawn (NOT_SHOWN) that the render plan asks to draw (env FIG_DRAW_EXTRA)."""
    return set(json.loads(os.environ.get("FIG_DRAW_EXTRA") or "[]"))
DT_RUN2_SENS_NOTE = T196_SENS_NOTE  # 09 @ ffba27984 (DT lead = Δ_full); wording per 09 RULE @ b0a3e0df9
ON_POLICY_NOTE = ("On-policy probes: own_frac* and own_all* plot the registered run-mean Δ_full, with Δ_full,run2 = TPR_sim(τ_sim) − "
                  "TPR_run2(τ_real) beside it as a labelled sensitivity (wording per probe); own_a, R7, I1 (where present) plot Δ_full,run2 (separate thresholds; run 1 "
                  "is a training arm), with the run-mean Δ_full, which includes run1 (training arm), in the numbers file. Their real-vs-real band "
                  "Δ_rr (run 2 − run 1) includes run 1.")


def on_policy(p):
    if p in TRAIN and "on_policy" in TRAIN[p]:  # the build profile states it per probe (e.g. sol_a_uncut off-policy, own_all_rs_uncut_L48 on)
        return bool(TRAIN[p]["on_policy"])
    return p.startswith(ON_POLICY_PREFIX) or p in ON_POLICY_EXACT


def beside_gap(p, pg):
    """For an on-policy probe, the other of {delta_full, sim_minus_run2} (kept beside the lead in the numbers file); else None."""
    return ("sim_minus_run2" if pg == "delta_full" else "delta_full") if on_policy(p) else None


def run2_sensitivity(p, pg):
    """True when the run-2 gap is DRAWN beside a Δ_full lead: own_frac*/own_all* (09 @ 635b14564), and own_a once its lead is Δ_full
    (09's conditional ruling, via a render-config override); never R7/I1."""
    if p in TRAIN and "on_policy" in TRAIN[p]:  # build profile: every on-policy probe with a Δ_full lead draws its run-2 sensitivity
        return bool(TRAIN[p]["on_policy"]) and pg == "delta_full"
    return (p.startswith(ON_POLICY_PREFIX) or p == "own_a") and pg == "delta_full"


def sens_label(p):
    if p in TRAIN and TRAIN[p].get("sens_label"):  # the build profile's wording for a retrain whose launch overlap differs
        return TRAIN[p]["sens_label"]
    return OWN_ALL_SENS_NOTE if p.startswith("own_all") else T196_SENS_NOTE  # own_frac*, own_a, DT: trained on T196 run1


def on_policy_note(entries):
    """Caption text built from what was actually plotted (Codex r7): entries = {label: {"gap", "sens", "on_policy"}}. '' if no on-policy row."""
    cat = {}  # (probe key, readout) -> category; a probe whose readouts differ is named with its readout (Codex r8); keys kept (Codex r26)
    for lab, v in entries.items():
        if v.get("on_policy"):
            pr, ro = lab.rsplit(" ", 1)
            cat[(pr, ro)] = "lead2" if v["gap"] == "sim_minus_run2" else "sens" if v.get("sens") else "plain"
    mixed = {pr for pr, _ in cat if len({c for (q, _), c in cat.items() if q == pr}) > 1}
    disp = lambda pr: DISPLAY.get(pr) or pr.replace("_L48", "")  # noqa: E731
    name = lambda pr, ro: f"{disp(pr)} ({ro})" if pr in mixed else disp(pr)  # noqa: E731
    keys = lambda c: sorted({(name(pr, ro), pr) for (pr, ro), c_ in cat.items() if c_ == c})  # noqa: E731  [(shown name, key)]
    lead2, sensk, plain = [n for n, _ in keys("lead2")], keys("sens"), [n for n, _ in keys("plain")]
    parts = []
    for note in sorted({sens_label(k) for _, k in sensk}):  # the profile's wording when it has one
        parts.append(f"{', '.join(n for n, k in sensk if sens_label(k) == note)}: the registered run-mean Δ_full leads, with Δ_full,run2 = "
                     f"TPR_sim(τ_sim) − TPR_run2(τ_real) drawn beside it (◆ / dotted ◇) as a {note}")
    if lead2:
        parts.append(f"{', '.join(lead2)}: Δ_full,run2 leads (separate thresholds; run 1 is a training arm); the run-mean Δ_full, which "
                     "includes run1 (training arm), is in the numbers file")
    if plain:
        parts.append(f"{', '.join(plain)}: Δ_full leads (record primary_comparator), no run-2 sensitivity drawn")
    return ("On-policy probes — " + "; ".join(parts) + ". Their real-vs-real band Δ_rr (run 2 − run 1) includes run 1. ") if parts else ""


def plotted_gap(rec, lab, p):
    """(gap key to plot, source). A per-probe 'primary_comparator' in the record wins (meta dict keyed by label or probe, the pooled entry, or
    any per-set row); conflicting or unknown values stop the render. Without one, the fixed on-policy list decides."""
    ov = overrides()
    if p in ov:
        return GAP_OF[ov[p]], f"render-config override ({ov[p]}) over the record's primary_comparator"
    found = []
    m = (rec.get("meta") or {}).get("primary_comparator")
    if isinstance(m, dict):
        found += [(m[k], f'meta.primary_comparator["{k}"]') for k in (lab, p) if k in m]
    pl = (rec.get("pooled") or {}).get(lab)
    if isinstance(pl, dict) and "primary_comparator" in pl:  # Codex r3: an explicit null is present (and rejected below), not ignored
        found.append((pl["primary_comparator"], f'pooled["{lab}"].primary_comparator'))
    for s_, x in ((rec.get("rows") or {}).get(lab) or {}).items():
        if isinstance(x, dict) and "primary_comparator" in x:
            found.append((x["primary_comparator"], f'rows["{lab}"]["{s_}"].primary_comparator'))
    vals = {json.dumps(v, sort_keys=True) for v, _ in found}
    if len(vals) > 1:
        sys.exit(f"conflicting primary_comparator for {lab}: {found}")
    if found:
        v, src = found[0]
        if not isinstance(v, str) or v not in GAP_OF:
            sys.exit(f"unknown primary_comparator {v!r} for {lab} at {src}: stop (adapter needed)")
        return GAP_OF[v], src
    return ("sim_minus_run2" if p in ON_POLICY_EXACT else "delta_full"), "fixed rule 09 @ 635b14564 (record has no primary_comparator)"
    # NOTE (0d): rows[*][*].registered is the APPENDIX comparator and is never consulted here; see lead_check().


def lead_check(src):
    """Assert a lead-gap source never comes from the rows' 'registered' field (0d's recheck note 4)."""
    assert "registered" not in src, f"lead gap read from the rows' 'registered' (appendix) field: {src}"
    return src


def gap_or_stop(gaps, name, where):
    """The gap node by key; a missing key stops the render (never a silent fallback to another gap)."""
    if name not in gaps:
        sys.exit(f"record lacks '{name}' at {where}: stop (no fallback)")
    return gaps[name]


WITHOUT_S4_NOTE = ("without S4: S4 cells removed from attack estimation only; calibration pools and thresholds include S4 (registered rule)")


def disclosure(rec):
    """The record's own meta.disclosure (Stage A) when present, else the constant above."""
    d = (rec.get("meta") or {}).get("disclosure")
    return d.strip() if isinstance(d, str) and d.strip() else DISCLOSURE


def pending_txt(rec):
    """Stage A meta.families_pending = {"by_root": {root: [families]}, "label": ...} or None: one caption sentence, or ''."""
    meta = rec.get("meta") or {}
    if "families_pending" not in meta or meta["families_pending"] is None:  # absent (today's records) or None (Stage A: nothing pending)
        return ""
    fp = meta["families_pending"]
    ok = (isinstance(fp, dict) and isinstance(fp.get("label"), str) and isinstance(fp.get("by_root"), dict) and fp["by_root"]
          and all(isinstance(r, str) and isinstance(v, list) and v and all(isinstance(x, str) for x in v) for r, v in fp["by_root"].items()))
    if not ok:  # Codex r6: validate the whole structure; anything else stops the render
        sys.exit(f"meta.families_pending has an unexpected shape: {fp!r}")
    roots = "; ".join(f"{r}: {', '.join(map(str, v))}" for r, v in sorted(fp["by_root"].items()))
    return f" Scorer families pending at cut-off ({fp.get('label', 'record meta.families_pending')}): {roots}."


# 0d's recheck notes (2e ~22:5x): name the interval mode; frozen-input label from meta.inputs_final; 'registered' is the APPENDIX comparator.
INTERVAL_NOTE = ("Intervals: JOINT 95% task-bootstrap (τ_sim and τ_real re-estimated inside every draw) in every panel; FIXED-threshold intervals "
                 "(threshold held at draw 0) are narrower and are in the record only, never plotted. Where the record withholds a joint interval "
                 "(fewer than 95% complete draws), the point is drawn alone and marked (‡; × on the FPR curve); no fixed interval is substituted.")


def inputs_txt(rec):
    """Frozen-input line: 'frozen inputs (INPUTS_FINAL d8be69e6): <filled label>' from meta.inputs_final; the record's top-level label /
    meta.pooled_label are stale for the frozen build and are never echoed. Records without inputs_final (today's dry runs): meta.row_label."""
    inf = (rec.get("meta") or {}).get("inputs_final")
    if isinstance(inf, dict) and inf.get("sha256"):
        label = inf.get("label", "")  # the full filled label (dozens of lines of replacements/exclusions) stays in the record, not the caption
        head = label.split(". Replacements")[0] if ". Replacements" in label else label[:160]
        return (f"Inputs: frozen inputs (INPUTS_FINAL {inf['sha256'][:8]}): {head}. Replacements, exclusions and pending items by key: the "
                "record's meta.inputs_final.label.")
    return f"Record label (pre-freeze record): {(rec.get('meta') or {}).get('row_label', '')}"


def appendix_txt(rows, labels):
    """The rows' 'registered' field is the APPENDIX comparator, never the lead (0d)."""
    return "appendix comparator per probe (record field 'registered', not plotted as the lead): " + registered_txt(rows, labels)


def undrawn_values(rec, labels, gaps_of_label):
    """Numbers-file entries for probes that are present but not drawn (e.g. own_a): the lead gap under the plan's overrides / the record's
    primary_comparator, the run-mean Δ_full, and the run-2 sensitivity when the lead is Δ_full. gaps_of_label(label) -> (gap dict, key prefix)."""
    out = {}
    for lab in labels:
        p = lab.rsplit(" ", 1)[0]
        pg, src = plotted_gap(rec, lab, p)
        lead_check(src)
        gs, kb = gaps_of_label(lab)
        e = {"lead_gap": pg, "source": src, "lead": {k: gap_or_stop(gs, pg, kb).get(k) for k in ("point", "ci95", "withheld")}, "lead_key": kb + f'["{pg}"]',
             "delta_full": {k: gap_or_stop(gs, "delta_full", kb).get(k) for k in ("point", "ci95", "withheld")}}
        if run2_sensitivity(p, pg):
            e["sensitivity_sim_minus_run2"] = {**{k: gap_or_stop(gs, "sim_minus_run2", kb).get(k) for k in ("point", "ci95", "withheld")},
                                               "key": kb + '["sim_minus_run2"]', "label": sens_label(p)}
        out[lab] = e
    return out


# Withheld intervals (Stage A: a joint interval is withheld when fewer than 95% of draws complete; the point is kept). Never substitute the
# fixed-threshold interval (0d: no mixing of modes): draw the point alone and say why.
def has_ci(node):
    return isinstance(node, dict) and isinstance(node.get("ci95"), (list, tuple)) and len(node["ci95"]) == 2


def xerr(node):
    """Errorbar half-widths in points, or None when the joint interval is withheld."""
    return [[100 * (node["point"] - node["ci95"][0])], [100 * (node["ci95"][1] - node["point"])]] if has_ci(node) else None


WITHHELD_MARK = "‡"  # 0d 10-16: a withheld joint interval is a footnote marker on the value, its reason in the caption (withheld_foot)
_WITHHELD = set()


def wmark(node):
    """'' when the joint interval is drawn, else ' ‡' (the record's reason is collected for withheld_foot)."""
    if has_ci(node):
        return ""
    _WITHHELD.add(str(node.get("withheld") or "no ci95 in the record"))
    return f" {WITHHELD_MARK}"


def withheld_foot(short=False):
    """Caption footnote for every ‡ drawn so far, or ''; short=True for the post figure's 4-line caption."""
    if not _WITHHELD:
        return ""
    if short:
        return f" {WITHHELD_MARK} joint interval withheld ({'; '.join(sorted(_WITHHELD))}): none drawn."
    return (f" {WITHHELD_MARK} joint interval withheld ({'; '.join(sorted(_WITHHELD))}): the point or ◇/◆ is drawn alone and a withheld yardstick band "
            "is not drawn; no fixed interval is substituted.")


def ci_txt(node):
    if has_ci(node):
        return f"[{100 * node['ci95'][0]:+.1f}, {100 * node['ci95'][1]:+.1f}]"
    return wmark(node).strip()


def right_end(node):
    """x position (points) for the printed value: the interval's upper end, or the point when withheld."""
    return 100 * (node["ci95"][1] if has_ci(node) else node["point"])
