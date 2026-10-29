"""COUNTSCHECK, laptop side: independent keys-and-counts derivations for two route C records.

Pre-stated in PRESTATED_OUTCOME_2026-10-20.md (@ aef74386b; 09 OK by key with two conditions). Stdlib only, Python 3.12+, so it
also runs inside the modal 1.6.1 client. It never parses the three value records itself: committed jq filters reduce them to keys,
labels and counts first. Shares and percents are NOT compared here (09's condition 1): `share_payload` hands my counts to the Modal
function, which compares them with the record and returns booleans. Items carry their label: INDEPENDENT, INTERNAL or CROSS-RECORD
(09's condition 2: only INDEPENDENT items count as an independent half).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

CELLS = ("command|L48", "command|L63", "reply|L48", "reply|L63")
SETS = ("S_old", "S_new", "S_newmains", "S_paper493")
CLASSES = ("A", "B", "sim")
PROBLEMS = {"simh_vs_b": ("sim", "B"), "a_vs_b": ("A", "B")}  # (class 1, class 0): RULE_HA L14-L15
SOL, ASTRA, QWEN = "openai/gpt-5.6-sol", "openai/gpt-6-astra", "vllm/Qwen/Qwen3.5-27B"
RATE_LIMIT = "infrastructure (rate limit)"
PILOT_PREFIX = "simh_evals_v1/"
MISSING = object()

PINS = {
    "simpool": ("104087089", "experiments/2026-10-20_routec_simpool_numbers/results/routec_simpool_numbers_2026-10-20_020501.json",
                "2e17de0c852612d8a2b1d83468997c0cb6f787dcfee9a3c93eef6edee711fd11"),
    "honest_routec": ("596bc767e", "experiments/2026-10-16_honest_sim_detector_all/results/honest_all_routec_2026-10-18_121211.json",
                      "10ed5645e0a8357150e07ac6416a06c6196e995d01eb5088b94de1d554d73c46"),
    "honest_prior": ("8b0b18e2a", "experiments/2026-10-16_honest_sim_detector_all/results/honest_all_2026-10-16_120353.json",
                     "c9ec2530e368c990136d536aa5ff74aefc3be44615698ed00f913f7b52d97b14"),
    "inputs_full": ("1a842a856", "experiments/2026-10-15_fpr_sweep_507_prep/config/INPUTS_FULL_2026-10-17_233212.json",
                    "1e1868586fe8a786f6e1094b9d5b62e362ec0e2ce241e9b0103d6584af351126"),
    "pilot_v3": ("344829af6", "experiments/2026-10-14_q3_q35_simhonest_purchase/results/simh_counted_keys_pilot_v3.json",
                 "2149c5cde368aef9e5012c8d9cda5b12fe5a93db3d27ad2d525225bb9f511b27"),
    "pilot_manifest": ("4af62db74", "experiments/2026-10-14_q3_q35_simhonest_purchase/results/manifest_simh_eval_upload_pilot.json",
                       "875167b9916560444f80be78bf516beba6711837a811bd677b82f66e704d182b"),
    "folds": ("ec56720ad", "experiments/2026-10-16_own_all_rs/results/own_all_rs_folds_2026-10-16.json",
              "c2fc5d40eb84a34a64cbb9ce7155572d1414f12ea1fbc2ec05708bd73061e299"),
    "catalogue": ("16682ef28", "experiments/2026-10-15_fpr_sweep_507_prep/results/routec_pwm_2026-10-18/"
                  "alltasks_CRD_FULL_r2_allorigins_pooledworld_catalogue_e12_2026-10-18_112818_catalogue.json.gz",
                  "4dc103729860dccffbffb4bfce522fbf2d3124502a080a79324f4fe29126f5a3"),
    "cont_headers": ("4326d76e9", "experiments/2026-10-15_e2e_sim_pipeline/results/simh_cont_headers_2026-10-17_081840.json",
                     "a7638c2d19f6a56f2953babca0ea98a7fd63854cbec21be6f1861ea452941e5d"),
    "scan_rows": ("ff8ea35a7", "experiments/2026-10-16_attack_prompt_provenance/results/openai_usage_2026-10-16_185845_scan_rows.json",
                  "0d0392fb773b1804a3ded3b4f3fd2f4dc8e7ae33b1ee660dd7200d0550410197"),
    "usage": ("de0bf0301", "experiments/2026-10-16_attack_prompt_provenance/results/openai_usage_reanalysis_2026-10-16_191817.json",
              "a0bcb2cef1ec21d598dda1795cc8b027eaf49235ba191f5d4c9643e108701102"),
    "rates_171": ("73cd0f2be", "experiments/2026-10-15_fpr_sweep_507_prep/results/appendix_a_rates_2026-10-16_121443.json",
                  "3c034ec9c74cc6989936a84e15ae07124cc93d94d0fe6fcb4bc6a4e958fd6e64"),
    "note_sf": ("d1db55766", "experiments/2026-10-15_e2e_sim_pipeline/results/NOTE_detector_solfree_launch_2026-10-17_v2.md",
                "15d0b11e056e4516fc825e5244c41c0f2bc64b0ca64bca8123839db1fc446f04"),
}
CATALOGUE_JSON_SHA256 = "72432318a3703d4675a5c4530d77f598a003af351340d895d305b9bd1050adc8"
FILTERS = ("extract_honest_all.jq", "extract_simpool_counts.jq", "extract_usage_bits.jq")

# Registered expectations, from text only (PRESTATED "expected" column).
REG_TABLE = {  # RULE_HA L22-L24 (+20 pilot in sim S_old, RULE_RC L4963): (e1, e2)
    "sim": {"S_old": (191, 0), "S_new": (305, 0), "S_newmains": (339, 0), "S_paper493": (34, 0)},
    "B": {"S_old": (194, 191), "S_new": (306, 303), "S_newmains": (344, 342), "S_paper493": (34, 0)},
    "A": {"S_old": (192, 0), "S_new": (309, 305), "S_newmains": (342, 334), "S_paper493": (34, 0)},
}
REG_TOTALS = {"A": 1516, "B": 1714, "sim": 869}
REG_V83_RELAY = (50, 848)  # NOTE_SP L11
NOMC = "sim|nm014_auto_workflow|e1|24a46578"  # NOTE_SP L114-L115
S7_LABEL = "DEFINITION-DIFFERENCE"  # PRESTATED Amendment 1 (09's terms): pre-stated FALSE; not printed; no independent half
S7_NOTE = ("definition difference: the producer's per-path source count over all seen paths, without a 0 bucket, is defined only in its "
           "code; the checker follows NOTE_SF L28")
S7_L28_READING = ("L28 names the buckets 1, 2 and 3 and no 0 bucket, and does not say which paths are counted (the pool's, or every path "
                  "seen), so it reads neither way clearly. The checker's 0 bucket and pool scope are not in L28; no producer deviation "
                  "from registered text is claimed.")


class Stop(RuntimeError):
    """A pre-stated STOP. Messages carry keys or counts only."""


# ---------------------------------------------------------------- loading (git show, sha256, jq)

def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def pinned_bytes(repo: Path, name: str) -> bytes:
    commit, path, want = PINS[name]
    b = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{path}"], capture_output=True, check=True).stdout
    if sha256(b) != want:
        raise Stop(f"sha256 mismatch: {name}")
    return b


def jq(filter_path: Path, data: bytes):
    out = subprocess.run(["jq", "-c", "-f", str(filter_path)], input=data, capture_output=True, check=True).stdout
    return json.loads(out)


def load_all(repo: Path, here: Path) -> dict:
    """Every input, sha-checked. Value records go through jq first; the rest are keys / labels / counts."""
    raw = {n: pinned_bytes(repo, n) for n in PINS}
    cat = gzip.decompress(raw["catalogue"])
    if sha256(cat) != CATALOGUE_JSON_SHA256:
        raise Stop("sha256 mismatch: catalogue (decompressed)")
    f = {n: here / n for n in FILTERS}
    return {
        "ext_rc": jq(f["extract_honest_all.jq"], raw["honest_routec"]),
        "ext_prior": jq(f["extract_honest_all.jq"], raw["honest_prior"]),
        "ext_sp": jq(f["extract_simpool_counts.jq"], raw["simpool"]),
        "scan_bits": jq(f["extract_usage_bits.jq"], raw["scan_rows"]),
        "usage_bits": jq(f["extract_usage_bits.jq"], raw["usage"]),
        "simpool_bytes": raw["simpool"],  # passed through to the Modal compare unparsed
        **{n: json.loads(raw[n]) for n in ("inputs_full", "pilot_v3", "pilot_manifest", "folds", "cont_headers", "rates_171")},
        "catalogue": json.loads(cat),
        "filter_sha256": {n: sha256(p.read_bytes()) for n, p in f.items()},
        "s7_note_sf_l28": s7_note_sf_l28(raw["note_sf"]),
    }


def s7_note_sf_l28(note_bytes: bytes) -> dict:
    """PRESTATED Amendment 1: NOTE_SF L28 verbatim (sha-checked through PINS) and my reading, stated before the run. Information only."""
    return {"source": "{}:{} L28".format(*PINS["note_sf"][:2]), "verbatim": note_bytes.decode().splitlines()[27].strip(),
            "reading": S7_L28_READING, "producer_deviation_claimed": False}


# ---------------------------------------------------------------- shared rules (D1-D9)

def fields(key: str) -> tuple[str, str, int]:
    """D1: `<class>|<cell>|<epoch>|<hash>`."""
    p = key.split("|")
    if len(p) != 4 or p[0] not in CLASSES or not re.fullmatch(r"e\d+", p[2]):
        raise Stop(f"unexpected run-key form ({len(p)} fields)")
    return p[0], p[1], int(p[2][1:])


def path_of(value: str) -> str:
    """D3: run_keys value minus its trailing `#e<n>`."""
    m = re.fullmatch(r"(.+)#e\d+", value)
    if not m:
        raise Stop("run_keys value without a #e<n> suffix")
    return m.group(1)


def copy_verdict(models, has_header=True, sol_first=True) -> str:
    """D8 per copy (NOTE_SF L13-L19, L23): SOL / INCOMPLETE / FREE. sol_first=False is the reported diagnostic order."""
    models = models or {}
    incomplete = (not has_header) or (not models) or (not models.get(ASTRA)) or (not models.get(QWEN))
    sol = bool(models.get(SOL))
    if sol_first:
        return "SOL" if sol else ("INCOMPLETE" if incomplete else "FREE")
    return "INCOMPLETE" if incomplete else ("SOL" if sol else "FREE")


def path_verdict(verdicts, stop_on_conflict=True) -> str:
    """D8 per path: SOL+FREE STOPs; any SOL -> sol_using; any INCOMPLETE -> unknown; no copy -> unknown."""
    v = set(verdicts)
    if "SOL" in v and "FREE" in v:
        if stop_on_conflict:
            raise Stop("a path has both SOL and FREE copies")
        return "conflict"
    if "SOL" in v:
        return "sol_using"
    if "INCOMPLETE" in v or not v:
        return "unknown"
    return "sol_free"


def rec(d, *path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return MISSING
        d = d[k]
    return d


class Items:
    """Result rows. A row never holds a record value: only my count, the key path and booleans."""

    def __init__(self):
        self.rows = []

    def add(self, iid, record, key, mine, record_value, label, cite, registered=MISSING, equal=MISSING, **extra):
        row = {"id": iid, "record": record, "record_key": key, "mine": mine, "label": label, "cite": cite, "where": "laptop", **extra}
        if record_value is MISSING and equal is MISSING:
            row["equal"], row["record_key_absent"] = False, True
        elif equal is not MISSING:
            row["equal"] = bool(equal)
        else:
            row["equal"] = same(record_value, mine)
        if registered is not MISSING:
            row["registered"], row["equal_registered"] = registered, mine == registered
        self.rows.append(row)


def same(a, b) -> bool:
    if isinstance(b, bool) or isinstance(a, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(b, int):
        return isinstance(a, (int, float)) and a == b
    return a == b


# ---------------------------------------------------------------- record (i): honest_all_routec

def derive_honest(ctx: dict, it: Items) -> dict:
    rc, pr, cell_set = ctx["ext_rc"], ctx["ext_prior"], ctx["folds"]["cell_set"]

    def set_of(cell):  # D2 (RULE_HA L26)
        s = cell_set.get(cell)
        if s not in SETS:
            raise Stop("a cell has no registered set in the fold list")
        return s

    rc_keys, pr_keys = set(rc["run_keys"]), set(pr["run_keys"])
    rc_path = {k: path_of(v) for k, v in rc["run_keys"].items()}
    pr_path = {k: path_of(v) for k, v in pr["run_keys"].items()}
    by_path = defaultdict(list)
    for k, p in rc_path.items():
        by_path[p].append(k)

    # D4: route C = prior + the 20 pilot runs, matched by path
    pilot = []
    for row in ctx["pilot_v3"]["keys"]:
        vp = row["volume_path"]
        if not vp.startswith(PILOT_PREFIX):
            raise Stop("pilot volume_path without the simh_evals_v1/ prefix")
        ks = [k for k in by_path.get(vp[len(PILOT_PREFIX):], []) if fields(k)[0] == "sim"]
        if len(ks) != 1:
            raise Stop(f"a pilot path matches {len(ks)} route C sim keys")
        pilot.append(ks[0])
    pilot_keys = set(pilot)
    if len(pilot_keys) != len(pilot):
        raise Stop("two pilot paths map to one key")
    if pilot_keys & pr_keys:
        raise Stop("a pilot key is in the 10-16 record")
    derived = pr_keys | pilot_keys
    R, IND, INT, XR = "i", "INDEPENDENT", "INTERNAL", "CROSS-RECORD"

    # H1, H2
    cls_n = Counter(fields(k)[0] for k in derived)
    for c in CLASSES:
        it.add(f"H1.{c}", R, f".epochs_per_class.{c}", cls_n[c], rec(rc, "epochs_per_class", c), IND,
               "RULE_HA L22-L24; RULE_RC L4963, L4976", registered=REG_TOTALS[c])
    it.add("H2.key_set", R, ".run_keys (key set)", len(derived), None, IND, "RULE_RC L4973, L4975; NOTE_HA L29-L32",
           registered=4099, equal=derived == rc_keys)
    moved = sum(1 for k in pr_keys & rc_keys if rc_path[k] != pr_path[k])
    it.add("H2.shared_path_moves", R, ".run_keys[<prior key>] (path)", moved, None, IND, "NOTE_HA L29-L32",
           registered=0, equal=moved == 0)
    it.add("H2.pilot_matched", R, ".run_keys (pilot paths)", len(pilot_keys), None, IND, "RULE_RC L4966",
           registered=20, equal=len(pilot_keys) == len(ctx["pilot_v3"]["keys"]))

    # H3, H4
    tally = Counter((fields(k)[0], set_of(fields(k)[1]), fields(k)[2]) for k in derived)
    grid = {(c, s, f"e{e}") for c in CLASSES for s in SETS for e in (1, 2)}
    rec_grid = {(c, s, e) for c, ss in (rc.get("per_class_set_epoch") or {}).items() for s, es in ss.items() for e in es}
    it.add("H3.grid", R, ".per_class_set_epoch (key grid)", len(grid), None, IND, "RULE_HA L20-L24", equal=rec_grid == grid)
    for c in CLASSES:
        for s in SETS:
            for e in (1, 2):
                it.add(f"H3.{c}.{s}.e{e}", R, f".per_class_set_epoch.{c}.{s}.e{e}", tally[(c, s, e)],
                       rec(rc, "per_class_set_epoch", c, s, f"e{e}"), IND, "RULE_HA L20-L26; RULE_RC L4963",
                       registered=REG_TABLE[c][s][e - 1])
                if c != "sim" and s != "S_paper493":
                    it.add(f"H4.{c}.{s}.e{e}", R, f".inputs.pool_sizes.{c}.{s}.e{e}", tally[(c, s, e)],
                           rec(rc, "pool_sizes", c, s, f"e{e}"), IND, "RULE_HA L30", registered=REG_TABLE[c][s][e - 1])

    # H5
    bad = {"set": 0, "epoch": 0, "cell": 0}
    for k in rc_keys:
        c, cell, e = fields(k)
        bad["set"] += rc["run_set"].get(k) != set_of(cell)
        bad["epoch"] += rc["run_epoch"].get(k) != e
        bad["cell"] += rc["run_cell"].get(k) != cell
    for n, v in bad.items():
        it.add(f"H5.run_{n}_disagreements", R, f".run_{n}", v, None, IND, "RULE_HA L26", registered=0, equal=v == 0)

    # H6 + D5
    def sold_sim(keys):
        return {k for k in keys if fields(k)[0] == "sim" and set_of(fields(k)[1]) == "S_old"}

    rc_sold, pr_sold = sold_sim(derived), sold_sim(pr_keys)
    for name, mine, reg in (("sold_sim", len(rc_sold), 191), ("registered", len(pr_sold), 171),
                            ("pilot", len(ctx["pilot_v3"]["keys"]), 20)):
        it.add(f"H6.{name}", R, f".inputs.routec_sold_sim.{name}", mine, rec(rc, "routec_sold_sim", name), IND,
               "RULE_RC L4963, L4966, L4973; NOTE_HA L103", registered=reg)
    adm = ctx["inputs_full"]["freeze_checks"]["tau_sim"]["S_old"]["fill_in_admitted"]

    def cells_of(entries):
        out = set()
        for e in entries:
            p = e.split("|")
            if len(p) != 2:
                raise Stop("fill-in admitted entry is not <world>|<cell>")
            out.add(p[1])
        return out

    parts = {
        "cont_149": {r["path"].split("/")[-2] for r in ctx["cont_headers"]["rows"]},
        "fill_8": cells_of(adm["simh_t196fill_v1__sim_honest"]["admitted"]),
        "g4_14": cells_of(adm["sp493_g4_simh_v1_try2__sim_honest"]["admitted"]),
        "pilot_20": {r["task"] for r in ctx["pilot_v3"]["keys"]},
    }
    union = set().union(*parts.values())
    d5 = len(union) == sum(len(v) for v in parts.values()) and union == {fields(k)[1] for k in rc_sold}
    it.add("H6.D5_composition", R, "S_old sim cells (by D5)", len(union), None, IND, "RULE_HA L22; RULE_RC L4963",
           registered=191, equal=d5)
    for n, v in parts.items():
        it.add(f"H6.D5_{n}", R, "S_old sim part", len(v), None, IND, "RULE_HA L22", registered=int(n.split("_")[-1]),
               equal=v <= {fields(k)[1] for k in rc_sold})
    pin = ctx["inputs_full"]["s_old_simh_extra_roots"]["simh_q4f_v1_pilot__sim_honest"]
    commit, path, sha = PINS["pilot_v3"]
    pin_ok = pin["sha256"] == sha and pin["path"] == path and (commit.startswith(pin["commit"]) or pin["commit"].startswith(commit))
    it.add("H6.inputs_full_pilot_pin", R, "inputs_full.s_old_simh_extra_roots (pin)", pin_ok, None, IND, "RULE_RC L4970",
           registered=True, equal=pin_ok)

    # H7
    tabs = ("run_set", "run_folds", "run_main", "run_cell", "run_epoch", "run_source")
    eq = all(k in rc_keys and rc_path[k] == pr_path[k] and all(rc[t].get(k) == pr[t].get(k) for t in tabs) for k in pr_keys)

    def fold_map(ext):
        m = defaultdict(set)
        for k, main in ext["run_main"].items():
            m[main].add(ext["run_folds"][k])
        return m

    fm_rc, fm_pr = fold_map(rc), fold_map(pr)
    eq = eq and fm_rc == fm_pr and all(len(v) == 1 for v in fm_rc.values())
    pm = "routec_prior_match"
    it.add("H7.equal", R, f".inputs.{pm}.equal", eq, rec(rc, pm, "equal"), XR, "NOTE_HA L29-L32; RULE_RC L4974-L4975",
           registered=True)
    it.add("H7.n_prior", R, f".inputs.{pm}.n_prior", len(pr_keys), rec(rc, pm, "n_prior"), XR, "NOTE_HA L104", registered=4079)
    it.add("H7.n_pilot", R, f".inputs.{pm}.n_pilot", len(rc_keys - pr_keys), rec(rc, pm, "n_pilot"), XR, "NOTE_HA L104",
           registered=20)
    it.add("H7.n_mains", R, f".inputs.{pm}.n_mains", len(set(rc["run_main"].values())), rec(rc, pm, "n_mains"), XR,
           "my reading (distinct .run_main); a mismatch is a definition difference")
    it.add("H8.meta_checked", R, ".skip.meta_checked", len(rc_keys), rec(rc, "meta_checked"), IND, "NOTE_HA L115",
           registered=4099)

    # H9-H13
    zr_rc, zr_pr = zero_rows(rc, rc_keys), zero_rows(pr, pr_keys)
    value_to_key = {v: k for k, v in rc["run_keys"].items()}
    for cell in CELLS:
        zr = zr_rc[cell]
        scored = set().union(*(set(p["run_score_keys"]) for p in rc["cells"][cell]["problems"].values()))
        it.add(f"H9.{cell}.unscored", R, f".cells[{cell}] run_scores", len(zr & scored), None, INT, "RULE_HA L51",
               registered=0, equal=not (zr & scored))
        z = rc["zero_row_runs"].get(cell, {})
        it.add(f"H9.{cell}.n", R, f".zero_row_runs[{cell}].n", len(zr), z.get("n", MISSING), INT, "record .settings.zero_row_rule")
        for c in CLASSES:
            it.add(f"H9.{cell}.by_class.{c}", R, f".zero_row_runs[{cell}].by_class.{c}", sum(fields(k)[0] == c for k in zr),
                   rec(z, "by_class", c), INT, "RULE_HA L51")
        for s in SETS:
            it.add(f"H9.{cell}.by_set.{s}", R, f".zero_row_runs[{cell}].by_set.{s}",
                   sum(set_of(fields(k)[1]) == s for k in zr), rec(z, "by_set", s), INT, "RULE_HA L26, L51")
        mine_runs = sorted((rc["run_keys"][k], k, set_of(fields(k)[1])) for k in zr)
        rec_runs = sorted((r["key"], r["name"], r["set"]) for r in z.get("runs", []))
        it.add(f"H9.{cell}.runs", R, f".zero_row_runs[{cell}].runs", len(mine_runs), None, INT, "RULE_HA L51",
               equal=mine_runs == rec_runs)
        for src, lst in (("uncut", rc["uncut_zero_row_runs"].get(cell)), ("runs_without_rows", rc["cells"][cell]["runs_without_rows"])):
            keys = as_keys(lst or [], rc_keys, value_to_key, by_path)
            it.add(f"H9.{cell}.{src}", R, f"{'.uncut_gates.zero_row_runs' if src == 'uncut' else '.cells[' + cell + '].runs_without_rows'}",
                   len(zr), None, INT, "RULE_HA L51", equal=lst is not None and keys == zr)
        for p, (c1, c0) in PROBLEMS.items():
            prob = rc["cells"][cell]["problems"][p]
            for cls_name, c in (("class1", c1), ("class0", c0)):
                mine = len({k for k in derived if fields(k)[0] == c} - zr)
                it.add(f"H10.{cell}.{p}.{cls_name}", R, f".cells[{cell}].problems.{p}.n_runs.{cls_name}", mine,
                       rec(prob, "n_runs", cls_name), IND, "RULE_HA L14-L15 (minus INTERNAL zero-row)")
                n_keys = sum(fields(k)[0] == c for k in prob["run_score_keys"])
                it.add(f"H10.{cell}.{p}.{cls_name}.score_keys", R, f".cells[{cell}].problems.{p}.run_scores (keys)", mine,
                       None, IND, "RULE_HA L14-L15", equal=n_keys == mine)
        rpr = rc["cells"][cell]["rows_per_run"]
        it.add(f"H13.{cell}.rows", R, f".cells[{cell}].rows", sum(rpr.values()), rc["cells"][cell].get("rows", MISSING),
               INT, "RULE_HA L54")
    for ro in ("command", "reply"):
        mine = rc["cells"][f"{ro}|L48"]["rows_per_run"] == rc["cells"][f"{ro}|L63"]["rows_per_run"]
        it.add(f"H13.equal_across_layers.{ro}", R, f".rows_per_run_equal_across_layers.{ro}", mine,
               rec(rc, "rows_per_run_equal_across_layers", ro), INT, "RULE_HA L54")

    n_fail = 0
    for cell in CELLS:
        for p in PROBLEMS:
            prob = rc["cells"][cell]["problems"][p]
            n_fail += sum(prob.get(f) is not True for f in ("observed_complete", "null_complete", "observed_certified", "null_certified"))
        n_fail += sum(not all(col) for col in rc["cells"][cell]["certified_per_column"])
    it.add("H11.n_failures_derived", R, "certification / completeness booleans", n_fail, None, INT, "RULE_HA L44-L47",
           registered=0, equal=n_fail == 0)
    for name, key in (("n_failed", ".failed"), ("n_required_gaps", ".required_gaps"), ("n_uncut_failures", ".uncut_gates.failures")):
        it.add(f"H11.{name}", R, f"{key} (length)", n_fail, None, INT, "RULE_HA L44-L47",
               equal=(rc[name] == 0) == (n_fail == 0))
    drops = sorted(tuple(d[f] for f in ("world", "source", "reason", "n")) for d in pr["drops"])
    it.add("H12.drops", R, ".inputs.drops", len(drops), None, XR, "RULE_HA L30-L31; RULE_RC L4963",
           equal=drops == sorted(tuple(d[f] for f in ("world", "source", "reason", "n")) for d in rc["drops"]))
    return {"derived": derived, "rc_path": rc_path, "pr_path": pr_path, "zr_rc": zr_rc, "zr_pr": zr_pr, "pilot_keys": pilot_keys}


def zero_rows(ext: dict, keys: set) -> dict:
    """D6: a run key with no entry, or 0, in the cell's rows_per_run."""
    return {c: {k for k in keys if not ext["cells"][c]["rows_per_run"].get(k, 0)} for c in CELLS}


def as_keys(lst, keys, value_to_key, by_path) -> set:
    out = set()
    for e in lst:
        if e in keys:
            out.add(e)
        elif e in value_to_key:
            out.add(value_to_key[e])
        elif len(by_path.get(e, [])) == 1:
            out.add(by_path[e][0])
        else:
            out.add(("UNMATCHED", len(out)))
    return out


# ---------------------------------------------------------------- record (ii): simpool items 1, 2, 4 (counts only)

def derive_simpool(ctx: dict, h: dict, bits: dict | None, it: Items) -> dict:
    """Counts and count-based booleans only. Returns the counts payload for the Modal share / percent compare."""
    sp, rc, pr = ctx["ext_sp"], ctx["ext_rc"], ctx["ext_prior"]
    R, IND, INT, XR = "ii", "INDEPENDENT", "INTERNAL", "CROSS-RECORD"
    role_s = [(ck, r) for ck, r in ctx["catalogue"]["runs"].items() if "S" in (r.get("roles") or [])]

    def index(keys):
        idx = defaultdict(list)
        for k in keys:
            c, cell, e = fields(k)
            if c == "sim":
                idx[(cell, e)].append(k)
        return idx

    def map_pool(entries, idx):  # D7
        out = {}
        for ck, r in entries:
            p = ck.split("|")
            if len(p) != 2 or not re.fullmatch(r"e\d+", str(r.get("epoch"))):
                raise Stop("unexpected catalogue key or epoch form")
            ks = idx.get((p[1], int(r["epoch"][1:])), [])
            if len(ks) != 1:
                raise Stop(f"a role-S run maps to {len(ks)} sim keys")
            out[ck] = ks[0]
        if len(set(out.values())) != len(out):
            raise Stop("two role-S runs map to one sim key")
        return out

    pool = {"routec": map_pool(role_s, index(h["derived"])),
            "v83": map_pool([e for e in role_s if e[1].get("world") != "simh_pilot"], index(set(pr["run_keys"])))}
    paths = {"routec": h["rc_path"], "v83": h["pr_path"]}

    base = defaultdict(list)  # path -> [(source, has_header, models)]
    for r in ctx["scan_bits"]:
        base[r["path"]].append(("scan_rows", True, r["models"]))
    for r in ctx["usage_bits"]:
        base[r["path"]].append(("usage", True, r["models"]))
    for r in ctx["cont_headers"]["rows"]:
        base[r["path"]].append(("header_scan", bool(r.get("has_header")), r.get("models_any_tokens")))
    copies = {"v83": base, "routec": defaultdict(list, {k: list(v) for k, v in base.items()})}
    if bits is not None:
        for r in bits["rows"]:
            copies["routec"][r["path"]].append(("header_scan", bool(r["has_header"]), r["models_any_tokens"]))

    def tally(keys, scope):
        out, alt, nsrc = Counter(), Counter(), Counter()
        for k in keys:
            cs = copies[scope].get(paths[scope][k], [])
            out[path_verdict([copy_verdict(m, hh) for _, hh, m in cs])] += 1
            alt[path_verdict([copy_verdict(m, hh, sol_first=False) for _, hh, m in cs], stop_on_conflict=False)] += 1
            nsrc[str(len({s for s, _, _ in cs}))] += 1
        return {"n": len(keys), "sol_using": out["sol_using"], "sol_free": out["sol_free"], "unknown": out["unknown"]}, alt, nsrc

    payload = {"item1": {"pool": {}, "cells": {"routec": {}, "v83": {}}}, "item2": {}}
    i1 = sp["item1"]
    for scope in ("routec", "v83"):
        counts, alt, nsrc = tally(set(pool[scope].values()), scope)
        payload["item1"]["pool"][scope] = counts
        reg_map = {"routec": {"n": 868}, "v83": {"n": 848, "sol_using": 50, "unknown": 0}}[scope]  # NOTE_SP L11, L114
        for f in ("n", "sol_using", "sol_free", "unknown"):
            reg = reg_map.get(f, MISSING)
            it.add(f"S{'1' if scope == 'routec' else '5'}.pool.{scope}.{f}", R, f'.items["1_sim_footnote"].pool.{scope}.{f}',
                   counts[f], rec(i1, "pool", scope, f), IND, "REC_TEXT .what; NOTE_SF L13-L19; NOTE_SP L11", registered=reg)
        it.add(f"S1.diag_incomplete_first.{scope}", R, "(diagnostic, no record key)", dict(alt), None, IND, "PRESTATED D8",
               equal=all(alt[x] == counts[y] for x, y in (("sol_using", "sol_using"), ("sol_free", "sol_free"), ("unknown", "unknown"))))
        rec_ns = rec(i1, "n_paths_by_number_of_sources", scope)
        rec_ns = rec_ns if isinstance(rec_ns, dict) else {}
        for k in sorted(set(nsrc) | set(rec_ns)):
            it.add(f"S7.{scope}.{k}", R, f'.items["1_sim_footnote"].n_paths_by_number_of_sources.{scope}.{k}', nsrc.get(k, 0),
                   rec_ns.get(k, 0 if k in nsrc else MISSING), S7_LABEL, "NOTE_SF L28; PRESTATED Amendment 1",
                   prestated_false=True, note=S7_NOTE)
        ext = rc if scope == "routec" else pr
        for cell in CELLS:
            ks = {k for k in ext["cells"][cell]["problems"]["simh_vs_b"]["run_score_keys"] if fields(k)[0] == "sim"}
            c_counts, _, _ = tally(ks, scope)
            payload["item1"]["cells"][scope][cell] = c_counts
            for f in ("n", "sol_using", "sol_free", "unknown"):
                reg = {"routec": 868, "v83": 848}[scope] if f == "n" else MISSING
                it.add(f"S6.{scope}.{cell}.{f}", R, f'.items["1_sim_footnote"].detector_cells.{scope}[{cell}].{f}', c_counts[f],
                       rec(i1, "detector_cells", scope, cell, f), IND, "REC_TEXT .v83_source; RULE_HA L14", registered=reg)
    v = payload["item1"]["pool"]["v83"]
    it.add("S9.v83_counts_equal_relay", R, '.items["1_sim_footnote"].v83_reproduces_printed["pool counts equal the 10-17 relay (sol, n)"]',
           (v["sol_using"], v["n"]) == REG_V83_RELAY, i1.get("v83_counts_equal_relay", MISSING), IND, "NOTE_SP L11", registered=True)

    # S8, S8b: pilot header counts (my Modal read)
    if bits is not None:
        pilot_paths = {r["path"] for r in bits["rows"]}
        n_unknown = sum(path_verdict([copy_verdict(m, hh) for _, hh, m in copies["routec"].get(p, [])]) == "unknown"
                        for p in pilot_paths)
        ph = i1["pilot_header_scan"]
        for f, mine, reg in (("n_files", len(bits["rows"]), 20), ("n_header_read", sum(bool(r["has_header"]) for r in bits["rows"]), 20),
                             ("n_unknown", n_unknown, 0)):
            it.add(f"S8.{f}", R, f'.items["1_sim_footnote"].pilot_header_scan.{f}', mine, ph.get(f, MISSING), IND,
                   "NOTE_SP L18-L22, L29-L30", registered=reg)
        rec_rows = {norm_pilot(r["path"]): r for r in ph.get("rows", [])}
        agree = 0
        for r in bits["rows"]:
            o = rec_rows.get(norm_pilot(r["path"]))
            if o is not None and bool(o.get("has_header")) == bool(r["has_header"]) and all(
                    (o.get("models_any_tokens") or {}).get(m) == r["models_any_tokens"].get(m) for m in (SOL, ASTRA, QWEN)):
                agree += 1
        it.add("S8b.rows_agree", R, '.items["1_sim_footnote"].pilot_header_scan.rows[]', agree, None, IND, "NOTE_SP L20-L21",
               registered=20, equal=agree == len(bits["rows"]) == len(rec_rows))

    # [^failed]: counts only (code-defined; stays on Aswin's list)
    rates, i2 = ctx["rates_171"], sp["item2"]
    common = list(rates["headline_common_sets"]["sets"])
    it.add("F1.headline_sets", R, '.items["2_failed_footnote"].headline_sets', len(common), None, "copy check", "RATES_DEF",
           registered=3, equal=sorted(i2.get("headline_sets") or []) == sorted(common))
    fa = [a for s in common for a in rates["sim"]["failed_attempts"].get(s, [])]
    v83 = {"failed": len(fa), "rate_limit": sum(a["category"] == RATE_LIMIT for a in fa),
           "launched": sum(rates["sim"]["by_set_arm"][f"{s}|sim_honest"]["launched"] for s in common)}
    y = rates["headline_common_sets"]["Y_sim"]
    for f, other in (("failed", y["failed"]), ("launched", y["launched"]), ("rate_limit", y["by_category"].get(RATE_LIMIT))):
        it.add(f"F2.internal.{f}", R, f"rates_171 .headline_common_sets.Y_sim ({f})", v83[f], None, INT, "RATES_DEF",
               equal=other == v83[f])
    pilot_common = sorted(ck for ck, r in ctx["catalogue"]["runs"].items()
                          if r.get("world") == "simh_pilot" and set(r.get("sets") or []) & set(common))
    routec = dict(v83) if not pilot_common else None
    for f in ("failed", "launched", "rate_limit"):
        it.add(f"F2.v83.{f}", R, f'.items["2_failed_footnote"].v83.{f}', v83[f], rec(i2, "v83", f), INT, "RATES_DEF; NOTE_SP L12")
        if routec is not None:
            it.add(f"F3.routec.{f}", R, f'.items["2_failed_footnote"].routec.{f}', routec[f], rec(i2, "routec", f),
                   INT, "NOTE_SP L12 (base INTERNAL to rates_171; pilot delta by catalogue sets)")
    it.add("F4.pilot_runs_in_headline_sets", R, '.items["2_failed_footnote"].pilot_runs_in_headline_sets', len(pilot_common),
           None, IND, "NOTE_SP L12", registered=0, equal=sorted(i2.get("pilot_runs_in_headline_sets") or []) == pilot_common)
    pm, pa = ctx["pilot_manifest"], i2.get("pilot_attempts_S_old_apart") or {}
    for f, mine, r in (("files", len(pm["files"]), pa.get("files", MISSING)),
                       ("counted_attempts_without_eval.simh", len(pm["counted_attempts_without_eval"]["simh"]),
                        pa.get("counted_attempts_without_eval_simh", MISSING)),
                       ("manifest_counts.files", pm["counts"]["files"], rec(pa, "manifest_counts", "files")),
                       ("manifest_counts.usable", pm["counts"]["usable"], rec(pa, "manifest_counts", "usable"))):
        it.add(f"F5.{f}", R, f'.items["2_failed_footnote"].pilot_attempts_S_old_apart.{f}', mine, r, "count / copy", "NOTE_SP L44")
    payload["item2"] = {"v83": v83, "routec": routec}

    # Item 4
    i4 = sp["item4"]
    rk = "run_keys (all sim runs of the record)"
    for scope, ext, keys, zr in (("routec", rc, h["derived"], h["zr_rc"]), ("v83", pr, set(pr["run_keys"]), h["zr_pr"])):
        sim_keys = {k for k in keys if fields(k)[0] == "sim"}
        pool_keys = set(pool[scope].values())
        n_det = {}
        for cell in CELLS:
            det = {k for k in ext["cells"][cell]["problems"]["simh_vs_b"]["run_score_keys"] if fields(k)[0] == "sim"}
            n_det[cell] = len(det)
            reg = {"routec": 868, "v83": 848}[scope]
            base_key = f'.items["4_honest_detector_cohort"].{scope}[{cell}]'
            it.add(f"C1.{scope}.{cell}", R, f"{base_key}.n_detector_sim", len(det), rec(i4, scope, cell, "n_detector_sim"), IND,
                   "REC_TEXT item 4 .v83_source; RULE_HA L14", registered=reg)
            it.add(f"C1.{scope}.{cell}.second_route", R, "(D4 sim minus D6 zero-row)", len(sim_keys - zr[cell]), None, INT,
                   "PRESTATED C1", equal=len(sim_keys - zr[cell]) == len(det))
            it.add(f"C2.{scope}.{cell}", R, f"{base_key}.n_pool", len(pool_keys), rec(i4, scope, cell, "n_pool"), XR,
                   "REC_TEXT .what (item 4)", registered=reg)
            it.add(f"C3.{scope}.{cell}.n_both", R, f"{base_key}.n_both", len(det & pool_keys), rec(i4, scope, cell, "n_both"), XR,
                   "REC_TEXT .what (item 4)", registered=reg)
            it.add(f"C3.{scope}.{cell}.equal", R, f"{base_key}.equal", det == pool_keys, rec(i4, scope, cell, "equal"), XR,
                   "REC_TEXT .what (item 4)", registered=True)
        it.add(f"C4.{scope}", R, f'.items["4_honest_detector_cohort"].pool_n.{scope}', len(pool_keys), rec(i4, "pool_n", scope), XR,
               "REC_TEXT pool .what; definition-difference note as pre-stated", registered={"routec": 868, "v83": 848}[scope])
        it.add(f"C5.{scope}", R, f'.items["4_honest_detector_cohort"].{scope}["{rk}"].n', len(sim_keys), rec(i4, scope, rk, "n"),
               IND, "RULE_RC L4963", registered={"routec": 869, "v83": 849}[scope])
        only = sorted(sim_keys - pool_keys)
        it.add(f"C6.{scope}", R, f'.items["4_honest_detector_cohort"].{scope}["{rk}"].detector_only_keys', only, None, XR,
               "REC_TEXT .what (item 4); registered = NOTE_SP L114-L115", registered=[NOMC],
               equal=sorted(rec(i4, scope, rk, "detector_only_keys") or []) == only)
        zc = rec(i4, scope, rk, "zero_row_cells_of_detector_only")
        zc = zc if isinstance(zc, dict) else {}
        mine_zc = {k: sorted(c for c in CELLS if k in zr[c]) for k in only}
        it.add(f"C7.{scope}", R, f'.items["4_honest_detector_cohort"].{scope}["{rk}"].zero_row_cells_of_detector_only',
               sum(len(v) for v in mine_zc.values()), None, INT, "NOTE_SP L115", registered=4,
               equal={k: sorted(v) for k, v in zc.items()} == mine_zc)
        payload.setdefault("item4", {})[scope] = {"n_det": n_det, "pool_n": len(pool_keys)}
    flags = i4.get("flags") or {}
    p4 = payload["item4"]
    for cell in CELLS:
        mine = "changed" if p4["routec"]["n_det"][cell] != p4["v83"]["n_det"][cell] else "unchanged"
        f = flags.get(f"cell {cell} n_detector_sim")
        it.add(f"C8.{cell}", R, f'.items["4_honest_detector_cohort"].flags["cell {cell} n_detector_sim"]', mine, None, IND,
               "NOTE_SP L114", registered="changed", equal=first_word(f) == mine)
    mine = "changed" if p4["routec"]["pool_n"] != p4["v83"]["pool_n"] else "unchanged"
    it.add("C8.pool_n", R, '.items["4_honest_detector_cohort"].flags.pool_n', mine, None, XR, "NOTE_SP L114",
           registered="changed", equal=first_word(flags.get("pool_n")) == mine)
    return payload


def norm_pilot(p: str) -> str:
    for pre in ("/a25/" + PILOT_PREFIX, PILOT_PREFIX):
        if p.startswith(pre):
            return p[len(pre):]
    return p


def first_word(s) -> str | None:
    if not isinstance(s, str) or not s.strip():
        return None
    return s.split()[0].strip(":;,.()")


def derive_all(ctx: dict, bits: dict | None) -> tuple[Items, dict]:
    it = Items()
    h = derive_honest(ctx, it)
    payload = derive_simpool(ctx, h, bits, it)
    payload["simpool_sha256"] = PINS["simpool"][2]
    return it, payload


def summary(rows: list) -> dict:
    by = defaultdict(lambda: {"n": 0, "equal": 0})
    for r in rows:
        by[r["label"]]["n"] += 1
        by[r["label"]]["equal"] += bool(r["equal"])
    return {"n_items": len(rows), "n_equal": sum(bool(r["equal"]) for r in rows),
            "false_ids": [r["id"] for r in rows if not r["equal"]],
            "false_ids_not_prestated": [r["id"] for r in rows if not r["equal"] and not r.get("prestated_false")],
            "prestated_false_ids_that_were_equal": [r["id"] for r in rows if r["equal"] and r.get("prestated_false")],
            "registered_false_ids": [r["id"] for r in rows if r.get("equal_registered") is False],
            "by_label": dict(by)}


def shape(obj, depth=2) -> object:
    """Key names and leaf TYPES only (0d's types-only allowance): never a value."""
    if isinstance(obj, dict):
        if depth == 0:
            return f"dict[{len(obj)}]"
        return {k: shape(v, depth - 1) for k, v in sorted(obj.items())[:12]}
    if isinstance(obj, list):
        return f"list[{len(obj)}]"
    return type(obj).__name__


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="COUNTSCHECK pre-launch shape check: loads and reduces every input, prints keys and types only")
    ap.add_argument("--shape-only", action="store_true", required=True)
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    ctx = load_all(here.parents[1], here)
    for name in ("ext_rc", "ext_prior", "ext_sp"):
        print(name, json.dumps(shape(ctx[name])))
    print("scan_bits", type(ctx["scan_bits"]).__name__, "usage_bits", type(ctx["usage_bits"]).__name__)
    print("SHAPE_OK all inputs sha-checked; filters", json.dumps(ctx["filter_sha256"]))
