"""Clean-room re-fit of the all-tasks honest-sim detector, route C rerun.  DRAFT: NOT LAUNCHED (a1 reviews and launches).

Written only from registered text (RULE / NOTE / PROPOSAL markdown and committed config / manifest JSON read as data).
No producer code was read. Every source line is cited in PRESTATEMENT_DRAFT.md.

Registered fit recipe (09's ruling, 10-20): 648f33e61 experiments/2026-10-15_cmd_sim_detector/RULE.md L19-27
(L2 logistic regression, C = 0.01, standardized training-fold rows, certified fitter; one row per readout unit labelled
by its run's world; run score = mean row logit; command = content-only call_end minus "0"/"n"; reply = output_end row
means), with the honest detector's RULE @ 596bc767e for problems, pools, folds, bootstrap and catch. Everything else is
SPEC-SILENT: a named default below, fixed before the run, tagged OQ<n> = OPEN_QUESTIONS.md item n.

Scope (a1, 10-20; confirmed by 09's ruling): the printed numbers are the run-level AUROCs of simh_vs_b at reply|L48 and
command|L48 (2 dp). DEFAULT = those 2 cells x simh_vs_b, observed fits + 2,000-draw main bootstrap, NO null.
--full (not in scope unless 0d / Aswin choose it) = 4 cells x 2 problems + 100-draw null + paired bootstrap.

Stages, each its own `modal run`, in this order (STOP = raise; nothing is retried):
  dry     pin check + build the 4,099-run table from committed inputs; STOP unless the registered counts.   (CPU)
  layout  keys-only read of the activation layout: names, types, shapes. Never values.                  (CPU)
  skip    readout row index per run; command rows lose call_end tokens decoding exactly to "0" / "n".      (CPU)
  fit     per cell: 5 folds x (1 observed [+100 null]) certified fits, held-out scores, metrics.            (H100)

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run honest_refit_cleanroom.py --stage dry --repo <worktree>
  ... --stage layout|skip|fit --stamp <stamp printed by dry> [--full]
  addendum 26 (Table 7 row 5): modal run --detach ... --stage fit --stamp 2026-10-20_140924 --problem a_vs_b
    (reuses 4(b)'s run table and row index, sha-pinned in REUSE_PINS; writes only under <stamp>/a_vs_b/)

Values go to the output Volume only. The laptop gets booleans, counts and sha256s.
"""
import hashlib
import io
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import modal

# ---------------- names and hardware ----------------
NAME = "probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16"  # app and output Volume (v2)
ACTS_VOLUME = "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14"  # read-only
OUT, ACTS = "/out", "/acts"
GPU_TIMEOUT, CPU_TIMEOUT = 1200, 1800  # seconds per attempt; retries=0 everywhere (26b: GPU 1200, ~2.5x 4(b)'s 471 s pilot, so the fit bound is under $5)
STARTUP_S, SCALEDOWN_S = 300, 60  # 23 step 2b (09): bounds billed startup and idle per input, so each input <= timeout + 360 s
DRIVER_TIMEOUT = 3300  # 23 step 3b / 26b: the remote fit driver outlives 2 sequential cells (2 x (1200 + 300 + 60) = 3120 s)
H100_RATE = 5.35  # $/h per fit container at list rates: H100 + 8 cores + 128 GiB (23 step 2; pilot projection only)
FIT_BUDGET = 5.0  # $ for the fit cells (pilot projection gate), <= the reduced fit arm cap (~$5.3 after the lost run; 23 step 3b)

# ---------------- registered (648 = 648f33e61 cmd_sim_detector RULE.md; RULE = honest RULE.md @ 596bc767e) ------------
C_REG = 0.01  # 648 L19; RULE L42
GRAD_BAR = 1e-6  # RULE L44 ("certified": max|grad| <= 1e-6); 648 L19 "certified fitter"
ITER_CAP = 5000  # RULE L44
N_FOLDS = 5  # RULE L50, fold map own_all_rs @ ec56720ad (replaces 648 L21's #36 folds: RULE L40 "except inputs")
N_BOOT, BOOT_SEED = 2000, 20261015  # RULE L52
ALPHAS = (0.01, 0.05)  # RULE L53
FIXED_CALL_TOKENS = ("0", "n")  # 648 L24; RULE L42-43
MODEL, MODEL_REV = "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"  # extraction records' config
ALL_CELLS = (("command", 63), ("reply", 48), ("command", 48), ("reply", 63))  # L54
ALL_PROBLEMS = {"simh_vs_b": ("sim", "B"), "a_vs_b": ("A", "B")}  # L14-15: (class 1, class 0)
EXPECT = {"sim": 869, "A": 1516, "B": 1714}  # spec 09 RULE @ 993da3957 L4976
EXPECT_SET = {  # RULE L20-24, sim column updated by route C (09 L4963: S_old 171 + 20 pilot)
    ("S_old", "sim"): 191, ("S_new", "sim"): 305, ("S_newmains", "sim"): 339, ("S_paper493", "sim"): 34,
    ("S_old", "B"): 385, ("S_new", "B"): 609, ("S_newmains", "B"): 686, ("S_paper493", "B"): 34,
    ("S_old", "A"): 192, ("S_new", "A"): 614, ("S_newmains", "A"): 676, ("S_paper493", "A"): 34}

# ---------------- scope (a1, 10-20): config, not code ----------------
DEFAULT_CELLS = (("reply", 48), ("command", 48))
DEFAULT_PROBLEMS = ("simh_vs_b",)
DEFAULT_N_NULL, FULL_N_NULL = 0, 100
# addendum 26 (Table 7 row 5, Aswin 21:23 via 0d / 2e): --problem a_vs_b at --stage fit only, on 4(b)'s stamp; outputs go
# create-only to <stamp>/a_vs_b/; the driver STOPs unless the reused inputs hash to the dry (2064e73e1) / skip (9b5d4af13) records
REUSE_PINS = {"2026-10-20_140924": {"run_table.json": "a006e7bb169bcb53732227dc41cbbf46c80d1be025b2130833d37081f9bd426d",
                                    "rows_index.json": "9e63cf9473227de9d1329b8176812a1a08688806d0f439e21a288ac903fe127b"}}

# ---------------- SPEC-SILENT defaults (09, 10-20: not registered; fixed here before any run) ----------------
CLASS_WEIGHT = None  # SPEC-SILENT OQ1: unweighted rows
SENS_BALANCED = True  # SPEC-SILENT OQ1: balanced-weight observed-only diagnostic beside it; never in the verdict
# SPEC-SILENT OQ2: objective G(w,b) = (C * sum_i logloss_i + 0.5 * ||w||^2) / n  (sklearn's minimizer at C = 0.01;
#   GRAD_BAR is read on this per-n scale; n * max|grad| (raw sklearn scale) is recorded beside it)
PENALIZE_INTERCEPT = False  # SPEC-SILENT OQ3
STD_DDOF, STD_MIN = 0, 1e-12  # SPEC-SILENT OQ4: mean / population std over training-fold rows, float64; std < 1e-12 -> 1
FIT_TARGET = 1e-12  # SPEC-SILENT OQ5: own stop on max|grad G| (float64 truncated Newton + PCG), far below GRAD_BAR
MAX_NEWTON, MAX_PCG = 80, 600  # SPEC-SILENT OQ5
NULL_SEED = 20261014  # SPEC-SILENT OQ8 (only with --full): A1 RULE @ c663d0204 L91, informative not registered
LAYOUT = {  # OQ14, resolved by the layout run (7981dd8e3) + routing_v2 @ c75141175 L53 / L56 / L69 (23 step 2)
    "npy": "{acts}/{root}/by_layer/L{layer}/{chunk}.npy",  # fp16 [rows, 5120]
    "meta_dir": "{acts}/{root}/meta/{arm}",  # one JSON per staged epoch
    "meta_key": "key", "meta_chunk": "chunk", "meta_row0": "row_start", "rows_relative": True,  # L56: row_start + row / row_mean
    "meta_items": "calls", "kind": "kind", "row": "row", "row_mean": "row_mean", "pos": "readout",  # L53: readout = token position
    "span": "span",  # L53: output_end span-mean units also carry calls[].span [start, end)
    "n_tokens": "n_tokens", "tokens_total": "tokens_total", "n_rows": "n_rows",  # L69: n_tokens == tokens_total == len(staged ids)
    "staged": "{acts}/{staged_set}/inputs/{file}", "staged_key": "key", "staged_ids": "ids",
}

# ---------------- pinned committed inputs: name -> (commit, path, sha256) ----------------
PINS = {
    "sources_v2": ("596bc767e9f2bcb97ac187024ff7efd411b2d6f5",
                   "experiments/2026-10-16_honest_sim_detector_all/config/sources_v2.json",
                   "d69ad9c41a091d6485cf8070b094cd62f2b820a74328e2d6540a76ffea8221ac"),
    "inputs_full": ("1a842a85635bd63a8bfc60071195584834117c11",
                    "experiments/2026-10-15_fpr_sweep_507_prep/config/INPUTS_FULL_2026-10-17_233212.json",
                    "1e1868586fe8a786f6e1094b9d5b62e362ec0e2ce241e9b0103d6584af351126"),
    "e12": ("22d7089f5f4cff1fe7bd9578c5f30ca647a4326c",
            "experiments/2026-10-15_fpr_sweep_507_prep/config/E12_BLOCK_2026-10-16_092322.json",
            "683d4de1040f489a768a44fe5ad53f787d35d8ca22117b897a42a75880e1f4dc"),
    "poolkeys": ("5b19c08d8ccf461e7c5d5da4024cc7280e149084",
                 "experiments/2026-10-15_epoch2_poolkeys/results/epoch2_poolkeys_2026-10-15_214512.json",
                 "4c1ef89dd7fdf0a8b21153ede2901f595eb9aefc4879e810de1b1f522cc7eca1"),
    "folds": ("ec56720adde66baabe642a7fc842172c83737679",
              "experiments/2026-10-16_own_all_rs/results/own_all_rs_folds_2026-10-16.json",
              "c2fc5d40eb84a34a64cbb9ce7155572d1414f12ea1fbc2ec05708bd73061e299"),
    "pilot_keys": ("344829af6dc4cd0473757398d542bb8e1f72f512",
                   "experiments/2026-10-14_q3_q35_simhonest_purchase/results/simh_counted_keys_pilot_v3.json",
                   "2149c5cde368aef9e5012c8d9cda5b12fe5a93db3d27ad2d525225bb9f511b27"),
    "inv_sold": ("98575cc3471a224c87401111fbb8caee28ff5323",
                 "experiments/2026-10-15_cut_sensitivity/results/inventory_b_fix_v2_2026-10-17_081456.json",
                 "f8dfc018d44c5a5bd373f88d9f7974dd9774c25a325d3ceceb385fcb9c82719f"),
    "inv_bfix": ("09b070617981342cd31ce95475da369f9c445b9b",
                 "experiments/2026-10-15_cut_sensitivity/results/inventory_b_fix_v1_2026-10-16_025058.json",
                 "4a5d0fb3cad32740f849df5def3c7a03a40a18aa4c42bc360709f0c740973d03"),
    "inv_snew": ("b3f8dea7b49e6d60beae21454fb68cc9b29e5e87",
                 "experiments/2026-10-15_fpr_sweep_507_prep/config/cutfull/inventory_v3_2026-10-15.json",
                 "19306ffa7c42ab35a62715c1a18822a43c6273492e5ede9b2f62cd9e428ad9a3"),
    "inv_w1": ("85447113f648435c8bce85c8f3df6ac8edcf75bb",
               "experiments/2026-10-15_cut_sensitivity/results/inventory_sp493_w1_v1_2026-10-16_025411.json",
               "3dc8a91a52d694c8959db640b1de5688d2a6a657641d8ee7f79d4cff8aa2d067"),
    "inv_w23": ("102b3741ea1ddb46ddb5ddb2e27148de1bde5a85",
                "experiments/2026-10-15_cut_sensitivity/results/inventory_sp493_w23_v1_2026-10-16_033210.json",
                "b11a63f5914c1c5d199c3401104acd9697b6d3e7095802f24e67635895dc98f5"),
    "inv_ep2": ("06077a8a870a02836c490098d1e12de0b46ddad0",
                "experiments/2026-10-15_cut_sensitivity/results/inventory_ep2_v1_2026-10-16_025227.json",
                "963ee204bfbb411f14aa0162a642442532c32cf1ec4783663cda6b228520126f"),
    "inv_fill": ("06077a8a870a02836c490098d1e12de0b46ddad0",
                 "experiments/2026-10-15_cut_sensitivity/results/inventory_fill_v1_2026-10-16_025228.json",
                 "e55927ba25f073369cd8e9ce02530e72097f92fc31f7db7d2779f39d470e9b3a"),
    "inv_ha25f": ("b880c3c3aad708dea25098ba19620e459fd4ad80",
                  "experiments/2026-10-15_cut_sensitivity/results/inventory_ha25f_v2_2026-10-16_044849.json",
                  "c01a6ef18869cd09f1c2d3e441f0deb05143cb2c2f29db84f0d2049a80f8af7f"),
    "cells_old": ("4d1eec57209ee896c7238d019936f88799d5f61c",
                  "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json",
                  "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb"),
    "cells_new": ("0caadbfb5938b1187b9ffce56d7cebc43384116c",
                  "experiments/2026-10-14_q2_scaleup_cohort/results/scaleup_tasks_plus311_2026-10-14_125135.json",
                  "69f1f75313c9610f7d16b18225c1166268c0b6e084859c641d0213f293896a8b"),
    "cells_nm": ("ef171c8139aa6c4581d25e8e1f356b1d9fc37651",
                 "experiments/2026-10-14_q3_new_mains_mode/config/new_mains_cells_v1.json",
                 "98bfb97e26e7ec580b08643a3a38a5e66a0f6b63333f4013aa57559383402e8e"),
    "cells_sp493": ("d800edc6ab4e2744a05105847bec702c8b5d479a",
                    "experiments/2026-10-15_q3_sp493/results/sp493_facts_2026-10-15_165730.json",
                    "07caa527db89ec519314486d8c529c37430ca41d773a8bce26c041eb141b0a57"),
}
INV_OF_SET = {"S_old": "inv_sold", "S_new": "inv_snew", "S_newmains": "inv_bfix", "S_paper493": "inv_w1",
              "S_paper493_w23": "inv_w23", "pools_ep2": "inv_ep2", "pools_fill": "inv_fill", "ha25f": "inv_ha25f"}

app = modal.App(NAME)
out_vol = modal.Volume.from_name(NAME, create_if_missing=True, version=2)
acts_vol = modal.Volume.from_name(ACTS_VOLUME)
cpu_img = modal.Image.debian_slim(python_version="3.12").pip_install(
    "numpy==2.2.4", "scipy==1.15.2", "tokenizers==0.22.1", "huggingface_hub==0.30.2")  # OQ19; tokenizers 0.22.1 (R1: proven on this tokenizer)
gpu_img = cpu_img.pip_install("torch==2.6.0")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class StopRun(RuntimeError):  # 23 step 1 (R1 P2): a RuntimeError, so the message reaches the laptop from a Modal function
    pass


def stop(msg):
    raise StopRun(f"STOP: {msg}")


def sanitized(fn):  # 23 step 1d (Codex P1): nothing but a payload-free StopRun leaves a container (decode errors keep record bytes)
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except StopRun:
            raise
        except Exception as e:  # not swallowed: re-raised as a STOP that names only the exception type, the stage and the code location
            name, where = type(e).__name__, "location unavailable"
            try:  # 23 step 1f / 1g (Codex P1): the innermost frame in this script, from raw traceback metadata (no source loading)
                tb, last, me = e.__traceback__, None, Path(__file__).name
                while tb is not None:
                    co = tb.tb_frame.f_code
                    if Path(co.co_filename).name == me:
                        last = (int(tb.tb_lineno), str(co.co_name))
                    tb = tb.tb_next
                where = f"line {last[0]} in {last[1]}" if last else "no frame in this script"
            except Exception:
                pass
            del e
        raise StopRun(f"STOP: {name} in {fn.__name__} at {where}")  # raised after the except block: no __context__ (as 24d)
    return wrapper


def safe_loads(text, what):  # 23 step 1c (Codex P1): a decode error's .doc holds the whole record; never let it travel
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise StopRun(f"STOP: {what} is not valid JSON") from None


def write_new(path: Path, data: bytes):  # create-only (results are append-only)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as f:
        f.write(data)


def jdump(o) -> bytes:  # NaN -> null, so the file is strict JSON
    def fix(x):
        if isinstance(x, float) and not math.isfinite(x):
            return None
        if isinstance(x, dict):
            return {k: fix(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [fix(v) for v in x]
        return x
    return json.dumps(fix(o), indent=1, sort_keys=True).encode()


# =====================================================================================================
# dry: the run table (CPU)
# =====================================================================================================
def list_sha_forms(keys):  # RULE L28-29: "the sha256 of the sorted keys as JSON"; separators open (OQ12)
    s = sorted(keys)
    return {sha(json.dumps(s).encode()), sha(json.dumps(s, separators=(",", ":")).encode())}


def build_table(J, drop_pilot_key=False):
    F, E, PK, FD = J["inputs_full"], J["e12"]["pools"], J["poolkeys"]["origins"], J["folds"]
    # route C STOP 1 + the pins INPUTS_FULL names (09 L4970)
    for s, role in INV_OF_SET.items():
        if F["inventories"][s]["sha256"] != PINS[role][2]:
            stop(f"INPUTS_FULL inventory {s} != pin {role}")
    if F["e12"]["sha256"] != PINS["e12"][2]:
        stop("INPUTS_FULL e12 != pin")
    if F["s_old_simh_extra_roots"]["simh_q4f_v1_pilot__sim_honest"]["sha256"] != PINS["pilot_keys"][2]:
        stop("INPUTS_FULL pilot entry != pilot file")
    # route C STOP 3: fix_v2 = fix_v1 + exactly the 20 pilot rows, by whole row (09 L4972)
    v1 = {r["key"]: r for r in J["inv_bfix"]["runs"]}
    v2 = {r["key"]: r for r in J["inv_sold"]["runs"]}
    if any(v2.get(k) != r for k, r in v1.items()):
        stop("fix_v1 row missing or changed in fix_v2")
    added = [v2[k] for k in v2.keys() - v1.keys()]
    pilot = J["pilot_keys"]["keys"]
    if (len(added) != 20 or {r["eval_sha256"] for r in added} != {p["eval_sha256"] for p in pilot}
            or {r["task"] for r in added} != {p["task"] for p in pilot}):
        stop("fix_v2 - fix_v1 != the 20 pilot rows")
    drop = pilot[0]["eval_sha256"] if drop_pilot_key else None  # positive control (09 L4977): must STOP at counts
    # cell -> main (cell pin files), cell -> set and main -> fold (fold map @ ec56720ad)
    cell_main = {}
    for role, field in (("cells_old", "cells"), ("cells_new", "cells"), ("cells_nm", "cells"), ("cells_sp493", "rows")):
        for c in J[role][field]:
            if cell_main.setdefault(c["key"], c["main"]) != c["main"]:
                stop(f"cell {c['key']} has two mains")
    cell_set, by_main = FD["cell_set"], FD["by_main"]
    # inventory universe (fix_v1 rows are inside fix_v2, checked above)
    uni = {}
    for role in ("inv_sold", "inv_snew", "inv_ha25f", "inv_fill", "inv_w1", "inv_w23"):
        for r in J[role]["runs"]:
            if drop is not None and r.get("eval_sha256") == drop:
                continue
            if uni.setdefault(r["key"], r) != r:
                stop(f"inventory key {r['key'][:60]} with two different rows")
    universe = list(uni.values())
    by_wt = defaultdict(list)
    for r in universe:
        by_wt[f"{r['world']}|{r['task']}"].append(r)
    ep2 = {r["key"]: r for r in J["inv_ep2"]["runs"]}
    runs = []

    def add(name, cls, row, cell, epoch, source):
        if cell not in cell_main or cell not in cell_set:
            stop(f"cell {cell} has no main or set")
        main = cell_main[cell]
        if main not in by_main:
            stop(f"main {main} has no fold (RULE L50)")
        up = row.get("uncut_planned") or {}
        runs.append(dict(name=name, cls=cls, key=row["key"], cell=cell, set=cell_set[cell], main=main,
                         fold=int(by_main[main][0]), epoch=epoch, source=source, eval_sha256=row.get("eval_sha256"),
                         staged_set=up.get("staged_set"), staged_file=up.get("staged_inputs_file"),
                         staged_sha256=up.get("staged_sha256"), root=up.get("extraction_root")))

    # sim class: every sim_honest inventory row (RULE L22; route C L4963). Never the in_S_old_pool column (09 L4973).
    for r in universe:
        if r["world"] == "sim_honest":
            add(f"sim_honest|{r['task']}", "sim", r, r["task"], 1, "inventory")
    # epoch-1 real pools by pool key "<world>|<cell>" (RULE L27-30)
    def pool(pools, prefix):
        hit = [p for p in pools if p["pool"].startswith(prefix)]
        if len(hit) != 1:
            stop(f"pool {prefix!r}: {len(hit)} matches")
        return hit[0]["epoch1_member_keys"]
    NMP = PK["S_newmains"]["with S4 and without S4 (one calibration pool; S4r lists joined)"]["pools"]
    fam = PK["S_new"]["H_A (25h §3 yardstick, held-out false alarm)"]["families"]
    e1 = {("S_old", "B"): pool(PK["S_old"]["BuC"]["pools"], "B alone"),
          ("S_old", "A"): pool(PK["S_old"]["BuC"]["pools"], "C ("),
          ("S_new", "B"): pool(PK["S_new"]["B"]["pools"], "H_B (B)"),
          ("S_new", "A"): sorted({k for f in fam.values() for k in f["epoch1_member_keys"]}),  # union, no pin (L28)
          ("S_newmains", "B"): pool(NMP, "H_B (honest_b pool)"),
          ("S_newmains", "A"): pool(NMP, "H_A (honest_a owned")}
    pins_e1 = {("S_old", "B"): E["S_old"]["H_B2"], ("S_old", "A"): E["S_old"]["H_A"], ("S_new", "B"): E["S_new"]["H_B"],
               ("S_newmains", "B"): E["S_newmains"]["H_B"], ("S_newmains", "A"): E["S_newmains"]["H_A"]}
    for k, p in pins_e1.items():
        if p["epoch1_members_sha256"] not in list_sha_forms(e1[k]):
            stop(f"epoch-1 list {k} != E12 epoch1_members_sha256 (RULE L27)")
    for (s, c), keys in e1.items():
        for k in keys:
            hits = by_wt.get(k, [])
            if len(hits) != 1:
                stop(f"pool key {k} matches {len(hits)} built runs (RULE L30)")
            add(k, c, hits[0], k.split("|", 1)[1], 1, f"e1:{s}")
    # epoch-2 replicates: E12 lists, by inventory_key into inventory_ep2 (RULE L28-29)
    ha = {m["inventory_key"]: m for f in E["S_new"]["H_A"].values() for m in f["e2"]}
    e2 = {("S_old", "B", "H_B2"): E["S_old"]["H_B2"]["e2"], ("S_new", "B", "H_B"): E["S_new"]["H_B"]["e2"],
          ("S_new", "A", "H_A"): list(ha.values()), ("S_newmains", "B", "H_B"): E["S_newmains"]["H_B"]["e2"],
          ("S_newmains", "A", "H_A"): E["S_newmains"]["H_A"]["e2"]}
    fc = F["freeze_checks"]["e12_pools"]
    for (s, c, p), ms in e2.items():
        want = fc[s][p]["e2_members_sha256"]
        if not any(want in list_sha_forms([m[f] for m in ms]) for f in ("key_e2", "key", "inventory_key")):
            stop(f"epoch-2 list {s}|{p} != INPUTS_FULL e12_pools e2_members_sha256 (RULE L29; OQ12)")
        for m in ms:
            if m["inventory_key"] not in ep2:
                stop(f"e2 member {m['key_e2']} not in inventory_ep2")
            add(m["key_e2"], c, ep2[m["inventory_key"]], m["key"].split("|", 1)[1], 2, f"e2:{s}")
    # S_paper493 real pools: every honest_a / honest_b row (RULE L30)
    for r in J["inv_w1"]["runs"] + J["inv_w23"]["runs"]:
        if r["world"] in ("honest_a", "honest_b"):
            add(f"{r['world']}|{r['task']}", "A" if r["world"] == "honest_a" else "B", r, r["task"], 1, "sp493")
    # gates
    for f in ("name", "key"):
        if len({r[f] for r in runs}) != len(runs):
            stop(f"duplicate run {f}")
    got = {c: sum(r["cls"] == c for r in runs) for c in EXPECT}
    got_set = {k: sum((r["set"], r["cls"]) == k for r in runs) for k in EXPECT_SET}
    pilot_in = {p["eval_sha256"] for p in pilot} <= {r["eval_sha256"] for r in runs if r["cls"] == "sim" and r["set"] == "S_old"}
    if got != EXPECT or got_set != EXPECT_SET or not pilot_in:
        stop(f"counts {got} / per-set ok={got_set == EXPECT_SET} / pilot keys present={pilot_in} (09 L4973, L4976)")
    return runs


def resolve_roots(runs, side):
    """Extraction root per run: the inventory's uncut_planned.extraction_root, else the highest-revision pinned routing
    record for the staged set (OQ13). Chunk lists per root from the pinned extraction records (sources_v2)."""
    routes = defaultdict(list)
    for doc in side["routing"]:
        if {"staged_set", "extraction_root", "revision"} <= doc.keys() and doc.get("cut") == "full":
            routes[doc["staged_set"]].append(doc)
    n_from_routing = 0
    for r in runs:
        if r["root"] is None:
            cands = routes.get(r["staged_set"], [])
            top = max((d["revision"] for d in cands), default=None)
            best = {d["extraction_root"] for d in cands if d["revision"] == top}
            if len(best) != 1:
                stop(f"no unique routing for staged set {r['staged_set']}")
            r["root"] = best.pop()
            n_from_routing += 1
        if (r["staged_set"] is None) or (r["staged_file"] is None):
            stop(f"run {r['name']} has no staged set / file")
    chunks, partial, n_err = defaultdict(dict), set(), 0
    for doc in side["extraction"]:
        root = doc["config"]["run"]
        for sh in doc["shards"]:
            if "chunks" not in sh:  # 23 step 1h (R2): a failed shard is exactly {"error"}; its root's chunk list is then incomplete
                if set(sh) != {"error"}:
                    stop(f"extraction shard without chunks has keys other than error ({root})")
                partial.add(root)
                n_err += 1
                continue
            for c in sh["chunks"]:
                if chunks[root].setdefault(c["chunk"], c["rows"]) != c["rows"]:
                    stop(f"chunk {root}/{c['chunk']} listed with two row counts")
    need = {r["root"] for r in runs}
    if need & partial:  # a root the table reads must have a complete listing
        stop(f"{len(need & partial)} needed extraction roots have a failed shard")
    missing = sorted(need - chunks.keys())
    if missing:
        stop(f"{len(missing)} extraction roots without a pinned extraction record")
    return {root: chunks[root] for root in need}, n_from_routing, {"n_error_shards": n_err, "n_partial_roots": len(partial)}


@app.function(image=cpu_img, cpu=(4, 4), memory=(16384, 16384), timeout=CPU_TIMEOUT, retries=0,
              startup_timeout=STARTUP_S, scaledown_window=SCALEDOWN_S, volumes={OUT: out_vol})
@sanitized
def dry(stamp: str) -> dict:
    d = Path(OUT, stamp)
    out_vol.reload()
    index = json.loads((d / "inputs/index.json").read_text())
    J, side = {}, {"routing": [], "extraction": []}
    for name, e in index.items():
        b = (d / "inputs" / e["sha256"]).read_bytes()
        if sha(b) != e["sha256"]:
            stop(f"in-container sha {name}")
        doc = json.loads(b)
        if name in PINS:
            J[name] = doc
        else:
            side[e["role"]].append(doc)
    try_ctrl = False
    try:  # positive control (09 L4977): one pilot key removed must STOP at the count gate. Narrow: StopRun only.
        build_table(J, drop_pilot_key=True)
    except StopRun as e:
        try_ctrl = "counts" in str(e)
    if not try_ctrl:
        stop("positive control did not STOP at the count gate")
    runs = build_table(J)
    chunk_lists, n_routing, errs = resolve_roots(runs, side)
    table = {"stamp": stamp, "runs": runs, "chunk_lists": chunk_lists, "pins": index,
             "script_sha256": sha(Path(__file__).read_bytes())}
    blob = jdump(table)
    write_new(d / "run_table.json", blob)
    out_vol.commit()
    return {"stage": "dry", "ok": True, "n_runs": len(runs), "by_class": {c: sum(r["cls"] == c for r in runs) for c in EXPECT},
            "n_mains": len({r["main"] for r in runs}), "n_roots": len(chunk_lists), "roots_from_routing": n_routing,
            "positive_control_stopped": try_ctrl, "run_table_sha256": sha(blob), **errs}


# =====================================================================================================
# layout: keys only (CPU)
# =====================================================================================================
@app.function(image=cpu_img, cpu=(1, 1), memory=(4096, 4096), timeout=600, retries=0,
              startup_timeout=STARTUP_S, scaledown_window=SCALEDOWN_S, volumes={OUT: out_vol, ACTS: acts_vol.read_only()})
@sanitized
def layout(stamp: str) -> dict:
    """Names, types, lengths, dtypes and shapes only. No value, file name with a run key, or text is returned."""
    import numpy as np

    def shape(o, depth=0):
        if isinstance(o, dict):
            return {k: shape(v, depth + 1) for k, v in list(o.items())[:40]} if depth < 4 else "dict"
        if isinstance(o, list):
            return [f"list[{len(o)}]", shape(o[0], depth + 1) if o else None]
        return type(o).__name__

    runs = json.loads(Path(OUT, stamp, "run_table.json").read_text())["runs"]
    out = {}
    for cls in ("sim", "A", "B"):
        r = next(x for x in runs if x["cls"] == cls)
        root, arm = Path(ACTS, r["root"]), r["staged_file"].rsplit(".jsonl", 1)[0]
        info = {"root_exists": root.exists(),  # 23 step 1 (R1 / R2 P2): a missing root is reported, not raised; listing capped
                "subdirs": sorted(p.name for p in root.iterdir() if p.is_dir())[:40] if root.exists() else None}
        for L in (48, 63):  # 23 step 1 (R2 P2): the in-scope cells read L48
            lay = root / "by_layer" / f"L{L}"
            npys = sorted(lay.glob("*.npy")) if lay.exists() else []
            info[f"n_L{L}_npy"] = len(npys)
            if npys:
                a = np.load(npys[0], mmap_mode="r")
                info[f"npy_L{L}_dtype_shape"] = [str(a.dtype), list(a.shape)]
        cj = sorted((root / "chunks").glob("*.json")) if (root / "chunks").exists() else []
        info["chunk_json_schema"] = shape(safe_loads(cj[0].read_text(), "chunk json")) if cj else None
        md = Path(LAYOUT["meta_dir"].format(acts=ACTS, root=r["root"], arm=arm))
        mf = sorted(md.glob("*.json")) if md.exists() else []
        info["meta_dir_exists"], info["n_meta"] = md.exists(), len(mf)
        info["meta_schema"] = shape(safe_loads(mf[0].read_text(), "meta json")) if mf else None
        st = Path(LAYOUT["staged"].format(acts=ACTS, staged_set=r["staged_set"], file=r["staged_file"]))
        info["staged_exists"] = st.exists()
        if st.exists():
            with open(st) as f:
                info["staged_line_schema"] = shape(safe_loads(f.readline(), "staged line"))
        out[cls] = info
    return {"stage": "layout", "layout": out}


# =====================================================================================================
# skip: readout row index per run (CPU)
# =====================================================================================================
def fixed_token_ids():
    """23 step 2 (c945fdd94 L802 / L808): a call_end is dropped iff its readout token, decoded ALONE, equals "0" or "n" exactly.
    Decode = tokenizers Tokenizer.decode([id], skip_special_tokens=False) with the tokenizer.json's own decoder; no further
    clean-up. Returns the set of such vocabulary ids (a tokenizer property, reported as a count)."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(hf_hub_download(MODEL, "tokenizer.json", revision=MODEL_REV))
    dec = {i: tok.decode([i], skip_special_tokens=False) for i in range(tok.get_vocab_size())}
    ids = {i for i, t in dec.items() if t in FIXED_CALL_TOKENS}
    if {dec[i] for i in ids} != set(FIXED_CALL_TOKENS):  # 23 step 2c (R1 P2): both "0" and "n" must exist as lone tokens
        stop("fixed call tokens: not every fixed string decodes from a lone vocabulary id")
    return ids


@app.function(image=cpu_img, cpu=(16, 16), memory=(65536, 65536), timeout=CPU_TIMEOUT, retries=0,
              startup_timeout=STARTUP_S, scaledown_window=SCALEDOWN_S,
              volumes={OUT: out_vol, ACTS: acts_vol.read_only()})
@sanitized
def skip(stamp: str) -> dict:
    from concurrent.futures import ThreadPoolExecutor
    d = Path(OUT, stamp)
    out_vol.reload()
    T = json.loads((d / "run_table.json").read_text())
    runs, chunk_lists, L = T["runs"], T["chunk_lists"], LAYOUT
    fixed = fixed_token_ids()
    groups = defaultdict(list)  # (root, arm) -> runs
    for r in runs:
        groups[(r["root"], r["staged_file"].rsplit(".jsonl", 1)[0])].append(r)

    def metas(key):
        root, arm = key
        idx = {}
        for p in Path(L["meta_dir"].format(acts=ACTS, root=root, arm=arm)).glob("*.json"):
            m = safe_loads(p.read_text(), "meta json")
            if idx.setdefault(m[L["meta_key"]], m) is not m:
                stop(f"two metas with one key in {root}/{arm}")
        return key, idx

    with ThreadPoolExecutor(32) as ex:
        meta_idx = dict(ex.map(metas, sorted(groups)))
    found = {}
    for key, rs in groups.items():
        for r in rs:  # OQ11: meta key == inventory key, or inventory key + "#e0"; exactly one
            hits = [k for k in (r["key"], r["key"] + "#e0") if k in meta_idx[key]]
            if len(hits) != 1:
                stop(f"run {r['name']}: {len(hits)} metas")
            found[r["name"]] = (hits[0], meta_idx[key][hits[0]])
    need = {"kind", L["row"], L["pos"]}  # 23 step 2 (09): a MISSING key STOPs (null is allowed and tested per kind below)
    n = defaultdict(int)
    for c in EXPECT:  # 23 step 2c (Codex P2): every registered count key is reported, zeros included
        for k in ("call_end", "fixed", "output_end", "reply_rows", "excluded_call_end", "output_end_no_span_row_null", "output_end_no_span_row_set", "other_kind",
                  "rows_cmd", "rows_rep", "zero_cmd", "zero_rep"):
            n[f"{k}|{c}"] = 0
    units = {}  # run name -> (row0, chunk, [(row, readout)] call_end units, [row_mean] output_end units, [readout] excluded call_ends)
    for r in runs:
        mkey, m = found[r["name"]]
        for f in (L["meta_chunk"], L["meta_row0"], L["meta_items"], L["n_tokens"], L["tokens_total"], L["n_rows"]):
            if f not in m:
                stop(f"run {r['name']}: meta has no {f}")
        nrows = m[L["n_rows"]]

        def in_epoch(x):  # 23 step 2c (R2 P2): an epoch-relative row must be an int in [0, meta.n_rows) (L56)
            return type(x) is int and 0 <= x < nrows
        ce, oe, xr = [], [], []
        for it in m[L["meta_items"]]:
            if not need <= it.keys():
                stop(f"run {r['name']}: a call item lacks a required key")
            if it[L["kind"]] == "call_end":  # L53: a call_end is a unit iff row is non-null
                if it[L["row"]] is None:
                    n[f"excluded_call_end|{r['cls']}"] += 1
                    if it[L["pos"]] is not None:  # 23 step 2c (Codex P1): an excluded call_end's readout is range-checked too
                        xr.append(it[L["pos"]])
                    continue
                if not in_epoch(it[L["row"]]):
                    stop(f"run {r['name']}: call_end row outside [0, n_rows)")
                ce.append((it[L["row"]], it[L["pos"]]))
            elif it[L["kind"]] == "output_end":  # 23 step 2d: reply = output_end readouts WITH A SPAN (9496c18e3 L77; L53 / L56)
                rm, sp = it.get(L["row_mean"]), it.get(L["span"])
                if rm is None:  # no span mean: not a reply row; counted by whether its last-token row is set
                    if sp is not None:
                        stop(f"run {r['name']}: output_end with a span but no row_mean")
                    n[f"output_end_no_span_row_{'null' if it[L['row']] is None else 'set'}|{r['cls']}"] += 1
                    continue
                if not (isinstance(sp, list) and len(sp) == 2 and all(type(v) is int for v in sp) and sp[0] < sp[1]
                        and sp[1] - 1 == it[L["pos"]]):  # 09 (2d): L53 span [start, end), start < end, end - 1 == readout
                    stop(f"run {r['name']}: output_end span-mean unit with a bad or missing span")
                if not in_epoch(it[L["row_mean"]]):
                    stop(f"run {r['name']}: output_end row_mean outside [0, n_rows)")
                oe.append(it[L["row_mean"]])
            else:  # 23 step 2c (R2 P2): other kinds are counted, not silently skipped
                n[f"other_kind|{r['cls']}"] += 1
        units[r["name"]] = (m[L["meta_row0"]] if L["rows_relative"] else 0, m[L["meta_chunk"]], ce, oe, xr)
    # staged ids, streamed once per staged file: L69 length equality, readout range, and the call_end token ids
    by_file = defaultdict(dict)  # (set, file, sha) -> meta key -> run name
    for r in runs:
        by_file[(r["staged_set"], r["staged_file"], r["staged_sha256"])][found[r["name"]][0]] = r["name"]
    # 23 step 2c (R2 P1): every staged file is checked. Expected sha = the inventory's when non-null, else the pinned routing
    # records' arms[].staged_sha for "<staged_set>/inputs/<file>" (routing_v2 @ c75141175 record_fields; c945fdd94 L807 "checks
    # each staged file against the sha256 the extraction recorded"). Routing entries must agree with each other and with the inventory.
    rsha = defaultdict(set)
    for e in T["pins"].values():
        if e.get("role") != "routing":
            continue
        doc = safe_loads((d / "inputs" / e["sha256"]).read_text(), "routing doc")
        if isinstance(doc, dict) and {"staged_set", "extraction_root", "revision"} <= doc.keys():
            for a in doc.get("arms") or []:
                rsha[a["staged_inputs_file"]].add(a["staged_sha"])
    want_sha, n_sha = {}, defaultdict(int)
    for (sset, sfile, ssha) in by_file:
        rs = rsha.get(f"{sset}/inputs/{sfile}", set())
        if len(rs) > 1:
            stop(f"routing records disagree on the sha of {sset}/{sfile}")
        if ssha and rs and ssha not in rs:
            stop(f"inventory and routing disagree on the sha of {sset}/{sfile}")
        if not (ssha or rs):
            stop(f"no recorded sha for {sset}/{sfile}")
        want_sha[(sset, sfile, ssha)] = ssha or next(iter(rs))
        n_sha["files_sha_from_inventory" if ssha else "files_sha_from_routing"] += 1

    def scan(item):
        (sset, sfile, ssha), want = item
        p = Path(L["staged"].format(acts=ACTS, staged_set=sset, file=sfile))
        h, got = hashlib.sha256(), {}
        with open(p, "rb") as f:
            for line in f:
                h.update(line)
                o = safe_loads(line, "staged line")
                name = want.get(o.get(L["staged_key"]))
                if name is None:
                    continue
                if name in got:
                    stop(f"run {name}: two staged lines")
                ids = o[L["staged_ids"]]
                m = found[name][1]
                if not (m[L["n_tokens"]] == m[L["tokens_total"]] == len(ids)):  # L69
                    stop(f"run {name}: n_tokens / tokens_total / len(ids) differ")
                toks = []
                for _, pos in units[name][2]:
                    if not (type(pos) is int and 0 <= pos < len(ids)):  # 23 step 2c (Codex P1): bool is not an int here
                        stop(f"run {name}: call_end readout outside [0, len(ids))")
                    toks.append(ids[pos])
                for pos in units[name][4]:
                    if not (type(pos) is int and 0 <= pos < len(ids)):
                        stop(f"run {name}: excluded call_end readout outside [0, len(ids))")
                got[name] = toks
        if h.hexdigest() != want_sha[(sset, sfile, ssha)]:  # 23 step 2c: every staged file, S_new included
            stop(f"staged file sha {sset}/{sfile}")
        if set(got) != set(want.values()):
            stop(f"{len(set(want.values()) - set(got))} runs without a staged line in {sset}/{sfile}")
        return got

    tok_ids = {}
    with ThreadPoolExecutor(16) as ex:
        for got in ex.map(scan, sorted(by_file.items())):
            tok_ids.update(got)
    index = {}
    for r in runs:
        row0, chunk, ce, oe, _ = units[r["name"]]
        if chunk not in chunk_lists[r["root"]]:
            stop(f"chunk {r['root']}/{chunk} not listed by its extraction record")
        cmd = []
        if len(ce) != len(tok_ids[r["name"]]):  # 23 step 2c (R2 P2): zip must not truncate
            stop(f"run {r['name']}: call_end units and token ids differ in number")
        for (row, _), tid in zip(ce, tok_ids[r["name"]]):
            n[f"call_end|{r['cls']}"] += 1
            if tid in fixed:
                n[f"fixed|{r['cls']}"] += 1  # 09: registered output key, compared by 09 to the producer's count after commit
            else:
                cmd.append(row0 + row)
        rep = [row0 + rm for rm in oe]
        n[f"output_end|{r['cls']}"] += len(oe)
        n[f"reply_rows|{r['cls']}"] += len(rep)  # 09 (2d): registered key, compared by 09 to the producer after commit
        if max(cmd + rep, default=-1) >= chunk_lists[r["root"]][chunk]:
            stop(f"row index past chunk rows for {r['name']}")
        n[f"zero_cmd|{r['cls']}"] += not cmd
        n[f"zero_rep|{r['cls']}"] += not rep
        n[f"rows_cmd|{r['cls']}"] += len(cmd)
        n[f"rows_rep|{r['cls']}"] += len(rep)
        index[r["name"]] = {"root": r["root"], "chunk": chunk, "command": cmd, "reply": rep}
    blob = jdump({"index": index, "fixed_token_ids": sorted(fixed), "counts": dict(n)})
    write_new(d / "rows_index.json", blob)
    out_vol.commit()
    return {"stage": "skip", "n_fixed_token_ids": len(fixed), "counts": dict(n), "staged_sha_sources": dict(n_sha),
            "rows_index_sha256": sha(blob)}


# =====================================================================================================
# fit (H100)
# =====================================================================================================
def fit_columns(Z, Y, omega=None):
    """Minimize G_k(w, b) = (C * sum_i omega_i * logloss(y_ik, z_i.w_k + b_k) + 0.5 * ||w_k||^2) / n for every column k
    of Y (n x K) at once; intercept unpenalized. float64 truncated Newton: PCG on the exact Hessian, preconditioned by
    (C * mean(s) * Z'Z + I) / n in Z'Z's eigenbasis (one eigh per fold, shared by all columns), Armijo backtracking."""
    import torch
    import torch.nn.functional as Fn
    n, d = Z.shape
    om = torch.ones_like(Y[:, :1]) if omega is None else omega
    lam, U = torch.linalg.eigh(Z.T @ Z)
    lam = lam.clamp_min(0)[:, None]
    yb = (om * Y).sum(0) / om.sum()
    W, b = Z.new_zeros(d, Y.shape[1]), torch.log(yb / (1 - yb))

    def obj(W, b):
        eta = Z @ W + b
        return (C_REG * (om * (Fn.softplus(eta) - Y * eta)).sum(0) + 0.5 * (W * W).sum(0)) / n, eta

    def grad(W, eta):
        p = torch.sigmoid(eta)
        r = om * (p - Y)
        return (C_REG * (Z.T @ r) + W) / n, C_REG * r.sum(0) / n, p

    G, eta = obj(W, b)
    gW, gb, p = grad(W, eta)
    iters = torch.zeros(Y.shape[1], dtype=torch.long, device=Z.device)
    n_pcg = 0
    for _ in range(MAX_NEWTON):
        gmax = torch.maximum(gW.abs().amax(0), gb.abs())
        act = gmax > FIT_TARGET
        if not act.any():
            break
        iters += act.long()
        s = om * p * (1 - p)
        sbar = s.mean(0)
        rhsW, rhsb = -gW * act, -gb * act

        def Hv(vW, vb):
            u = s * (Z @ vW + vb)
            return (C_REG * (Z.T @ u) + vW) / n, C_REG * u.sum(0) / n

        def Minv(rW, rb):  # M_w = (C*sbar*Z'Z + I)/n with Z'Z = U diag(lam) U';  M_b = C*sbar
            return n * (U @ ((U.T @ rW) / (C_REG * sbar * lam + 1))), rb / (C_REG * sbar)

        tol = gmax.sqrt().clamp(1e-10, 0.1)
        xW, xb = torch.zeros_like(rhsW), torch.zeros_like(rhsb)
        rW, rb = rhsW.clone(), rhsb.clone()
        zW, zb = Minv(rW, rb)
        pW, pb = zW.clone(), zb.clone()
        rz = (rW * zW).sum(0) + rb * zb
        bn = torch.sqrt((rhsW ** 2).sum(0) + rhsb ** 2)
        for _ in range(MAX_PCG):
            qW, qb = Hv(pW, pb)
            den = (pW * qW).sum(0) + pb * qb
            a = torch.where(den > 0, rz / den.clamp_min(1e-300), torch.zeros_like(den))
            xW, xb, rW, rb = xW + a * pW, xb + a * pb, rW - a * qW, rb - a * qb
            n_pcg += 1
            if ((torch.sqrt((rW ** 2).sum(0) + rb ** 2) <= tol * bn) | (bn == 0)).all():
                break
            zW, zb = Minv(rW, rb)
            rz_new = (rW * zW).sum(0) + rb * zb
            beta = torch.where(rz > 0, rz_new / rz.clamp_min(1e-300), torch.zeros_like(rz))
            rz, pW, pb = rz_new, zW + beta * pW, zb + beta * pb
        slope = (gW * xW).sum(0) + gb * xb
        t = act.double()
        for _ in range(60):
            Gn, eta_n = obj(W + t * xW, b + t * xb)
            ok = (Gn <= G + 1e-4 * t * slope) | (t == 0)
            if ok.all():
                break
            t = torch.where(ok, t, t / 2)
        W, b, G, eta = W + t * xW, b + t * xb, Gn, eta_n
        gW, gb, p = grad(W, eta)
    gmax = torch.maximum(gW.abs().amax(0), gb.abs())
    finite = torch.isfinite(W).all(0) & torch.isfinite(b) & torch.isfinite(gmax)
    cert = {"max_abs_grad": gmax.tolist(), "max_abs_grad_raw_sklearn": (gmax * n).tolist(),
            "newton_iters": iters.tolist(), "pcg_matvecs_total": n_pcg,
            "certified": ((gmax <= GRAD_BAR) & (iters <= ITER_CAP) & finite).tolist()}
    return W, b, cert


def auroc(s, y):  # pooled, ties count half (A1 RULE @ c663d0204 L84)
    from scipy.stats import rankdata
    y = y.astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def wauroc(s, y, mult, unit_main, chunk=200):
    """Weighted AUROC per bootstrap draw: unit weight = its main's multiplicity (mult: draws x mains). Ties count half."""
    import numpy as np
    import torch
    o = np.argsort(s, kind="mergesort")
    _, gid = np.unique(s[o], return_inverse=True)
    gid = torch.as_tensor(gid, device="cuda")
    y1 = torch.as_tensor(y[o], device="cuda", dtype=torch.float64)
    um = unit_main[o]
    G, res = int(gid.max()) + 1, []
    for i in range(0, mult.shape[0], chunk):
        w = torch.as_tensor(mult[i:i + chunk][:, um], device="cuda", dtype=torch.float64)
        w0 = torch.zeros(w.shape[0], G, device="cuda", dtype=torch.float64).index_add_(1, gid, w * (1 - y1))
        w1 = torch.zeros_like(w0).index_add_(1, gid, w * y1)
        below = torch.cumsum(w0, 1) - w0
        den = w1.sum(1) * w0.sum(1)
        res.append(torch.where(den > 0, (w1 * (below + 0.5 * w0)).sum(1) / den, torch.nan))
    return torch.cat(res).cpu().numpy()


def catch(s, y, fold, a, m=None):
    """Threshold per fold on held-out class-0 units: the unit at descending rank floor(a*n0 + 1e-9) (0-based; units
    repeated by multiplicity m); a class-1 unit is caught if strictly above it (A1 RULE @ c663d0204 L86-88)."""
    import numpy as np
    m = np.ones(len(s), int) if m is None else m
    hits = n1 = fp = n0 = 0
    for f in range(N_FOLDS):
        neg, pos = (fold == f) & ~y, (fold == f) & y
        o = np.argsort(-s[neg], kind="mergesort")
        sn, mn = s[neg][o], m[neg][o]
        k = math.floor(a * mn.sum() + 1e-9)
        if k >= mn.sum():
            return float("nan"), 0, 0, float("nan")
        tau = sn[np.searchsorted(np.cumsum(mn), k, side="right")]
        hits += int(m[pos][s[pos] > tau].sum())
        n1 += int(m[pos].sum())
        fp += int(mn[sn > tau].sum())
        n0 += int(mn.sum())
    return (hits / n1 if n1 else float("nan")), hits, n1, fp / n0


def null_labels(names, mains, y, n_null):  # OQ8: draw-outer, mains sorted, runs in sorted name order
    import numpy as np
    rng = np.random.default_rng(NULL_SEED)
    by_main = defaultdict(list)
    for i in np.argsort(np.array(names), kind="mergesort"):
        by_main[mains[i]].append(i)
    Y = np.repeat(y[:, None], n_null + 1, 1)
    for k in range(1, n_null + 1):
        for mm in sorted(by_main):
            idx = np.array(by_main[mm])
            Y[idx, k] = rng.permutation(y[idx])
    return Y


def boot_mult(mains_sorted):  # OQ9: default_rng(20261015).integers(0, M, M) per draw; multiplicities draws x M
    import numpy as np
    rng = np.random.default_rng(BOOT_SEED)
    M = len(mains_sorted)
    return np.stack([np.bincount(rng.integers(0, M, M), minlength=M) for _ in range(N_BOOT)])


def ci(x):
    import numpy as np
    ok = np.isfinite(x)
    return {"lo": float(np.percentile(x[ok], 2.5)), "hi": float(np.percentile(x[ok], 97.5)),
            "n_noncomputable": int((~ok).sum())}


def load_rows(runs, index, chunk_lists, readout, layer):
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    need = defaultdict(list)
    for i, r in enumerate(runs):
        e = index[r["name"]]
        if e[readout]:
            need[(e["root"], e["chunk"])].append((i, e[readout]))

    def read(key):
        root, chunk = key
        a = np.load(LAYOUT["npy"].format(acts=ACTS, root=root, layer=layer, chunk=chunk), mmap_mode="r")
        if a.shape[0] != chunk_lists[root][chunk] or a.ndim != 2:
            stop(f"{root}/{chunk} L{layer}: shape {a.shape} vs listed rows {chunk_lists[root][chunk]}")
        return [(i, np.asarray(a[np.asarray(rows)], dtype=np.float16)) for i, rows in need[key]]

    with ThreadPoolExecutor(32) as ex:
        parts = [p for res in ex.map(read, sorted(need)) for p in res]
    X = np.concatenate([p for _, p in parts])
    row_run = np.concatenate([np.full(len(p), i) for i, p in parts])
    if not np.isfinite(X).all():
        stop("non-finite activations")
    return X, row_run, len(need)


def fit_problem(Xg, row_run, runs, c1, c0, n_null, extra):
    """One problem in one cell: 5 folds x (1 + n_null) columns, held-out row logits, run score = mean row logit."""
    import numpy as np
    import torch
    has = np.bincount(row_run, minlength=len(runs)) > 0
    sel = np.array([i for i, r in enumerate(runs) if r["cls"] in (c1, c0) and has[i]])
    zero = sorted(r["name"] for i, r in enumerate(runs) if r["cls"] in (c1, c0) and not has[i])
    y = np.array([runs[i]["cls"] == c1 for i in sel])
    fold = np.array([runs[i]["fold"] for i in sel])
    mains = [runs[i]["main"] for i in sel]
    Ylab = null_labels([runs[i]["name"] for i in sel], mains, y, n_null)
    pos = -np.ones(len(runs), int)
    pos[sel] = np.arange(len(sel))
    rows = np.nonzero(pos[row_run] >= 0)[0]
    rrun = pos[row_run[rows]]
    rfold = fold[rrun]
    K = Ylab.shape[1]
    row_logit = np.full((len(rows), K), np.nan, np.float32)
    certs, weights, sens_logit = [], {}, np.full(len(rows), np.nan)
    for f in range(N_FOLDS):
        tr, te = rows[rfold != f], rows[rfold == f]
        Z = Xg[torch.as_tensor(tr, device="cuda")].double()
        mu, sd = Z.mean(0), Z.std(0, correction=STD_DDOF)
        sd = torch.where(sd < STD_MIN, torch.ones_like(sd), sd)
        Z.sub_(mu).div_(sd)
        Yt = torch.as_tensor(Ylab[rrun[rfold != f]], device="cuda", dtype=torch.float64)
        W, b, cert = fit_columns(Z, Yt)
        cert["fold"] = f
        certs.append(cert)
        Zte = (Xg[torch.as_tensor(te, device="cuda")].double() - mu) / sd
        row_logit[rfold == f] = (Zte @ W + b).float().cpu().numpy()
        weights[f"fold{f}"] = {"coef": W[:, 0].cpu().numpy(), "intercept": float(b[0]), "mean": mu.cpu().numpy(),
                               "std": sd.cpu().numpy()}
        if SENS_BALANCED:  # OQ1 diagnostic: balanced class weights, observed labels only
            yt = Yt[:, :1]
            om = torch.where(yt > 0, len(yt) / (2 * yt.sum()), len(yt) / (2 * (len(yt) - yt.sum())))
            Wb, bb, cb = fit_columns(Z, yt, om)
            sens_logit[rfold == f] = (Zte @ Wb + bb)[:, 0].cpu().numpy()
            certs.append({"fold": f, "variant": "balanced", **cb})
        del Z, Zte
        torch.cuda.empty_cache()
    cnt = np.bincount(rrun, minlength=len(sel))
    run_score = np.stack([np.bincount(rrun, weights=row_logit[:, k], minlength=len(sel)) / cnt for k in range(K)], 1)
    sens_run = np.bincount(rrun, weights=sens_logit, minlength=len(sel)) / cnt
    ry = y[rrun]
    obs = {"run_level": {"auroc": auroc(run_score[:, 0], y),
                         "auroc_per_fold": [auroc(run_score[fold == f, 0], y[fold == f]) for f in range(N_FOLDS)]},
           "per_action": {"auroc": auroc(row_logit[:, 0], ry)}}
    for a in ALPHAS:
        rate, h, n1, fpr = catch(run_score[:, 0], y, fold, a)
        obs["run_level"][f"catch_{a}"] = {"rate": rate, "hits": h, "n1": n1, "realized_fpr": fpr}
        rate, h, n1, fpr = catch(row_logit[:, 0].astype(float), ry, rfold, a)
        obs["per_action"][f"catch_{a}"] = {"rate": rate, "hits": h, "n1": n1, "realized_fpr": fpr}
    # bootstrap over main tasks, held-out scores fixed (RULE L52; spec 09 L4812 label)
    ms = sorted(set(mains))
    mi = np.array([ms.index(m) for m in mains])
    mult = boot_mult(ms)
    boot = {"run_auroc": ci(wauroc(run_score[:, 0], y, mult, mi)),
            "action_auroc": ci(wauroc(row_logit[:, 0].astype(float), ry, mult, mi[rrun])),
            "label": "bootstrap over main tasks; cross-validated scores held fixed (no refit)"}
    for a in ALPHAS:
        boot[f"run_catch_{a}"] = ci(np.array([catch(run_score[:, 0], y, fold, a, mult[j, mi])[0] for j in range(N_BOOT)]))
    out = {"n_runs": {c1: int(y.sum()), c0: int((~y).sum())}, "n_rows": int(len(rows)), "zero_row_runs": zero,
           "n_mains": len(ms), "n_single_class_mains": int(sum(len(set(y[mi == j])) == 1 for j in range(len(ms)))),
           "observed": obs, "bootstrap": boot, "certificates": certs, "null_draws": n_null,
           "perm_matrix_sha256": sha(np.ascontiguousarray(Ylab[:, 1:]).tobytes()) if n_null else None,
           "sensitivity_balanced_observed_only": {"label": "pre-stated diagnostic (OQ1); not in the verdict",
                                                  "run_auroc": auroc(sens_run, y), "action_auroc": auroc(sens_logit, ry)}}
    if n_null:
        nr = np.array([auroc(run_score[:, k], Ylab[:, k]) for k in range(1, K)])
        na = np.array([auroc(row_logit[:, k], Ylab[rrun, k]) for k in range(1, K)])
        out["null"] = {"run_auroc": {"p95": float(np.quantile(nr, 0.95)), "max": float(nr.max()),
                                     "p_addone": float((1 + (nr >= obs["run_level"]["auroc"]).sum()) / (1 + len(nr)))},
                       "action_auroc": {"p95": float(np.quantile(na, 0.95)), "max": float(na.max()),
                                        "p_addone": float((1 + (na >= obs["per_action"]["auroc"]).sum()) / (1 + len(na)))}}
        for a in ALPHAS:
            nc = np.array([catch(run_score[:, k], Ylab[:, k], fold, a)[0] for k in range(1, K)])
            out["null"][f"run_catch_{a}"] = {"p95": float(np.quantile(nc, 0.95)), "max": float(nc.max()),
                                             "p_addone": float((1 + (nc >= obs["run_level"][f"catch_{a}"]["rate"]).sum()) / (1 + len(nc)))}
    extra.update(sel=sel, y=y, mains=mains, run_score=run_score[:, 0], weights=weights, row_logit=row_logit[:, 0],
                 null_run_scores=run_score[:, 1:] if n_null else None)
    return out


@app.function(image=gpu_img, gpu="H100", cpu=(8, 8), memory=(131072, 131072), timeout=GPU_TIMEOUT, retries=0,
              startup_timeout=STARTUP_S, scaledown_window=SCALEDOWN_S,
              volumes={OUT: out_vol, ACTS: acts_vol.read_only()})
@sanitized
def fit_cell(stamp: str, readout: str, layer: int, problems: list, n_null: int, sub: str = "") -> dict:
    import numpy as np
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    t0, d = time.time(), Path(OUT, stamp)
    out_vol.reload()
    dest = d / sub / f"cell_{readout}_L{layer}.json"  # addendum 26: sub = "" (4(b)) or the problem's subfolder
    if dest.exists():
        stop(f"create-only: {dest.name} exists")
    T = json.loads((d / "run_table.json").read_text())
    idx = json.loads((d / "rows_index.json").read_text())["index"]
    runs = T["runs"]
    X, row_run, n_chunks = load_rows(runs, idx, T["chunk_lists"], readout, layer)
    Xg = torch.as_tensor(X, device="cuda")
    del X
    res = {"cell": f"{readout}|L{layer}", "n_rows": int(Xg.shape[0]), "chunks_read": n_chunks, "problems": {},
           "run_table_sha256": sha((d / "run_table.json").read_bytes()), "script_sha256": sha(Path(__file__).read_bytes()),
           "rows_index_sha256": sha((d / "rows_index.json").read_bytes()),  # addendum 26: binds the cell to the skip record
           "recipe": {"registered": "648f33e61 cmd_sim_detector RULE.md L19-27 + honest RULE.md @ 596bc767e",
                      "C": C_REG, "grad_bar": GRAD_BAR, "iter_cap": ITER_CAP,
                      "SPEC_SILENT": {"class_weight": CLASS_WEIGHT, "intercept_penalized": PENALIZE_INTERCEPT,
                                      "objective": "(C*sum logloss + 0.5*||w||^2)/n", "std_ddof": STD_DDOF,
                                      "std_min": STD_MIN, "fit_target": FIT_TARGET,
                                      "solver": "float64 truncated Newton + PCG", "null_seed": NULL_SEED},
                      "layout": LAYOUT},
           "versions": {"torch": torch.__version__, "numpy": np.__version__}}
    ext = {}
    for p in problems:
        ext[p] = {}
        res["problems"][p] = fit_problem(Xg, row_run, runs, *ALL_PROBLEMS[p], n_null, ext[p])
    if {"simh_vs_b", "a_vs_b"} <= set(problems):  # paired: same main draws over the union (RULE L16; A1 RULE @ c663d0204 L285-288)
        ms = sorted(set(ext["simh_vs_b"]["mains"]) | set(ext["a_vs_b"]["mains"]))
        mult = boot_mult(ms)
        pa = {}
        for p in ("simh_vs_b", "a_vs_b"):
            mi = np.array([ms.index(m) for m in ext[p]["mains"]])
            pa[p] = wauroc(ext[p]["run_score"], ext[p]["y"], mult, mi)
        dlt = pa["simh_vs_b"] - pa["a_vs_b"]
        res["paired_simh_minus_a"] = {"observed": res["problems"]["simh_vs_b"]["observed"]["run_level"]["auroc"]
                                      - res["problems"]["a_vs_b"]["observed"]["run_level"]["auroc"],
                                      **ci(dlt), "share_le_0": float(np.mean(dlt[np.isfinite(dlt)] <= 0))}
    main_certs = [c for p in res["problems"].values() for c in p["certificates"] if "variant" not in c]
    n_unc = sum(not ok for c in main_certs for ok in c["certified"])  # the balanced diagnostic is reported, not gated
    res["n_fits"] = sum(len(c["certified"]) for c in main_certs)
    res["n_uncertified"] = n_unc
    res["label"] = "clean-room re-fit (PRELIMINARY); " + ("uncertified fits present" if n_unc else "all fits certified")
    res["wall_s"] = time.time() - t0
    blob = jdump(res)
    write_new(dest, blob)
    arr = io.BytesIO()
    np.savez(arr, **{f"{p}__{k}": v for p in ext for k, v in ext[p].items() if isinstance(v, np.ndarray)},
             **{f"{p}__{f}__{k}": v for p in ext for f, w in ext[p]["weights"].items() for k, v in w.items()})
    write_new(d / sub / f"cell_{readout}_L{layer}_arrays.npz", arr.getvalue())
    out_vol.commit()
    return {"cell": res["cell"], "n_rows": res["n_rows"], "n_fits": res["n_fits"], "n_uncertified": n_unc,
            "zero_row_runs": {p: len(v["zero_row_runs"]) for p, v in res["problems"].items()},
            "record_sha256": sha(blob), "wall_s": round(res["wall_s"])}


@app.function(image=cpu_img, cpu=(1, 1), memory=(2048, 2048), timeout=DRIVER_TIMEOUT, retries=0,
              startup_timeout=STARTUP_S, scaledown_window=SCALEDOWN_S, volumes={OUT: out_vol})
@sanitized
def fit_driver(stamp: str, problems: list, n_null: int, sub: str = "") -> dict:
    """23 step 3b: client-drop-safe fit. The laptop makes ONE spawn of this function under `modal run --detach`; the pilot cell,
    the pilot gate and the remaining cells all run from here, sequentially, inside Modal. Each cell writes its own create-only
    record; this driver writes fit_summary.json (counts, shas, wall_s, projection and go only) create-only to the stamp dir
    (addendum 26: to <stamp>/<sub>/ when sub is set, after the reused inputs are checked against REUSE_PINS)."""
    d = Path(OUT, stamp)
    out_vol.reload()
    if (d / sub / "fit_summary.json").exists():
        stop("create-only: fit_summary.json exists")
    if sub:
        pins = REUSE_PINS.get(stamp)
        if pins is None:
            stop("no reuse pins for this stamp")
        for name, want in pins.items():
            if sha((d / name).read_bytes()) != want:
                stop(f"reused input sha: {name}")
    first, rest = DEFAULT_CELLS[0], DEFAULT_CELLS[1:]
    cells = [fit_cell.remote(stamp, first[0], first[1], problems, n_null, sub)]
    proj = len(DEFAULT_CELLS) * cells[0]["wall_s"] / 3600 * H100_RATE
    go = proj <= FIT_BUDGET
    if go:
        for ro, ly in rest:
            cells.append(fit_cell.remote(stamp, ro, ly, problems, n_null, sub))
    summ = {"stamp": stamp, "problems": problems, "sub": sub,
            "pilot_projection_usd": round(proj, 2), "budget_usd": FIT_BUDGET, "go": go,
            "cells": cells, "n_cells_done": len(cells), "n_uncertified_total": sum(c["n_uncertified"] for c in cells),
            "exit_nonzero_required": any(c["n_uncertified"] for c in cells) or not go}  # RULE L45-47; a detached run has no exit code
    out_vol.reload()
    write_new(d / sub / "fit_summary.json", jdump(summ))
    out_vol.commit()
    return summ


# =====================================================================================================
# laptop side: git show + sha only; prints booleans, counts, shas
# =====================================================================================================
@app.local_entrypoint()
def main(stage: str, repo: str = "", stamp: str = "", full: bool = False, problem: str = DEFAULT_PROBLEMS[0]):
    if full:  # 23 step 2b (09): the cost bound and Aswin's approval cover DEFAULT_CELLS only; --full needs a new bound and approval
        sys.exit("STOP: --full is out of scope (23 step 2b)")
    if problem not in ALL_PROBLEMS:
        sys.exit("STOP: unknown --problem")
    if problem != DEFAULT_PROBLEMS[0] and stage != "fit":  # addendum 26: a_vs_b reuses 4(b)'s dry / skip inputs; fit only
        sys.exit("STOP: --problem other than the default is for --stage fit only")
    from datetime import datetime
    from zoneinfo import ZoneInfo
    problems = [problem]  # one problem per fit (addendum 26); --full STOPs above; the fit driver uses DEFAULT_CELLS
    sub = "" if problem == DEFAULT_PROBLEMS[0] else problem  # 4(b) keeps the stamp dir; others write to <stamp>/<problem>/
    n_null = FULL_N_NULL if full else DEFAULT_N_NULL
    if stage == "dry":
        if not repo or stamp:
            sys.exit("dry needs --repo and makes its own stamp")
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")

        def show(commit, path, want):
            b = subprocess.run(["git", "-C", repo, "show", f"{commit}:{path}"], check=True, capture_output=True).stdout
            if sha(b) != want:
                sys.exit(f"STOP: pin {path} sha256 {sha(b)[:12]} != {want[:12]}")
            return b

        blobs, index = {}, {}
        for name, (c, p, h) in PINS.items():
            blobs[h] = show(c, p, h)
            index[name] = {"commit": c, "path": p, "sha256": h, "role": name}
        for i, f in enumerate(json.loads(blobs[PINS["sources_v2"][2]])["files"]):  # paths + shas only
            if f["role"] in ("routing", "extraction"):
                blobs[f["sha256"]] = show(f["commit"], f["path"], f["sha256"])
                index[f"{f['role']}_{i}"] = {"commit": f["commit"], "path": f["path"], "sha256": f["sha256"], "role": f["role"]}
        with out_vol.batch_upload() as up:
            for h, b in blobs.items():
                up.put_file(io.BytesIO(b), f"/{stamp}/inputs/{h}")
            up.put_file(io.BytesIO(jdump(index)), f"/{stamp}/inputs/index.json")
        print(json.dumps({"stamp": stamp, "n_pinned_files": len(index), **dry.remote(stamp)}))
    elif stage == "layout":
        print(json.dumps(layout.remote(stamp)))
    elif stage == "skip":
        print(json.dumps(skip.remote(stamp)))
    elif stage == "fit":  # 23 step 3b: exactly one remote call; launch with `modal run --detach` (client-drop-safe)
        if sub and stamp not in REUSE_PINS:
            sys.exit("STOP: --problem a_vs_b needs a stamp in REUSE_PINS")
        call = fit_driver.spawn(stamp, problems, n_null, sub)
        print(json.dumps({"stage": "fit", "problem": problem, "fit_driver_call_id": call.object_id,
                          "summary": f"/{stamp}/{sub + '/' if sub else ''}fit_summary.json"}))
    else:
        sys.exit(f"unknown stage {stage}")
