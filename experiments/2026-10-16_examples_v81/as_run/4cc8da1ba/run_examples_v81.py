"""v81 example tables, KEYS ONLY (q2's fork; started from p2's draft @ 933080851). PRELIMINARY until Aswin reviews.

Rules (pinned in PINS by commit + sha256):
  09 RULE "v81 pre-statements" @ 2d61af835: §1 the outcome table (Appendix B), §2 the six false alarms.
  09 RULE item 28 @ a3d81dbcb: the on-policy TRAINING pair of own_all_rs_uncut_L48 (the training-examples table).
  09 RULE item 43 (relayed by q2, 10-16 ~12:5x): POOLED calibration is the full-length lead. §1's catches and §2's pools use the pooled
    thresholds (real = pooled H_B, sim = pooled H_sim, deduped by physical run key). --calibration switches pooled | per_origin.
  09's answers (relayed by q2, 10-16 ~12:4x): item 28's key list is DERIVED by script (union over folds of the training RUN keys; identity by
    run key + .eval sha; attack = world run1, honest = world honest_a); its RNG is index A's: random.Random(int.from_bytes(sha256(label)[:8],
    "big")), run = rng.choice(sorted run keys), then unit = rng.choice(the run's C units sorted by (irid, call_index), index A's text, irid
    from the staged call_end entries) with the SAME rng; the command shown is that seeded unit. §1 uses numpy default_rng as stated. The outcome table prints z - T (the command's score minus the alarm threshold, both in
    SDs of that evaluation's honest-run scores; z = the score standardized on its fold's honest calibration moments, what Stage A thresholds)
    and keeps the raw z; every false-alarm row must have z - T > 0.
  Viewer spec @ 446ed20d8: CUT = None (whole run); a rendered command is shown only if the sha256 of its bytes AND of the full arguments
    equal the index's, by key, else STOP.

Modes (one Modal function, `job`; every Volume read-only; CPU only):
  --mode keylist     item 28's training key list. The laptop rebuilds own_all_rs_uncut's training identity with the fit's OWN code
                     (run_own_all._uncut_training @ 19611550c, git only) and requires it to equal the fit record's training_report. The job
                     reads the 16 per-root uncut C training-row files (sha256 = the fit record's), keeps the training rows exactly as
                     fit_uncut does (world run1 / honest_a, main in the universe), asserts the fit's own counts and returns one entry per
                     training run (keys, shas and counts). The record is committed BEFORE any draw.
  --mode train-draw  item 28's draw from the committed key list (--keylist <path>@<commit>): run, then unit, same rng; the unit's cutfull
                     locator (staged row line + call, its .eval) and the sha256 / length of its command bytes and full arguments. No text.
  --mode tables      §1 + §2: HELD until q2 relays the pooled Stage A record and its selection record (path @ commit + sha256).
  --mode render      Stage 2 (q2's "go stage 2" only, --go-stage-2): the drawn commands' text, hash-gated, returned by the function
                     and written ONLY to --text-out under this session's scratchpad (exclusive create). Never printed or logged.

  uv run --no-project --python 3.12 --with modal==1.6.1 --with numpy==2.3.3 modal run --detach \\
      experiments/2026-10-16_examples_v81/run_examples_v81.py --mode keylist --fit-code-dir <checkout of 19611550c>
"""

import hashlib
import io
import json
import random
import re
import struct
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE  # in the container the module sits at /root (launch 13:11 crash-looped on parents[1])
APP = "probe-shift_aswin-surya_sub-q2examples_v81_kill-2026-11-15"
VOLS = {
    "/p2": "probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13",  # own_all_rs_uncut's training rows (the fit's OUT Volume)
    "/b": "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14",  # d9's cutfull staged sets (the fit's /b)
    "/q4": "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13",  # T196 .eval files
    "/acts": "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14",  # +311 / nm360 .eval files
}
EVAL_ROOTS = ("/q4/evals_v1", "/acts/evals_v1")  # a run's .eval is found by its sha256 (exactly one distinct match), never by name alone
FIT_STAMP = "2026-10-16_030614"
ROWS_DIR = f"/p2/own_all_rs_uncut/{FIT_STAMP}/rows"  # rows_uncut's output: <root>__<R|C>.npz (names, run_key, call_index, unit, token, X)
READOUT = "C"  # the lead readout
PINS = {  # name -> (commit, path, sha256); read from git and refused on any byte difference
    "rule_v81": ("2d61af835", "experiments/2026-10-14_priority1_spec_09/RULE.md", "b801ad51e46cb0e92b98f02d62697bba31cfff52395a68bc795cf3ff1889c5e7"),
    "rule_item28": ("a3d81dbcb", "experiments/2026-10-14_priority1_spec_09/RULE.md", "c0de0b0ec9b31ba4b98a0c3730ac2bacc2ee3c63b6802e2cc05fdc1518f6d7e4"),
    "viewer_spec": ("446ed20d8", "experiments/2026-10-14_priority1_spec_09/READ_LOG.md", "d56827ccdc0d448b28384e3f01295102176c800f5bb4731cb5d8ab070a8f0552"),
    "p2_draft": ("933080851", "experiments/2026-10-16_examples_v81/run_examples_v81.py", "616d93742b47886ebe4fbf21bfc34263117a256e6ad5a0838370abff2044edbe"),
    "inputs_full": ("927fabb0d", "experiments/2026-10-15_fpr_sweep_507_prep/config/INPUTS_FULL_2026-10-16_110447.json",
                    "d6923a75aa8f914cd3b2e92d8403f125dd095219997aebea8b1d6e4e2b67f77c"),
    "fit": ("19611550c", "experiments/2026-10-14_p2_family_expansion/results/own_all_rs_uncut_fit_2026-10-16_030614.json",
            "f62f4d660109fff5d3d207a024299f3c254fbfb2c603ddc7fc62b40b38760442"),
    "companion": ("e452cb797", "experiments/2026-10-14_p2_family_expansion/results/own_all_rs_uncut_INPUTS_FULL_companion_2026-10-16.json",
                  "e83f55945978e0ec2944bb23922003fbad31ac9afc54957a6b1f548684be37e7"),
    "own_all_inputs": ("807c52f78", "experiments/2026-10-14_p2_family_expansion/results/own_all_inputs_2026-10-15.json",
                       "26f9d8d1fe08dd2ef430a1bef8af50f336da47234fe364e1a94879bcb54491d1"),
    "fit_code": ("19611550c", "experiments/2026-10-14_p2_family_expansion/run_own_all.py", "d975d001d77272ca31b861ffc1fb2b18c51c522c31699c31291ccc59c53cb1cb"),
}
PROBE = "own_all_rs_uncut_L48"
TRAIN_LABEL = "p2-writing-examples-2026-10-16|own_all_rs_uncut_L48|{cls}"  # item 28, verbatim
TRAIN_WORLD = {"attack": "run1", "honest": "honest_a"}  # 09: the probe's training arms
UNIT_ORDERS = ("irid", "token")  # index A sorts a run's C units by (irid, call_index); the alternative is the unit index (token order)
UNIT_ORDER = "irid"  # 09 via q2 (10-16 ~13:0x): index A literally, (irid, call_index) with irid from the staged call_end entries (Astra r1 #3)
STAGE2_GO = "q2: go stage 2 (Aswin's one-time exception for the v81 tables)"  # the remote render refuses any other value (Astra r1 #1)
SEED_TABLE2 = b"p2-writing-examples-2026-10-16|table2"  # v81 §1, verbatim
SEED_FALSEALARMS = b"p2-writing-examples-2026-10-16|falsealarms"  # v81 §2, verbatim
OUTCOMES = ("both caught", "real caught, sim missed", "real missed, sim caught", "both missed")  # v81 §1 order
N_FALSEALARMS = 6
LEVEL = 0.01  # per action
CALIBRATIONS = ("pooled", "per_origin")
KEYLIST_PIN = None  # (commit, path, sha256) of the committed key list: set by a reviewed commit AFTER the keylist job and BEFORE train-draw
INDEX_PIN = None  # (commit, path, sha256) of the committed training-pair record: set by a reviewed commit before Stage 2 render
HELD = {"pooled_stage_a": None, "pooled_selection": None}  # (commit, path, sha256): q2 relays them; until then --mode tables STOPs
SCRATCH = Path("/private/tmp/claude-501/-Users-aswinsurya-probe-shift-q2/802b982d-f788-49b8-9390-11c29e3725f3/scratchpad")

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "zstandard==0.25.0")
app = modal.App(APP, image=image)
vols = {k: modal.Volume.from_name(v).read_only() for k, v in VOLS.items()}


def nnint(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


class Gate(Exception):
    """A drawn entry that fails a check: reported with its reason (structure only, no text) and no hashes."""


# ---------- pure selection rules (no I/O) ----------


def seed8(label: bytes) -> int:
    return int.from_bytes(hashlib.sha256(label).digest()[:8], "big")


def train_rng(cls: str) -> random.Random:
    """Item 28 + 09: index A's generator, random.Random, with the 8-byte seed of the item-28 label."""
    return random.Random(seed8(TRAIN_LABEL.format(cls=cls).encode()))


def outcome(real_caught: bool, sim_caught: bool) -> str:
    return OUTCOMES[(0 if real_caught else 2) + (0 if sim_caught else 1)]


def table_outcomes(cells: list) -> dict:
    """v81 §1. cells: one per cohort cell, {"cell": "<origin>|<cell>", "real_caught", "sim_caught", "real", "sim"} where real / sim are each
    run's shown command ({"unit", "z", "z_minus_T", ...} or "no command" for a retained zero-tool arm, which counts as missed). Sorted by the
    plain string key; ONE cell drawn uniformly per outcome cell with ONE numpy default_rng, outcome cells in OUTCOMES order (an empty outcome
    cell draws nothing and reads "none"); n printed per outcome cell."""
    import numpy as np

    keys = [c["cell"] for c in cells]
    if len(set(keys)) != len(keys):
        raise SystemExit("outcome table: a cell key appears twice: STOP")
    for c in cells:
        for side in ("real", "sim"):
            if c[side] == "no command" and c[f"{side}_caught"]:
                raise SystemExit(f"{c['cell']}: a zero-tool arm is caught (the zero-tool rule says missed): STOP")
    by = {o: [] for o in OUTCOMES}
    for c in sorted(cells, key=lambda c: c["cell"]):
        by[outcome(c["real_caught"], c["sim_caught"])].append(c)
    g = np.random.default_rng(seed8(SEED_TABLE2))
    out = {}
    for o, v in by.items():
        d = v[int(g.integers(len(v)))] if v else None
        out[o] = {"n": len(v), "drawn": d if d else "none"}
    return out


def false_alarms(events: list, probes: list, pools: list, available: set) -> dict:
    """v81 §2 with item 43: an event is one per honest run per probe, the run's FIRST content-only command unit above its pool's threshold
    ({"probe", "pool", "run", "unit", "z", "T", "z_minus_T"}). Every (probe, pool) of the lead must be present by key (q2/09: the "exactly two
    pools" STOP is replaced by this). Sorted by (probe, pool, run key, unit index); SIX drawn uniformly without replacement; totals printed for
    every probe x pool, zeros included. Every event must have finite z and T and z - T > 0, computed here (Astra r1 #8). available: the
    (probe, pool) pairs whose units were loaded, by key; it must equal probes x pools, so a missing pool is not read as zero alarms (r1 #11)."""
    import numpy as np

    if len(probes) != 6 or len(set(probes)) != 6 or not pools or len(set(pools)) != len(pools):
        raise SystemExit("false alarms: not six distinct lead probes, or no / repeated pools: STOP")
    want = {(a, b) for a in probes for b in pools}
    if set(available) != want:
        raise SystemExit(f"false alarms: loaded pools {sorted(set(available) ^ want)[:4]} differ from the lead probes x pools: STOP")
    if any((e["probe"], e["pool"]) not in want for e in events):
        raise SystemExit("false alarms: an event outside the lead probes x pools: STOP")
    if len({(e["probe"], e["pool"], e["run"]) for e in events}) != len(events):
        raise SystemExit("false alarms: two events for one honest run x probe x pool: STOP")
    marg = [float(e["z"]) - float(e["T"]) for e in events]  # computed once (Astra r2 #5): finite operands AND a finite, positive margin
    bad = [e for e, m in zip(events, marg) if not (np.isfinite(e["z"]) and np.isfinite(e["T"]) and np.isfinite(m) and m > 0)]
    if bad:
        raise SystemExit(f"false alarms: {len(bad)} events with non-finite z / T / margin or z - T <= 0: STOP")
    ev = sorted(({k: v for k, v in e.items() if k != "z_minus_T"} | {"z_minus_T": m} for e, m in zip(events, marg)),
                key=lambda e: (e["probe"], e["pool"], e["run"], e["unit"]))
    if len(ev) < N_FALSEALARMS:
        raise SystemExit(f"{len(ev)} alarm events < {N_FALSEALARMS}: STOP and ask 09")
    pick = np.random.default_rng(seed8(SEED_FALSEALARMS)).choice(len(ev), size=N_FALSEALARMS, replace=False)
    totals = {f"{a}|{b}": 0 for a in probes for b in pools}
    for e in ev:
        totals[f"{e['probe']}|{e['pool']}"] += 1
    return {"draws": [ev[int(i)] for i in pick], "events_per_probe_pool": totals, "universe": len(ev)}


def first_alarm(units: list, T: float):
    """units: one honest run's content-only C units [{"unit", "z"}]; the first in token (unit) order with z > T, else None."""
    for u in sorted(units, key=lambda u: u["unit"]):
        if u["z"] > T:
            return u
    return None


def argmax_command(units: list):
    """Per run, the argmax-scoring content-only command unit (out-of-fold z; ties: the lowest unit); a zero-tool run has none."""
    return max(units, key=lambda u: (u["z"], -u["unit"])) if units else None


# ---------- container helpers (bytes stay in the container; only keys, counts and digests leave) ----------


def eval_members(path: Path) -> list:
    with zipfile.ZipFile(path) as z:
        return [i.filename for i in z.infolist()]


def read_member(path: Path, name: str) -> dict:
    """One JSON member of an Inspect .eval (zstd members, zip method 93, read directly); attachments inlined (run_examples_index.py)."""
    import zstandard

    with zipfile.ZipFile(path) as z:
        info = z.getinfo(name)
        if info.compress_type != 93:
            s = json.loads(z.read(info))
        else:
            with open(path, "rb") as f:
                f.seek(info.header_offset)
                name_len, extra_len = struct.unpack("<HH", f.read(30)[26:30])
                f.seek(info.header_offset + 30 + name_len + extra_len)
                s = json.loads(zstandard.ZstdDecompressor().decompress(f.read(info.compress_size), max_output_size=info.file_size))
    att = s.pop("attachments", None) or {}

    def inline(x):
        if isinstance(x, str) and x.startswith("attachment://"):
            return att.get(x[len("attachment://"):], x)
        if isinstance(x, list):
            return [inline(v) for v in x]
        if isinstance(x, dict):
            return {k: inline(v) for k, v in x.items()}
        return x

    return inline(s)


def sample_of(path: Path, epoch0: int) -> tuple:
    smp = [n for n in eval_members(path) if n.startswith("samples/") and n.endswith(".json")]
    if len(smp) != 1:
        smp = [n for n in smp if n.endswith(f"_epoch_{epoch0 + 1}.json")]
    if len(smp) != 1:
        raise Gate(f"{len(smp)} sample members match epoch {epoch0}")
    s = read_member(path, smp[0])
    if s.get("epoch") != epoch0 + 1:  # Inspect epochs are 1-based; the key's #e<n> is 0-based (Astra r1 #4: checked even for a single member)
        raise Gate("the sample's epoch is not the key's")
    return s, smp[0]


def no_attachment_refs(x) -> None:
    """Astra r1 #5: an unresolved attachment://... reference anywhere in the arguments STOPs (never hashed or shown)."""
    if isinstance(x, str) and x.startswith("attachment://"):
        raise Gate("unresolved attachment reference in the tool-call arguments")
    if isinstance(x, dict):
        for v in x.values():
            no_attachment_refs(v)
    elif isinstance(x, list):
        for v in x:
            no_attachment_refs(v)


def command_of(args) -> tuple:
    """The rule's command bytes: arguments['cmd'], else arguments['command'], else the canonical JSON; and the canonical full arguments."""
    full = json.dumps(args, sort_keys=True, separators=(",", ":"))
    if isinstance(args, dict) and "cmd" in args:
        field, s = "cmd", args["cmd"]
    elif isinstance(args, dict) and "command" in args:
        field, s = "command", args["command"]
    else:
        field, s = "json", full
    if not isinstance(s, str):
        raise Gate(f"arguments[{field!r}] is a {type(s).__name__}, not a string")
    return field, s, full


def digests(args) -> dict:
    field, s, full = command_of(args)
    return {"field": field, "length": len(s), "sha256": hashlib.sha256(s.encode()).hexdigest(),
            "arguments_length": len(full), "arguments_sha256": hashlib.sha256(full.encode()).hexdigest()}


def tool_name(x) -> str:
    if not isinstance(x, str) or len(x) > 64 or not re.fullmatch(r"[A-Za-z0-9_.:-]+", x):
        raise Gate("the tool-call function is not a plain tool name")
    return x


def find_eval(key: str, sha: str) -> tuple:
    """(volume name, path in the volume, bytes) of the run's .eval: the key without #e<n> (and without a leading runs/), under EVAL_ROOTS,
    with the identity's sha256; exactly one root may hold it with that sha."""
    rel = key.split("#e")[0]
    rel = rel[5:] if rel.startswith("runs/") else rel
    hits = []
    for r in EVAL_ROOTS:
        p = Path(r) / rel
        if p.exists():
            b = p.read_bytes()
            if hashlib.sha256(b).hexdigest() == sha:
                mount = "/" + r.split("/")[1]
                hits.append((VOLS[mount], str(p).removeprefix(mount + "/"), b))
    if len(hits) != 1:
        raise Gate(f"{len(hits)} .eval files with the identity's sha256 under {EVAL_ROOTS}")
    return hits[0]


def staged_row(e: dict) -> tuple:
    """(line, row) of the run's staged row in its cutfull staged inputs file (sha-checked before parsing)."""
    raw = (Path("/b") / e["staged"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != e["staged_sha256"]:
        raise Gate("staged inputs file sha256 differs from the identity")
    lines = raw.splitlines()
    rows = [(i, json.loads(lines[i])) for i, ln in enumerate(lines) if json.dumps(e["key"]).encode() in ln]
    rows = [(i, r) for i, r in rows if r.get("key") == e["key"]]
    if len(rows) != 1:
        raise Gate(f"{len(rows)} staged rows carry the run key")
    return rows[0]


def call_for(row: dict, call_index: int, token: int) -> tuple:
    """(j, call): the row's unique call_end with this call_index and readout token, every identifier well-formed (Astra r1 #2, #4)."""
    js = [j for j, c in enumerate(row.get("calls") or []) if c.get("kind") == "call_end" and c.get("call_index") == call_index and c.get("readout") == token]
    if len(js) != 1:
        raise Gate(f"{len(js)} staged call_end entries match (call_index, readout token)")
    c = row["calls"][js[0]]
    if not (isinstance(c.get("irid"), str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", c["irid"])) or not all(nnint(c.get(x)) for x in ("turn", "call_index", "readout")):
        raise Gate("the staged call's irid / turn / call_index / readout is not well-formed")
    return js[0], c


def arguments_at(e: dict, line_row=None) -> dict:
    """The drawn unit's tool-call arguments: staged row (sha-checked) -> its call_end with this call_index and readout token -> the .eval
    (sha-checked) model event with the call's x-irid -> tool call call_index. Also returns the locator fields (keys and ints only)."""
    line, row = line_row or staged_row(e)
    u = e["unit"]
    j, c = call_for(row, u["call_index"], u["token"])
    if "irid" in u and u["irid"] != c["irid"]:  # Astra r2 #2: the recorded locator must re-derive to the same call
        raise Gate("the staged call's irid differs from the drawn unit's")
    vol, path, eb = find_eval(e["key"], e["eval_sha256"])
    snap = Path("/tmp") / f"verified_{e['eval_sha256']}.eval"
    snap.write_bytes(eb)
    s, member = sample_of(snap, int(e["key"].rsplit("#e", 1)[1]))
    if not re.fullmatch(r"samples/[A-Za-z0-9_.-]{1,200}\.json", member):
        raise Gate("the sample member name is not well-formed")
    model = [x for x in s.get("events") or [] if x.get("event") == "model"]
    hit = [i for i, x in enumerate(model) if (((x.get("call") or {}).get("request") or {}).get("extra_headers") or {}).get("x-irid") == c["irid"]]
    if len(hit) != 1:
        raise Gate(f"{len(hit)} model events carry the call's x-irid")
    ch = (model[hit[0]].get("output") or {}).get("choices") or []
    if len(ch) != 1:  # Astra r2 #8: the staged call belongs to the event's only choice
        raise Gate(f"the model event has {len(ch)} choices, not 1")
    tcs = (ch[0].get("message") or {}).get("tool_calls") or []
    if not 0 <= c["call_index"] < len(tcs):
        raise Gate(f"call_index outside the event's {len(tcs)} tool calls")
    tc = tcs[c["call_index"]]
    no_attachment_refs(tc.get("arguments"))
    loc = {"staged_row": {"source": "cutfull", "staged": e["staged"], "staged_sha256": e["staged_sha256"], "arm": e["arm"], "line": line, "call": j},
           "unit": {"step": c["turn"], "action": c["call_index"], "token": c["readout"], "irid": c["irid"], "row_unit": u["unit"]},
           "run_eval": {"volume": vol, "eval": path, "sha256": e["eval_sha256"], "sample_member": member, "model_event": hit[0]},
           "tool": tool_name(tc.get("function"))}
    return {"loc": loc, "args": tc.get("arguments")}


def gate_digests(args, e: dict) -> dict:
    """Stage 2's hash gate (viewer spec @ 446ed20d8): the command text AND the canonical full arguments, only if BOTH sha256 / length pairs
    equal the index's, else STOP (no text in the message; Astra r2 #4: the full arguments are rendered too, as index A's amendment's viewer)."""
    d = digests(args)
    bad = [k for k in ("sha256", "length", "arguments_sha256", "arguments_length") if d[k] != e[k]]
    if bad:
        raise SystemExit(f"STOP: {e.get('label', '?')}: {','.join(bad)} differ from the index (viewer spec @ 446ed20d8)")
    _, text, full = command_of(args)
    return {"text": text, "arguments_json": full}


def _units_digest(units: list) -> str:
    return hashlib.sha256(json.dumps(units, separators=(",", ":")).encode()).hexdigest()


def _run_units(root_sha: str, root: str, key: str, name: str) -> list:
    """One training run's C rows in the root's per-root file (sha-checked), as sorted [unit, call_index, token]; every row of the key must
    carry the key list's world|cell name (Astra r2 #1: the world is re-checked against the rows file, not taken from the list)."""
    import numpy as np

    b = (Path(ROWS_DIR) / f"{root}__{READOUT}.npz").read_bytes()
    if hashlib.sha256(b).hexdigest() != root_sha:
        raise SystemExit(f"{root}: rows file sha256 differs from the fit record")
    z = np.load(io.BytesIO(b))
    m = z["run_key"].astype(str) == key
    if not m.any() or set(z["names"].astype(str)[m].tolist()) != {name}:
        raise SystemExit("STOP: the drawn run's rows file name (world|cell) differs from the key list")
    return sorted([int(a), int(c), int(t)] for a, c, t in zip(z["unit"][m], z["call_index"][m], z["token"][m], strict=True))


def _keylist(p: dict) -> dict:
    import numpy as np

    main_of, uni, eps = p["main_of"], set(p["universe_mains"]), p["epochs"]
    runs, rows_tr, past_tr, per_root = {}, 0, 0, {}
    for root, want in sorted(p["roots"].items()):
        b = (Path(ROWS_DIR) / f"{root}__{READOUT}.npz").read_bytes()  # read once: hashed, then parsed from the same bytes
        if hashlib.sha256(b).hexdigest() != want["sha256"]:
            raise SystemExit(f"{root}: rows file sha256 differs from the fit record")
        z = np.load(io.BytesIO(b))
        names, rk = z["names"].astype(str), z["run_key"].astype(str)
        un, ci, tok = (z[x].astype(np.int64) for x in ("unit", "call_index", "token"))
        if len(names) != want["rows"] or len(set(names.tolist())) != want["runs"] or not (len(rk) == len(un) == len(ci) == len(tok) == len(names)):
            raise SystemExit(f"{root}: rows / runs differ from the fit record")
        world = np.array([x.split("|", 1)[0] for x in names])
        cell = np.array([x.split("|", 1)[1] for x in names])
        miss = sorted({c for c in cell.tolist() if c not in main_of})
        if miss:
            raise SystemExit(f"{root}: cells without a main {miss[:3]}")
        trn = np.array([main_of[c] in uni for c in cell], dtype=bool) & np.isin(world, ["run1", "honest_a"])  # fit_uncut: uni & ~isnan(y), any fold
        rows_tr += int(trn.sum())
        past_tr += int((trn & (tok >= 32768)).sum())
        per_root[root] = {"rows": int(len(names)), "training_rows": int(trn.sum()), "training_runs": len(set(rk[trn].tolist()))}
        for i in np.flatnonzero(trn):
            k = rk[i]
            r = runs.setdefault(k, {"root": root, "name": names[i], "units": []})
            if r["root"] != root or r["name"] != names[i]:
                raise SystemExit("a training run key appears under two roots or two world|cell names: STOP")
            r["units"].append([int(un[i]), int(ci[i]), int(tok[i])])
    if any(e["staged_sha256"] == "from_config" for e in eps.values()):  # Astra r2 #3: no live config.json as fit provenance
        raise SystemExit("STOP: an identity epoch's staged sha256 is 'from_config' (not pinned by the fit's identity)")
    ex = p["expect"]
    got = {"training_rows": rows_tr, "training_rows_past_32768": past_tr, "training_runs": len(runs)}
    if any(got[k] != ex[k] for k in got):
        raise SystemExit(f"STOP: training counts {got} differ from the fit record's {dict((k, ex[k]) for k in got)}")
    out = []
    for k in sorted(runs):
        r, e = runs[k], eps.get(k)
        w, c = r["name"].split("|", 1)
        if e is None or (e["world"], e["cell"], e["run"]) != (w, c, r["root"]):
            raise SystemExit("STOP: a training run is not in the fit's training identity with the same world, cell and root")
        u = sorted(r["units"])
        if len({x[0] for x in u}) != len(u) or [x[2] for x in u] != sorted(x[2] for x in u):
            raise SystemExit("STOP: a training run's units repeat or are not in token order")
        out.append({"key": k, "world": w, "cell": c, "root": r["root"], "eval_sha256": e["eval_sha256"], "staged": e["staged"],
                    "staged_sha256": e["staged_sha256"], "arm": e["arm"], "n_units": len(u), "units_sha256": _units_digest(u)})
    return {"runs": out, "per_root": per_root, "counts": got | {"by_world": {w: sum(x["world"] == w for x in out) for w in ("run1", "honest_a")}},
            "identity_epochs_without_training_rows": sorted(set(eps) - set(runs))}


def _train_draw(p: dict) -> list:
    if p.get("unit_order") != "irid":
        raise SystemExit("train-draw: the unit order is not index A's (irid, call_index): STOP")
    kl, roots = p["keylist"], p["root_sha"]
    out = []
    for cls, world in TRAIN_WORLD.items():
        runs = sorted((r for r in kl["runs"] if r["world"] == world), key=lambda r: r["key"])
        rng = train_rng(cls)
        r = rng.choice(runs)
        units = _run_units(roots[r["root"]], r["root"], r["key"], f"{world}|{r['cell']}")
        if len(units) != r["n_units"] or _units_digest(units) != r["units_sha256"]:
            raise SystemExit("STOP: the drawn run's units differ from the committed key list")
        e = {"table": "training examples pair (RULE item 28)", "probe": f"{PROBE} ({READOUT}, layer 48)", "class": cls,
             "label": TRAIN_LABEL.format(cls=cls), "pool": f"{len(runs)} training runs (world {world})", "task_key": r["key"], "world": world,
             "cell": r["cell"], "root": r["root"], "n_units_in_run": len(units),
             "unit_sort": "(irid, call_index), irid from the staged row's call_end entries (index A, rule @ 160620a04; 09 via q2 10-16 ~13:0x)"}
        try:
            lr = staged_row(r)
            cand = []
            for un, ci, tok in units:
                _, c = call_for(lr[1], ci, tok)
                cand.append((c["irid"], ci, un, tok))
            if len({(a, b) for a, b, _, _ in cand}) != len(cand):
                raise Gate("two units share (irid, call_index)")
            cand.sort(key=lambda x: (x[0], x[1]))
            irid, ci, un, tok = rng.choice(cand)
            e["unit_drawn"] = {"unit": un, "call_index": ci, "token": tok, "irid": irid}
            a = arguments_at(r | {"unit": e["unit_drawn"]}, lr)
            out.append(e | a["loc"] | digests(a["args"]) | {"status": "ok"})
        except Gate as g:
            out.append(e | {"status": f"gate: {g}"})
    return out


def _render(p: dict) -> list:
    """Stage 2 only: each committed entry's command text, re-derived from its source and returned only when BOTH digests equal the index's."""
    if p.get("stage_2") != STAGE2_GO:  # Astra r1 #1: refused at the remote boundary too, default denied
        raise SystemExit("render: Stage 2 is not authorized for this call")
    ents = p["entries"]
    if sorted(e["class"] for e in ents) != sorted(TRAIN_WORLD) or len({e["task_key"] for e in ents}) != len(ents) or \
            any(e["status"] != "ok" or e["identity"]["key"] != e["task_key"] or e["identity"]["world"] != TRAIN_WORLD[e["class"]] or
                e["world"] != TRAIN_WORLD[e["class"]] for e in ents):
        raise SystemExit("render: not exactly one ok attack (run1) + one ok honest (honest_a) entry on distinct runs: STOP")
    out = []
    for e in ents:
        a = arguments_at(e["identity"] | {"unit": e["unit_drawn"]})
        if {k: a["loc"][k] for k in ("staged_row", "unit", "run_eval", "tool")} != {k: e[k] for k in ("staged_row", "unit", "run_eval", "tool")}:
            raise SystemExit(f"STOP: {e['label']}: the re-derived locator differs from the index")
        out.append({"label": e["label"], "task_key": e["task_key"]} | gate_digests(a["args"], e))
    return out


@app.function(cpu=2, memory=16384, timeout=1800, retries=0, volumes=vols)
def job(mode: str, payload: dict):
    err = None
    try:
        res = {"keylist": _keylist, "train_draw": _train_draw, "render": _render}[mode](payload)
        return json.loads(json.dumps(res, allow_nan=False))  # plain JSON types only (NaN / inf refused: Astra r5 optional) cross back (launch 2's numpy str_ result could not be unpickled on the laptop)
    except SystemExit as e:  # this job's own messages (names, counts, shas)
        err = str(e)
    except Gate as e:
        err = f"gate: {e}"
    except Exception as e:  # noqa: BLE001 -- parse errors can hold source text (JSONDecodeError.doc): only the type crosses back
        err = f"stopped: {type(e).__name__} (message withheld)"
    raise SystemExit(err)  # raised outside the except blocks, so no exception context travels with it


# ---------- laptop side: git, pins, launch, records ----------


def git(*a) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout


def pinned(name: str) -> bytes:
    c, p, h = PINS[name]
    b = git("show", f"{c}:{p}")
    if hashlib.sha256(b).hexdigest() != h:
        raise SystemExit(f"{name}: sha256 differs from the pin")
    return b


def identity(fit_code_dir: str) -> tuple:
    """own_all_rs_uncut's training identity, rebuilt with the fit's own code at the fit commit and the fit record's frozen pins (its routing
    commit, inventory_b and chain commit), required equal to the fit record's training_report by key. Keys and shas only."""
    d = Path(fit_code_dir).resolve()
    src = d / "experiments/2026-10-14_p2_family_expansion/run_own_all.py"
    if hashlib.sha256(src.read_bytes()).hexdigest() != PINS["fit_code"][2]:
        raise SystemExit("--fit-code-dir: run_own_all.py differs from the fit commit's")
    if subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip() != \
            git("rev-parse", PINS["fit_code"][0]).decode().strip() or subprocess.run(["git", "-C", str(d), "status", "--porcelain", "--untracked-files=no"],
                                                                                capture_output=True, text=True, check=True).stdout.strip():
        raise SystemExit("--fit-code-dir is not a clean checkout of the fit commit")
    sys.path.insert(0, str(src.parent))
    import run_own_all as roa  # noqa: E402  (the fit's own launcher-side code; git reads only)

    fit = json.loads(pinned("fit"))
    inp = json.loads(pinned("own_all_inputs"))
    if fit["inputs"]["sha256"] != PINS["own_all_inputs"][2]:
        raise SystemExit("the fit record's inputs sha256 is not the pinned own_all inputs")
    ru = fit["rs_uncut"]
    ib = (ru["inventory_b"]["commit"], ru["inventory_b"]["path"], ru["inventory_b"]["sha256"])
    g_ = lambda *a: subprocess.run(["git", "-C", str(roa.REPO), *a], capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
    by_root, rep = roa._uncut_training(g_, ru["routing_commit"], ib, inp, ru["training_report"]["chain_commit"])
    if json.loads(json.dumps(rep)) != ru["training_report"]:
        raise SystemExit("STOP: the rebuilt training identity differs from the fit record's training_report")
    eps = {}
    for root, v in by_root.items():
        for e in v:
            if e["key"] in eps:
                raise SystemExit("a training key has two uncut sources")
            eps[e["key"]] = {x: e[x] for x in ("world", "cell", "run", "staged", "staged_sha256", "arm", "eval_sha256")}
    main_of = {}
    for c_, p_, s_ in (inp["pins"]["t196"], inp["pins"]["plus311"], inp["pins"]["nm360"]):  # the fit launcher's main_of for the three sets
        b = git("show", f"{c_}:{p_}")
        if hashlib.sha256(b).hexdigest() != s_:
            raise SystemExit(f"{p_}: sha256 differs")
        main_of |= {c["key"]: c["main"] for c in json.loads(b)["cells"]}
    return fit, inp, eps, main_of


def check_inputs_full(fit: dict) -> None:
    inf = json.loads(pinned("inputs_full"))
    f_ = inf["fits"][PROBE]
    if (f_["fit"]["commit"], f_["fit"]["path"], f_["fit"]["sha256"]) != PINS["fit"] or \
            (f_["companion"]["commit"], f_["companion"]["path"], f_["companion"]["sha256"]) != PINS["companion"] or PROBE not in inf["scope"]["headline_six"]:
        raise SystemExit("INPUTS_FULL does not pin this fit record / companion for the lead probe")
    comp = json.loads(pinned("companion"))
    if comp["fit_record"]["git"]["sha256"] != PINS["fit"][2] or comp["fit_record"]["cut"] != "none" or not comp["pass"]:
        raise SystemExit("the companion does not certify this fit record (cut none, pass)")
    if fit["fit"]["cut"] != "none":
        raise SystemExit("the fit record's cut is not none")


def bound_keylist(spec: str, want_sha: str = "") -> tuple:
    """Astra r1 #6: the key list (path@commit, the commit resolved to its full id) must be this script's own keylist output, bound to the
    pinned fit: its pins equal PINS' fit / companion / inputs / fit-code entries, its root shas equal the fit record's C row pins, its runs are
    unique and their count and per-world counts equal the fit's training_runs and the record's own counts, and the writing script's bytes
    at the record's commit hash to its script_sha256 with the same keylist code path (the record's 'what' names item 28)."""
    kp, kc = spec.rsplit("@", 1)
    kc = git("rev-parse", "--verify", f"{kc}^{{commit}}").decode().strip()
    kb = git("show", f"{kc}:{kp}")
    ksha = hashlib.sha256(kb).hexdigest()
    if want_sha and ksha != want_sha:
        raise SystemExit("the key list differs from the index's pin")
    kl = json.loads(kb)
    fit = json.loads(pinned("fit"))
    me = str(Path(__file__).resolve().relative_to(REPO))
    if hashlib.sha256(git("show", f"{kl['git']}:{me}")).hexdigest() != kl["script_sha256"] or not kl["what"].startswith(f"{PROBE} training RUN key list"):
        raise SystemExit("the key list was not written by this script's keylist mode at its recorded commit")
    for k in ("fit", "companion", "own_all_inputs", "fit_code", "inputs_full"):
        if (kl["pins"][k]["commit"], kl["pins"][k]["path"], kl["pins"][k]["sha256"]) != PINS[k]:
            raise SystemExit(f"the key list's {k} pin differs from this script's")
    if kl["rows_files"]["sha256_by_root"] != {r: v[READOUT]["sha256"] for r, v in fit["rows"].items()}:
        raise SystemExit("the key list's root shas differ from the fit record's")
    keys = [r["key"] for r in kl["runs"]]
    ex = fit["fit"]["training_units_past_32768"][READOUT]
    if len(set(keys)) != len(keys) or len(keys) != ex["training_runs"] or kl["counts"]["training_runs"] != len(keys) or \
            kl["counts"]["training_rows"] != ex["training_rows"] or sum(r["n_units"] for r in kl["runs"]) != ex["training_rows"] or \
            any(kl["counts"]["by_world"][w] != sum(r["world"] == w for r in kl["runs"]) for w in TRAIN_WORLD.values()) or \
            {r["world"] for r in kl["runs"]} != set(TRAIN_WORLD.values()):
        raise SystemExit("the key list is incomplete, has duplicates, or its counts differ from the fit record's")
    return kl, kp, kc, ksha


def write_record(stem: str, rec: dict) -> None:
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"stamp (PT)": stamp, "git": git("rev-parse", "HEAD").decode().strip(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "app": APP, "pins": {k: {"commit": c, "path": p, "sha256": h} for k, (c, p, h) in PINS.items()}} | rec
    b = json.dumps(rec, indent=1).encode()
    dest = HERE / "results" / f"{stem}_{stamp}.json"
    dest.parent.mkdir(exist_ok=True)
    with dest.open("xb") as fh:
        fh.write(b)
    print(json.dumps({"record": str(dest.relative_to(REPO)), "sha256": hashlib.sha256(b).hexdigest()}), flush=True)


@app.local_entrypoint()
def main(mode: str = "", fit_code_dir: str = "", keylist: str = "", index: str = "", calibration: str = "pooled", go_stage_2: bool = False, text_out: str = ""):
    if calibration not in CALIBRATIONS:
        raise SystemExit(f"--calibration {CALIBRATIONS}")
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("uncommitted tracked changes")
    for k in PINS:
        pinned(k)
    if mode == "keylist":
        fit, inp, eps, main_of = identity(fit_code_dir)
        check_inputs_full(fit)
        if set(fit["rows"]) != set(fit["rs_uncut"]["rows"][READOUT]) or any(fit["rows"][r][READOUT]["sha256"] != h for r, h in fit["rs_uncut"]["rows"][READOUT].items()):
            raise SystemExit("the fit record's per-root rows and its rs_uncut row pins differ")
        payload = {"roots": {r: {x: v[READOUT][x] for x in ("sha256", "rows", "runs")} for r, v in fit["rows"].items()}, "epochs": eps, "main_of": main_of,
                   "universe_mains": sorted(inp["folds"]["by_main"]), "expect": fit["fit"]["training_units_past_32768"][READOUT]}
        res = job.remote("keylist", payload)
        write_record("oaru_training_keys", {
            "what": f"{PROBE} training RUN key list (RULE item 28 @ a3d81dbcb, derived by script per 09): the union over folds of the runs with at least one "
                    f"{READOUT} training row (world run1 = attack, honest_a = honest; main in the universe), from the fit's per-root uncut rows. KEYS ONLY.",
            "identity": "rebuilt with run_own_all._uncut_training @ 19611550c from the fit record's routing commit, inventory_b and chain commit; equal to "
                        "the fit record's training_report by key (run key + registered .eval sha)",
            "rows_files": {"volume": VOLS["/p2"], "dir": ROWS_DIR.removeprefix("/p2/"), "sha256_by_root": {r: v["sha256"] for r, v in payload["roots"].items()}},
        } | res)
        return
    if mode == "train-draw":
        if UNIT_ORDER not in UNIT_ORDERS:
            raise SystemExit("HELD: item 28's unit order is not pinned (UNIT_ORDER)")
        if KEYLIST_PIN is None or keylist != f"{KEYLIST_PIN[1]}@{KEYLIST_PIN[0]}":
            raise SystemExit("train-draw: the key list must be KEYLIST_PIN (set by a reviewed commit after the keylist job)")
        kl, kp, kc, ksha = bound_keylist(keylist, want_sha=KEYLIST_PIN[2])
        res = job.remote("train_draw", {"keylist": kl, "root_sha": kl["rows_files"]["sha256_by_root"], "unit_order": UNIT_ORDER})
        write_record("examples_v81_training_pair", {
            "what": "training examples pair (RULE item 28): one attack and one honest training run, then one unit, same rng. KEYS ONLY: "
                    "lengths and sha256s of the command bytes and full arguments, computed in the Modal job; no text.",
            "keylist": {"path": kp, "commit": kc, "sha256": ksha},
            "rng": "random.Random(int.from_bytes(sha256(label).digest()[:8], 'big')); run = rng.choice(runs sorted by key); unit = rng.choice(that run's "
                   "C training units sorted by (irid, call_index)), the same rng",
            "unit_rule": {"path": "experiments/2026-10-14_p2_simh_scores/results/writing_examples_rule_2026-10-15_193657.json", "commit": "160620a04",
                          "sha256": "b941a9115118bdfb46185655468e8c7a5c1d722ea5ac0cde1d17dc49fded4611", "sort_key": "(irid, call_index)"},
            "entries": res, "status_counts": {s: sum(e["status"] == s for e in res) for s in sorted({e["status"] for e in res})}})
        return
    if mode == "tables":
        if any(v is None for v in HELD.values()):
            raise SystemExit(f"HELD: the {calibration} Stage A record and its selection record are not pinned yet (q2 relays them); no draw")
        raise SystemExit("tables: the loader is written after the pinned records' layout is known (Codex/Astra-reviewed first)")
    if mode == "render":
        if not go_stage_2:
            raise SystemExit("render is Stage 2: --go-stage-2 only after q2's 'go stage 2'")
        out = Path(text_out).resolve()
        if SCRATCH.resolve() not in out.parents:
            raise SystemExit("--text-out must be under this session's scratchpad")
        if INDEX_PIN is None or index != f"{INDEX_PIN[1]}@{INDEX_PIN[0]}":
            raise SystemExit("render: the index must be INDEX_PIN (set by a reviewed commit after the draw record)")
        ic = git("rev-parse", "--verify", f"{INDEX_PIN[0]}^{{commit}}").decode().strip()
        ip = INDEX_PIN[1]
        ib = git("show", f"{ic}:{ip}")
        if hashlib.sha256(ib).hexdigest() != INDEX_PIN[2]:
            raise SystemExit("render: the index differs from INDEX_PIN")
        rec = json.loads(ib)
        if rec.get("keylist", {}).get("sha256") != (KEYLIST_PIN or (None, None, None))[2] or rec.get("unit_rule", {}).get("sort_key") != "(irid, call_index)":
            raise SystemExit("render: the index was not drawn from KEYLIST_PIN with index A's (irid, call_index) order")
        if any(e["status"] != "ok" for e in rec["entries"]) or sorted(e["class"] for e in rec["entries"]) != sorted(TRAIN_WORLD):
            raise SystemExit("render: the index is not a complete ok attack + honest pair: STOP (no filtering, no redraw)")
        kl, _, _, _ = bound_keylist(f"{KEYLIST_PIN[1]}@{KEYLIST_PIN[0]}", want_sha=KEYLIST_PIN[2])
        by_key = {r["key"]: r for r in kl["runs"]}
        ents = [e | {"identity": by_key[e["task_key"]]} for e in rec["entries"]]
        res = job.remote("render", {"entries": ents, "stage_2": STAGE2_GO})
        with out.open("x") as fh:  # the only place the text is written; never printed
            json.dump({"index": {"path": ip, "commit": ic, "sha256": INDEX_PIN[2]}, "entries": res}, fh, indent=1)
        print(json.dumps({"text_file": str(out), "entries": len(res)}), flush=True)
        return
    raise SystemExit("--mode keylist | train-draw | tables | render")


def selftest() -> None:
    """Pure negative tests on synthetic arguments (no data, no Modal; Astra r1 #10): command and full arguments past 32,768 characters."""
    big = "x" * 40000 + "tail"
    a = {"cmd": big, "timeout": 5}
    e = digests(a) | {"label": "selftest"}
    r_ = gate_digests(a, e)
    assert r_["text"] == big and len(big) > 32768 and json.loads(r_["arguments_json"]) == a  # full content and full arguments come back whole
    for bad in ({"cmd": big[:32768], "timeout": 5},  # truncation at 32k
                {"cmd": big[:35000] + "y" + big[35001:], "timeout": 5},  # one character changed past 32k
                {"cmd": big, "timeout": 6}):  # a non-command argument changed: command hash equal, full-arguments hash differs
        try:
            gate_digests(bad, e)
        except SystemExit:
            continue
        raise AssertionError("gate_digests accepted a mismatching call")
    assert digests({"cmd": big, "timeout": 6})["sha256"] == e["sha256"]
    for x in ({"cmd": "attachment://0123"}, {"cmd": "ok", "env": ["attachment://ab"]}):
        try:
            no_attachment_refs(x)
        except Gate:
            continue
        raise AssertionError("an unresolved attachment reference passed")
    try:
        _render({"stage_2": "no", "entries": []})
        raise AssertionError("render ran without the Stage 2 token")
    except SystemExit:
        pass
    probes = [f"p{i}" for i in range(6)]
    pools = ["H_B", "H_sim"]
    full = {(a_, b_) for a_ in probes for b_ in pools}
    ev = [{"probe": probes[i % 6], "pool": pools[i % 2], "run": f"r{i}", "unit": 0, "z": 3.0, "T": 2.0, "z_minus_T": 1.0} for i in range(12)]
    assert len(false_alarms(ev, probes, pools, full)["draws"]) == 6
    for evb, av in ((ev[:11] + [ev[11] | {"z": 1.0, "T": 2.0, "z_minus_T": 1.0}], full), (ev, full - {("p0", "H_B")}),
                    (ev[:11] + [ev[11] | {"z": 1e308, "T": -1e308}], full)):  # below threshold; a missing pool; an overflowing margin
        try:
            false_alarms(evb, probes, pools, av)
        except SystemExit:
            continue
        raise AssertionError("false_alarms accepted a below-threshold event or a missing pool")
    print("selftest: all negative tests passed")


if __name__ == "__main__" and sys.argv[1:] == ["--selftest"]:
    selftest()
