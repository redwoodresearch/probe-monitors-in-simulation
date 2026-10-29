"""By-key check of the sim-honest scorer's cutfull grammar against the pinned (a'') inventory and d9's naming / routing spec (pure string
and JSON checks; no Modal, no activation, no score). For every inventory run: its cutfull folder passes d9's scorer regex (naming v1
routed_folder without the gate group) exactly when it passes 09's CUTFULL regex; its gate folder is refused; the arm token is registered
(no "@"); sim-honest rows are admitted by this scorer's grammar (flag ON), derive to their original up to _try, and the original is
authoritative (a tracked step-3 record whose committed message holds the key); other worlds are outside this scorer (S_new adapter).
Writes results/cutfull_grammar_check_<stamp>.json.

  python3 -I check_cutfull_grammar.py
"""

import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
INVENTORY = ("b8a39f329", "experiments/2026-10-15_cut_sensitivity/results/inventory_v3_2026-10-15.json", "19306ffa7c42ab35a62715c1a18822a43c6273492e5ede9b2f62cd9e428ad9a3")
NAMING = ("f7e47563f", "experiments/2026-10-14_q3_overnight_chain/cutfull/cutfull_naming_v1.json")
ROUTING = ("f7e47563f", "experiments/2026-10-14_q3_overnight_chain/cutfull/cutfull_routing_v1.json", "cdcc2ac76c654ca3d19ac9037ec7abd0d404127faa3e54963b9d52126010d1c4")
git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout  # noqa: E731


def pinned(c, p, h=None):
    b = git("show", f"{c}:{p}")
    if h and hashlib.sha256(b).hexdigest() != h:
        raise SystemExit(f"{p} @ {c}: sha256 differs from its pin")
    return json.loads(b), {"commit": git("rev-parse", c).decode().strip(), "path": p, "sha256": hashlib.sha256(b).hexdigest()}


def scorer_grammar():
    """The scorer's own grammar code (from _FOLDER_RE_BASE to FOLDER_RE), executed alone: the scorer module imports modal."""
    src = (HERE / "score_step3_simh_batch.py").read_text()
    i, j = src.index("_FOLDER_RE_BASE = "), src.index("FOLDER_RE = _Grammar()")
    ns = {"re": re}
    exec(src[i:j] + "FOLDER_RE = _Grammar()\n", ns)
    return ns


def main():
    inv, inv_pin = pinned(*INVENTORY)
    naming, naming_pin = pinned(*NAMING)
    routing, routing_pin = pinned(*ROUTING)
    d9 = re.compile(naming["regex"]["routed_folder"].replace("(?P<gate>_gate)?", ""))
    if "gate group absent" not in routing["scorer_regex"] or "(?P<gate>" in d9.pattern:
        raise SystemExit("d9's scorer regex is not naming v1 routed_folder minus the gate group")
    g = scorer_grammar()
    g["CUTFULL"]["enabled"] = True
    strip_try = lambda f: re.sub(r"_try[0-9]+__", "__", f)  # noqa: E731
    tracked = set(git("ls-files", "--", str((HERE / "results").relative_to(REPO))).decode().split())
    msgs = {hashlib.sha256(q.read_bytes()).hexdigest(): q for q in (HERE / "results" / "simh_msgs").glob("msg_*.json") if str(q.relative_to(REPO)) in tracked}
    okeys = {}
    for rp in sorted((HERE / "results").glob("simh_batch_*.json")):
        r = json.loads(rp.read_text()) if str(rp.relative_to(REPO)) in tracked else {}
        if r.get("message_sha256") in msgs:
            m = json.loads(msgs[r["message_sha256"]].read_text())
            for fo in r.get("result") or {}:
                okeys.setdefault(fo, set()).update(x["key"] for x in m["epochs"] + [y for q in m.get("provenance", []) for y in q.get("excluded_by_staging", [])]
                                                   if x["folder"] == fo)
    fails, n = {}, Counter()
    for r in inv["runs"]:
        cf, gf, of, k, w = r["uncut_planned"]["folder"], r["uncut_planned"]["gate_folder"], r["original_folder"], r["key"], r["world"]
        checks = {"d9_regex_eq_09_regex": bool(d9.fullmatch(cf)) == bool(re.fullmatch(g["CUTFULL"]["regex"], cf)) and bool(d9.fullmatch(cf)),
                  "gate_folder_refused": not g["FOLDER_RE"].fullmatch(gf) and not re.fullmatch(d9, gf),
                  "registered_arm_token": "@" not in cf and "@" not in of and cf.split("__")[1] == of.split("__")[1],
                  "derives_to_original_up_to_try": strip_try(g["orig_name"](cf)) == strip_try(of)}
        if w == "sim_honest":
            m_ = g["FOLDER_RE"].fullmatch(cf)
            checks |= {"simh_grammar_admits": bool(m_) and m_.group(1) == g["_FOLDER_RE_BASE"].fullmatch(of).group(1),
                       "original_authoritative_by_key": k in okeys.get(of, set())}
        else:
            checks["outside_simh_scorer"] = not g["FOLDER_RE"].fullmatch(cf)
        n[w] += 1
        for c, ok in checks.items():
            n[f"{c}:{'pass' if ok else 'FAIL'}"] += 1
            if not ok:
                fails.setdefault(c, []).append(k)
    folders = sorted({r["uncut_planned"]["folder"] for r in inv["runs"]})
    out = {"label": "by-key check: sim-honest scorer cutfull grammar vs inventory v3 and d9's naming / routing spec (strings only)",
           "stamp (PT)": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S"), "git": git("rev-parse", "HEAD").decode().strip(),
           "scorer_sha256": hashlib.sha256((HERE / "score_step3_simh_batch.py").read_bytes()).hexdigest(),
           "pins": {"inventory": inv_pin, "naming": naming_pin, "routing": routing_pin}, "d9_scorer_regex": d9.pattern,
           "runs": len(inv["runs"]), "cutfull_folders": len(folders), "folders_with_try": sum("_try" in f for f in folders),
           "counts": dict(sorted(n.items())), "failures_by_check": {c: {"n": len(v), "keys": v} for c, v in fails.items()}, "pass": not fails}
    p = HERE / "results" / f"cutfull_grammar_check_{out['stamp (PT)']}.json"
    p.open("x").write(json.dumps(out, indent=1))
    print(p.name, json.dumps({k: out[k] for k in ("runs", "cutfull_folders", "folders_with_try", "counts", "pass")}, indent=1))
    print({c: v["n"] for c, v in out["failures_by_check"].items()})
    if fails:
        raise SystemExit(f"FAIL: {len(fails)} check(s) failed by key (see {p.name})")


if __name__ == "__main__":
    main()
