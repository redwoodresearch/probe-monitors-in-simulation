"""Prerequisite check (PLAN.md 5): does every listed call join to a stored call_end reading? IDs and numbers only, $0.

Reads local copies of the attack-run extraction metas (each extraction's meta count must equal its complete.json epochs),
keeps the runs the 196-task scorer keeps (redrawn and rerun attempts skipped), and joins each key row on
(world, irid, call_index) to the meta's call_end entry with a stored row. Also counts the stage-1 calls that the lists'
join (scorer `call` = call_index + 1) left unscored although the scorer's record exists under the command entry's own
`call` number (the extraction metas number entries within a request, so the two differ for zero-argument and
multi-call requests).

  uv run --no-project --python 3.12 --with pandas --with numpy python check_joins.py --metas <acts_meta root> --key <key_stage1.csv> --inputs <dir>
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import prepare_stage1 as P

EXTRACTIONS = {"real": ("attack", ["2026-10-09_rerun_strict_segends_v1", "2026-10-10_scaleup_canary_segends_v1", "2026-10-10_scaleup_l2_segends_v1",
                                   "2026-10-10_scaleup_topup_segends_v1"]),
               "run2": ("attack", ["2026-10-10_second_attack_segends_v1"]),
               "sim": ("sim_attack@september", ["2026-10-09_rerun_strict_segends_v1", "2026-10-10_scaleup_canary_segends_v1",
                                                "2026-10-10_scaleup_l2_segends_v1", "2026-10-10_scaleup_topup_segends_v1"])}


def entries(metas: str, out: dict | None = None):
    """(world, irid, call_index) -> the command entry's `call`, and -> its call_end row, from the kept runs' metas."""
    root, out = Path(metas), out if out is not None else {"extractions": {}}
    command, call_end = {}, {}
    for world, (arm, xs) in EXTRACTIONS.items():
        for x in xs:
            files = sorted((root / x / "meta").glob("*/*.json"))
            done = json.loads((root / x / "complete.json").read_text())["epochs"]
            out["extractions"][x] = {"metas": len(files), "complete_epochs": done}
            if len(files) != done:
                raise SystemExit(f"{x}: {len(files)} metas vs {done} epochs")
            for f in sorted((root / x / "meta" / arm).glob("*.json")):
                m = json.loads(f.read_text())
                if m.get("redrawn_because") or m.get("attempt_kind") == "rerun":
                    continue
                for c in m["calls"]:
                    if "call_index" not in c:
                        continue
                    k = (world, c["irid"], c["call_index"])
                    if c.get("kind", "command") == "command":
                        command[k] = c["call"]
                    elif c.get("kind") == "call_end" and c["row"] is not None:
                        call_end[k] = (x, m["chunk"], m["row_start"] + c["row"])
    return command, call_end


def rescore(C: pd.DataFrame, command: dict, inputs: str) -> pd.DataFrame:
    """C with the monitor's record joined through the command entry's own `call` (the corrected join): scored, y, flag."""
    D = json.loads((Path(inputs) / "draw0.json").read_text())
    M = pd.DataFrame(D["meta"])
    flag = np.full(len(M), np.nan)
    for _, world, cell in P.ARMS.values():
        pc = D["draws"][0]["cells"][cell]["per_call"]
        flag[np.asarray(pc["row"])] = np.asarray(pc["flag_frozen"], dtype=float)
    M["flag"] = flag
    C = C.drop(columns=["y", "flag", "row", "task", "scored", "flagged"]).assign(call=[command.get((w, i, c), -1) for w, i, c in zip(C.world, C.irid, C.call_index)])
    C = C.merge(M.assign(row=np.arange(len(M)))[["world", "irid", "call", "y", "row", "flag"]], on=["world", "irid", "call"], how="left", validate="many_to_one")
    C["scored"] = C.row.notna() & C.flag.notna()
    C["flagged"] = C.scored & (C.flag == 1)
    return C


def main(metas: str, key: str, inputs: str):
    out = {"extractions": {}}
    command, call_end = entries(metas, out)
    K = pd.read_csv(key, usecols=["world", "side", "irid", "call_index", "call", "source"])
    K["joined"] = [(w, i, c) in call_end for w, i, c in zip(K.world, K.irid, K.call_index)]
    K["call_matches_command_entry"] = [command.get((w, i, c)) == cc for w, i, c, cc in zip(K.world, K.irid, K.call_index, K.call)]
    out["listed_rows_by_side_world"] = {f"{s} / {w}": {"rows": len(g), "joined_to_call_end": int(g.joined.sum()), "unjoined": int((~g.joined).sum()),
                                                       "call_number_matches": int(g.call_matches_command_entry.sum())}
                                        for (s, w), g in K.groupby(["side", "world"])}
    # stage-1 calls the lists' join left unscored although the scorer's record exists under the command entry's number
    C = P.load(inputs)
    sp = json.loads(P.SPLIT_FILE.read_text())
    S = C[C.task_key.isin([t for t, v in sp["tasks"].items() if v["stage1"]])].copy()
    D = json.loads((Path(inputs) / "draw0.json").read_text())
    M = pd.DataFrame(D["meta"])
    have = set(zip(M.world, M.irid, M.call))
    S["true_call"] = [command.get((w, i, c), -1) for w, i, c in zip(S.world, S.irid, S.call_index)]
    S["scorer_record_true"] = [(w, i, c) in have for w, i, c in zip(S.world, S.irid, S.true_call)]
    lost = S[~S.scored & S.scorer_record_true]
    yt = dict(zip(zip(M.world, M.irid, M.call), M.y))
    out["stage1_calls_lost_to_call_numbering"] = {"count": len(lost), "by_world_side": {f"{s} / {w}": int(n) for (s, w), n in
                                                  Counter(zip(lost.task_key.map(lambda t: sp["tasks"][t]["side"]), lost.world)).items()},
                                                  "impactful": int(sum(yt[(w, i, c)] for w, i, c in zip(lost.world, lost.irid, lost.true_call))),
                                                  "target_touching": int(lost.touch.sum())}
    u = S[~S.scored & S.touch]
    out["unscored_target_touching"] = {"count": len(u), "replay_reader_excluded": int(u.q2_excluded.sum()),
                                       "with_call_end_row": int(sum((w, i, c) in call_end for w, i, c in zip(u.world, u.irid, u.call_index)))}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metas", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--inputs", required=True)
    main(**vars(ap.parse_args()))
