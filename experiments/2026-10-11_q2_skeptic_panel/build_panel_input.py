"""Skeptic panel input from the status page's answer lines (the viewer's verbatim export): per thread, the question
(header), the claimed status and plain line (SETTLED), the numbers (ANSWER, verbatim, with its inline source labels),
and the checks (the ANSWER's clauses that mention a real-vs-real control, a null, a planted check or a control,
extracted verbatim). Nothing is added or rewritten.

  python3 experiments/2026-10-11_q2_skeptic_panel/build_panel_input.py <answer_lines.txt> <version tag> [IDs still changing, comma-separated]
"""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
KEYS = re.compile(r"real-vs-real|two real|real run 2|null|planted|plant|control|shuffl", re.I)


def main(src: Path, tag: str, skip: list[str]):
    text = src.read_text(); qs = []; skipped = []
    for sec in re.split(r"\n(?=## )", text):
        if not sec.startswith("## "):
            continue
        head = sec.splitlines()[0][3:].strip()
        settled = re.search(r"^SETTLED:\s*(.+)$", sec, re.M); answer = re.search(r"^ANSWER:\s*(.+)$", sec, re.M)
        if not (settled and answer):
            raise ValueError(f"{head}: missing SETTLED or ANSWER")
        qid = head.split()[0]
        if qid in skip:
            skipped.append(qid)
            continue
        clauses = [c.strip() for c in re.split(r"(?<=[.;])\s+", answer.group(1)) if KEYS.search(c)]
        state = re.search(r"\s(open|closed(?:, merged| for tonight)?)$", head)
        if not state:
            raise ValueError(f"{head}: no thread state at the end of the header")
        qs.append({"id": qid, "question": head[: state.start()].strip(), "status": state.group(1), "line": settled.group(1),
                   "numbers": [answer.group(1)], "checks": clauses or ["(the answer text names no control, null or planted check)"]})
    out = {"source": str(src), "source sha256": hashlib.sha256(src.read_bytes()).hexdigest(), "version": tag,
           "note": "line = the page's SETTLED text; numbers = its ANSWER text verbatim; checks = ANSWER clauses naming a control, null or planted check",
           "skipped (still changing at assembly)": skipped, "questions": qs}
    dest = HERE / "results" / f"panel_input_{tag}.json"; dest.parent.mkdir(exist_ok=True)
    with open(dest, "x") as fh:
        fh.write("{\n" + ",\n".join(f"{json.dumps(k)}: {json.dumps(v)}" for k, v in out.items()) + "\n}\n")  # one top-level key per line
    print(f"{len(qs)} questions: {[q['id'] for q in qs]}"); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], sys.argv[3].split(",") if len(sys.argv) > 3 else [])
