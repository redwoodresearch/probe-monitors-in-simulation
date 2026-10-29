"""HTML for the transcript grid: one row per step, one column per run, each segment a card with a score slot.

A step is one agent turn and the tool outputs that answer it, so step n of every run sits on the same row (the
sims repo's comparison view does the same). Segments are the reader's: one card per agent turn and per tool
output, numbered by (step, action) as in the activation extraction, which is also how probe scores are keyed.
Every piece of transcript text is escaped.
"""

import json
from html import escape

# Arm -> (column title, colour, one-line description). Real runs are blue, the simulated run orange.
ARM_STYLE = {
    "real-attack-e1": ("Real attack · run 1", "#2563eb", "Tools really executed in the environment"),
    "sim-attack": ("Simulated attack", "#ea580c", "Tool outputs written by a simulator model, not executed"),
    "real-attack-e2": ("Real attack · run 2", "#0891b2", "A second real run of the same task: how much two real runs differ by chance"),
    "real-honest": ("Real honest", "#16a34a", "Same main task, no side task, tools really executed"),
}
UNLABELLED = {"cmd", "code"}  # a bash command or python code: the tool badge already says what it is
ALWAYS_BLOCK = {"file_text", "new_str", "old_str", "insert_text", "content"}

CSS = """
<style>
.pg{--bd:rgba(128,128,128,.28);--soft:rgba(128,128,128,.07);--mut:rgba(128,128,128,.95);font-size:13px;line-height:1.4}
.pg .grid{display:grid;gap:0 10px;align-items:start}
.pg .hd{position:sticky;top:3.75rem;z-index:2;padding:6px 8px;font-weight:600;border-bottom:3px solid var(--c);
 background:var(--bg)}
.pg .hd small{display:block;font-weight:400;color:var(--mut);font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pg .step{font-size:11px;color:var(--mut);padding-top:10px;text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.pg .cell{border-top:1px solid var(--bd);padding:4px 0;min-width:0}
.pg .seg{border:1px solid var(--bd);border-left:6px solid var(--score,var(--bd));border-radius:5px;margin:4px 0;padding:3px 8px;min-width:0}
.pg .seg.agent{background:var(--soft)}
.pg .seg.gone,.pg .gone{opacity:.38}
.pg .sh{display:flex;flex-wrap:wrap;gap:6px;align-items:center;font-size:11px;color:var(--mut)}
.pg .sh b{color:inherit;font-weight:600;color:var(--fg,inherit)}
.pg .tool{font-family:ui-monospace,Menlo,monospace;font-size:11px;padding:0 6px;border-radius:9px;background:rgba(128,128,128,.18);color:inherit}
.pg .tool.text_editor{background:rgba(147,51,234,.18)}
.pg .tool.bash{background:rgba(37,99,235,.15)}
.pg .sc{margin-left:auto;font-family:ui-monospace,Menlo,monospace;font-size:11px}
.pg pre{white-space:pre-wrap;word-break:break-word;font-size:11.5px;margin:3px 0;padding:4px 6px;border-radius:4px;
 background:rgba(128,128,128,.08);max-height:var(--maxh,340px);overflow:auto}
.pg .call{margin:4px 0}
.pg .k{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:var(--mut)}
.pg .text{white-space:pre-wrap;margin:3px 0}
.pg .cut{border-top:2px dashed #dc2626;color:#dc2626;font-size:11px;font-weight:600;margin:4px 0;padding-top:2px}
.pg .end{color:var(--mut);font-size:11px;font-style:italic;padding:6px 0}
.pg .empty{color:var(--mut);font-size:12px;padding:8px 0}
.pg details summary{cursor:pointer;font-size:12px;color:var(--mut)}
/* Probe readouts: content-end tokens in green, pre_call in purple, template tokens dashed grey (sensitivity only). */
.pg .ro-output_end,.pg .ro-call_end{background:rgba(16,185,129,.35);outline:1px solid rgba(16,185,129,.9);border-radius:2px}
.pg .rob{display:inline-block;font-family:ui-monospace,Menlo,monospace;font-size:10px;line-height:1.3;padding:0 4px;
 margin:0 2px;border-radius:3px;cursor:help;vertical-align:baseline}
.pg .rob.ro-output_end,.pg .rob.ro-call_end{background:rgba(16,185,129,.25);outline:none;color:inherit}
.pg .rob.ro-pre_call{background:rgba(147,51,234,.22)}
.pg .rob.ro-call_marker,.pg .rob.ro-output_marker{border:1px dashed rgba(128,128,128,.8);color:var(--mut)}
.pg .seg.dropped{border-style:dashed;opacity:.75}
.pg details.raw pre{background:rgba(234,179,8,.10)}
.pg .ws{color:#ca8a04;font-size:10px;margin-left:2px}
</style>
"""


def group_steps(messages: list[dict]) -> tuple[list[int], list[list[int]]]:
    """Message indices before the first agent turn, and per step the agent turn plus what follows it."""
    prompt, steps = [], []
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            steps.append([i])
        elif steps:
            steps[-1].append(i)
        else:
            prompt.append(i)
    return prompt, steps


def _block(text: str) -> str:
    return f"<pre>{escape(text)}</pre>"


def _marked(text: str, marks: list[tuple]) -> str:
    """`text`, escaped, with marks: (start, end, cls, title) wraps text[start:end] in a span (classes stack where
    marks overlap, e.g. a readout token past the cut); (pos, html) inserts html before character pos."""
    edges = sorted({0, len(text)} | {m[0] for m in marks} | {m[1] for m in marks if len(m) == 4})
    out = []
    for a, b in zip(edges, edges[1:] + [None]):
        out += [m[1] for m in marks if len(m) == 2 and m[0] == a]
        if b is None:
            break
        on = [m for m in marks if len(m) == 4 and m[0] <= a and b <= m[1]]
        piece = escape(text[a:b])
        if on:
            title = "; ".join(m[3] for m in on if m[3])
            title = f' title="{escape(title)}"' if title else ""
            piece = f'<span class="{" ".join(m[2] for m in on)}"{title}>{piece}</span>'
        out.append(piece)
    return "".join(out)


CUT = '<div class="cut">✂ A 32k-token reader window ends here</div>'
READOUT_ABOUT = {
    "output_end": "the output's last content token",
    "call_end": "the call's last content token",
    "pre_call": "the <tool_call> token opening the call: sees only what came before it",
    "call_marker": "the closing </tool_call> template token: sensitivity check only",
    "output_marker": "the closing </tool_response> template token: sensitivity check only",
}


def _cut_marks(text: str, at: int) -> list[tuple]:
    return [(at, CUT), (at, len(text), "gone", "")]


def _readout(r: dict, view: dict) -> tuple[str, str]:
    """(span class, hover title) for one readout, with its probe score when one is loaded."""
    score = (view.get("ro_scores") or {}).get((r["step"], r["action"], r["kind"]))
    title = (f'Probe reads here: {r["kind"]}, {READOUT_ABOUT[r["kind"]]} (token {r["token"]:,}: {r["token_str"]!r})'
             + (f"; score {score:.2f}" if score is not None else "")
             + ("" if r["kept"] else "; past the 32k window, so not read"))
    return f'ro-{r["kind"]}' + ("" if r["kept"] else " gone"), title


def _badge(r: dict, view: dict, label: str) -> str:
    cls, title = _readout(r, view)
    score = (view.get("ro_scores") or {}).get((r["step"], r["action"], r["kind"]))
    return f'<span class="rob {cls}" title="{escape(title)}">{escape(label)}{f" {score:.2f}" if score is not None else ""}</span>'


def _span_marks(r: dict, view: dict, start: int, end: int) -> list[tuple]:
    """A content-end readout over characters [start, end): highlighted, or a thin badge where it covers none."""
    cls, title = _readout(r, view)
    if end > start:
        score = (view.get("ro_scores") or {}).get((r["step"], r["action"], r["kind"]))
        return [(start, end, cls, title)] + ([(end, _badge(r, view, "▸"))] if score is not None else [])
    return [(start, _badge(r, view, "▸"))]


def _raw_html(sample: dict, i: int, view: dict) -> tuple[str, str]:
    """(before, after) the card in the raw view: the messages the reader dropped, and the original message."""
    if not view.get("raw"):
        return "", ""
    raw = sample["raw"][i]
    dump = lambda o: escape(json.dumps(o, indent=1, ensure_ascii=False))
    drop = lambda o: f'<div class="seg dropped"><div class="sh"><b>dropped by the reader</b></div><pre>{dump(o)}</pre></div>'
    orig = dump(raw["orig"]) if raw["orig"] is not None else "(none: the run had no system message; the reader adds one)"
    after = f'<details class="raw" open><summary>raw message</summary><pre>{orig}</pre></details>'
    return "".join(map(drop, raw["dropped"])), after + "".join(map(drop, raw.get("dropped_after", [])))


def _args(arguments, part: dict | None = None, marker: str = "", ro: dict | None = None) -> str:
    """A tool call's arguments. With `part` (the 32k cut inside this call: the argument it falls in and how many
    characters of it are read), that argument is split at the cut and every later one is dimmed. `ro` adds marks
    per argument (readouts)."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return _block(arguments)
    if not isinstance(arguments, dict):
        return _block(json.dumps(arguments, indent=1))
    out, line, state = [], [], "read"  # in rendered order (the template dumps them as given); short ones share a line
    for k, v in arguments.items():
        state = "cut" if part and k == part["arg"] else "gone" if state != "read" else "read"
        text = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)  # as the chat template serializes it
        marks = list((ro or {}).get(k, []))
        in_key = state == "cut" and part.get("in_key")
        if state == "cut" and not in_key:  # for a non-string value, chars counts its serialized JSON
            marks += [(part["chars"], marker), (part["chars"], len(text), "gone", "")]
        html, name = _marked(text, marks), escape(k)
        if in_key:  # the reader stops inside the argument's name
            name, html = f'{marker}<span class="gone">{name}</span>', f'<span class="gone">{html}</span>'
        gone = ' class="gone"' if state == "gone" else ""
        if isinstance(v, str) and (k in UNLABELLED or k in ALWAYS_BLOCK or "\n" in v or len(v) > 90):
            out += [f'<div>{" · ".join(line)}</div>'] if line else []
            line = []
            label = (marker if in_key else "") if k in UNLABELLED else f'<div class="k">{name}</div>'
            out.append(f"<div{gone}>{label}<pre>{html}</pre></div>")
        else:
            line.append(f'<span{gone}><span class="k">{name}</span> {html}</span>')
    return "".join(out + ([f'<div>{" · ".join(line)}</div>'] if line else []))


def _score_slot(score: float | None, scale: tuple[float, float] | None) -> tuple[str, str]:
    """(style for the card's left bar, label). Empty until a probe is picked."""
    if scale is None:
        return "", ""
    if score is None:
        return "", '<span class="sc">no score</span>'
    lo, hi = scale
    alpha = min(max((score - lo) / (hi - lo) if hi > lo else 0.5, 0.0), 1.0)
    return f' style="--score:rgba(220,38,38,{0.12 + 0.88 * alpha:.2f})"', f'<span class="sc">{score:.2f}</span>'


def _seg_card(sample: dict, i: int, seg: dict | None, view: dict) -> str:
    m, cut = sample["messages"][i], sample.get("cut")
    cls = " gone" if cut is not None and i > cut["message"] else ""
    key = (seg["step"], seg["action"]) if seg else None
    style, label = _score_slot((view.get("scores") or {}).get(key), view.get("scale")) if seg else ("", "")
    if seg and view.get("scale") is not None and seg["n_tokens"] == 0:
        label = '<span class="sc">empty, not scored</span>'  # zero tokens: the extraction pools nothing here
    ntok = f'{seg["n_tokens"]:,} tok' if seg else ""
    here = cut is not None and i == cut["message"]
    shown = {"output_end", "call_end", "pre_call"} | ({"call_marker", "output_marker"} if view.get("markers") else set())
    ros = [r for r in sample.get("readouts", []) if r["message"] == i and r["kind"] in shown] if view.get("readouts") else []
    # A template-token readout is shown only beside its content-token readout (AGENTS.md); an empty output has none.
    content = {(r["step"], r["action"]) for r in ros if r["kind"] in ("call_end", "output_end")}
    ros = [r for r in ros if r["kind"] not in ("call_marker", "output_marker") or (r["step"], r["action"]) in content]
    before, after = _raw_html(sample, i, view)
    if m["role"] == "assistant":
        calls = m.get("tool_calls") or []
        head = f'<div class="sh"><b>Agent</b> step {seg["step"] + 1} · {ntok}{label}</div>'
        # With a cut in this turn: the call it falls in is split inside its arguments (`part`), or, when the cut
        # is before a call's arguments, the marker goes before that call; everything after is dimmed.
        cut_call = (cut["call"] if cut["call"] is not None else -1) if here else len(calls)
        part = cut["part"] if here else None
        marker = (f'<div class="cut">✂ A 32k-token reader window ends here ({cut["tokens_seen"]:,} of '
                  f'{seg["n_tokens"]:,} tokens of this turn)</div>') if here else ""
        text_chars = cut.get("text_chars") if here and cut_call == -1 else None
        body = marker if cut_call == -1 and text_chars is None else ""
        if m["content"].strip():  # the cut may fall inside the turn's own text: split it there
            t = m["content"]
            marks = [(text_chars, marker), (text_chars, len(t), "gone", "")] if text_chars is not None else []
            body += f'<div class="text{" gone" if cut_call == -1 and text_chars is None else ""}">{_marked(t, marks)}</div>'
        for n, c in enumerate(calls):
            name = c["function"]["name"]
            mine = [r for r in ros if r["call"] == n]
            split = part is not None and n == cut_call
            pre = "".join(_badge(r, view, "◆ pre_call") for r in mine if r["kind"] == "pre_call")
            if n == cut_call and not split:  # the cut is before this call's arguments: a read opener stays above it
                body += "".join(_badge(r, view, "◆ pre_call") for r in mine if r["kind"] == "pre_call" and r["kept"]) + marker
                pre = "".join(_badge(r, view, "◆ pre_call") for r in mine if r["kind"] == "pre_call" and not r["kept"])
            gone = " gone" if n > cut_call or (n == cut_call and not split) else ""
            ro_args, tail = {}, ""
            for r in mine:
                args = c["function"]["arguments"]
                if r["kind"] == "call_end" and isinstance(args, dict) and r["arg"] in args:
                    v = args[r["arg"]]
                    n_chars = len(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
                    ro_args.setdefault(r["arg"], []).extend(_span_marks(r, view, r["start"], n_chars if r["end"] is None else r["end"]))
                elif r["kind"] == "call_end":
                    tail += _badge(r, view, "▸ call_end")
                elif r["kind"] == "call_marker":
                    tail += _badge(r, view, "⟂ </tool_call>")
            body += (f'<div class="call{gone}">{pre}<span class="k">call {n + 1}</span> '
                     f'<span class="tool {escape(name)}">{escape(name)}</span>'
                     f'{_args(c["function"]["arguments"], part if split else None, marker, ro_args)}{tail}</div>')
        body += marker if here and cut_call == len(calls) else ""
        return f'{before}<div class="seg agent{cls}"{style}>{head}{body}{after}</div>'
    if m["role"] == "tool":
        name = m.get("tool_name") or "?"
        head = (f'<div class="sh"><span class="tool {escape(name)}">{escape(name)}</span> output of call '
                f'{seg["action"] + 1} · {ntok}{label}</div>')
        text = m["content"]
        marks = _cut_marks(text, cut["chars"]) if here else []
        for r in ros:
            if r["kind"] == "output_end":
                marks += _span_marks(r, view, r["start"], r["end"])
            else:
                marks.append((len(text), _badge(r, view, "⟂ </tool_response>")))
        trailing = len(text) - len(text.rstrip("\n"))
        ws = f'<span class="ws" title="trailing newlines, as the reader sees them">⏎×{trailing}</span>' if view.get("raw") and trailing else ""
        body = f"<pre>{_marked(text, marks)}{ws}</pre>" if text or marks else '<div class="end">(empty output)</div>'
        return f'{before}<div class="seg tool{cls}"{style}>{head}{body}{after}</div>'
    marks = _cut_marks(m["content"], cut["text_chars"]) if here else []  # a user message the window ends in
    return (f'{before}<div class="seg{cls}"><div class="sh"><b>{escape(m["role"])}</b></div>'
            f'<pre>{_marked(m["content"], marks)}</pre>{after}</div>')


def _cell(sample: dict | None, rows: list[int] | None, seg_of: dict, view: dict) -> str:
    if sample is None or rows is None:
        return '<div class="cell"></div>'
    return '<div class="cell">' + "".join(_seg_card(sample, i, seg_of.get(i), view) for i in rows) + "</div>"


def _prompt_cell(sample: dict | None, rows: list[int], view: dict) -> str:
    if sample is None:
        return '<div class="cell"></div>'
    raw = [_raw_html(sample, i, view) for i in rows]  # (dropped originals before, original after) per message
    parts = "".join(f'{before}<div class="k">{escape(sample["messages"][i]["role"])}</div>'
                    f'{_block(sample["messages"][i]["content"] or "(empty)")}{after}' for i, (before, after) in zip(rows, raw))
    return f'<div class="cell"><details><summary>System prompt and task, as the reader sees them</summary>{parts}</details></div>'


def grid(columns: list[dict], expand: bool = False, dark: bool = False, show: dict | None = None) -> str:
    """columns: [{"arm", "subtitle", "sample" (cached sample or None), "note" (shown when there is no sample),
    "scores" ({(step, action): score} or None), "ro_scores" ({(step, action, readout): score} or None), "scale"
    ((lo, hi) or None)}]. show: {"readouts", "markers" (template-token readouts), "raw"}, each a bool."""
    show = show or {}
    grouped, seg_maps = [], []
    for col in columns:
        s = col["sample"]
        grouped.append(group_steps(s["messages"]) if s else ([], []))
        seg_maps.append({seg["message"]: seg for seg in s["segments"]} if s else {})
    n_steps = max((len(g[1]) for g in grouped), default=0)
    out = [f'<div class="grid" style="grid-template-columns:52px repeat({len(columns)},minmax(0,1fr))">', '<div class="hd" style="--c:transparent"></div>']
    for col in columns:
        title, colour, about = ARM_STYLE[col["arm"]]
        out.append(f'<div class="hd" style="--c:{colour}" title="{escape(col.get("subtitle") or "")}">{escape(title)}'
                   f'<small>{escape(about)}</small></div>')
    out.append('<div class="step">prompt</div>')
    for col, (prompt, _) in zip(columns, grouped):
        out.append(_prompt_cell(col["sample"], prompt, show) if col["sample"] else f'<div class="cell empty">{escape(col.get("note") or "")}</div>')
    for k in range(n_steps):
        out.append(f'<div class="step">step {k + 1}</div>')
        for col, (_, steps), seg_of in zip(columns, grouped, seg_maps):
            if col["sample"] and k == len(steps):
                out.append('<div class="cell"><div class="end">transcript ends</div></div>')
            else:
                view = show | {"scores": col.get("scores"), "ro_scores": col.get("ro_scores"), "scale": col.get("scale")}
                out.append(_cell(col["sample"], steps[k] if k < len(steps) else None, seg_of, view))
    out.append("</div>")
    # The pinned column titles need an opaque background matching Streamlit's theme (3.75rem clears its top bar).
    style = f' style="--bg:{"#0e1117" if dark else "#ffffff"}' + (';--maxh:none"' if expand else '"')
    return CSS + f'<div class="pg"{style}>' + "".join(out) + "</div>"
