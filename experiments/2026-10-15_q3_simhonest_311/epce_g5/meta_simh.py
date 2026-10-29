# Adapted from meta.py @ 02f460da2 for sim-honest .evals (2 changes marked ADAPT). Read-only metadata of .eval files. Prints NO message/transcript/tool text: only status, the sample error
# message string, model-event counts, and for the last two model events: stop_reason, token usage, and LENGTHS
# (characters) of visible completion text and reasoning text, plus the count of tool calls in the output.
# Args: <run_dir_abs> <policy_model> <glob relative to run_dir> [...]   (only sim_honest@september/ globs are accepted)
import collections
import glob
import os
import sys

from inspect_ai.log import read_eval_log

run_dir, policy = sys.argv[1], sys.argv[2]
for pat in sys.argv[3:]:
    assert pat.startswith("sim_honest@september/"), pat  # ADAPT 1: sim-honest folder
    for p in sorted(glob.glob(os.path.join(run_dir, pat, "*.eval"))):
        log = read_eval_log(p)
        print("==", os.path.relpath(p, run_dir))
        print("  status", log.status, "n_samples", len(log.samples or []))
        for s in log.samples or []:
            print("  sample_error_message", repr(s.error.message if s.error else None))
            mev = [(i, e) for i, e in enumerate(s.events) if e.event == "model"]
            print("  n_events", len(s.events), "n_model_events", len(mev),
                  "n_tool_events", sum(e.event == "tool" for e in s.events),
                  "model_events_by_model", dict(collections.Counter(e.model for _, e in mev)))
            pev = [(i, e) for i, e in mev if e.model == policy]  # ADAPT 2: last two POLICY calls (simulator calls excluded)
            print("  n_policy_model_events", len(pev), "last_model_event_index", mev[-1][0] if mev else None)
            for i, e in pev[-2:]:
                out = e.output
                ch = out.choices[0] if out and out.choices else None
                msg = ch.message if ch else None
                vis_len, rea_len, n_rea = 0, 0, 0
                if msg is not None:
                    if isinstance(msg.content, str):
                        vis_len = len(msg.content)
                    else:
                        for c in msg.content:
                            if c.type == "text":
                                vis_len += len(c.text)
                            elif c.type == "reasoning":
                                n_rea += 1
                                rea_len += len(c.reasoning or "")
                u = out.usage if out else None
                print("  model_event", "event_index", i, "model", e.model,
                      "stop_reason", ch.stop_reason if ch else None,
                      "input_tokens", u.input_tokens if u else None,
                      "output_tokens", u.output_tokens if u else None,
                      "reasoning_tokens", u.reasoning_tokens if u else None,
                      "visible_text_chars", vis_len, "output_completion_chars", len(out.completion) if out else None,
                      "n_tool_calls", len(msg.tool_calls or []) if msg else None,
                      "n_reasoning_blocks", n_rea, "reasoning_chars", rea_len,
                      "event_error_set", e.error is not None, "output_error_set", bool(out and out.error))
