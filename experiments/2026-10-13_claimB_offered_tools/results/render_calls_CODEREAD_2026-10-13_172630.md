# Reader render calls: is the offered-tool list rendered? (code reading, 10-13 17:26 PT)

Part 1 of RULE.md in this folder. It is read from the code at commit with `git show <commit>:<path> | grep -n -E
"apply_chat_template\(|segment_end_positions\(|tools="`. There are no numbers from data. Codex checked the same calls
with `git show` when it reviewed the rule (all single-line, none passes `tools=`).

| Reader / pipeline | File @ commit | Render call | `tools=` passed |
|---|---|---|---|
| Qwen3-32B and Qwen3.5 (#36, fix939, fix-full) | `src/probe_shift/reader.py` @ 1345a54e | line 43, `_layout`: `tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)` | no |
| Qwen3-32B, Step 1 placeholder extractor | `experiments/2026-10-13_q2_step1_placeholder/run_extract_qwen3_ph.py` @ dc1fcbe1 | line 161: `segment_end_positions(tok, it["messages"], cfg["max_tokens"])`, which renders through `reader._layout` (reader.py line 43) | no |
| Olmo | `experiments/2026-10-12_q2_olmo_reader/run_extract_olmo.py` @ 1345a54e | line 86: `tok.apply_chat_template(msgs, tokenize=False)` | no |
| Llama | `experiments/2026-10-12_q2_llama_reader/run_extract_llama.py` @ 1345a54e | line 112: `tok.apply_chat_template(prep(msgs), tokenize=False, date_string=DATE)` | no |
| Seed | `experiments/2026-10-13_g3_new_reader/run_extract_g3.py` @ a3efa7ea | lines 122 and 130: `tok.apply_chat_template(msgs, tokenize=False)` and `(sent, tokenize=False)` | no |

The string `tools=` appears nowhere in any of these five files.

**Staged rows:** fix-full's staged honest shard (2026-10-13_q2_fixfull_inputs_v2/inputs/shard_12.jsonl, sha256
94ae36d6…b799) has no tools field. Its row keys are arm, cell, cohort, environment, epoch, file, key, main_task, messages,
n_submit_calls, sample_id, side_task, sim_reference, tokens and truncated. Its message keys are role, content, tool_name and
tool_calls. (Read 17:15 PT, before Aswin's 17:24 no-local-compute rule.)

**Verdict, per reader:** the structured offered-tool list (the chat template's tools block) is NOT rendered for Qwen3-32B
(all pipelines, including Step 1), Qwen3.5, Olmo, Llama or Seed. This does not cover tool names written into message
text (system or user prose); that is Part 3 of RULE.md.

**Setup observation** (3 samples read during setup, 17:15–17:20 PT; not a result): grep, list_dir and read_file were
offered to the SIMULATOR model, not the agent.
- openai/gpt-6-astra model events, inside the agent's bash or text_editor tool spans (span type "tool"), have
  [grep, list_dir, read_file].
- The agent, openai/gpt-5.6-sol in the arena_policy solver span, has [bash, submit, text_editor] in a simulated honest,
  a simulated attack and a real honest sample.
- Part 2 counts this over all runs.
