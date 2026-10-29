"""The text baselines' harness-normalizing scrub, copied verbatim so the probe and TF-IDF read identical text.

Source: baselines-q3's experiments/2026-10-06_q1_text_baselines/analyze_tfidf.py at commit bbf355e2 (file sha256
7ffec621cf2f0d3e3c944b6bc4477d75e08bc0fabcd126e13fd6b1f151be756c), lines from "# Level 1." to `scrub`; since then both
copies changed together once (2026-10-07: UUID and hex IDs joined by `_` are masked too, so text at levels 2-7 differs
from extractions and scores made before that). `scrub(text, level)`, levels cumulative: 1 times and dates, 2 numbers and
IDs, 3 whitespace, 4 harness format, 5 diff lines, 6 empty diff lines, 7 editor output format. Keep in sync
(tests/test_scrub.py checks the ID rules match).
"""

import re

# Bumped when a rule changes what a level outputs. 2 (2026-10-07): UUID and hex IDs joined by `_` are masked too
# (levels 2-7). An extraction staged at one version must not be resumed at another (run_extract_segment_end.upload).
SCRUB_VERSION = 2

# Level 1. Wall-clock times identify when a cohort ran (real epochs 1 and 2 ran about 2 h apart): ISO dates, clock
# times (not inside IPs or ports), month-day dates, Unix epoch seconds or ms.
MONTHS = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
TIME_PATTERNS = [(re.compile(r"\d{4}-\d{2}-\d{2}"), "<date>"),
                 (re.compile(r"(?<![\d.:])\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?(?![\d:])"), "<time>"),
                 (re.compile(rf"\b(?:{MONTHS})\s+\d{{1,2}}\b"), "<date>"),
                 (re.compile(r"\b1[5-9]\d{8}(?:\d{3})?\b"), "<epoch>")]
# Level 2. Numbers and IDs: PIDs, sizes, inodes, ports, hashes, container IDs. UUIDs and hex IDs are bounded like
# `\b` except that `_` also separates (`(?<![^\W_])`, `(?![^\W_])`): with `\b`, an ID joined by an underscore
# (`run_46a7d3b90e5c`) escaped the hex rule and the digit rule chopped it into `<n>a<n>d<n>b…`, while the hyphen-joined
# form (`run-46a7d3b90e5c`) became `<hex>`, a tell made by the scrub itself (fixed 2026-10-07).
ID_PATTERNS = [(re.compile(r"(?<![^\W_])[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}(?![^\W_])"), "<uuid>"),
               (re.compile(r"(?<![\d.])\d{1,3}(?:\.\d{1,3}){3}(?![\d.])"), "<ip>"),
               (re.compile(r"(?<![^\W_])(?=[0-9a-fA-F]*[a-fA-F])(?=[0-9a-fA-F]*\d)[0-9a-fA-F]{6,}(?![^\W_])"), "<hex>"),
               (re.compile(r"\d+"), "<n>")]
# Level 3. Whitespace.
TRAILING_SPACE, BLANK_RUNS = re.compile(r"[^\S\n]+$", re.M), re.compile(r"\n{3,}")
# Level 4. Harness format: real editor diffs carry a line-number gutter (" 142|+ code"), simulated ones don't, so the
# gutter is stripped from real diffs; stderr-looking lines are dropped (their order in the output differs by harness);
# weekday and month names masked; blank lines removed and runs of spaces or tabs collapsed.
EDITOR_DIFF, GUTTER = "Here's a diff showing the changes made", re.compile(r"^ *\d+\|", re.M)
STDERR_LINE = re.compile(r"^(?:(?:fatal|error|warning|bash|sh|/bin/sh|sed|grep|ls|cat|find|rm|cp|mv|mkdir|chmod|head|tail|awk|"
                         r"wc|xargs|cd|python3?|node|npm WARN|E): .*|.*(?:No such file or directory|Permission denied|"
                         r"command not found|not a git repository).*)\n?", re.M | re.I)
# Level 5. Diff lines canonicalized: real diff lines put one space between the +/-/context sign and the code (" 46|+ code"),
# simulated ones don't ("+code"), so that space goes with the gutter; hunk headers are reduced to "@@" (the simulator
# sometimes writes the short "@@ -N +N @@" form).
GUTTER_AND_SPACE, HUNK = re.compile(r"^ *\d+\|([+\- ]?) ?", re.M), re.compile(r"^@@ [^\n]*@@[^\n]*$", re.M)
# Level 6. Empty diff lines dropped: the real tool's diff algorithm matches new blank lines against existing ones and shows
# them as context, while the simulator writes them as added lines ("+" alone; 0.88 per simulated diff vs 0.47 real), so
# lines that are only a sign are removed from editor diffs (blank context lines are already gone at level 4).
EMPTY_DIFF_LINE = re.compile(r"^[+-]$\n?", re.M)
# Level 7. Editor output format canonicalized: the real tool answers some edits (all inserts) with a `cat -n` snippet of
# the edited file plus a review note, the simulator with a diff, so both are reduced to plain code: preambles, the review
# note, diff headers and removed lines dropped, +/space diff markers and `cat -n` line numbers stripped.
EDITED = re.compile(r"^The file \S+ has been edited\. Here's (?:a diff showing the changes made|the result of running `cat -n` "
                    r"on a snippet of (?:the edited file|\S+)):\n?", re.M)
CAT_N_VIEW = re.compile(r"^Here's the result of running `cat -n` on \S+:\n?", re.M)
REVIEW_NOTE = re.compile(r"\n?Review the changes and make sure they are as expected \(correct indentation, no duplicate lines, "
                         r"etc\)\. Edit the file again if necessary\.?")
DIFF_HEADER, REMOVED_LINE = re.compile(r"^(?:--- old_version|\+\+\+ edited_version|@@.*)$\n?", re.M), re.compile(r"^-.*$\n?", re.M)
# A `cat -n` line number is followed by a tab, or ends the line when the readout cut falls right after it.
DIFF_MARKER, CAT_N_PREFIX = re.compile(r"^[+ ]", re.M), re.compile(r"^ *\d+(?:\t|$)", re.M)
WEEKDAYS = re.compile(r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b")
MONTH_NAMES = re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|"
                         r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)\b")


def mask_times(text: str) -> str:
    for pattern, placeholder in TIME_PATTERNS:
        text = pattern.sub(placeholder, text)
    return text


def scrub(text: str, level: int) -> str:
    """Cumulative scrub: 1 times and dates, 2 numbers and IDs, 3 whitespace, 4 harness format, 5 diff lines, 6 empty diff
    lines, 7 editor output format (diff and `cat -n` snippet both reduced to plain code)."""
    if level >= 4:  # before the digit masks, which would turn the gutter into "<n>|"
        diff = EDITOR_DIFF in text
        if diff:
            text = GUTTER_AND_SPACE.sub(r"\1", text) if level >= 5 else GUTTER.sub("", text)
            if level >= 5:
                text = HUNK.sub("@@", text)
        if level >= 7 and (diff or "result of running `cat -n`" in text):
            text = REVIEW_NOTE.sub("", CAT_N_VIEW.sub("", EDITED.sub("", text)))
            if diff:
                text = DIFF_MARKER.sub("", REMOVED_LINE.sub("", DIFF_HEADER.sub("", text)))
            else:
                text = CAT_N_PREFIX.sub("", text)
        text = STDERR_LINE.sub("", text)
    if level >= 1:
        text = mask_times(text)
    if level >= 2:
        for pattern, placeholder in ID_PATTERNS:
            text = pattern.sub(placeholder, text)
    if level >= 3:
        text = BLANK_RUNS.sub("\n\n", TRAILING_SPACE.sub("", text)).rstrip()
    if level >= 4:
        text = MONTH_NAMES.sub("<month>", WEEKDAYS.sub("<wday>", text))
        text = re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n", "\n", text))
    if level >= 6 and EDITOR_DIFF in text:
        text = EMPTY_DIFF_LINE.sub("", text).rstrip("\n")
    return text
