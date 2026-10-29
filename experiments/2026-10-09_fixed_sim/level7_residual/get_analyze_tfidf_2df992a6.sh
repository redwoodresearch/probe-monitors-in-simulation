#!/bin/sh
# The level-7 residual job (run_level7_residual_modal.py) reads analyze_tfidf_2df992a6.py: q1's analyze_tfidf.py as of
# commit 2df992a6 (on main), byte for byte. Git keeps this loader instead of a 655-line copy; run it once, from the repo.
set -eu
here=$(dirname "$0")
git show 2df992a6:experiments/2026-10-06_q1_text_baselines/analyze_tfidf.py > "$here/analyze_tfidf_2df992a6.py"
echo "f317903f76a7275c184c8f59dbe27e5dafe5e4c78dea97c56c0daf9515562a7e  $here/analyze_tfidf_2df992a6.py" | shasum -a 256 -c -
