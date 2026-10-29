"""Codes-only copies of the coding results, for git. The coders' paraphrases, arm-guess reasons and the settling
rule text can repeat short runs of the agent's own words, so (until the raw-text policy is decided) they stay in
git-ignored data/full/ and on S3. The committed copies keep codes, turn numbers, counts, flags, confidences, arm
guesses and salted ids. Fails if any remaining string repeats 5 consecutive words of a coded transcript.

  python scrub_for_git.py data/full/<file>.json [...]   -> results/<file stem>__codes_only.json
"""

import glob
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
FREE_TEXT = {"paraphrase", "arm_guess_basis", "rule", "paraphrases", "quote"}


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in FREE_TEXT}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for k, v in x.items():
            yield k
            yield from strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from strings(v)


EXPECTED_TRANSCRIPTS = 14  # the coded set (README); without them the overlap check would guard nothing


def main():
    files = sorted(glob.glob(str(HERE / "data/coding/*.txt")))
    if len(files) != EXPECTED_TRANSCRIPTS:  # data/ is git-ignored: on a fresh checkout, fetch it from S3 first (README)
        raise SystemExit(f"found {len(files)} coded transcripts in data/coding, expected {EXPECTED_TRANSCRIPTS}: refusing to scrub "
                         "without the corpus the overlap check needs")
    corpus = re.sub(r"\s+", " ", " ".join(Path(f).read_text().lower() for f in files))
    for src in map(Path, sys.argv[1:]):
        out = strip(json.loads(src.read_text()))
        for s in strings(out):
            w = s.lower().split()
            for i in range(len(w) - 4):
                if " ".join(w[i : i + 5]) in corpus:
                    raise ValueError(f"{src.name}: still repeats transcript words: {' '.join(w[i:i + 5])!r}")
        dst = HERE / "results" / f"{src.stem}__codes_only.json"
        with dst.open("x") as f:
            f.write(json.dumps(out, indent=1))
        print(src.name, "->", dst.name)


if __name__ == "__main__":
    main()
