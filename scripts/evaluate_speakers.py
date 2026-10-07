"""
Measure Doctor/Patient speaker-separation accuracy.

The benchmark transcripts are labelled ("Doctor: ...", "Patient: ..."). This script removes the
labels, so the text looks like raw Whisper output, runs the preprocessor's speaker separation,
and reports the percentage of words attributed to the correct speaker.

Usage:
    python scripts/evaluate_speakers.py                     # this project, on data/benchmark_dataset.json
    python scripts/evaluate_speakers.py --root <project>    # another copy, e.g. the original project
    python scripts/evaluate_speakers.py --quiet --dataset data/mts_dialog/*.csv   # held-out MTS-Dialog
"""
import os
import re
import sys
import csv
import json
import difflib
import argparse
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser()
parser.add_argument("--root", default=str(Path(__file__).resolve().parent.parent),
                    help="project whose app/core/preprocessor.py is evaluated")
parser.add_argument("--dataset", nargs="+",
                    default=[str(Path(__file__).resolve().parent.parent / "data" / "benchmark_dataset.json")],
                    help="benchmark JSON and/or MTS-Dialog CSV files")
parser.add_argument("--quiet", action="store_true", help="print only the overall accuracy")
args = parser.parse_args()
sys.path.insert(0, str(Path(args.root).resolve()))

from app.core.preprocessor import preprocessor  # noqa: E402  (imported from --root)


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def score_case(transcript):
    """Return (correct, matched) word counts for one labelled transcript."""
    gold, plain_lines = [], []
    for line in transcript.split("\n"):
        match = re.match(r"^\s*(doctor|patient)\s*:\s*(.*)$", line, re.I)
        if not match:
            continue
        speaker, text = match.group(1).capitalize(), match.group(2)
        gold += [(tok, speaker) for tok in tokens(text)]
        plain_lines.append(text)

    turns = preprocessor.process(" ".join(plain_lines))["dialogue_turns"]
    predicted = [(tok, turn["speaker"]) for turn in turns for tok in tokens(turn["text"])]

    # Align predicted words to gold words (cleaning removes fillers, so sequences can differ slightly)
    matcher = difflib.SequenceMatcher(a=[t for t, _ in gold], b=[t for t, _ in predicted], autojunk=False)
    correct = matched = 0
    for block in matcher.get_matching_blocks():
        for i in range(block.size):
            matched += 1
            correct += gold[block.a + i][1] == predicted[block.b + i][1]
    return correct, matched, len(turns)


def load_cases(paths):
    """Benchmark JSON, or MTS-Dialog CSV files (conversations with family members are skipped)."""
    cases = []
    for path in paths:
        if path.endswith(".csv"):
            with open(path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if not re.search(r"^\s*Guest_", row["dialogue"], re.M):
                        cases.append({"id": row["ID"], "title": row["section_header"], "transcript": row["dialogue"]})
        else:
            with open(path, "r", encoding="utf-8") as f:
                cases += json.load(f)
    return cases


cases = load_cases(args.dataset)
total_correct = total_matched = 0
print(f"Speaker attribution on {len(cases)} unlabelled transcripts ({args.root})\n")
for case in cases:
    correct, matched, n_turns = score_case(case["transcript"])
    total_correct += correct
    total_matched += matched
    if not args.quiet:
        print(f"  {case['id']:<10}{correct / matched:>7.1%}  ({n_turns} turns)  {case['title']}")
print(f"\nOverall word-level speaker accuracy: {total_correct / total_matched:.1%}")
