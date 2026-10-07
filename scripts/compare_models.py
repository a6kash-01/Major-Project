"""
Compare pretrained BART / T5 checkpoints on the clinical benchmark (ablation study).

For every checkpoint it reports ROUGE for the raw model summary and for the final
fact-grounded summary that the system puts in the SOAP report.

Usage:
    python scripts/compare_models.py                       # all checkpoints below, on the 10-case benchmark
    python scripts/compare_models.py --only samsum flan    # checkpoints whose label contains a word
    python scripts/compare_models.py --dataset data/mts_dialog/MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv
                                                           # held-out MTS-Dialog GENHX conversations

Writes data/model_comparison.json (or the path given with --out).
"""
import os
import sys
import gc
import csv
import json
import argparse
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import BENCHMARK_PATH, DATA_DIR
from app.core.evaluator import rouge_evaluator
from app.core.summarizer_bart import BARTSummarizer
from app.core.summarizer_t5 import T5Summarizer

RESULTS_PATH = DATA_DIR / "model_comparison.json"

# (label, family, Hugging Face checkpoint, T5 task prompt) - all pretrained, none trained by this project
CHECKPOINTS = [
    ("BART-large-CNN (original)", "BART", "facebook/bart-large-cnn", None),
    ("BART-large-CNN-SAMSum", "BART", "philschmid/bart-large-cnn-samsum", None),
    ("BART-large dialogue (MEETING_SUMMARY)", "BART", "knkarthick/MEETING_SUMMARY", None),
    ("T5-base (original)", "T5", "t5-base", "summarize: "),
    ("Flan-T5-base", "T5", "google/flan-t5-base", "Summarize this doctor-patient conversation: "),
    ("Flan-T5-base-SAMSum", "T5", "philschmid/flan-t5-base-samsum", "summarize: "),
]


def build(family, checkpoint, prompt):
    summarizer = BARTSummarizer(model_name=checkpoint) if family == "BART" else T5Summarizer(model_name=checkpoint, prompt=prompt)
    summarizer._load_model()
    # The summarizers fall back to a smaller model if loading fails; that would mislabel the results
    if summarizer.model_name != checkpoint:
        raise RuntimeError(f"Could not load '{checkpoint}' (fell back to '{summarizer.model_name}')")
    return summarizer


def load_cases(path, section):
    """The project benchmark (JSON), or one section type of an MTS-Dialog CSV file."""
    if not path.endswith(".csv"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return [{"id": row["ID"], "title": row["section_header"], "transcript": row["dialogue"],
                 "reference_summary": row["section_text"]}
                for row in csv.DictReader(f) if row["section_header"] == section]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="evaluate only checkpoints whose label contains one of these words")
    parser.add_argument("--dataset", default=str(BENCHMARK_PATH), help="benchmark JSON or MTS-Dialog CSV")
    parser.add_argument("--section", default="GENHX", help="MTS-Dialog section to evaluate (default GENHX)")
    parser.add_argument("--out", default=str(RESULTS_PATH))
    args = parser.parse_args()

    cases = load_cases(args.dataset, args.section)

    selected = [c for c in CHECKPOINTS
                if not args.only or any(word.lower() in c[0].lower() for word in args.only)]

    rows = []
    for label, family, checkpoint, prompt in selected:
        print(f"Evaluating {label} ({checkpoint})...", flush=True)
        summarizer = build(family, checkpoint, prompt)
        result = rouge_evaluator.evaluate_summarizers(cases, {label: summarizer})
        rows.append({
            "label": label,
            "family": family,
            "checkpoint": checkpoint,
            "prompt": prompt,
            "model_only": result["models_raw"][label],
            "fact_grounded": result["models"][label],
            "seconds_per_case": result["avg_inference_seconds"][label],
            "cases": result["detailed_cases"]
        })
        del summarizer
        gc.collect()

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"\nROUGE F1 on {len(cases)} cases from {Path(args.dataset).name} (model only -> with fact grounding)\n")
    print(f"{'Checkpoint':<40}{'ROUGE-1':>17}{'ROUGE-2':>17}{'ROUGE-L':>17}{'Fact recall':>17}{'sec/case':>10}")
    print("-" * 118)
    for row in rows:
        cells = "".join(
            f"{row['model_only'][m]['f1']:>8.4f} ->{row['fact_grounded'][m]['f1']:>6.4f}"
            for m in ("rouge1", "rouge2", "rougeL")
        )
        facts = f"{row['model_only']['fact_recall']:>8.1%} ->{row['fact_grounded']['fact_recall']:>6.1%}"
        print(f"{row['label']:<40}{cells}{facts}{row['seconds_per_case']:>10}")
    print(f"\nFull results saved to {args.out}")


if __name__ == "__main__":
    main()
