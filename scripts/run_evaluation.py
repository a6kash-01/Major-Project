"""
Run the ROUGE benchmark (BART vs T5) from the command line and save the results.

Usage:
    python scripts/run_evaluation.py

Writes data/evaluation_results.json and prints a comparison table for the project report.
"""
import os
import sys
import json
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import BENCHMARK_PATH, DATA_DIR
from app.core.evaluator import rouge_evaluator

RESULTS_PATH = DATA_DIR / "evaluation_results.json"
METRIC_NAMES = {"rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L"}


def main():
    results = rouge_evaluator.evaluate_benchmark(str(BENCHMARK_PATH))

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    bart, t5 = results["models"]["BART"], results["models"]["T5"]
    print(f"\nROUGE evaluation over {results['num_test_cases']} clinical test cases\n")
    print(f"{'Metric':<10}{'BART P':>9}{'BART R':>9}{'BART F1':>10}{'T5 P':>9}{'T5 R':>9}{'T5 F1':>10}")
    print("-" * 66)
    for key, name in METRIC_NAMES.items():
        b, t = bart[key], t5[key]
        print(f"{name:<10}{b['precision']:>9.4f}{b['recall']:>9.4f}{b['f1']:>10.4f}"
              f"{t['precision']:>9.4f}{t['recall']:>9.4f}{t['f1']:>10.4f}")

    secs = results["avg_inference_seconds"]
    print(f"\nAverage summarization time per case: BART {secs['BART']}s, T5 {secs['T5']}s")
    print(f"Better model (by ROUGE-L F1): {results['overall_winner']}")

    print("\nPer-case ROUGE-L F1:")
    for case in results["detailed_cases"]:
        print(f"  {case['case_id']:<10}BART {case['bart']['metrics']['rougeL']['f1']:.4f}   "
              f"T5 {case['t5']['metrics']['rougeL']['f1']:.4f}   {case['title']}")

    print(f"\nFull results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
