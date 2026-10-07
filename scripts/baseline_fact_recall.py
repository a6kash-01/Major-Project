"""Score the ORIGINAL project's saved summaries with the improved project's fact-recall metric.

Usage: python baseline_fact_recall.py <improved_project_root> <baseline_results_json> [...]
"""
import os
import sys
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, sys.argv[1])
from app.core.evaluator import rouge_evaluator  # improved project's metric

for path in sys.argv[2:]:
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    for model in ("bart", "t5"):
        scores = [rouge_evaluator.fact_recall(c["reference_summary"], c[model]["summary"])
                  for c in results["detailed_cases"]]
        scores = [s for s in scores if s is not None]
        print(f"{os.path.basename(path)}  {model.upper():<5} fact recall {sum(scores) / len(scores):.1%}  "
              f"({len(scores)} cases)")
