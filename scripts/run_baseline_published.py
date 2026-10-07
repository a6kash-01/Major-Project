"""Run the ORIGINAL project's ROUGE evaluator unchanged and save its results.

Usage: python run_baseline.py <original_project_root> <output_json> [dataset_json]
"""
import os
import sys
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
root, out_path = sys.argv[1], sys.argv[2]
dataset = sys.argv[3] if len(sys.argv) > 3 else os.path.join(root, "data", "benchmark_dataset.json")
sys.path.insert(0, root)
os.chdir(root)

from app.core.evaluator import rouge_evaluator  # original code

results = rouge_evaluator.evaluate_benchmark(dataset)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

for model in ("BART", "T5"):
    m = results["models"][model]
    print(model, "R1 %.4f  R2 %.4f  RL %.4f" % (m["rouge1"]["f1"], m["rouge2"]["f1"], m["rouge"+"L"]["f1"]))
