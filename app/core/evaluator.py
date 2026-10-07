import json
import os
import re
import time
from typing import Dict, Any, List, Optional
from rouge_score import rouge_scorer
from app.config import BENCHMARK_PATH, logger
from app.core.preprocessor import preprocessor
from app.core.report_generator import report_generator
from app.core.summarizer_bart import bart_summarizer
from app.core.summarizer_t5 import t5_summarizer

class ROUGEEvaluator:
    """
    Evaluation Module for Clinical Summarization Models.
    Calculates ROUGE-1, ROUGE-2, and ROUGE-L metrics (Precision, Recall, F1)
    to compare BART and T5 models on standardized test datasets.
    """
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def calculate_metrics(self, reference: str, candidate: str) -> Dict[str, Dict[str, float]]:
        """
        Compute ROUGE-1, ROUGE-2, and ROUGE-L precision, recall, and f1 scores.
        """
        scores = self.scorer.score(reference.strip(), candidate.strip())
        
        return {
            "rouge1": {
                "precision": round(scores["rouge1"].precision, 4),
                "recall": round(scores["rouge1"].recall, 4),
                "f1": round(scores["rouge1"].fmeasure, 4)
            },
            "rouge2": {
                "precision": round(scores["rouge2"].precision, 4),
                "recall": round(scores["rouge2"].recall, 4),
                "f1": round(scores["rouge2"].fmeasure, 4)
            },
            "rougeL": {
                "precision": round(scores["rougeL"].precision, 4),
                "recall": round(scores["rougeL"].recall, 4),
                "f1": round(scores["rougeL"].fmeasure, 4)
            }
        }

    def evaluate_benchmark(self, dataset_path: str = str(BENCHMARK_PATH)) -> Dict[str, Any]:
        """
        Run comparative evaluation of BART vs T5 on ground-truth reference dataset.
        Returns aggregated scores and case-by-case comparison.
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Benchmark dataset file not found at: {dataset_path}")

        with open(dataset_path, "r", encoding="utf-8") as f:
            test_cases = json.load(f)

        results = self.evaluate_summarizers(test_cases, {"BART": bart_summarizer, "T5": t5_summarizer})

        # Superior model winner determination based on ROUGE-L F1 of the final reported summary
        models = results["models"]
        results["overall_winner"] = "BART" if models["BART"]["rougeL"]["f1"] >= models["T5"]["rougeL"]["f1"] else "T5"
        return results

    def fact_recall(self, reference: str, candidate: str) -> Optional[float]:
        """
        Clinical fact recall: the share of the reference's critical facts - numbers (doses, vitals,
        durations, ages) and medication names - that also appear in the candidate summary.
        Returns None when the reference states no such facts.
        """
        number = r"\d+(?:[./]\d+)?"
        facts = set(re.findall(number, reference))
        facts |= {m.lower() for m in report_generator.extract_medications(reference)
                  if not m.startswith("No medications")}
        if not facts:
            return None
        candidate_lower = candidate.lower()
        candidate_numbers = set(re.findall(number, candidate_lower))
        found = sum(1 for fact in facts
                    if (fact in candidate_numbers if fact[0].isdigit()
                        else re.search(r"\b" + re.escape(fact) + r"\b", candidate_lower)))
        return round(found / len(facts), 4)

    def average(self, metric_list: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
        """Average per-case ROUGE scores (and clinical fact recall over cases that state facts)."""
        averaged = {
            metric: {k: round(sum(m[metric][k] for m in metric_list) / len(metric_list), 4)
                     for k in ("precision", "recall", "f1")}
            for metric in ("rouge1", "rouge2", "rougeL")
        }
        recalls = [m["fact_recall"] for m in metric_list if m.get("fact_recall") is not None]
        averaged["fact_recall"] = round(sum(recalls) / len(recalls), 4) if recalls else None
        return averaged

    def evaluate_summarizers(self, test_cases: List[Dict[str, Any]], summarizers: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score each summarizer on the test cases. "models" holds the scores of the final summary the
        system reports (model output + fact grounding); "models_raw" holds the model output alone.
        """
        # Load models up front so model loading is not counted in the inference timings
        for summarizer in summarizers.values():
            summarizer._load_model()

        num_cases = len(test_cases)
        scores = {name: {"raw": [], "final": [], "seconds": 0.0} for name in summarizers}
        detailed_cases = []

        logger.info(f"[ROUGEEvaluator] Running benchmark evaluation across {num_cases} clinical test cases...")

        for idx, case in enumerate(test_cases):
            ref_summary = case.get("reference_summary", "")
            proc = preprocessor.process(case.get("transcript", ""))
            entry = {
                "case_id": case.get("id", f"case_{idx+1}"),
                "title": case.get("title", f"Test Case {idx+1}"),
                "reference_summary": ref_summary
            }

            for name, summarizer in summarizers.items():
                start = time.perf_counter()
                model_summary = summarizer.summarize(proc["chunks"])["summary"]
                seconds = time.perf_counter() - start
                final_summary = report_generator.ground_summary(
                    model_summary, proc["dialogue_turns"], proc["clean_transcript"]
                )
                raw_metrics = self.calculate_metrics(ref_summary, model_summary)
                final_metrics = self.calculate_metrics(ref_summary, final_summary)
                raw_metrics["fact_recall"] = self.fact_recall(ref_summary, model_summary)
                final_metrics["fact_recall"] = self.fact_recall(ref_summary, final_summary)

                scores[name]["raw"].append(raw_metrics)
                scores[name]["final"].append(final_metrics)
                scores[name]["seconds"] += seconds
                entry[name.lower()] = {
                    "summary": final_summary,
                    "metrics": final_metrics,
                    "model_summary": model_summary,
                    "model_metrics": raw_metrics,
                    "seconds": round(seconds, 2)
                }
            detailed_cases.append(entry)

        return {
            "num_test_cases": num_cases,
            "models": {name: self.average(s["final"]) for name, s in scores.items()},
            "models_raw": {name: self.average(s["raw"]) for name, s in scores.items()},
            "avg_inference_seconds": {name: round(s["seconds"] / num_cases, 2) for name, s in scores.items()},
            "detailed_cases": detailed_cases
        }

# Global Singleton Instance
rouge_evaluator = ROUGEEvaluator()
